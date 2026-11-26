"""
M3 scenario generation (docs/m3_spec.md §1)
===========================================
Turns the hourly calibrated quantile forecasts (q10, q50, q90 for horizons
h = 0..23 at one origin hour) into S possible 24-hour paths of carbon
intensity.

1. Per hour: a piecewise-linear inverse CDF through (0.1, q10), (0.5, q50),
   (0.9, q90), with straight-line tails to p = 0 and p = 1 that continue
   the neighbouring slope. Values are floored at 0.
2. Across hours: a Gaussian copula with AR(1) correlation rho^|h-k|.
3. Seeds: one random stream per origin hour (common random numbers), so
   every variant and every beta sees the same scenarios at the same hour.
   The first S rows of a larger draw are exactly a smaller draw.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import norm

P_KNOTS = np.array([0.0, 0.1, 0.5, 0.9, 1.0])
SCENARIO_SEED = 20261126
PIT_CLIP = 1e-4


# ----------------------------------------------------------------------
# Marginals
# ----------------------------------------------------------------------
def quantile_knots(q10, q50, q90):
    """
    Arrays of shape (..., 5): Q(p) at p = 0, 0.1, 0.5, 0.9, 1.
    Inputs are sorted first, so crossed quantiles still give a valid Q.
    """
    q = np.sort(np.stack([np.asarray(q10, float), np.asarray(q50, float),
                          np.asarray(q90, float)], axis=-1), axis=-1)
    lo, mid, hi = q[..., 0], q[..., 1], q[..., 2]
    return np.stack([lo - (mid - lo) / 4, lo, mid, hi, hi + (hi - mid) / 4], axis=-1)


def inverse_cdf(u, knots):
    """u: (S, H) probabilities; knots: (H, 5). Returns (S, H) values (not yet floored)."""
    u = np.asarray(u, float)
    out = np.empty_like(u)
    for h in range(knots.shape[0]):
        out[:, h] = np.interp(u[:, h], P_KNOTS, knots[h])
    return out


def cdf(y, knots):
    """
    Forward CDF F(y) for one value per horizon: y (H,), knots (H, 5) -> (H,).
    Values below Q(0) give 0, values above Q(1) give 1.
    """
    out = np.empty(len(y))
    for h in range(len(y)):
        k = knots[h] + np.arange(5) * 1e-9  # strictly increasing for np.interp
        out[h] = np.interp(y[h], k, P_KNOTS, left=0.0, right=1.0)
    return out


def scenario_mean(knots):
    """Exact mean of the piecewise-linear distribution: trapezoid area under Q(p)."""
    return np.sum(np.diff(P_KNOTS) * (knots[..., 1:] + knots[..., :-1]) / 2, axis=-1)


# ----------------------------------------------------------------------
# Copula
# ----------------------------------------------------------------------
def ar1_corr(rho: float, n: int = 24) -> np.ndarray:
    idx = np.arange(n)
    return rho ** np.abs(idx[:, None] - idx[None, :])


def origin_rng(origin: pd.Timestamp, seed: int = SCENARIO_SEED) -> np.random.Generator:
    """One stream per origin hour: SeedSequence([seed, hours since 1970])."""
    hours = int(pd.Timestamp(origin).timestamp() // 3600)
    return np.random.default_rng(np.random.SeedSequence([seed, hours]))


def sample_paths(q10, q50, q90, rho: float, n_scenarios: int, origin: pd.Timestamp,
                 seed: int = SCENARIO_SEED):
    """
    Returns (paths, n_floored): paths has shape (S, H), H = len(q50).
    paths[s, h] = scenario s of the carbon intensity of the hour origin + h.
    """
    knots = quantile_knots(q10, q50, q90)
    n_h = knots.shape[0]
    z = origin_rng(origin, seed).standard_normal((n_scenarios, n_h))
    if rho != 0:
        z = z @ np.linalg.cholesky(ar1_corr(rho, n_h)).T
    paths = inverse_cdf(norm.cdf(z), knots)
    n_floored = int((paths < 0).sum())
    return np.maximum(paths, 0.0), n_floored


# ----------------------------------------------------------------------
# Estimating rho (on C1 forecast errors only; spec §5)
# ----------------------------------------------------------------------
def pit_z(y, knots, clip: float = PIT_CLIP):
    """z = Phi^-1(F(y)), clipped. Returns (z, number of values that hit a bound)."""
    u = cdf(y, knots)
    hit = int(((u <= clip) | (u >= 1 - clip)).sum())
    return norm.ppf(np.clip(u, clip, 1 - clip)), hit


def ar1_neg_loglik(rho: float, z: np.ndarray) -> float:
    """
    Negative log-likelihood of rows of z under a stationary AR(1) with unit
    variance: z_0 ~ N(0,1), z_h | z_{h-1} ~ N(rho z_{h-1}, 1 - rho^2).
    Constants are dropped; z has shape (n_origins, H).
    """
    var = 1 - rho ** 2
    resid = z[:, 1:] - rho * z[:, :-1]
    n_cond = resid.size
    return 0.5 * (n_cond * np.log(var) + np.sum(resid ** 2) / var + np.sum(z[:, 0] ** 2))


def estimate_rho(z: np.ndarray) -> float:
    res = minimize_scalar(ar1_neg_loglik, bounds=(-0.999, 0.999), args=(z,),
                          method="bounded", options={"xatol": 1e-6})
    return float(res.x)


def lag_correlations(z: np.ndarray, max_lag: int = 6) -> dict:
    """Empirical correlation between z_h and z_{h+lag}, pooled over h and origins."""
    return {lag: float(np.corrcoef(z[:, :-lag].ravel(), z[:, lag:].ravel())[0, 1])
            for lag in range(1, max_lag + 1)}
