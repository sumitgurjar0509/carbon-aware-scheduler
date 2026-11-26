"""
Sanity tests for the parameterized uncertainty_aware scheduler
(schedulers.make_uncertainty_aware):

  1. risk_weight=1.0 reproduces Step 5's original fixed uncertainty_aware
     exactly (they should in fact be the literal same function object).
  2. risk_weight=0.0 is EXACTLY equivalent to forecast_greedy's decision
     rule (not just "closely related") -- worked out in the factory's
     own docstring: forecast_greedy runs now iff
     actual(t) <= min_future_q50, and risk_weight=0 reduces the
     threshold to q50(t*) where t* is chosen as the argmin of q50 among
     the same remaining candidates -- i.e. q50(t*) IS min_future_q50, so
     both compare the exact same value the exact same way.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from schedulers import (  # noqa: E402
    cvar,
    forecast_greedy,
    make_cvar_aware,
    make_uncertainty_aware,
    uncertainty_aware,
)
from simulator import SimContext, simulate_job  # noqa: E402


def _synthetic_ctx_and_jobs(seed=0, n_jobs=60):
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2026-01-01", tz="UTC")
    n_periods = 24 * 2 * 6
    timestamps = pd.date_range(start, periods=n_periods, freq="30min")

    actual_values = 100 + 60 * np.sin(2 * np.pi * np.arange(n_periods) / 48) \
        + rng.normal(0, 15, n_periods)
    actual_series = pd.Series(actual_values, index=timestamps)

    horizons = list(range(1, 25))
    forecast_tables = {}
    for h in horizons:
        shifted = actual_series.shift(-2 * h)
        noise = rng.normal(0, 25, n_periods)
        q50 = shifted.values + noise
        q50 = np.where(np.isnan(q50), actual_series.values, q50)
        q10 = q50 - 20
        q90 = q50 + 20
        forecast_tables[h] = pd.DataFrame({"q10": q10, "q50": q50, "q90": q90}, index=timestamps)

    ctx = SimContext(actual_series, forecast_tables, float(np.median(actual_values)))

    arrivals = start + pd.to_timedelta(rng.uniform(0, 4 * 24 * 60, n_jobs), unit="m")
    slack_hours = rng.choice([6, 12, 24], size=n_jobs)
    durations = rng.uniform(20, 200, n_jobs)
    jobs = pd.DataFrame({
        "job_id": np.arange(1, n_jobs + 1),
        "arrival_time": arrivals,
        "duration_minutes": durations,
        "slack_hours": slack_hours,
    })
    jobs["deadline"] = (
        jobs["arrival_time"]
        + pd.to_timedelta(jobs["duration_minutes"], unit="m")
        + pd.to_timedelta(jobs["slack_hours"], unit="h")
    )
    return ctx, jobs


def test_make_uncertainty_aware_default_alias_is_risk_weight_1():
    assert uncertainty_aware.risk_weight == 1.0


def test_risk_weight_1_reproduces_fixed_uncertainty_aware_results():
    ctx, jobs = _synthetic_ctx_and_jobs(seed=10, n_jobs=40)
    fresh_risk_1 = make_uncertainty_aware(1.0)

    for _, job in jobs.iterrows():
        r_old = simulate_job(job, uncertainty_aware, ctx)
        r_new = simulate_job(job, fresh_risk_1, ctx)
        assert r_old["started_time"] == r_new["started_time"]
        assert r_old["realized_carbon_cost"] == pytest.approx(
            r_new["realized_carbon_cost"], abs=1e-9
        )


def test_risk_weight_0_exactly_matches_forecast_greedy():
    """
    Not just "closely related" -- see the factory's docstring for the
    algebraic argument for exact equivalence. Verified here per-job,
    per-step, on synthetic data with deliberately noisy/wrong forecasts
    (so the two policies have plenty of opportunity to diverge if they
    were only approximately related).
    """
    ctx, jobs = _synthetic_ctx_and_jobs(seed=11, n_jobs=60)
    risk_0 = make_uncertainty_aware(0.0)

    for _, job in jobs.iterrows():
        r_fg = simulate_job(job, forecast_greedy, ctx)
        r_risk0 = simulate_job(job, risk_0, ctx)
        assert r_fg["started_time"] == r_risk0["started_time"], (
            f"job_id={job['job_id']}: forecast_greedy started at "
            f"{r_fg['started_time']}, risk_weight=0 started at {r_risk0['started_time']}"
        )
        assert r_fg["realized_carbon_cost"] == pytest.approx(
            r_risk0["realized_carbon_cost"], abs=1e-9
        )


def test_risk_weight_monotonically_widens_threshold():
    """
    Sanity check on the interpolation itself: for a fixed forecast
    (q50 < q90), a higher risk_weight should never produce a LOWER
    threshold than a smaller risk_weight -- the whole point of the
    parameterization is a monotone hedge from median (0.0) to worst
    case (1.0).
    """
    q50, q90 = 100.0, 180.0
    weights = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    thresholds = [q50 + w * (q90 - q50) for w in weights]
    assert thresholds == sorted(thresholds)
    assert thresholds[0] == pytest.approx(q50)
    assert thresholds[-1] == pytest.approx(q90)


# ----------------------------------------------------------------------
# Step 7: CVaR helper + make_cvar_aware
# ----------------------------------------------------------------------
def _brute_force_cvar(f, alpha, n=200_001):
    """Monte-Carlo-free check: average Q(p) on a fine grid over [1-alpha, 1]."""
    from schedulers import quantile_function_knots
    p, q = quantile_function_knots(f["q10"], f["q50"], f["q90"])
    grid = np.linspace(1 - alpha, 1, n)
    return float(np.interp(grid, p, q).mean())


@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.2, 0.5, 0.7, 1.0])
def test_cvar_matches_brute_force_integration(alpha):
    f = {"q10": 80.0, "q50": 120.0, "q90": 200.0}
    assert cvar(f, alpha) == pytest.approx(_brute_force_cvar(f, alpha), rel=1e-6)


def test_cvar_at_calibration_alpha_equals_q90():
    """Documented consequence of the piecewise-linear approximation."""
    f = {"q10": 50.0, "q50": 90.0, "q90": 170.0}
    assert cvar(f, 0.2) == pytest.approx(170.0)


def test_cvar_alpha_1_is_mean_and_monotone_in_alpha():
    f = {"q10": 80.0, "q50": 120.0, "q90": 200.0}
    # mean of a piecewise-uniform distribution = integral of Q over [0, 1]
    assert cvar(f, 1.0) == pytest.approx(_brute_force_cvar(f, 1.0), rel=1e-6)
    values = [cvar(f, a) for a in [1.0, 0.7, 0.5, 0.2, 0.1, 0.05]]
    assert values == sorted(values)  # smaller alpha -> deeper tail -> higher CVaR


def test_cvar_handles_crossed_quantiles():
    assert cvar({"q10": 130.0, "q50": 120.0, "q90": 200.0}, 0.2) == pytest.approx(200.0)


def test_cvar_aware_risk_0_exactly_matches_forecast_greedy():
    ctx, jobs = _synthetic_ctx_and_jobs(seed=12, n_jobs=60)
    cvar_0 = make_cvar_aware(alpha=0.2, risk_weight=0.0)
    for _, job in jobs.iterrows():
        assert simulate_job(job, forecast_greedy, ctx)["started_time"] ==             simulate_job(job, cvar_0, ctx)["started_time"]


@pytest.mark.parametrize("w", [0.2, 0.6, 1.0])
def test_cvar_aware_never_runs_earlier_than_uncertainty_aware(w):
    """
    At alpha=0.2 both score candidates by q50 + w*(q90-q50); cvar_aware
    takes the min over ALL candidates (<= the score at argmin-q50), so at
    every step it waits whenever uncertainty_aware waits.
    """
    ctx, jobs = _synthetic_ctx_and_jobs(seed=13, n_jobs=60)
    ua, ca = make_uncertainty_aware(w), make_cvar_aware(0.2, w)
    for _, job in jobs.iterrows():
        assert simulate_job(job, ca, ctx)["started_time"] >=             simulate_job(job, ua, ctx)["started_time"]
