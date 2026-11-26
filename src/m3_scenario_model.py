"""
M3 scenario model: C1 CQR, tail quantile models, and C1 calibration diagnostics
================================================================================
docs/m3_spec.md §1a. The scenario model was revised after Stage 1 v1,
before any test-period run, and is judged ONLY by C1 calibration
diagnostics. No scheduling result is computed or read here.

    $env:CAS_DATA_TAG = "20250923_20260923"
    python src/m3_scenario_model.py --train-tails  # q02/q05/q95/q98 on train + CQR on C1
    python src/m3_scenario_model.py --diagnose     # A1-A3 for v1 and candidates, picks the model

Acceptance criteria (declared in the spec before any candidate was tried):
  A1  share of actual hourly values outside [Q(0), Q(1)]             <= 1%
  A2  coverage of the scenario 10-90% range of 6-hour sums,
      pooled over windows h0-5, 6-11, 12-17, 18-23, S = 1000          in [75%, 85%]
  A3  every decile bin of the PIT u = F(y)                            in [5%, 15%]
Candidates, in order, stop at the first that passes:
  (i)  7 levels (q02 q05 q10 q50 q90 q95 q98) + AR(1) rho re-estimated on C1
  (ii) 7 levels + empirical 24x24 Ledoit-Wolf correlation of z on C1 (only if (i) fails A2)
"""
import json
import subprocess
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lp_scheduler as lp  # noqa: E402
import scenarios as sc  # noqa: E402
from calibration import apply_calibration, conformal_correction, nonconformity_scores  # noqa: E402
from data_config import check_model_label, get_data_tag, write_model_label  # noqa: E402
from forecasting import get_feature_columns, load_clean_data, train_quantile_models  # noqa: E402
from hourly_forecasting import (  # noqa: E402
    ALPHA as CQR_ALPHA,
    HOURLY_HORIZONS,
    HOURLY_MODELS_DIR,
    forecast_tables,
    horizon_dataset,
    hourly_features,
    m1_boundaries,
    split_by_boundaries,
    to_hourly,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
C1_END = pd.Timestamp("2026-06-06 00:00", tz="UTC")   # approved split
CQR_C1_DIR = PROJECT_ROOT / "results" / "models" / "m3_cqr_c1"
CQR_C1_PATH = CQR_C1_DIR / "cqr_corrections.csv"
TAILS_DIR = PROJECT_ROOT / "results" / "models" / "m3_hourly_tails"
TAIL_LEVELS = (0.02, 0.05, 0.95, 0.98)
TAIL_PAIRS = {(0.05, 0.95): 0.10, (0.02, 0.98): 0.04}   # CQR alpha per interval pair
OUT_DIR = PROJECT_ROOT / "results" / "tables" / "m3" / "scenario_model"
STAGE1_V1 = PROJECT_ROOT / "results" / "tables" / "m3" / "stage1_v1" / "frozen_params.json"

DIAG_S = 1000
WINDOWS = ((0, 6), (6, 12), (12, 18), (18, 24))
A1_MAX, A2_RANGE, A3_RANGE = 0.01, (0.75, 0.85), (0.05, 0.15)


def level_key(level: float) -> str:
    return f"q{int(round(level * 100)):02d}"


# ----------------------------------------------------------------------
# C1 rows and CQR
# ----------------------------------------------------------------------
def c1_rows(df_h: pd.DataFrame, calib_start, c1_end=C1_END) -> pd.DataFrame:
    """Rows whose origin is in C1 and whose whole target hour ends by C1_END."""
    keep = (df_h["timestamp"] >= calib_start) & (df_h["target_timestamp"] + lp.ONE_HOUR <= c1_end)
    return df_h[keep].reset_index(drop=True)


def _booster(models_dir, h, level):
    return lgb.Booster(model_file=str(models_dir / f"h{h:02d}h_q{int(round(level * 100)):02d}.txt"))


def refit_cqr_c1(hourly, calib_start, c1_end=C1_END, models_dir=HOURLY_MODELS_DIR) -> pd.DataFrame:
    """CQR for (q10, q90), as in M2, fitted on C1 rows only with the unchanged m2_hourly models."""
    feat = hourly_features(hourly)
    cols = get_feature_columns(feat)
    rows = []
    for h in HOURLY_HORIZONS:
        rows_h = c1_rows(horizon_dataset(feat, hourly, h), calib_start, c1_end)
        lo, hi = (_booster(models_dir, h, q).predict(rows_h[cols]) for q in (0.1, 0.9))
        rows.append({"horizon_h": h,
                     "cqr_correction": conformal_correction(
                         nonconformity_scores(lo, hi, rows_h["target"]), CQR_ALPHA),
                     "n_calib": len(rows_h), "alpha": CQR_ALPHA,
                     "c1_last_target": str(rows_h["target_timestamp"].max())})
    return pd.DataFrame(rows)


def train_tail_models(df30, tag):
    """q02/q05/q95/q98 per horizon on the TRAINING split (same rules as M2), then CQR on C1."""
    hourly = to_hourly(df30)
    calib_start, test_start, _ = m1_boundaries(df30)
    feat = hourly_features(hourly)
    cols = get_feature_columns(feat)
    TAILS_DIR.mkdir(parents=True, exist_ok=True)
    (TAILS_DIR / "data_tag.json").unlink(missing_ok=True)
    rows = []
    for h in HOURLY_HORIZONS:
        df_h = horizon_dataset(feat, hourly, h)
        train, _, _ = split_by_boundaries(df_h, calib_start, test_start)
        models = train_quantile_models(train, cols, target_col="target", quantiles=TAIL_LEVELS)
        for q, booster in models.items():
            booster.save_model(str(TAILS_DIR / f"h{h:02d}h_q{int(round(q * 100)):02d}.txt"))
        c1 = c1_rows(df_h, calib_start)
        for (lo_q, hi_q), alpha in TAIL_PAIRS.items():
            lo, hi = models[lo_q].predict(c1[cols]), models[hi_q].predict(c1[cols])
            rows.append({"horizon_h": h, "pair": f"{level_key(lo_q)}-{level_key(hi_q)}",
                         "alpha": alpha, "n_train": len(train), "n_calib": len(c1),
                         "cqr_correction": conformal_correction(
                             nonconformity_scores(lo, hi, c1["target"]), alpha)})
        print(f"  h={h:>2}: trained on {len(train)} rows, CQR on {len(c1)} C1 rows", flush=True)
    corr = pd.DataFrame(rows)
    corr.to_csv(TAILS_DIR / "cqr_corrections_c1.csv", index=False)
    write_model_label(TAILS_DIR, tag, stage="calibration")
    return corr


# ----------------------------------------------------------------------
# Quantile tables for any set of levels
# ----------------------------------------------------------------------
def quantile_tables(hourly, origins: pd.DatetimeIndex, tag: str, levels=sc.LEVELS_V2) -> dict:
    """
    level key -> DataFrame (origins x 24). q10/q50/q90 from m2_hourly with the
    C1 corrections; tail levels from m3_hourly_tails with their C1 corrections.
    All levels are then sorted per (origin, horizon) (monotone rearrangement).
    """
    base = forecast_tables(hourly, origins, tag=tag, corrections_path=CQR_C1_PATH)
    out = {level_key(q): base[level_key(q)] for q in (0.1, 0.5, 0.9) if q in levels}
    extra = [q for q in levels if q in TAIL_LEVELS]
    if extra:
        check_model_label(TAILS_DIR, tag)
        corr = pd.read_csv(TAILS_DIR / "cqr_corrections_c1.csv")
        feat = hourly_features(hourly).set_index("timestamp")
        X = feat.loc[origins, get_feature_columns(feat.reset_index())]
        for q in extra:
            out[level_key(q)] = pd.DataFrame(index=origins, columns=HOURLY_HORIZONS, dtype=float)
        for h in HOURLY_HORIZONS:
            for (lo_q, hi_q) in TAIL_PAIRS:
                if lo_q not in extra:
                    continue
                c = float(corr[(corr.horizon_h == h)
                               & (corr.pair == f"{level_key(lo_q)}-{level_key(hi_q)}")].cqr_correction.iloc[0])
                lo, hi = apply_calibration(_booster(TAILS_DIR, h, lo_q).predict(X),
                                           _booster(TAILS_DIR, h, hi_q).predict(X), c)
                out[level_key(lo_q)][h], out[level_key(hi_q)][h] = lo, hi
    keys = [level_key(q) for q in sorted(levels)]
    stacked = np.sort(np.stack([out[k].to_numpy(float) for k in keys], axis=-1), axis=-1)
    return {k: pd.DataFrame(stacked[..., i], index=origins, columns=HOURLY_HORIZONS)
            for i, k in enumerate(keys)}


def stack_levels(tables: dict, levels) -> np.ndarray:
    """(n_origins, 24, L) array in level order."""
    return np.stack([tables[level_key(q)].to_numpy(float) for q in levels], axis=-1)


# ----------------------------------------------------------------------
# Diagnostics A1-A3 (spec §1a)
# ----------------------------------------------------------------------
def c1_origins(hourly, calib_start, c1_end=C1_END) -> pd.DatetimeIndex:
    ci = hourly.set_index("timestamp")["carbon_intensity"]
    o = ci.index[(ci.index >= calib_start) & (ci.index + 24 * lp.ONE_HOUR <= c1_end)]
    return o.intersection(pd.DatetimeIndex(hourly_features(hourly)["timestamp"]))


def pit_values(values, levels, y):
    """values (n, 24, L), y (n, 24) -> (u (n, 24), outside-support mask (n, 24))."""
    u = np.empty(y.shape)
    outside = np.zeros(y.shape, bool)
    for r in range(y.shape[0]):
        p, knots = sc.knots_from_levels(values[r], levels)
        u[r] = sc.cdf(y[r], knots, p)
        outside[r] = (y[r] < knots[:, 0]) | (y[r] > knots[:, -1])
    return u, outside


def z_from_u(u, clip=sc.PIT_CLIP):
    from scipy.stats import norm
    return norm.ppf(np.clip(u, clip, 1 - clip))


def shrunk_correlation(z: np.ndarray) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf covariance of z, rescaled to a correlation matrix. Returns (corr, shrinkage)."""
    from sklearn.covariance import LedoitWolf
    lw = LedoitWolf().fit(z)
    sd = np.sqrt(np.diag(lw.covariance_))
    return lw.covariance_ / np.outer(sd, sd), float(lw.shrinkage_)


def window_coverage(values, levels, y, origins, rho=0.0, corr=None, n_scen=DIAG_S):
    """Per window, the share of 6-hour actual sums inside the scenario 10-90% range."""
    hits = np.zeros((len(origins), len(WINDOWS)), bool)
    for r, o in enumerate(origins):
        paths, _ = sc.sample_paths_levels(values[r], levels, n_scen, o, rho=rho, corr=corr)
        for w, (a, b) in enumerate(WINDOWS):
            sums = paths[:, a:b].sum(axis=1)
            lo, hi = np.quantile(sums, [0.1, 0.9])
            hits[r, w] = lo <= y[r, a:b].sum() <= hi
    return hits.mean(axis=0)


def diagnose(name, values, levels, y, origins, rho=0.0, corr=None):
    """Returns (row of A1-A3 results, PIT decile shares, u)."""
    u, outside = pit_values(values, levels, y)
    hist = np.histogram(u.ravel(), bins=10, range=(0, 1))[0] / u.size
    cov = window_coverage(values, levels, y, origins, rho=rho, corr=corr)
    a1, a2 = float(outside.mean()), float(cov.mean())
    row = {"model": name, "levels": " ".join(level_key(q) for q in levels),
           "copula": "empirical Ledoit-Wolf" if corr is not None else f"AR(1) rho={rho:.4f}",
           "A1_outside_support": a1, "A1_pass": a1 <= A1_MAX,
           "A2_cov_6h_pooled": a2, "A2_pass": A2_RANGE[0] <= a2 <= A2_RANGE[1],
           "cov_6h_h0_5": float(cov[0]), "cov_6h_h6_11": float(cov[1]),
           "cov_6h_h12_17": float(cov[2]), "cov_6h_h18_23": float(cov[3]),
           "A3_min_bin": float(hist.min()), "A3_max_bin": float(hist.max()),
           "A3_pass": bool(A3_RANGE[0] <= hist.min() and hist.max() <= A3_RANGE[1]),
           "n_origins": len(origins), "n_values": int(u.size)}
    row["all_pass"] = bool(row["A1_pass"] and row["A2_pass"] and row["A3_pass"])
    return row, hist, u


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main(train=False, run_diagnose=False):
    tag = get_data_tag()
    df30 = load_clean_data()
    hourly = to_hourly(df30)
    calib_start, test_start, _ = m1_boundaries(df30)
    assert C1_END < test_start

    if train:
        CQR_C1_DIR.mkdir(parents=True, exist_ok=True)
        cqr = refit_cqr_c1(hourly, calib_start)
        assert (pd.to_datetime(cqr["c1_last_target"]) + lp.ONE_HOUR <= C1_END).all()
        cqr.to_csv(CQR_C1_PATH, index=False)
        write_model_label(CQR_C1_DIR, tag, stage="calibration")
        print("Training q02/q05/q95/q98 on the training split, CQR on C1 ...")
        train_tail_models(df30, tag)

    if not run_diagnose:
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    origins = c1_origins(hourly, calib_start)
    ci = hourly.set_index("timestamp")["carbon_intensity"]
    y = np.stack([ci.loc[o: o + 23 * lp.ONE_HOUR].to_numpy(float) for o in origins])
    tables = quantile_tables(hourly, origins, tag, sc.LEVELS_V2)

    rows, hists = [], {}
    # v1, as frozen in Stage 1 v1
    v1_rho = json.loads(STAGE1_V1.read_text(encoding="utf-8"))["rho"]
    v1_vals = stack_levels(quantile_tables(hourly, origins, tag, sc.LEVELS_V1), sc.LEVELS_V1)
    r, hists["v1"], _ = diagnose("v1", v1_vals, sc.LEVELS_V1, y, origins, rho=v1_rho)
    rows.append(r)

    # (i) 7 levels + AR(1), rho re-estimated on C1 from the new PIT values
    v2_vals = stack_levels(tables, sc.LEVELS_V2)
    u_i, _ = pit_values(v2_vals, sc.LEVELS_V2, y)
    z_i = z_from_u(u_i)
    rho_i = sc.estimate_rho(z_i)
    r, hists["i"], _ = diagnose("i", v2_vals, sc.LEVELS_V2, y, origins, rho=rho_i)
    rows.append(r)
    chosen = {"model": "i", "levels": list(sc.LEVELS_V2), "copula": "ar1", "rho": rho_i} \
        if r["all_pass"] else None

    corr_ii = None
    if chosen is None and r["A1_pass"] and r["A3_pass"]:
        corr_ii, shrink = shrunk_correlation(z_i)
        r, hists["ii"], _ = diagnose("ii", v2_vals, sc.LEVELS_V2, y, origins, corr=corr_ii)
        r["ledoit_wolf_shrinkage"] = shrink
        rows.append(r)
        if r["all_pass"]:
            chosen = {"model": "ii", "levels": list(sc.LEVELS_V2), "copula": "empirical",
                      "corr_path": str((OUT_DIR / "corr_c1.csv").relative_to(PROJECT_ROOT)),
                      "ledoit_wolf_shrinkage": shrink}
            pd.DataFrame(corr_ii).to_csv(OUT_DIR / "corr_c1.csv", index=False)

    diag = pd.DataFrame(rows)
    diag.to_csv(OUT_DIR / "c1_diagnostics.csv", index=False)
    pd.DataFrame({k: v for k, v in hists.items()},
                 index=[f"{b / 10:.1f}-{(b + 1) / 10:.1f}" for b in range(10)]).rename_axis(
        "u_bin").to_csv(OUT_DIR / "pit_histograms.csv")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                            cwd=PROJECT_ROOT).stdout.strip()
    (OUT_DIR / "scenario_model.json").write_text(json.dumps({
        "chosen": chosen, "decided_by": "C1 diagnostics A1-A3 only (spec §1a)",
        "code_commit_at_run": commit, "data_tag": tag,
        "c1_origins": [str(origins.min()), str(origins.max())], "n_c1_origins": len(origins),
        "diagnostics": rows}, indent=2, default=str), encoding="utf-8")
    print(diag.round(4).T.to_string())
    print("\nPIT histograms (share per decile):")
    print(pd.read_csv(OUT_DIR / "pit_histograms.csv").round(4).to_string(index=False))
    print(f"\nChosen scenario model: {chosen}")


if __name__ == "__main__":
    main(train="--train-tails" in sys.argv, run_diagnose="--diagnose" in sys.argv)
