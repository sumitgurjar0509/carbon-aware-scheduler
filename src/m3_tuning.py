"""
M3 Stage 1: parameter selection on C1 / C2 only (docs/m3_spec.md §5)
====================================================================
    $env:CAS_DATA_TAG = "20250923_20260923"
    python src/m3_tuning.py --timing     # time solves on the first 100 C2 jobs only
    python src/m3_tuning.py              # full Stage 1 -> results/tables/m3/frozen_params.json

The TEST period is never read here: every series is cut before it starts,
and the code checks this.

Steps:
  1. Re-fit the CQR corrections on C1 only -> results/models/m3_cqr_c1/
  2. Estimate rho by AR(1) maximum likelihood on C1 forecast errors (PIT -> z)
  3. Choose S on C2 by Monte Carlo agreement with S = 2000 (beta = 0.5, no re-plan)
  4. Choose beta on C2 with the declared rule, separately with and without re-planning
  5. Write frozen_params.json (with the C2 table that justified the choice)
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lp_scheduler as lp  # noqa: E402
import m3_scheduler as m3  # noqa: E402
import scenarios as sc  # noqa: E402
from calibration import conformal_correction, nonconformity_scores  # noqa: E402
from data_config import get_data_tag, write_model_label  # noqa: E402
from forecasting import get_feature_columns, load_clean_data  # noqa: E402
from hourly_forecasting import (  # noqa: E402
    ALPHA as CQR_ALPHA,
    HOURLY_HORIZONS,
    HOURLY_MODELS_DIR,
    forecast_tables,
    horizon_dataset,
    hourly_features,
    m1_boundaries,
    to_hourly,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "results" / "tables" / "m3"
C1_DIR, C2_DIR = OUT_DIR / "tuning_c1", OUT_DIR / "tuning_c2"
CQR_C1_DIR = PROJECT_ROOT / "results" / "models" / "m3_cqr_c1"
CQR_C1_PATH = CQR_C1_DIR / "cqr_corrections.csv"

C1_END = pd.Timestamp("2026-06-06 00:00", tz="UTC")   # approved split
TUNE_SEED = 20260606
BETA_GRID = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
S_GRID = (100, 200, 500, 1000)
S_REF = 2000
S_CHECK_BETA = 0.5
AGREE_MIN = 0.99
BETA_TIE_REL = 0.001       # ties within 0.1% of the best total -> smallest beta
RU_CHECK = 200
ALPHA_TUNE = 1.0


# ----------------------------------------------------------------------
# 1-2. C1: CQR re-fit and rho
# ----------------------------------------------------------------------
def c1_rows(df_h: pd.DataFrame, calib_start, c1_end) -> pd.DataFrame:
    """Rows whose origin is in C1 and whose whole target hour ends by C1_END."""
    keep = (df_h["timestamp"] >= calib_start) & (df_h["target_timestamp"] + lp.ONE_HOUR <= c1_end)
    return df_h[keep].reset_index(drop=True)


def refit_cqr_c1(hourly, calib_start, c1_end, models_dir=HOURLY_MODELS_DIR) -> pd.DataFrame:
    """Same CQR as M2, fitted on C1 rows only, with the unchanged m2_hourly models."""
    feat = hourly_features(hourly)
    cols = get_feature_columns(feat)
    rows = []
    for h in HOURLY_HORIZONS:
        rows_h = c1_rows(horizon_dataset(feat, hourly, h), calib_start, c1_end)
        pred = {q: lgb.Booster(model_file=str(models_dir / f"h{h:02d}h_q{int(q * 100):02d}.txt"))
                .predict(rows_h[cols]) for q in (0.1, 0.9)}
        scores = nonconformity_scores(pred[0.1], pred[0.9], rows_h["target"])
        rows.append({"horizon_h": h, "cqr_correction": conformal_correction(scores, CQR_ALPHA),
                     "n_calib": len(rows_h), "alpha": CQR_ALPHA,
                     "c1_last_target": str(rows_h["target_timestamp"].max())})
    return pd.DataFrame(rows)


def c1_pit(hourly, calib_start, c1_end, tag):
    """z matrix (origins x 24) on C1 with the C1 corrections. Returns (z, n_clipped, origins)."""
    ci = hourly.set_index("timestamp")["carbon_intensity"]
    origins = ci.index[(ci.index >= calib_start) & (ci.index + 24 * lp.ONE_HOUR <= c1_end)]
    feat_ok = hourly_features(hourly)["timestamp"]
    origins = origins.intersection(pd.DatetimeIndex(feat_ok))
    q = forecast_tables(hourly, origins, tag=tag, corrections_path=CQR_C1_PATH)
    z = np.empty((len(origins), 24))
    clipped = 0
    for r, o in enumerate(origins):
        knots = sc.quantile_knots(*(q[k].loc[o].to_numpy(float) for k in ("q10", "q50", "q90")))
        y = ci.loc[o: o + 23 * lp.ONE_HOUR].to_numpy(float)
        z[r], hit = sc.pit_z(y, knots)
        clipped += hit
    return z, clipped, origins


# ----------------------------------------------------------------------
# Selection rules (declared in the spec; tested)
# ----------------------------------------------------------------------
def select_s(agreement: dict, s_grid=S_GRID, threshold=AGREE_MIN) -> int:
    """Smallest S whose agreement with S_REF is >= threshold; S_REF if none."""
    ok = [s for s in s_grid if agreement[s] >= threshold]
    return min(ok) if ok else S_REF


def select_beta(totals: dict, tie_rel=BETA_TIE_REL):
    """
    totals: beta -> total emissions. Lowest total wins; any beta within
    tie_rel of the best counts as tied, and the smallest tied beta is chosen.
    Returns (chosen beta, best beta > 0 by the same rule over positive betas).
    """
    def pick(cands):
        best = min(totals[b] for b in cands)
        return min(b for b in cands if totals[b] <= best * (1 + tie_rel))
    return pick(list(totals)), pick([b for b in totals if b > 0])


# ----------------------------------------------------------------------
# C2 setup
# ----------------------------------------------------------------------
def c2_world(df30, tag):
    hourly = to_hourly(df30)
    calib_start, test_start, _ = m1_boundaries(df30)
    ci = hourly.set_index("timestamp")["carbon_intensity"]
    ci_c2 = ci[(ci.index >= C1_END) & (ci.index + lp.ONE_HOUR <= test_start)]
    assert ci_c2.index.max() + lp.ONE_HOUR <= test_start, "C2 would touch the test period"
    jobs = lp.generate_jobs(ci_c2.index.min(), ci_c2.index.max(), seed=TUNE_SEED)
    quant = forecast_tables(hourly, pd.DatetimeIndex(ci_c2.index), tag=tag,
                            corrections_path=CQR_C1_PATH)
    return hourly, ci_c2, jobs, quant, calib_start, test_start


def window_sum_coverage(ci_c2, quant, rho, n_scen=500, width=6):
    """Diagnostic on C2: share of 6-hour actual sums inside the scenario 10-90% range."""
    hits = []
    for o in ci_c2.index[: len(ci_c2) - 24]:
        paths, _ = sc.sample_paths(*(quant[k].loc[o].to_numpy(float) for k in ("q10", "q50", "q90")),
                                   rho=rho, n_scenarios=n_scen, origin=o)
        sums = paths[:, :width].sum(axis=1)
        y = ci_c2.loc[o: o + (width - 1) * lp.ONE_HOUR].sum()
        hits.append(np.quantile(sums, 0.1) <= y <= np.quantile(sums, 0.9))
    return float(np.mean(hits))


def run_variant(name, jobs, ci, quant, rho, S, beta, replan, ru_check=0, alt_sizes=()):
    kind = "point" if name.startswith("base") else "scenario"
    planner = m3.Planner(kind, ci, table=quant["q50"], quantiles=quant, rho=rho, n_scenarios=S)
    t0 = time.perf_counter()
    res, diag = m3.simulate(jobs, ci, planner, alpha=ALPHA_TUNE, beta=beta, replan=replan,
                            ru_check=ru_check, label=name, alt_sizes=alt_sizes)
    diag["wall_s"] = time.perf_counter() - t0
    print(f"  {name:<28} total {res.emissions_g.sum() / 1e6:8.3f} t  solves {diag['solves']:>6}  "
          f"{diag['wall_s'] / 60:5.1f} min  bf_dis {diag['bf_disagree']}  "
          f"ru {diag['ru_checked']}/{diag['ru_disagree_option']}", flush=True)
    return res, diag


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main(timing_only=False):
    tag = get_data_tag()
    df30 = load_clean_data()
    hourly = to_hourly(df30)
    calib_start, test_start, _ = m1_boundaries(df30)
    for d in (C1_DIR, C2_DIR, CQR_C1_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # 1. CQR on C1
    cqr = refit_cqr_c1(hourly, calib_start, C1_END)
    assert (pd.to_datetime(cqr["c1_last_target"]) + lp.ONE_HOUR <= C1_END).all()
    cqr.to_csv(CQR_C1_PATH, index=False)
    write_model_label(CQR_C1_DIR, tag, stage="calibration")
    cqr.to_csv(C1_DIR / "cqr_corrections_c1.csv", index=False)
    print(f"CQR re-fitted on C1: {int(cqr.n_calib.min())}-{int(cqr.n_calib.max())} rows per horizon")

    # 2. rho on C1
    z, clipped, origins = c1_pit(hourly, calib_start, C1_END, tag)
    rho = sc.estimate_rho(z)
    lags = sc.lag_correlations(z, 6)
    rho_info = {"rho": rho, "method": "AR(1) maximum likelihood on z = Phi^-1(F(y)), C1 only",
                "n_origins": len(origins), "origins": [str(origins.min()), str(origins.max())],
                "n_values": int(z.size), "n_clipped_outside_tails": clipped,
                "share_clipped": clipped / z.size,
                "lag_corr_empirical": lags, "lag_corr_ar1": {k: rho ** k for k in lags}}
    (C1_DIR / "rho_estimate.json").write_text(json.dumps(rho_info, indent=2), encoding="utf-8")
    print(f"rho (C1) = {rho:.4f}; empirical lag-1..3 corr "
          f"{[round(lags[k], 3) for k in (1, 2, 3)]}; clipped {clipped}/{z.size}")

    # C2 world
    hourly, ci_c2, jobs, quant, _, _ = c2_world(df30, tag)
    print(f"C2 hours {ci_c2.index.min()} -> {ci_c2.index.max()} ({len(ci_c2)} h); jobs {len(jobs)}")

    if timing_only:
        rows = []
        sub = jobs.head(100)
        for name, S, ru in [("base_lp", 1, 0), ("cvar_S200", 200, 100), ("cvar_S1000", 1000, 20)]:
            _, d = run_variant(name, sub, ci_c2, quant, rho, S, S_CHECK_BETA, False, ru_check=ru)
            rows.append({"config": name, "S": S, "solves": d["solves"],
                         "per_option_solve_s": d["solve_time_s"] / d["solves"],
                         "wall_per_solve_s": d["wall_s"] / d["solves"],
                         "ru_mip_checked": d["ru_checked"],
                         "ru_mip_solve_s": d["ru_time_s"] / d["ru_checked"] if d["ru_checked"] else np.nan,
                         "ru_disagree_option": d["ru_disagree_option"]})
        t = pd.DataFrame(rows)
        t.to_csv(C2_DIR / "timing.csv", index=False)
        print(t.round(4).to_string(index=False))
        return

    # 3. S on C2 (beta = 0.5, no re-planning, reference S = 2000)
    print(f"Choosing S: reference S={S_REF}, beta={S_CHECK_BETA}")
    _, d_ref = run_variant("s_reference", jobs, ci_c2, quant, rho, S_REF, S_CHECK_BETA, False,
                           alt_sizes=S_GRID)
    agreement = {s: d_ref["alt_agree"][s] / d_ref["alt_n"] for s in S_GRID}
    S = select_s(agreement)
    pd.DataFrame([{"S": s, "agreement_with_S2000": agreement[s]} for s in S_GRID]).to_csv(
        C2_DIR / "s_selection.csv", index=False)
    print(f"S agreement {agreement} -> S = {S}")

    # 4. beta grid on C2
    runs, diags = {}, {}
    runs["random_fifo"] = lp.run_random_fifo(jobs, ci_c2)
    runs["actual_lp"], _ = lp.run_lp(jobs, ci_c2, ALPHA_TUNE, "actual")
    m2_check, _ = lp.run_lp(jobs, ci_c2, ALPHA_TUNE, "forecast", quant["q50"])
    for replan in (False, True):
        sfx = "_replan" if replan else ""
        runs[f"base_lp{sfx}"], diags[f"base_lp{sfx}"] = run_variant(
            f"base_lp{sfx}", jobs, ci_c2, quant, rho, 1, 0.0, replan)
        for beta in BETA_GRID:
            name = f"cvar_b{beta:g}{sfx}"
            runs[name], diags[name] = run_variant(name, jobs, ci_c2, quant, rho, S, beta, replan,
                                                  ru_check=RU_CHECK if beta > 0 else 0)
    a = m2_check.set_index("job_id")[["cluster", "start_hour"]]
    b = runs["base_lp"].set_index("job_id").loc[a.index, ["cluster", "start_hour"]]
    m2_identical = bool(a.equals(b))
    print(f"base_lp (M3 engine) identical to M2 lp_scheduler on C2: {m2_identical}")

    summary = m3.summarise(runs, "random_fifo", "base_lp", "actual_lp")
    for col in ("beta", "replan", "solves", "bf_disagree", "ru_checked", "ru_disagree_option",
                "ru_disagree_value", "n_floored", "wall_s"):
        summary[col] = [np.nan if p not in diags else
                        (diags[p].get(col) if col not in ("beta", "replan") else None)
                        for p in summary.policy]
    summary["beta"] = [float(p.split("_b")[1].split("_")[0]) if p.startswith("cvar_b") else np.nan
                       for p in summary.policy]
    summary["replan"] = [p.endswith("_replan") for p in summary.policy]
    summary.to_csv(C2_DIR / "beta_grid.csv", index=False)
    pd.concat(runs.values(), ignore_index=True).to_csv(C2_DIR / "per_job_results_c2.csv", index=False)

    tot = summary.set_index("policy")["total_tCO2"]
    chosen = {}
    for replan in (False, True):
        sfx = "_replan" if replan else ""
        totals = {b: tot[f"cvar_b{b:g}{sfx}"] for b in BETA_GRID}
        chosen[replan] = select_beta(totals)

    cov_rho = window_sum_coverage(ci_c2, quant, rho)
    cov_ind = window_sum_coverage(ci_c2, quant, 0.0)

    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                            cwd=PROJECT_ROOT).stdout.strip()
    frozen = {
        "frozen_at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
        "code_commit_at_run": commit,
        "data_tag": tag,
        "rho": rho, "S": S, "a_cvar": m3.A_CVAR,
        "beta_cvar_lp": chosen[False][0],
        "beta_cvar_lp_replan": chosen[True][0],
        "best_positive_beta_cvar_lp": chosen[False][1],
        "best_positive_beta_cvar_lp_replan": chosen[True][1],
        "selection_rules": {
            "rho": "AR(1) MLE on C1 PIT z-values",
            "S": f"smallest S in {list(S_GRID)} with >= {AGREE_MIN:.0%} decision agreement "
                 f"with S={S_REF} on C2 (beta={S_CHECK_BETA}, no re-planning)",
            "beta": f"lowest total C2 emissions over {list(BETA_GRID)}, alpha={ALPHA_TUNE}; "
                    f"ties within {BETA_TIE_REL:.1%} -> smallest beta; separately per re-plan setting",
            "a": "fixed at 0.2 (not tuned)"},
        "seeds": {"tune_jobs": TUNE_SEED, "fifo": lp.FIFO_SEED, "scenarios": sc.SCENARIO_SEED,
                  "test_jobs": lp.JOB_SEED},
        "periods": {"C1": [str(calib_start), str(C1_END)],
                    "C2_hours": [str(ci_c2.index.min()), str(ci_c2.index.max())],
                    "test_start_not_used": str(test_start)},
        "cqr_corrections": str(CQR_C1_PATH.relative_to(PROJECT_ROOT)),
        "rho_diagnostics": rho_info,
        "s_agreement": {str(k): v for k, v in agreement.items()},
        "c2_window6h_coverage_80pct": {"with_rho": cov_rho, "rho_0": cov_ind},
        "m3_engine_reproduces_m2_on_c2": m2_identical,
        "c2_n_jobs": len(jobs),
        "c2_table": summary.round(6).replace({np.nan: None}).to_dict(orient="records"),
    }
    (OUT_DIR / "frozen_params.json").write_text(json.dumps(frozen, indent=2, default=str),
                                                encoding="utf-8")
    print("\n" + summary[["policy", "total_tCO2", "cut_vs_fifo_pct", "gap_closed_pct",
                          "regret_rate_vs_base_pct", "cvar10_intensity_g_per_kwh",
                          "mean_delay_h"]].round(3).to_string(index=False))
    print(f"\nFrozen: rho={rho:.4f}, S={S}, beta (no re-plan)={chosen[False]}, "
          f"beta (re-plan)={chosen[True]}")
    print(f"6-h window-sum 80% coverage on C2: rho={cov_rho:.3f}, rho=0: {cov_ind:.3f}")


if __name__ == "__main__":
    main(timing_only="--timing" in sys.argv)
