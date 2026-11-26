"""
M3 Stage 1: parameter selection on C1 / C2 only (docs/m3_spec.md §1a, §5)
=========================================================================
    $env:CAS_DATA_TAG = "20250923_20260923"
    python src/m3_scenario_model.py --train-tails --diagnose   # scenario model, C1 only
    python src/m3_tuning.py --timing     # time solves on the first 100 C2 jobs only
    python src/m3_tuning.py              # full Stage 1 -> results/tables/m3/frozen_params.json

The TEST period is never read here: every series is cut before it starts,
and the code checks this.

Stage 1 v2 (after the scenario-model revision of spec §1a):
  1. The scenario model (levels, copula, rho or correlation matrix) is read
     from results/tables/m3/scenario_model/scenario_model.json, which was
     chosen on C1 diagnostics only.
  2. S on C2: smallest S with >= 99% decision agreement with S = 2000 (beta = 0.5).
  3. beta on C2: lowest total emissions (ties within 0.1% -> smallest beta),
     separately with and without re-planning.
  4. Secondary tail-risk beta (re-planning): beta > 0 with the lowest CVaR_0.1
     per-job emission intensity among those with total <= 1.01 x base_lp_replan.
  5. Write frozen_params.json with the C2 table that justified the choices.
v1 of Stage 1 (3 quantile levels, AR(1)) is kept in results/tables/m3/stage1_v1/.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lp_scheduler as lp  # noqa: E402
import m3_scheduler as m3  # noqa: E402
import scenarios as sc  # noqa: E402
from data_config import get_data_tag  # noqa: E402
from forecasting import load_clean_data  # noqa: E402
from hourly_forecasting import forecast_tables, m1_boundaries, to_hourly  # noqa: E402
from m3_scenario_model import C1_END, CQR_C1_PATH, c1_rows, quantile_tables  # noqa: E402,F401

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "results" / "tables" / "m3"
C2_DIR = OUT_DIR / "tuning_c2"
SCENARIO_MODEL = OUT_DIR / "scenario_model" / "scenario_model.json"

TUNE_SEED = 20260606
BETA_GRID = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
S_GRID = (100, 200, 500, 1000)
S_REF = 2000
S_CHECK_BETA = 0.5
AGREE_MIN = 0.99
BETA_TIE_REL = 0.001       # ties within 0.1% of the best total -> smallest beta
TAIL_MAX_EXCESS = 0.01     # tail-risk beta: total <= 1% above base_lp_replan
RU_CHECK = 200
ALPHA_TUNE = 1.0


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


def select_beta_tail(totals: dict, cvar10: dict, base_replan_total: float,
                     max_excess=TAIL_MAX_EXCESS):
    """
    Spec §5 rule 5: among beta > 0 with total <= (1 + max_excess) x base_lp_replan,
    the lowest CVaR_0.1 per-job intensity; ties -> smallest beta. None if none qualifies.
    """
    ok = [b for b in totals if b > 0 and totals[b] <= base_replan_total * (1 + max_excess)]
    if not ok:
        return None
    best = min(cvar10[b] for b in ok)
    return min(b for b in ok if cvar10[b] == best)


# ----------------------------------------------------------------------
# C2 setup
# ----------------------------------------------------------------------
def load_scenario_model() -> dict:
    model = json.loads(SCENARIO_MODEL.read_text(encoding="utf-8"))
    chosen = model["chosen"]
    if chosen is None:
        raise SystemExit("No scenario model passed the C1 criteria (spec §1a). Stopping.")
    if chosen["copula"] == "empirical":
        chosen["corr"] = pd.read_csv(PROJECT_ROOT / chosen["corr_path"]).to_numpy(float)
    return chosen


def c2_world(df30, tag, levels, method="cqr"):
    """
    Returns the C2 world plus three point tables:
      raw_q50   M2's exact q50 (base_lp, base_lp_replan: H1/H2 unchanged)
      recal_q50 the candidate-(iii) recalibrated q50 (base_lp*_recal controls)
    and `quant`, the scenario model's level tables (method of the chosen model).
    """
    hourly = to_hourly(df30)
    calib_start, test_start, _ = m1_boundaries(df30)
    ci = hourly.set_index("timestamp")["carbon_intensity"]
    ci_c2 = ci[(ci.index >= C1_END) & (ci.index + lp.ONE_HOUR <= test_start)]
    assert ci_c2.index.max() + lp.ONE_HOUR <= test_start, "C2 would touch the test period"
    jobs = lp.generate_jobs(ci_c2.index.min(), ci_c2.index.max(), seed=TUNE_SEED)
    origins = pd.DatetimeIndex(ci_c2.index)
    quant = quantile_tables(hourly, origins, tag, levels, method)
    raw_q50 = forecast_tables(hourly, origins, tag=tag)["q50"]
    recal_q50 = quantile_tables(hourly, origins, tag, sc.LEVELS_V2, "recal")["q50"]
    return hourly, ci_c2, jobs, quant, raw_q50, recal_q50, calib_start, test_start


def run_variant(name, jobs, ci, quant, model, S, beta, replan, ru_check=0, alt_sizes=(),
                table=None):
    kind = "point" if name.startswith("base") else "scenario"
    planner = m3.Planner(kind, ci, table=table, quantiles=quant,
                         rho=model.get("rho", 0.0), corr=model.get("corr"),
                         levels=model["levels"], n_scenarios=S)
    t0 = time.perf_counter()
    res, diag = m3.simulate(jobs, ci, planner, alpha=ALPHA_TUNE, beta=beta, replan=replan,
                            ru_check=ru_check, label=name, alt_sizes=alt_sizes)
    diag["wall_s"] = time.perf_counter() - t0
    print(f"  {name:<28} total {res.emissions_g.sum() / 1e6:8.3f} t  solves {diag['solves']:>6}  "
          f"{diag['wall_s'] / 60:5.1f} min  bf_dis {diag['bf_disagree']}  "
          f"ru {diag['ru_checked']}/{diag['ru_disagree_option']}/{diag['ru_disagree_value']}",
          flush=True)
    return res, diag


def git_head() -> tuple[str, bool]:
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                          cwd=PROJECT_ROOT).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "src", "tests"],
                           capture_output=True, text=True, cwd=PROJECT_ROOT).stdout.strip()
    return head, not dirty


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main(timing_only=False):
    commit, clean = git_head()
    if not clean:
        raise SystemExit("src/ or tests/ has uncommitted changes. Commit before running (spec §5).")
    tag = get_data_tag()
    df30 = load_clean_data()
    model = load_scenario_model()
    C2_DIR.mkdir(parents=True, exist_ok=True)
    hourly, ci_c2, jobs, quant, raw_q50, recal_q50, calib_start, test_start = c2_world(
        df30, tag, model["levels"], model.get("method", "cqr"))
    print(f"Scenario model: {model['model']} ({model['copula']}); "
          f"C2 hours {ci_c2.index.min()} -> {ci_c2.index.max()}; jobs {len(jobs)}")

    if timing_only:
        rows = []
        for name, S, ru in [("base_lp", 1, 0), ("cvar_S1000", 1000, 20), ("cvar_S2000", 2000, 10)]:
            _, d = run_variant(name, jobs.head(100), ci_c2, quant, model, S, S_CHECK_BETA, False,
                               ru_check=ru, table=raw_q50)
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

    # S on C2
    print(f"Choosing S: reference S={S_REF}, beta={S_CHECK_BETA}")
    _, d_ref = run_variant("s_reference", jobs, ci_c2, quant, model, S_REF, S_CHECK_BETA, False,
                           alt_sizes=S_GRID)
    agreement = {s: d_ref["alt_agree"][s] / d_ref["alt_n"] for s in S_GRID}
    S = select_s(agreement)
    pd.DataFrame([{"S": s, "agreement_with_S2000": agreement[s]} for s in S_GRID]).to_csv(
        C2_DIR / "s_selection.csv", index=False)
    print(f"S agreement {agreement} -> S = {S}")

    # beta grid on C2
    runs, diags = {}, {}
    runs["random_fifo"] = lp.run_random_fifo(jobs, ci_c2)
    runs["actual_lp"], _ = lp.run_lp(jobs, ci_c2, ALPHA_TUNE, "actual")
    m2_check, _ = lp.run_lp(jobs, ci_c2, ALPHA_TUNE, "forecast", raw_q50)
    for replan in (False, True):
        sfx = "_replan" if replan else ""
        runs[f"base_lp{sfx}"], diags[f"base_lp{sfx}"] = run_variant(
            f"base_lp{sfx}", jobs, ci_c2, quant, model, 1, 0.0, replan, table=raw_q50)
        runs[f"base_lp{sfx}_recal"], diags[f"base_lp{sfx}_recal"] = run_variant(
            f"base_lp{sfx}_recal", jobs, ci_c2, quant, model, 1, 0.0, replan, table=recal_q50)
        for beta in BETA_GRID:
            name = f"cvar_b{beta:g}{sfx}"
            runs[name], diags[name] = run_variant(name, jobs, ci_c2, quant, model, S, beta, replan,
                                                  ru_check=RU_CHECK if beta > 0 else 0)
    a = m2_check.set_index("job_id")[["cluster", "start_hour"]]
    b = runs["base_lp"].set_index("job_id").loc[a.index, ["cluster", "start_hour"]]
    m2_identical = bool(a.equals(b))
    print(f"base_lp (M3 engine) identical to M2 lp_scheduler on C2: {m2_identical}")

    summary = m3.summarise(runs, "random_fifo", "base_lp", "actual_lp")
    summary["beta"] = [float(p.split("cvar_b")[1].split("_")[0]) if p.startswith("cvar_b")
                       else np.nan for p in summary.policy]
    summary["replan"] = [p.endswith("_replan") for p in summary.policy]
    for col in ("solves", "bf_disagree", "ru_checked", "ru_disagree_option", "ru_disagree_value",
                "n_floored", "wall_s"):
        summary[col] = [diags[p].get(col) if p in diags else np.nan for p in summary.policy]
    summary.to_csv(C2_DIR / "beta_grid.csv", index=False)
    pd.concat(runs.values(), ignore_index=True).to_csv(C2_DIR / "per_job_results_c2.csv", index=False)
    mismatches = {p: d.get("ru_mismatches", []) for p, d in diags.items() if d.get("ru_mismatches")}
    (C2_DIR / "ru_mismatches.json").write_text(json.dumps(mismatches, indent=2, default=str),
                                               encoding="utf-8")

    s_ = summary.set_index("policy")
    chosen = {}
    for replan in (False, True):
        sfx = "_replan" if replan else ""
        chosen[replan] = select_beta({b: s_.loc[f"cvar_b{b:g}{sfx}", "total_tCO2"] for b in BETA_GRID})
    beta_tail = select_beta_tail(
        {b: s_.loc[f"cvar_b{b:g}_replan", "total_tCO2"] for b in BETA_GRID},
        {b: s_.loc[f"cvar_b{b:g}_replan", "cvar10_intensity_g_per_kwh"] for b in BETA_GRID},
        s_.loc["base_lp_replan", "total_tCO2"])

    def eff(a, b):  # total of a minus total of b, in tCO2 and % of b
        ta, tb = s_.loc[a, "total_tCO2"], s_.loc[b, "total_tCO2"]
        return {"minus": f"{a} - {b}", "tCO2": ta - tb, "pct": 100 * (ta / tb - 1)}
    effects = {
        "bias_correction": {"no_replan": eff("base_lp_recal", "base_lp"),
                            "replan": eff("base_lp_replan_recal", "base_lp_replan")},
        "uncertainty_cvar_vs_mean": {
            f"beta_{b:g}{sfx}": eff(f"cvar_b{b:g}{sfx}", f"cvar_b0{sfx}")
            for sfx in ("", "_replan") for b in BETA_GRID if b > 0},
    }
    model_out = {k: v for k, v in model.items() if k != "corr"}
    frozen = {
        "version": "stage1_v2 (scenario model revised after Stage 1 v1, before any test run)",
        "frozen_at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
        "code_commit_at_run": commit, "code_clean_at_run": clean,
        "data_tag": tag,
        "scenario_model": model_out, "S": S, "a_cvar": m3.A_CVAR,
        "beta_cvar_lp": chosen[False][0],
        "beta_cvar_lp_replan": chosen[True][0],
        "best_positive_beta_cvar_lp": chosen[False][1],
        "best_positive_beta_cvar_lp_replan": chosen[True][1],
        "beta_tail_cvar_lp_replan": beta_tail,
        "selection_rules": {
            "scenario_model": "first candidate passing A1-A3 on C1 (spec §1a)",
            "S": f"smallest S in {list(S_GRID)} with >= {AGREE_MIN:.0%} decision agreement "
                 f"with S={S_REF} on C2 (beta={S_CHECK_BETA}, no re-planning); S={S_REF} if none",
            "beta": f"lowest total C2 emissions over {list(BETA_GRID)}, alpha={ALPHA_TUNE}; "
                    f"ties within {BETA_TIE_REL:.1%} -> smallest beta; separately per re-plan setting",
            "beta_tail": f"re-planning: beta > 0 with lowest CVaR_0.1 per-job intensity among those "
                         f"with total <= {1 + TAIL_MAX_EXCESS:.2f} x base_lp_replan; ties -> smallest",
            "a": "fixed at 0.2 (not tuned)", "ru_check": "CBC gapRel = 0"},
        "seeds": {"tune_jobs": TUNE_SEED, "fifo": lp.FIFO_SEED, "scenarios": sc.SCENARIO_SEED,
                  "test_jobs": lp.JOB_SEED},
        "periods": {"C1": [str(calib_start), str(C1_END)],
                    "C2_hours": [str(ci_c2.index.min()), str(ci_c2.index.max())],
                    "test_start_not_used": str(test_start)},
        "s_agreement": {str(k): v for k, v in agreement.items()},
        "c2_effects": effects,
        "m3_engine_reproduces_m2_on_c2": m2_identical,
        "c2_n_jobs": len(jobs),
        "c2_table": summary.round(6).replace({np.nan: None}).to_dict(orient="records"),
    }
    (OUT_DIR / "frozen_params.json").write_text(json.dumps(frozen, indent=2, default=str),
                                                encoding="utf-8")
    print("\n" + summary[["policy", "total_tCO2", "cut_vs_fifo_pct", "gap_closed_pct",
                          "regret_rate_vs_base_pct", "cvar10_intensity_g_per_kwh",
                          "mean_delay_h"]].round(3).to_string(index=False))
    print(f"\nFrozen: S={S}, beta (no re-plan)={chosen[False]}, beta (re-plan)={chosen[True]}, "
          f"beta_tail={beta_tail}")


if __name__ == "__main__":
    main(timing_only="--timing" in sys.argv)
