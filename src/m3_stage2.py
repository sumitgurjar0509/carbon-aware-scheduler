"""
M3 Stage 2: the ONE test-period run, with the frozen parameters
===============================================================
docs/m3_spec.md §4-§6. Run it yourself, once:

    $env:CAS_DATA_TAG = "20250923_20260923"
    .\\venv\\Scripts\\python.exe -u src\\m3_stage2.py --log results\\tables\\m3\\test\\stage2_log.txt

Safety rules (checked BEFORE any test data is read):
  1. src/ and tests/ must have no uncommitted changes; the commit is recorded.
  2. results/tables/m3/test/ must not exist or must be empty. The first thing
     the run writes there is RUN_STARTED.json, so even a crashed run blocks a
     second one. Re-running needs a deliberate manual deletion.
  3. The log file is opened in exclusive-create mode: it is never overwritten.
All parameters come from results/tables/m3/frozen_params.json; nothing is
chosen here. Jobs are M2's 2,359 test jobs (seed 20260923), regenerated and
checked byte-for-byte against results/tables/m2_base_lp/jobs_seed20260923.csv.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lp_scheduler as lp  # noqa: E402
import m3_scheduler as m3  # noqa: E402
from data_config import get_data_tag  # noqa: E402
from m3_tuning import git_head  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "results" / "tables" / "m3"
TEST_DIR = OUT_DIR / "test"
FROZEN = OUT_DIR / "frozen_params.json"
M2_DIR = PROJECT_ROOT / "results" / "tables" / "m2_base_lp"
STARTED, COMPLETED = "RUN_STARTED.json", "RUN_COMPLETED.json"
ALPHAS = (1.0, 0.5)
RU_CHECK = 200


# ----------------------------------------------------------------------
# Guards
# ----------------------------------------------------------------------
def refuse_if_dirty(clean: bool):
    if not clean:
        raise SystemExit("REFUSED: src/ or tests/ has uncommitted changes. Commit first.")


def refuse_if_results_exist(test_dir: Path = TEST_DIR):
    if test_dir.exists() and any(test_dir.iterdir()):
        raise SystemExit(f"REFUSED: {test_dir} already contains files "
                         f"({', '.join(sorted(p.name for p in test_dir.iterdir()))}). "
                         "The test period is run only once. Nothing was changed.")


def same_csv_text(a: str, b: str) -> bool:
    """Equal CSV text, ignoring Windows/Unix line endings."""
    return a.replace("\r\n", "\n").strip() == b.replace("\r\n", "\n").strip()


class Tee:
    """Writes to the console and to a log file opened in exclusive-create mode."""

    def __init__(self, path: Path):
        self.file = open(path, "x", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, text):
        self.stdout.write(text)
        self.file.write(text)
        self.file.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()


# ----------------------------------------------------------------------
# The planned variants (spec §4), from the frozen parameters only
# ----------------------------------------------------------------------
def planned_variants(frozen: dict) -> tuple[list, dict]:
    """
    Returns (runs, aliases). Each run: dict(name, kind, table, beta, replan, alpha).
    kind: "fifo", "actual", "point" (table "raw" or "recal") or "scenario".
    When a selected beta is 0, that cvar variant IS the mean variant; it is
    recorded as an alias instead of being run twice.
    """
    runs = [{"name": "random_fifo", "kind": "fifo", "alpha": None}]
    aliases = {}
    b_lp, b_rp = frozen["beta_cvar_lp"], frozen["beta_cvar_lp_replan"]
    b_pos_lp, b_pos_rp = frozen["best_positive_beta_cvar_lp"], frozen["best_positive_beta_cvar_lp_replan"]
    b_tail = frozen["beta_tail_cvar_lp_replan"]
    for a in ALPHAS:
        sfx = f"_a{a:g}"
        add = lambda name, **kw: runs.append({"name": name + sfx, "alpha": a, **kw})  # noqa: E731
        add("actual_lp", kind="actual", beta=0.0, replan=False)
        add("base_lp", kind="point", table="raw", beta=0.0, replan=False)
        add("base_lp_replan", kind="point", table="raw", beta=0.0, replan=True)
        add("base_lp_recal", kind="point", table="recal", beta=0.0, replan=False)
        add("base_lp_replan_recal", kind="point", table="recal", beta=0.0, replan=True)
        add("mean_lp", kind="scenario", beta=0.0, replan=False)
        add("mean_lp_replan", kind="scenario", beta=0.0, replan=True)
        for name, beta, replan, base in [("cvar_lp", b_lp, False, "mean_lp"),
                                         ("cvar_lp_replan", b_rp, True, "mean_lp_replan")]:
            if beta == 0:
                aliases[name + sfx] = base + sfx
            else:
                add(name, kind="scenario", beta=beta, replan=replan)
        if b_lp == 0:
            add("cvar_lp_bestpos", kind="scenario", beta=b_pos_lp, replan=False)
        if b_rp == 0:
            add("cvar_lp_replan_bestpos", kind="scenario", beta=b_pos_rp, replan=True)
        if b_tail is not None:
            add("cvar_lp_replan_tail", kind="scenario", beta=b_tail, replan=True)
    return runs, aliases


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main(log_path: Path):
    commit, clean = git_head()
    refuse_if_dirty(clean)
    refuse_if_results_exist()
    frozen_bytes = FROZEN.read_bytes()
    frozen = json.loads(frozen_bytes)
    tag = get_data_tag()
    if frozen["data_tag"] != tag:
        raise SystemExit(f"REFUSED: frozen parameters are for {frozen['data_tag']}, CAS_DATA_TAG={tag}")

    TEST_DIR.mkdir(parents=True, exist_ok=True)
    with open(TEST_DIR / STARTED, "x", encoding="utf-8") as f:
        json.dump({"started_at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
                   "code_commit_at_run": commit,
                   "frozen_params_sha256": hashlib.sha256(frozen_bytes).hexdigest()}, f, indent=2)
    if log_path is not None:
        sys.stdout = Tee(Path(log_path))
    print(f"Stage 2 started; code {commit}; frozen params sha256 "
          f"{hashlib.sha256(frozen_bytes).hexdigest()[:12]}", flush=True)

    # Heavy imports only after the guards.
    from forecasting import load_clean_data
    from hourly_forecasting import forecast_tables, m1_boundaries, to_hourly
    from m3_scenario_model import quantile_tables

    df30 = load_clean_data()
    hourly = to_hourly(df30)
    _, test_start, test_end = m1_boundaries(df30)
    ci = hourly.set_index("timestamp")["carbon_intensity"]
    ci = ci[(ci.index >= test_start) & (ci.index + lp.ONE_HOUR <= test_end + pd.Timedelta(minutes=30))]
    jobs = lp.generate_jobs(ci.index.min(), ci.index.max(), window_start=test_start)
    if not same_csv_text(jobs.to_csv(index=False),
                         (M2_DIR / f"jobs_seed{lp.JOB_SEED}.csv").read_bytes().decode("utf-8")):
        raise SystemExit("Regenerated test jobs differ from M2's jobs file -- stopping.")
    jobs.to_csv(TEST_DIR / f"jobs_seed{lp.JOB_SEED}.csv", index=False)

    model = frozen["scenario_model"]
    levels, method = tuple(model["levels"]), model.get("method", "cqr")
    origins = pd.DatetimeIndex(ci.index)
    quant = quantile_tables(hourly, origins, tag, levels, method)
    tables = {"raw": forecast_tables(hourly, origins, tag=tag)["q50"],
              "recal": quantile_tables(hourly, origins, tag, levels, "recal")["q50"]}
    S = int(frozen["S"])
    print(f"Test hours {ci.index.min()} -> {ci.index.max()} ({len(ci)} h); jobs {len(jobs)}; "
          f"model {model['model']} rho={model.get('rho')}; S={S}", flush=True)

    plan, aliases = planned_variants(frozen)
    results, diags = {}, {}
    t_all = time.perf_counter()
    for v in plan:
        t0 = time.perf_counter()
        if v["kind"] == "fifo":
            res, d = lp.run_random_fifo(jobs, ci), {}
        elif v["kind"] == "actual":
            res, d = m3.simulate(jobs, ci, m3.Planner("actual", ci), alpha=v["alpha"], label=v["name"])
        else:
            planner = m3.Planner(v["kind"], ci, table=tables.get(v.get("table")), quantiles=quant,
                                 rho=model.get("rho", 0.0), levels=levels,
                                 n_scenarios=S if v["kind"] == "scenario" else 1)
            res, d = m3.simulate(jobs, ci, planner, alpha=v["alpha"], beta=v["beta"],
                                 replan=v["replan"], label=v["name"],
                                 ru_check=RU_CHECK if v["kind"] == "scenario" and v["beta"] > 0 else 0)
        d["wall_s"] = time.perf_counter() - t0
        res["policy"] = v["name"]
        results[v["name"]], diags[v["name"]] = res, d
        print(f"  {v['name']:<32} total {res.emissions_g.sum() / 1e6:8.3f} t  "
              f"{d['wall_s'] / 60:5.1f} min  bf_dis {d.get('bf_disagree', '-')}  "
              f"ru {d.get('ru_checked', 0)}/{d.get('ru_disagree_option', 0)}", flush=True)

    # M2 reproduction checks (same jobs, same code paths)
    m2 = pd.read_csv(M2_DIR / "per_job_results.csv")
    repro = {}
    for mine, theirs in [("random_fifo", "random_fifo"), ("base_lp_a1", "lp_forecast_a1"),
                         ("base_lp_a0.5", "lp_forecast_a0.5"), ("actual_lp_a1", "lp_actual_a1"),
                         ("actual_lp_a0.5", "lp_actual_a0.5")]:
        a = m2[m2.policy == theirs].set_index("job_id")
        b = results[mine].set_index("job_id").loc[a.index]
        repro[f"{mine} == M2 {theirs}"] = bool(
            (a["cluster"].to_numpy() == b["cluster"].to_numpy()).all()
            and np.allclose(a["emissions_g"].to_numpy(), b["emissions_g"].to_numpy()))
    print(f"M2 reproduction: {repro}", flush=True)

    per_job = pd.concat(results.values(), ignore_index=True)
    per_job.to_csv(TEST_DIR / "per_job_results.csv", index=False)
    summaries = []
    for a in ALPHAS:
        sfx = f"_a{a:g}"
        runs_a = {k: v for k, v in results.items() if k == "random_fifo" or k.endswith(sfx)}
        for base, label in [(f"base_lp{sfx}", "H1_reference_base_lp"),
                            (f"base_lp_replan{sfx}", "H2_reference_base_lp_replan")]:
            s = m3.summarise(runs_a, "random_fifo", base, f"actual_lp{sfx}")
            s.insert(0, "alpha", a)
            s.insert(1, "reference", label)
            for alias, target in aliases.items():
                if alias.endswith(sfx):
                    row = s[s.policy == target].copy()
                    row["policy"] = f"{alias} (= {target}, selected beta 0)"
                    s = pd.concat([s, row], ignore_index=True)
            summaries.append(s)
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(TEST_DIR / "summary.csv", index=False)

    def total(name):
        return results[name]["emissions_g"].sum() / 1e6

    effects = {}
    for a in ALPHAS:
        sfx = f"_a{a:g}"
        cv_rp = aliases.get(f"cvar_lp_replan{sfx}", f"cvar_lp_replan{sfx}")
        e = {"H1_cvar_lp_replan_minus_base_lp_tCO2": total(cv_rp) - total(f"base_lp{sfx}"),
             "H2_cvar_lp_replan_minus_base_lp_replan_tCO2": total(cv_rp) - total(f"base_lp_replan{sfx}"),
             "bias_correction_no_replan_tCO2": total(f"base_lp_recal{sfx}") - total(f"base_lp{sfx}"),
             "bias_correction_replan_tCO2": total(f"base_lp_replan_recal{sfx}") - total(f"base_lp_replan{sfx}")}
        for v in plan:
            if v["kind"] == "scenario" and v.get("beta", 0) > 0 and v["name"].endswith(sfx):
                ref = f"mean_lp_replan{sfx}" if v["replan"] else f"mean_lp{sfx}"
                e[f"uncertainty_{v['name']}_minus_{ref}_tCO2"] = total(v["name"]) - total(ref)
        effects[f"alpha_{a:g}"] = e

    info = {"code_commit_at_run": commit, "code_clean_at_run": clean, "data_tag": tag,
            "frozen_params_sha256": hashlib.sha256(frozen_bytes).hexdigest(),
            "frozen_params": {k: frozen[k] for k in (
                "scenario_model", "S", "a_cvar", "beta_cvar_lp", "beta_cvar_lp_replan",
                "best_positive_beta_cvar_lp", "best_positive_beta_cvar_lp_replan",
                "beta_tail_cvar_lp_replan", "seeds")},
            "test_hours": [str(ci.index.min()), str(ci.index.max())], "n_jobs": len(jobs),
            "variants": plan, "aliases": aliases, "m2_reproduction": repro,
            "effects_descriptive_no_significance": effects,
            "diagnostics": {k: {kk: vv for kk, vv in d.items() if kk != "alt_agree"}
                            for k, d in diags.items()},
            "wall_time_min": (time.perf_counter() - t_all) / 60,
            "note": "Single test run (one period, one job seed): no significance claims (spec §6, M4)."}
    (TEST_DIR / "run_info.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    with open(TEST_DIR / COMPLETED, "x", encoding="utf-8") as f:
        json.dump({"completed_at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")}, f)
    cols = ["alpha", "reference", "policy", "total_tCO2", "cut_vs_fifo_pct", "gap_closed_pct",
            "regret_rate_vs_base_pct", "cvar10_intensity_g_per_kwh", "mean_delay_h"]
    print("\n" + summary[summary.reference == "H1_reference_base_lp"][cols].round(3).to_string(index=False))
    print(f"\nEffects (descriptive): {json.dumps(effects, indent=1)}")
    print(f"\nStage 2 finished in {info['wall_time_min']:.1f} min. Results in {TEST_DIR}")


if __name__ == "__main__":
    log = None
    if "--log" in sys.argv:
        log = sys.argv[sys.argv.index("--log") + 1]
    main(Path(log) if log else None)
