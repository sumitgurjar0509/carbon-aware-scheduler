"""
Scheduler Evaluation Metrics
==============================
Part of: AI-Based Energy and Carbon Optimization in Cloud Computing Environments

Turns simulator.py's per-job, per-scheduler results into the thesis's
headline numbers, overall and broken down by deadline-slack bucket
(6h/12h/24h, per RQ1):

  - % carbon saved vs run_immediately (the baseline)
  - average and p95 delay
  - deadline violation count (must be ~0 by construction -- flagged
    loudly, not quietly reported, if ever nonzero: see simulator.py's
    module docstring point 5)
  - REGRET: the fraction of jobs where a scheduler did WORSE than just
    running immediately, and the average size of that regret in gCO2 --
    reported specifically for forecast_greedy vs uncertainty_aware,
    since this is the direct empirical answer to RQ3 (does trusting an
    uncertain forecast, vs hedging against its uncertainty, produce
    outcomes that are ever worse than not shifting the job at all?)
  - Gap to oracle: (oracle_savings_pct - scheduler_savings_pct) /
    oracle_savings_pct -- what fraction of the theoretically achievable
    ceiling (perfect future knowledge) each realistic scheduler
    captures (RQ2).
  - The observational peak-concurrency metric (job-minutes per
    settlement period) from simulator.py, per scheduler -- NOT broken
    down by slack bucket, since it is a whole-system property.

"% carbon saved" is computed as an AGGREGATE ratio, not an average of
per-job percentages:

    pct_saved = 1 - sum(scheduler_cost) / sum(baseline_cost)

over the relevant job set (a bucket, or all jobs). This weights jobs by
how much carbon they actually involve (duration x intensity), which is
what actually matters for total emissions -- an unweighted average of
per-job percentages would let a handful of tiny jobs with large relative
swings dominate the headline number even though they barely move total
emissions.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from schedulers import SCHEDULERS  # noqa: E402
from simulator import run_simulation  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
PLOTS_DIR = PROJECT_ROOT / "results" / "plots"

BASELINE = "run_immediately"
BUCKET_LABELS = {6: "6h", 12: "12h", 24: "24h"}


def _job_ids_for_bucket(slack_map: pd.Series, bucket: str):
    if bucket == "overall":
        return slack_map.index
    hours = {v: k for k, v in BUCKET_LABELS.items()}[bucket]
    return slack_map[slack_map == hours].index


def compute_metrics_table(combined_df: pd.DataFrame, scheduler_names=None) -> pd.DataFrame:
    """
    `scheduler_names` defaults to the 5 fixed schedulers in
    `schedulers.SCHEDULERS`, but accepts any explicit list of scheduler
    labels present in `combined_df["scheduler"]` -- reused as-is by
    `risk_sweep.py` for the risk_weight frontier, whose per-weight
    scheduler labels ("risk_0.0", "risk_0.2", ...) aren't in that dict.
    `run_immediately` must still be present in combined_df (as the
    baseline) even if excluded from `scheduler_names` itself.
    """
    if scheduler_names is None:
        scheduler_names = list(SCHEDULERS.keys())

    baseline_cost = combined_df.loc[
        combined_df["scheduler"] == BASELINE
    ].set_index("job_id")["realized_carbon_cost"]

    slack_map = combined_df.drop_duplicates("job_id").set_index("job_id")["slack_hours"]
    buckets = ["6h", "12h", "24h", "overall"]

    rows = []
    for bucket in buckets:
        job_ids = _job_ids_for_bucket(slack_map, bucket)
        for scheduler in scheduler_names:
            sub = combined_df[
                (combined_df["scheduler"] == scheduler) & (combined_df["job_id"].isin(job_ids))
            ].set_index("job_id").loc[job_ids]

            total_cost = sub["realized_carbon_cost"].sum()
            total_baseline = baseline_cost.loc[job_ids].sum()
            pct_saved = 0.0 if scheduler == BASELINE else 1 - total_cost / total_baseline

            diff = sub["realized_carbon_cost"] - baseline_cost.loc[job_ids]
            regret_mask = diff > 0
            regret_rate = regret_mask.mean()
            avg_regret_magnitude = diff[regret_mask].mean() if regret_mask.any() else 0.0

            rows.append({
                "scheduler": scheduler,
                "bucket": bucket,
                "n_jobs": len(job_ids),
                "pct_carbon_saved": pct_saved,
                "avg_delay_hours": sub["delay_hours"].mean(),
                "p95_delay_hours": sub["delay_hours"].quantile(0.95),
                "deadline_violations": int(sub["deadline_violation"].sum()),
                "forced_start_count": int(sub["was_forced_start"].sum()),
                "regret_rate": regret_rate,
                "avg_regret_magnitude_gco2": avg_regret_magnitude,
            })

    metrics = pd.DataFrame(rows)

    # Gap to oracle: computed per bucket, relative to oracle's OWN pct_saved
    # in that same bucket (never mixing buckets).
    oracle_pct = metrics[metrics["scheduler"] == "oracle"].set_index("bucket")["pct_carbon_saved"]
    metrics["gap_to_oracle_pct"] = metrics.apply(
        lambda r: np.nan if r["scheduler"] in (BASELINE, "oracle") or oracle_pct[r["bucket"]] == 0
        else (oracle_pct[r["bucket"]] - r["pct_carbon_saved"]) / oracle_pct[r["bucket"]],
        axis=1,
    )

    return metrics


def check_deadline_violations(metrics: pd.DataFrame):
    bad = metrics[metrics["deadline_violations"] > 0]
    if not bad.empty:
        print("\n" + "=" * 70)
        print("!! BUG WARNING: nonzero deadline violations detected -- this should")
        print("   be structurally impossible per simulator.py's design. Do NOT")
        print("   trust these results until this is fixed.")
        print(bad[["scheduler", "bucket", "deadline_violations"]].to_string(index=False))
        print("=" * 70)
    else:
        print("Deadline violation check: 0 violations for every scheduler/bucket (as expected).")


def report_regret_comparison(metrics: pd.DataFrame):
    print("\n=== Regret: forecast_greedy vs uncertainty_aware (the core RQ3 comparison) ===")
    sub = metrics[
        (metrics["scheduler"].isin(["forecast_greedy", "uncertainty_aware"]))
    ][["scheduler", "bucket", "regret_rate", "avg_regret_magnitude_gco2"]]
    print(sub.round(4).to_string(index=False))


def report_gap_to_oracle(metrics: pd.DataFrame):
    print("\n=== Gap to oracle (RQ2: fraction of achievable ceiling captured) ===")
    sub = metrics[
        ~metrics["scheduler"].isin([BASELINE, "oracle"])
    ][["scheduler", "bucket", "pct_carbon_saved", "gap_to_oracle_pct"]]
    print(sub.round(4).to_string(index=False))


def report_concurrency(concurrency_by_scheduler: dict):
    print("\n=== Observational concurrency (job-minutes per 30-min settlement period) ===")
    print("(No capacity constraint is enforced -- this is purely descriptive, see")
    print(" simulator.py point 4. Higher values mean a scheduler bunches more jobs")
    print(" into the same low-carbon windows -- the 'stampede effect' concern.)")
    rows = []
    for name, series in concurrency_by_scheduler.items():
        rows.append({
            "scheduler": name,
            "max_job_minutes": series.max(),
            "p95_job_minutes": series.quantile(0.95),
            "mean_job_minutes": series.mean(),
        })
    conc_df = pd.DataFrame(rows).set_index("scheduler").round(1)
    print(conc_df.to_string())
    return conc_df


def plot_carbon_saved_by_bucket(metrics: pd.DataFrame, out_path: Path):
    """One line per scheduler, carbon saved (%) vs slack bucket."""
    schedulers_to_plot = [s for s in SCHEDULERS if s != BASELINE]
    bucket_order = ["6h", "12h", "24h"]

    # Fixed categorical palette, applied in a stable order across schedulers.
    colors = {
        "threshold": "#eda100",
        "forecast_greedy": "#eb6834",
        "oracle": "#2a78d6",
        "uncertainty_aware": "#4a3aa7",
    }
    INK_SECONDARY = "#52514e"
    INK_MUTED = "#898781"
    GRIDLINE = "#e1e0d9"
    SURFACE = "#fcfcfb"

    fig, ax = plt.subplots(figsize=(8, 5.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    x = np.arange(len(bucket_order))
    for scheduler in schedulers_to_plot:
        sub = metrics[metrics["scheduler"] == scheduler].set_index("bucket").loc[bucket_order]
        ax.plot(x, sub["pct_carbon_saved"] * 100, color=colors[scheduler], linewidth=2,
                marker="o", markersize=6, label=scheduler)

    ax.set_title("Carbon saved (%) vs run_immediately, by deadline-slack bucket",
                 color="#0b0b0b", fontsize=12, fontweight="bold", loc="left")
    ax.set_xlabel("deadline slack", color=INK_SECONDARY, fontsize=9)
    ax.set_ylabel("carbon saved (%)", color=INK_SECONDARY, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(bucket_order)
    ax.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(INK_MUTED)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY, loc="best")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    combined_df, concurrency_by_scheduler, jobs, test_start, test_end = run_simulation()

    metrics = compute_metrics_table(combined_df)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = TABLES_DIR / "scheduler_comparison.csv"
    metrics.to_csv(metrics_path, index=False)
    print(f"\nSaved full results table to {metrics_path}")

    print("\n" + "=" * 100)
    print("=== Full results: all 5 schedulers x {6h, 12h, 24h, overall} ===")
    print("=" * 100)
    display_cols = ["scheduler", "bucket", "n_jobs", "pct_carbon_saved", "avg_delay_hours",
                     "p95_delay_hours", "deadline_violations", "forced_start_count",
                     "regret_rate", "avg_regret_magnitude_gco2", "gap_to_oracle_pct"]
    print(metrics[display_cols].round(4).to_string(index=False))

    check_deadline_violations(metrics)
    report_regret_comparison(metrics)
    report_gap_to_oracle(metrics)
    report_concurrency(concurrency_by_scheduler)

    plot_carbon_saved_by_bucket(metrics, PLOTS_DIR / "carbon_saved_by_bucket.png")
