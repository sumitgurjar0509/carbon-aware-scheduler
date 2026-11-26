"""
Step 7: CVaR-Based Scheduler vs the Threshold-Based Risk Frontier
====================================================================
Part of: AI-Based Energy and Carbon Optimization in Cloud Computing Environments

risk_sweep.py traced one risk-tolerance frontier: make_uncertainty_aware
(bound = q50 + w*(q90 - q50) at the single argmin-q50 candidate) swept
over risk_weight w. This module asks whether a different DECISION RULE
can shift that frontier (more savings at the same regret, or less regret
at the same savings), rather than just move along it:

    make_cvar_aware(alpha=0.2, w): score EVERY remaining candidate by
        (1 - w) * q50 + w * CVaR_alpha, take the lowest, wait iff it
        beats the current actual intensity.

Both sweeps are simulated here in ONE run, against the SAME SimContext
(same forecast tables, same jobs), so the comparison can't be confounded
by a re-trained model or a re-generated trace. risk_sweep.py's own
outputs (risk_sweep_*.csv, risk_frontier_*.png) are not touched.

Read schedulers.cvar's docstring before interpreting the result: under
the three-quantile piecewise-linear approximation, CVaR_0.2 == q90
exactly, so at alpha=0.2 the per-candidate score is identical to
uncertainty_aware's bound and the ONLY difference between the two
frontiers is "min over all candidates" vs "bound at argmin-q50".
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibration import ALPHA, three_way_split  # noqa: E402
from evaluate import compute_metrics_table  # noqa: E402
from forecasting import create_features, load_clean_data  # noqa: E402
from risk_sweep import RISK_WEIGHTS, simulate_with_fn  # noqa: E402
from schedulers import SCHEDULERS, make_cvar_aware, make_uncertainty_aware  # noqa: E402
from simulator import (  # noqa: E402
    PROCESSED_DIR,
    SimContext,
    build_forecast_tables,
    filter_jobs_to_test_period,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
PLOTS_DIR = PROJECT_ROOT / "results" / "plots"

CVAR_ALPHA = ALPHA  # 0.2, the same alpha the CQR intervals are calibrated to
FRONTIERS = {"threshold": "risk_", "cvar": "cvar_"}


def run_cvar_sweep():
    df = load_clean_data()
    df_feat = create_features(df)
    train_feat, _, test_feat = three_way_split(df_feat)
    test_start, test_end = test_feat["timestamp"].min(), test_feat["timestamp"].max()
    threshold_value = train_feat["carbon_intensity"].median()

    workload_df = pd.read_csv(PROCESSED_DIR / "workload_trace.csv",
                               parse_dates=["arrival_time", "deadline"])
    jobs = filter_jobs_to_test_period(workload_df, test_start, test_end)

    print("\nBuilding CQR-calibrated forecast lookup tables (1h-24h horizons)...")
    forecast_tables = build_forecast_tables(df_feat, test_start, test_end)
    actual_series = df.set_index("timestamp")["carbon_intensity"]
    ctx = SimContext(actual_series, forecast_tables, threshold_value)

    all_results = []
    for name in ["run_immediately", "forecast_greedy", "oracle"]:
        print(f"Simulating reference scheduler: {name} ...")
        all_results.append(simulate_with_fn(jobs, name, SCHEDULERS[name], ctx))

    labels = []
    for w in RISK_WEIGHTS:
        for prefix, fn in [("risk_", make_uncertainty_aware(w)),
                           ("cvar_", make_cvar_aware(CVAR_ALPHA, w))]:
            label = f"{prefix}{w}"
            labels.append(label)
            print(f"Simulating {label} ...")
            all_results.append(simulate_with_fn(jobs, label, fn, ctx))

    combined = pd.concat(all_results, ignore_index=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(TABLES_DIR / "cvar_sweep_simulation_results.csv", index=False)

    metrics = compute_metrics_table(combined, scheduler_names=["forecast_greedy", "oracle"] + labels)
    metrics.to_csv(TABLES_DIR / "cvar_sweep_results.csv", index=False)
    return metrics, combined


def _frontier(metrics: pd.DataFrame, prefix: str, bucket: str) -> pd.DataFrame:
    sub = metrics[(metrics["bucket"] == bucket) & metrics["scheduler"].str.startswith(prefix)].copy()
    sub["risk_weight"] = sub["scheduler"].str.replace(prefix, "").astype(float)
    return sub.sort_values("risk_weight")


def dominance_report(metrics: pd.DataFrame, regret_metric: str, bucket: str = "overall") -> pd.DataFrame:
    """
    For every point on each frontier: is it strictly Pareto-dominated by
    any point on the OTHER frontier (>= savings and <= regret, at least
    one strict)? Plus, for each CVaR point, the threshold frontier's
    savings at the same regret, linearly interpolated along the
    threshold frontier's non-dominated points (NaN outside its regret
    range -- no extrapolation).
    """
    thr = _frontier(metrics, "risk_", bucket)
    cv = _frontier(metrics, "cvar_", bucket)

    def dominated_by(row, others):
        hits = others[(others["pct_carbon_saved"] >= row["pct_carbon_saved"])
                      & (others[regret_metric] <= row[regret_metric])
                      & ((others["pct_carbon_saved"] > row["pct_carbon_saved"])
                         | (others[regret_metric] < row[regret_metric]))]
        return ", ".join(f"{w:g}" for w in hits["risk_weight"])

    # threshold frontier's own Pareto-efficient points, sorted by regret
    eff = thr[[dominated_by(r, thr) == "" for _, r in thr.iterrows()]].sort_values(regret_metric)

    rows = []
    for name, own, other in [("threshold", thr, cv), ("cvar", cv, thr)]:
        for _, r in own.iterrows():
            x = r[regret_metric]
            interp = (np.interp(x, eff[regret_metric], eff["pct_carbon_saved"])
                      if name == "cvar" and eff[regret_metric].min() <= x <= eff[regret_metric].max()
                      else np.nan)
            rows.append({
                "frontier": name,
                "risk_weight": r["risk_weight"],
                regret_metric: x,
                "pct_carbon_saved": r["pct_carbon_saved"],
                "dominated_by_other_frontier_at_w": dominated_by(r, other),
                "threshold_savings_at_same_regret": interp,
            })
    return pd.DataFrame(rows)


def plot_overlay(metrics: pd.DataFrame, out_path: Path, bucket: str = "overall"):
    VIOLET = "#4a3aa7"
    AQUA = "#1baf7a"
    ORANGE = "#eb6834"
    BLUE = "#2a78d6"
    INK = "#0b0b0b"
    INK_SECONDARY = "#52514e"
    INK_MUTED = "#898781"
    GRIDLINE = "#e1e0d9"
    SURFACE = "#fcfcfb"

    sub = metrics[metrics["bucket"] == bucket]
    fg = sub[sub["scheduler"] == "forecast_greedy"].iloc[0]
    oracle = sub[sub["scheduler"] == "oracle"].iloc[0]

    panels = [("regret_rate", 100, "regret rate (% of jobs worse off than running immediately)"),
              ("avg_regret_magnitude_gco2", 1, "avg regret magnitude when it occurs (gCO2)")]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.8), facecolor=SURFACE, sharey=True)

    series = [("risk_", VIOLET, "o", "-", (-34, 6),
               "threshold-based: q50 + w·(q90−q50) at argmin-q50 candidate"),
              ("cvar_", AQUA, "s", "--", (6, -11),
               f"CVaR-based: min over all candidates of (1−w)·q50 + w·CVaR$_{{{CVAR_ALPHA:g}}}$")]

    for ax, (metric, scale, xlabel) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        for prefix, color, marker, ls, offset, label in series:
            fr = _frontier(metrics, prefix, bucket)
            x = fr[metric].values * scale
            y = fr["pct_carbon_saved"].values * 100
            ax.plot(x, y, color=color, linewidth=2, linestyle=ls, zorder=2)
            ax.scatter(x, y, color=color, s=64, marker=marker, zorder=3,
                       edgecolors=SURFACE, linewidths=2, label=label)
            for xi, yi, w in zip(x, y, fr["risk_weight"]):
                ax.annotate(f"w={w:g}", xy=(xi, yi), textcoords="offset points",
                            xytext=offset, fontsize=7.5, color=INK_SECONDARY)

        ax.scatter([fg[metric] * scale], [fg["pct_carbon_saved"] * 100], color=ORANGE, s=90,
                   marker="D", zorder=4, edgecolors=SURFACE, linewidths=2,
                   label="forecast_greedy (≡ w=0 on both)")
        ax.scatter([oracle[metric] * scale], [oracle["pct_carbon_saved"] * 100], color=BLUE, s=140,
                   marker="*", zorder=4, edgecolors=SURFACE, linewidths=1, label="oracle")

        ax.set_xlabel(xlabel, color=INK_SECONDARY, fontsize=9)
        ax.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(INK_MUTED)
        ax.tick_params(colors=INK_SECONDARY, labelsize=9)

    axes[0].set_ylabel("carbon saved (%) vs run_immediately", color=INK_SECONDARY, fontsize=9)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, frameon=False, fontsize=9, labelcolor=INK_SECONDARY,
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"Risk-tolerance frontiers ({bucket}): threshold-based vs CVaR-based scheduler "
                 f"(alpha={CVAR_ALPHA:g}, w = risk_weight)",
                 color=INK, fontsize=12, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0.09, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    print(f"Saved plot to {out_path}")


def decision_divergence(combined: pd.DataFrame) -> pd.DataFrame:
    """Per risk_weight: how many jobs got a different start time under the two rules."""
    rows = []
    for w in RISK_WEIGHTS:
        a = combined[combined["scheduler"] == f"risk_{w}"].set_index("job_id")
        b = combined[combined["scheduler"] == f"cvar_{w}"].set_index("job_id").loc[a.index]
        diff = a["started_time"] != b["started_time"]
        rows.append({"risk_weight": w,
                     "jobs_with_different_start": int(diff.sum()),
                     "cvar_later_start": int((b["started_time"] > a["started_time"]).sum()),
                     "cvar_lower_cost_among_differing": int(
                         (b["realized_carbon_cost"] < a["realized_carbon_cost"])[diff].sum())})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    metrics, combined = run_cvar_sweep()

    display_cols = ["scheduler", "n_jobs", "pct_carbon_saved", "regret_rate",
                    "avg_regret_magnitude_gco2", "avg_delay_hours", "deadline_violations"]
    print("\n=== Overall: threshold-based (risk_*) vs CVaR-based (cvar_*) ===")
    print(metrics[metrics["bucket"] == "overall"][display_cols].round(4).to_string(index=False))

    print("\n=== Decision divergence between the two rules (overall) ===")
    print(decision_divergence(combined).to_string(index=False))

    reports = []
    for metric in ["regret_rate", "avg_regret_magnitude_gco2"]:
        for bucket in ["overall", "6h", "12h", "24h"]:
            rep = dominance_report(metrics, metric, bucket)
            rep.insert(0, "bucket", bucket)
            rep.insert(0, "regret_metric", metric)
            reports.append(rep.rename(columns={metric: "regret"}))
            if bucket == "overall":
                print(f"\n=== Pareto dominance, overall, regret = {metric} ===")
                print(rep.round(4).to_string(index=False))
    pd.concat(reports, ignore_index=True).to_csv(TABLES_DIR / "cvar_dominance.csv", index=False)

    plot_overlay(metrics, PLOTS_DIR / "cvar_vs_threshold_frontier.png", bucket="overall")
