"""
Risk-Tolerance Sweep for the Uncertainty-Aware Scheduler
============================================================
Part of: AI-Based Energy and Carbon Optimization in Cloud Computing Environments

Step 5 found that the uncertainty-aware scheduler, at its single fixed
risk_weight=1.0 (compare the CQR-calibrated q90 worst case against the
current actual), regrets far less often than forecast_greedy but barely
shifts any jobs at all (~1% carbon saved vs forecast_greedy's ~22.5%) --
because the CQR-widened q90 bound is a very hard bar for the future to
clear. That result is one END of a spectrum, not "the" answer: this
module sweeps `schedulers.make_uncertainty_aware(risk_weight)` across
risk_weight in {0.0, 0.2, 0.4, 0.6, 0.8, 1.0} and asks whether an
INTERMEDIATE risk tolerance Pareto-dominates either extreme -- more
carbon saved than the fully risk-averse policy at similar regret, or
less regret than the fully greedy policy (risk_weight=0, ≡
forecast_greedy) at similar savings.

No changes to simulator.py were needed: `simulate_job()` already accepts
any callable with the standard (t, job, remaining_times, ctx) -> bool
signature, so each risk_weight's scheduler function from
`make_uncertainty_aware()` is used directly without registering it in
`schedulers.SCHEDULERS` (that dict stays exactly as Step 5 left it, so
the fixed 5-scheduler comparison in evaluate.py is unaffected).
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibration import three_way_split  # noqa: E402
from evaluate import BASELINE, compute_metrics_table  # noqa: E402
from forecasting import create_features, load_clean_data  # noqa: E402
from schedulers import SCHEDULERS, make_uncertainty_aware  # noqa: E402
from simulator import (  # noqa: E402
    SimContext,
    build_forecast_tables,
    filter_jobs_to_test_period,
    load_workload,
    simulate_job,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
PLOTS_DIR = PROJECT_ROOT / "results" / "plots"

RISK_WEIGHTS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def simulate_with_fn(jobs_df: pd.DataFrame, scheduler_label: str, scheduler_fn, ctx) -> pd.DataFrame:
    """Same as simulator.simulate_all, but takes a callable directly instead of a SCHEDULERS name."""
    records = [simulate_job(job, scheduler_fn, ctx) for _, job in jobs_df.iterrows()]
    df = pd.DataFrame(records)
    df["scheduler"] = scheduler_label
    return df


def run_sweep():
    df = load_clean_data()
    df_feat = create_features(df)
    train_feat, calib_feat, test_feat = three_way_split(df_feat)
    test_start, test_end = test_feat["timestamp"].min(), test_feat["timestamp"].max()
    threshold_value = train_feat["carbon_intensity"].median()

    workload_df = load_workload()
    jobs = filter_jobs_to_test_period(workload_df, test_start, test_end)

    print("\nBuilding CQR-calibrated forecast lookup tables (1h-24h horizons)...")
    forecast_tables = build_forecast_tables(df_feat, test_start, test_end)
    actual_series = df.set_index("timestamp")["carbon_intensity"]
    ctx = SimContext(actual_series, forecast_tables, threshold_value)

    all_results = []

    # Reference schedulers: run_immediately (baseline, required by
    # compute_metrics_table), forecast_greedy and oracle (plotted as
    # distinct reference points -- they have no risk_weight).
    for name in ["run_immediately", "forecast_greedy", "oracle"]:
        print(f"Simulating reference scheduler: {name} ...")
        res = simulate_with_fn(jobs, name, SCHEDULERS[name], ctx)
        all_results.append(res)

    risk_labels = []
    for w in RISK_WEIGHTS:
        label = f"risk_{w}"
        risk_labels.append(label)
        print(f"Simulating uncertainty_aware, risk_weight={w} ...")
        fn = make_uncertainty_aware(w)
        res = simulate_with_fn(jobs, label, fn, ctx)
        all_results.append(res)

    combined = pd.concat(all_results, ignore_index=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(TABLES_DIR / "risk_sweep_simulation_results.csv", index=False)

    scheduler_names = ["forecast_greedy", "oracle"] + risk_labels
    metrics = compute_metrics_table(combined, scheduler_names=scheduler_names)
    metrics.to_csv(TABLES_DIR / "risk_sweep_results.csv", index=False)

    return metrics, risk_labels


def plot_pareto_frontier(metrics: pd.DataFrame, risk_labels: list, out_path: Path,
                          regret_metric: str = "regret_rate", bucket: str = "overall"):
    """
    x = regret metric, y = % carbon saved, one point per risk_weight
    (connected in risk_weight order and labeled), plus forecast_greedy
    and oracle as distinct, unconnected reference markers.
    """
    sub = metrics[metrics["bucket"] == bucket]

    risk_sub = sub[sub["scheduler"].isin(risk_labels)].copy()
    risk_sub["risk_weight"] = risk_sub["scheduler"].str.replace("risk_", "").astype(float)
    risk_sub = risk_sub.sort_values("risk_weight")

    fg = sub[sub["scheduler"] == "forecast_greedy"].iloc[0]
    oracle = sub[sub["scheduler"] == "oracle"].iloc[0]

    BLUE = "#2a78d6"
    ORANGE = "#eb6834"
    VIOLET = "#4a3aa7"
    INK_SECONDARY = "#52514e"
    INK_MUTED = "#898781"
    GRIDLINE = "#e1e0d9"
    SURFACE = "#fcfcfb"

    metric_label = {
        "regret_rate": "regret rate (fraction of jobs worse off than running immediately)",
        "avg_regret_magnitude_gco2": "avg regret magnitude (gCO2, when it occurs)",
    }[regret_metric]

    fig, ax = plt.subplots(figsize=(8.5, 6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    x = risk_sub[regret_metric].values * (100 if regret_metric == "regret_rate" else 1)
    y = risk_sub["pct_carbon_saved"].values * 100
    ax.plot(x, y, color=VIOLET, linewidth=1.6, zorder=2)
    ax.scatter(x, y, color=VIOLET, s=60, zorder=3, label="uncertainty_aware (risk_weight sweep)")
    for xi, yi, w in zip(x, y, risk_sub["risk_weight"]):
        ax.annotate(f"{w:g}", xy=(xi, yi), textcoords="offset points",
                    xytext=(6, 4), fontsize=8, color=INK_SECONDARY)

    fg_x = fg[regret_metric] * (100 if regret_metric == "regret_rate" else 1)
    fg_y = fg["pct_carbon_saved"] * 100
    ax.scatter([fg_x], [fg_y], color=ORANGE, s=90, marker="D", zorder=4, label="forecast_greedy")

    oracle_x = oracle[regret_metric] * (100 if regret_metric == "regret_rate" else 1)
    oracle_y = oracle["pct_carbon_saved"] * 100
    ax.scatter([oracle_x], [oracle_y], color=BLUE, s=90, marker="*", zorder=4, label="oracle")

    unit = "%" if regret_metric == "regret_rate" else "gCO2"
    ax.set_title(f"Risk-tolerance frontier ({bucket}): carbon saved vs regret",
                 color="#0b0b0b", fontsize=12, fontweight="bold", loc="left")
    ax.set_xlabel(f"{metric_label} ({unit})", color=INK_SECONDARY, fontsize=9)
    ax.set_ylabel("carbon saved (%) vs run_immediately", color=INK_SECONDARY, fontsize=9)
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
    print(f"Saved plot to {out_path}")


def plot_pareto_frontier_faceted(metrics: pd.DataFrame, risk_labels: list, out_path: Path,
                                  regret_metric: str = "regret_rate"):
    """Same frontier, small-multiples by slack bucket (6h/12h/24h)."""
    buckets = ["6h", "12h", "24h"]
    VIOLET = "#4a3aa7"
    ORANGE = "#eb6834"
    BLUE = "#2a78d6"
    INK_SECONDARY = "#52514e"
    INK_MUTED = "#898781"
    GRIDLINE = "#e1e0d9"
    SURFACE = "#fcfcfb"

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), facecolor=SURFACE, sharey=True)

    for ax, bucket in zip(axes, buckets):
        ax.set_facecolor(SURFACE)
        sub = metrics[metrics["bucket"] == bucket]
        risk_sub = sub[sub["scheduler"].isin(risk_labels)].copy()
        risk_sub["risk_weight"] = risk_sub["scheduler"].str.replace("risk_", "").astype(float)
        risk_sub = risk_sub.sort_values("risk_weight")

        x_scale = 100 if regret_metric == "regret_rate" else 1
        x = risk_sub[regret_metric].values * x_scale
        y = risk_sub["pct_carbon_saved"].values * 100
        ax.plot(x, y, color=VIOLET, linewidth=1.6, zorder=2)
        ax.scatter(x, y, color=VIOLET, s=50, zorder=3)
        for xi, yi, w in zip(x, y, risk_sub["risk_weight"]):
            ax.annotate(f"{w:g}", xy=(xi, yi), textcoords="offset points",
                        xytext=(5, 4), fontsize=7, color=INK_SECONDARY)

        fg = sub[sub["scheduler"] == "forecast_greedy"].iloc[0]
        oracle = sub[sub["scheduler"] == "oracle"].iloc[0]
        ax.scatter([fg[regret_metric] * x_scale], [fg["pct_carbon_saved"] * 100],
                   color=ORANGE, s=70, marker="D", zorder=4)
        ax.scatter([oracle[regret_metric] * x_scale], [oracle["pct_carbon_saved"] * 100],
                   color=BLUE, s=70, marker="*", zorder=4)

        ax.set_title(bucket, color="#0b0b0b", fontsize=11, fontweight="bold", loc="left")
        unit = "%" if regret_metric == "regret_rate" else "gCO2"
        ax.set_xlabel(f"regret ({unit})", color=INK_SECONDARY, fontsize=8)
        ax.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(INK_MUTED)
        ax.tick_params(colors=INK_SECONDARY, labelsize=8)

    axes[0].set_ylabel("carbon saved (%)", color=INK_SECONDARY, fontsize=9)
    fig.suptitle("Risk-tolerance frontier by slack bucket (violet=risk sweep, "
                 "diamond=forecast_greedy, star=oracle)",
                 color="#0b0b0b", fontsize=11, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    metrics, risk_labels = run_sweep()

    print("\n" + "=" * 100)
    print("=== Risk-weight sweep results (uncertainty_aware) + forecast_greedy/oracle reference ===")
    print("=" * 100)
    display_cols = ["scheduler", "bucket", "n_jobs", "pct_carbon_saved", "regret_rate",
                     "avg_regret_magnitude_gco2", "gap_to_oracle_pct"]
    print(metrics[display_cols].round(4).to_string(index=False))

    plot_pareto_frontier(metrics, risk_labels, PLOTS_DIR / "risk_frontier_regret_rate.png",
                          regret_metric="regret_rate", bucket="overall")
    plot_pareto_frontier(metrics, risk_labels, PLOTS_DIR / "risk_frontier_regret_magnitude.png",
                          regret_metric="avg_regret_magnitude_gco2", bucket="overall")
    plot_pareto_frontier_faceted(metrics, risk_labels, PLOTS_DIR / "risk_frontier_by_bucket.png",
                                  regret_metric="regret_rate")
