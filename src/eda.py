"""
Exploratory look at the cleaned carbon intensity series.

WHY THIS PLOT (not just describe()):
The whole thesis premise -- "shifting delay-tolerant jobs to
low-carbon windows saves carbon" -- only works if carbon intensity
actually varies enough, on a timescale schedulers can exploit (hours,
within a deadline), to be worth shifting for. This script is the first
sanity check of that premise: it shows (1) the raw series over time,
so gaps/outliers are visible, and (2) the diurnal (hour-of-day) profile,
which is the specific pattern a same-day scheduler exploits.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_config import clean_data_path  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "plots"

# Dataviz reference palette (see thesis skill notes): single sequential
# hue (blue) for a single series, muted ink/grid so the data reads first.
BLUE = "#2a78d6"
BLUE_BAND = "#9ec5f4"   # step 200 of the same ramp, for the IQR band
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"


def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(INK_MUTED)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)


def plot_overview(df: pd.DataFrame, out_path: Path):
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 7), facecolor=SURFACE, gridspec_kw={"height_ratios": [1.3, 1]}
    )
    fig.suptitle(
        "GB Grid Carbon Intensity — actual, 30-min settlement periods",
        color=INK_PRIMARY, fontsize=13, fontweight="bold", x=0.02, ha="left",
    )

    # --- Panel 1: full time series ---
    ax1.plot(df["timestamp"], df["carbon_intensity"], color=BLUE, linewidth=0.9)
    ax1.set_title(
        f"{df['timestamp'].min().date()} to {df['timestamp'].max().date()}",
        color=INK_SECONDARY, fontsize=10, loc="left",
    )
    ax1.set_ylabel("gCO2/kWh", color=INK_SECONDARY, fontsize=9)
    _style_axes(ax1)

    # --- Panel 2: diurnal profile (mean + IQR band by hour of day) ---
    hourly = df.copy()
    hourly["hour"] = hourly["timestamp"].dt.hour
    stats = hourly.groupby("hour")["carbon_intensity"].agg(
        mean="mean", q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75)
    )

    ax2.fill_between(stats.index, stats["q25"], stats["q75"], color=BLUE_BAND, alpha=0.5,
                      linewidth=0, label="IQR (25th-75th pct)")
    ax2.plot(stats.index, stats["mean"], color=BLUE, linewidth=2, marker="o", markersize=3,
              label="mean")
    ax2.set_title(
        "Diurnal profile — this hour-of-day swing is what temporal shifting exploits",
        color=INK_SECONDARY, fontsize=10, loc="left",
    )
    ax2.set_xlabel("hour of day (UTC)", color=INK_SECONDARY, fontsize=9)
    ax2.set_ylabel("gCO2/kWh", color=INK_SECONDARY, fontsize=9)
    ax2.set_xticks(range(0, 24, 2))
    ax2.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY, loc="upper right")
    _style_axes(ax2)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    data_file = clean_data_path()  # the dataset named by CAS_DATA_TAG
    df = pd.read_csv(data_file, parse_dates=["timestamp"])
    print(f"Loaded {data_file.name}: {len(df)} rows")

    plot_overview(df, RESULTS_DIR / "carbon_intensity_overview.png")

    print("\n=== Diurnal mean by hour (UTC) ===")
    hourly_mean = df.copy()
    hourly_mean["hour"] = hourly_mean["timestamp"].dt.hour
    print(hourly_mean.groupby("hour")["carbon_intensity"].mean().round(1))

    swing = hourly_mean.groupby("hour")["carbon_intensity"].mean()
    print(f"\nPeak hour: {swing.idxmax()}:00 ({swing.max():.1f} gCO2/kWh)")
    print(f"Trough hour: {swing.idxmin()}:00 ({swing.min():.1f} gCO2/kWh)")
    print(f"Diurnal swing (peak - trough of hourly means): {swing.max() - swing.min():.1f} gCO2/kWh")

    print("\n=== Monthly mean (seasonal structure -- the reason for the 365-day window) ===")
    monthly = df.copy()
    monthly["month"] = monthly["timestamp"].dt.to_period("M")
    monthly_mean = monthly.groupby("month")["carbon_intensity"].mean().round(1)
    print(monthly_mean)
