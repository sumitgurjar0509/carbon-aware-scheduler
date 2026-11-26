"""
Multi-Horizon Carbon Intensity Forecasting
============================================
Part of: AI-Based Energy and Carbon Optimization in Cloud Computing Environments

Extends the prototype `carbon_quantile_forecast.py` (kept at the project
root, unchanged, as the original one-step-ahead proof of concept) to:
  1. load the REAL cleaned data produced by `data_pipeline.py` (columns
     timestamp / carbon_intensity / forecast / was_interpolated) instead
     of the prototype's assumed 2-column synthetic CSV, and
  2. forecast MULTIPLE horizons (1..24 hours ahead), not just one
     30-minute step -- the scheduler needs a forecast for every point in
     a job's deadline window (6h/12h/24h slack), not a single number.

REUSED UNCHANGED from carbon_quantile_forecast.py (per instruction: these
are already correct, do not rewrite them):
    create_features, time_based_split, persistence_baseline,
    seasonal_naive_baseline, train_quantile_models, predict_quantiles,
    pinball_loss, evaluate_forecast

ADAPTED for the real schema (this is the actual "reuse, don't rewrite
from scratch" work):
  - feature_cols now explicitly excludes 'forecast' and 'was_interpolated'
    in addition to 'timestamp'/'carbon_intensity'. The prototype's
    feature_cols line (`everything except timestamp and target`) would
    silently have handed the model the grid operator's OWN forecast as
    an input feature on real data -- see "Why we exclude `forecast`"
    below for why that would defeat the point of this comparison.
  - baseline/target alignment is done by TIMESTAMP, not row position,
    because real data can (rarely) have gaps; see "Multi-step-ahead
    design" below for why that matters here specifically.

IMPORTANT -- SUPERSEDED BY calibration.py FOR DOWNSTREAM USE
--------------------------------------------------------------------
This module's own run found the LightGBM quantile model's 80% interval
was overconfident (~62.4% actual coverage, not 80%). `calibration.py`
fixes this with Conformalized Quantile Regression: it RETRAINS all 72
models on a smaller train-only split (a calibration set must be held
out and never trained on) and OVERWRITES the model files in
results/models/quantile/, then applies a per-horizon correction. From
Step 5 (the scheduler) onward, use `calibration.predict_calibrated()`,
not the raw q10/q90 straight out of a model file loaded here -- see
calibration.py's module docstring for the full reasoning.

======================================================================
MULTI-STEP-AHEAD DESIGN (read this before the viva)
======================================================================
The job is: given data up to "now" (time t), forecast carbon intensity
at t+1h, t+2h, ..., t+24h -- one trajectory per origin time, because a
scheduler deciding when to run a job needs to compare candidate slots
across the whole window up to its deadline.

Three standard strategies exist for this with tree-based models
(LightGBM has no native sequence output):
  (a) RECURSIVE: train one 1-step model, feed its own predictions back
      in as lag features to roll forward 48 steps. Cheapest (3 models
      total for q10/q50/q90), but errors compound over 24h, AND it is
      awkward for QUANTILE forecasts specifically -- feeding q10's
      predicted value back in as if it were a real lag treats one
      quantile path as "the future", which breaks the interpretation
      of the resulting q10 trajectory as a genuine 10th-percentile band.
  (b) MULTI-OUTPUT: one model that predicts all horizons at once.
      LightGBM doesn't support this natively for quantile regression.
  (c) DIRECT: train a SEPARATE model per horizon h, each predicting
      carbon_intensity h steps ahead directly from features computed at
      the origin time. No compounding error, and each quantile at each
      horizon is a well-defined, independently-fit conditional quantile.
      Cost: more models to train.

We use DIRECT, because avoiding compounding error and keeping each
quantile trajectory statistically well-defined matters more here than
saving training time -- and training time turns out to be a non-issue
(see below).

Resolution of the horizon grid: the data is natively 30-minute, so
"24 hours ahead" is 48 native steps. We deliberately forecast at HOURLY
steps (h = 1..24 hours = 2..48 native steps in steps of 2), not all 48
half-hourly steps, for three reasons: (1) it halves the model count
(24 horizons x 3 quantiles = 72 models instead of 144), keeping this
comfortably in "minutes not hours" on a CPU-only laptop; (2) hourly
resolution is standard in carbon-aware scheduling literature and is
fine-grained enough for a scheduler choosing among candidate hours
within a 6h/12h/24h window; (3) it directly matches how deadline slack
is specified in this thesis (whole hours). This can be revisited if the
simulator later needs sub-hourly placement.

Target alignment by TIMESTAMP, not row position: for horizon h, the
target for the row at origin time t is the actual value at t + h hours.
The naive way to build this is `df[target_col].shift(-h)` (a POSITIONAL
shift). That is only correct if every row is exactly one native step
after the previous row with no rows missing. `create_features()`
internally drops rows where lag/rolling features are NaN (correctly,
per its own docstring) -- and if any long/edge data gap ever caused
additional rows to be dropped from the middle of the series, a
positional shift would silently pair each origin with the wrong future
row across that gap. We build the target (and the persistence/seasonal
baseline predictions) by looking up the value at `timestamp + h` in the
full cleaned series instead, which is correct regardless of gaps. In
the one year of data actually fetched here there happen to be zero
gaps (see gap_report_*.csv), so this made no numerical difference this
run -- but it is the right way to do it and costs nothing extra.
"""
import sys
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_config import (  # noqa: E402
    clean_data_path,
    clear_model_label,
    get_data_tag,
    write_model_label,
)
from data_pipeline import PROCESSED_DIR  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "results" / "models" / "quantile"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
PLOTS_DIR = PROJECT_ROOT / "results" / "plots"

NATIVE_STEP_MINUTES = 30
HORIZON_HOURS = list(range(1, 25))  # 1h .. 24h, matches the thesis's slack buckets
QUANTILES = (0.1, 0.5, 0.9)


# ----------------------------------------------------------------------
# REUSED UNCHANGED from carbon_quantile_forecast.py
# ----------------------------------------------------------------------
def create_features(df, target_col='carbon_intensity',
                     lags=(1, 2, 4, 8, 48),
                     rolling_windows=(4, 8, 48)):
    """
    Builds the input features the model learns from.

    lags: how many time-steps back to look (in units of your data's
          resolution). If data is 30-min resolution, lag=48 means
          "value 24 hours ago" -- carbon intensity is strongly
          periodic on a daily cycle, so this matters a lot.

    rolling_windows: recent trend features (mean/std over the last
          N steps), computed using ONLY past data (shift(1) first)
          so we never leak future information into a feature.
    """
    df = df.copy()

    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)

    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    for lag in lags:
        df[f'lag_{lag}'] = df[target_col].shift(lag)

    for window in rolling_windows:
        df[f'roll_mean_{window}'] = df[target_col].shift(1).rolling(window).mean()
        df[f'roll_std_{window}'] = df[target_col].shift(1).rolling(window).std()

    df = df.dropna().reset_index(drop=True)
    return df


def time_based_split(df, test_frac=0.2):
    """Splits chronologically: train on the past, test on the future."""
    split_idx = int(len(df) * (1 - test_frac))
    train = df.iloc[:split_idx].reset_index(drop=True)
    test = df.iloc[split_idx:].reset_index(drop=True)
    return train, test


def persistence_baseline(df, target_col='carbon_intensity', horizon=1):
    """Naive baseline: 'the future will look like right now'."""
    return df[target_col].shift(horizon)


def seasonal_naive_baseline(df, target_col='carbon_intensity', period=48):
    """Naive baseline: 'the future will look like it did at this time yesterday'."""
    return df[target_col].shift(period)


def train_quantile_models(train_df, feature_cols, target_col='carbon_intensity',
                           quantiles=(0.1, 0.5, 0.9)):
    """One LightGBM model per quantile, via objective='quantile' + alpha=<level>."""
    models = {}
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]

    for q in quantiles:
        params = {
            'objective': 'quantile',
            'alpha': q,
            'metric': 'quantile',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'min_data_in_leaf': 20,
            'verbosity': -1,
        }
        train_set = lgb.Dataset(X_train, label=y_train)
        model = lgb.train(params, train_set, num_boost_round=300)
        models[q] = model

    return models


def predict_quantiles(models, X):
    """Returns a DataFrame with one column per quantile (e.g. q10, q50, q90)."""
    preds = {}
    for q, model in models.items():
        preds[f'q{int(q * 100)}'] = model.predict(X)
    return pd.DataFrame(preds, index=X.index)


def pinball_loss(y_true, y_pred, quantile):
    """The loss function quantile regression actually optimizes. Lower is better."""
    errors = np.asarray(y_true) - np.asarray(y_pred)
    return np.mean(np.maximum(quantile * errors, (quantile - 1) * errors))


def evaluate_forecast(test_df, preds_df, target_col='carbon_intensity'):
    y_true = test_df[target_col].values
    results = {}

    if 'q50' in preds_df.columns:
        results['MAE_median'] = mean_absolute_error(y_true, preds_df['q50'])

    for col in preds_df.columns:
        q = int(col[1:]) / 100
        results[f'pinball_{col}'] = pinball_loss(y_true, preds_df[col].values, q)

    if 'q10' in preds_df.columns and 'q90' in preds_df.columns:
        inside = ((y_true >= preds_df['q10']) & (y_true <= preds_df['q90'])).mean()
        results['coverage_80pct_interval'] = inside

    return results


# ----------------------------------------------------------------------
# NEW: loading real data + multi-horizon target/baseline construction
# ----------------------------------------------------------------------
def load_clean_data(tag: str | None = None,
                    processed_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    """Loads the cleaned dataset named by `tag` (default: CAS_DATA_TAG)."""
    path = clean_data_path(tag, processed_dir)
    df = pd.read_csv(path, parse_dates=["timestamp"])
    print(f"Loaded {path.name}: {len(df)} rows "
          f"({df['timestamp'].min()} -> {df['timestamp'].max()})")
    return df


def get_feature_columns(df_feat: pd.DataFrame) -> list:
    """
    Everything create_features() added, MINUS the columns that must not
    be fed to the model: timestamp/carbon_intensity (identifiers/target),
    and -- specific to the REAL schema -- 'forecast' and 'was_interpolated'.

    Why we exclude `forecast`: the API's `forecast` field is National
    Grid ESO's OWN forecast for that exact future settlement period. If
    we gave our LightGBM model the answer key for the very thing it's
    trying to predict, it would trivially achieve near-perfect skill by
    just copying it -- which would say nothing about how well OUR
    forecaster (built from historical patterns alone) does, and would
    make the "how much of the oracle's savings can a realistic
    forecaster capture" comparison meaningless. We exclude it so this
    forecaster is a genuinely independent, historical-pattern-only model.

    Why we exclude `was_interpolated`: it flags whether a given row's
    OWN actual value was interpolated (a data-quality artifact of
    cleaning), not a physically meaningful predictor of future carbon
    intensity -- including it risks the model fitting noise tied to
    where gaps happened to occur rather than the actual grid dynamics.
    """
    exclude = {"timestamp", "carbon_intensity", "forecast", "was_interpolated"}
    return [c for c in df_feat.columns if c not in exclude]


def build_horizon_dataset(df_feat: pd.DataFrame, clean_df: pd.DataFrame,
                           horizon_steps: int, target_col: str = "carbon_intensity"):
    """
    Attaches the multi-step-ahead target to df_feat, looked up by
    TIMESTAMP (origin_time + horizon_steps * 30min) rather than by row
    position -- see module docstring for why. Drops rows where that
    future timestamp has no known actual value (either past the end of
    the dataset, or an undocumented long/edge gap).
    """
    df_h = df_feat.copy()
    df_h["target_timestamp"] = df_h["timestamp"] + pd.Timedelta(
        minutes=NATIVE_STEP_MINUTES * horizon_steps
    )
    lookup = clean_df.set_index("timestamp")[target_col]
    df_h["target"] = df_h["target_timestamp"].map(lookup)
    df_h = df_h.dropna(subset=["target"]).reset_index(drop=True)
    return df_h


def add_baseline_predictions(df_h: pd.DataFrame, clean_df: pd.DataFrame,
                              horizon_steps: int, target_col: str = "carbon_intensity"):
    """
    Persistence and seasonal-naive predictions, computed on the full
    (gap-safe, positionally-contiguous) `clean_df` grid and then joined
    onto df_h by target_timestamp -- so they line up with the correct
    future point regardless of any rows create_features() dropped.
    """
    baseline = clean_df[["timestamp"]].copy()
    baseline["persistence_pred"] = persistence_baseline(
        clean_df, target_col=target_col, horizon=horizon_steps
    )
    baseline["seasonal_pred"] = seasonal_naive_baseline(
        clean_df, target_col=target_col, period=48
    )
    lookup = baseline.set_index("timestamp")
    return df_h.merge(lookup, left_on="target_timestamp", right_index=True, how="left")


# ----------------------------------------------------------------------
# Main multi-horizon experiment
# ----------------------------------------------------------------------
def run_multi_horizon_experiment(df: pd.DataFrame, horizons_hours=HORIZON_HOURS,
                                  quantiles=QUANTILES, test_frac=0.2,
                                  save_models=True):
    """
    For each horizon (in hours): builds the target, trains one LightGBM
    quantile model set (q10/q50/q90), evaluates all 4 forecasters
    against the same test rows, and optionally saves the models.

    Returns (per_horizon_df, trained_models, test_sets) where
    trained_models[h] = {quantile: booster} and test_sets[h] is the
    test_df used for horizon h hours (kept for the representative-week
    plot -- avoids retraining).
    """
    df_feat = create_features(df)
    feature_cols = get_feature_columns(df_feat)
    print(f"Feature columns ({len(feature_cols)}): {feature_cols}")

    if save_models:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        clear_model_label(MODELS_DIR)

    rows = []
    trained_models = {}
    test_sets = {}

    for h in horizons_hours:
        horizon_steps = h * (60 // NATIVE_STEP_MINUTES)  # hours -> native 30-min steps

        df_h = build_horizon_dataset(df_feat, df, horizon_steps)
        df_h = add_baseline_predictions(df_h, df, horizon_steps)

        train_df, test_df = time_based_split(df_h, test_frac=test_frac)

        models = train_quantile_models(train_df, feature_cols, target_col="target",
                                        quantiles=quantiles)
        preds = predict_quantiles(models, test_df[feature_cols])
        lgb_metrics = evaluate_forecast(test_df, preds, target_col="target")

        baseline_valid = test_df.dropna(subset=["persistence_pred", "seasonal_pred"])
        persistence_mae = mean_absolute_error(
            baseline_valid["target"], baseline_valid["persistence_pred"]
        )
        seasonal_mae = mean_absolute_error(
            baseline_valid["target"], baseline_valid["seasonal_pred"]
        )
        persistence_pinball50 = pinball_loss(
            baseline_valid["target"], baseline_valid["persistence_pred"], 0.5
        )
        seasonal_pinball50 = pinball_loss(
            baseline_valid["target"], baseline_valid["seasonal_pred"], 0.5
        )

        rows.append({
            "horizon_hours": h,
            "persistence_MAE": persistence_mae,
            "persistence_pinball_q50": persistence_pinball50,
            "seasonal_naive_MAE": seasonal_mae,
            "seasonal_naive_pinball_q50": seasonal_pinball50,
            "lgbm_point_MAE": lgb_metrics["MAE_median"],
            "lgbm_quantile_pinball_q10": lgb_metrics["pinball_q10"],
            "lgbm_quantile_pinball_q50": lgb_metrics["pinball_q50"],
            "lgbm_quantile_pinball_q90": lgb_metrics["pinball_q90"],
            "lgbm_quantile_coverage_80pct": lgb_metrics["coverage_80pct_interval"],
            "n_train": len(train_df),
            "n_test": len(test_df),
        })

        trained_models[h] = models
        test_sets[h] = test_df

        if save_models:
            for q, model in models.items():
                model.save_model(str(MODELS_DIR / f"h{h:02d}h_q{int(q*100):02d}.txt"))

        print(f"  horizon={h:>2}h  lgbm_MAE={lgb_metrics['MAE_median']:.2f}  "
              f"persistence_MAE={persistence_mae:.2f}  seasonal_MAE={seasonal_mae:.2f}  "
              f"coverage80={lgb_metrics['coverage_80pct_interval']:.2%}")

    return pd.DataFrame(rows), trained_models, test_sets


def summarize_forecasters(per_horizon_df: pd.DataFrame) -> pd.DataFrame:
    """
    Averages each metric across all 24 horizons into one row per
    forecaster -- the top-level "compare all 4 forecasters" table.
    Cells that don't apply to a forecaster (e.g. a quantile-only metric
    for a point forecaster) are left as NaN, not zero, to avoid implying
    a value that was never computed.
    """
    m = per_horizon_df.mean(numeric_only=True)
    return pd.DataFrame([
        {"forecaster": "Persistence", "MAE": m["persistence_MAE"],
         "pinball_q10": np.nan, "pinball_q50": m["persistence_pinball_q50"],
         "pinball_q90": np.nan, "coverage_80pct": np.nan},
        {"forecaster": "Seasonal-naive (24h)", "MAE": m["seasonal_naive_MAE"],
         "pinball_q10": np.nan, "pinball_q50": m["seasonal_naive_pinball_q50"],
         "pinball_q90": np.nan, "coverage_80pct": np.nan},
        {"forecaster": "LightGBM point (=q50)", "MAE": m["lgbm_point_MAE"],
         "pinball_q10": np.nan, "pinball_q50": m["lgbm_quantile_pinball_q50"],
         "pinball_q90": np.nan, "coverage_80pct": np.nan},
        {"forecaster": "LightGBM quantile (q10/q50/q90)", "MAE": m["lgbm_point_MAE"],
         "pinball_q10": m["lgbm_quantile_pinball_q10"],
         "pinball_q50": m["lgbm_quantile_pinball_q50"],
         "pinball_q90": m["lgbm_quantile_pinball_q90"],
         "coverage_80pct": m["lgbm_quantile_coverage_80pct"]},
    ]).set_index("forecaster")


# ----------------------------------------------------------------------
# Representative-week plot: actual vs LightGBM 24h-ahead forecast
# ----------------------------------------------------------------------
def plot_forecast_vs_actual_week(test_df_24h: pd.DataFrame, out_path: Path,
                                  horizon_hours: int = 24):
    """
    Shows actual vs the LightGBM quantile forecast (q10-q90 band) for
    one representative week, at the 24h horizon -- the longest, hardest
    horizon, and the one that matters most for 24h-slack jobs (the
    thesis's most flexible scheduling case, where the scheduler leans
    most heavily on the forecast rather than ground truth close at hand).

    We pick the FIRST full week available in the test split (rather
    than the last) so there's always a clean 7-day block with no
    partial-week edge effects, regardless of test-set length.
    """
    df = test_df_24h.sort_values("target_timestamp").reset_index(drop=True)
    week_start = df["target_timestamp"].iloc[0].normalize()
    week_end = week_start + pd.Timedelta(days=7)
    week = df[(df["target_timestamp"] >= week_start) & (df["target_timestamp"] < week_end)]

    if len(week) < 24 * 2 * 0.5:  # sanity: at least ~half a week of points
        week = df.iloc[: 7 * 24 * 2]

    BLUE = "#2a78d6"
    ORANGE = "#eb6834"
    ORANGE_BAND = "#f7c5ab"
    INK_SECONDARY = "#52514e"
    INK_MUTED = "#898781"
    GRIDLINE = "#e1e0d9"
    SURFACE = "#fcfcfb"

    fig, ax = plt.subplots(figsize=(11, 5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    ax.fill_between(week["target_timestamp"], week["q10"], week["q90"],
                     color=ORANGE_BAND, alpha=0.6, linewidth=0,
                     label="LightGBM q10-q90 band")
    ax.plot(week["target_timestamp"], week["q50"], color=ORANGE, linewidth=1.6,
             label=f"LightGBM forecast ({horizon_hours}h ahead, median)")
    ax.plot(week["target_timestamp"], week["target"], color=BLUE, linewidth=1.8,
             label="Actual")

    ax.set_title(
        f"Actual vs {horizon_hours}h-ahead LightGBM forecast — "
        f"{week['target_timestamp'].iloc[0].date()} to {week['target_timestamp'].iloc[-1].date()}",
        color="#0b0b0b", fontsize=12, fontweight="bold", loc="left",
    )
    ax.set_ylabel("gCO2/kWh", color=INK_SECONDARY, fontsize=9)
    ax.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(INK_MUTED)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY, loc="upper right")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    df = load_clean_data()

    print(f"\nRunning direct multi-horizon experiment for horizons "
          f"{HORIZON_HOURS[0]}h..{HORIZON_HOURS[-1]}h "
          f"({len(HORIZON_HOURS)} horizons x {len(QUANTILES)} quantiles = "
          f"{len(HORIZON_HOURS) * len(QUANTILES)} models)...")
    per_horizon_df, trained_models, test_sets = run_multi_horizon_experiment(df)
    write_model_label(MODELS_DIR, get_data_tag(), stage="forecasting")

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    per_horizon_path = TABLES_DIR / "forecast_by_horizon.csv"
    per_horizon_df.to_csv(per_horizon_path, index=False)
    print(f"\nSaved per-horizon results to {per_horizon_path}")

    summary = summarize_forecasters(per_horizon_df)
    summary_path = TABLES_DIR / "forecast_summary.csv"
    summary.to_csv(summary_path)

    print("\n=== Forecaster comparison (averaged across 1h-24h horizons) ===")
    print(summary.round(3).to_string())

    print("\n=== Key horizons (matching thesis slack buckets: 6h, 12h, 24h) ===")
    key_rows = per_horizon_df[per_horizon_df["horizon_hours"].isin([1, 6, 12, 24])]
    print(key_rows.round(2).to_string(index=False))

    # Representative-week plot at the 24h horizon.
    test_df_24 = test_sets[24].copy()
    models_24 = trained_models[24]
    feature_cols = get_feature_columns(create_features(df))
    preds_24 = predict_quantiles(models_24, test_df_24[feature_cols])
    test_df_24 = pd.concat([test_df_24.reset_index(drop=True), preds_24.reset_index(drop=True)],
                            axis=1)
    plot_forecast_vs_actual_week(test_df_24, PLOTS_DIR / "forecast_vs_actual_24h_week.png",
                                  horizon_hours=24)
