"""
Hourly LightGBM + CQR forecaster for the M2 LP scheduler
========================================================
The base paper's LP works on HOURLY carbon intensity (docs/m2_spec.md).
The M1 models predict single 30-minute values, so this module trains a
SEPARATE set of models whose target is the hourly mean, and saves them to
results/models/m2_hourly/. The M1 models in results/models/quantile/ are
never read or written here.

What is reused from M1 (unchanged functions):
  create_features, get_feature_columns, build_horizon_dataset,
  train_quantile_models, predict_quantiles (forecasting.py), and
  nonconformity_scores, conformal_correction, apply_calibration
  (calibration.py).

Hourly data (spec G18)
----------------------
Hour H = mean of the H:00 and H:30 settlement periods. Hours missing
either half are left out (NaN), never filled.

Horizons
--------
A forecast made at origin hour o uses data up to hour o-1 (the last
finished hour). Horizon h = 0..23 predicts the hour starting at o+h, so
the 24 slots o, o+1, ..., o+23 cover the LP's T = 24 hour horizon. h=0
is still a genuine forecast: hour o has not happened yet.

Features: the same time spans as M1, in hourly steps:
  lags 1h, 2h, 4h, 24h; rolling mean/std over 2h, 4h, 24h;
  plus M1's calendar features. M1's 30-minute lag has no hourly equivalent.

Split: the SAME calendar boundaries as M1
-----------------------------------------
train / calibration / test use M1's canonical split boundaries
(simulator.get_canonical_split), so the M2 test period is exactly M1's.
The split is slightly stricter than M1's: a training row is kept only if
its whole TARGET hour ends before the calibration period starts, and a
calibration row only if its target hour ends before the test period
starts. No target value ever crosses a boundary.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibration import apply_calibration, conformal_correction, nonconformity_scores  # noqa: E402
from data_config import check_model_label, clear_model_label, get_data_tag, write_model_label  # noqa: E402
from forecasting import (  # noqa: E402
    QUANTILES,
    build_horizon_dataset,
    create_features,
    get_feature_columns,
    load_clean_data,
    predict_quantiles,
    train_quantile_models,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HOURLY_MODELS_DIR = PROJECT_ROOT / "results" / "models" / "m2_hourly"
M2_TABLES_DIR = PROJECT_ROOT / "results" / "tables" / "m2_base_lp"

HOURLY_HORIZONS = list(range(24))  # h = 0..23 -> hours o .. o+23
HOURLY_LAGS = (1, 2, 4, 24)
HOURLY_ROLLING = (2, 4, 24)
ALPHA = 0.2  # 80% interval, as in M1
ONE_HOUR = pd.Timedelta(hours=1)


# ----------------------------------------------------------------------
# Hourly series
# ----------------------------------------------------------------------
def to_hourly(df30: pd.DataFrame) -> pd.DataFrame:
    """
    30-min clean data -> hourly means on a complete hourly grid.
    An hour is kept only if BOTH halves exist and are not NaN; otherwise
    its value is NaN. The grid covers only hours whose :00 half lies in the data.
    """
    s = df30.set_index("timestamp")["carbon_intensity"]
    hour = s.index.floor("h")
    grouped = s.groupby(hour)
    mean = grouped.mean()
    complete = (grouped.size() == 2) & (grouped.count() == 2)
    mean[~complete] = np.nan
    full = pd.date_range(mean.index.min(), mean.index.max(), freq="h")
    mean = mean.reindex(full)
    # Drop incomplete hours at either end (e.g. the dataset starts at 03:30).
    valid = mean.notna()
    mean = mean.loc[valid.idxmax(): valid[::-1].idxmax()]
    return pd.DataFrame({"timestamp": mean.index, "carbon_intensity": mean.values})


def hourly_features(hourly: pd.DataFrame) -> pd.DataFrame:
    return create_features(hourly, lags=HOURLY_LAGS, rolling_windows=HOURLY_ROLLING)


def horizon_dataset(hourly_feat: pd.DataFrame, hourly: pd.DataFrame, h: int) -> pd.DataFrame:
    """Target = hourly mean of the hour starting at origin + h hours."""
    # build_horizon_dataset counts in 30-min steps, so h hours = 2h steps.
    return build_horizon_dataset(hourly_feat, hourly, horizon_steps=2 * h)


def split_by_boundaries(df_h: pd.DataFrame, calib_start: pd.Timestamp,
                        test_start: pd.Timestamp):
    """Strict chronological split by calendar boundaries (see module docstring)."""
    origin = df_h["timestamp"]
    target_end = df_h["target_timestamp"] + ONE_HOUR
    train = df_h[(origin < calib_start) & (target_end <= calib_start)]
    calib = df_h[(origin >= calib_start) & (target_end <= test_start)]
    test = df_h[origin >= test_start]
    return (train.reset_index(drop=True), calib.reset_index(drop=True),
            test.reset_index(drop=True))


def m1_boundaries(df30: pd.DataFrame):
    """Calibration and test start times of M1's canonical split."""
    from simulator import get_canonical_split  # local import: heavy module
    _, _, calib_feat, test_feat = get_canonical_split(df30)
    return (calib_feat["timestamp"].min(), test_feat["timestamp"].min(),
            test_feat["timestamp"].max())


# ----------------------------------------------------------------------
# Train + calibrate
# ----------------------------------------------------------------------
def run_hourly_experiment(df30: pd.DataFrame, horizons=HOURLY_HORIZONS,
                          models_dir: Path | None = HOURLY_MODELS_DIR):
    """
    Trains q10/q50/q90 per horizon on TRAIN, computes the CQR correction
    on CALIBRATION, and evaluates on TEST (evaluation only, nothing is
    chosen from it). models_dir=None keeps everything in memory (tests).
    """
    hourly = to_hourly(df30)
    calib_start, test_start, _ = m1_boundaries(df30)
    feat = hourly_features(hourly)
    feature_cols = get_feature_columns(feat)

    if models_dir is not None:
        models_dir.mkdir(parents=True, exist_ok=True)
        clear_model_label(models_dir)

    rows, models_by_h = [], {}
    for h in horizons:
        df_h = horizon_dataset(feat, hourly, h)
        train, calib, test = split_by_boundaries(df_h, calib_start, test_start)
        models = train_quantile_models(train, feature_cols, target_col="target",
                                       quantiles=QUANTILES)
        cp = predict_quantiles(models, calib[feature_cols])
        correction = conformal_correction(
            nonconformity_scores(cp["q10"], cp["q90"], calib["target"]), alpha=ALPHA)

        tp = predict_quantiles(models, test[feature_cols])
        y = test["target"].to_numpy()
        lo, hi = apply_calibration(tp["q10"], tp["q90"], correction)
        rows.append({
            "horizon_h": h, "n_train": len(train), "n_calib": len(calib), "n_test": len(test),
            "test_MAE_q50": float(np.mean(np.abs(y - tp["q50"]))),
            "test_MSE_q50": float(np.mean((y - tp["q50"]) ** 2)),
            "test_coverage_raw": float(np.mean((y >= tp["q10"]) & (y <= tp["q90"]))),
            "test_coverage_calibrated": float(np.mean((y >= lo) & (y <= hi))),
            "cqr_correction": correction,
        })
        models_by_h[h] = models
        if models_dir is not None:
            for q, booster in models.items():
                booster.save_model(str(models_dir / f"h{h:02d}h_q{int(q * 100):02d}.txt"))
        print(f"  h={h:>2}  MAE={rows[-1]['test_MAE_q50']:.2f}  "
              f"coverage raw={rows[-1]['test_coverage_raw']:.1%} "
              f"-> calibrated={rows[-1]['test_coverage_calibrated']:.1%}")

    per_h = pd.DataFrame(rows)
    if models_dir is not None:
        per_h[["horizon_h", "cqr_correction", "n_calib"]].assign(alpha=ALPHA).to_csv(
            models_dir / "cqr_corrections.csv", index=False)
    return per_h, models_by_h


# ----------------------------------------------------------------------
# Forecast tables for the scheduler
# ----------------------------------------------------------------------
def forecast_tables(hourly: pd.DataFrame, origins: pd.DatetimeIndex,
                    models_dir: Path = HOURLY_MODELS_DIR, tag: str | None = None,
                    models_by_h: dict | None = None,
                    corrections_path: Path | None = None) -> dict:
    """
    Returns {"q10","q50","q90"} -> DataFrame indexed by origin hour, with
    columns 0..23 (horizon h = the hour starting at origin + h).
    q10/q90 are CQR-calibrated; q50 is the raw median (CQR never changes it).
    Saved models are used only if their label matches the dataset tag.
    corrections_path: CQR corrections to apply instead of the ones saved
    with the models (M3 uses corrections re-fitted on C1 only). Its folder
    must carry a matching dataset label too.
    """
    import lightgbm as lgb

    feat = hourly_features(hourly).set_index("timestamp")
    missing = origins.difference(feat.index)
    if len(missing):
        raise ValueError(f"No features for {len(missing)} origin hours, e.g. {missing[0]}")
    X = feat.loc[origins, get_feature_columns(feat.reset_index())]

    if models_by_h is None:
        check_model_label(models_dir, tag or get_data_tag())
        if corrections_path is None:
            corrections_path = models_dir / "cqr_corrections.csv"
        else:
            check_model_label(Path(corrections_path).parent, tag or get_data_tag())
        corrections = pd.read_csv(corrections_path).set_index("horizon_h")
    out = {q: pd.DataFrame(index=origins, columns=HOURLY_HORIZONS, dtype=float)
           for q in ("q10", "q50", "q90")}
    for h in HOURLY_HORIZONS:
        if models_by_h is not None:
            preds = predict_quantiles(models_by_h[h], X)
            correction = 0.0
        else:
            preds = pd.DataFrame({
                f"q{int(q * 100)}": lgb.Booster(
                    model_file=str(models_dir / f"h{h:02d}h_q{int(q * 100):02d}.txt")).predict(X)
                for q in QUANTILES}, index=X.index)
            correction = float(corrections.loc[h, "cqr_correction"])
        lo, hi = apply_calibration(preds["q10"], preds["q90"], correction)
        out["q10"][h], out["q50"][h], out["q90"][h] = lo, preds["q50"].to_numpy(), hi
    return out


if __name__ == "__main__":
    df30 = load_clean_data()
    tag = get_data_tag()
    print("Training hourly LightGBM + CQR (h = 0..23) on M1's split boundaries ...")
    per_h, _ = run_hourly_experiment(df30)
    write_model_label(HOURLY_MODELS_DIR, tag, stage="calibration")
    M2_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    per_h.to_csv(M2_TABLES_DIR / "hourly_forecast_by_horizon.csv", index=False)
    m = per_h.mean(numeric_only=True)
    print(f"\nMean over horizons: MAE={m['test_MAE_q50']:.2f}, MSE={m['test_MSE_q50']:.1f}, "
          f"coverage raw={m['test_coverage_raw']:.1%} -> calibrated={m['test_coverage_calibrated']:.1%}")
    print(f"Saved models to {HOURLY_MODELS_DIR} and table to "
          f"{M2_TABLES_DIR / 'hourly_forecast_by_horizon.csv'}")
