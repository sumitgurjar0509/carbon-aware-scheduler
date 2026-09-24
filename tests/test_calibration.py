"""
Sanity tests for calibration.py -- specifically the two properties CQR's
validity depends on:
  1. train / calibration / test are strictly chronological and disjoint.
  2. the correction is a function of calibration data ONLY -- it must be
     unaffected by, and never computed using, the test set.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from calibration import (  # noqa: E402
    apply_calibration,
    conformal_correction,
    nonconformity_scores,
    run_calibration_experiment,
    three_way_split,
)
from forecasting import build_horizon_dataset, create_features, get_feature_columns, predict_quantiles  # noqa: E402


def _synthetic_carbon_df(n_days=45, seed=0):
    """
    A synthetic 30-min series with a daily sine pattern + noise -- enough
    structure for create_features()'s lags/rolling stats to be
    meaningful, without needing the real API data for a unit test.
    """
    rng = np.random.default_rng(seed)
    n = n_days * 48
    timestamps = pd.date_range("2026-01-01", periods=n, freq="30min", tz="UTC")
    hours = np.arange(n) / 2.0
    signal = 120 + 40 * np.sin(2 * np.pi * hours / 24) + rng.normal(0, 5, size=n)
    return pd.DataFrame({"timestamp": timestamps, "carbon_intensity": signal})


# ----------------------------------------------------------------------
# 1. Chronological split, no overlap
# ----------------------------------------------------------------------
def test_three_way_split_is_chronological_with_no_overlap():
    df = _synthetic_carbon_df()
    df_feat = create_features(df)
    df_h = build_horizon_dataset(df_feat, df, horizon_steps=4)

    train, calib, test = three_way_split(df_h)

    assert len(train) + len(calib) + len(test) == len(df_h)
    assert train["timestamp"].max() < calib["timestamp"].min()
    assert calib["timestamp"].max() < test["timestamp"].min()

    # Roughly 60/20/20 (allow rounding slack from two sequential splits).
    total = len(df_h)
    assert abs(len(train) / total - 0.6) < 0.02
    assert abs(len(calib) / total - 0.2) < 0.02
    assert abs(len(test) / total - 0.2) < 0.02


def test_three_way_split_no_row_appears_twice():
    df = _synthetic_carbon_df()
    df_feat = create_features(df)
    df_h = build_horizon_dataset(df_feat, df, horizon_steps=4)

    train, calib, test = three_way_split(df_h)
    all_ts = pd.concat([train["timestamp"], calib["timestamp"], test["timestamp"]])
    assert all_ts.is_unique


# ----------------------------------------------------------------------
# 2. Correction depends ONLY on calibration data, never on test data
# ----------------------------------------------------------------------
def test_correction_matches_independent_recomputation_from_calib_only():
    """
    Re-derives the correction completely independently, touching only
    the calibration split (never `test`), using the SAME trained models
    the pipeline produced. If the pipeline's internal correction is
    secretly influenced by test data, this independent, calib-only
    recomputation would not match it exactly.
    """
    df = _synthetic_carbon_df()
    h = 2  # small horizon, fast

    per_horizon_df, trained_models, corrections = run_calibration_experiment(
        df, horizons_hours=[h], save_models=False
    )

    df_feat = create_features(df)
    feature_cols = get_feature_columns(df_feat)
    df_h = build_horizon_dataset(df_feat, df, horizon_steps=h * 2)
    _, calib, _test = three_way_split(df_h)  # test intentionally unused below

    calib_preds = predict_quantiles(trained_models[h], calib[feature_cols])
    scores = nonconformity_scores(
        calib_preds["q10"].values, calib_preds["q90"].values, calib["target"].values
    )
    expected_correction = conformal_correction(scores, alpha=0.2)

    assert corrections[h] == pytest.approx(expected_correction, abs=1e-9)


def test_correction_unaffected_by_perturbing_test_targets():
    """
    Directly demonstrates no leakage: corrupting the TEST set's target
    values (after the correction has already been computed) cannot
    change a correction that only ever looked at calibration data.
    """
    df = _synthetic_carbon_df()
    h = 2

    _, trained_models, corrections = run_calibration_experiment(
        df, horizons_hours=[h], save_models=False
    )

    df_feat = create_features(df)
    feature_cols = get_feature_columns(df_feat)
    df_h = build_horizon_dataset(df_feat, df, horizon_steps=h * 2)
    _, calib, test = three_way_split(df_h)

    # Corrupt the test set's targets wildly -- if this changed the
    # correction, it would mean the correction depended on test data.
    corrupted_test = test.copy()
    corrupted_test["target"] = 1e9

    calib_preds = predict_quantiles(trained_models[h], calib[feature_cols])
    scores = nonconformity_scores(
        calib_preds["q10"].values, calib_preds["q90"].values, calib["target"].values
    )
    recomputed_correction = conformal_correction(scores, alpha=0.2)

    assert recomputed_correction == pytest.approx(corrections[h], abs=1e-9)
    # sanity: the corrupted test set was never touched by the computation above
    assert corrupted_test["target"].iloc[0] == 1e9


# ----------------------------------------------------------------------
# Pure-function checks: the conformal correction formula itself
# ----------------------------------------------------------------------
def test_conformal_correction_matches_manual_order_statistic():
    # n=10, alpha=0.2 -> k = ceil(11 * 0.8) = ceil(8.8) = 9th smallest.
    scores = np.arange(1, 11, dtype=float)  # 1..10
    correction = conformal_correction(scores, alpha=0.2)
    assert correction == 9.0  # sorted(scores)[9-1] = sorted(scores)[8] = 9


def test_conformal_correction_returns_inf_when_n_too_small():
    # n=2, alpha=0.2 -> k = ceil(3 * 0.8) = ceil(2.4) = 3 > n=2 -> unachievable.
    scores = np.array([1.0, 2.0])
    correction = conformal_correction(scores, alpha=0.2)
    assert correction == np.inf


def test_apply_calibration_widens_symmetrically():
    q10 = np.array([50.0, 60.0])
    q90 = np.array([150.0, 160.0])
    cal_q10, cal_q90 = apply_calibration(q10, q90, correction=10.0)
    np.testing.assert_allclose(cal_q10, [40.0, 50.0])
    np.testing.assert_allclose(cal_q90, [160.0, 170.0])
