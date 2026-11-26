"""
Conformalized Quantile Regression (CQR) Calibration
=====================================================
Part of: AI-Based Energy and Carbon Optimization in Cloud Computing Environments

WHY THIS STEP EXISTS
---------------------
`forecasting.py` found that the LightGBM quantile model's nominal 80%
[q10, q90] interval only actually covers the true value ~62.4% of the
time on the test set (averaged across the 1h-24h horizons) -- the model
is overconfident. That directly undermines this thesis's RQ3 (does an
uncertainty-aware scheduler reduce deadline violations?): if the
"80% confident" interval is really only ~62% reliable, an
uncertainty-aware scheduler built on it will violate deadlines more
often than its own stated confidence would suggest.

This module fixes that with Conformalized Quantile Regression (CQR):

    Romano, Y., Patterson, E., & Candes, E. (2019). "Conformalized
    Quantile Regression." Advances in Neural Information Processing
    Systems (NeurIPS 2019), 32, pp. 3543-3553.

CQR wraps ANY quantile regressor with a post-hoc correction, computed on
a held-out calibration set, that provably achieves the target marginal
coverage (in expectation over a fresh draw, distribution-free, no
assumptions on the model or the data distribution beyond exchangeability
between calibration and test points) -- it does not require the base
quantile model to be well-specified or well-calibrated to begin with.

THREE-WAY CHRONOLOGICAL SPLIT (why the calibration set must be separate)
--------------------------------------------------------------------------
CQR's coverage guarantee relies on the calibration points being
exchangeable with the test point, conditional on the model -- i.e. the
model must never have seen the calibration set during training, exactly
as it must never see the test set. Reusing training data (or the test
set) as the calibration set breaks the guarantee: nonconformity scores
computed on data the model already fit to are systematically smaller
than genuinely-unseen scores, which would produce an UNDER-corrected
(too narrow) interval -- precisely the overconfidence problem we are
trying to fix. So we split chronologically into three disjoint,
time-ordered blocks:

    train (60%) -> calibration (20%) -> test (20%)

train/test boundaries stay the same 80/20 chronological split already
used in forecasting.py (so the final test-set evaluation numbers are
comparable); the calibration set is carved out of what was previously
the training block, immediately before the test block. All three sets
never overlap and always satisfy train < calibration < test in time,
enforced by reusing `time_based_split` (unchanged) twice in sequence --
never a random/shuffled split, for the same reason as everywhere else
in this project (time series -> chronological splits only).

MODELS ARE RETRAINED HERE (train-only, not train+calib)
---------------------------------------------------------
The 72 models forecasting.py saved to results/models/quantile/ were
trained on an 80% train block that INCLUDED what is now the calibration
set. For CQR's guarantee to hold, the calibration set must be unseen by
the model, so this module retrains all 72 models from scratch on the
smaller 60% train-only block and OVERWRITES those files -- they become
the canonical models used from here on (see `predict_calibrated`).
forecasting.py's original run remains valid as the evidence that
motivated this step (the miscalibration finding), even though its
specific model files are superseded.

NONCONFORMITY SCORE AND THE CORRECTION (Algorithm 1 in the paper)
--------------------------------------------------------------------
For each calibration point i with true value y_i and raw predicted
interval [q10_pred_i, q90_pred_i]:

    s_i = max(q10_pred_i - y_i, y_i - q90_pred_i)

s_i is POSITIVE when y_i fell outside the interval (on either side --
whichever side, by however much) and NEGATIVE when it fell safely
inside. The correction is the following EMPIRICAL QUANTILE of the
scores {s_i}:

    q_level = ceil((n_calib + 1) * (1 - alpha)) / n_calib

Why not just the naive (1-alpha)=80th percentile of the scores? Because
a plain percentile (e.g. `np.quantile(scores, 0.8)`) is only
asymptotically correct as n_calib -> infinity. For a FINITE calibration
set, split-conformal prediction's exact, finite-sample, distribution-
free coverage guarantee comes specifically from taking the
ceil((n+1)(1-alpha))-th order statistic (not the n*(1-alpha)-th) -- the
"+1" accounts for the (unobserved) test point itself in the
exchangeability argument that proves the guarantee (a new draw is
equally likely to rank anywhere among the n+1 calibration-plus-test
scores). Using the naive percentile instead would systematically
under-cover by a small but non-zero amount for any finite n -- exactly
the kind of subtle gap that would otherwise reintroduce part of the
overconfidence problem this whole module exists to fix.

Given the correction Q_h (one per horizon h), the calibrated interval
for a new point is:

    q10_calibrated = q10_pred - Q_h
    q90_calibrated = q90_pred + Q_h

(q50 is left untouched -- CQR only ever widens/shifts the outer
quantiles, never the median.)
"""
import sys
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_config import clear_model_label, get_data_tag, write_model_label  # noqa: E402
from forecasting import (  # noqa: E402
    HORIZON_HOURS,
    MODELS_DIR,
    NATIVE_STEP_MINUTES,
    PLOTS_DIR,
    QUANTILES,
    TABLES_DIR,
    build_horizon_dataset,
    create_features,
    get_feature_columns,
    load_clean_data,
    predict_quantiles,
    time_based_split,
    train_quantile_models,
)

ALPHA = 0.2          # 1 - alpha = 80% target coverage, matching the q10/q90 interval
TARGET_COVERAGE = 1 - ALPHA
TEST_FRAC = 0.2      # unchanged from forecasting.py, for comparable test-set numbers
CALIB_FRAC = 0.2     # carved out of what used to be the training block
CORRECTIONS_PATH = MODELS_DIR / "cqr_corrections.csv"


# ----------------------------------------------------------------------
# Three-way chronological split
# ----------------------------------------------------------------------
def three_way_split(df_h: pd.DataFrame, test_frac: float = TEST_FRAC,
                     calib_frac: float = CALIB_FRAC):
    """
    Chronological train / calibration / test split, built from two
    applications of the unmodified `time_based_split`:
        1. peel off the final `test_frac` as TEST.
        2. of what remains (the leading 1-test_frac block), peel off the
           final `calib_frac / (1 - test_frac)` as CALIBRATION -- this
           relative fraction is what makes the calibration block equal
           exactly `calib_frac` of the ORIGINAL total (e.g. 0.2/0.8=0.25
           of the remaining 80% = 20% of the original whole).
    Result: train ~60%, calibration ~20%, test ~20%, strictly
    chronological with zero overlap (train.timestamp.max() <
    calib.timestamp.min() and calib.timestamp.max() < test.timestamp.min()
    by construction, since time_based_split never shuffles).
    """
    train_calib, test = time_based_split(df_h, test_frac=test_frac)
    relative_calib_frac = calib_frac / (1 - test_frac)
    train, calib = time_based_split(train_calib, test_frac=relative_calib_frac)
    return train, calib, test


# ----------------------------------------------------------------------
# CQR core: nonconformity score + finite-sample-corrected quantile
# ----------------------------------------------------------------------
def nonconformity_scores(q10_pred: np.ndarray, q90_pred: np.ndarray,
                          y_true: np.ndarray) -> np.ndarray:
    """s_i = max(q10_pred_i - y_i, y_i - q90_pred_i). See module docstring."""
    q10_pred = np.asarray(q10_pred)
    q90_pred = np.asarray(q90_pred)
    y_true = np.asarray(y_true)
    return np.maximum(q10_pred - y_true, y_true - q90_pred)


def conformal_correction(scores: np.ndarray, alpha: float = ALPHA) -> float:
    """
    The finite-sample-corrected (1-alpha) quantile of the calibration
    nonconformity scores: the ceil((n+1)(1-alpha))-th order statistic,
    NOT a plain percentile -- see module docstring for why this
    specific form is what gives CQR its exact finite-sample coverage
    guarantee.

    Edge case: if ceil((n+1)(1-alpha)) > n (only possible when n is very
    small), the target coverage is unachievable from this many
    calibration points -- the correct, honest behaviour per the theory
    is to return +inf (the calibrated interval becomes [-inf, +inf],
    i.e. "we cannot certify 80% coverage with this little calibration
    data" rather than silently under-covering).
    """
    scores = np.sort(np.asarray(scores))
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return np.inf
    return float(scores[k - 1])  # k-th smallest, 1-indexed


def apply_calibration(q10_pred: np.ndarray, q90_pred: np.ndarray, correction: float):
    """q10_calibrated = q10_pred - Q; q90_calibrated = q90_pred + Q. q50 untouched."""
    return np.asarray(q10_pred) - correction, np.asarray(q90_pred) + correction


# ----------------------------------------------------------------------
# Main per-horizon calibration experiment
# ----------------------------------------------------------------------
def run_calibration_experiment(df: pd.DataFrame, horizons_hours=HORIZON_HOURS,
                                quantiles=QUANTILES, alpha=ALPHA, save_models=True):
    """
    For each horizon: train on TRAIN only, compute the CQR correction on
    CALIBRATION, evaluate both raw and calibrated intervals on the
    held-out TEST set (never touched by training or calibration).

    Returns (per_horizon_df, trained_models, corrections) where
    trained_models[h] = {quantile: booster} and corrections[h] = the
    scalar CQR correction for horizon h -- both kept in memory so the
    sanity tests can independently re-verify the correction without
    re-reading from disk.
    """
    df_feat = create_features(df)
    feature_cols = get_feature_columns(df_feat)

    if save_models:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        clear_model_label(MODELS_DIR)

    rows = []
    trained_models = {}
    corrections = {}

    for h in horizons_hours:
        horizon_steps = h * (60 // NATIVE_STEP_MINUTES)

        df_h = build_horizon_dataset(df_feat, df, horizon_steps)
        train_df, calib_df, test_df = three_way_split(df_h)

        models = train_quantile_models(train_df, feature_cols, target_col="target",
                                        quantiles=quantiles)

        # --- calibrate: nonconformity scores on the CALIBRATION set only ---
        calib_preds = predict_quantiles(models, calib_df[feature_cols])
        scores = nonconformity_scores(
            calib_preds["q10"].values, calib_preds["q90"].values, calib_df["target"].values
        )
        correction = conformal_correction(scores, alpha=alpha)

        # --- evaluate raw vs calibrated on the held-out TEST set ---
        test_preds = predict_quantiles(models, test_df[feature_cols])
        y_test = test_df["target"].values

        raw_q10, raw_q90 = test_preds["q10"].values, test_preds["q90"].values
        raw_coverage = np.mean((y_test >= raw_q10) & (y_test <= raw_q90))
        raw_width = np.mean(raw_q90 - raw_q10)

        cal_q10, cal_q90 = apply_calibration(raw_q10, raw_q90, correction)
        cal_coverage = np.mean((y_test >= cal_q10) & (y_test <= cal_q90))
        cal_width = np.mean(cal_q90 - cal_q10)

        rows.append({
            "horizon_hours": h,
            "n_train": len(train_df),
            "n_calib": len(calib_df),
            "n_test": len(test_df),
            "correction": correction,
            "coverage_raw": raw_coverage,
            "coverage_calibrated": cal_coverage,
            "width_raw": raw_width,
            "width_calibrated": cal_width,
        })

        trained_models[h] = models
        corrections[h] = correction

        if save_models:
            for q, model in models.items():
                model.save_model(str(MODELS_DIR / f"h{h:02d}h_q{int(q*100):02d}.txt"))

        print(f"  horizon={h:>2}h  correction={correction:6.2f}  "
              f"coverage raw={raw_coverage:.1%} -> calibrated={cal_coverage:.1%}  "
              f"width raw={raw_width:5.1f} -> calibrated={cal_width:5.1f}")

    per_horizon_df = pd.DataFrame(rows)

    if save_models:
        corrections_df = per_horizon_df[["horizon_hours", "correction", "n_calib"]].copy()
        corrections_df["alpha"] = alpha
        corrections_df.to_csv(CORRECTIONS_PATH, index=False)
        print(f"Saved corrections to {CORRECTIONS_PATH}")

    return per_horizon_df, trained_models, corrections


# ----------------------------------------------------------------------
# Public API for the scheduler (Step 5 onward): calibrated predictions
# ----------------------------------------------------------------------
def load_correction(horizon_hours: int, corrections_path: Path = CORRECTIONS_PATH) -> float:
    """Reads the saved CQR correction for one horizon."""
    corrections_df = pd.read_csv(corrections_path)
    match = corrections_df[corrections_df["horizon_hours"] == horizon_hours]
    if match.empty:
        raise ValueError(f"No saved CQR correction for horizon={horizon_hours}h "
                          f"in {corrections_path}")
    return float(match["correction"].iloc[0])


def predict_calibrated(horizon_hours: int, X: pd.DataFrame,
                        models_dir: Path = MODELS_DIR,
                        corrections_path: Path = CORRECTIONS_PATH) -> pd.DataFrame:
    """
    THE function the simulator/schedulers should call from Step 5
    onward -- loads the saved (train-only-fit) models for `horizon_hours`,
    predicts q10/q50/q90, and applies the saved CQR correction to q10/q90.
    q50 (the point forecast) is returned unchanged, per CQR's definition.
    """
    preds = {}
    for q in QUANTILES:
        model_path = models_dir / f"h{horizon_hours:02d}h_q{int(q*100):02d}.txt"
        booster = lgb.Booster(model_file=str(model_path))
        preds[f"q{int(q*100)}"] = booster.predict(X)

    correction = load_correction(horizon_hours, corrections_path)
    q10_cal, q90_cal = apply_calibration(preds["q10"], preds["q90"], correction)

    return pd.DataFrame({
        "q10": q10_cal,
        "q50": preds["q50"],
        "q90": q90_cal,
    }, index=X.index)


# ----------------------------------------------------------------------
# Comparison plot: coverage vs horizon, before/after calibration
# ----------------------------------------------------------------------
def plot_coverage_comparison(per_horizon_df: pd.DataFrame, out_path: Path,
                              target_coverage: float = TARGET_COVERAGE):
    BLUE = "#2a78d6"      # calibrated -- the one we actually use going forward
    ORANGE = "#eb6834"    # raw -- the miscalibrated baseline being replaced
    INK_SECONDARY = "#52514e"
    INK_MUTED = "#898781"
    GRIDLINE = "#e1e0d9"
    SURFACE = "#fcfcfb"

    fig, ax = plt.subplots(figsize=(9, 5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    ax.axhline(target_coverage, color=INK_MUTED, linewidth=1.2, linestyle="--",
               label=f"target ({target_coverage:.0%})")
    ax.plot(per_horizon_df["horizon_hours"], per_horizon_df["coverage_raw"],
            color=ORANGE, linewidth=2, marker="o", markersize=4,
            label="Raw (uncalibrated)")
    ax.plot(per_horizon_df["horizon_hours"], per_horizon_df["coverage_calibrated"],
            color=BLUE, linewidth=2, marker="o", markersize=4,
            label="CQR-calibrated")

    ax.set_title("80% interval coverage vs horizon — before/after CQR calibration",
                 color="#0b0b0b", fontsize=12, fontweight="bold", loc="left")
    ax.set_xlabel("forecast horizon (hours)", color=INK_SECONDARY, fontsize=9)
    ax.set_ylabel("empirical coverage", color=INK_SECONDARY, fontsize=9)
    ax.set_ylim(0.4, 1.0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.set_xticks(per_horizon_df["horizon_hours"][::2])
    ax.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(INK_MUTED)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY, loc="lower right")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    df = load_clean_data()

    print(f"\nRunning CQR calibration for horizons "
          f"{HORIZON_HOURS[0]}h..{HORIZON_HOURS[-1]}h "
          f"(train {1-TEST_FRAC-CALIB_FRAC:.0%} / calib {CALIB_FRAC:.0%} / "
          f"test {TEST_FRAC:.0%}, chronological)...")
    per_horizon_df, trained_models, corrections = run_calibration_experiment(df)
    write_model_label(MODELS_DIR, get_data_tag(), stage="calibration")

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    per_horizon_path = TABLES_DIR / "calibration_by_horizon.csv"
    per_horizon_df.to_csv(per_horizon_path, index=False)
    print(f"\nSaved per-horizon results to {per_horizon_path}")

    summary = per_horizon_df[[
        "coverage_raw", "coverage_calibrated", "width_raw", "width_calibrated"
    ]].mean()
    summary_path = TABLES_DIR / "calibration_summary.csv"
    summary.to_frame("mean_across_horizons").to_csv(summary_path)

    print("\n=== Calibration summary (averaged across 1h-24h horizons) ===")
    print(f"Coverage: raw={summary['coverage_raw']:.1%} -> "
          f"calibrated={summary['coverage_calibrated']:.1%} "
          f"(target={TARGET_COVERAGE:.0%})")
    print(f"Width:    raw={summary['width_raw']:.1f} -> "
          f"calibrated={summary['width_calibrated']:.1f} gCO2/kWh "
          f"(+{summary['width_calibrated'] - summary['width_raw']:.1f}, "
          f"{(summary['width_calibrated']/summary['width_raw'] - 1):.1%} wider)")

    print("\n=== Key horizons (6h, 12h, 24h) ===")
    key_rows = per_horizon_df[per_horizon_df["horizon_hours"].isin([1, 6, 12, 24])]
    print(key_rows.round(2).to_string(index=False))

    plot_coverage_comparison(per_horizon_df, PLOTS_DIR / "coverage_before_after_cqr.png")
