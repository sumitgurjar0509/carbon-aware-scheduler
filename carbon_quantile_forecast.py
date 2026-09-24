"""
Carbon Intensity Quantile Forecasting using LightGBM
======================================================
Part of: AI-Based Energy and Carbon Optimization in Cloud Computing Environments

WHAT THIS DOES
--------------
Instead of predicting a single number for future carbon intensity
(e.g. "150 gCO2/kWh at 2pm tomorrow"), this trains THREE models that
together give a range with confidence:
    q10 -> "10% chance it's lower than this"
    q50 -> "the median / best-guess point forecast"
    q90 -> "10% chance it's higher than this"

So instead of one number, you get: "likely between 120-180, best
guess 150". This range is what your uncertainty-aware scheduler
will use to decide whether a shift is safe.

HOW TO GET THIS DATA
---------------------
Download historical carbon intensity from the UK Carbon Intensity API:
  https://api.carbonintensity.org.uk/
Save it as a CSV with two columns: timestamp, carbon_intensity
(30-minute resolution is what the API provides).

REQUIREMENTS
------------
pip install lightgbm pandas numpy scikit-learn --break-system-packages
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error


# ----------------------------------------------------------------------
# 1. Load data
# ----------------------------------------------------------------------
def load_carbon_data(csv_path):
    """
    Expects a CSV with columns: timestamp, carbon_intensity
    timestamp should be at regular intervals (e.g. every 30 minutes).
    """
    df = pd.read_csv(csv_path, parse_dates=['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df


# ----------------------------------------------------------------------
# 2. Feature engineering
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

    # --- Calendar features: carbon intensity has strong daily/weekly patterns ---
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)

    # Cyclical encoding: tells the model that hour 23 and hour 0 are
    # adjacent, not far apart (plain integer hour would confuse it).
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    # --- Lag features: past values of the target itself ---
    for lag in lags:
        df[f'lag_{lag}'] = df[target_col].shift(lag)

    # --- Rolling statistics: recent trend/volatility ---
    for window in rolling_windows:
        df[f'roll_mean_{window}'] = df[target_col].shift(1).rolling(window).mean()
        df[f'roll_std_{window}'] = df[target_col].shift(1).rolling(window).std()

    # Drop rows with NaN (the first few rows won't have enough lag history)
    df = df.dropna().reset_index(drop=True)
    return df


# ----------------------------------------------------------------------
# 3. Train/test split -- TIME-BASED, never shuffle time series data
# ----------------------------------------------------------------------
def time_based_split(df, test_frac=0.2):
    """
    Splits chronologically: train on the past, test on the future.
    Never use train_test_split() with shuffle=True on time series --
    that lets the model "see the future" during training, which is
    a classic mistake examiners will ask about.
    """
    split_idx = int(len(df) * (1 - test_frac))
    train = df.iloc[:split_idx].reset_index(drop=True)
    test = df.iloc[split_idx:].reset_index(drop=True)
    return train, test


# ----------------------------------------------------------------------
# 4. Train one LightGBM model per quantile
# ----------------------------------------------------------------------
def train_quantile_models(train_df, feature_cols, target_col='carbon_intensity',
                           quantiles=(0.1, 0.5, 0.9)):
    """
    LightGBM supports quantile regression natively via
    objective='quantile' + alpha=<the quantile level>.
    We train a SEPARATE model for each quantile -- this is the
    standard, simplest approach (as opposed to more complex
    multi-quantile-in-one-model methods).
    """
    models = {}
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]

    for q in quantiles:
        params = {
            'objective': 'quantile',
            'alpha': q,              # <-- this is what makes it predict THIS quantile
            'metric': 'quantile',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'min_data_in_leaf': 20,
            'verbosity': -1,
        }
        train_set = lgb.Dataset(X_train, label=y_train)
        model = lgb.train(params, train_set, num_boost_round=300)
        models[q] = model
        print(f"Trained model for quantile {q}")

    return models


# ----------------------------------------------------------------------
# 5. Predict
# ----------------------------------------------------------------------
def predict_quantiles(models, X):
    """Returns a DataFrame with one column per quantile (e.g. q10, q50, q90)."""
    preds = {}
    for q, model in models.items():
        preds[f'q{int(q * 100)}'] = model.predict(X)
    return pd.DataFrame(preds, index=X.index)


# ----------------------------------------------------------------------
# 6. Baselines (for comparison -- your thesis needs these)
# ----------------------------------------------------------------------
def persistence_baseline(df, target_col='carbon_intensity', horizon=1):
    """Naive baseline: 'the future will look like right now'."""
    return df[target_col].shift(horizon)


def seasonal_naive_baseline(df, target_col='carbon_intensity', period=48):
    """Naive baseline: 'the future will look like it did at this time yesterday'."""
    return df[target_col].shift(period)


# ----------------------------------------------------------------------
# 7. Evaluation
# ----------------------------------------------------------------------
def pinball_loss(y_true, y_pred, quantile):
    """
    THE metric for quantile forecasts -- this is the actual loss
    function quantile regression optimizes. Report this in your
    thesis, not just MAE (MAE only makes sense for the median q50).
    Lower is better.
    """
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

    # Coverage check: of all the true values, what % actually fell
    # inside the predicted [q10, q90] range? Should be close to 80%.
    # If it's much lower, your uncertainty range is too narrow
    # (overconfident). If much higher, it's too wide (unhelpful).
    if 'q10' in preds_df.columns and 'q90' in preds_df.columns:
        inside = ((y_true >= preds_df['q10']) & (y_true <= preds_df['q90'])).mean()
        results['coverage_80pct_interval'] = inside

    return results


# ----------------------------------------------------------------------
# Example usage -- replace 'carbon_intensity.csv' with your real data
# ----------------------------------------------------------------------
if __name__ == '__main__':
    df = load_carbon_data('carbon_intensity.csv')

    df_feat = create_features(df)
    feature_cols = [c for c in df_feat.columns if c not in ['timestamp', 'carbon_intensity']]

    train_df, test_df = time_based_split(df_feat, test_frac=0.2)
    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

    models = train_quantile_models(train_df, feature_cols, quantiles=(0.1, 0.5, 0.9))

    preds = predict_quantiles(models, test_df[feature_cols])

    results = evaluate_forecast(test_df, preds)
    print("\n=== Results ===")
    for k, v in results.items():
        print(f"{k}: {v:.3f}")
