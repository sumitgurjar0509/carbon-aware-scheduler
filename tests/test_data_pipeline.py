"""
Sanity tests for data_pipeline.py.

These deliberately avoid hitting the live API (network calls in unit
tests are slow/flaky and not what we're testing here) -- they construct
synthetic "raw API response" DataFrames and check that clean_and_resample
handles gaps the way we claim it does.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from data_pipeline import clean_and_resample  # noqa: E402


def _make_raw(timestamps, actuals, forecasts=None):
    forecasts = forecasts if forecasts is not None else actuals
    return pd.DataFrame(
        {
            "from": pd.to_datetime(timestamps, utc=True),
            "to": pd.to_datetime(timestamps, utc=True) + pd.Timedelta(minutes=30),
            "forecast": forecasts,
            "actual": actuals,
            "index": ["moderate"] * len(timestamps),
        }
    )


def test_no_gaps_passthrough():
    ts = pd.date_range("2026-01-01", periods=10, freq="30min", tz="UTC")
    raw = _make_raw(ts, actuals=list(range(100, 110)))
    clean, report = clean_and_resample(raw)

    assert report["missing_periods_total"] == 0
    assert clean["carbon_intensity"].isna().sum() == 0
    assert list(clean["carbon_intensity"]) == list(range(100, 110))


def test_short_gap_is_interpolated():
    # Missing settlement periods 3 and 4 (a 2-period internal gap).
    ts = pd.date_range("2026-01-01", periods=10, freq="30min", tz="UTC")
    keep = [0, 1, 2, 5, 6, 7, 8, 9]
    raw = _make_raw(ts[keep], actuals=[100, 101, 102, 105, 106, 107, 108, 109])

    clean, report = clean_and_resample(raw, max_gap_periods=4)

    assert report["missing_periods_total"] == 2
    assert report["short_gaps_interpolated"] == 2
    assert report["long_gaps_left_as_nan"] == 0
    # linearly interpolated between 102 (idx2) and 105 (idx5) -> 103, 104
    filled = clean.loc[[3, 4], "carbon_intensity"].tolist()
    assert np.allclose(filled, [103.0, 104.0])
    assert clean.loc[[3, 4], "was_interpolated"].all()


def test_long_gap_left_as_nan_not_fabricated():
    # Missing 6 consecutive periods (> default max_gap_periods=4) --
    # must NOT be filled, must remain NaN.
    ts = pd.date_range("2026-01-01", periods=12, freq="30min", tz="UTC")
    keep = [0, 1, 2, 9, 10, 11]
    raw = _make_raw(ts[keep], actuals=[100, 101, 102, 150, 151, 152])

    clean, report = clean_and_resample(raw, max_gap_periods=4)

    assert report["long_gaps_left_as_nan"] == 6
    assert report["short_gaps_interpolated"] == 0
    long_gap_rows = clean.iloc[3:9]
    assert long_gap_rows["carbon_intensity"].isna().all()
    assert not long_gap_rows["was_interpolated"].any()


def test_trailing_edge_gap_not_fabricated():
    # A single missing period at the very END of the range (e.g. the
    # most recent, not-yet-settled settlement period, which the real
    # API returns WITH a row but actual=null) must be left as NaN, not
    # extrapolated, even though its "gap length" (1) is below the
    # short-gap threshold.
    ts = pd.date_range("2026-01-01", periods=5, freq="30min", tz="UTC")
    raw = _make_raw(ts, actuals=[100, 101, 102, 103, np.nan])

    clean, report = clean_and_resample(raw, max_gap_periods=4)

    assert clean["carbon_intensity"].isna().sum() == 1
    assert not clean["was_interpolated"].iloc[-1]
    assert report["edge_gaps_left_as_nan"] == 1
