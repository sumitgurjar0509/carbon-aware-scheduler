import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from workload_generator import generate_workload  # noqa: E402


def _data_range():
    start = pd.Timestamp("2025-01-01", tz="UTC")
    end = start + pd.Timedelta(days=60)
    return start, end


def test_job_count_and_deadline_within_data_range():
    start, end = _data_range()
    df = generate_workload(start, end, n_jobs=200, seed=1)

    assert len(df) == 200
    assert (df["deadline"] <= end).all()
    assert (df["arrival_time"] >= start).all()


def test_deadline_equals_arrival_plus_duration_plus_slack():
    start, end = _data_range()
    df = generate_workload(start, end, n_jobs=50, seed=2)

    expected = (
        df["arrival_time"]
        + pd.to_timedelta(df["duration_minutes"], unit="m")
        + pd.to_timedelta(df["slack_hours"], unit="h")
    )
    assert (df["deadline"] == expected).all()


def test_slack_buckets_are_stratified_evenly():
    start, end = _data_range()
    df = generate_workload(start, end, n_jobs=300, seed=3)

    counts = df["slack_hours"].value_counts()
    assert set(counts.index) == {6, 12, 24}
    # Exact even split (300 / 3 = 100 each) via round-robin stratification.
    assert (counts == 100).all()


def test_duration_is_capped():
    start, end = _data_range()
    df = generate_workload(start, end, n_jobs=500, duration_cap_minutes=240, seed=4)

    assert df["duration_minutes"].max() <= 240
    assert df["duration_minutes"].min() > 0


def test_arrivals_are_sorted_and_within_window():
    start, end = _data_range()
    df = generate_workload(start, end, n_jobs=100, seed=5)

    assert df["arrival_time"].is_monotonic_increasing
    assert (df["arrival_time"] < end).all()
