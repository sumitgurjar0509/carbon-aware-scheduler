"""
Tests for the data window, the dataset tag and the model label
(src/data_config.py) and for the loaders that use them.
All files are made in a temporary folder; no real data is touched.
"""
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import data_config  # noqa: E402
from data_config import (  # noqa: E402
    ConfigError,
    check_model_label,
    clean_data_path,
    clear_model_label,
    existing_workload_path,
    gap_report_path,
    get_data_tag,
    get_window,
    list_tags,
    workload_path,
    write_model_label,
)

NOW = datetime(2026, 11, 26, 10, 42, 31, 5, tzinfo=timezone.utc)
OLD = "20250101_20260101"
NEW = "20250201_20260201"


def _utc(*parts):
    return datetime(*parts, tzinfo=timezone.utc)


def _make_dataset(folder, tag, value):
    """Writes a tiny cleaned dataset and gap report for `tag`."""
    pd.DataFrame({"timestamp": ["2025-01-01 00:00:00+00:00"],
                  "carbon_intensity": [value]}).to_csv(folder / f"clean_{tag}.csv", index=False)
    pd.DataFrame([{"range_start": f"{tag[:4]}-{tag[4:6]}-01 00:00:00+00:00",
                   "range_end": "2026-03-01 00:00:00+00:00"}]).to_csv(
        folder / f"gap_report_{tag}.csv", index=False)


@pytest.fixture
def two_datasets(tmp_path):
    """Two datasets where NEW has the later modification time."""
    _make_dataset(tmp_path, OLD, 111)
    _make_dataset(tmp_path, NEW, 222)
    for name in ("clean", "gap_report"):
        os.utime(tmp_path / f"{name}_{OLD}.csv", (1_000_000, 1_000_000))
        os.utime(tmp_path / f"{name}_{NEW}.csv", (2_000_000, 2_000_000))
    return tmp_path


# ----------------------------------------------------------------------
# Window
# ----------------------------------------------------------------------
def test_window_default_is_last_365_days_to_current_hour():
    start, end = get_window(env={}, now=NOW)
    assert end == _utc(2026, 11, 26, 10)
    assert start == end - timedelta(days=365)


def test_window_both_set():
    env = {"CAS_START": "2025-09-23T03:00Z", "CAS_END": "2026-09-23T03:00Z"}
    assert get_window(env=env, now=NOW) == (_utc(2025, 9, 23, 3), _utc(2026, 9, 23, 3))


def test_window_only_end_counts_back_365_days():
    start, end = get_window(env={"CAS_END": "2026-09-23T03:00Z"}, now=NOW)
    assert end == _utc(2026, 9, 23, 3)
    assert start == end - timedelta(days=365)


def test_window_only_start_ends_at_current_hour():
    assert get_window(env={"CAS_START": "2026-01-01"}, now=NOW) == (
        _utc(2026, 1, 1), _utc(2026, 11, 26, 10))


def test_window_other_timezones_are_converted_to_utc():
    env = {"CAS_START": "2026-01-01T05:30+05:30", "CAS_END": "2026-02-01T00:00"}
    assert get_window(env=env, now=NOW) == (_utc(2026, 1, 1), _utc(2026, 2, 1))


def test_window_reads_real_environment_variables(monkeypatch):
    monkeypatch.setenv("CAS_START", "2026-01-01")
    monkeypatch.setenv("CAS_END", "2026-02-01")
    assert get_window() == (_utc(2026, 1, 1), _utc(2026, 2, 1))


def test_window_bad_format_names_the_variable():
    with pytest.raises(ConfigError, match="CAS_END"):
        get_window(env={"CAS_END": "23/09/2026"}, now=NOW)


@pytest.mark.parametrize("start", ["2026-02-01", "2026-03-01"])
def test_window_start_must_be_before_end(start):
    with pytest.raises(ConfigError, match="before"):
        get_window(env={"CAS_START": start, "CAS_END": "2026-02-01"}, now=NOW)


# ----------------------------------------------------------------------
# Tag and file paths
# ----------------------------------------------------------------------
def test_missing_tag_fails_and_lists_available_tags(two_datasets):
    with pytest.raises(ConfigError) as error:
        get_data_tag(env={}, processed_dir=two_datasets)
    message = str(error.value)
    assert "CAS_DATA_TAG is not set" in message
    assert OLD in message and NEW in message


@pytest.mark.parametrize("bad", ["latest", "20250101", "../20250101_20260101"])
def test_malformed_tag_is_rejected(bad, tmp_path):
    with pytest.raises(ConfigError, match="YYYYMMDD_YYYYMMDD"):
        get_data_tag(env={"CAS_DATA_TAG": bad}, processed_dir=tmp_path)


def test_list_tags(two_datasets):
    assert list_tags(two_datasets) == [OLD, NEW]


def test_tag_wins_over_newest_file(two_datasets, monkeypatch):
    """The old bug: the newest file was used. Now the tag decides."""
    monkeypatch.setenv("CAS_DATA_TAG", OLD)
    assert clean_data_path(processed_dir=two_datasets).name == f"clean_{OLD}.csv"
    assert gap_report_path(processed_dir=two_datasets).name == f"gap_report_{OLD}.csv"


@pytest.mark.parametrize("finder", [clean_data_path, gap_report_path, existing_workload_path])
def test_unknown_tag_file_fails_clearly(finder, two_datasets):
    with pytest.raises(ConfigError, match="File not found"):
        finder("19990101_20000101", two_datasets)


def test_workload_path_is_tagged_and_ignores_untagged_file(two_datasets):
    (two_datasets / "workload_trace.csv").write_text("old untagged trace\n")
    assert workload_path(OLD, two_datasets).name == f"workload_trace_{OLD}.csv"
    with pytest.raises(ConfigError, match="workload_generator"):
        existing_workload_path(OLD, two_datasets)
    workload_path(OLD, two_datasets).write_text("job_id\n1\n")
    assert existing_workload_path(OLD, two_datasets).exists()


# ----------------------------------------------------------------------
# Loaders in the pipeline scripts
# ----------------------------------------------------------------------
def test_load_clean_data_uses_the_tag(two_datasets, monkeypatch):
    from forecasting import load_clean_data

    monkeypatch.setenv("CAS_DATA_TAG", OLD)
    df = load_clean_data(processed_dir=two_datasets)
    assert df["carbon_intensity"].tolist() == [111]


def test_load_clean_data_without_tag_fails(two_datasets, monkeypatch):
    from forecasting import load_clean_data

    monkeypatch.delenv("CAS_DATA_TAG", raising=False)
    with pytest.raises(ConfigError, match="CAS_DATA_TAG is not set"):
        load_clean_data(processed_dir=two_datasets)


def test_workload_data_range_comes_from_tagged_report(two_datasets):
    from workload_generator import _load_data_range

    start, _ = _load_data_range(OLD, two_datasets)
    assert start == pd.Timestamp("2025-01-01", tz="UTC")


def test_simulator_load_workload_reads_tagged_file(tmp_path, monkeypatch):
    import simulator

    path = tmp_path / f"workload_trace_{OLD}.csv"
    path.write_text("job_id,arrival_time,deadline\n"
                    "1,2025-01-01 00:00:00+00:00,2025-01-01 07:00:00+00:00\n")
    monkeypatch.setattr(simulator, "existing_workload_path", lambda tag=None: path)
    df = simulator.load_workload(OLD)
    assert str(df["arrival_time"].dt.tz) == "UTC"


# ----------------------------------------------------------------------
# Model label
# ----------------------------------------------------------------------
def test_model_label_accepts_matching_calibrated_models(tmp_path):
    write_model_label(tmp_path, OLD, stage="calibration")
    check_model_label(tmp_path, OLD)  # must not raise


@pytest.mark.parametrize("tag, stage, message", [
    (NEW, "calibration", "trained on dataset"),
    (OLD, "forecasting", "not calibrated"),
])
def test_model_label_rejects_wrong_models(tmp_path, tag, stage, message):
    write_model_label(tmp_path, tag, stage=stage)
    with pytest.raises(ConfigError, match=message):
        check_model_label(tmp_path, OLD)


def test_model_label_missing_after_clear(tmp_path):
    clear_model_label(tmp_path)  # fine when there is nothing to remove
    write_model_label(tmp_path, OLD, stage="calibration")
    clear_model_label(tmp_path)
    assert not (tmp_path / data_config.MODEL_LABEL_NAME).exists()
    with pytest.raises(ConfigError, match="no dataset label"):
        check_model_label(tmp_path, OLD)


def test_simulator_refuses_models_from_another_dataset(tmp_path, monkeypatch):
    import calibration
    import simulator

    write_model_label(tmp_path, NEW, stage="calibration")
    monkeypatch.setattr(calibration, "MODELS_DIR", tmp_path)
    monkeypatch.setenv("CAS_DATA_TAG", OLD)
    with pytest.raises(ConfigError, match="trained on dataset"):
        simulator.build_forecast_tables(pd.DataFrame(), None, None)
