"""Tests for scripts/compare_tables.py."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from compare_m2_tiebreak import compare_versions  # noqa: E402
from compare_tables import compare_dirs, compare_frames, main  # noqa: E402
import pytest  # noqa: E402


def _summary(cut, delay, local):
    return pd.DataFrame({"policy": ["random_fifo", "lp_actual_a1"], "alpha": [np.nan, 1.0],
                         "pct_reduction_vs_random_fifo": cut, "mean_start_delay_h": delay,
                         "local_job_share": local})


def test_compare_m2_versions_side_by_side():
    out = compare_versions(_summary([0, 30], [0, 10], [0.1, 0.2]),
                           _summary([0, 29], [0, 4], [0.1, 0.3]))
    row = out.set_index("policy").loc["lp_actual_a1"]
    assert row["cut_vs_fifo_pct_change"] == pytest.approx(-1)
    assert row["mean_delay_h_v1"] == 10 and row["mean_delay_h_v2"] == 4
    assert row["local_share_change"] == pytest.approx(0.1)


def test_compare_m2_versions_requires_same_policies():
    with pytest.raises(ValueError):
        compare_versions(_summary([0, 1], [0, 1], [0, 1]),
                         _summary([0, 1], [0, 1], [0, 1]).iloc[::-1])


def _table(**changes):
    data = {"name": ["a", "b"], "value": [100.0, np.nan], "flag": [True, False]}
    data.update(changes)
    return pd.DataFrame(data)


def test_identical_tables_match():
    result = compare_frames(_table(), _table())
    assert result == {"problems": [], "max_abs_diff": 0.0}


def test_tiny_difference_is_within_tolerance():
    assert compare_frames(_table(), _table(value=[100.0 + 1e-7, np.nan]))["problems"] == []


def test_real_number_difference_is_reported():
    result = compare_frames(_table(), _table(value=[100.01, np.nan]))
    assert "value" in result["problems"][0]
    assert abs(result["max_abs_diff"] - 0.01) < 1e-9


def test_number_versus_empty_cell_is_a_difference():
    assert compare_frames(_table(), _table(value=[100.0, 0.0]))["problems"]


def test_text_and_boolean_must_match_exactly():
    assert compare_frames(_table(), _table(name=["a", "c"]))["problems"]
    assert compare_frames(_table(), _table(flag=[True, True]))["problems"]


def test_column_and_row_count_changes_are_reported():
    assert "columns" in compare_frames(_table(), _table().drop(columns="flag"))["problems"][0]
    assert "row count" in compare_frames(_table(), _table().head(1))["problems"][0]


def _dirs(tmp_path):
    committed, new = tmp_path / "committed", tmp_path / "new"
    committed.mkdir()
    new.mkdir()
    return committed, new


def test_compare_dirs_statuses(tmp_path):
    committed, new = _dirs(tmp_path)
    for folder in (committed, new):
        _table().to_csv(folder / "same.csv", index=False)
    _table().to_csv(committed / "changed.csv", index=False)
    _table(name=["x", "y"]).to_csv(new / "changed.csv", index=False)
    _table().to_csv(committed / "gone.csv", index=False)
    _table().to_csv(new / "extra.csv", index=False)

    statuses = {r["file"]: r["status"] for r in compare_dirs(committed, new)}
    assert statuses == {"same.csv": "MATCH", "changed.csv": "DIFFERENT",
                        "gone.csv": "MISSING", "extra.csv": "NO_BASELINE"}


def test_main_exit_codes(tmp_path):
    committed, new = _dirs(tmp_path)
    args = ["--committed", str(committed), "--new", str(new)]
    assert main(args) == 2  # empty baseline must never pass

    _table().to_csv(committed / "t.csv", index=False)
    assert main(args) == 1  # missing

    _table().to_csv(new / "t.csv", index=False)
    _table().to_csv(new / "extra.csv", index=False)
    assert main(args) == 0  # no-baseline files do not fail the run

    _table(name=["x", "y"]).to_csv(new / "t.csv", index=False)
    assert main(args) == 1
