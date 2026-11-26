"""Tests for the Stage 2 guards and variant list (no test-period data is touched)."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import lp_scheduler as lp  # noqa: E402
import m3_stage2 as s2  # noqa: E402

FROZEN = {"beta_cvar_lp": 0.0, "beta_cvar_lp_replan": 0.0, "best_positive_beta_cvar_lp": 0.1,
          "best_positive_beta_cvar_lp_replan": 0.1, "beta_tail_cvar_lp_replan": 0.25}


def test_refuses_when_test_dir_has_any_file(tmp_path):
    s2.refuse_if_results_exist(tmp_path / "missing")        # absent: fine
    s2.refuse_if_results_exist(tmp_path)                    # empty: fine
    (tmp_path / "stage2_log.txt").write_text("x")
    with pytest.raises(SystemExit, match="REFUSED"):
        s2.refuse_if_results_exist(tmp_path)


def test_refuses_when_code_is_dirty():
    with pytest.raises(SystemExit, match="uncommitted"):
        s2.refuse_if_dirty(False)
    s2.refuse_if_dirty(True)


def test_log_is_never_overwritten(tmp_path, monkeypatch):
    log = tmp_path / "stage2_log.txt"
    log.write_text("first run")
    with pytest.raises(FileExistsError):
        s2.Tee(log)
    assert log.read_text() == "first run"


def test_variant_list_with_beta_zero_uses_aliases_and_extras():
    runs, aliases = s2.planned_variants(FROZEN)
    names = [r["name"] for r in runs]
    assert len(names) == len(set(names))
    assert names.count("random_fifo") == 1
    for a in ("a1", "a0.5"):
        for n in ("actual_lp", "base_lp", "base_lp_replan", "base_lp_recal", "base_lp_replan_recal",
                  "mean_lp", "mean_lp_replan", "cvar_lp_bestpos", "cvar_lp_replan_bestpos",
                  "cvar_lp_replan_tail"):
            assert f"{n}_{a}" in names
        assert aliases[f"cvar_lp_{a}"] == f"mean_lp_{a}"
        assert aliases[f"cvar_lp_replan_{a}"] == f"mean_lp_replan_{a}"
    tail = next(r for r in runs if r["name"] == "cvar_lp_replan_tail_a1")
    assert tail["beta"] == 0.25 and tail["replan"] is True
    base = next(r for r in runs if r["name"] == "base_lp_a1")
    assert base["table"] == "raw" and base["kind"] == "point"


def test_variant_list_with_positive_beta_and_no_tail():
    runs, aliases = s2.planned_variants(dict(FROZEN, beta_cvar_lp_replan=0.5,
                                             beta_tail_cvar_lp_replan=None))
    names = [r["name"] for r in runs]
    assert "cvar_lp_replan_a1" in names and "cvar_lp_replan_a1" not in aliases
    assert not any(n.startswith("cvar_lp_replan_tail") for n in names)
    assert not any(n.startswith("cvar_lp_replan_bestpos") for n in names)


def test_same_csv_text_ignores_line_endings():
    assert s2.same_csv_text("a,b\r\n1,2\r\n", "a,b\n1,2\n")
    assert not s2.same_csv_text("a,b\n1,2\n", "a,b\n1,3\n")


def test_regenerated_m2_jobs_match_the_committed_file():
    """The exact job generation call used by Stage 2 reproduces M2's committed jobs file."""
    m2 = Path(__file__).resolve().parent.parent / "results" / "tables" / "m2_base_lp" / \
        f"jobs_seed{lp.JOB_SEED}.csv"
    first = pd.Timestamp("2026-07-12 09:00", tz="UTC")
    last = pd.Timestamp("2026-09-23 03:00", tz="UTC")
    start = pd.Timestamp("2026-07-12 08:30", tz="UTC")
    jobs = lp.generate_jobs(first, last, window_start=start)
    assert s2.same_csv_text(jobs.to_csv(index=False), m2.read_bytes().decode("utf-8"))
