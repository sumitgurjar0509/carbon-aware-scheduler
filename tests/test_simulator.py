"""
Sanity tests for simulator.py / schedulers.py, per the task's own
correctness requirements:

  1. Deadline violations are genuinely 0 under normal conditions (the
     forced-start boundary is always <= the true latest allowable
     start -- see simulator.py's module docstring, point 2).
  2. Oracle's realized carbon cost is <= every other scheduler's, for
     EVERY job -- oracle is a per-job upper bound by construction
     (after the design correction documented in schedulers.py's
     oracle() docstring: it compares true INTEGRATED cost, not point
     intensity, specifically so this holds even when a job's duration
     spans multiple settlement periods).
  3. The carbon-integration logic (realized_carbon_cost) matches a
     hand-computed example for a job spanning exactly 2 settlement
     periods.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from simulator import (  # noqa: E402
    SimContext,
    build_decision_grid,
    realized_carbon_cost,
    simulate_job,
)
from schedulers import SCHEDULERS  # noqa: E402


# ----------------------------------------------------------------------
# 3. Carbon-integration hand-computed example
# ----------------------------------------------------------------------
def test_realized_carbon_cost_exact_two_periods():
    """
    Job starts exactly on a period boundary and runs exactly 60 minutes
    (2 full settlement periods) -- the simplest possible case, hand
    verified: cost = 1.0 * intensity_period1 + 1.0 * intensity_period2.
    """
    actual_series = pd.Series(
        {
            pd.Timestamp("2026-01-01 00:00", tz="UTC"): 100.0,
            pd.Timestamp("2026-01-01 00:30", tz="UTC"): 200.0,
            pd.Timestamp("2026-01-01 01:00", tz="UTC"): 300.0,
        }
    )
    cost = realized_carbon_cost(
        actual_series, pd.Timestamp("2026-01-01 00:00", tz="UTC"), duration_minutes=60.0
    )
    assert cost == pytest.approx(100.0 + 200.0)


def test_realized_carbon_cost_partial_final_period():
    """
    Job starts on a period boundary, runs 45 minutes: full 30 min of
    period 1 + 15/30 = 0.5 of period 2.
    Hand-computed: 1.0*100 + 0.5*200 = 200.0.
    """
    actual_series = pd.Series(
        {
            pd.Timestamp("2026-01-01 00:00", tz="UTC"): 100.0,
            pd.Timestamp("2026-01-01 00:30", tz="UTC"): 200.0,
        }
    )
    cost = realized_carbon_cost(
        actual_series, pd.Timestamp("2026-01-01 00:00", tz="UTC"), duration_minutes=45.0
    )
    assert cost == pytest.approx(1.0 * 100.0 + 0.5 * 200.0)


def test_realized_carbon_cost_raises_on_missing_actual():
    actual_series = pd.Series(
        {pd.Timestamp("2026-01-01 00:00", tz="UTC"): np.nan}
    )
    with pytest.raises(ValueError):
        realized_carbon_cost(
            actual_series, pd.Timestamp("2026-01-01 00:00", tz="UTC"), duration_minutes=30.0
        )


# ----------------------------------------------------------------------
# Synthetic end-to-end fixture for the two structural properties below
# ----------------------------------------------------------------------
def _synthetic_ctx_and_jobs(seed=0, n_jobs=40):
    """
    Builds a small synthetic actual/forecast world and a batch of
    synthetic jobs, entirely in-memory (no real fetched/trained
    artifacts needed) -- fast, deterministic, and isolates these tests
    from whatever the currently-fetched carbon data happens to contain.

    Forecasts are made DELIBERATELY NOISY/WRONG relative to the actual
    series (a different, uncorrelated random draw) specifically so that
    forecast_greedy's choices can genuinely differ from oracle's --
    otherwise property 2 (oracle <= everything, always) would be
    trivially true even with a broken implementation.
    """
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2026-01-01", tz="UTC")
    n_periods = 24 * 2 * 5  # 5 days of 30-min periods
    timestamps = pd.date_range(start, periods=n_periods, freq="30min")

    actual_values = 100 + 60 * np.sin(2 * np.pi * np.arange(n_periods) / 48) \
        + rng.normal(0, 15, n_periods)
    actual_series = pd.Series(actual_values, index=timestamps)

    horizons = list(range(1, 25))
    forecast_tables = {}
    for h in horizons:
        # Deliberately noisy forecast: actual value h hours ahead, plus
        # independent noise -- close enough to be informative, wrong
        # enough that forecast_greedy will sometimes pick a worse slot
        # than oracle.
        shifted = actual_series.shift(-2 * h)
        noise = rng.normal(0, 25, n_periods)
        q50 = (shifted.values + noise)
        q50 = np.where(np.isnan(q50), actual_series.values, q50)  # fill tail
        q10 = q50 - 20
        q90 = q50 + 20
        forecast_tables[h] = pd.DataFrame(
            {"q10": q10, "q50": q50, "q90": q90}, index=timestamps
        )

    threshold_value = float(np.median(actual_values))
    ctx = SimContext(actual_series, forecast_tables, threshold_value)

    # Synthetic jobs: arrivals spread through the first 3 days, slack in
    # {6, 12, 24}h, duration a few multiples of 30 min (so periods divide
    # evenly is NOT guaranteed -- keep some fractional durations too).
    arrivals = start + pd.to_timedelta(
        rng.uniform(0, 3 * 24 * 60, n_jobs), unit="m"
    )
    slack_options = np.array([6, 12, 24])
    slack_hours = rng.choice(slack_options, size=n_jobs)
    durations = rng.uniform(20, 200, n_jobs)  # minutes, deliberately fractional

    jobs = pd.DataFrame({
        "job_id": np.arange(1, n_jobs + 1),
        "arrival_time": arrivals,
        "duration_minutes": durations,
        "slack_hours": slack_hours,
    })
    jobs["deadline"] = (
        jobs["arrival_time"]
        + pd.to_timedelta(jobs["duration_minutes"], unit="m")
        + pd.to_timedelta(jobs["slack_hours"], unit="h")
    )

    return ctx, jobs


# ----------------------------------------------------------------------
# 1. Deadline violations are genuinely 0
# ----------------------------------------------------------------------
def test_no_deadline_violations_for_any_scheduler():
    ctx, jobs = _synthetic_ctx_and_jobs(seed=1, n_jobs=60)

    for name, fn in SCHEDULERS.items():
        for _, job in jobs.iterrows():
            record = simulate_job(job, fn, ctx)
            assert record["end_time"] <= record["deadline"], (
                f"Deadline violation for scheduler={name}, job_id={job['job_id']}: "
                f"end_time={record['end_time']} > deadline={record['deadline']}"
            )


def test_decision_grid_never_exceeds_true_latest_start():
    """
    The forced boundary (last grid point) must always be <= arrival +
    slack_hours exactly (the true latest allowable start) -- this is
    the structural guarantee that makes zero deadline violations
    possible in the first place.
    """
    rng = np.random.default_rng(2)
    for _ in range(200):
        arrival = pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(
            minutes=float(rng.uniform(0, 10000))
        )
        slack_hours = int(rng.choice([6, 12, 24]))
        grid = build_decision_grid(arrival, slack_hours)
        true_latest_start = arrival + pd.Timedelta(hours=slack_hours)
        assert grid[-1] <= true_latest_start
        assert grid[0] >= arrival


# ----------------------------------------------------------------------
# 2. Oracle is a genuine per-job upper bound
# ----------------------------------------------------------------------
def test_oracle_cost_never_exceeds_any_other_scheduler_per_job():
    ctx, jobs = _synthetic_ctx_and_jobs(seed=3, n_jobs=50)

    oracle_fn = SCHEDULERS["oracle"]
    other_names = [n for n in SCHEDULERS if n != "oracle"]

    for _, job in jobs.iterrows():
        oracle_record = simulate_job(job, oracle_fn, ctx)
        oracle_cost = oracle_record["realized_carbon_cost"]

        for name in other_names:
            other_record = simulate_job(job, SCHEDULERS[name], ctx)
            other_cost = other_record["realized_carbon_cost"]
            assert oracle_cost <= other_cost + 1e-6, (
                f"Oracle upper-bound violated for job_id={job['job_id']} vs {name}: "
                f"oracle={oracle_cost:.3f} > {name}={other_cost:.3f}"
            )
