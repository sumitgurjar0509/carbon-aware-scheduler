"""
Discrete-Time Job Scheduling Simulator
=========================================
Part of: AI-Based Energy and Carbon Optimization in Cloud Computing Environments

Runs each of the 5 schedulers (schedulers.py) against the synthetic job
trace (workload_generator.py) and the real carbon intensity data,
producing per-job outcomes that evaluate.py turns into the thesis's
headline metrics.

======================================================================
KEY DESIGN DECISIONS (read before the viva)
======================================================================

1. ONLY jobs whose entire decision window falls inside the TEST split.
   calibration.py's chronological split reserves the final ~20% of the
   year as TEST, genuinely unseen by every trained model. A job whose
   arrival or decision window fell in the train/calibration period would
   let the "forecast" schedulers use models that had already seen that
   period during training/calibration -- not a fair or honest test. We
   recompute the SAME train/calibration/test boundaries used in
   calibration.py (via `calibration.three_way_split` applied to the
   full engineered feature set, independent of any specific forecast
   horizon -- see `get_canonical_split()`) and keep only jobs whose
   entire decision grid (arrival through the forced boundary) sits
   inside that test window. See the printed report for how many jobs
   this leaves.

2. THE DECISION GRID: hourly steps, snapped to the settlement grid.
   Forecasts only exist at whole-hour horizons (1..24h, from
   forecasting.py's design). A job's continuous arrival_time is
   therefore snapped to the settlement grid (ceil to the next 30-min
   boundary -- can't start before it arrives) as its first decision
   point t0; the true latest allowable start (arrival + slack_hours,
   exactly -- duration cancels out of `deadline - duration`) is snapped
   DOWN (floor to 30-min) as the safety boundary -- flooring can only
   move the boundary EARLIER, so it can never cause a deadline
   violation. From t0, decision points step forward in exact 1-hour
   increments (t0, t0+1h, t0+2h, ...) until the last one that does not
   exceed the floored boundary; that last point is the FORCED start.
   Every point on this grid is therefore both settlement-grid-aligned
   (valid for feature/actual lookups) and an exact whole-hour multiple
   apart (matching the forecast horizons exactly, 1, 2, 3, ... hours
   ahead of whichever point you're standing at) -- no fractional-hour
   horizon ever needs to be predicted.
   Trade-off, stated plainly: because BOTH ends are independently
   rounded (up at arrival, down at the deadline boundary), a job can
   lose up to ~1 hour of its nominal slack to grid discretization. This
   is a deliberate, minor conservatism (never a deadline violation risk)
   -- not modeled away, because pretending away discretization would be
   dishonest about what an hourly-forecast-driven scheduler can actually
   promise.
   ALL FIVE schedulers share this exact same per-job grid, so
   differences in outcomes reflect the scheduling POLICY, not a
   difference in available candidate start times.

3. CARBON COST = ACTUAL usage, integrated over the run, always.
   A scheduler may have decided using a forecast, but once a job is
   actually running, it consumes whatever the grid actually was during
   that window -- never the forecast. `realized_carbon_cost()` sums
   (overlap_minutes / 30) * actual_intensity over every settlement
   period the job's [start, start+duration) window overlaps, handling
   partial-period overlap at the tail exactly (duration is rarely an
   exact multiple of 30 minutes).

4. NO capacity constraints -- jobs are scheduled fully independently.
   This is a deliberate scope decision (see the task's own framing,
   citing the "stampede effect" literature, e.g. Hanafy et al.): many
   schedulers converging on the same attractive low-carbon window is a
   real phenomenon this simulator does NOT prevent or penalize. We do,
   however, compute an OBSERVATIONAL concurrency metric (total
   overlapping job-minutes per settlement period, across the whole test
   window) per scheduler, purely to let the thesis discuss whether a
   given policy bunches jobs together -- without enforcing any limit on
   it.

5. Deadline violations should be ~0 for every scheduler, by construction
   (see point 2 -- the forced boundary is always <= the true latest
   start). If evaluate.py ever reports a nonzero violation count, that
   indicates a BUG in this simulator, not a real scheduling trade-off --
   flagged loudly rather than quietly reported as a finding.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calibration  # noqa: E402
from data_config import check_model_label, existing_workload_path, get_data_tag  # noqa: E402
from forecasting import HORIZON_HOURS, create_features, get_feature_columns, load_clean_data  # noqa: E402
from schedulers import SCHEDULERS  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TABLES_DIR = PROJECT_ROOT / "results" / "tables"

NATIVE_PERIOD = pd.Timedelta(minutes=30)


# ----------------------------------------------------------------------
# Canonical split (matches calibration.py exactly, horizon-independent)
# ----------------------------------------------------------------------
def get_canonical_split(df: pd.DataFrame):
    """
    Re-derives the SAME chronological train/calibration/test boundaries
    calibration.py used, but applied once to the full engineered feature
    set rather than per-horizon (build_horizon_dataset trims a handful
    of extra rows off the tail depending on horizon -- at most 24h out
    of a ~73-day test window, immaterial here, and this way there is
    exactly ONE test-period definition shared by every horizon rather
    than 24 almost-identical ones).
    """
    df_feat = create_features(df)
    train_feat, calib_feat, test_feat = calibration.three_way_split(df_feat)
    return df_feat, train_feat, calib_feat, test_feat


# ----------------------------------------------------------------------
# Job filtering + per-job decision grid
# ----------------------------------------------------------------------
def build_decision_grid(arrival_time: pd.Timestamp, slack_hours: int) -> list:
    """See module docstring, point 2, for the full rationale."""
    t0 = arrival_time.ceil("30min")
    true_latest_start = arrival_time + pd.Timedelta(hours=int(slack_hours))
    decision_end = true_latest_start.floor("30min")

    h_max = int((decision_end - t0) / pd.Timedelta(hours=1))  # truncates toward 0
    h_max = max(h_max, 0)
    return [t0 + pd.Timedelta(hours=k) for k in range(h_max + 1)]


def filter_jobs_to_test_period(workload_df: pd.DataFrame, test_start: pd.Timestamp,
                                test_end: pd.Timestamp) -> pd.DataFrame:
    """
    Keeps only jobs whose ENTIRE decision grid (first to last point)
    falls within [test_start, test_end] -- the only period genuinely
    out-of-sample for every trained model.
    """
    kept_rows = []
    for _, job in workload_df.iterrows():
        grid = build_decision_grid(job["arrival_time"], job["slack_hours"])
        if grid[0] >= test_start and grid[-1] <= test_end:
            kept_rows.append(job)
    filtered = pd.DataFrame(kept_rows).reset_index(drop=True)
    print(f"Jobs within test period: {len(filtered)} / {len(workload_df)} "
          f"({len(filtered) / len(workload_df):.1%})")
    print(filtered["slack_hours"].value_counts().sort_index().rename("jobs_by_slack_bucket"))
    return filtered


def load_workload(tag: str | None = None) -> pd.DataFrame:
    """Reads the workload trace that belongs to dataset `tag` (default: CAS_DATA_TAG)."""
    path = existing_workload_path(tag)
    print(f"Loaded {path.name}")
    return pd.read_csv(path, parse_dates=["arrival_time", "deadline"])


# ----------------------------------------------------------------------
# Forecast lookup tables (precomputed once, vectorized, reused by every job)
# ----------------------------------------------------------------------
def build_forecast_tables(df_feat: pd.DataFrame, test_start: pd.Timestamp,
                           test_end: pd.Timestamp, horizons=HORIZON_HOURS) -> dict:
    """
    For each horizon h, one batched call to calibration.predict_calibrated
    over every test-period feature row -- returns a DataFrame indexed by
    ORIGIN timestamp with columns q10/q50/q90, i.e. table[h].loc[t] is
    "the CQR-calibrated forecast made at t for time t+h hours". Batched
    per horizon (not per job-step) purely for speed: a single vectorized
    LightGBM predict() over ~3500 rows is far cheaper than tens of
    thousands of one-row predict() calls with the model reloaded from
    disk each time.
    """
    # Refuse models trained on another dataset, or not yet calibrated.
    check_model_label(calibration.MODELS_DIR, get_data_tag())

    feature_cols = get_feature_columns(df_feat)
    mask = (df_feat["timestamp"] >= test_start) & (df_feat["timestamp"] <= test_end)
    test_feat = df_feat.loc[mask].reset_index(drop=True)

    tables = {}
    for h in horizons:
        preds = calibration.predict_calibrated(h, test_feat[feature_cols])
        # NOTE: assigning test_feat["timestamp"].values here would silently
        # strip tz-awareness (numpy datetime64 has no tz), causing every
        # later .loc[timestamp] lookup with a tz-aware key to raise KeyError.
        preds.index = pd.DatetimeIndex(test_feat["timestamp"])
        tables[h] = preds
    return tables


# ----------------------------------------------------------------------
# Carbon-cost integration (point 3) and the shared simulation context
# ----------------------------------------------------------------------
def _period_overlaps(start_time: pd.Timestamp, end_time: pd.Timestamp):
    """
    Yields (period_start, overlap_minutes) for every native 30-min
    settlement period overlapping [start_time, end_time). General
    interval-overlap logic -- correct regardless of whether start_time
    happens to land exactly on a period boundary (it always does here,
    since every decision time is 30-min-grid-aligned, but end_time
    rarely is, since duration_minutes is fractional).
    """
    first_period_start = start_time.floor("30min")
    p_start = first_period_start
    while p_start < end_time:
        p_end = p_start + NATIVE_PERIOD
        overlap_start = max(p_start, start_time)
        overlap_end = min(p_end, end_time)
        overlap_minutes = (overlap_end - overlap_start).total_seconds() / 60.0
        if overlap_minutes > 0:
            yield p_start, overlap_minutes
        p_start = p_end


def realized_carbon_cost(actual_series: pd.Series, started_time: pd.Timestamp,
                          duration_minutes: float) -> float:
    """sum of (overlap_minutes / 30) * actual_intensity -- see point 3."""
    end_time = started_time + pd.Timedelta(minutes=duration_minutes)
    cost = 0.0
    for p_start, overlap_minutes in _period_overlaps(started_time, end_time):
        if p_start not in actual_series.index:
            raise ValueError(f"No actual carbon intensity available for period {p_start} "
                              f"-- cannot compute realized cost (never fabricate this).")
        intensity = actual_series.loc[p_start]
        if pd.isna(intensity):
            raise ValueError(f"Actual carbon intensity at {p_start} is missing (NaN) "
                              f"-- cannot compute realized cost (never fabricate this).")
        cost += (overlap_minutes / 30.0) * intensity
    return cost


class SimContext:
    """Bundles what every scheduler decision function needs (see schedulers.py)."""

    def __init__(self, actual_series: pd.Series, forecast_tables: dict, threshold_value: float):
        self.actual_series = actual_series
        self.forecast_tables = forecast_tables
        self.threshold_value = threshold_value

    def actual_at(self, t: pd.Timestamp) -> float:
        if t not in self.actual_series.index or pd.isna(self.actual_series.loc[t]):
            raise ValueError(f"Missing actual carbon intensity at {t}")
        return float(self.actual_series.loc[t])

    def realized_cost_at(self, start_time: pd.Timestamp, job: pd.Series) -> float:
        """
        The TRUE, fully-integrated realized carbon cost the job would
        incur if started at `start_time` -- i.e. exactly the same
        function used to score the job's actual outcome after the fact
        (see `realized_carbon_cost`). Only the oracle scheduler uses
        this (see schedulers.py's oracle() docstring for why): it is
        the one scheduler allowed to know this in advance, since it
        represents perfect knowledge of the whole future, not a
        deployable forecast.
        """
        return realized_carbon_cost(self.actual_series, start_time, job["duration_minutes"])

    def forecast(self, origin_t: pd.Timestamp, horizon_hours: int) -> dict:
        row = self.forecast_tables[horizon_hours].loc[origin_t]
        return {"q10": float(row["q10"]), "q50": float(row["q50"]), "q90": float(row["q90"])}


# ----------------------------------------------------------------------
# Per-job simulation
# ----------------------------------------------------------------------
def simulate_job(job: pd.Series, scheduler_fn, ctx: SimContext) -> dict:
    grid = build_decision_grid(job["arrival_time"], job["slack_hours"])

    started_time = None
    was_forced = False
    for i, t in enumerate(grid):
        remaining_times = grid[i + 1:]
        if i == len(grid) - 1:
            started_time = t
            was_forced = True
            break
        if scheduler_fn(t, job, remaining_times, ctx):
            started_time = t
            was_forced = False
            break

    realized_cost = realized_carbon_cost(ctx.actual_series, started_time, job["duration_minutes"])
    delay = (started_time - job["arrival_time"]).total_seconds() / 3600.0  # hours
    true_deadline = job["deadline"]
    end_time = started_time + pd.Timedelta(minutes=job["duration_minutes"])

    return {
        "job_id": job["job_id"],
        "slack_hours": job["slack_hours"],
        "arrival_time": job["arrival_time"],
        "started_time": started_time,
        "end_time": end_time,
        "deadline": true_deadline,
        "delay_hours": delay,
        "realized_carbon_cost": realized_cost,
        "was_forced_start": was_forced,
        "deadline_violation": end_time > true_deadline,
    }


def simulate_all(jobs_df: pd.DataFrame, scheduler_name: str, ctx: SimContext) -> pd.DataFrame:
    scheduler_fn = SCHEDULERS[scheduler_name]
    records = [simulate_job(job, scheduler_fn, ctx) for _, job in jobs_df.iterrows()]
    df = pd.DataFrame(records)
    df["scheduler"] = scheduler_name
    return df


# ----------------------------------------------------------------------
# Observational concurrency metric (point 4) -- no capacity limit enforced
# ----------------------------------------------------------------------
def compute_concurrency(results_df: pd.DataFrame, test_start: pd.Timestamp,
                         test_end: pd.Timestamp) -> pd.Series:
    """
    Total overlapping job-MINUTES per native 30-min settlement period,
    summed across all jobs in `results_df` (one scheduler's run) --
    purely observational, never used to constrain scheduling decisions.
    """
    period_index = pd.date_range(test_start.floor("30min"), test_end.ceil("30min"),
                                  freq="30min")
    concurrency = pd.Series(0.0, index=period_index)
    for _, row in results_df.iterrows():
        for p_start, overlap_minutes in _period_overlaps(row["started_time"], row["end_time"]):
            if p_start in concurrency.index:
                concurrency.loc[p_start] += overlap_minutes
    return concurrency


# ----------------------------------------------------------------------
# Full pipeline
# ----------------------------------------------------------------------
def run_simulation():
    df = load_clean_data()
    df_feat, train_feat, calib_feat, test_feat = get_canonical_split(df)

    test_start, test_end = test_feat["timestamp"].min(), test_feat["timestamp"].max()
    threshold_value = train_feat["carbon_intensity"].median()
    print(f"Test period: {test_start} -> {test_end} ({len(test_feat)} settlement periods)")
    print(f"Threshold scheduler value (train-split median): {threshold_value:.1f} gCO2/kWh")

    workload_df = load_workload()
    jobs = filter_jobs_to_test_period(workload_df, test_start, test_end)

    print("\nBuilding CQR-calibrated forecast lookup tables (1h-24h horizons)...")
    forecast_tables = build_forecast_tables(df_feat, test_start, test_end)

    actual_series = df.set_index("timestamp")["carbon_intensity"]
    ctx = SimContext(actual_series, forecast_tables, threshold_value)

    all_results = {}
    concurrency_by_scheduler = {}
    for name in SCHEDULERS:
        print(f"\nSimulating scheduler: {name} ...")
        results_df = simulate_all(jobs, name, ctx)
        all_results[name] = results_df
        concurrency_by_scheduler[name] = compute_concurrency(results_df, test_start, test_end)

        n_violations = results_df["deadline_violation"].sum()
        if n_violations > 0:
            print(f"  !! BUG WARNING: {n_violations} deadline violation(s) for {name} -- "
                  f"this should be structurally impossible. Investigate before trusting results.")
        print(f"  jobs={len(results_df)}  mean_delay={results_df['delay_hours'].mean():.2f}h  "
              f"forced_starts={results_df['was_forced_start'].sum()}  "
              f"total_carbon={results_df['realized_carbon_cost'].sum():.1f} gCO2")

    combined = pd.concat(all_results.values(), ignore_index=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TABLES_DIR / "simulation_results.csv"
    combined.to_csv(out_path, index=False)
    print(f"\nSaved per-job simulation results to {out_path}")

    return combined, concurrency_by_scheduler, jobs, test_start, test_end


if __name__ == "__main__":
    run_simulation()
