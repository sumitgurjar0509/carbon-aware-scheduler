"""
Synthetic Delay-Tolerant Job Workload Generator
=================================================
Part of: AI-Based Energy and Carbon Optimization in Cloud Computing Environments

WHAT THIS DOES
--------------
Generates a synthetic trace of delay-tolerant BATCH jobs (think: nightly
ETL, batch inference, non-urgent model training/re-training) spread
across the same period as the fetched carbon intensity data. Each job
has an arrival time, a duration, and a deadline slack bucket; its
deadline is simply arrival + duration + slack. This is the workload the
simulator (next step) will schedule against the carbon intensity trace.

ASSUMPTIONS (documented here so they're defensible at the viva)
------------------------------------------------------------------
1. ARRIVALS -- homogeneous Poisson process.
   Job arrivals are modeled as a Poisson process (i.e. exponentially
   distributed inter-arrival times), the standard assumption for
   independent job/request arrivals in queueing and cloud-workload
   literature, rather than a fixed schedule or uniform random
   timestamps. This gives realistic clustering/gaps instead of an
   artificially even spacing.

2. DURATION -- lognormal, capped at 4 hours.
   Real cloud batch-job runtimes are strictly positive and heavily
   right-skewed: most jobs are short, a smaller number run much longer
   (this is the standard finding in published cluster-trace studies,
   e.g. Google/Azure workload analyses -- many short tasks, a long
   tail). A lognormal distribution is the standard choice for modeling
   this shape (as opposed to e.g. uniform or normal, which would put
   unrealistic mass on very short or negative durations).
   Parameters: median duration = 45 minutes (mu = ln(45)), sigma = 0.8
   (moderate spread -> a noticeable but not extreme long tail).
   We CAP duration at 240 minutes (4 hours): these jobs are meant to be
   meaningfully shorter than the shortest deadline-slack bucket (6h),
   so that "shifting" the job within its slack window is a real choice
   about WHEN a short job runs, not dominated by the job's own runtime
   eating most of the slack. Uncapped lognormal tails would occasionally
   produce jobs longer than their own slack bucket, which doesn't
   represent a meaningfully "delay-tolerant, shiftable" job.

3. DEADLINE SLACK -- {6h, 12h, 24h}, stratified to an even 1/3 split.
   The thesis explicitly compares carbon savings AT each slack level
   (RQ1), so we deliberately stratify job COUNTS evenly across the
   three buckets (via a shuffled round-robin assignment) rather than
   leaving the split to chance -- this guarantees comparable sample
   sizes per bucket regardless of random seed, which matters for a fair
   later comparison.

4. DATA-COVERAGE CONSTRAINT on arrival times.
   Every job's [arrival, deadline] window must fall entirely within the
   fetched carbon intensity range, since the simulator needs real
   ground-truth intensity data for every candidate execution slot. We
   therefore only draw arrivals from
       [data_start, data_end - max_slack - max_duration]
   (24h + 4h = 28h trimmed off the end of ~365 days -- negligible loss
   of usable arrival window, and avoids ever needing to guess at
   intensity data that doesn't exist).

5. JOB COUNT -- 1000 jobs by default.
   Falls within the "a few hundred to a couple thousand" range asked
   for; ~2.7 jobs/day on average over the year, which is plausible for
   a batch-workload trace and gives each slack bucket (~333 jobs) a
   reasonable sample size for later aggregate comparisons.

Everything here is a documented, deliberately-chosen scope decision --
none of it is fabricated data (it's synthetic BY DESIGN, since we don't
have a real cluster trace, and the point is to test scheduling logic
under a controlled, explainable workload).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_config import gap_report_path, get_data_tag, workload_path  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

SLACK_HOURS_OPTIONS = (6, 12, 24)
DURATION_MEDIAN_MINUTES = 45
DURATION_SIGMA = 0.8
DURATION_CAP_MINUTES = 240
N_JOBS = 1000
RANDOM_SEED = 42


def _load_data_range(tag: str, processed_dir: Path = PROCESSED_DIR):
    """Reads the carbon-data date range from the gap report of dataset `tag`."""
    report = pd.read_csv(gap_report_path(tag, processed_dir)).iloc[0]
    return pd.Timestamp(report["range_start"]), pd.Timestamp(report["range_end"])


def generate_workload(
    data_start: pd.Timestamp,
    data_end: pd.Timestamp,
    n_jobs: int = N_JOBS,
    slack_options=SLACK_HOURS_OPTIONS,
    duration_median_minutes: float = DURATION_MEDIAN_MINUTES,
    duration_sigma: float = DURATION_SIGMA,
    duration_cap_minutes: float = DURATION_CAP_MINUTES,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Builds the synthetic job trace. See module docstring for the
    rationale behind each modeling choice.
    """
    rng = np.random.default_rng(seed)

    max_slack_hours = max(slack_options)
    arrival_window_end = data_end - pd.Timedelta(hours=max_slack_hours) \
        - pd.Timedelta(minutes=duration_cap_minutes)
    arrival_window_seconds = (arrival_window_end - data_start).total_seconds()
    if arrival_window_seconds <= 0:
        raise ValueError("Carbon data range too short for the requested slack/duration caps.")

    # --- Arrivals: homogeneous Poisson process over the usable window ---
    mean_rate_per_second = n_jobs / arrival_window_seconds
    inter_arrival_seconds = rng.exponential(1 / mean_rate_per_second, size=n_jobs * 2)
    arrival_seconds = np.cumsum(inter_arrival_seconds)
    arrival_seconds = arrival_seconds[arrival_seconds < arrival_window_seconds][:n_jobs]
    # Poisson arrivals can occasionally fall short of n_jobs within the
    # window on an unlucky draw; top up with uniform draws if needed
    # rather than silently returning fewer jobs than requested.
    if len(arrival_seconds) < n_jobs:
        extra = rng.uniform(0, arrival_window_seconds, size=n_jobs - len(arrival_seconds))
        arrival_seconds = np.concatenate([arrival_seconds, extra])
    arrival_seconds = np.sort(arrival_seconds[:n_jobs])
    arrival_times = data_start + pd.to_timedelta(arrival_seconds, unit="s")

    # --- Duration: lognormal, capped ---
    mu = np.log(duration_median_minutes)
    duration_minutes = rng.lognormal(mean=mu, sigma=duration_sigma, size=n_jobs)
    duration_minutes = np.minimum(duration_minutes, duration_cap_minutes)

    # --- Slack: stratified even split via shuffled round-robin ---
    base_labels = np.array(slack_options)
    reps = int(np.ceil(n_jobs / len(base_labels)))
    slack_hours = np.tile(base_labels, reps)[:n_jobs]
    rng.shuffle(slack_hours)

    df = pd.DataFrame({
        "job_id": np.arange(1, n_jobs + 1),
        "arrival_time": arrival_times,
        "duration_minutes": np.round(duration_minutes, 1),
        "slack_hours": slack_hours,
    })
    df["deadline"] = (
        df["arrival_time"]
        + pd.to_timedelta(df["duration_minutes"], unit="m")
        + pd.to_timedelta(df["slack_hours"], unit="h")
    )

    assert (df["deadline"] <= data_end).all(), \
        "Bug: a job's deadline falls outside the available carbon data range."

    return df.sort_values("arrival_time").reset_index(drop=True)


def summarize_workload(df: pd.DataFrame):
    print(f"Total jobs: {len(df)}")
    print(f"Trace span: {df['arrival_time'].min()} -> {df['deadline'].max()}")

    print("\n=== Duration distribution (minutes) ===")
    print(df["duration_minutes"].describe().round(1))

    print("\n=== Slack bucket breakdown ===")
    counts = df["slack_hours"].value_counts().sort_index()
    pct = (counts / len(df) * 100).round(1)
    print(pd.DataFrame({"count": counts, "pct": pct}))

    print("\n=== Arrivals per day (should look Poisson-ish, not perfectly flat) ===")
    per_day = df.set_index("arrival_time").resample("D").size()
    print(per_day.describe().round(2))


if __name__ == "__main__":
    tag = get_data_tag()
    data_start, data_end = _load_data_range(tag)
    print(f"Carbon data range: {data_start} -> {data_end}")

    workload = generate_workload(data_start, data_end)

    out_path = workload_path(tag)
    workload.to_csv(out_path, index=False)
    print(f"\nSaved {len(workload)} jobs to {out_path}")

    summarize_workload(workload)
