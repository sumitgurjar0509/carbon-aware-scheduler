"""
Carbon Intensity Data Pipeline
==============================
Part of: AI-Based Energy and Carbon Optimization in Cloud Computing Environments

WHAT THIS DOES
--------------
Downloads GB national carbon intensity history from the UK Carbon
Intensity API (https://api.carbonintensity.org.uk/), caches it locally
as raw JSON-derived CSV so repeated runs don't re-hit the network, then
produces a cleaned, gap-documented 30-minute series for forecasting.

WHY THE UK CARBON INTENSITY API
--------------------------------
- Free, no API key, half-hourly settlement-period granularity already
  matches the "actual" grid-mix data used elsewhere in GB carbon-aware
  computing literature (e.g. the WattTime / Electricity Maps style of
  study).
- It reports both `forecast` and `actual` intensity per settlement
  period. We use `actual` as ground truth for the simulator and as the
  training target; `forecast` is the *grid operator's own* forecast,
  which is a legitimate baseline to compare our LightGBM forecaster
  against later if needed (out of current SCOPE, noted for later).

API SHAPE (checked live on 2026-09-23)
---------------------------------------
GET /intensity/{from}/{to}   (ISO8601, 'Z' suffix, max range = 14 days)
-> {"data": [{"from": "...", "to": "...",
              "intensity": {"forecast": N, "actual": N, "index": "low"}}, ...]}

`actual` can be null for very recent settlement periods (not yet
settled) or for gaps in National Grid ESO's own reporting. We keep
these as NaN -- we do NOT invent a value.

CACHING STRATEGY
-----------------
Each 14-day chunk is fetched once and cached as its own raw CSV under
data/raw/. Re-running the pipeline with the same date range reads from
disk. Delete files under data/raw/ to force a re-download.
"""

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

API_BASE = "https://api.carbonintensity.org.uk"
CHUNK_DAYS = 14  # API hard limit per request
NATIVE_RESOLUTION = timedelta(minutes=30)  # settlement period length

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def _fmt(dt: datetime) -> str:
    """API wants ISO8601 with a literal 'Z', no microseconds."""
    return dt.strftime("%Y-%m-%dT%H:%MZ")


def _chunk_ranges(start: datetime, end: datetime, chunk: timedelta):
    """
    Yield (chunk_start, chunk_end) pairs covering [start, end), anchored
    to `end` and walking backwards. This is deliberate: `end` is always
    "now" (rounded to the hour) and `start` is whatever window we ask
    for. Anchoring from `end` means the chunk boundaries near "now" are
    identical across runs even as `start` moves further back (e.g. when
    extending the fetch window from 180 to 365 days) -- so a widened
    window reuses the already-cached recent chunks and only downloads
    the newly-added older chunks, instead of re-fetching everything
    under a shifted grid.
    """
    boundaries = []
    cur_end = end
    while cur_end > start:
        cur_start = max(start, cur_end - chunk)
        boundaries.append((cur_start, cur_end))
        cur_end = cur_start
    return reversed(boundaries)


def fetch_carbon_intensity(
    start: datetime,
    end: datetime,
    raw_dir: Path = RAW_DIR,
    session: requests.Session | None = None,
    max_retries: int = 3,
) -> pd.DataFrame:
    """
    Fetch GB national carbon intensity for [start, end) in 14-day chunks,
    caching each chunk to disk. Returns the raw, un-cleaned long-form
    DataFrame with columns: from, to, forecast, actual, index.

    Both `start` and `end` must be timezone-aware UTC datetimes.
    """
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("start/end must be timezone-aware (use timezone.utc)")

    raw_dir.mkdir(parents=True, exist_ok=True)
    session = session or requests.Session()

    frames = []
    for chunk_start, chunk_end in _chunk_ranges(start, end, timedelta(days=CHUNK_DAYS)):
        cache_name = f"{chunk_start.strftime('%Y%m%d')}_{chunk_end.strftime('%Y%m%d')}.csv"
        cache_path = raw_dir / cache_name

        if cache_path.exists():
            frames.append(pd.read_csv(cache_path, parse_dates=["from", "to"]))
            continue

        url = f"{API_BASE}/intensity/{_fmt(chunk_start)}/{_fmt(chunk_end)}"
        last_err = None
        for attempt in range(1, max_retries + 1):
            try:
                resp = session.get(url, timeout=30)
                resp.raise_for_status()
                payload = resp.json()
                break
            except (requests.RequestException, ValueError) as e:
                last_err = e
                if attempt < max_retries:
                    time.sleep(2 * attempt)  # simple backoff
        else:
            raise RuntimeError(
                f"Failed to fetch {url} after {max_retries} attempts: {last_err}. "
                "Not substituting fabricated data -- fix connectivity and retry."
            )

        records = payload.get("data", [])
        if not records:
            print(f"WARNING: API returned zero records for {chunk_start} -> {chunk_end}")

        chunk_df = pd.DataFrame(
            [
                {
                    "from": r["from"],
                    "to": r["to"],
                    "forecast": r["intensity"]["forecast"],
                    "actual": r["intensity"]["actual"],
                    "index": r["intensity"]["index"],
                }
                for r in records
            ]
        )
        if not chunk_df.empty:
            chunk_df["from"] = pd.to_datetime(chunk_df["from"], utc=True)
            chunk_df["to"] = pd.to_datetime(chunk_df["to"], utc=True)

        chunk_df.to_csv(cache_path, index=False)
        frames.append(chunk_df)
        time.sleep(0.2)  # be polite to a free public API

    if not frames:
        raise RuntimeError(f"No data fetched for range {start} -> {end}")

    raw = pd.concat(frames, ignore_index=True)
    raw = raw.drop_duplicates(subset=["from"]).sort_values("from").reset_index(drop=True)
    return raw


def clean_and_resample(
    raw: pd.DataFrame,
    max_gap_periods: int = 4,
) -> tuple[pd.DataFrame, dict]:
    """
    Turns the raw API dump into a clean, regularly-spaced 30-minute
    series indexed by the settlement-period start time (`from`).

    MISSING DATA POLICY (explicit, not silent):
    - The API's native resolution IS 30 minutes, so "resampling" here
      mostly means: reindex onto a complete 30-min grid so that missing
      settlement periods become visible as NaN rows, rather than
      silently shrinking the series.
    - `actual` intensity can be null even for periods that ARE present
      in the API response (not yet settled by National Grid ESO). We
      treat null-`actual` exactly like a missing row.
    - Short gaps (<= max_gap_periods, default 4 = 2 hours) are filled
      by linear interpolation, since carbon intensity is smooth over
      such short windows and this is a defensible, disclosed choice.
    - Long gaps (> max_gap_periods) are LEFT AS NaN. We do not
      extrapolate or interpolate across long gaps -- doing so would
      fabricate grid behaviour we have no basis for. Callers (the
      forecaster, the simulator) must handle NaN explicitly (e.g. drop
      those rows, or exclude that period from an experiment).

    Returns (clean_df, gap_report) where gap_report documents exactly
    what was found and what was done, for the thesis writeup.
    """
    if raw.empty:
        raise ValueError("clean_and_resample received an empty raw DataFrame")

    df = raw[["from", "actual", "forecast"]].copy()
    df = df.rename(columns={"from": "timestamp", "actual": "carbon_intensity"})
    df = df.drop_duplicates(subset="timestamp").set_index("timestamp").sort_index()

    full_index = pd.date_range(
        start=df.index.min(), end=df.index.max(), freq="30min", tz=df.index.tz
    )
    n_expected = len(full_index)
    n_present_rows = len(df)

    df = df.reindex(full_index)
    df.index.name = "timestamp"

    missing_mask = df["carbon_intensity"].isna()
    n_missing_total = int(missing_mask.sum())

    # Identify contiguous missing runs to classify short vs. long gaps.
    gap_id = (missing_mask != missing_mask.shift()).cumsum()
    gap_groups = df[missing_mask].groupby(gap_id[missing_mask])
    gap_lengths = gap_groups.size()

    short_gap_ids = gap_lengths[gap_lengths <= max_gap_periods].index
    long_gap_ids = gap_lengths[gap_lengths > max_gap_periods].index

    short_gap_index = df.index[missing_mask & gap_id.isin(short_gap_ids)]
    long_gap_index = df.index[missing_mask & gap_id.isin(long_gap_ids)]

    df["was_interpolated"] = False
    if len(short_gap_index) > 0:
        interpolated = df["carbon_intensity"].interpolate(
            method="linear", limit=max_gap_periods, limit_area="inside"
        )
        df.loc[short_gap_index, "carbon_intensity"] = interpolated.loc[short_gap_index]
        # limit_area="inside" deliberately refuses to fill a gap that
        # touches the start/end of the series (e.g. the most recent
        # settlement period, which the API often returns with
        # actual=null because it hasn't been settled yet). Only mark
        # rows that were ACTUALLY filled -- checking the classification
        # alone would mislabel those still-NaN edge rows as interpolated.
        actually_filled = short_gap_index[df.loc[short_gap_index, "carbon_intensity"].notna()]
        df.loc[actually_filled, "was_interpolated"] = True
        short_gap_index = actually_filled

    still_missing_mask = df["carbon_intensity"].isna()
    n_unfillable_edge_gaps = int(still_missing_mask.sum()) - len(long_gap_index)

    gap_report = {
        "range_start": str(full_index.min()),
        "range_end": str(full_index.max()),
        "expected_periods": n_expected,
        "periods_returned_by_api": n_present_rows,
        "missing_periods_total": n_missing_total,
        "missing_pct": round(100 * n_missing_total / n_expected, 3) if n_expected else 0.0,
        "short_gaps_interpolated": len(short_gap_index),
        "long_gaps_left_as_nan": len(long_gap_index),
        "num_long_gap_runs": len(long_gap_ids),
        "edge_gaps_left_as_nan": n_unfillable_edge_gaps,
        "max_gap_periods_threshold": max_gap_periods,
    }

    return df.reset_index(), gap_report


def load_or_build_dataset(
    start: datetime,
    end: datetime,
    processed_dir: Path = PROCESSED_DIR,
    force_rebuild: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """
    High-level entry point: fetch (cached) raw data, clean it, cache the
    cleaned result too, and return (clean_df, gap_report).
    """
    processed_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}"
    clean_path = processed_dir / f"clean_{tag}.csv"
    report_path = processed_dir / f"gap_report_{tag}.csv"

    if clean_path.exists() and report_path.exists() and not force_rebuild:
        clean_df = pd.read_csv(clean_path, parse_dates=["timestamp"])
        gap_report = pd.read_csv(report_path).iloc[0].to_dict()
        return clean_df, gap_report

    raw = fetch_carbon_intensity(start, end)
    clean_df, gap_report = clean_and_resample(raw)

    clean_df.to_csv(clean_path, index=False)
    pd.DataFrame([gap_report]).to_csv(report_path, index=False)
    return clean_df, gap_report


if __name__ == "__main__":
    # Window = 365 days up to now, UTC.
    #
    # The API's history actually goes back to at least 2018 (checked
    # live: /intensity/2018-01-01T00:00Z/... returns real data), so we
    # deliberately did NOT fetch "as much as the API has" -- we capped
    # it at one full year. Two reasons:
    #   1. Concept drift: GB grid carbon intensity has fallen sharply
    #      over the years as coal was phased out and renewables grew.
    #      2018-2020 intensity levels are systematically higher than
    #      today's and are not representative of the CURRENT grid the
    #      scheduler will actually run against -- training on several
    #      years of that data would teach the model a decarbonization
    #      trend baked into the past rather than today's diurnal/
    #      seasonal pattern, which is what we actually want it to learn.
    #   2. One year is the minimum needed to see a full seasonal cycle
    #      (summer solar peak vs winter) exactly once, which is the
    #      stated goal here, without inflating fetch/train time.
    # This is a judgment call, not a hard constraint -- worth defending
    # at the viva as "one full seasonal cycle of *current-regime* grid
    # behaviour" rather than "maximum available history".
    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=365)

    print(f"Building dataset for {start} -> {end} ...")
    clean_df, gap_report = load_or_build_dataset(start, end)

    print("\n=== Gap report ===")
    for k, v in gap_report.items():
        print(f"{k}: {v}")

    print("\n=== Basic stats (carbon_intensity, gCO2/kWh) ===")
    print(clean_df["carbon_intensity"].describe())
