# Carbon-Aware Scheduler

M.Tech thesis project: *AI-Based Energy and Carbon Optimization in Cloud
Computing Environments*.

Research question: for delay-tolerant cloud workloads, how much carbon
does temporal shifting save at different deadline-slack levels, how
much of an oracle scheduler's savings can a realistic forecaster
capture, and does an uncertainty-aware (quantile-forecast) scheduler
reduce deadline violations versus a naive point-forecast scheduler,
without giving up carbon savings?

## Scope (current)

- **Grid**: GB national carbon intensity, via the
  [UK Carbon Intensity API](https://api.carbonintensity.org.uk/) (free,
  no key, 30-minute settlement periods, `actual` + `forecast` fields).
- **Workload**: synthetic jobs with deadline slack in {6h, 12h, 24h}.
- **Forecasters**: persistence, seasonal-naive, LightGBM point,
  LightGBM quantile (q10/q50/q90).
- **Schedulers**: run-immediately, threshold-based, forecast-greedy,
  oracle, and the proposed uncertainty-aware scheduler.
- **Metrics**: % carbon saved, mean delay, p95 delay, deadline
  violation rate, peak-load increase.

Anything beyond this list is out of scope until agreed on explicitly.

## Project layout

```
data/
  raw/          cached raw API responses, one CSV per 14-day chunk
  processed/    cleaned 30-min series + gap reports
src/
  data_config.py        data window, dataset tag and model label rules
  data_pipeline.py      fetch / cache / clean carbon intensity data
  eda.py                exploratory plots (overview + diurnal + monthly)
  forecasting.py        multi-horizon baselines + LightGBM quantile forecasting
  calibration.py        CQR calibration of the quantile intervals (see below)
  workload_generator.py synthetic delay-tolerant job trace
  simulator.py          discrete-time job scheduling simulator
  schedulers.py         the 5 scheduler strategies (+ parameterized uncertainty_aware)
  evaluate.py           metrics computation + results table/plot
  risk_sweep.py         risk_weight frontier for uncertainty_aware (see below)
notebooks/          exploratory analysis
scripts/
  compare_tables.py     compare regenerated tables with the reference copy
results/
  plots/, tables/, tables_committed/, models/
tests/              sanity tests per module
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

**Note on pinned versions**: `numpy` and `lightgbm` are pinned to
specific versions (not just floors) because their newest builds are
broken on the machine this was developed on: `lightgbm==4.7.0` crashes
with an access-violation `OSError` inside its C library when building a
`Dataset` (reproduced with a bare numpy array, unrelated to our code),
and `numpy>=2.0` paired with `lightgbm==4.3.0` (the version that avoids
that crash) hits a separate `ValueError` from LightGBM's older
`copy=False` usage against numpy 2.0's stricter copy semantics.
`numpy==1.26.4` + `lightgbm==4.3.0` is the combination that actually
works here. `scikit-learn` is similarly pinned to `1.4.2` because
`>=1.5`'s compiled `sparsefuncs_fast` extension gets blocked outright by
this machine's Windows Application Control policy. If you're on a
different machine and hit none of these, the pins can likely be
loosened — but verify with `pytest tests/ -v` and a full pipeline run
before trusting a relaxed version.

## Reproducing the results (data window and dataset tag)

Every script that reads processed data needs to be told **which dataset**
to use, through the `CAS_DATA_TAG` environment variable. A tag is the
date part of a file name: `data/processed/clean_<tag>.csv`. If the
variable is not set, the script stops and lists the available tags. It
never picks "the newest file".

The committed results use tag `20250923_20260923` (2025-09-23 03:30 to
2026-09-23 03:30 UTC, 17,521 periods, 0 missing). In PowerShell:

```powershell
$env:CAS_DATA_TAG = "20250923_20260923"
python src/eda.py
python src/forecasting.py
python src/calibration.py          # must run before any simulation
python src/workload_generator.py   # writes workload_trace_<tag>.csv (seed 42)
python src/evaluate.py
python src/risk_sweep.py
python src/cvar_sweep.py
python scripts/compare_tables.py
```

These commands **overwrite** `results/tables/`, `results/plots/` and
`results/models/`. They do not use the network.

- `scripts/compare_tables.py` compares every CSV in `results/tables/`
  with the reference copy in `results/tables_committed/` (relative
  tolerance 1e-6, absolute 1e-8). The `cvar_*` tables are reported as
  `NO_BASELINE` because no earlier committed version exists.
- Training writes `results/models/quantile/data_tag.json`. The simulator
  refuses models trained on another tag, or not yet calibrated.

### Fetching a different window

`data_pipeline.py` fetches the last 365 days by default. To fix the
window, set one or both of these (ISO 8601, UTC if no timezone given):

```powershell
$env:CAS_START = "2025-09-23T03:00Z"
$env:CAS_END   = "2026-09-23T03:00Z"
python src/data_pipeline.py
```

With only `CAS_END`, the start is 365 days earlier. With only
`CAS_START`, the end is the current hour. Remove a variable with
`Remove-Item Env:CAS_END`.

**Limitation:** cache file names and tags contain dates only. Two windows
that differ only in the hour share the same cached files, so the second
run silently reuses the first one's data.

### Software versions

`results/environment_m1.txt` holds the exact package versions used for
the M1 reproduction (Python 3.11.9, Windows 11). It was created with:

```bash
pip freeze > results/environment_m1.txt
```

## Data pipeline

```bash
python src/data_pipeline.py   # fetch (cached) + clean last 365 days
python src/eda.py             # basic stats + overview/diurnal/monthly plots
```

### Design choices (for the viva)

- **Why 30-min resolution, no further resampling**: this is the API's
  native settlement-period granularity, so "cleaning" means *reindexing
  onto a complete grid*, not resampling/aggregating.
- **Missing data is never silently filled.** `clean_and_resample()` in
  `data_pipeline.py`:
  - Short gaps (default: ≤4 periods = 2 hours) are linearly
    interpolated (defensible over such a short, smooth window) and
    flagged `was_interpolated=True`.
  - Long gaps are left as `NaN` — never interpolated or extrapolated.
  - Edge gaps (e.g. the most recent settlement period, which the API
    often returns with `actual=null` because it isn't settled yet)
    cannot be interpolated (no data on one side) and are correctly
    left as `NaN` and reported separately (`edge_gaps_left_as_nan`),
    not mislabeled as filled.
  - Every run produces a `gap_report_*.csv` documenting exactly what
    was found and what was done — cite this directly in the thesis.
- **Caching**: each 14-day API chunk (the API's own hard limit per
  request) is cached to `data/raw/` by date range, so reruns don't
  re-hit the network. Chunk boundaries are anchored to the *end* of the
  window (now) and walk backwards, so widening the window later (as we
  just did, 180 → 365 days) reuses the already-cached recent chunks and
  only fetches the newly-added older ones. Delete files there to force
  a full re-fetch.
- **Why 365 days, not more, even though the API has data back to 2018**
  (checked live). GB grid carbon intensity has fallen substantially
  over the years as coal was phased out and renewables grew — training
  on several years of that history would teach the model a
  decarbonization trend baked into the past, not today's diurnal/
  seasonal pattern, which is what we actually want it to learn. One
  year is the minimum needed to see a full seasonal cycle exactly once
  under the *current* grid regime. This is a judgment call worth
  defending as "one full seasonal cycle of current-regime behaviour"
  rather than "maximum available history."
- **Time-based splits only.** Every train/test split in this project
  (see `forecasting.py`, once written) splits chronologically —
  shuffling time series data leaks future information into training,
  which is the standard mistake examiners check for.

### Sanity check performed on this run

Fetched the trailing 365 days (2025-09-23 → 2026-09-23, the actual
range used, recorded in `data/processed/gap_report_*.csv`): 17,521
expected settlement periods, all 17,521 returned by the API, **zero**
missing `actual` values. (An earlier 180-day pull had exactly 1 missing
value — the most-recent, not-yet-settled period — which had since
settled by the time this 365-day pull ran; nothing was fabricated
either way.)

Diurnal profile over the full year: ~53 gCO2/kWh swing between the
midday trough (~104 gCO2/kWh, 11:00 UTC) and the evening peak
(~157 gCO2/kWh, 18:00 UTC) — smaller than the summer-only 180-day
estimate (~78 gCO2/kWh), because winter has a much weaker midday solar
dip. This swing is what a carbon-aware scheduler exploits, and is the
reason the deadline-slack levels chosen (6h/12h/24h) matter (24h slack
can always reach the next day's trough; 6h often cannot).

Monthly means confirm real seasonal structure but it is *not* a clean
sinusoid: April 2026 was lowest (~93 gCO2/kWh) and September 2025
highest (~164 gCO2/kWh) in this window, with a fair amount of
month-to-month noise on top of the seasonal signal — consistent with
GB's grid mix being wind-heavy (weather-driven variability), not purely
solar-driven. Worth stating plainly at the viva: "seasonal" here means
a real but noisy signal, not a smooth annual curve.

## Forecasting

```bash
python src/forecasting.py
```

Trains DIRECT multi-horizon LightGBM quantile models (q10/q50/q90) for
every hourly horizon 1h..24h (72 models total, ~20s on this CPU-only
laptop), plus persistence and seasonal-naive baselines at each horizon,
and evaluates all of them on the same time-based test split.

### Why "direct" multi-horizon, not recursive (for the viva)

LightGBM has no native multi-step output. The alternative (recursive:
train one 1-step model, feed its own predictions back in as lags to
roll forward 48 steps) lets errors compound over 24h and is awkward for
quantiles specifically — feeding a q10 prediction back in as if it were
a real lag treats one quantile path as "the future," which breaks the
interpretation of the resulting trajectory as a genuine percentile
band. We train a separate model per horizon instead: no compounding
error, and every quantile at every horizon is an independently-fit,
well-defined conditional quantile. Full reasoning and the choice of
hourly (not half-hourly) horizon resolution is in the module docstring
of `forecasting.py`.

### Two things adapted from the prototype for the real schema

- **`forecast` (the grid operator's own forecast) is deliberately
  excluded from the model's features.** Handing our model National Grid
  ESO's own forecast for the exact value it's trying to predict would
  let it win by copying the answer key, which would defeat the point of
  comparing "our forecaster" against an oracle later.
- **Target and baseline alignment is done by timestamp, not row
  position** (`build_horizon_dataset`, `add_baseline_predictions`), so
  a positional `shift(-h)` can't silently misalign an origin with the
  wrong future row if any rows were ever dropped mid-series by a gap.

### Results (1-year data, 24 hourly horizons, saved to `results/tables/`)

Averaged across all 24 horizons:

| Forecaster | MAE | pinball q50 | 80% coverage |
|---|---|---|---|
| Persistence | 42.4 | 21.2 | — |
| Seasonal-naive (24h) | 36.8 | 18.4 | — |
| LightGBM point (=q50) | 27.4 | 13.7 | — |
| LightGBM quantile (q10/q50/q90) | 27.4 | 13.7 | **62.4%** |

LightGBM beats both baselines at every single horizon from 1h to 24h.
One interesting numerical check: at h=24, persistence and seasonal-naive
MAE are *exactly* equal (36.87 = 36.87) — this isn't a coincidence, it's
because a 24h-ahead persistence forecast ("value now") and a "same time
yesterday" seasonal-naive forecast with a 24h period are mathematically
identical when the forecast horizon equals the seasonal period.

**Honest finding to flag, not hide**: the nominal 80% [q10, q90]
interval only actually contains the true value ~58-70% of the time
across horizons (avg 62.4%), i.e. the quantile model is *overconfident*
— its uncertainty bands are too narrow. This is visible in
`results/plots/forecast_vs_actual_24h_week.png` (actual dips below the
band on 2026-07-15 and 2026-07-18). This is exactly the kind of
miscalibration the thesis's RQ3 (does uncertainty-awareness reduce
deadline violations?) needs to grapple with — an uncalibrated
uncertainty-aware scheduler could still miss deadlines more than its
80%-interval "should" suggest. Worth flagging as a limitation and a
candidate follow-up (e.g. conformal calibration) rather than a Step-2
fix, since it's outside this step's scope. **Addressed next, in
Calibration below.**

## Calibration (Conformalized Quantile Regression)

```bash
python src/calibration.py
```

### Why this step exists

The forecasting step above found that the LightGBM quantile model's
nominal 80% `[q10, q90]` interval only actually covered the true value
~62.4% of the time on the test set, averaged across the 1h-24h
horizons — the model is overconfident. This is a direct threat to RQ3
(*does an uncertainty-aware scheduler reduce deadline violations
relative to a naive point-forecast scheduler?*): if the "80%-confident"
interval the scheduler reasons about is really only ~62% reliable in
practice, any conclusion drawn from it about deadline-violation rates
would rest on a false premise. This had to be fixed before the
scheduler could be built on top of it.

### Method: Conformalized Quantile Regression (CQR)

> Romano, Y., Patterson, E., & Candès, E. (2019). *Conformalized
> Quantile Regression.* Advances in Neural Information Processing
> Systems (NeurIPS 2019), 32, pp. 3543–3553.

CQR wraps a trained quantile regressor with a post-hoc correction,
computed on a held-out **calibration** set, that achieves the target
marginal coverage exactly in a finite-sample, distribution-free sense —
it makes no assumption that the underlying quantile model is
well-specified or already well-calibrated, which is precisely the
situation found above.

**Three-way chronological split.** CQR's guarantee relies on the
calibration points being exchangeable with the test point conditional
on the model — i.e. the model must never have seen the calibration data
during training, exactly as it must never see the test data. This
requires a third, disjoint split carved out of what would otherwise be
the training set:

```
train (60%)  →  calibration (20%)  →  test (20%)
```

built chronologically (never shuffled, same reasoning as every other
split in this project) via two sequential applications of the
unmodified `time_based_split`. All 72 models (24 horizons × 3
quantiles) were **retrained from scratch on the smaller train-only
60% block** — the models forecasting.py originally saved were trained
on an 80% block that *included* what is now the calibration set, which
would have invalidated CQR's guarantee. The retrained models overwrite
those files in `results/models/quantile/` and are now the canonical
ones.

**Nonconformity score and correction (Algorithm 1 of the paper).** For
each calibration point with true value `y` and raw predicted interval
`[q10, q90]`:

```
s = max(q10_pred - y, y - q90_pred)
```

positive when `y` fell outside the interval, negative when safely
inside. The correction `Q` is the following order statistic of the
calibration scores (not a plain percentile — see below):

```
n = number of calibration points, alpha = 0.2 (for an 80% interval)
k = ceil((n + 1) * (1 - alpha))
Q = k-th smallest calibration score
```

The calibrated interval is then `[q10_pred - Q, q90_pred + Q]` (the
median `q50` is left untouched — CQR only ever adjusts the outer
quantiles).

**Why `ceil((n+1)(1-alpha))/n`, not the naive 80th percentile of the
scores?** A plain percentile is only *asymptotically* correct as
`n → ∞`. Split-conformal prediction's exact, finite-sample,
distribution-free coverage guarantee specifically requires the
`ceil((n+1)(1-alpha))`-th order statistic — the "+1" accounts for the
(unobserved) test point itself in the exchangeability argument that
proves the guarantee: a fresh draw is equally likely to rank anywhere
among the `n+1` calibration-plus-test scores. Using the naive
percentile instead would systematically under-cover by a small but
non-zero amount for any finite `n` — reintroducing a version of the
very overconfidence problem this step exists to fix.

### Results (1-year data, 24 hourly horizons)

Averaged across all 24 horizons:

| | Coverage | Interval width (gCO2/kWh) |
|---|---|---|
| Raw (uncalibrated) | 55.2% | 69.8 |
| CQR-calibrated | **80.0%** | 107.3 |
| Target | 80% | — |

Calibration brings average coverage to within 0.0 points of the 80%
target. The real, honest cost: intervals are **53.9% wider** on
average (+37.6 gCO2/kWh) to achieve that guarantee. This is a genuine
trade-off to discuss in the thesis, not a free fix — a scheduler
reasoning over a wider interval will treat more candidate execution
slots as "possibly high-carbon," making it more conservative about
shifting jobs, which is likely to show up later as smaller carbon
savings (relative to using the raw, overconfident interval) but with
fewer deadline violations. That trade-off is exactly what RQ3 is asking
about, and this calibration step is what makes it possible to study
honestly.

Note the raw (uncalibrated) coverage measured here (55.2%) differs from
the 62.4% found in the Forecasting section above — expected, not a
discrepancy to worry about: this run's models were trained on the
smaller 60% train-only block (vs. 80% before), so the underlying model
is somewhat different. CQR's guarantee does not depend on how
well- or badly-calibrated the raw model happens to be; the plot below
shows it pulls coverage to the ~80% target regardless of the sizeable
gap in the raw baseline (which ranged as low as ~45% at some horizons).

Full per-horizon numbers: `results/tables/calibration_by_horizon.csv`.
Comparison plot: `results/plots/coverage_before_after_cqr.png`.

## Workload generator

```bash
python src/workload_generator.py
```

Generates a synthetic trace of delay-tolerant batch jobs
(`data/processed/workload_trace_<tag>.csv`): Poisson-process arrivals,
lognormal durations (median 45 min, capped at 4h), and job counts
stratified evenly across the three deadline-slack buckets. Full
modeling rationale is in the module docstring of
`workload_generator.py`.

### Generated trace summary (1000 jobs)

- **Duration**: median 46.4 min, mean 62.4 min (right-skewed as
  intended), max 240 min (cap reached for the longest jobs).
- **Slack split**: 6h=334 (33.4%), 12h=333 (33.3%), 24h=333 (33.3%) —
  stratified, not left to chance.
- **Arrivals**: mean 2.75 jobs/day, std 1.76 — Poisson-consistent
  (variance roughly tracking the mean), not a flat/uniform spacing.

## Simulator and schedulers

```bash
python src/evaluate.py    # runs simulator.py internally, then evaluates
```

Simulates all 5 scheduler strategies (`schedulers.py`) against the
synthetic job trace and the real carbon data (`simulator.py`), then
computes the thesis's headline metrics (`evaluate.py`).

### Key design decisions

- **Only test-period jobs are used.** A job's entire decision window
  (arrival through its forced-start boundary) must fall inside the
  final ~20% chronological test split from `calibration.py` — the only
  period genuinely unseen by every trained model. Of the 1000 generated
  jobs, **201 (20.1%)** qualify (67 at 6h slack, 73 at 12h, 61 at 24h).
- **Decision grid**: hourly steps, snapped to the settlement grid (see
  `simulator.py`'s module docstring for the full reasoning). A job's
  continuous arrival time is snapped UP to the next 30-min boundary as
  its first decision point; its true latest allowable start
  (`arrival + slack_hours`, exact, since duration cancels out of
  `deadline - duration`) is snapped DOWN as the forced boundary —
  flooring can only move the boundary earlier, so it can never cause a
  violation. This costs a job up to ~1 hour of nominal slack to
  discretization, stated plainly rather than modeled away.
- **Carbon cost is always the ACTUAL, integrated cost** of the settlement
  periods the job's run window overlaps, regardless of what any
  scheduler believed when it decided to start — a scheduler can act on a
  forecast, but a job burns whatever the grid actually was.
- **No capacity constraints.** Jobs are scheduled fully independently;
  the "stampede effect" (many schedulers converging on the same
  low-carbon window) is not prevented, only observed (see the
  concurrency metric below).

### A bug the sanity tests were designed to catch — and did

The task specified that oracle should use "the same comparison logic as
forecast_greedy" (comparing point carbon intensity at candidate start
times) but with true future values instead of forecasts. Implemented
literally, this is **not** a valid per-job upper bound once a job's
duration spans multiple settlement periods: the single lowest-point-
intensity slot can be immediately followed by a spike, giving it a worse
*fully-integrated* realized cost than a slightly-higher-starting slot
with a calmer follow-on period. This showed up empirically as 16 of 201
jobs (~8%) where the point-based oracle's realized cost was *higher*
than forecast_greedy's — even though oracle still won in aggregate
(35,199 vs 38,317 gCO2 total) — exactly the discrepancy the task's own
sanity test ("oracle ≥ forecast_greedy for every job") was designed to
surface.

**Fix**: oracle's decision rule was corrected from "lowest point
intensity at the candidate start" to "lowest true *integrated* cost
across the job's full duration" (`ctx.realized_cost_at` in
`simulator.py`, used only by oracle — perfect information a deployable
scheduler could never have). This makes oracle directly optimize the
same objective it is scored on, so it is now a genuine per-job upper
bound by construction: verified for every job, against every other
scheduler, in `tests/test_simulator.py`. `forecast_greedy` and
`uncertainty_aware` were deliberately left comparing point forecasts,
unchanged — a realistic simplification for a deployable heuristic (a
real scheduler would have to forecast the full integrated cost of every
candidate, not just its starting intensity), not something that needed
the same fix. Worth a sentence in the methodology chapter as a concrete
example of a subtle evaluation-design bug caught by a correctness test
before it could distort a headline number.

### Results (201 test-period jobs, all figures gCO2/kWh unless noted)

Carbon saved (%) vs `run_immediately`, by scheduler and slack bucket:

| Scheduler | 6h | 12h | 24h | Overall |
|---|---|---|---|---|
| threshold | 7.4% | 12.5% | 22.5% | 14.3% |
| forecast_greedy | 11.5% | 20.1% | 35.2% | 22.5% |
| oracle | 18.7% | 28.0% | 44.3% | 30.6% |
| uncertainty_aware | 1.6% | 1.0% | 0.5% | 1.0% |

**RQ1 (does more slack save more carbon?)** — yes, clearly, for every
carbon-aware scheduler except uncertainty_aware: savings roughly double
from 6h→12h and again 12h→24h, tracking the diurnal swing found in the
EDA step (more slack = more chances to reach the next low-carbon
trough).

**RQ2 (gap to oracle — fraction of the achievable ceiling captured)**:

| Scheduler | 6h | 12h | 24h | Overall |
|---|---|---|---|---|
| threshold | 60.5% gap | 55.3% gap | 49.1% gap | 53.3% gap |
| forecast_greedy | 38.8% gap | 28.2% gap | 20.5% gap | 26.5% gap |
| uncertainty_aware | 91.3% gap | 96.4% gap | 98.9% gap | 96.6% gap |

forecast_greedy captures roughly **73.5%** of oracle's achievable
savings overall (26.5% gap), and its gap *narrows* as slack grows — more
slack gives the forecaster more usable low-carbon candidates to find
even with imperfect forecasts. threshold captures under half. See
`results/tables/scheduler_comparison.csv` for the full breakdown.

**RQ3 (regret — the key finding, reported honestly)**:

| Scheduler | Regret rate | Avg regret magnitude (gCO2) |
|---|---|---|
| forecast_greedy | 11.0% (overall) | 44.7 |
| uncertainty_aware | 2.0% (overall) | 12.6 |

uncertainty_aware regrets **less often and less severely** than
forecast_greedy when it does — exactly the trade-off RQ3 asks about.
But the honest, complete picture from the table above: it achieves this
by barely shifting jobs at all (0.5–1.6% carbon saved vs
forecast_greedy's 11.5–35.2%), because the CQR-calibrated q90 bound
(53.9% wider than the raw, overconfident interval — see Calibration
above) is wide enough that "the worst case at the best future slot beats
running now for certain" is a very hard bar to clear. This scheduler, as
specified, is safe almost to the point of being inert on this workload
and grid. That is a real, useful, and somewhat uncomfortable finding for
the thesis to discuss — not a bug: it directly demonstrates that CQR's
correctly-widened intervals, while solving the coverage problem from
Step 4, create a new design tension (a scheduler that only trusts a
forecast when its full calibrated uncertainty band supports it will act
conservatively almost by default). **Addressed next, in the
risk-tolerance sweep below**: rather than treating this fixed policy as
the end of the story, we parameterize it with a tunable risk weight and
ask whether the fixed q90 threshold was simply one extreme of a broader,
genuinely useful spectrum.

**Deadline violations**: 0 for every scheduler, every bucket — the
correctness check passed cleanly (see `evaluate.py`'s loud-failure
check and `tests/test_simulator.py`).

**Observational concurrency** (job-minutes per 30-min settlement
period, no capacity limit enforced): max ranges from 60.0
(run_immediately) to 150.0 (oracle) job-minutes; p95 is identical
(30.0) across all five schedulers, meaning the carbon-aware schedulers'
extra bunching is concentrated in occasional peaks, not a systematic
shift of the whole distribution. Full table in
`results/tables/scheduler_comparison.csv`.

Full per-job records: `results/tables/simulation_results.csv`. Plot:
`results/plots/carbon_saved_by_bucket.png`.

## Risk-tolerance sweep (the real contribution)

```bash
python src/risk_sweep.py
```

Step 5's `uncertainty_aware` was a single fixed policy: compare the
CQR-calibrated **q90** (full worst case) against the current actual, and
only shift if the future's worst case still wins for certain. That
result — barely any carbon saved, but very low regret — is one endpoint
of a spectrum, not "the" answer. `schedulers.make_uncertainty_aware
(risk_weight)` generalizes it: interpolate a threshold between the
median and worst-case forecast at the best future candidate t\*,

```
threshold = q50(t*) + risk_weight * (q90(t*) - q50(t*))
```

`risk_weight=0.0` trusts the median only (proven **exactly** — not just
"similar in spirit" — equivalent to forecast_greedy's decision rule,
verified per-job in `tests/test_schedulers.py`); `risk_weight=1.0`
reproduces Step 5's original policy exactly. No changes to
`simulator.py` were needed — `simulate_job()` already accepts any
callable with the standard scheduler signature.

### The frontier: carbon saved vs regret, overall (201 jobs)

| risk_weight | % carbon saved | regret rate | avg regret magnitude (gCO2) | gap to oracle |
|---|---|---|---|---|
| 0.0 (= forecast_greedy) | 22.5% | 11.0% | 44.7 | 26.5% |
| 0.2 | 18.0% | 8.5% | 31.4 | 41.2% |
| 0.4 | 12.6% | 7.5% | 26.6 | 58.8% |
| 0.6 | 5.6% | 6.5% | 22.2 | 81.6% |
| 0.8 | 3.5% | 4.5% | 31.0 | 88.6% |
| 1.0 (= Step 5 uncertainty_aware) | 1.0% | 2.0% | 12.6 | 96.6% |

`regret_rate` told a much cleaner story than `avg_regret_magnitude`: the
rate decreases essentially monotonically across the sweep (a clean
staircase — see `results/plots/risk_frontier_regret_rate.png`), while
the magnitude is noisy and non-monotonic (`results/plots/
risk_frontier_regret_magnitude.png`), because at high risk_weight so few
jobs regret at all (as low as 0-2 jobs at some bucket/weight
combinations) that its average is dominated by whichever one or two
jobs happened to regret. Full table (with the 6h/12h/24h breakdown):
`results/tables/risk_sweep_results.csv`.

### Honest read: is there a genuine sweet spot?

**No single risk_weight Pareto-dominates both endpoints.** Both %
carbon saved and regret rate decrease monotonically as risk_weight
increases from 0 to 1 (overall) — every point on the frontier is a
real, non-dominated trade-off, not a free lunch. This is itself worth
stating plainly rather than searching for a result that isn't there.

What the frontier does show is **uneven marginal value** along its
length — the exchange rate between savings given up and regret avoided
is far from constant:

- **0.0 → 0.2** (the cheapest step): gives up 4.5 points of savings to
  avoid 2.5 points of regret — roughly 1.8 points of savings per point
  of regret avoided.
- **0.2 → 0.6** (the expensive middle): gives up 12.4 points of savings
  to avoid only 2.0 points of regret — roughly 6.2 points of savings
  per point of regret avoided, more than 3× worse value than the first
  step.
- **0.6 → 1.0** (cheap again): gives up 4.6 points of savings to avoid
  4.5 points of regret — close to 1:1.

So the practical recommendation isn't "pick risk_weight=0.5" — it's
that **risk_weight ≈ 0.2 is the best cheap first step away from
forecast_greedy's regret**, and if more regret-aversion is wanted, it is
far more efficient to move most of the way toward 1.0 than to linger in
the 0.2–0.6 range, which buys comparatively little regret reduction for
the savings it gives up. This is a genuinely useful, nuanced result —
not "intermediate values are always better," but "the frontier has a
cheap region near each end and an expensive region in the middle,"
which is itself a finding worth a paragraph in the discussion chapter.

**Caveat on the per-bucket breakdown**: the 24h bucket (61 jobs, and as
few as 2-5 jobs regretting at some risk_weight values) shows a
non-monotonic, visibly noisy frontier (`results/plots/
risk_frontier_by_bucket.png`) — small-sample noise, not a real reversal
of the trend. The 12h bucket (73 jobs) shows the cleanest, most
textbook-monotonic frontier of the three. Report the overall (pooled)
frontier as the headline result; treat the per-bucket frontiers as
suggestive, not independently conclusive, given the sample sizes
involved.

## Tests

```bash
pytest tests/ -v
```
