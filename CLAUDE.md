# CLAUDE.md — carbon-aware-scheduler (M.Tech CSE thesis)

## What this project is
M.Tech thesis: uncertainty-aware carbon-aware scheduling of delay-tolerant cloud jobs.
Student: beginner-to-intermediate Python. Explain every change in plain language.

Base paper: Zanotto, Padovani, Iacca, Fiore, "Carbon Intensity Forecasting with On-Site
Renewables for Sustainable AI Workload Scheduling", GreenSys '26 (ACM),
DOI 10.1145/3802973.3804455. PDF should be at `docs/base_paper.pdf` — read it before
implementing anything that claims to reproduce it.

Base paper reported: LP scheduler with Chronos 2 forecasts cuts emissions 27.7% vs random
FIFO; ground-truth carbon intensity gives 32.09%. Their scheduler uses point forecasts only.

Thesis contribution (Technique 1): an uncertainty-aware version of the base paper's LP that
uses calibrated forecast quantiles (CQR) via scenario sampling and a CVaR risk term, compared
like-for-like against a reimplementation of the base paper's LP.

## Existing code (src/)
- data_pipeline.py — GB carbon intensity from UK Carbon Intensity API, 30-min, cached
- forecasting.py — 72 LightGBM models (24 horizons x q10/q50/q90), persistence, seasonal-naive
- calibration.py — split-conformal CQR, 60/20/20 chronological split
- workload_generator.py — synthetic single jobs (seed 42)
- simulator.py — hourly decision grid, integrated cost, forced start at latest slot
- schedulers.py — run_immediately, threshold, forecast_greedy, oracle, uncertainty_aware
- evaluate.py, risk_sweep.py — metrics (pct saved, regret, gap to oracle)
- tests/ — pytest suite; keep it passing

## Hard rules (research integrity)
1. Never tune any parameter (tau, rho, beta, alpha, CVaR level) on test data. Tune on the
   calibration period only; run the test period once with the chosen values.
2. Every policy compared on IDENTICAL jobs, carbon data, capacity and horizon.
3. Fix all random seeds; record them in the results files.
4. Never overwrite committed results in results/tables/ without saying so; write new
   experiments to results/tables/<experiment_name>/.
5. Never invent or "expect" numbers. Report only what the code produced.
6. Keep published, reproduced and proposed results in separate, clearly labelled columns.
7. Linear expected cost: minimising the mean of linear scenario costs equals using the mean
   forecast. Any gain from uncertainty must come from the risk term (CVaR) or re-planning,
   and the write-up must say so.

## Working style
- One milestone at a time. At the end of each: run `pytest tests/ -v`, show a short summary
  of what changed and the key numbers, then STOP and wait for the student.
- Work on a git branch per milestone (e.g. `m1-reproducibility`). Commit with clear messages.
  Never push unless asked.
- Add a test for every new function.
- Windows machine. Pinned: numpy==1.26.4, lightgbm==4.3.0, scikit-learn==1.4.2 — do not
  upgrade these (they break on this PC). New dependency allowed: pulp (bundles the CBC solver).

## Milestones for Technique 1
- M1 Reproducibility: fixed data window via CAS_START/CAS_END env vars; scripts load the
  dataset by explicit tag CAS_DATA_TAG instead of newest file; regenerate tables and compare
  with results/tables_committed/ (scripts/compare_tables.py).
- M2 Base-paper LP reimplementation (src/lp_scheduler.py, PuLP + CBC): job generator per the
  paper (read docs/base_paper.pdf for exact parameters), capacity constraints, alpha weight,
  random-FIFO baseline, LP with point forecast, LP with ground truth. Hourly data (average of
  the two half-hours). Document every assumption where the paper is unclear in
  docs/assumptions.md. If only GB data is available, restrict to one grid and say so.
- M3 Uncertainty-aware LP: sample S scenarios of future intensity from calibrated q10/q50/q90
  (piecewise-linear inverse CDF, Gaussian copula with correlation rho across horizons);
  objective = (1-beta)*mean scenario emissions + beta*CVaR_a (Rockafellar-Uryasev
  linearisation). Optional: rolling re-plan every hour.
- M4 Evaluation harness: many months x seeds, paired Wilcoxon test vs base-paper LP,
  bootstrap 95% CIs, "share of gap to ground truth closed".
- M5 Ablations: forecaster (point vs calibrated), beta, rho, S, re-planning on/off.
