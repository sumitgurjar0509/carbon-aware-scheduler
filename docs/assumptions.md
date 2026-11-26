# M2 assumptions (base-paper LP reimplementation)

Where the base paper (Zanotto et al., GreenSys '26) is unclear, this is
what the code does. The full reasoning and page references are in
`docs/m2_spec.md`. Each row names the code that implements it.

| # | Assumption | Code |
|---|---|---|
| G1 | Eq. 1 is printed with `min{…, 0}`; implemented as `max{…, 0}`, so the grid share is in [0, 1] | `lp_scheduler.local_adjusted_ci` |
| G2 | A job of `D` hours occupies the `D` hourly slots `t … t+D−1` (not `D+1`) | `feasible_options`, `Grid.actual_emissions_g` |
| G3/G4/G19 | `N` and `D` are independent `Normal(6, 2)` draws, rounded and redrawn until valid: `D ∈ [1, 24]` h, `N ∈ [1, 20]` nodes | `_truncated_rounded_normal`, `generate_jobs` |
| G5 | Planning horizon `T = 24` h (limited by our forecaster) | `HORIZON_T` |
| G6 | Decision hour `o` = arrival rounded up to the hour; forecasts at `o` use data up to hour `o−1`; slots `o … o+23` | `generate_jobs`, `hourly_forecasting` |
| G7 | `Ē = c / max c` over the job's feasible options; `L̄ = 1` on the local cluster, else 0 | `objective_coefficients` |
| G8 | Objective uses `c_tj` (summed CI) as in Eq. 3; `N · C_hw` appears only in the reported emissions | `objective_coefficients`, `actual_emissions_g` |
| G9 | Ties broken by lowest cluster index, then earliest start (penalty ≤ 1e-6) | `TIE_EPS` |
| G10 | A fresh forecast for every decision hour | `forecast_tables` |
| G11 | Random FIFO: start at the decision hour on a random cluster with room (local included); if none, the next hour | `run_random_fifo` |
| G12/G13 | Reduction = `1 − total(policy) / total(random FIFO)`; every policy scored with ACTUAL hourly CI | `summarise` |
| G14 | Headline comparison uses `α = 1`; `α = 0.5` and `0` also reported | `ALPHAS` |
| G15 | 1000 jobs per 30.44-day month over the M1 test period's arrival window → 2,359 jobs | `generate_jobs` |
| G16 | Binary model solved by CBC; each job checked against brute force | `solve_lp`, `solve_brute_force` |
| G17 | No on-site solar: `P_gen = 0`, so local CI = GB CI | `local_adjusted_ci` defaults |
| G18 | Hourly CI = mean of the `:00` and `:30` periods; incomplete hours dropped, never filled | `hourly_forecasting.to_hourly` |
| G20 | Seeds: jobs 20260923, FIFO 20260924 | `JOB_SEED`, `FIFO_SEED` |
| §3.1 | 1 local (10 nodes) + 8 cloud clusters (20 nodes each), all with GB CI | `CAPACITIES` |
| §3.3 | Separate hourly LightGBM + CQR models, h = 0…23, same feature time spans as M1, trained on M1's train period, calibrated on M1's calibration period (targets never cross a boundary) | `hourly_forecasting.py` |
| L1 | Test period only (≈73 days, summer 2026); multi-season evaluation in M4 | — |
