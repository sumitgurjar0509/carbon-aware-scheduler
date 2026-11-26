# M3 specification: uncertainty-aware LP (scenarios + CVaR + re-planning)

Status: **draft for approval**. No code yet.

Builds on M2 (`docs/m2_spec.md`, branch `m2-base-lp`, results in
`results/tables/m2_base_lp/`). Everything not mentioned here is kept
**exactly as in M2**:

- clusters (1 local of 10 nodes + 8 cloud of 20, all GB CI);
- job generator, `T = 24` h, `C_hw = 8.5` kW;
- the paper's α / local-cluster term;
- the corrected G9 tie-break;
- scoring every policy with the actual hourly CI.

Notation:

| Symbol | Meaning |
|---|---|
| `o` | decision (origin) hour |
| `h = 0…23` | horizon |
| `k = (t, j)` | an option: start offset `t` on cluster `j`, as in M2 |
| `K` | set of feasible options |
| `S` | number of scenarios |
| `s` | one scenario |
| `a` | CVaR tail fraction (`CVaR_a` = mean cost of the worst `a` share of scenarios) |
| `β` | risk weight |
| `ρ` | copula correlation |

---

## 1. Scenario generation

**Inputs.** For each origin `o` and horizon `h`, the hourly models in
`results/models/m2_hourly/` give `q10, q50, q90`. `q10` and `q90` are
CQR-calibrated; `q50` is raw (CQR never changes it). The CQR corrections
used in M3 are re-fitted on C1 only (§5); the quantile models
themselves are **not retrained**.

**Marginal per horizon: a piecewise-linear inverse CDF with linear tails.**

1. Sort the three values (monotone rearrangement), so a crossed forecast still gives a valid distribution.
2. Knots: `(0, Q0), (0.1, q10), (0.5, q50), (0.9, q90), (1, Q1)`, joined by straight lines, where
   - `Q0 = q10 − (q50 − q10)/4`
   - `Q1 = q90 + (q90 − q50)/4`
   (each tail continues the slope of its neighbouring inner segment).
3. Floor every value at 0 gCO2/kWh. Carbon intensity cannot be negative; the floor is applied after sampling and the number of floored values is reported.

This is the same shape as `schedulers.quantile_function_knots` (M1 step
7). It is reused, with new hourly inputs.

**Dependence across horizons: a Gaussian copula.**

- `Z ~ N(0, Σ)` with `Σ_hk = ρ^|h−k|` (AR(1) form: neighbouring hours are more correlated than distant ones, as in a real time series).
- `U_h = Φ(Z_h)`, and `CI_h^s = Q_h(U_h^s)`.
- `ρ = 0` gives independent hours; `ρ → 1` gives fully comonotone paths.
- Alternative not proposed: equal correlation for every pair of hours. It ignores that errors decay with distance.

**One path for all clusters.** All 9 clusters share the GB grid, so a
scenario is one 24-hour path used for every cluster. Eq. 1–2 with
`P_gen = 0` keeps the local cluster's CI equal to the GB CI.

**Seeds: common random numbers.** The `Z` draws for origin `o` use
`seed = hash(SCENARIO_SEED, o)`, with `SCENARIO_SEED = 20261126`. So every
variant, every `β` and every re-plan at the same hour sees **identical
scenarios**. This is fixed and recorded.

**Known limitations, to be written up:**

- CQR guarantees only the marginal 80% coverage of `[q10, q90]`. `q50` is not calibrated.
- Anything beyond `q10`/`q90` is a modelling assumption: linear, bounded tails.
- `ρ` is a single number shared by all origins.
- With `a ≤ 0.1`, `CVaR_a` of a single hour lies entirely in the assumed upper tail (`p > 0.9`). For job costs summed over `D` hours with `ρ < 1`, this is softened but not removed.

---

## 2. CVaR model for one job

**Scenario cost.** For a feasible option `k = (t, j)`:

```
c_k^s = Σ_{τ=t}^{t+D−1} CI_j^s(o+τ)
```

(planned CI in scenario `s`, local cluster adjusted by Eq. 1–2 as in M2).

**Normalisation (M2's G7, extended).** `C = max_{k∈K} c̄_k`, where
`c̄_k = (1/S) Σ_s c_k^s`. Then `Ē_k^s = c_k^s / C`, and `L̄_k = 1` if
`j = local`, else 0. The same normaliser is used for the mean and CVaR
terms, so `β` trades like for like.

**Model (Rockafellar–Uryasev).** Binary `x_k`, free `η`, and `u_s ≥ 0`:

```
minimise   Σ_k [ α(1−β)·(1/S)Σ_s Ē_k^s − (1−α)·L̄_k + ε·(t_k·9 + j_k)/(9·T) ] · x_k
           + αβ · ( η + 1/(a·S) · Σ_s u_s )
subject to Σ_k x_k = 1
           u_s ≥ Σ_k Ē_k^s · x_k − η        for every s
           u_s ≥ 0,   x_k ∈ {0, 1},   η free,   ε = 1e-6
```

- **Emissions term:** `α · [(1−β)·mean + β·CVaR_a]`.
- **Local-use term:** `−(1−α)·L̄`, as in Eq. 5.
- **Tie-break:** the corrected G9 rule (earliest start, then local, then lowest cloud index).
- `S` is chosen so that `a·S` is a whole number; the empirical CVaR is then exact.

**Why binary variables are needed.** Unlike M2, the relaxation is not
automatically 0/1. A fractional mix of options can have a lower CVaR than
any single option, because averaging spreads the risk. So `x` must stay
binary, and CBC solves it as a MIP.

**Equivalent per-option form (proposed for bulk runs).**

- For a fixed choice `x = e_k`, the inner minimum over `η, u` equals exactly the empirical `CVaR_a` of `{Ē_k^s}` (Rockafellar & Uryasev 2000, discrete case). Because exactly one option is chosen, the MIP optimum is `min_k` of
  `α[(1−β)·mean_s Ē_k^s + β·CVaR_a(Ē_k^s)] − (1−α)L̄_k + tie_k`.
- That is an M2-shaped problem (one binary per option, one equality), with each option's CVaR precomputed.
- **Proposal:**
  - Implement the full RU MIP as the **reference model**. Check it against both the per-option form and brute-force enumeration in tests, and on the first 200 jobs of every test run, reporting the number of disagreements.
  - Use the per-option form (still PuLP + CBC) for the full runs and the tuning grid, because of solve time (§7).
- **Your decision is needed here.**

**What `β = 0` means (CLAUDE.md rule 7).**

- With `β = 0`, the emissions term is `α·(1/S)Σ_s Ē_k^s`. Cost is linear in CI, so this equals planning on the **scenario-mean forecast** `mean_s CI^s`. **It adds no uncertainty information beyond a point forecast.**
- That mean is *not* M2's `q50`: the piecewise-linear distribution is usually skewed, so its mean differs from its median.
- So a gain of `β = 0` over the M2 base LP comes from **using the mean instead of the median**, which is a point-forecast choice, **not** a benefit of modelling uncertainty.
- **Any gain attributed to uncertainty must come from `β > 0` (the CVaR term) or from re-planning.** The write-up must say so.
- To keep the two effects apart, the `β = 0` scenario-mean LP is run as a control on identical scenarios (§4).

---

## 3. Re-planning (hourly)

**Rule.** At every hour `H`, in arrival order (earliest first):

1. Jobs that have **started** are fixed. They are non-preemptive, and their nodes stay reserved until they finish.
2. Every job that has arrived (decision hour `o ≤ H`) and not yet started has its tentative reservation released and is re-solved:
   - with a **fresh forecast at origin `H`** (fresh scenarios too, for CVaR);
   - over its **remaining window** `H … o+T−D` for the start (its original latest start never moves);
   - against capacity used by started jobs and by the new reservations of jobs re-planned earlier this hour.
3. Jobs arriving at `H` are planned last.
4. Every job whose new plan starts at `H` starts and becomes fixed.

**Capacity rule during a re-plan (recommended).** When waiting job `i` is
re-planned at hour `H`, the capacity it sees as taken is:

- nodes of started jobs;
- the **old** reservation of every other waiting job;
- the **new** reservation of every waiting job already re-planned this hour.

In words: a job may move only into capacity that was free at the start of
the hour, plus its own released slot. A slot that another job gives up
becomes usable only from the next hour.

**Why a feasible plan always exists.** Job `i`'s previous plan (start ≥
`H`, otherwise it would already have started) is still feasible when it
is re-planned:

- It did not overlap any started job or any old reservation when it was made.
- No job re-planned earlier this hour can have moved into it: they could only use capacity free at the start of the hour, and `i`'s slot was reserved then.

So keeping the old plan is always an option, and re-planning can never
leave a waiting job without a plan.

**New arrivals.** Jobs arriving at `H` have no old plan. They are planned
after all waiting jobs, against started jobs plus everyone's new
reservations (capacity freed this hour is available to them). As in M2,
the run stops with an error if one of them finds no feasible option.

**Cost of the rule.** It is slightly conservative: a freed slot waits one
hour before others can use it. That keeps the model simple and safe.

**Tests** will check that capacity is never exceeded, that no job ever
loses its plan, and that every job starts no later than `o + T − D`.

**Forecast coverage.** The forecast at origin `H` covers hours `H … H+23`,
which includes the remaining window `H … o+T−1`, because `H ≥ o`.

**Actual-CI bound.** M2's `lp_actual_*` (plan once on arrival with the
true CI) stays the bound. With perfect information, re-planning cannot
change any job's own best choice. It could only change how jobs share
capacity, so it is not part of the bound.

---

## 4. Variants (test run, all at α = 1 for the headline, plus α = 0.5)

| Name | Planning CI | Re-plan | Purpose |
|---|---|---|---|
| `base_lp` | `q50` point forecast | no | M2's `lp_forecast` (identical code path; must reproduce M2 exactly) |
| `base_lp_replan` | `q50`, fresh each hour | yes | effect of re-planning alone |
| `mean_lp` (control, β = 0) | scenario mean | no | separates "mean vs median" from the risk term (§2) |
| `cvar_lp` | scenarios, chosen β > 0 | no | effect of the CVaR term alone |
| `cvar_lp_replan` | scenarios, chosen β > 0, fresh each hour | yes | both together |
| `mean_lp_replan` (control) | scenario mean, fresh each hour | yes | control for the above |
| `actual_lp` | actual CI | no | bound (M2's `lp_actual`) |
| `random_fifo` | — | — | baseline (M2's, same seed) |

**Identical everything (rule 2):**

- M2's 2,359 jobs (`jobs_seed20260923.csv`), FIFO seed 20260924;
- the same data, capacities, `T` and α;
- the same scenarios per origin hour.

The two `mean_*` controls are added to your four variants because of
rule 7. They cost little: no CVaR, and the same solver path as `base_lp`.

---

## 5. Choosing parameters without touching the test period

**The problem.** CQR's corrections were fitted on the whole calibration
period (2026-04-30 13:00 → 2026-07-12 08:00 UTC, 72.8 days). Tuning `β`
on the same period would use intervals whose coverage is in-sample there,
so it would look better calibrated than it will be on the test period.

**Proposal: split the calibration period by time.**

| Block | Dates (UTC) | Use |
|---|---|---|
| Train | 2025-09-24 → 2026-04-30 12:30 | LightGBM models (unchanged from M2) |
| **C1** | 2026-04-30 13:00 → 2026-06-06 00:00 (≈ 36.5 days) | Re-fit the CQR corrections only |
| **C2** | 2026-06-06 00:00 → 2026-07-12 08:30 (≈ 36.4 days) | Estimate `ρ`, choose `S`, choose `β` (and `a`, if not fixed) |
| Test | 2026-07-12 08:30 → 2026-09-23 03:30 | One final run with the chosen values |

- **Boundaries:** a C1 row is used only if its target hour ends by 2026-06-06 00:00 (M2's strict rule).
- **New corrections** go to `results/models/m3_cqr_c1/`; `m2_hourly/` is not changed.
- **M2 results are unaffected**, because the base LP uses only `q50`, which CQR never changes.
- **Cost:** CQR now uses about half the points (≈ 870 per horizon instead of 1,747). Its finite-sample guarantee still holds; the correction just varies a bit more.
- **Alternative rejected:** tuning on the end of the training period. LightGBM was fitted on those rows, so its errors there are optimistic.

**How each parameter is chosen.** Order: ρ, then S, then a, then β. Each
step uses C2 only.

1. **ρ: estimated, not tuned on schedules.**
   - For each C2 origin and horizon, compute `u_h = F_h(y_h)` (the actual value's position in the forecast distribution) and `z_h = Φ⁻¹(u_h)`.
   - Fit the AR(1) `ρ` by maximum likelihood on `z_0 … z_23`.
   - Report the fit and a check: coverage of 6-hour window sums in C2.
2. **S: chosen for Monte Carlo stability, not performance.**
   - Pick the smallest `S ∈ {100, 200, 500, 1000}` such that, on the C2 jobs, decisions agree with `S = 2000` for at least 99% of jobs.
   - Use the median `β` of the grid for this check. `a·S` must be a whole number.
3. **a: proposed fixed a priori at `a = 0.2`, not tuned.**
   - It matches CQR's α.
   - At `a = 0.2`, half the tail mass (`p` in 0.8–0.9) lies inside the calibrated region. At `a = 0.1` it would be entirely in the assumed tail.
   - Fewer tuned parameters on a 36-day block also means less overfitting.
   - Alternative, if you prefer: tune `a ∈ {0.1, 0.2}` together with `β`.
4. **β: tuned on C2 with a rule declared before any run.**
   - Grid `β ∈ {0, 0.1, 0.25, 0.5, 0.75, 1}`, separately for with and without re-planning, at α = 1.
   - C2 workload: the M2 generator with a **different seed** (`TUNE_SEED = 20260606`), 1000 jobs per month, giving ≈ 1,163 jobs, plus the FIFO seed.
   - **Selection rule:** choose the `β` with the **lowest total realised emissions** on C2.
   - Ties within 0.1% of the best go to the **smallest β** (simplest model).
   - The whole curve is reported, not just the winner.
   - If the winner is `β = 0`, that is reported as the result ("the CVaR term did not help on C2"), and the test run still includes `cvar_lp` with the best `β > 0` from the grid, **labelled as such**.

After that, the test period is run **once** with the chosen values
(rule 1). The chosen values and seeds are written into
`results/tables/m3_uncertainty_lp/run_info.json`, and the C2 tuning
tables go to `results/tables/m3_uncertainty_lp/tuning_c2/`.

---

## 6. Metrics

All metrics are computed on the test period, scored with actual CI, for
every variant at α = 1 (and α = 0.5).

| Metric | Definition |
|---|---|
| Cut vs random FIFO (%) | `100·(1 − Σ E_variant / Σ E_fifo)`, as in M2 |
| Share of actual-CI gap closed (%) | `100·(E_base − E_variant) / (E_base − E_actual)`, using totals. 0 = `base_lp`, 100 = `actual_lp`; can be negative |
| Regret rate (%) | Share of jobs whose realised emissions are higher than under `base_lp` (paired by `job_id`, tolerance 1e-9 g). Also the mean regret in kgCO2 |
| CVaR₀.₁ of per-job emissions | Mean of the worst 10% of jobs. Reported in two ways: (i) per-job emission intensity, `E / (N·C_hw·D)` in gCO2/kWh (the main one, so that big jobs don't dominate); (ii) raw per-job kgCO2 |
| Mean start delay (h) | as in M2 |
| Also | Share of jobs on the local cluster; number of re-plans per job; solve times; RU-vs-per-option disagreements; scenario values floored at 0 |

**Caution.** Per-job pairing is not a clean experiment. Jobs share
capacity, so one job's change can move another job. Regret is therefore
reported as an outcome comparison, not as a measure of each decision.

**Statistical plan (to run in M4, not in M3):**

- **Unit:** a run = (period, job seed). Within one run, jobs are correlated through shared days and capacity, so the run, not the job, is the unit of analysis.
- **Primary comparison** (declared now): `cvar_lp_replan` vs `base_lp`, total emissions, α = 1.
- **Secondary:** each variant vs `base_lp`, and `cvar_*` vs `mean_*` (the pure risk-term effect).
- **Test:** paired Wilcoxon signed-rank on per-run differences, with Holm correction across the secondary comparisons.
- **Intervals:** bootstrap 95% CIs over runs, and a moving-block bootstrap by day within a run to describe single-run uncertainty.
- **Runs:**
  - Several job seeds (e.g. 10) on each available period.
  - More periods need **rolling-origin retraining**: the models are trained only on the first 60% of the year, so later months are the only true out-of-sample data. This is an M4 design decision.
- M3 itself reports **one run** (test period, M2 jobs) **with no significance claims**.

---

## 7. Problem size and solve-time estimates

All figures below are **estimates, not measurements**. Phase B times 100
solves of each kind before any full run. Reference measurement from M2:
the M2-sized model takes 0.025 s per solve, including PuLP's overhead.

**Per job, `K ≤ 216` options (≈ 171 at `D = 6`):**

| Model | Variables | Rows | Non-zeros | Estimated time per solve |
|---|---|---|---|---|
| M2 / per-option CVaR | K binaries | 1 | K | ≈ 0.03 s (measured in M2) + scenario costs (K × S × D additions; a few ms with numpy) |
| RU MIP, S = 200 | K binaries + 201 continuous | 201 | ≈ 34,000 | ≈ 0.3–2 s |
| RU MIP, S = 1000 | K + 1001 | 1001 | ≈ 171,000 | ≈ 2–10 s |

For the RU MIP, PuLP building and writing the model file in Python
dominates, and the relaxation may branch.

**Number of solves:**

- **Without re-planning:** 2,359 jobs per variant.
- **With re-planning:** a job is solved on arrival and then once per hour until it starts. M2's α = 1 mean delays (7.6 h forecast, 10.6 h actual) suggest about 8–12 solves per job, so ≈ 19,000–28,000 per variant.

**Total time, test run** (α ∈ {1, 0.5}, 6 forecast-based variants):

- Per-option form: ≈ 2 × (3 × 2,359 + 3 × ~24,000) × 0.03–0.05 s ≈ **1.3–2.2 h**.
- Using the RU MIP everywhere: roughly 12–90 h. Too slow, hence the proposal in §2.

**Tuning on C2:** ≈ 1,163 jobs, 6 β × 2 (re-plan on/off), plus ρ and S checks. Re-planning dominates, ≈ 12 × ~12,000 × 0.04 s ≈ **1.6 h** with the per-option form.

**Reference checks:** the RU MIP on the first 200 jobs of every test run, ≈ 200 × 6 × ~1 s ≈ 20 min.

**Total M3 compute:** about 4–5 h on this laptop, if the estimates hold. If the timing check comes in higher, the fallback is fewer β grid points (e.g. {0, 0.25, 0.5, 1}). That choice would be made before any C2 result is seen.

---

## 8. Decisions needed

1. **§2:** use the per-option form (still PuLP + CBC) for bulk runs, with the full RU MIP as a checked reference on 200 jobs per run?
2. **§3:** re-plan capacity rule: a job may move only into capacity free at the start of the hour, plus its own released slot (recommended)?
3. **§4:** add the two `mean_*` controls (β = 0) to the four variants?
4. **§5:** C1/C2 split at 2026-06-06 00:00 UTC; ρ estimated by maximum likelihood; S chosen for stability; `a = 0.2` fixed; β tuned with the declared rule?
5. **§6:** primary comparison for M4 as declared: `cvar_lp_replan` vs `base_lp`, total emissions, α = 1?
