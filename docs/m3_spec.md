# M3 specification: uncertainty-aware LP (scenarios + CVaR + re-planning)

Status: **approved 2026-11-26** with these changes:

- §2: the equivalence argument and the exact definition of `Ē` are now written out.
- §5: ρ is estimated on **C1** forecast errors; C2 is used only for `S` and `β`; `a = 0.2` is fixed.
- §6: two pre-registered primary hypotheses, H1 and H2.
- Output folder: `results/tables/m3/`.

Phase B runs in two stages. Stage 1 does parameter selection on C1/C2
only and freezes the values in `results/tables/m3/frozen_params.json`.
Stage 2, the single test-period run, starts only after approval.

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
| `a` | CVaR tail fraction, **fixed at 0.2**. Convention: `CVaR_0.2` = mean cost of the **worst 20%** of scenarios, i.e. the **upper** (high-emission) tail |
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

**Normalisation (M2's G7, extended), exact definition.** For one job
solved at hour `H`, let `K` be its set of feasible options (start offsets
allowed by the remaining window and capacity, on any cluster). Then:

```
c_k^s = Σ_{τ=t_k}^{t_k+D−1} CI_{j_k}^s(H+τ)          scenario cost of option k   (gCO2/kWh · h)
c̄_k   = (1/S) Σ_{s=1}^{S} c_k^s                       mean cost of option k
C     = max_{k ∈ K} c̄_k                               one number per job and solve
Ē_k^s = c_k^s / C                                     normalised scenario cost
L̄_k   = 1 if j_k = local, else 0
```

- One constant `C` scales both the mean term and the CVaR term, so `β` trades like for like, and multiplying all costs by a constant never changes the decision.
- `C` is the largest **mean** cost, not the largest scenario cost. So `mean_s Ē_k^s ∈ (0, 1]`, as in M2. `Ē_k^s` itself can exceed 1 in bad scenarios, which is intended.
- With `S = 1` (a point path, e.g. `q50` or the actual CI), this is exactly M2's `Ē = c / max c`. That is why `base_lp` reproduces M2.
- Carbon intensity is floored at 0 and positive in practice, so `C > 0`. If `C ≤ 0` ever occurred, the run would stop with an error.

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

**Equivalent per-option form (approved for bulk runs).**

*Claim.* With exactly one binary choice per job, the RU MIP above and the
following M2-shaped problem have the same optimal option and the same
optimal value:

```
minimise  Σ_k f_k · x_k    subject to  Σ_k x_k = 1,  x_k ∈ {0,1}
f_k = α·[ (1−β)·m_k + β·CVaR_a(k) ] − (1−α)·L̄_k + tie_k
m_k = (1/S) Σ_s Ē_k^s,    CVaR_a(k) = mean of the a·S largest values of {Ē_k^s}_s
```

*Argument.*

1. **Fix the option.** Because `Σ x_k = 1` with binary `x`, every feasible point has `x = e_k` for exactly one `k`. The MIP is therefore the minimum, over `k`, of the problem with `x` fixed to `e_k`.
2. **Write out that problem.** With `x = e_k`, the constraints become `u_s ≥ Ē_k^s − η` and `u_s ≥ 0`, and only `η, u` remain free. At the optimum each `u_s` sits at its lower bound, `u_s = max(Ē_k^s − η, 0)`, so the remaining problem is
   `g_k(η) = η + (1/(a·S)) Σ_s max(Ē_k^s − η, 0)`.
3. **Apply Rockafellar–Uryasev.** `min_η g_k(η)` equals the empirical CVaR at level `a` of the `S` equally likely values `Ē_k^s`: the mean of the worst `a` share (Rockafellar & Uryasev 2000, Thm. 1, discrete case). The minimum is reached at `η` = the `(1−a)` empirical quantile. When `a·S` is a whole number, this is exactly the mean of the `a·S` largest values.
4. **Assemble.** The other terms of the objective don't depend on `η, u`. So the value at `x = e_k` is exactly `f_k`, the MIP optimum is `min_k f_k`, and the minimising option (with the same `ε` tie-break) is the same in both forms.

*Checks kept:*
- In tests, the RU MIP (CBC), the per-option form (CBC) and brute-force `argmin_k f_k` must pick the same option on random small cases.
- In every run with `β > 0` and `S > 1`, the **first 200 job solves** are also solved with the full RU MIP. Disagreements in option and in objective value (tolerance 1e-6) are counted and reported.

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

**How each parameter is chosen.** Order: ρ (on C1), then S (on C2),
then β (on C2). `a` is fixed.

1. **ρ: estimated on C1 forecast errors only.**
   - For each C1 origin whose 24 target hours all end by 2026-06-06 00:00, and each horizon `h`, compute `u_h = F_h(y_h)` (where the actual value falls in the calibrated forecast distribution) and `z_h = Φ⁻¹(u_h)`.
   - Actual values outside the bounded tails (`u = 0` or `1`) are clipped to `[1e-4, 1 − 1e-4]`, and the number clipped is reported. That count measures how often the assumed tails were too short.
   - Fit the AR(1) `ρ` by maximum likelihood on `z_0 … z_23`, treating origins as independent rows.
   - The empirical lag-1 to lag-6 correlations of `z` are reported next to `ρ^lag`.
   - **Caveat:** the C1-fitted CQR corrections are in-sample on C1. This affects the marginal widths only slightly and does not use C2 or the test period.
2. **S: chosen on C2 for Monte Carlo stability, not performance.**
   - Pick the smallest `S ∈ {100, 200, 500, 1000}` whose decisions agree with `S = 2000` for at least 99% of C2 jobs.
   - The check uses `β = 0.5` and no re-planning. (The grid's median, 0.375, is not a grid point; 0.5 is the nearest grid value above it.)
   - Fair comparison: the reference run uses `S = 2000`. At every job's decision, in that same state, the decision is recomputed with the first `S` of the same 2,000 scenarios (common random numbers).
   - `a·S` is a whole number for every candidate.
3. **a: fixed at `a = 0.2`, not tuned.**
   - It matches CQR's α.
   - At `a = 0.2`, half the tail mass (`p` in 0.8–0.9) lies inside the calibrated region. At `a = 0.1` it would be entirely in the assumed tail.
4. **β: tuned on C2 with a rule declared before any run.**
   - Grid `β ∈ {0, 0.1, 0.25, 0.5, 0.75, 1}`, separately for with and without re-planning, at α = 1.
   - C2 workload: the M2 generator with a **different seed** (`TUNE_SEED = 20260606`), 1000 jobs per month, giving ≈ 1,163 jobs, plus the FIFO seed.
   - **Selection rule:** choose the `β` with the **lowest total realised emissions** on C2.
   - Ties within 0.1% of the best go to the **smallest β** (simplest model).
   - The whole curve is reported, not just the winner.
   - If the winner is `β = 0`, that is reported as the result ("the CVaR term did not help on C2"), and the test run still includes `cvar_lp` with the best `β > 0` from the grid, **labelled as such**.

The chosen ρ, S and β (one for each re-plan setting), the C2 table that
justified them, and all seeds are written to
`results/tables/m3/frozen_params.json` and committed **before** any
test-period run. Supporting tables go to `results/tables/m3/tuning_c1/`
and `results/tables/m3/tuning_c2/`. The test period is then run
**once** with exactly those values (rule 1), and its results go to
`results/tables/m3/test/`.

---

## 6. Metrics

All metrics are computed on the test period, scored with actual CI, for
every variant at α = 1 (and α = 0.5).

| Metric | Definition |
|---|---|
| Cut vs random FIFO (%) | `100·(1 − Σ E_variant / Σ E_fifo)`, as in M2 |
| Share of actual-CI gap closed (%) | `100·(E_base − E_variant) / (E_base − E_actual)`, using totals. 0 = `base_lp`, 100 = `actual_lp`; can be negative |
| Regret rate (%) | Share of jobs whose realised emissions are higher than under the comparison policy: `base_lp`, or for H2 `base_lp_replan`. Paired by `job_id`, tolerance 1e-9 g. Also the mean regret in kgCO2 |
| CVaR₀.₁ of per-job emissions | Mean of the worst 10% of jobs. Reported in two ways: (i) per-job emission intensity, `E / (N·C_hw·D)` in gCO2/kWh (the main one, so that big jobs don't dominate); (ii) raw per-job kgCO2 |
| Mean start delay (h) | as in M2 |
| Also | Share of jobs on the local cluster; number of re-plans per job; solve times; RU-vs-per-option disagreements; scenario values floored at 0 |

**Caution.** Per-job pairing is not a clean experiment. Jobs share
capacity, so one job's change can move another job. Regret is therefore
reported as an outcome comparison, not as a measure of each decision.

**Statistical plan (to run in M4, not in M3):**

- **Unit:** a run = (period, job seed). Within one run, jobs are correlated through shared days and capacity, so the run, not the job, is the unit of analysis.
- **Pre-registered primary hypotheses** (declared 2026-11-26, before any M3 run). Both use **total emissions at α = 1**:
  - **H1:** `cvar_lp_replan` vs `base_lp`. Does the full method beat the paper's method?
  - **H2:** `cvar_lp_replan` vs `base_lp_replan`. Does uncertainty add anything beyond re-planning?
  - **Holm** correction across H1 and H2 at family-wise α = 0.05. Each test is a two-sided paired Wilcoxon signed-rank test on per-run differences.
- **Declared secondary metrics** for the same two comparisons, reported with effect sizes and bootstrap CIs and **not** used for confirmatory claims:
  - regret rate (jobs worse than the comparison policy);
  - CVaR₀.₁ of per-job emission intensity.
- **Exploratory, labelled as such:** every other variant pair, including `cvar_*` vs `mean_*` (the pure risk-term effect).
- **Intervals:** bootstrap 95% CIs over runs, and a moving-block bootstrap by day within a run to describe single-run uncertainty.
- **Runs:**
  - Several job seeds (e.g. 10) on each available period.
  - More periods need **rolling-origin retraining**: the models are trained only on the first 60% of the year, so later months are the only true out-of-sample data. This is an M4 design decision.
- M3 itself reports **one run** (test period, M2 jobs) **with no significance claims**.

---

## 7. Problem size and solve-time estimates

**Measured in Stage 1** (first 100 C2 jobs, `results/tables/m3/tuning_c2/timing.csv`):

- per-option form (CBC): 0.025 s per solve at S = 200 and S = 1000;
- full RU MIP: 0.15 s at S = 200 and 0.81 s at S = 1000;
- RU MIP vs per-option: 0 disagreements in 120 checks.

The estimates below were made before measuring and are kept for the record; they were too pessimistic for the RU MIP.

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

## 8. Decisions (resolved 2026-11-26)

1. §2: per-option form for bulk runs, with the full RU MIP checked on the first 200 solves of every run with β > 0. **Approved.**
2. §3: re-plan capacity rule as recommended. **Approved.**
3. §4: both `mean_*` (β = 0) controls. **Approved.**
4. §5: split at 2026-06-06 00:00 UTC; ρ on **C1** only; S and β on C2; `a = 0.2` fixed. **Approved with changes.**
5. §6: two pre-registered primary hypotheses, H1 and H2, with Holm correction. **Approved with changes.**
