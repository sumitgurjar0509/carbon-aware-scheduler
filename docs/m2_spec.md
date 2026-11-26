# M2 specification: reimplementing the base paper's LP scheduler

Status: **draft for approval**. No code yet.

Source: Zanotto, Padovani, Iacca, Fiore, *Carbon Intensity Forecasting with
On-Site Renewables for Sustainable AI Workload Scheduling*, GreenSys '26,
DOI 10.1145/3802973.3804455 (`docs/base_paper.pdf`).

Page numbers below are the **printed proceedings pages 13–18** (PDF page
1 = p. 13). "§" means section, "Eq." means equation number in the paper.
The text was extracted with `pdftotext`. The figure axes could not be read
reliably, so no number here is taken from a figure.

---

## 1. What the paper specifies

### 1.1 System model (§3.1, p. 14–15)

| Item | What the paper says | Where |
|---|---|---|
| Clusters | `J = {0, 1, …, J_max}`; `l = 0` is the local (on-premise) cluster, the rest are public cloud (CSP) clusters | §3.1, p. 14 |
| Cluster CI | Each cluster `j` has carbon intensity `CI_j(t)` of its region | §3.1, p. 14 |
| Local capacity | `N_max` = maximum reservable nodes in the local cluster | §3.1, p. 14 |
| CSP capacity | `Q_max_j` = usage quota (max reservable nodes) at CSP cluster `j` | §3.1, p. 14 |
| Job model | Non-preemptive, runs to completion, runs on `N` nodes of **one** cluster, exclusive node reservation, uniform load, constant power | §3.1, p. 14 |
| Job definition | `W := <D, N>`: `D` = wall-time (hours), `N` = number of nodes | §3.1, p. 14 |
| Arrival handling | "Workloads are scheduled upon arrival without being queued, although their execution start may be delayed by the scheduler." One job is planned at a time; jobs already placed are stored as `N_l(τ)`, `N_j(τ)` | §3.1, p. 14; §3.3, p. 16 |

### 1.2 Local-cluster adjusted CI with on-site solar (§3.1, Eq. 1–2, p. 15)

```
%P_grid(t) = min{ C_hw·(N + N_l(t)) − P_gen(t), 0 } / ( C_hw·(N + N_l(t)) )      (Eq. 1)
CI_l(t)   = %P_grid(t)·CI_j(t) + (1 − %P_grid(t))·EF_gen                         (Eq. 2)
```

- `P_gen(t)` is the forecast on-site generator power. All of it is used by the cluster, and the grid covers the rest.
- `CI_j(t)` is the grid CI of the local cluster's region.
- `EF_gen` is the generator's emission factor in gCO2eq/kWh.
- `C_hw` is the power per node.

As printed, Eq. 1 cannot be right: `min{…, 0}` is never positive, so the grid share would be ≤ 0. It must mean `max{…, 0}` (see G1).

### 1.3 Decision variables and objective (§3.3, Eq. 3–5, p. 16)

| Item | What the paper says | Where |
|---|---|---|
| Time | `t ∈ [0, T]`, **hourly** steps of a scheduling horizon of length `T` | §3.3, p. 16 |
| Variable | `x_tj` binary, 1 if the job (one job `W`) starts at time `t` on cluster `j` | §3.3, p. 16 |
| Emission cost | `E(t,j) = x_tj · c_tj`, with `c_tj = Σ_{τ=t}^{t+D} CI_j(τ)` | Eq. 3, p. 16 |
| Where CI comes from | "The Forecaster component … is used to build the CI data matrix"; `CI_l` is the adjusted CI from Eq. 2 | §3.3, p. 16 |
| Local-usage reward | `L(t,j) = x_tj · a_tj`, with `a_tj = N` if `j = l`, else 0 | Eq. 4, p. 16 |
| Objective | `t*, j* = argmin_{t,j} { α·Ē(t,j) − (1−α)·L̄(t,j) }` | Eq. 5, p. 16 |
| Normalisation | `Ē`, `L̄` are Eq. 3 and Eq. 4 "normalized over their maximum value", so both are in [0, 1] | §3.3, p. 16 |
| `α` | `α ∈ [0, 1]`; experiments use `α ∈ {1, 0.5, 0}` | Eq. 5, p. 16; §4.2, p. 17 |

### 1.4 Constraints (§3.3, p. 16)

1. `Σ_t Σ_j x_tj = 1`: each job is placed exactly once. The summation symbols are garbled in the extracted text; the meaning is clear from the wording.
2. `t + D ≤ T`: the job must finish within the horizon.
3. `∀τ ∈ [t, t+D]: N + N_l(τ) ≤ N_max`: local capacity.
4. `∀τ ∈ [t, t+D]: N + N_j(τ) ≤ Q_max_j, ∀j ≠ l`: CSP quota.

Solver: PuLP with CBC (§3.3–4, p. 17).

### 1.5 Experimental setup (§3.1, §4.2, p. 15 and 17)

| Item | Value | Where |
|---|---|---|
| Regions | 8: AU-NSW, DE, GB, IT-NO, JP, KR, US-TEX-ERCO, US-MIDA-PJM (one CSP cluster per region) | §3.1, p. 15; §4.2, p. 17 |
| Data | Electricity Maps, year **2023**, hourly | §3.1, p. 15; §3.2, p. 15 |
| Jobs | "a total of 1000 jobs arriving per month" | §4.2, p. 17 |
| Arrival times | "uniform distribution over the whole period" | §4.2, p. 17 |
| `N` and `D` | "normal distributions with parameters μ = 6 and σ = 2"; `D` in **hours** | §4.2, p. 17 |
| Local cluster | `N_max = 10` nodes, with a solar generator modelled from Electricity Maps solar production data | §4.2, p. 17 |
| CSP clusters | `Q_max_j = 20` for every `j` | §4.2, p. 17 |
| Power per node | `C_hw = 8.5 kW`, constant over the job | §4.2, p. 17 |
| Reported emissions | `Em(D,N,t,j) = Σ_{τ=t}^{t+D} N · C_hw · CI_j(τ)` | §4.2, p. 17 |
| Forecaster | Chronos 2, zero-shot, context 512 hourly steps, output 96 hourly steps | §3.2, p. 16; §4.2, p. 17 |

### 1.6 Baseline and headline figures (§4.2, p. 17)

- **Random FIFO**: "assigns workloads to randomly chosen regions, respecting resource constraints".
- **Optimised, forecast**: the LP built on Chronos 2 forecasts. It "reduces emissions by **27.7%** compared to the random FIFO scheduler".
- **Optimised, ground truth**: the LP built on actual Electricity Maps CI, described as "a theoretical upper bound". It gives **32.09%**.
- **α trade-off**: with `α = 0.5`, most jobs go to the local cluster, with "an increase of only **9.97%** of emissions compared to the α = 1 profile".
- The abstract rounds the 27.7% result to "27%" (p. 13).

The paper does not give the formula behind the percentages. The natural
reading is `1 − total_emissions(LP) / total_emissions(random FIFO)`, summed
over all jobs (see G12).

---

## 2. Gaps and assumptions

Every item below is something the paper does not state, or states
inconsistently. Each one will be copied into `docs/assumptions.md` when
the code is written.

| # | Gap | Proposed assumption | Why |
|---|---|---|---|
| G1 | Eq. 1 uses `min{…, 0}`, which makes the grid share ≤ 0 | Read it as `max{…, 0}` | Only reading that gives a share in [0, 1] |
| G2 | `Σ_{τ=t}^{t+D}` has `D+1` terms, but the job runs `D` hours | Sum over `D` hourly slots `τ = t, …, t+D−1`; same for constraints 3–4 and `Em` | Energy then equals `N · C_hw · D`, which matches "constant power for the duration" |
| G3 | `N` and `D` come from a normal distribution, but must be whole numbers ≥ 1 for hourly slots; rounding is not stated | Draw from `Normal(6, 2)`, round to the nearest integer, and redraw any value < 1 | Keeps the stated mean and spread; redrawing is rare (≈0.3% of draws per variable) |
| G4 | "normal distributions with μ = 6 and σ = 2": one shared pair, or the same values for both? | Independent draws, both `Normal(6, 2)` | Simplest reading |
| G5 | Planning horizon `T` is not given (forecasts are 96 h; "increasing the delay time window" is mentioned as future analysis, p. 18) | `T = 24` hours from the job's arrival hour, the most our forecaster covers (see §3.3) | We cannot plan further than our forecasts |
| G6 | What `t = 0` is, and whether the current hour can be used | Arrival time is rounded **up** to the next whole hour `o`. Forecasts are made at `o` with data up to `o`. Candidate starts are `o, …, o+T−D`, covered by horizons 1…24 (one-hour shift; detail fixed in Phase B) | Never uses information from after the decision time |
| G7 | Normalisation "over their maximum value": the maximum over what? | Per job: `Ē = c_tj / max c` over that job's feasible `(t, j)`, and `L̄ = a_tj / N`. If no local option is feasible, `L̄ = 0` | Gives [0, 1] per job, as the paper requires |
| G8 | Eq. 3's objective leaves out `N · C_hw`; `Em` includes it | Use `c_tj` in the objective as written. This changes nothing, because `N · C_hw` is the same for every option of one job (with no solar, see §3.2) | Faithful to Eq. 3 |
| G9 | Ties: several `(t, j)` can have exactly the same cost (common for us: all clusters share GB data, see §3.1) | Deterministic tie-break: lowest cluster index (local first), then earliest start. Implemented as tiny penalty terms (≤ 1e-6 of the objective range), and tested against brute-force enumeration | Results must be reproducible; CBC's own choice among ties is not guaranteed |
| G10 | How long a forecast is reused ("a configurable time validity", p. 15) | A fresh forecast at each arrival hour; jobs arriving in the same hour share it | Smallest possible validity; no extra parameter |
| G11 | Random FIFO details: does it include the local cluster? When does a job start? | Each job, in arrival order, picks uniformly at random among **all** clusters (local included) that have room for `[o, o+D)`, and starts at `o`. If none has room, it starts at the earliest hour where one does. Fixed seed | "FIFO" = start on arrival; "respecting resource constraints" = only clusters with room |
| G12 | Formula behind 27.7% / 32.09% | `1 − Σ Em(LP) / Σ Em(random FIFO)` over all jobs, using the **actual** CI for both | The usual definition; matches our `pct_carbon_saved` |
| G13 | Is the forecast LP's emission scored with forecast or actual CI? | Always actual CI, whichever CI the scheduler planned with | A job burns the real grid mix (same rule as our simulator) |
| G14 | Which `α` produced 27.7% / 32.09%? | `α = 1` (pure emissions); `α ∈ {0, 0.5, 1}` also reported as in Fig. 3c | Only `α = 1` matches "carbon-aware vs carbon-agnostic" |
| G15 | Length of the simulated period (one month? all of 2023?) | Our whole test period (≈ 72.8 days), with 1000 jobs per 30.44-day month over the ≈ 71.8-day arrival window, i.e. about 2,359 jobs | See §3.4 |
| G16 | Each job is "an LP", but `x_tj` is binary, so it is really an integer program | Solve as written (binary variables, CBC). For one job with a single "place it once" constraint, the LP relaxation already has a 0/1 optimum, so the paper's "LP / global optimum" statement holds | Faithful; easy to check in a test |
| G17 | `EF_gen` (solar emission factor) is not given | Not needed: we have no on-site generator (§3.2). Eq. 1–2 are implemented with `P_gen = 0` and tested on toy numbers | Keeps the code ready if solar data is added later |
| G18 | Hourly CI from 30-min data | Hour `H` = mean of the `H:00` and `H:30` settlement periods (CLAUDE.md M2). The dataset starts at 03:30, so the first, incomplete hour is dropped | As CLAUDE.md asks |

None of these assumptions is tuned. `α` comes from the paper, `T` from
our forecaster's limit, and both seeds are fixed in advance. So nothing
touches the test period before the single final run (CLAUDE.md rule 1).

---

## 3. What we cannot reproduce exactly

The aim is to reimplement **the method** faithfully. Our numbers **will
not be comparable** to 27.7% / 32.09%, and the write-up must say so plainly.

### 3.1 One grid (GB) instead of eight regions

- **Difference**: we have only GB data from the UK Carbon Intensity API (tag `20250923_20260923`, 2025–2026), not Electricity Maps 2023 for 8 regions. The paper's savings come from **moving jobs between regions** (very different grids, e.g. GB vs KR) as well as **delaying** them. With one grid, region-shifting saves nothing, and only time-shifting remains.
- **Proposal**:
  - Keep the paper's cluster layout and capacities exactly: 1 local cluster (`N_max = 10`) plus 8 CSP clusters (`Q_max = 20` each), **all using GB carbon intensity**.
  - The LP, its size, its capacity constraints and the job load are then identical to the paper. Only the CI data differs.
  - Report this as "**GB-only, time-shifting only**", in its own clearly labelled column, never next to 27.7% as if comparable.
- **Alternative considered and rejected**: synthetic CI for the other 7 regions. That would be made-up data.
- **Capacity check** (paper parameters, plain arithmetic):
  - Demand ≈ 1000 jobs × 6 nodes × 6 h = 36,000 node-hours per month.
  - Capacity ≈ (10 + 8×20) nodes × 730 h ≈ 124,100 node-hours.
  - Average usage is therefore about 29%, so placing jobs should rarely fail.
  - With only one GB cloud cluster (30 nodes), demand would exceed capacity. That is why all 8 CSP clusters are kept.

### 3.2 No on-site solar data

- **Difference**: the local cluster's adjusted CI (Eq. 1–2) needs on-site solar output, which we do not have.
- **Proposal**:
  - Set `P_gen(t) = 0`, so `CI_l(t) = CI_GB(t)` exactly. The local cluster is then no greener than the cloud clusters.
  - The `α` trade-off then only changes *where* jobs run, not their emissions, unless capacity forces a later start. That will show in the `α = 0 / 0.5` results, and we should not expect anything like the paper's 9.97%.
  - Not proposed: the API's national solar share is the whole country's mix, not a local panel. Using it would be a different model, not a reproduction.

### 3.3 LightGBM 1–24 h instead of Chronos 2 at 96 h

- **Difference**: our forecaster covers 24 hourly horizons (and predicts 30-min values), not 96 h.
- **Proposal**:
  - Set `T = 24` (see G5), so each job can be delayed at most `24 − D` hours. Their horizon may be longer (not stated), and this is part of why our numbers differ.
  - **Decision needed — hourly forecasts.** Our current models predict a single 30-min value `h` hours ahead, not the hourly average the LP needs. Two options:
    - **(a) Recommended.** Train a separate set of hourly-target LightGBM models (same features, target = the hourly mean, horizons 1–24). Use the same chronological 60/20/20 split and CQR calibration as M1. Save them to `results/models/m2_hourly/` so the M1 models stay untouched. M3 also needs calibrated hourly quantiles, so this work is reused there. Cost: one more training run (the 24×3-model run took about a minute in M1).
    - **(b)** Reuse the M1 models and use the 30-min forecast at `o+h` as the whole hour's value. No retraining, but it ignores the second half-hour, and M3 would need new models anyway.
- The "LP with ground truth" run uses the actual hourly CI and needs no forecaster.

### 3.4 Period: test split only, not a full year

- **Difference**: the paper's forecasters are zero-shot, so the whole of 2023 is out-of-sample. Ours are trained, so only the final 20% of our year is unseen.
- **Proposal**:
  - Simulate only the test period, 2026-07-12 08:30 to 2026-09-23 03:30 UTC (≈ 72.8 days, summer only).
  - Keep the paper's rate of 1000 jobs per month, with arrivals uniform over `[test_start, test_end − T]` so every job's window lies in the test period. That window is ≈ 71.8 days, giving ≈ 2,359 jobs.
  - All policies run on the identical jobs, CI, capacities and horizon (rule 2). The job seed and FIFO seed are written into every results file (rule 3).
  - Summer-only results are not an annual figure, and the write-up will say so. M4 will extend this to more periods and seeds.

### 3.5 Reporting (rule 6)

`results/tables/m2_base_lp/` will hold one summary table with
**separate, labelled columns**:

- `published` (27.7, 32.09, 9.97, from the paper);
- `reproduced_gb_only` (our method on GB data, LightGBM forecast vs ground truth vs random FIFO, for each `α`).

There will be no column comparing the two directly.

---

## 4. Problem size and solve time

### 4.1 Per-job formulation (what the paper does)

One optimisation per arriving job:

| | Paper (estimated) | Ours (proposed) |
|---|---|---|
| Clusters | 9 | 9 |
| Candidate starts per job | up to `T − D + 1` (T unknown) | `24 − D + 1` ≈ 19 for `D = 6` |
| Binary variables | 9 × (T − D + 1) | ≈ 171 (max 9 × 24 = 216) |
| Equality constraints | 1 | 1 |
| Capacity constraints | ≤ 9 × (T−D+1) × D | ≈ 171 × 6 ≈ 1,000 one-variable rows |

- **How capacity is handled**: `N_l(τ)` and `N_j(τ)` are already known when a job arrives, so each capacity constraint involves just one variable. A start `(t, j)` either fits for its whole run or does not. Starts that don't fit are fixed to 0 instead of adding rows: same feasible set, smaller model.
- **Solve time estimate** (not measured; PuLP is not installed yet):
  - CBC on a ~200-variable problem with one equality should take a few milliseconds.
  - On Windows, the larger cost is PuLP writing a model file and starting `cbc.exe`, about 0.05–0.3 s per call.
  - Full M2 run: ≈ 2,359 jobs × 6 LP runs (forecast and ground truth, each for α ∈ {1, 0.5, 0}) ≈ 14,200 solves. That is roughly **12–70 minutes**, plus the random FIFO baseline, which needs no solver.
  - These are estimates. Phase B will time 100 solves first and report the measured figure.

### 4.2 A whole month as one optimisation (not proposed)

- **Size**: placing all ~1,000 jobs of a month jointly would need about 1,000 × 171 ≈ 171,000 binary variables. The capacity rows (9 clusters × ~744 hours) would each contain many jobs, giving about 1 million non-zeros.
- **Why not**: this is a hard scheduling problem, and CBC's time is unpredictable (minutes to hours). It is also **not what the paper does**: the paper plans each job on arrival, with earlier jobs fixed.
- **Fallback** (only if 4.1 turns out too slow): plan in batches of the jobs arriving in each hour. That gives about 1–2 jobs per batch, only a few hundred variables, and stays close to "schedule on arrival". It would be listed as a deviation.

**Recommendation**: per-job planning, as in the paper.
- **Correctness test**: a brute-force check (try every `(t, j)` and take the lowest objective, with the same tie-break) must give the same answer as CBC on every job.
- **Speed**: the measured time from the first 100 solves is reported before the full run.

---

## 5. Planned Phase B deliverables (after approval)

- `src/lp_scheduler.py`:
  - job generator (G3–G4, G15);
  - hourly CI (G18);
  - Eq. 1–2 with `P_gen = 0`;
  - per-job LP with PuLP and CBC;
  - random FIFO;
  - runners for the forecast LP and the ground-truth LP.
- Hourly forecaster, option (a) or (b) from §3.3, depending on your choice.
- `docs/assumptions.md`: G1–G18 plus the §3 decisions.
- `tests/test_lp_scheduler.py`:
  - LP equals brute force;
  - capacity is never exceeded;
  - every job is placed exactly once;
  - `t + D ≤ T`;
  - normalisation gives values in [0, 1];
  - FIFO respects capacity;
  - fixed seeds give identical outputs.
- Results in `results/tables/m2_base_lp/` with the seeds recorded.
- `pulp` added to `requirements.txt` (the only new dependency allowed).
