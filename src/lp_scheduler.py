"""
M2: reimplementation of the base paper's LP scheduler (GB-only)
================================================================
Zanotto et al., GreenSys '26 (docs/base_paper.pdf). The full specification,
with page references and every assumption (G1-G20), is in docs/m2_spec.md.

In plain words
--------------
Jobs arrive one at a time. When a job arrives, the scheduler picks ONE
start hour (within the next T = 24 hours) and ONE cluster, by solving a
small 0/1 optimisation with PuLP + CBC. Jobs placed earlier are fixed,
and their nodes count against capacity.

Clusters: 1 local cluster (10 nodes) + 8 cloud clusters (20 nodes each).
All use GB carbon intensity (we have no other grids). With no on-site
solar data, the local cluster's adjusted CI equals the GB CI.

Policies compared on identical jobs, data, capacity and horizon:
  random_fifo           start on arrival, random cluster with room (carbon-agnostic)
  lp_forecast_aX        LP planned with the hourly LightGBM q50 forecast, alpha = X
  lp_actual_aX          LP planned with the actual hourly CI (the paper's "ground truth")
Every policy is SCORED with the actual hourly CI.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_config import get_data_tag  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "tables" / "m2_base_lp"

# ---- Paper parameters (docs/m2_spec.md §1.5) ----
N_MAX_LOCAL = 10            # local cluster nodes
Q_MAX_CSP = 20              # quota per cloud cluster
N_CSP = 8                   # one cloud cluster per paper region; all GB data here
CAPACITIES = np.array([N_MAX_LOCAL] + [Q_MAX_CSP] * N_CSP)
N_CLUSTERS = len(CAPACITIES)
LOCAL = 0
C_HW_KW = 8.5               # power per node
JOBS_PER_MONTH = 1000
JOB_MU, JOB_SIGMA = 6.0, 2.0
ALPHAS = (1.0, 0.5, 0.0)

# ---- Our choices (spec G5, G9, G19, G20) ----
HORIZON_T = 24              # hours; limited by the hourly forecaster
TIE_EPS = 1e-6
DAYS_PER_MONTH = 365.25 / 12
JOB_SEED = 20260923
FIFO_SEED = 20260924
ONE_HOUR = pd.Timedelta(hours=1)


# ----------------------------------------------------------------------
# Job generator (spec G3, G4, G15, G19)
# ----------------------------------------------------------------------
def _truncated_rounded_normal(rng, size, low, high, mu=JOB_MU, sigma=JOB_SIGMA):
    """Normal(mu, sigma) rounded to integers; values outside [low, high] are redrawn."""
    values = np.rint(rng.normal(mu, sigma, size)).astype(int)
    bad = (values < low) | (values > high)
    while bad.any():
        values[bad] = np.rint(rng.normal(mu, sigma, bad.sum())).astype(int)
        bad = (values < low) | (values > high)
    return values


def generate_jobs(first_hour: pd.Timestamp, last_hour: pd.Timestamp, seed: int = JOB_SEED,
                  jobs_per_month: float = JOBS_PER_MONTH, horizon: int = HORIZON_T,
                  window_start: pd.Timestamp | None = None) -> pd.DataFrame:
    """
    Arrivals uniform over [window_start, last_hour + 1h - horizon], so each
    job's whole planning window (decision hour o .. o + horizon - 1) lies
    inside [first_hour, last_hour]. o = arrival rounded UP to the hour.
    Job count = jobs_per_month x window length in months (rounded).
    Durations D in [1, horizon] hours, nodes N in [1, largest capacity].
    """
    window_start = first_hour if window_start is None else window_start
    window_end = last_hour + ONE_HOUR - horizon * ONE_HOUR
    span = (window_end - window_start).total_seconds()
    if span <= 0:
        raise ValueError("Period too short for the planning horizon.")
    n_jobs = int(round(jobs_per_month * span / 86400 / DAYS_PER_MONTH))

    rng = np.random.default_rng(seed)
    arrivals = window_start + pd.to_timedelta(np.sort(rng.uniform(0, span, n_jobs)), unit="s")
    nodes = _truncated_rounded_normal(rng, n_jobs, 1, int(CAPACITIES.max()))
    duration = _truncated_rounded_normal(rng, n_jobs, 1, horizon)
    decision = pd.DatetimeIndex(arrivals).ceil("h")
    decision = decision.where(decision >= first_hour, first_hour)
    return pd.DataFrame({"job_id": np.arange(1, n_jobs + 1), "arrival_time": arrivals,
                         "decision_hour": decision, "nodes": nodes, "duration_h": duration})


# ----------------------------------------------------------------------
# Local-cluster adjusted CI (paper Eq. 1-2, corrected per spec G1)
# ----------------------------------------------------------------------
def local_adjusted_ci(grid_ci, n_job, n_local_allocated, p_gen_kw=0.0, ef_gen=0.0,
                      c_hw_kw=C_HW_KW):
    """
    %P_grid = max{C_hw(N + N_l) - P_gen, 0} / (C_hw(N + N_l))   (printed with min; see G1)
    CI_l    = %P_grid * CI_grid + (1 - %P_grid) * EF_gen
    With p_gen_kw = 0 (M2: no solar data) this returns grid_ci unchanged.
    """
    demand = c_hw_kw * (n_job + np.asarray(n_local_allocated, dtype=float))
    share_grid = np.maximum(demand - np.asarray(p_gen_kw, dtype=float), 0.0) / demand
    return share_grid * np.asarray(grid_ci, dtype=float) + (1.0 - share_grid) * ef_gen


# ----------------------------------------------------------------------
# One job: options, objective, solvers
# ----------------------------------------------------------------------
def feasible_options(nodes: int, duration: int, plan_ci: np.ndarray, used: np.ndarray,
                     capacities: np.ndarray = CAPACITIES, horizon: int = HORIZON_T):
    """
    plan_ci, used: arrays (clusters x horizon) for hours o .. o+horizon-1.
    Returns [(t, j, c_tj)] for every start offset t (0..horizon-duration)
    and cluster j where the job fits for all its hours. c_tj = summed planned CI.
    """
    options = []
    for j in range(len(capacities)):
        for t in range(horizon - duration + 1):
            if np.all(used[j, t:t + duration] + nodes <= capacities[j]):
                options.append((t, j, float(plan_ci[j, t:t + duration].sum())))
    return options


def objective_coefficients(options, alpha: float, horizon: int = HORIZON_T,
                           n_clusters: int = N_CLUSTERS):
    """
    Spec §1.3a:  alpha*E_bar - (1-alpha)*L_bar + eps*(t*J + j)/(J*T),  J = n_clusters
    E_bar = c_tj / max c over the feasible options; L_bar = 1 on the local cluster.
    Tie-break (G9, corrected after the first test run): earliest start first,
    then local (j = 0) before cloud, then lowest cloud index -- so a job never
    waits while a cluster with room is idle.
    Returns [(t, j, coefficient)].
    """
    if not options:
        return []
    c_max = max(abs(c) for _, _, c in options) or 1.0
    return [(t, j, alpha * (c / c_max) - (1 - alpha) * (1.0 if j == LOCAL else 0.0)
             + TIE_EPS * (t * n_clusters + j) / (n_clusters * horizon))
            for t, j, c in options]


def solve_lp(coefficients):
    """The paper's model for one job, solved with PuLP + CBC. Returns (t, j)."""
    import warnings

    import pulp

    # PuLP 3.3 warns that this API changes in PuLP 4.0; requirements.txt pins 3.3.2.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        prob = pulp.LpProblem("one_job", pulp.LpMinimize)
        x = {(t, j): pulp.LpVariable(f"x_{t}_{j}", cat="Binary") for t, j, _ in coefficients}
        prob += pulp.lpSum(coef * x[(t, j)] for t, j, coef in coefficients)
        prob += pulp.lpSum(x.values()) == 1
        status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"CBC status {pulp.LpStatus[status]}")
    chosen = [key for key, var in x.items() if var.value() is not None and var.value() > 0.5]
    if len(chosen) != 1:
        raise RuntimeError(f"Expected exactly one chosen option, got {chosen}")
    return chosen[0]


def solve_brute_force(coefficients):
    """Same model by trying every option. Used only to check the LP."""
    t, j, _ = min(coefficients, key=lambda option: option[2])
    return t, j


# ----------------------------------------------------------------------
# Simulation over many jobs
# ----------------------------------------------------------------------
class Grid:
    """Hourly actual CI plus a node-usage ledger per cluster and hour."""

    def __init__(self, hourly_ci: pd.Series, capacities: np.ndarray = CAPACITIES):
        if hourly_ci.isna().any():
            raise ValueError("Hourly CI has missing values; refusing to simulate on them.")
        self.hours = pd.DatetimeIndex(hourly_ci.index)
        self.ci = hourly_ci.to_numpy(dtype=float)
        self.capacities = np.asarray(capacities)
        self.used = np.zeros((len(capacities), len(self.ci)), dtype=int)
        self._pos = {h: i for i, h in enumerate(self.hours)}

    def pos(self, hour: pd.Timestamp) -> int:
        return self._pos[hour]

    def place(self, j: int, start: int, duration: int, nodes: int):
        if start + duration > self.used.shape[1]:
            raise ValueError("Job runs past the end of the data.")
        self.used[j, start:start + duration] += nodes
        if (self.used[j, start:start + duration] > self.capacities[j]).any():
            raise AssertionError("Capacity exceeded -- this is a bug.")

    def actual_emissions_g(self, j: int, start: int, duration: int, nodes: int) -> float:
        """Paper's Em = sum over the D hours of N * C_hw * CI (spec G2, G13), in gCO2."""
        ci = self.ci[start:start + duration]
        if j == LOCAL:  # Eq. 1-2 with the job itself included; P_gen = 0 in M2
            ci = local_adjusted_ci(ci, nodes, self.used[LOCAL, start:start + duration] - nodes)
        return float(nodes * C_HW_KW * ci.sum())


def _record(policy, alpha, job, j, offset, grid, start, solve_s=np.nan):
    return {"policy": policy, "alpha": alpha, "job_id": int(job.job_id),
            "decision_hour": job.decision_hour, "nodes": int(job.nodes),
            "duration_h": int(job.duration_h), "cluster": int(j),
            "start_offset_h": int(offset), "start_hour": grid.hours[start],
            "emissions_g": grid.actual_emissions_g(j, start, int(job.duration_h), int(job.nodes)),
            "solve_s": solve_s}


def run_lp(jobs: pd.DataFrame, hourly_ci: pd.Series, alpha: float, plan: str,
           forecast_q50: pd.DataFrame | None = None, horizon: int = HORIZON_T,
           check_brute_force: bool = True, label: str | None = None):
    """
    plan = "actual": plan with the true hourly CI (ground truth).
    plan = "forecast": plan with forecast_q50.loc[decision_hour, h], h = 0..horizon-1.
    Returns (records DataFrame, number of jobs where CBC and brute force disagree).
    """
    grid = Grid(hourly_ci)
    label = label or f"lp_{plan}_a{alpha:g}"
    records, disagreements = [], 0
    for job in jobs.itertuples(index=False):
        p = grid.pos(job.decision_hour)
        window = slice(p, p + horizon)
        if plan == "actual":
            gb = grid.ci[window]
        elif plan == "forecast":
            gb = forecast_q50.loc[job.decision_hour, list(range(horizon))].to_numpy(dtype=float)
        else:
            raise ValueError(f"plan must be 'actual' or 'forecast', got {plan!r}")
        used = grid.used[:, window]
        plan_ci = np.tile(gb, (N_CLUSTERS, 1))
        plan_ci[LOCAL] = local_adjusted_ci(gb, job.nodes, used[LOCAL])

        options = feasible_options(int(job.nodes), int(job.duration_h), plan_ci, used,
                                   grid.capacities, horizon)
        if not options:
            raise RuntimeError(f"Job {job.job_id}: no feasible start within {horizon} h.")
        coefs = objective_coefficients(options, alpha, horizon)
        t0 = time.perf_counter()
        t, j = solve_lp(coefs)
        solve_s = time.perf_counter() - t0
        if check_brute_force and (t, j) != solve_brute_force(coefs):
            disagreements += 1
        grid.place(j, p + t, int(job.duration_h), int(job.nodes))
        records.append(_record(label, alpha, job, j, t, grid, p + t, solve_s))
    return pd.DataFrame(records), disagreements


def run_random_fifo(jobs: pd.DataFrame, hourly_ci: pd.Series, seed: int = FIFO_SEED):
    """
    Spec G11: in arrival order, start at the decision hour on a cluster
    chosen uniformly at random among those with room for the whole job.
    If none has room, try the next hour.
    """
    rng = np.random.default_rng(seed)
    grid = Grid(hourly_ci)
    records = []
    for job in jobs.itertuples(index=False):
        p, n, d = grid.pos(job.decision_hour), int(job.nodes), int(job.duration_h)
        offset = 0
        while True:
            start = p + offset
            if start + d > len(grid.ci):
                raise RuntimeError(f"Job {job.job_id}: no room before the end of the data.")
            fits = [j for j in range(N_CLUSTERS)
                    if np.all(grid.used[j, start:start + d] + n <= grid.capacities[j])]
            if fits:
                break
            offset += 1
        j = int(rng.choice(fits))
        grid.place(j, start, d, n)
        records.append(_record("random_fifo", np.nan, job, j, offset, grid, start))
    return pd.DataFrame(records)


# ----------------------------------------------------------------------
# Summaries
# ----------------------------------------------------------------------
def summarise(results: pd.DataFrame, baseline: str = "random_fifo") -> pd.DataFrame:
    base = results.loc[results["policy"] == baseline, "emissions_g"].sum()
    rows = []
    for policy, g in results.groupby("policy", sort=False):
        total = g["emissions_g"].sum()
        rows.append({"policy": policy, "alpha": g["alpha"].iloc[0], "n_jobs": len(g),
                     "total_tCO2": total / 1e6,
                     "pct_reduction_vs_random_fifo": 100 * (1 - total / base),
                     "local_job_share": float((g["cluster"] == LOCAL).mean()),
                     "mean_start_delay_h": float(g["start_offset_h"].mean()),
                     "mean_solve_s": float(g["solve_s"].mean()) if g["solve_s"].notna().any()
                     else np.nan})
    return pd.DataFrame(rows)


def comparison_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Published vs reproduced (GB-only) vs not applicable yet (CLAUDE.md rule 6)."""
    s = summary.set_index("policy")
    red = s["pct_reduction_vs_random_fifo"]
    tot = s["total_tCO2"]

    def inc(plan):  # % emission increase of alpha=0.5 vs alpha=1
        return 100 * (tot[f"lp_{plan}_a0.5"] / tot[f"lp_{plan}_a1"] - 1)

    na = ""
    rows = [
        ("Emission reduction vs random FIFO, LP with forecast, alpha=1 (%)",
         "27.7 (Chronos 2, 8 regions)", f"{red['lp_forecast_a1']:.2f} (LightGBM q50, GB only)", na),
        ("Emission reduction vs random FIFO, LP with ground truth, alpha=1 (%)",
         "32.09 (8 regions)", f"{red['lp_actual_a1']:.2f} (GB only)", na),
        ("Forecast LP reduction / ground-truth LP reduction (%)",
         f"{100 * 27.7 / 32.09:.1f} (derived from 27.7 and 32.09)",
         f"{100 * red['lp_forecast_a1'] / red['lp_actual_a1']:.1f}", na),
        ("Emission increase, alpha=0.5 vs alpha=1 (%)",
         "9.97 (CI source not stated)",
         f"forecast LP {inc('forecast'):.2f}; ground-truth LP {inc('actual'):.2f}", na),
        ("Per-region emission breakdown (Fig. 3b)", "shown as a figure only", "",
         "only one grid (GB); all clusters share the same CI"),
        ("Solar-adjusted local CI (Eq. 1-2, Fig. 2)", "used (Electricity Maps solar)", "",
         "no on-site solar data; P_gen = 0, so local CI = GB CI"),
        ("Forecaster MSE (Table 1)", "176.58 (Chronos 2, 96 h, 8 regions, 2023)", "",
         "different data, regions and horizon; our hourly MSE is in "
         "hourly_forecast_by_horizon.csv"),
        ("Planning horizon T (h)", "not stated", f"{HORIZON_T}", na),
        ("Jobs", "1000 per month (period not stated)",
         f"{int(s['n_jobs'].iloc[0])} over the M1 test period (summer 2026)", na),
    ]
    return pd.DataFrame(rows, columns=["metric", "published", "reproduced_gb_only",
                                       "not_applicable_yet"])


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main(time_only: int | None = None):
    from forecasting import load_clean_data
    from hourly_forecasting import forecast_tables, m1_boundaries, to_hourly

    tag = get_data_tag()
    df30 = load_clean_data()
    hourly = to_hourly(df30)
    _, test_start, test_end = m1_boundaries(df30)
    hourly_ci = hourly.set_index("timestamp")["carbon_intensity"]
    hourly_ci = hourly_ci[(hourly_ci.index >= test_start)
                          & (hourly_ci.index + ONE_HOUR <= test_end + pd.Timedelta(minutes=30))]
    first_hour, last_hour = hourly_ci.index.min(), hourly_ci.index.max()

    jobs = generate_jobs(first_hour, last_hour, window_start=test_start)
    origins = pd.DatetimeIndex(jobs["decision_hour"].unique())
    q50 = forecast_tables(hourly, origins, tag=tag)["q50"]
    print(f"Test hours: {first_hour} -> {last_hour} ({len(hourly_ci)} h); jobs: {len(jobs)}")

    if time_only:
        sub = jobs.head(time_only)
        _, dis = run_lp(sub, hourly_ci, 1.0, "forecast", q50)
        recs, _ = run_lp(sub, hourly_ci, 1.0, "forecast", q50)
        per = recs["solve_s"].to_numpy()
        n_solves = len(jobs) * 2 * len(ALPHAS)
        print(f"{len(per)} solves (second pass): mean {per.mean():.3f} s, "
              f"median {np.median(per):.3f} s, max {per.max():.3f} s; "
              f"brute-force disagreements: {dis}")
        print(f"Estimated full run: {n_solves} solves x {per.mean():.3f} s "
              f"= {n_solves * per.mean() / 60:.1f} min")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    jobs.to_csv(RESULTS_DIR / f"jobs_seed{JOB_SEED}.csv", index=False)
    all_results = [run_random_fifo(jobs, hourly_ci)]
    disagreements = {}
    t_start = time.perf_counter()
    for plan in ("forecast", "actual"):
        for alpha in ALPHAS:
            label = f"lp_{plan}_a{alpha:g}"
            print(f"Running {label} ...", flush=True)
            recs, dis = run_lp(jobs, hourly_ci, alpha, plan, q50 if plan == "forecast" else None,
                               label=label)
            all_results.append(recs)
            disagreements[label] = dis
    elapsed = time.perf_counter() - t_start

    results = pd.concat(all_results, ignore_index=True)
    results.to_csv(RESULTS_DIR / "per_job_results.csv", index=False)
    summary = summarise(results)
    summary["brute_force_disagreements"] = summary["policy"].map(disagreements)
    summary.to_csv(RESULTS_DIR / "summary.csv", index=False)
    comparison = comparison_table(summary)
    comparison.to_csv(RESULTS_DIR / "published_vs_reproduced.csv", index=False)

    import pulp
    info = {"data_tag": tag, "job_seed": JOB_SEED, "fifo_seed": FIFO_SEED,
            "horizon_T_h": HORIZON_T, "alphas": list(ALPHAS), "capacities": CAPACITIES.tolist(),
            "c_hw_kw": C_HW_KW, "jobs_per_month": JOBS_PER_MONTH,
            "job_mu_sigma": [JOB_MU, JOB_SIGMA], "tie_eps": TIE_EPS, "p_gen_kw": 0.0,
            "n_jobs": len(jobs), "test_hours": [str(first_hour), str(last_hour)],
            "forecaster": "hourly LightGBM q50, results/models/m2_hourly",
            "pulp_version": pulp.__version__, "lp_wall_time_s": round(elapsed, 1),
            "brute_force_disagreements": disagreements}
    (RESULTS_DIR / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    print("\n" + summary.round(3).to_string(index=False))
    print("\n" + comparison.to_string(index=False))
    print(f"\nLP wall time: {elapsed / 60:.1f} min. Results in {RESULTS_DIR}")


if __name__ == "__main__":
    main(time_only=int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[1] == "--time" else None)
