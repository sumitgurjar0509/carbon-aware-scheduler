"""
M3: uncertainty-aware LP scheduler (docs/m3_spec.md)
=====================================================
Extends the M2 base-paper LP (src/lp_scheduler.py) with:
  - scenario costs from scenarios.py (S paths per origin hour),
  - the objective alpha*[(1-beta)*mean + beta*CVaR_a] - (1-alpha)*L_bar + tie,
    solved in the per-option form with PuLP + CBC (lp_scheduler.solve_lp),
    with the full Rockafellar-Uryasev MIP as a checked reference (spec §2),
  - optional hourly re-planning of jobs that have arrived but not started
    (spec §3, with the approved capacity rule).

Everything else (clusters, capacities, T, C_hw, tie-break, scoring with
actual CI) is taken unchanged from lp_scheduler.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lp_scheduler as lp  # noqa: E402
from scenarios import LEVELS_V1, SCENARIO_SEED, sample_paths_levels  # noqa: E402

A_CVAR = 0.2      # fixed (spec §5): CVaR_0.2 = mean of the worst 20% of scenarios (upper tail)
J = lp.N_CLUSTERS
T = lp.HORIZON_T
CVAR_JOBS_TAIL = 0.1  # metric: CVaR_0.1 over jobs


# ----------------------------------------------------------------------
# CVaR and option scoring
# ----------------------------------------------------------------------
def cvar_upper(values: np.ndarray, a: float = A_CVAR) -> np.ndarray:
    """
    values: (K, S). Mean of the a*S LARGEST values per row (worst = highest
    emissions). With S = 1 the single value is returned. For S > 1, a*S must
    be a whole number so that the empirical CVaR is exact.
    """
    values = np.asarray(values, float)
    s = values.shape[1]
    if s == 1:
        return values[:, 0].copy()
    m = a * s
    if abs(m - round(m)) > 1e-9:
        raise ValueError(f"a*S = {m} is not a whole number")
    m = int(round(m))
    return np.partition(values, s - m, axis=1)[:, s - m:].mean(axis=1)


def feasible_starts(nodes: int, duration: int, latest_offset: int, taken: np.ndarray,
                    capacities: np.ndarray = lp.CAPACITIES):
    """
    taken: (J, latest_offset + duration) nodes already in use from hour H on.
    Returns arrays (t_idx, j_idx) of every start offset 0..latest_offset and
    cluster where the job fits in all its hours.
    """
    windows = sliding_window_view(taken[:, :latest_offset + duration], duration, axis=1)
    ok = windows.max(axis=2) + nodes <= capacities[:, None]   # (J, latest_offset + 1)
    j_idx, t_idx = np.nonzero(ok)
    order = np.lexsort((j_idx, t_idx))  # earliest start, then cluster (matches tie order)
    return t_idx[order], j_idx[order]


def scenario_costs(paths: np.ndarray, nodes: int, duration: int, taken_local: np.ndarray):
    """
    paths: (S, W) planned GB CI for hours H .. H+W-1. Returns costs (J, S, W-D+1):
    summed planned CI over the job's D hours for every start offset and cluster.
    The local cluster uses Eq. 1-2 (identity while P_gen = 0).
    """
    def window_sums(x):
        c = np.concatenate([np.zeros((x.shape[0], 1)), np.cumsum(x, axis=1)], axis=1)
        return c[:, duration:] - c[:, :-duration]

    cloud = window_sums(paths)
    local = window_sums(lp.local_adjusted_ci(paths, nodes, taken_local[None, :]))
    return np.concatenate([local[None], np.repeat(cloud[None], J - 1, axis=0)], axis=0)


def option_scores(t_idx, j_idx, costs, alpha: float, beta: float, a: float = A_CVAR):
    """
    Spec §2 (exact normalisation): C = max_k mean_s c_k^s, E_bar = c / C.
    Returns dict with E (K, S), m, cvar, L, tie and f (the per-option objective).
    """
    raw = costs[j_idx, :, t_idx]                     # (K, S)
    c_bar = raw.mean(axis=1)
    big_c = c_bar.max()
    if not big_c > 0:
        raise ValueError(f"Normaliser C = {big_c} is not positive")
    e = raw / big_c
    m = e.mean(axis=1)
    cv = cvar_upper(e, a)
    local = (j_idx == lp.LOCAL).astype(float)
    tie = lp.TIE_EPS * (t_idx * J + j_idx) / (J * T)
    f = alpha * ((1 - beta) * m + beta * cv) - (1 - alpha) * local + tie
    return {"E": e, "m": m, "cvar": cv, "L": local, "tie": tie, "f": f}


def coefficient_list(t_idx, j_idx, f):
    return [(int(t), int(j), float(c)) for t, j, c in zip(t_idx, j_idx, f)]


def solve_ru_mip(scores: dict, t_idx, j_idx, alpha: float, beta: float, a: float = A_CVAR):
    """
    The full Rockafellar-Uryasev model (spec §2), solved by CBC.
    Returns ((t, j), objective value).
    """
    import warnings

    import pulp

    e = scores["E"]
    k, s = e.shape
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        prob = pulp.LpProblem("ru_cvar", pulp.LpMinimize)
        x = [pulp.LpVariable(f"x{i}", cat="Binary") for i in range(k)]
        u = [pulp.LpVariable(f"u{q}", lowBound=0) for q in range(s)]
        eta = pulp.LpVariable("eta")
        lin = alpha * (1 - beta) * scores["m"] - (1 - alpha) * scores["L"] + scores["tie"]
        obj = pulp.LpAffineExpression([(x[i], float(lin[i])) for i in range(k)])
        obj += alpha * beta * eta
        obj += pulp.LpAffineExpression([(uq, alpha * beta / (a * s)) for uq in u])
        prob += obj
        prob += pulp.lpSum(x) == 1
        for q in range(s):
            terms = [(x[i], -float(e[i, q])) for i in range(k)] + [(u[q], 1.0), (eta, 1.0)]
            prob += pulp.LpAffineExpression(terms) >= 0
        # gapRel = 0: the reference must be exact (spec §5, run hygiene).
        status = prob.solve(pulp.PULP_CBC_CMD(msg=False, gapRel=0))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"CBC status {pulp.LpStatus[status]}")
    i = int(np.argmax([v.value() for v in x]))
    return (int(t_idx[i]), int(j_idx[i])), float(pulp.value(prob.objective))


# ----------------------------------------------------------------------
# Planners: where the planned CI paths come from
# ----------------------------------------------------------------------
class Planner:
    """
    kind = "point":    one path, e.g. the q50 forecast table (origins x 24)
    kind = "actual":   one path, the true hourly CI
    kind = "scenario": S paths from the calibrated quantiles (scenarios.py);
                       `quantiles` maps "q02".."q98" to tables, `levels` lists
                       the levels used (default q10/q50/q90), and the copula is
                       AR(1) with `rho` or the matrix `corr`.
    paths(H) returns (S, 24) for hours H .. H+23 (truncated at the data end).
    """

    def __init__(self, kind, hourly_ci: pd.Series, table=None, quantiles=None, rho=0.0,
                 n_scenarios=1, seed=SCENARIO_SEED, levels=LEVELS_V1, corr=None):
        self.kind, self.table, self.quantiles = kind, table, quantiles
        self.rho, self.n_scenarios, self.seed = rho, n_scenarios, seed
        self.levels, self.corr = tuple(levels), corr
        self.hours = pd.DatetimeIndex(hourly_ci.index)
        self.ci = hourly_ci.to_numpy(float)
        self._cache_h, self._cache = None, None
        self.n_floored = 0

    @property
    def size(self):
        return self.n_scenarios if self.kind == "scenario" else 1

    def paths(self, h_pos: int) -> np.ndarray:
        if self._cache_h == h_pos:
            return self._cache
        origin = self.hours[h_pos]
        if self.kind == "actual":
            out = self.ci[h_pos:h_pos + T][None, :]
        elif self.kind == "point":
            out = self.table.loc[origin, list(range(T))].to_numpy(float)[None, :]
        elif self.kind == "scenario":
            values = np.stack([self.quantiles[f"q{int(round(q * 100)):02d}"]
                               .loc[origin, list(range(T))].to_numpy(float)
                               for q in self.levels], axis=-1)
            out, floored = sample_paths_levels(values, self.levels, self.n_scenarios, origin,
                                               rho=self.rho, corr=self.corr, seed=self.seed)
            self.n_floored += floored
        else:
            raise ValueError(self.kind)
        self._cache_h, self._cache = h_pos, out
        return out


# ----------------------------------------------------------------------
# Simulation engine (spec §3)
# ----------------------------------------------------------------------
def simulate(jobs: pd.DataFrame, hourly_ci: pd.Series, planner: Planner, alpha: float = 1.0,
             beta: float = 0.0, replan: bool = False, a: float = A_CVAR, ru_check: int = 0,
             label: str = "run", alt_sizes=(), verbose: bool = False,
             solver=lp.solve_lp):
    """
    Jobs in arrival order. At each hour H:
      1. (replan) re-solve every waiting job, earliest arrival first, with the
         capacity rule of spec §3 (only capacity free at the start of H plus
         its own released slot);
      2. plan jobs whose decision hour is H, against everything reserved;
      3. start the jobs whose plan starts at H (fixed from then on).
    solver: lp.solve_lp (PuLP + CBC) for every real run; tests of the engine
    logic may pass lp.solve_brute_force (shown equal to CBC in other tests).
    alt_sizes: scenario counts S' < S; at every solve the decision with the
    first S' scenarios is also computed (brute force) to measure agreement.
    Returns (per-job DataFrame, diagnostics dict).
    """
    hours = pd.DatetimeIndex(hourly_ci.index)
    pos = {h: i for i, h in enumerate(hours)}
    n_hours = len(hours)
    jobs = jobs.sort_values("job_id").reset_index(drop=True)
    o_pos = np.array([pos[h] for h in jobs["decision_hour"]])
    nodes = jobs["nodes"].to_numpy(int)
    dur = jobs["duration_h"].to_numpy(int)
    latest = o_pos + T - dur
    if (o_pos + T > n_hours).any():
        raise ValueError("A job's planning window runs past the end of the data.")

    started_used = np.zeros((J, n_hours), int)
    res_used = np.zeros((J, n_hours), int)
    res = {}
    waiting = []
    n_solves = np.zeros(len(jobs), int)
    by_hour = pd.Series(range(len(jobs))).groupby(o_pos).apply(list).to_dict()
    diag = {"solves": 0, "bf_disagree": 0, "ru_checked": 0, "ru_disagree_option": 0,
            "ru_disagree_value": 0, "ru_time_s": 0.0, "solve_time_s": 0.0,
            "alt_agree": {s: 0 for s in alt_sizes}, "alt_n": 0}

    def plan(i, h_pos, taken):
        latest_offset = latest[i] - h_pos
        t_idx, j_idx = feasible_starts(nodes[i], dur[i], latest_offset, taken)
        if len(t_idx) == 0:
            raise RuntimeError(f"Job {jobs.job_id[i]}: no feasible option at hour {hours[h_pos]}")
        width = latest_offset + dur[i]
        paths = planner.paths(h_pos)[:, :width]
        costs = scenario_costs(paths, nodes[i], dur[i], taken[lp.LOCAL, :width])
        sc = option_scores(t_idx, j_idx, costs, alpha, beta, a)
        coefs = coefficient_list(t_idx, j_idx, sc["f"])
        t0 = time.perf_counter()
        choice = solver(coefs)
        diag["solve_time_s"] += time.perf_counter() - t0
        diag["solves"] += 1
        n_solves[i] += 1
        if choice != lp.solve_brute_force(coefs):
            diag["bf_disagree"] += 1
        if diag["ru_checked"] < ru_check and beta > 0 and planner.size > 1:
            t0 = time.perf_counter()
            ru_choice, ru_value = solve_ru_mip(sc, t_idx, j_idx, alpha, beta, a)
            diag["ru_time_s"] += time.perf_counter() - t0
            diag["ru_checked"] += 1
            best = dict(((t, j), f) for t, j, f in coefs)[choice]
            diag["ru_disagree_option"] += int(ru_choice != choice)
            diag["ru_disagree_value"] += int(abs(ru_value - best) > 1e-6)
            if ru_choice != choice or abs(ru_value - best) > 1e-6:
                f_of = dict(((t, j), f) for t, j, f in coefs)
                diag.setdefault("ru_mismatches", []).append({
                    "job_id": int(jobs.job_id[i]), "hour": str(hours[h_pos]),
                    "per_option_choice": choice, "per_option_value": best,
                    "ru_choice": ru_choice, "ru_reported_value": ru_value,
                    "true_f_of_ru_choice": f_of[ru_choice]})
        if alt_sizes:
            diag["alt_n"] += 1
            for s_alt in alt_sizes:
                sa = option_scores(t_idx, j_idx, costs[:, :s_alt, :], alpha, beta, a)
                k = int(np.argmin(sa["f"]))
                diag["alt_agree"][s_alt] += int((int(t_idx[k]), int(j_idx[k])) == choice)
        return choice

    def cells(i, j, start):
        return j, slice(start, start + dur[i])

    last_h = int(latest.max())
    for h_pos in range(int(o_pos.min()), last_h + 1):
        if replan and waiting:
            base = started_used + res_used
            extras = np.zeros_like(base)
            new = {}
            for i in waiting:
                j_old, s_old = res[i]
                width = latest[i] + dur[i] - h_pos
                taken = base[:, h_pos:h_pos + width].copy()
                taken[j_old, s_old - h_pos:s_old - h_pos + dur[i]] -= nodes[i]
                taken += extras[:, h_pos:h_pos + width]
                # Spec §3 guarantee: the previous plan must still fit. If not, it is a bug.
                old_cells = taken[j_old, s_old - h_pos:s_old - h_pos + dur[i]]
                if (old_cells + nodes[i] > lp.CAPACITIES[j_old]).any():
                    raise AssertionError(f"Job {jobs.job_id[i]}: previous plan no longer fits (bug).")
                t, j = plan(i, h_pos, taken)
                s_new = h_pos + t
                add = np.zeros(n_hours, bool)
                add[s_new:s_new + dur[i]] = True
                if j == j_old:
                    add[s_old:s_old + dur[i]] = False
                extras[j, add] += nodes[i]
                new[i] = (j, s_new)
            for i, (j, s_new) in new.items():
                j_old, s_old = res[i]
                res_used[cells(i, j_old, s_old)] -= nodes[i]
                res_used[cells(i, j, s_new)] += nodes[i]
                res[i] = (j, s_new)

        for i in by_hour.get(h_pos, []):
            taken = (started_used + res_used)[:, h_pos:h_pos + T]
            t, j = plan(i, h_pos, taken)
            res[i] = (j, h_pos + t)
            res_used[cells(i, j, h_pos + t)] += nodes[i]
            waiting.append(i)

        still = []
        for i in waiting:
            j, s = res[i]
            if s == h_pos:
                res_used[cells(i, j, s)] -= nodes[i]
                started_used[cells(i, j, s)] += nodes[i]
            else:
                if s < h_pos or s > latest[i]:
                    raise AssertionError(f"Job {jobs.job_id[i]} has an invalid plan: bug.")
                still.append(i)
        waiting = still
        if (started_used + res_used > lp.CAPACITIES[:, None]).any():
            raise AssertionError("Capacity exceeded: bug.")
        if verbose and h_pos % 168 == 0:
            print(f"    {label}: hour {h_pos - int(o_pos.min())}/{last_h - int(o_pos.min())}",
                  flush=True)
    if waiting:
        raise AssertionError("Jobs left unstarted: bug.")

    # Score with ACTUAL CI, independently re-checking capacity (lp.Grid asserts).
    grid = lp.Grid(hourly_ci)
    starts = {i: res[i] for i in range(len(jobs))}
    for i, (j, s) in starts.items():
        grid.place(j, s, int(dur[i]), int(nodes[i]))
    rows = []
    for i, (j, s) in starts.items():
        rows.append({"policy": label, "alpha": alpha, "beta": beta, "replan": replan,
                     "job_id": int(jobs.job_id[i]), "decision_hour": jobs.decision_hour[i],
                     "nodes": int(nodes[i]), "duration_h": int(dur[i]), "cluster": int(j),
                     "start_offset_h": int(s - o_pos[i]), "start_hour": hours[s],
                     "emissions_g": grid.actual_emissions_g(j, s, int(dur[i]), int(nodes[i])),
                     "n_solves": int(n_solves[i])})
    diag["n_floored"] = planner.n_floored
    return pd.DataFrame(rows), diag


# ----------------------------------------------------------------------
# Metrics (spec §6)
# ----------------------------------------------------------------------
def tail_mean(values, tail: float = CVAR_JOBS_TAIL) -> float:
    """Mean of the worst (largest) ceil(tail * n) values."""
    v = np.sort(np.asarray(values, float))
    m = int(np.ceil(tail * len(v)))
    return float(v[-m:].mean())


def summarise(runs: dict, fifo: str, base: str, actual: str) -> pd.DataFrame:
    """runs: label -> per-job DataFrame (all on the same jobs)."""
    tot = {k: df["emissions_g"].sum() for k, df in runs.items()}
    base_by_job = runs[base].set_index("job_id")["emissions_g"]
    rows = []
    for label, df in runs.items():
        d = df.set_index("job_id").loc[base_by_job.index]
        diff = d["emissions_g"] - base_by_job
        gap = tot[base] - tot[actual]
        intensity = d["emissions_g"] / (d["nodes"] * lp.C_HW_KW * d["duration_h"])
        rows.append({
            "policy": label,
            "total_tCO2": tot[label] / 1e6,
            "cut_vs_fifo_pct": 100 * (1 - tot[label] / tot[fifo]),
            "gap_closed_pct": 100 * (tot[base] - tot[label]) / gap if gap else np.nan,
            "regret_rate_vs_base_pct": 100 * float((diff > 1e-9).mean()),
            "mean_regret_kg": float(diff[diff > 1e-9].mean() / 1e3) if (diff > 1e-9).any() else 0.0,
            "cvar10_intensity_g_per_kwh": tail_mean(intensity),
            "cvar10_job_kg": tail_mean(d["emissions_g"] / 1e3),
            "mean_delay_h": float(d["start_offset_h"].mean()),
            "local_share": float((d["cluster"] == lp.LOCAL).mean()),
            "solves_per_job": float(d["n_solves"].mean()) if "n_solves" in d else np.nan,
        })
    return pd.DataFrame(rows)
