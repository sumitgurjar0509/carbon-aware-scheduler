"""Tests for M3: scenarios.py and m3_scheduler.py (synthetic data only)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import lp_scheduler as lp  # noqa: E402
import m3_scheduler as m3  # noqa: E402
import scenarios as sc  # noqa: E402

ORIGIN = pd.Timestamp("2026-06-10 05:00", tz="UTC")


# ----------------------------------------------------------------------
# Scenario marginals
# ----------------------------------------------------------------------
def test_knots_pass_through_quantiles_with_linear_tails():
    k = sc.quantile_knots([100.0], [120.0], [180.0])[0]
    np.testing.assert_allclose(k, [95.0, 100.0, 120.0, 180.0, 195.0])


def test_crossed_quantiles_are_sorted():
    k = sc.quantile_knots(130.0, 120.0, 200.0)
    assert np.all(np.diff(k) >= 0)


def test_inverse_cdf_hits_quantiles_and_cdf_inverts_it():
    knots = sc.quantile_knots(np.array([50.0, 80.0]), np.array([70.0, 90.0]), np.array([120.0, 95.0]))
    u = np.array([[0.1, 0.5], [0.9, 0.9], [0.3, 0.7]])
    y = sc.inverse_cdf(u, knots)
    assert y[0, 0] == pytest.approx(50.0) and y[0, 1] == pytest.approx(90.0)
    assert y[1, 0] == pytest.approx(120.0)
    np.testing.assert_allclose(sc.cdf(y[2], knots), u[2], atol=1e-6)


def test_cdf_is_0_or_1_outside_the_bounded_tails():
    knots = sc.quantile_knots([100.0], [120.0], [180.0])
    assert sc.cdf(np.array([10.0]), knots)[0] == 0.0
    assert sc.cdf(np.array([500.0]), knots)[0] == 1.0


def test_exact_mean_matches_large_sample():
    q = (np.full(3, 60.0), np.full(3, 100.0), np.full(3, 220.0))
    paths, _ = sc.sample_paths(*q, rho=0.0, n_scenarios=200_000, origin=ORIGIN)
    np.testing.assert_allclose(paths.mean(axis=0), sc.scenario_mean(sc.quantile_knots(*q)), rtol=5e-3)


def test_flooring_only_when_lower_tail_goes_negative():
    _, none = sc.sample_paths([5.0], [10.0], [50.0], rho=0.0, n_scenarios=5000, origin=ORIGIN)
    paths, some = sc.sample_paths([2.0], [20.0], [50.0], rho=0.0, n_scenarios=5000, origin=ORIGIN)
    assert none == 0          # Q(0) = 3.75 > 0
    assert some > 0           # Q(0) = 2 - 4.5 < 0
    assert paths.min() == 0.0


# ----------------------------------------------------------------------
# Copula and seeds
# ----------------------------------------------------------------------
def _flat_q(n=24):
    return np.full(n, 80.0), np.full(n, 100.0), np.full(n, 130.0)


def test_copula_correlation_follows_ar1():
    rho = 0.8
    paths, _ = sc.sample_paths(*_flat_q(), rho=rho, n_scenarios=40_000, origin=ORIGIN)
    knots = sc.quantile_knots(*_flat_q())
    z = np.array([sc.pit_z(p, knots)[0] for p in paths[:5000]])
    for lag in (1, 3):
        emp = np.corrcoef(z[:, :-lag].ravel(), z[:, lag:].ravel())[0, 1]
        assert emp == pytest.approx(rho ** lag, abs=0.03)


def test_marginal_quantiles_are_respected():
    paths, _ = sc.sample_paths(*_flat_q(), rho=0.6, n_scenarios=50_000, origin=ORIGIN)
    assert (paths[:, 7] <= 80.0).mean() == pytest.approx(0.1, abs=0.01)
    assert (paths[:, 7] <= 130.0).mean() == pytest.approx(0.9, abs=0.01)


def test_seeds_are_reproducible_per_origin_and_nested():
    a, _ = sc.sample_paths(*_flat_q(), rho=0.5, n_scenarios=2000, origin=ORIGIN)
    b, _ = sc.sample_paths(*_flat_q(), rho=0.5, n_scenarios=100, origin=ORIGIN)
    c, _ = sc.sample_paths(*_flat_q(), rho=0.5, n_scenarios=100, origin=ORIGIN + lp.ONE_HOUR)
    np.testing.assert_array_equal(a[:100], b)       # first S rows = a smaller draw
    assert not np.allclose(b, c)                    # another origin, another stream


def test_rho_mle_recovers_true_value():
    rng = np.random.default_rng(0)
    rho, n = 0.7, 3000
    z = np.empty((n, 24))
    z[:, 0] = rng.standard_normal(n)
    for h in range(1, 24):
        z[:, h] = rho * z[:, h - 1] + np.sqrt(1 - rho ** 2) * rng.standard_normal(n)
    assert sc.estimate_rho(z) == pytest.approx(rho, abs=0.01)
    assert sc.lag_correlations(z, 2)[2] == pytest.approx(rho ** 2, abs=0.02)


# ----------------------------------------------------------------------
# CVaR and the three solvers
# ----------------------------------------------------------------------
def test_cvar_is_mean_of_worst_20_percent():
    v = np.arange(1, 11, dtype=float)[None, :]  # 1..10
    assert m3.cvar_upper(v, 0.2)[0] == pytest.approx((9 + 10) / 2)
    assert m3.cvar_upper(np.array([[7.0]]), 0.2)[0] == 7.0
    with pytest.raises(ValueError):
        m3.cvar_upper(np.ones((1, 7)), 0.2)


def _random_case(seed, s=20, width=10):
    rng = np.random.default_rng(seed)
    nodes, dur = int(rng.integers(1, 9)), int(rng.integers(1, 5))
    taken = rng.integers(0, 15, (m3.J, width))
    taken[lp.LOCAL] = rng.integers(0, 8, width)
    paths = rng.uniform(40, 300, (s, width))
    t_idx, j_idx = m3.feasible_starts(nodes, dur, width - dur, taken)
    costs = m3.scenario_costs(paths, nodes, dur, taken[lp.LOCAL])
    return t_idx, j_idx, costs, rng


@pytest.mark.parametrize("seed", range(15))
def test_ru_mip_per_option_and_brute_force_agree(seed):
    t_idx, j_idx, costs, rng = _random_case(seed)
    if len(t_idx) == 0:
        pytest.skip("no feasible option")
    alpha = float(rng.choice([1.0, 0.5, rng.uniform()]))
    beta = float(rng.choice([0.25, 0.5, 1.0]))
    s = m3.option_scores(t_idx, j_idx, costs, alpha, beta)
    coefs = m3.coefficient_list(t_idx, j_idx, s["f"])
    per_option = lp.solve_lp(coefs)
    ru_choice, ru_value = m3.solve_ru_mip(s, t_idx, j_idx, alpha, beta)
    assert per_option == lp.solve_brute_force(coefs) == ru_choice
    assert ru_value == pytest.approx(dict(((t, j), f) for t, j, f in coefs)[per_option], abs=1e-6)


def test_normalisation_uses_largest_mean_cost():
    t_idx, j_idx, costs, _ = _random_case(3)
    s = m3.option_scores(t_idx, j_idx, costs, 1.0, 0.5)
    assert s["m"].max() == pytest.approx(1.0)
    assert s["E"].max() > 1.0  # bad scenarios may exceed 1, as intended


def test_beta0_equals_planning_on_the_scenario_mean_path():
    t_idx, j_idx, costs, _ = _random_case(4, s=40)
    s0 = m3.option_scores(t_idx, j_idx, costs, 1.0, 0.0)
    mean_costs = costs.mean(axis=1, keepdims=True)
    s_mean = m3.option_scores(t_idx, j_idx, mean_costs, 1.0, 0.0)
    np.testing.assert_allclose(s0["f"], s_mean["f"])


def test_feasible_starts_respect_window_and_capacity():
    taken = np.zeros((m3.J, 8), int)
    taken[1, 2] = 18
    t_idx, j_idx = m3.feasible_starts(4, 3, 5, taken)
    opts = set(zip(t_idx.tolist(), j_idx.tolist()))
    assert (0, 1) not in opts and (2, 1) not in opts and (3, 1) in opts
    assert t_idx.max() == 5


# ----------------------------------------------------------------------
# Engine
# ----------------------------------------------------------------------
def _world(seed=3, n_days=6, jobs_per_month=1500):
    rng = np.random.default_rng(seed)
    hours = pd.date_range("2026-06-06", periods=24 * n_days, freq="h", tz="UTC")
    ci = pd.Series(150 + 60 * np.sin(np.arange(len(hours)) * 2 * np.pi / 24)
                   + rng.normal(0, 10, len(hours)), index=hours)
    jobs = lp.generate_jobs(hours[0], hours[-1], seed=seed, jobs_per_month=jobs_per_month)
    noisy = {h: ci.shift(-h).to_numpy() + rng.normal(0, 25, len(hours)) for h in range(m3.T)}
    q50 = pd.DataFrame(noisy, index=hours).ffill()
    quant = {"q10": q50 - 30, "q50": q50, "q90": q50 + 40}
    return ci, jobs, quant


def _check_run(res, ci, jobs):
    used = np.zeros((m3.J, len(ci)), int)
    p = {h: i for i, h in enumerate(ci.index)}
    for r in res.itertuples():
        used[r.cluster, p[r.start_hour]:p[r.start_hour] + r.duration_h] += r.nodes
    assert (used <= lp.CAPACITIES[:, None]).all()
    assert sorted(res["job_id"]) == sorted(jobs["job_id"])
    assert (res["start_offset_h"] >= 0).all()
    assert (res["start_offset_h"] + res["duration_h"] <= m3.T).all()


def test_point_planner_without_replanning_reproduces_m2_exactly():
    ci, jobs, quant = _world()
    m2, _ = lp.run_lp(jobs, ci, 1.0, "forecast", quant["q50"])
    planner = m3.Planner("point", ci, table=quant["q50"])
    new, diag = m3.simulate(jobs, ci, planner, alpha=1.0)
    a = m2.set_index("job_id")[["cluster", "start_hour"]]
    b = new.set_index("job_id").loc[a.index, ["cluster", "start_hour"]]
    pd.testing.assert_frame_equal(a, b)
    assert diag["bf_disagree"] == 0


BF = lp.solve_brute_force  # engine-logic tests; CBC == brute force is tested separately


@pytest.mark.parametrize("kind, beta", [("point", 0.0), ("scenario", 0.5)])
def test_replanning_keeps_capacity_windows_and_places_every_job(kind, beta):
    ci, jobs, quant = _world(seed=5)
    planner = m3.Planner(kind, ci, table=quant["q50"], quantiles=quant, rho=0.6, n_scenarios=20)
    res, diag = m3.simulate(jobs, ci, planner, alpha=1.0, beta=beta, replan=True, ru_check=10,
                            solver=BF)
    _check_run(res, ci, jobs)
    assert (res["n_solves"] >= 1).all()
    assert (res["n_solves"] == res["start_offset_h"] + 1).all()  # once at arrival + once per waiting hour
    if kind == "scenario":
        assert diag["ru_checked"] == 10 and diag["ru_disagree_option"] == 0


def test_replanning_with_cbc_solver_on_a_short_run():
    ci, jobs, quant = _world(seed=6)
    planner = m3.Planner("scenario", ci, quantiles=quant, rho=0.6, n_scenarios=20)
    res, diag = m3.simulate(jobs.head(25), ci, planner, beta=0.5, replan=True, ru_check=5)
    _check_run(res, ci, jobs.head(25))
    assert diag["bf_disagree"] == 0 and diag["ru_disagree_option"] == 0


def test_replanning_single_job_with_unchanging_forecast_keeps_its_plan():
    ci, jobs, _ = _world()
    one = jobs.head(1)
    table = pd.DataFrame({h: ci.shift(-h).to_numpy() for h in range(m3.T)}, index=ci.index).ffill()
    planner = m3.Planner("point", ci, table=table)
    a, _ = m3.simulate(one, ci, planner, replan=False, solver=BF)
    b, _ = m3.simulate(one, ci, m3.Planner("point", ci, table=table), replan=True, solver=BF)
    assert a.start_hour.iloc[0] == b.start_hour.iloc[0] and a.cluster.iloc[0] == b.cluster.iloc[0]


@pytest.mark.parametrize("seed", range(6))
def test_replanning_under_load_keeps_every_previous_plan_feasible(seed):
    """
    ~1500 jobs/month on 6 days: many capacity conflicts. The engine asserts
    at every re-plan that the job's previous plan still fits (spec §3).
    (At much higher load NEW arrivals can find no room, with or without
    re-planning; that is M2's rule and not what is tested here.)
    """
    ci, jobs, quant = _world(seed=seed, jobs_per_month=1500)
    planner = m3.Planner("scenario", ci, quantiles=quant, rho=0.6, n_scenarios=20)
    res, _ = m3.simulate(jobs, ci, planner, beta=0.5, replan=True, solver=BF)
    _check_run(res, ci, jobs)


def test_alt_sizes_report_agreement_with_reference():
    ci, jobs, quant = _world()
    planner = m3.Planner("scenario", ci, quantiles=quant, rho=0.5, n_scenarios=100)
    _, diag = m3.simulate(jobs.head(30), ci, planner, beta=0.5, alt_sizes=(10, 100), solver=BF)
    assert diag["alt_agree"][100] == diag["alt_n"] == diag["solves"]
    assert 0 <= diag["alt_agree"][10] <= diag["alt_n"]


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
def test_tail_mean_and_summary():
    assert m3.tail_mean(np.arange(1, 21), 0.1) == pytest.approx(19.5)
    base = pd.DataFrame({"job_id": [1, 2], "emissions_g": [100.0, 100.0], "nodes": [1, 1],
                         "duration_h": [1, 1], "start_offset_h": [0, 2], "cluster": [0, 1],
                         "n_solves": [1, 3]})
    runs = {"fifo": base.assign(emissions_g=[200.0, 200.0]), "base": base,
            "actual": base.assign(emissions_g=[50.0, 50.0]),
            "x": base.assign(emissions_g=[60.0, 110.0])}
    s = m3.summarise(runs, "fifo", "base", "actual").set_index("policy")
    assert s.loc["x", "cut_vs_fifo_pct"] == pytest.approx(57.5)
    assert s.loc["x", "gap_closed_pct"] == pytest.approx(30.0)
    assert s.loc["x", "regret_rate_vs_base_pct"] == pytest.approx(50.0)
    assert s.loc["actual", "gap_closed_pct"] == pytest.approx(100.0)


# ----------------------------------------------------------------------
# Stage-1 selection rules (m3_tuning.py)
# ----------------------------------------------------------------------
def test_select_s_picks_smallest_stable_size():
    import m3_tuning as tu
    assert tu.select_s({100: 0.97, 200: 0.992, 500: 0.999, 1000: 1.0}) == 200
    assert tu.select_s({100: 0.9, 200: 0.95, 500: 0.98, 1000: 0.985}) == tu.S_REF


def test_select_beta_lowest_total_with_tie_to_smallest():
    import m3_tuning as tu
    totals = {0.0: 100.0, 0.1: 99.0, 0.25: 98.95, 0.5: 98.0, 0.75: 98.05, 1.0: 101.0}
    assert tu.select_beta(totals) == (0.5, 0.5)
    tied = {0.0: 98.0, 0.1: 98.05, 0.25: 99.0}          # 0.1 within 0.1% of best (0.0)
    assert tu.select_beta(tied) == (0.0, 0.1)


def test_c1_rows_keep_targets_before_the_boundary():
    import m3_tuning as tu
    ts = pd.date_range("2026-06-05 20:00", periods=8, freq="h", tz="UTC")
    df_h = pd.DataFrame({"timestamp": ts, "target_timestamp": ts + 2 * lp.ONE_HOUR})
    kept = tu.c1_rows(df_h, ts[0], tu.C1_END)
    assert (kept["target_timestamp"] + lp.ONE_HOUR <= tu.C1_END).all()
    assert len(kept) == 2  # origins 20:00, 21:00 -> targets end 23:00, 00:00


# ----------------------------------------------------------------------
# Scenario-model revision (spec §1a) and tail-risk beta
# ----------------------------------------------------------------------
def test_general_knots_reduce_to_v1_for_three_levels():
    vals = np.array([[100.0, 120.0, 180.0]])
    p, q = sc.knots_from_levels(vals, sc.LEVELS_V1)
    np.testing.assert_allclose(p, sc.P_KNOTS)
    np.testing.assert_allclose(q[0], sc.quantile_knots(100.0, 120.0, 180.0))


def test_seven_level_knots_tails_and_sampling():
    levels = sc.LEVELS_V2
    vals = np.tile([60.0, 75.0, 85.0, 100.0, 130.0, 150.0, 180.0], (24, 1))
    p, q = sc.knots_from_levels(vals, levels)
    assert q[0, 0] == pytest.approx(60 - (75 - 60) / 0.03 * 0.02)   # slope of the first segment
    assert q[0, -1] == pytest.approx(180 + (180 - 150) / 0.03 * 0.02)
    paths, _ = sc.sample_paths_levels(vals, levels, 50_000, ORIGIN, rho=0.5)
    assert (paths[:, 3] <= 75.0).mean() == pytest.approx(0.05, abs=0.006)
    assert (paths[:, 3] <= 150.0).mean() == pytest.approx(0.95, abs=0.006)


def test_sampling_with_a_correlation_matrix():
    corr = np.full((24, 24), 0.3) + 0.7 * np.eye(24)
    vals = np.tile([80.0, 100.0, 130.0], (24, 1))
    paths, _ = sc.sample_paths_levels(vals, sc.LEVELS_V1, 40_000, ORIGIN, corr=corr)
    knots = sc.quantile_knots(vals[:, 0], vals[:, 1], vals[:, 2])
    z = np.array([sc.pit_z(r, knots)[0] for r in paths[:4000]])
    assert np.corrcoef(z[:, 2], z[:, 17])[0, 1] == pytest.approx(0.3, abs=0.05)


def test_planner_uses_requested_levels():
    ci, jobs, quant = _world()
    wide = dict(quant, q05=quant["q10"] - 30, q95=quant["q90"] + 30)
    p3 = m3.Planner("scenario", ci, quantiles=wide, n_scenarios=2000)
    p5 = m3.Planner("scenario", ci, quantiles=wide, n_scenarios=2000,
                    levels=(0.05, 0.1, 0.5, 0.9, 0.95))
    assert p5.paths(5).std() > p3.paths(5).std()


def test_pit_outside_support_and_histogram():
    import m3_scenario_model as sm
    vals = np.tile([80.0, 100.0, 130.0], (2, 24, 1))
    y = np.full((2, 24), 100.0)
    y[0, 0] = 10.0        # far below Q(0) = 75
    u, outside = sm.pit_values(vals, sc.LEVELS_V1, y)
    assert outside.sum() == 1 and u[0, 0] == 0.0
    assert u[1, 5] == pytest.approx(0.5, abs=1e-6)


def test_shrunk_correlation_is_a_valid_correlation_matrix():
    import m3_scenario_model as sm
    rng = np.random.default_rng(1)
    z = rng.standard_normal((300, 24)) @ np.linalg.cholesky(sc.ar1_corr(0.8)).T
    corr, shrink = sm.shrunk_correlation(z)
    np.testing.assert_allclose(np.diag(corr), 1.0)
    assert 0 <= shrink <= 1 and np.all(np.linalg.eigvalsh(corr) > 0)
    assert corr[0, 1] == pytest.approx(0.8, abs=0.1)


def test_select_beta_tail_rule():
    import m3_tuning as tu
    totals = {0.0: 100.0, 0.1: 100.5, 0.25: 101.5, 0.5: 100.9}
    cvar = {0.0: 200.0, 0.1: 199.0, 0.25: 150.0, 0.5: 190.0}
    # base_replan = 100 -> limit 101: 0.25 excluded; lowest CVaR among 0.1, 0.5 -> 0.5
    assert tu.select_beta_tail(totals, cvar, 100.0) == 0.5
    assert tu.select_beta_tail({0.0: 90.0, 0.1: 120.0}, {0.0: 1.0, 0.1: 1.0}, 100.0) is None
