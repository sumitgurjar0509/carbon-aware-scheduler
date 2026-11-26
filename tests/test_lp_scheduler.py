"""
Tests for the M2 base-paper LP reimplementation (src/lp_scheduler.py)
and the hourly forecaster (src/hourly_forecasting.py).
Everything uses small synthetic data; no real files are touched.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import lp_scheduler as lp  # noqa: E402
from hourly_forecasting import horizon_dataset, hourly_features, split_by_boundaries, to_hourly  # noqa: E402

T = 6  # small horizon for hand-checkable cases
START = pd.Timestamp("2026-01-01", tz="UTC")


def _plan(gb, n_clusters=lp.N_CLUSTERS):
    return np.tile(np.asarray(gb, dtype=float), (n_clusters, 1))


def _choose(nodes, duration, gb, used, alpha, solver=lp.solve_lp, horizon=T):
    opts = lp.feasible_options(nodes, duration, _plan(gb), used, lp.CAPACITIES, horizon)
    return solver(lp.objective_coefficients(opts, alpha, horizon))


# ----------------------------------------------------------------------
# Condition 1(a): alpha = 1 picks the lowest-emission FEASIBLE start
# ----------------------------------------------------------------------
def test_alpha1_picks_lowest_emission_start():
    gb = [300, 250, 100, 120, 400, 500]  # 2-hour windows: 550, 350, 220, 520, 900
    used = np.zeros((lp.N_CLUSTERS, T), dtype=int)
    t, j = _choose(nodes=4, duration=2, gb=gb, used=used, alpha=1.0)
    assert t == 2


def test_alpha1_picks_lowest_feasible_start_when_best_hour_is_full():
    # Hour 2 is full on every cluster, so starts 1 and 2 are impossible.
    # Remaining 2-hour windows: t=0 -> 550, t=3 -> 520, t=4 -> 900. Best: t=3.
    gb = [300, 250, 100, 120, 400, 500]
    used = np.zeros((lp.N_CLUSTERS, T), dtype=int)
    used[:, 2] = lp.CAPACITIES
    opts = lp.feasible_options(4, 2, _plan(gb), used, lp.CAPACITIES, T)
    assert sorted({t for t, _, _ in opts}) == [0, 3, 4]
    t, _ = lp.solve_lp(lp.objective_coefficients(opts, 1.0, T))
    assert t == 3


# ----------------------------------------------------------------------
# Condition 1(b): alpha = 0 with spare local capacity picks the local cluster
# ----------------------------------------------------------------------
def test_alpha0_picks_local_cluster_when_it_has_room():
    gb = [100, 100, 100, 100, 100, 100]
    gb_local_worse = _plan(gb)
    gb_local_worse[lp.LOCAL] = 900  # local is much dirtier: alpha=0 must still choose it
    used = np.zeros((lp.N_CLUSTERS, T), dtype=int)
    opts = lp.feasible_options(5, 3, gb_local_worse, used, lp.CAPACITIES, T)
    t, j = lp.solve_lp(lp.objective_coefficients(opts, 0.0, T))
    assert j == lp.LOCAL


def test_alpha0_goes_to_cloud_when_local_is_full():
    used = np.zeros((lp.N_CLUSTERS, T), dtype=int)
    used[lp.LOCAL, :] = lp.N_MAX_LOCAL
    _, j = _choose(nodes=3, duration=2, gb=[100] * T, used=used, alpha=0.0)
    assert j != lp.LOCAL


def test_job_bigger_than_local_cluster_never_goes_local():
    used = np.zeros((lp.N_CLUSTERS, T), dtype=int)
    _, j = _choose(nodes=15, duration=2, gb=[100] * T, used=used, alpha=0.0)
    assert j != lp.LOCAL


# ----------------------------------------------------------------------
# Condition 1(c): CBC equals brute force on many small random cases
# ----------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(25))
def test_lp_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    horizon = 8
    plan = rng.uniform(50, 300, (lp.N_CLUSTERS, horizon))
    used = rng.integers(0, 15, (lp.N_CLUSTERS, horizon))
    used[lp.LOCAL] = rng.integers(0, 10, horizon)
    nodes, duration = int(rng.integers(1, 9)), int(rng.integers(1, 6))
    alpha = float(rng.choice([0.0, 0.5, 1.0, rng.uniform()]))
    opts = lp.feasible_options(nodes, duration, plan, used, lp.CAPACITIES, horizon)
    if not opts:
        pytest.skip("no feasible option in this random case")
    coefs = lp.objective_coefficients(opts, alpha, horizon)
    assert lp.solve_lp(coefs) == lp.solve_brute_force(coefs)


def test_ties_broken_by_lowest_cluster_then_earliest_start():
    used = np.zeros((lp.N_CLUSTERS, T), dtype=int)
    assert _choose(2, 2, [100] * T, used, alpha=1.0) == (0, lp.LOCAL)
    assert _choose(2, 2, [100] * T, used, alpha=1.0, solver=lp.solve_brute_force) == (0, lp.LOCAL)


# ----------------------------------------------------------------------
# Objective pieces
# ----------------------------------------------------------------------
def test_normalised_terms_are_in_0_1():
    used = np.zeros((lp.N_CLUSTERS, T), dtype=int)
    opts = lp.feasible_options(3, 2, _plan([80, 200, 150, 90, 60, 300]), used, lp.CAPACITIES, T)
    c_max = max(c for _, _, c in opts)
    for alpha in (0.0, 0.5, 1.0):
        for (t, j, coef), (_, _, c) in zip(lp.objective_coefficients(opts, alpha, T), opts):
            e_bar, l_bar = c / c_max, float(j == lp.LOCAL)
            assert 0 <= e_bar <= 1
            assert coef == pytest.approx(alpha * e_bar - (1 - alpha) * l_bar, abs=1e-6)


def test_feasible_options_respect_horizon_and_capacity():
    used = np.zeros((lp.N_CLUSTERS, T), dtype=int)
    used[3, 1] = 18
    opts = lp.feasible_options(4, 3, _plan([1] * T), used, lp.CAPACITIES, T)
    assert all(t + 3 <= T for t, _, _ in opts)
    assert (0, 3, 3.0) not in opts and (1, 3, 3.0) not in opts and (2, 3, 3.0) in opts


# ----------------------------------------------------------------------
# Eq. 1-2 (corrected sign, spec G1)
# ----------------------------------------------------------------------
def test_local_ci_without_solar_equals_grid():
    np.testing.assert_allclose(lp.local_adjusted_ci([100, 200], 4, [0, 6]), [100, 200])


def test_local_ci_with_solar_mixes_by_grid_share():
    # demand = 8.5 * (2 + 2) = 34 kW; generator 17 kW -> half from the grid.
    assert lp.local_adjusted_ci(200, 2, 2, p_gen_kw=17, ef_gen=40) == pytest.approx(120)
    # generator covers everything -> emission factor of the generator only.
    assert lp.local_adjusted_ci(200, 2, 2, p_gen_kw=100, ef_gen=40) == pytest.approx(40)


# ----------------------------------------------------------------------
# Condition 2: job generator
# ----------------------------------------------------------------------
def _jobs(seed=lp.JOB_SEED):
    return lp.generate_jobs(START, START + pd.Timedelta(days=60) - lp.ONE_HOUR, seed=seed)


def test_generator_values_are_valid_and_count_matches_rate():
    jobs = _jobs()
    assert jobs["duration_h"].between(1, lp.HORIZON_T).all()
    assert jobs["nodes"].between(1, lp.CAPACITIES.max()).all()
    expected = 1000 * (60 - 1) / lp.DAYS_PER_MONTH  # arrival window = 60 days - 24 h
    assert abs(len(jobs) - expected) <= 1
    assert abs(jobs["nodes"].mean() - 6) < 0.3 and abs(jobs["duration_h"].mean() - 6) < 0.3


def test_generator_windows_fit_the_data():
    jobs = _jobs()
    last = START + pd.Timedelta(days=60) - lp.ONE_HOUR
    assert (jobs["decision_hour"] >= START).all()
    assert (jobs["decision_hour"] + (lp.HORIZON_T - 1) * lp.ONE_HOUR <= last).all()
    assert (jobs["decision_hour"] >= jobs["arrival_time"]).all()
    assert jobs["arrival_time"].is_monotonic_increasing


def test_generator_is_reproducible_with_fixed_seed():
    pd.testing.assert_frame_equal(_jobs(), _jobs())
    assert not _jobs().equals(_jobs(seed=1))


def test_truncation_redraws_out_of_range_values():
    rng = np.random.default_rng(0)
    values = lp._truncated_rounded_normal(rng, 20000, 5, 7)
    assert set(np.unique(values)) == {5, 6, 7}


# ----------------------------------------------------------------------
# Whole-run properties
# ----------------------------------------------------------------------
def _small_world(n_days=6, seed=3):
    rng = np.random.default_rng(seed)
    hours = pd.date_range(START, periods=24 * n_days, freq="h")
    ci = pd.Series(150 + 60 * np.sin(np.arange(len(hours)) * 2 * np.pi / 24)
                   + rng.normal(0, 10, len(hours)), index=hours)
    jobs = lp.generate_jobs(hours[0], hours[-1], seed=seed, jobs_per_month=1500)
    forecast = pd.DataFrame(
        {h: ci.shift(-h).reindex(jobs["decision_hour"].unique()).to_numpy() + rng.normal(0, 20)
         for h in range(lp.HORIZON_T)}, index=pd.DatetimeIndex(jobs["decision_hour"].unique()))
    return ci, jobs, forecast


def _usage(results, ci):
    used = np.zeros((lp.N_CLUSTERS, len(ci)), dtype=int)
    pos = {h: i for i, h in enumerate(ci.index)}
    for r in results.itertuples():
        used[r.cluster, pos[r.start_hour]: pos[r.start_hour] + r.duration_h] += r.nodes
    return used


@pytest.mark.parametrize("plan, alpha", [("actual", 1.0), ("forecast", 1.0), ("actual", 0.0)])
def test_lp_run_places_every_job_once_within_capacity_and_horizon(plan, alpha):
    ci, jobs, forecast = _small_world()
    res, disagreements = lp.run_lp(jobs, ci, alpha, plan, forecast)
    assert disagreements == 0
    assert sorted(res["job_id"]) == sorted(jobs["job_id"])
    assert (res["start_offset_h"] + res["duration_h"] <= lp.HORIZON_T).all()
    assert (_usage(res, ci) <= lp.CAPACITIES[:, None]).all()


def test_fifo_respects_capacity_starts_on_arrival_and_is_reproducible():
    ci, jobs, _ = _small_world()
    a, b = lp.run_random_fifo(jobs, ci), lp.run_random_fifo(jobs, ci)
    pd.testing.assert_frame_equal(a, b)
    assert (_usage(a, ci) <= lp.CAPACITIES[:, None]).all()
    assert (a["start_offset_h"] == 0).mean() > 0.95


def test_ground_truth_lp_beats_fifo_and_emissions_formula():
    ci, jobs, _ = _small_world()
    lp_res, _ = lp.run_lp(jobs, ci, 1.0, "actual")
    fifo = lp.run_random_fifo(jobs, ci)
    assert lp_res["emissions_g"].sum() < fifo["emissions_g"].sum()
    r = lp_res.iloc[0]
    p = ci.index.get_loc(r.start_hour)
    assert r.emissions_g == pytest.approx(r.nodes * lp.C_HW_KW * ci.iloc[p:p + r.duration_h].sum())


def test_summary_reduction_is_relative_to_fifo():
    df = pd.DataFrame({"policy": ["random_fifo", "x"], "alpha": [np.nan, 1.0],
                       "emissions_g": [200.0, 150.0], "cluster": [0, 1],
                       "start_offset_h": [0, 2], "solve_s": [np.nan, 0.1]})
    s = lp.summarise(df).set_index("policy")
    assert s.loc["x", "pct_reduction_vs_random_fifo"] == pytest.approx(25.0)


# ----------------------------------------------------------------------
# Hourly forecaster data handling
# ----------------------------------------------------------------------
def _df30(n_hours=24 * 30, start="2026-01-01 00:30"):
    ts = pd.date_range(start, periods=2 * n_hours, freq="30min", tz="UTC")
    return pd.DataFrame({"timestamp": ts, "carbon_intensity": np.arange(len(ts), dtype=float)})


def test_to_hourly_averages_halves_and_drops_incomplete_first_hour():
    hourly = to_hourly(_df30())
    assert hourly["timestamp"].iloc[0] == pd.Timestamp("2026-01-01 01:00", tz="UTC")
    assert hourly["carbon_intensity"].iloc[0] == pytest.approx((1 + 2) / 2)


def test_to_hourly_leaves_hour_with_missing_half_as_nan():
    df = _df30()
    df.loc[10, "carbon_intensity"] = np.nan
    hourly = to_hourly(df).set_index("timestamp")["carbon_intensity"]
    assert hourly.isna().sum() == 1


def test_hourly_split_keeps_targets_inside_each_period():
    hourly = to_hourly(_df30())
    feat = hourly_features(hourly)
    df_h = horizon_dataset(feat, hourly, h=5)
    calib_start = pd.Timestamp("2026-01-15", tz="UTC")
    test_start = pd.Timestamp("2026-01-22", tz="UTC")
    train, calib, test = split_by_boundaries(df_h, calib_start, test_start)
    assert (train["target_timestamp"] + lp.ONE_HOUR <= calib_start).all()
    assert (calib["target_timestamp"] + lp.ONE_HOUR <= test_start).all()
    assert (calib["timestamp"] >= calib_start).all() and (test["timestamp"] >= test_start).all()
    assert (df_h["target_timestamp"] - df_h["timestamp"] == pd.Timedelta(hours=5)).all()
