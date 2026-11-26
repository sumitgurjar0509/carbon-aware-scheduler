"""
The Five Scheduler Strategies
===============================
Part of: AI-Based Energy and Carbon Optimization in Cloud Computing Environments

Each scheduler is a DECISION FUNCTION with the same signature, called by
`simulator.py` at every feasible decision time for a job:

    scheduler(t, job, remaining_times, ctx) -> bool
        t               : current decision time (pd.Timestamp, on the job's
                           hourly decision grid -- see simulator.py)
        job              : the job's row (arrival_time, duration_minutes,
                           slack_hours, deadline, ...)
        remaining_times  : list of the job's remaining FUTURE decision times
                           after t (empty only at the final, forced step --
                           simulator.py handles forcing directly and never
                           actually calls a scheduler with an empty list in
                           normal operation, but every scheduler below
                           still guards for it defensively)
        ctx              : a `simulator.SimContext` giving access to
                           ctx.actual_at(t) (ground-truth carbon intensity)
                           and ctx.forecast(origin_t, horizon_hours) (CQR-
                           calibrated {q10, q50, q90} forecast)

Returns True ("run now") or False ("wait, re-evaluate at the next step").

All five schedulers see the exact same discretized set of candidate start
times per job (see simulator.py's decision-grid construction) -- so any
difference in outcomes between them reflects the SCHEDULING POLICY, not a
difference in what candidates were available to choose from.
"""
import numpy as np
import pandas as pd


def _horizon_hours(t: pd.Timestamp, future_t: pd.Timestamp) -> int:
    """
    Every decision time on a job's grid is exactly on an hourly stride
    from every other (see simulator.py), so this is always an exact
    integer -- round() only guards against floating-point timedelta
    noise, never a real fractional horizon.
    """
    return int(round((future_t - t) / pd.Timedelta(hours=1)))


def run_immediately(t, job, remaining_times, ctx) -> bool:
    """
    Baseline: no carbon-awareness at all. Always runs at the very first
    decision time (the job's own arrival, snapped to the settlement
    grid) -- this is what every other scheduler's savings/regret are
    measured against.
    """
    return True


def threshold(t, job, remaining_times, ctx) -> bool:
    """
    Runs as soon as the CURRENT actual carbon intensity is at or below a
    fixed threshold, else waits.

    Threshold = the median actual carbon intensity of the TRAIN split
    (computed once, offline, in simulator.py -- never per-job or from
    calibration/test data, which would leak information a real
    deployment wouldn't have at decision time). Using the train-set
    median specifically (rather than an arbitrary fixed number) means
    the threshold reflects "typical" grid carbon intensity for this grid
    as of the historical period the model was trained on, without
    needing any forecast at all -- the simplest carbon-aware policy that
    isn't just "always run now".
    """
    return ctx.actual_at(t) <= ctx.threshold_value


def forecast_greedy(t, job, remaining_times, ctx) -> bool:
    """
    Runs now if the current actual intensity already beats the best
    (lowest) CALIBRATED q50 forecast among all remaining feasible future
    start times; otherwise waits one step and re-evaluates. Uses
    `calibration.predict_calibrated()`-derived forecasts exclusively
    (via ctx.forecast) -- never the raw, uncalibrated quantiles.

    This is a point-forecast policy: it trusts the median forecast at
    face value and has no notion of how uncertain that forecast is.
    """
    if not remaining_times:
        return True
    horizons = [_horizon_hours(t, ft) for ft in remaining_times]
    best_future_q50 = min(ctx.forecast(t, h)["q50"] for h in horizons)
    return ctx.actual_at(t) <= best_future_q50


def oracle(t, job, remaining_times, ctx) -> bool:
    """
    Perfect-knowledge upper bound on achievable savings -- an upper
    bound on achievable savings, not a deployable policy (a real
    scheduler cannot see the future). Only valid in simulation, where we
    already know the ground truth for the whole test period.

    DESIGN NOTE -- corrected to compare INTEGRATED cost, not point
    intensity (read before the viva): an earlier version of this
    scheduler mirrored forecast_greedy exactly, comparing the actual
    POINT intensity at each candidate start time (same comparison unit
    as forecast_greedy's q50 point forecast). That is not a valid
    per-job upper bound once a job's duration spans multiple settlement
    periods: the single lowest-point-intensity slot can be immediately
    followed by a spike, giving it a WORSE fully-integrated realized
    cost than a slightly-higher-starting slot with a calmer follow-on
    period. Empirically, this made the point-based oracle lose to
    forecast_greedy on a real per-job basis for ~8% of jobs in a full
    simulation run, even though it still won in aggregate -- exactly the
    kind of subtle evaluation-design bug the task's own sanity test
    ("oracle >= forecast_greedy for every job") is designed to catch.
    Oracle therefore compares the TRUE, fully-integrated realized cost
    (via ctx.realized_cost_at, perfect information only oracle is
    allowed to use) of running now vs. the best remaining future
    candidate -- this makes it a genuine per-job upper bound by
    construction, since it directly optimizes the same objective
    realized_carbon_cost() scores it on.

    forecast_greedy and uncertainty_aware are NOT changed to match --
    comparing point forecasts (rather than forecasting the full
    integrated cost of every candidate) is a realistic simplification
    for a deployable heuristic, and is left as-is deliberately.

    Implemented with the SAME step-by-step interface as the other
    schedulers (for a uniform simulation loop), but this is
    mathematically equivalent to picking the single globally-best
    feasible start time at arrival (same argument as before, just with
    "actual point intensity" replaced by "true integrated cost"): the
    first t at which cost_now is <= every remaining future candidate's
    cost is, by definition, the point achieving the global minimum
    integrated cost over the whole window.
    """
    if not remaining_times:
        return True
    cost_now = ctx.realized_cost_at(t, job)
    best_future_cost = min(ctx.realized_cost_at(ft, job) for ft in remaining_times)
    return cost_now <= best_future_cost


def make_uncertainty_aware(risk_weight: float):
    """
    Factory for the thesis's proposed scheduler, parameterized by a
    RISK-TOLERANCE WEIGHT in [0, 1] -- this is the actual contribution,
    not the single fixed (risk_weight=1.0) policy Step 5 first tried.

    At each step: find t* = the remaining future time with the lowest
    calibrated q50 forecast (the single most attractive future
    candidate, same rule as forecast_greedy). Interpolate a
    RISK-ADJUSTED bound between its median and its CQR-calibrated
    worst case:

        threshold = q50(t*) + risk_weight * (q90(t*) - q50(t*))

      - risk_weight=0.0: threshold = q50(t*) -- trusts the median
        forecast at face value, with no hedge against uncertainty at
        all. This is EXACTLY forecast_greedy's decision rule restated
        (not just "equivalent in spirit"): forecast_greedy runs now iff
        actual(t) <= min_future_q50, and since t* is chosen as the
        argmin of q50 among the same remaining candidates, q50(t*) IS
        min_future_q50 -- same value, same comparison, same outcome for
        every job and every step. Verified exactly (within floating-
        point tolerance) in tests/test_schedulers.py.
      - risk_weight=1.0: threshold = q90(t*) -- the fully worst-case
        bound, reproducing Step 5's original fixed `uncertainty_aware`
        exactly (kept below as `make_uncertainty_aware(1.0)`, aliased
        to the old name for Step 5's results to stay reproducible).
      - 0 < risk_weight < 1: a genuine intermediate hedge -- willing to
        bet on the future being at least this much better than its
        median forecast, but not necessarily as good as its full
        calibrated worst case.

    Decision: "wait" if threshold < actual(t) (even the risk-adjusted
    bound beats running now for certain); otherwise "run now". Every
    other aspect (re-evaluated at each step, never commits to t*
    upfront) is unchanged from the risk_weight=1.0 version.
    """
    def scheduler(t, job, remaining_times, ctx) -> bool:
        if not remaining_times:
            return True
        horizons = [_horizon_hours(t, ft) for ft in remaining_times]
        best_h = min(horizons, key=lambda h: ctx.forecast(t, h)["q50"])
        f = ctx.forecast(t, best_h)
        thresh = f["q50"] + risk_weight * (f["q90"] - f["q50"])
        if thresh < ctx.actual_at(t):
            return False  # wait
        return True  # run now

    scheduler.__name__ = f"uncertainty_aware_risk_{risk_weight}"
    scheduler.risk_weight = risk_weight
    return scheduler


# Step 5's original fixed policy, kept as an alias so its results stay
# exactly reproducible -- see make_uncertainty_aware's docstring.
uncertainty_aware = make_uncertainty_aware(1.0)


# ----------------------------------------------------------------------
# Step 7: CVaR-based scheduler
# ----------------------------------------------------------------------
_QUANTILE_LEVELS = (0.1, 0.5, 0.9)


def quantile_function_knots(q10: float, q50: float, q90: float):
    """
    Approximates a candidate's carbon-intensity distribution from its
    three CQR-calibrated quantiles as a PIECEWISE-LINEAR QUANTILE
    FUNCTION Q(p) (equivalently, a piecewise-uniform density), returned
    as knots (p, Q(p)) spanning p in [0, 1]:

        (0.1, q10) -- (0.5, q50) -- (0.9, q90)   interpolated linearly
        tails [0, 0.1] and [0.9, 1] extended linearly with the slope of
        the adjacent inner segment, i.e.
            Q(0) = q10 - (q50 - q10) / 4
            Q(1) = q90 + (q90 - q50) / 4

    The three quantiles are first sorted (monotone rearrangement), so a
    crossed forecast (e.g. q10 > q50) still yields a valid,
    non-decreasing Q(p).

    LIMITATIONS -- read before interpreting any CVaR number:
      - Three quantiles say nothing about the tail SHAPE beyond q90.
        Linear extrapolation gives a BOUNDED upper tail (max =
        q90 + (q90 - q50)/4): real grid carbon intensity can have
        heavier tails (sudden low-wind spells), which this cannot
        represent. A heavier-tailed choice (e.g. exponential beyond
        q90) would make CVaR strictly larger than the values used here.
      - Inside [0.1, 0.9] the density is assumed uniform between
        adjacent knots, which is the least-informative choice given
        only three points, not an estimate of the true shape.
      - CQR calibrates only the 80% INTERVAL [q10, q90] to marginal
        coverage; it does not calibrate the q50 or any other level, and
        the coverage is marginal across all origins, not per-candidate.
      - It is a distribution of POINT intensity at the candidate's
        start, not of the job's integrated cost over its duration --
        the same simplification forecast_greedy and uncertainty_aware
        already make (see oracle()'s design note).
    """
    lo, mid, hi = sorted((q10, q50, q90))
    p = np.array([0.0, 0.1, 0.5, 0.9, 1.0])
    q = np.array([lo - (mid - lo) / 4.0, lo, mid, hi, hi + (hi - mid) / 4.0])
    return p, q


def cvar(forecast: dict, alpha: float) -> float:
    """
    CVaR_alpha of a candidate's (approximate) carbon-intensity
    distribution: the expected intensity in the WORST alpha-fraction of
    outcomes, where "worst" means HIGHEST intensity (upper tail):

        CVaR_alpha = (1/alpha) * integral_{1-alpha}^{1} Q(p) dp

    with Q from `quantile_function_knots`. Computed exactly (Q is
    piecewise linear, so the trapezoid rule over its knots plus the
    1-alpha cut point is exact, not a numerical approximation).

    Consequence worth stating up front: for any alpha <= 0.5 the whole
    upper tail [1-alpha, 1] lies on the single linear piece through
    (0.5, q50) and (0.9, q90), so

        CVaR_alpha = Q(1 - alpha/2) = q50 + (1.25 - 1.25 * alpha) * (q90 - q50)

    and in particular CVaR_0.2 = q90 EXACTLY, and q10 plays no role.
    Under this three-quantile approximation, CVaR at the calibration
    alpha adds no tail information beyond q90 itself.
    """
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    p, q = quantile_function_knots(forecast["q10"], forecast["q50"], forecast["q90"])
    cut = 1.0 - alpha
    q_cut = float(np.interp(cut, p, q))
    keep = p > cut
    pp = np.concatenate([[cut], p[keep]])
    qq = np.concatenate([[q_cut], q[keep]])
    return float(np.sum((pp[1:] - pp[:-1]) * (qq[1:] + qq[:-1]) / 2.0) / alpha)


def make_cvar_aware(alpha: float, risk_weight: float):
    """
    Step 7: CVaR-based alternative to make_uncertainty_aware, testing
    whether a different DECISION RULE can shift the risk-tolerance
    frontier rather than just move along it.

    Two differences from make_uncertainty_aware:
      1. Risk measure: CVaR_alpha (expected intensity in the worst
         alpha-fraction of outcomes, see `cvar`) instead of the single
         q90 quantile.
      2. Candidate selection: evaluates EVERY remaining feasible
         candidate, not only the argmin-q50 one. For each:

             risk_adjusted = (1 - risk_weight) * q50 + risk_weight * CVaR_alpha

         and takes the candidate with the lowest risk_adjusted cost.
         A candidate with a slightly worse median but a much tighter
         forecast can therefore win -- uncertainty_aware never even
         looks at it.

    Decision: wait if that best risk_adjusted cost < actual(t), else run
    now (same tie-breaking direction as make_uncertainty_aware).

    IMPORTANT, given `cvar`'s docstring: at alpha=0.2 (the calibration
    alpha) CVaR_0.2 = q90 under the piecewise-linear approximation, so
    the per-candidate score is q50 + risk_weight * (q90 - q50) -- the
    SAME bound uncertainty_aware applies, at the same risk_weight. The
    only behavioral difference at alpha=0.2 is point 2 (min over all
    candidates vs. the bound at argmin-q50 only). Since
    min_c score(c) <= score(argmin-q50), this policy waits at least as
    often as uncertainty_aware at the same risk_weight, never less.

    risk_weight=0.0 reduces to min over candidates of q50, i.e. exactly
    forecast_greedy (verified in tests/test_schedulers.py).
    """
    def scheduler(t, job, remaining_times, ctx) -> bool:
        if not remaining_times:
            return True
        best = np.inf
        for ft in remaining_times:
            f = ctx.forecast(t, _horizon_hours(t, ft))
            score = (1 - risk_weight) * f["q50"] + risk_weight * cvar(f, alpha)
            best = min(best, score)
        if best < ctx.actual_at(t):
            return False  # wait
        return True  # run now

    scheduler.__name__ = f"cvar_aware_a{alpha}_risk_{risk_weight}"
    scheduler.alpha = alpha
    scheduler.risk_weight = risk_weight
    return scheduler


SCHEDULERS = {
    "run_immediately": run_immediately,
    "threshold": threshold,
    "forecast_greedy": forecast_greedy,
    "oracle": oracle,
    "uncertainty_aware": uncertainty_aware,
}
