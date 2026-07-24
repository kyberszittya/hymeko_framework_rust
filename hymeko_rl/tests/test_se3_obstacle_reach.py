"""6D-1 — SE(3) obstacle reach (EE-route-selection benchmark) tests.

Covers the env (obstacle injection + seating + EE-AABB collision + direct-path-blocking), the route-mode machinery
(via placement, execution-based feasibility, the three allocation strategies as prob-reweightings, top-k limiting,
mode non-crossing), and — the validity gate in miniature — single-head@infeasible-route FAILS while a K-mode covering
a feasible route RECOVERS on the same state/budget/executor. Physics kept minimal (few states, short horizons) for speed.
"""
import numpy as np
import pytest

from hymeko_rl.env.se3_obstacle_reach_env import SE3ObstacleReachEnv
from hymeko_rl.env.se3_reach_option import (
    ROUTE_DIRS, RouteModeProposal, RouteOptionScorer, route_execution_feasible, route_via)
from hymeko_rl.option_rl import MultimodalBudgetSearch, allocate_budget


def _env(**kw):
    d = dict(control_mode="position", max_steps=300, reach_thresh=0.06, ang_thresh=0.4, min_separation=0.16)
    d.update(kw)
    return SE3ObstacleReachEnv(**d)


class _ViaGen:
    def sample(self, center, n, rng):
        c = np.asarray(center, np.float64)
        return c[None, :] if n == 1 else c + rng.normal(0, 0.02, (int(n), len(c)))


# ── env: obstacle injection, seating, collision, direct-path blocking ──
def test_obstacle_injected_and_seated_between_start_and_goal():
    env = _env()
    _o, info = env.reset(seed=0)
    assert env._obstacle_gid >= 0
    mid = 0.5 * (env._start_ee + env._target)
    assert np.allclose(env.obstacle_center(), mid, atol=1e-5)   # seated at the start↔goal midpoint
    assert info["direct_blocked"] is True                       # midpoint obstacle blocks the straight EE path


def test_ee_in_obstacle_aabb():
    env = _env()
    env.reset(seed=0)
    c = env.obstacle_center()
    assert env.ee_in_obstacle(c)                                # centre is inside
    assert not env.ee_in_obstacle(c + np.array([0.0, 0.5, 0.0]))  # far outside


def test_direct_path_blocked_across_seeds():
    env = _env()
    assert all((env.reset(seed=s), env.direct_path_blocked())[1] for s in range(6))


# ── route machinery: via, allocation reweighting, top-k, non-crossing ──
def test_route_via_is_lateral_offset_from_midpoint():
    env = _env()
    env.reset(seed=1)
    mid = 0.5 * (env._start_ee + env._target)
    via = route_via(env, ROUTE_DIRS["over"], offset=0.14)
    assert np.allclose(via - mid, np.array([0.0, 0.0, 0.14]), atol=1e-5)


def test_allocation_strategies_are_prob_reweightings():
    env = _env()
    env.reset(seed=1)
    dirs = list(ROUTE_DIRS.values())
    eq = RouteModeProposal(env, dirs, "equal").modes(None)
    assert all(abs(m.prob - 0.25) < 1e-6 for m in eq)          # equal-minimum: uniform probs
    tp = RouteModeProposal(env, dirs, "top_probe").modes(None)
    assert abs(tp[0].prob - 0.97) < 1e-6 and all(abs(m.prob - 0.01) < 1e-6 for m in tp[1:])  # top-refined + probes
    pr = RouteModeProposal(env, dirs, "prob").modes(None)
    assert pr[0].prob >= pr[-1].prob                            # prob-weighted: descending prior


def test_top_probe_allocation_realizes_9_1_1_1_at_b12():
    assert allocate_budget([0.97, 0.01, 0.01, 0.01], 12) == [9, 1, 1, 1]
    assert allocate_budget([0.25, 0.25, 0.25, 0.25], 12) == [3, 3, 3, 3]


def test_top_k_limits_modes():
    env = _env()
    env.reset(seed=1)
    dirs = list(ROUTE_DIRS.values())
    assert len(RouteModeProposal(env, dirs, "prob", k=1).modes(None)) == 1   # single-head
    assert len(RouteModeProposal(env, dirs, "prob", k=2).modes(None)) == 2


def test_mode_candidates_stay_in_route_family_basin():
    env = _env()
    env.reset(seed=1)
    mid = 0.5 * (env._start_ee + np.asarray(env._target, np.float32))
    for nm, d in ROUTE_DIRS.items():
        d = np.asarray(d, np.float32)
        base = route_via(env, d)
        cands = _ViaGen().sample(base, 24, np.random.default_rng(0))
        assert np.all(np.sign((cands - mid) @ d) == np.sign((base - mid) @ d)), nm   # never cross basins


# ── validity gate in miniature: single-head@wrong FAILS, K-mode RECOVERS ──
def _first_eligible(env, scan=30):
    for s in range(scan):
        env.reset(seed=s)
        if not env.direct_path_blocked():
            continue
        feas = {nm: route_execution_feasible(env, d, seed=100 + s) for nm, d in ROUTE_DIRS.items()}
        good = [nm for nm, f in feas.items() if f]
        bad = [nm for nm, f in feas.items() if not f]
        if good and bad:
            return s, good, bad
    return None


def test_single_head_wrong_route_fails_kmode_recovers():
    env = _env()
    found = _first_eligible(env)
    assert found is not None, "no eligible state found in scan"
    s, good, bad = found
    env.reset(seed=s)
    obs = env.node_features().reshape(-1)
    sh = MultimodalBudgetSearch(_ViaGen(), RouteOptionScorer(env), budget=12).select(
        RouteModeProposal(env, [ROUTE_DIRS[bad[0]]], "prob"), obs, np.random.default_rng(1))
    env.reset(seed=s)
    obs = env.node_features().reshape(-1)
    km = MultimodalBudgetSearch(_ViaGen(), RouteOptionScorer(env), budget=12).select(
        RouteModeProposal(env, list(ROUTE_DIRS.values()), "equal"), obs, np.random.default_rng(1))
    assert km.outcome["success"] == 1 and sh.outcome["success"] == 0   # K-mode recovers where the wrong single route fails


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
