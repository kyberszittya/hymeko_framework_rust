"""Option-parameter CEM: box bounds, elitism floor (never worse than the scripted default), θ stays in the box."""
from __future__ import annotations

import numpy as np

from hymeko_rl.env.planar_grasp_env import PlanarGraspEnv
from hymeko_rl.train.option_search import (
    PARAM_NAMES,
    PHASE_PUSH_HI,
    PHASE_PUSH_LO,
    THETA_SCRIPTED,
    OptionSearchConfig,
    _objective,
    option_cem,
)


def test_scripted_theta_in_box():
    assert np.all(THETA_SCRIPTED >= PHASE_PUSH_LO) and np.all(THETA_SCRIPTED <= PHASE_PUSH_HI)
    assert len(PARAM_NAMES) == 5 == len(THETA_SCRIPTED)


def test_objective_rewards_coverage_penalises_exploit():
    cfg = OptionSearchConfig()
    good = {"delivery": 0.9, "sustained_push_per_ep": 2.0, "ft_progress_in_contact": 0.02,
            "body_progress_in_contact": 0.0, "exploit_rate": 0.0, "arm_body_rate": 0.0}
    exploit = dict(good, exploit_rate=1.0, body_progress_in_contact=0.05)
    assert _objective(good, THETA_SCRIPTED, cfg) > _objective(exploit, THETA_SCRIPTED, cfg)
    # large θ departure is penalised
    far = THETA_SCRIPTED + (PHASE_PUSH_HI - THETA_SCRIPTED)
    assert _objective(good, far, cfg) < _objective(good, THETA_SCRIPTED, cfg)


def test_option_cem_elitism_floor_and_box():
    def make_env():
        return PlanarGraspEnv(robot=None, max_steps=120, difficulty=0.3)
    cfg = OptionSearchConfig(pop=6, elite=2, iters=2, n_eval_eps=2, max_steps=120, log_every=0)
    res = option_cem(make_env, cfg)
    assert res.best_objective >= res.baseline_objective - 1e-9   # scripted θ kept in the population
    theta = np.array(res.theta)
    assert np.all(theta >= PHASE_PUSH_LO - 1e-9) and np.all(theta <= PHASE_PUSH_HI + 1e-9)
    assert len(res.history) == cfg.iters + 1
    assert set(res.theta_named) == set(PARAM_NAMES)
