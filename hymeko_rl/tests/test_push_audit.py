"""Sustained-PUSH audit: window definition (pure) + an end-to-end audit that reproduces the coverage gap."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from hymeko_rl.agents.multichannel_ctde import build_collaborative_offpolicy
from hymeko_rl.env.planar_grasp_env import PlanarGraspEnv
from hymeko_rl.eval.evaluate import greedy_action_fn
from hymeko_rl.eval.push_audit import audit_policy, sustained_push_windows
from hymeko_rl.experiments.galambos_demo import PushDemonstrator
from hymeko_rl.viz.render_planar_gifs import demonstrator_action_fn


def _ctx(both, toward, body_prog, body_contact):
    return SimpleNamespace(both_contact=np.array(both, bool), toward=np.array(toward, float),
                           body_prog_step=np.array(body_prog, float), body_contact=np.array(body_contact, bool))


def _contract():
    return SimpleNamespace(progress_eps=0.002, body_eps=0.005)


def test_sustained_window_requires_length_progress_and_clean_contact():
    n = 12
    # a clean 6-step two-finger run (idx 3..8) with progress and no body contact → qualifies at k=5
    both = [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0]
    toward = [0.0] * 12
    for i in range(3, 9):
        toward[i] = 0.01
    wins = sustained_push_windows(_ctx(both, toward, [0.0] * n, [False] * n), _contract(), k=5)
    assert wins == [(3, 9)]
    # too short (k=7) → none
    assert sustained_push_windows(_ctx(both, toward, [0.0] * n, [False] * n), _contract(), k=7) == []
    # body contact in the window → rejected
    bc = [False] * n
    bc[5] = True
    assert sustained_push_windows(_ctx(both, toward, [0.0] * n, bc), _contract(), k=5) == []
    # no toward-progress → rejected
    assert sustained_push_windows(_ctx(both, [0.0] * n, [0.0] * n, [False] * n), _contract(), k=5) == []


def test_audit_reproduces_coverage_gap():
    """The scripted PushDemonstrator must show materially more sustained-PUSH coverage than the frozen DAgger MLP."""
    def make_env():
        return PlanarGraspEnv(robot=None, max_steps=300, difficulty=0.3)
    env = make_env()
    actor = build_collaborative_offpolicy(env, kind="mlp", hidden=64)[0]
    import torch
    actor.load_state_dict(torch.load("experiments/v2_dagger/FROZEN_selected/mlp_s1_selected_d3.pt", map_location="cpu"))
    actor.eval()

    scripted = audit_policy(make_env, lambda e: demonstrator_action_fn(PushDemonstrator(e)),
                            name="scripted", n_episodes=8, seed0=9000, k_sustained=5)
    dagger = audit_policy(make_env, lambda e: greedy_action_fn(actor),
                          name="dagger", n_episodes=8, seed0=9000, k_sustained=5)
    assert scripted.both_contact_frac > dagger.both_contact_frac
    assert scripted.sustained_push_per_ep >= dagger.sustained_push_per_ep
    assert set(scripted.phase_dist) == {"APPROACH", "CONTACT", "PUSH", "DELIVERY"}
    assert scripted.exploit_rate == 0.0   # the scripted controller must not body-shove
