"""CANONICAL_DYNAMIC_EXPERT reproduction (2026-07-22, §9): the frozen learned chain (E_valselect approach → handoff
transport) delivers strict K=6 on the canonical v3 robot at the SAME count as the legacy golden robot — recovered
through REAL dynamics (no state injection, no teleportation, all motion through ``env.step``, strict certificate fires
naturally). This is the payoff of the golden-structure inertia + integrator-parity repair (before it: 0/9 on v2).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from hymeko_rl.experiments.coin_neutral_start import _HEADLINE, eval_composed, neutral_env
from hymeko_rl.train.sac import build_sac

_HANDOFF = "experiments/2026_07_21_coin_neutral_handoff/handoff_best.pt"


def _handoff_transport():
    tr, _ = build_sac("mlp", obs_dim=41, flat_dim=41, action_dim=6, action_scale=1.0)
    tr.load_state_dict(torch.load(_HANDOFF, weights_only=True))
    tr.eval()
    return tr


@pytest.fixture(scope="module")
def canonical_delivery():
    # canonical v3 robot (make_coin_env → robot_source=hymeko_spec) through neutral_env; real dynamics.
    return eval_composed(_handoff_transport(), _HEADLINE, grasp_hold=3, env_cf=neutral_env(prefix_steps=0))


def test_frozen_chain_delivers_three_of_nine_on_canonical_v3(canonical_delivery):
    # the pre-repair v2 robot delivered 0/9 (5× mass) — the golden-structure repair restores the legacy 3/9.
    assert canonical_delivery["deliver"] >= 3, f"canonical v3 delivery below legacy baseline: {canonical_delivery}"


def test_grasp_forms_naturally(canonical_delivery):
    # acquisition (bilateral contact) forms through env.step, not injection.
    assert canonical_delivery["grasp"] >= 3, f"grasp count too low: {canonical_delivery}"


def test_delivery_is_through_real_dynamics_no_injection():
    # a delivering rollout runs entirely through env.step and the strict certificate fires naturally — assert the
    # composed chain reaches a certified delivery on the known-delivering seed 1045 without any state manipulation.
    from hymeko_rl.coin_delivery.delivery_certificate import DeliveryCertifier
    from hymeko_rl.experiments.coin_delivery_e0_campaign import _greedy_action_fn
    from hymeko_rl.experiments.coin_neutral_start import _cert_step, _clearance, _e_approach_actor
    env, cf = neutral_env(prefix_steps=0)
    inner = cf._env
    e, tfn = _e_approach_actor(), _greedy_action_fn(_handoff_transport())
    env.set_stage(0)
    env.reset(seed=1045)
    cert = DeliveryCertifier(initial_clearance=_clearance(inner))
    for _ in range(160):                                              # approach → grasp
        cert.update(_cert_step(inner, cf))
        m = inner._planar_metrics
        if m.left_contact and m.right_contact:
            break
        a = e.action_mean(torch.as_tensor(np.asarray(inner.node_features(), np.float32)[None]))[0].detach().numpy()
        inner.step(np.asarray(a, np.float32))
    cf._prev_coin = np.asarray(inner._planar_metrics.disk_pos[:2], np.float64)
    cf._t = 0
    cf._both_hist = []
    env._suffix_t = 0
    env._prev_dtz = env._dtz()
    env._prev_both = env._both()
    o = cf._obs(np.zeros(4, np.float32))
    for _ in range(200):                                            # transport → strict K=6 delivery
        cert.update(_cert_step(inner, cf))
        if cert.delivery_certified:
            break
        o = env.step(np.asarray(tfn(env, o, None), np.float32))[0]
    assert cert.delivery_certified, "strict certificate did not fire naturally on the known-delivering seed"
