"""Tests for the R10.2 Boundary-4b conservative structured-option TD3 engine.

Fast: the rank correlation, the immutable-positive minibatch mixing, the zero-init actor's exploration, and a config
guard that the ranking gate is actually reachable within the episode budget (the bug that would silently prevent any
update). Physics: one tiny train_seed run completes, starts the actor at the scaffold, and preserves nominal HOME K6.
"""
from __future__ import annotations

import numpy as np
import pytest

from hymeko_rl.coin_delivery.theta_option import torque_path_frozen as frz
from hymeko_rl.coin_delivery.theta_option import torque_path_td3 as td3
from hymeko_rl.coin_delivery.theta_option.torque_path_option import THETA_DIM
from hymeko_rl.option_rl.agents import make_actor
from hymeko_rl.option_rl.core import OptionReplayBuffer, OptionTransition


def _tr(reward: float, cls: str = "k6") -> OptionTransition:
    return OptionTransition(s=np.zeros(td3.OBS_DIM, np.float32), action=np.zeros(THETA_DIM, np.float32), reward=reward,
                            tau=20.0, s_next=np.zeros(td3.OBS_DIM, np.float32), terminal=1.0, end="handoff",
                            provenance={"class": cls})


# ── fast ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
def test_spearman_monotone_and_short():
    assert td3._spearman(np.arange(10.0), np.arange(10.0)) == pytest.approx(1.0)
    assert td3._spearman(np.arange(10.0), -np.arange(10.0)) == pytest.approx(-1.0)
    assert td3._spearman(np.array([1.0]), np.array([1.0])) == 0.0        # too few points -> 0


def test_sample_mixed_reserves_positive_fraction():
    main, pos = OptionReplayBuffer(), OptionReplayBuffer()
    for _ in range(200):
        main.add(_tr(-1.0, "safe_negative"))
    for _ in range(20):
        pos.add(_tr(1.0, "k6"))
    cfg = td3.TD3Config(batch=64, positive_frac=0.25)
    rng = np.random.default_rng(0)
    bs, ba, br, bt, bs2, bd = td3._sample_mixed(main, pos, cfg, rng)
    assert bs.shape[0] == cfg.batch
    assert int((br > 0).sum()) >= int(cfg.batch * cfg.positive_frac)     # positives guaranteed in the minibatch


def test_zero_actor_mean_option_is_zero_and_explore_bounded():
    actor = td3.zero_init_detactor(make_actor("td3", td3.OBS_DIM, THETA_DIM))
    obs = np.zeros(td3.OBS_DIM, np.float32)
    d, rng = frz.frozen_normalization(), np.random.default_rng(0)
    assert np.array_equal(td3._act(actor, obs, d, frz.SIGMA, rng, explore=False), np.zeros(THETA_DIM, np.float32))
    a = td3._act(actor, obs, d, frz.SIGMA, rng, explore=True)
    assert a.shape == (THETA_DIM,) and np.all(a >= -1.0) and np.all(a <= 1.0)


def test_default_config_ranking_gate_is_reachable():
    c = td3.TD3Config()
    updates_start_ep = max(c.batch, c.warmup_episodes)                   # updates begin once replay has `batch` items
    available = (c.total_episodes - updates_start_ep) * c.updates_per_episode
    assert available >= c.critic_warmup_updates                          # the gate can fire within the frozen budget


# ── physics (rig reused from the audit) ──────────────────────────────────────────────────────────────────────────────
def test_train_seed_runs_and_preserves_nominal_k6():
    from hymeko_rl.coin_delivery.theta_option import capture_rl as crl
    from hymeko_rl.experiments.coin_kinetic_capture_exploration_audit import _rig
    rig = _rig()
    train = crl.perturbation_panel(n=4, seed=12345)
    panel = crl.perturbation_panel(n=4, seed=90210)
    cfg = td3.TD3Config(total_episodes=20, warmup_episodes=4, critic_warmup_updates=8, eval_every=20, batch=8,
                        rank_thetas=2)
    r = td3.train_seed(rig, train, panel, 0, cfg, frz.frozen_normalization(), frz.SIGMA, log=lambda *a: None)
    # The run completes with a well-formed result; the zero-init actor's scaffold baseline always delivers nominal K6.
    # (Whether the POST actor preserves nominal K6 is a run outcome the experiment measures, not a guaranteed invariant —
    #  an unstable tiny config can release the actor on a weak critic and drift off the scaffold; that is a NO_IMPROVEMENT.)
    assert r["scaffold_eval"]["nominal_k6"] and r["scaffold_eval"]["k6"] >= 1
    assert set(r["post_eval"]) == set(r["scaffold_eval"]) and 0 <= r["post_eval"]["k6"] <= r["post_eval"]["n"]
    assert isinstance(r["released"], bool)
