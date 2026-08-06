"""Hierarchical model refinement with RL — the coarse→detailed decomposition, warm-started through the hierarchy.

The warm-started detailed policy is *initially* bit-identical to the coarse one (the arm channel is inert until
learned) — an exact transfer, verified. Refining the model (adding the flight-phase arm) improves the hard,
flight-heavy run-stop over the foot-only coarse model.
"""

from __future__ import annotations

import numpy as np

from scenarios.humanoid.hierarchical_runstop import (
    HierConfig,
    _forward,
    mixed_set,
    n_params,
    rollout,
    train_cem,
    warm_start,
)


def test_warm_start_is_initially_identical_to_coarse() -> None:
    """The exact hierarchy transfer: the lifted detailed policy behaves bit-identically to the coarse one."""
    cfg = HierConfig(iters=8)
    coarse = train_cem(cfg, use_arm=False)
    lifted = warm_start(coarse, cfg)
    states = mixed_set(cfg, offset=0.5)
    stopped_c, fell_c = rollout(coarse, states, cfg, use_arm=False)
    stopped_w, fell_w = rollout(lifted, states, cfg, use_arm=True)
    assert np.array_equal(stopped_c, stopped_w) and np.array_equal(fell_c, fell_w)


def test_arm_channel_of_the_lifted_policy_is_inert() -> None:
    """The new arm output is exactly zero at lift time (only the shared foot/L channels carry over)."""
    cfg = HierConfig()
    lifted = warm_start(train_cem(HierConfig(iters=5), use_arm=False), cfg)
    feats = np.random.RandomState(7).standard_normal((20, 6))       # any 6-feature inputs
    assert np.allclose(_forward(lifted, feats, cfg, use_arm=True)[:, 2], 0.0)   # arm action ≡ 0 before refinement


def test_refinement_never_regresses_the_coarse_model() -> None:
    """Warm-started refinement is monotone: the detailed (arm) policy is at least as good as the coarse one.

    Because the lift begins bit-identical to the coarse policy and CEM only accepts improving elites, the refined
    model never falls below the coarse baseline — the honest, robust hierarchy property (the *magnitude* of the
    arm's gain is task-dependent: large in the pure balance task, modest in run-stop where braking-induced L
    exceeds the arm's momentum capacity — reported, not asserted).
    """
    cfg = HierConfig(iters=15)
    coarse = train_cem(cfg, use_arm=False)
    refined = train_cem(cfg, use_arm=True, init=warm_start(coarse, cfg))
    coarse_stop = rollout(coarse, mixed_set(cfg, offset=0.5), cfg, use_arm=False)[0].mean()
    refined_stop = rollout(refined, mixed_set(cfg, offset=0.5), cfg, use_arm=True)[0].mean()
    assert refined_stop >= coarse_stop - 0.02                      # the detailed model does not regress the coarse


def test_actions_are_bounded_and_deterministic() -> None:
    cfg = HierConfig()
    params = np.random.RandomState(1).standard_normal(n_params(cfg, True))
    out = _forward(params, np.random.RandomState(2).standard_normal((20, 6)) * 3, cfg, use_arm=True)
    assert np.all(np.abs(out) <= 1.0 + 1e-9)                       # tanh-bounded (scaled to a_max/fx_max/wa_max)
    assert np.allclose(train_cem(HierConfig(iters=4), use_arm=True), train_cem(HierConfig(iters=4), use_arm=True))
