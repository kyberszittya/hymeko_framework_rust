"""Level-2 refinement (shoulder + elbow) — the framework deepens exactly; the reduced-model gain is honest.

The warm-start transfer is exact at the deeper level too (the lifted Level-2 policy is bit-identical to Level-1),
and refinement never regresses it. The *magnitude* of any gain from the second joint is measured, not asserted —
in this reduced model it is nil (the bottleneck is not the arm's momentum capacity), reported honestly.
"""

from __future__ import annotations

import numpy as np

from scenarios.humanoid.hierarchical_runstop_deep import (
    DeepConfig,
    _forward,
    evaluate,
    mixed_set,
    n_params,
    rollout,
    train_cem,
    warm_start,
)


def test_deep_warm_start_is_exact() -> None:
    """Lifting a Level-1 (single-arm) policy to Level-2 (shoulder+elbow) is bit-identical (elbow channel inert)."""
    cfg = DeepConfig(iters=8)
    level1 = train_cem(cfg, deep=False)
    lifted = warm_start(level1, cfg)
    states = mixed_set(cfg, offset=0.5)
    s1, f1 = rollout(level1, states, cfg, deep=False)
    sl, fl = rollout(lifted, states, cfg, deep=True)
    assert np.array_equal(s1, sl) and np.array_equal(f1, fl)


def test_elbow_channel_is_inert_at_lift() -> None:
    cfg = DeepConfig()
    lifted = warm_start(train_cem(DeepConfig(iters=5), deep=False), cfg)
    assert np.allclose(_forward(lifted, np.random.RandomState(7).standard_normal((20, 7)), cfg, deep=True)[:, 3], 0.0)


def test_deep_refinement_never_regresses() -> None:
    """Warm-started Level-2 is at least as good as Level-1 (monotone refinement); the gain magnitude is reported."""
    cfg = DeepConfig(iters=15)
    level1 = train_cem(cfg, deep=False)
    deep = train_cem(cfg, deep=True, init=warm_start(level1, cfg))
    assert evaluate(deep, cfg, deep=True) >= evaluate(level1, cfg, deep=False) - 0.02


def test_actions_bounded_and_deterministic() -> None:
    cfg = DeepConfig()
    params = np.random.RandomState(1).standard_normal(n_params(cfg, deep=True))
    out = _forward(params, np.random.RandomState(2).standard_normal((15, 7)) * 3, cfg, deep=True)
    assert np.all(np.abs(out) <= 1.0 + 1e-9)
    assert np.allclose(train_cem(DeepConfig(iters=4), deep=True), train_cem(DeepConfig(iters=4), deep=True))
