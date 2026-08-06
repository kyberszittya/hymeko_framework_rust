"""Unit tests for demo-mix tagged pools + ratio mixing."""
from __future__ import annotations

import numpy as np
import pytest

from hymeko_rl.train.demo_mix import TaggedPools, mix_pools


def _pools(n_s: int, n_d: int) -> TaggedPools:
    return TaggedPools(
        sustained_obs=np.zeros((n_s, 6, 8), np.float32), sustained_acts=np.ones((n_s, 4), np.float32),
        deliver_obs=np.zeros((n_d, 6, 8), np.float32), deliver_acts=np.full((n_d, 4), 2.0, np.float32),
        n_episodes=10, n_delivered=8)


def test_mix_ratio_respected() -> None:
    pools = _pools(500, 500)
    obs, acts = mix_pools(pools, 0.25, total=400, seed=0)
    assert obs.shape == (400, 6, 8) and acts.shape == (400, 4)
    # sustained acts are 1.0, deliver acts are 2.0 → count how many rows are all-ones
    n_sustained = int(np.all(np.isclose(acts, 1.0), axis=1).sum())
    assert abs(n_sustained - 100) <= 5   # ~25% of 400 (sampling is exact count here, no randomness in the split)


def test_mix_endpoints() -> None:
    pools = _pools(300, 300)
    o0, a0 = mix_pools(pools, 0.0, total=200, seed=1)
    assert np.all(np.isclose(a0, 2.0))          # all delivery-completion
    o1, a1 = mix_pools(pools, 1.0, total=200, seed=1)
    assert np.all(np.isclose(a1, 1.0))          # all sustained


def test_mix_empty_sustained_pool_falls_back_to_deliver() -> None:
    pools = _pools(0, 300)
    obs, acts = mix_pools(pools, 0.75, total=200, seed=2)   # asks for 75% sustained but none exist
    assert obs.shape[0] == 200 and np.all(np.isclose(acts, 2.0))


def test_mix_invalid_frac_raises() -> None:
    with pytest.raises(ValueError):
        mix_pools(_pools(10, 10), 1.5, total=10)


def test_tagged_pools_summary() -> None:
    p = _pools(120, 380)
    s = p.summary()
    assert s == {"n_episodes": 10, "n_delivered": 8, "n_sustained_states": 120, "n_deliver_states": 380}
