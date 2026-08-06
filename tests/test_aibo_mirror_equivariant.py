"""Phase B — the Reynolds symmetrization produces an EXACTLY mirror-equivariant policy.

Locks the mathematical contract the central-hypothesis test relies on: mirror_obs/mirror_act are
order-2 involutions, and ``symmetrize(f)`` satisfies ``a(g·s) == g·a(s)`` to float precision for an
ARBITRARY base ``f`` (so any residual asymmetry in the measured crab is the DYNAMICS, never a leaky
equivariance).
"""

from __future__ import annotations

import numpy as np

from scenarios.aibo.mirror_equivariant import equivariance_residual, symmetrize
from scenarios.aibo.residual_trot import ResidualTrotEnv

_MOBS = ResidualTrotEnv.mirror_obs
_MACT = ResidualTrotEnv.mirror_act


def test_mirror_obs_is_involution() -> None:
    rng = np.random.default_rng(0)
    o = rng.standard_normal(9).astype(np.float32)
    assert np.allclose(_MOBS(_MOBS(o)), o)


def test_mirror_act_is_involution() -> None:
    rng = np.random.default_rng(1)
    a = rng.standard_normal(4).astype(np.float32)
    assert np.allclose(_MACT(_MACT(a)), a)


def test_symmetrize_is_exactly_equivariant_for_arbitrary_base() -> None:
    rng = np.random.default_rng(2)
    w = rng.standard_normal((4, 9))                      # an arbitrary (asymmetric) linear base policy

    def base(o: np.ndarray) -> np.ndarray:
        return np.tanh(w @ np.asarray(o, np.float64)).astype(np.float32)

    sym = symmetrize(base, _MOBS, _MACT)
    for _ in range(20):
        o = rng.standard_normal(9).astype(np.float32)
        assert equivariance_residual(sym, _MOBS, _MACT, o) < 1e-6


def test_symmetrized_of_symmetric_base_is_unchanged() -> None:
    # if the base is already equivariant, symmetrization is a no-op (Reynolds average is a projection)
    rng = np.random.default_rng(3)

    def base(o: np.ndarray) -> np.ndarray:
        return np.zeros(4, np.float32)                   # the trivially-equivariant zero policy

    sym = symmetrize(base, _MOBS, _MACT)
    o = rng.standard_normal(9).astype(np.float32)
    assert np.allclose(sym(o), base(o))
