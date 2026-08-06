"""Exact mirror-equivariance by Reynolds symmetrization — Phase B of the symmetry-closure campaign.

The omni crab is one-sided (+y reached, -y not). A left-right MIRROR-EQUIVARIANT policy cannot be
one-sided *by construction*: if it reaches +y it must, by symmetry, apply the mirrored recipe for -y.
The cleanest exact equivariance is the Reynolds average over the order-2 mirror group
``G = {e, g}`` (``g`` = swap left/right + flip lateral sign, an involution ``g = g^{-1}``):

    a(s) = 1/2 ( f(s) + g_act( f( g_obs(s) ) ) )

This ``a`` is EXACTLY equivariant: ``a(g_obs(s)) == g_act(a(s))`` (proof: substitute and use
``g^2 = e``). It wraps ANY base policy ``f`` — MLP, HSiKAN, or the sunflower-per-node HSiKAN — so the
sunflower's ``S_4`` structural prior and this ``Z_2`` exact equivariance compose. The campaign's central
hypothesis is that this PRESERVES a symmetry but cannot MANUFACTURE one the dynamics lacks: over the
asymmetric diagonal-trot scaffold the mirrored -y recipe does not produce a -y crab, so symmetrization
either leaves -y unreached or cancels the working +y (symmetric mediocrity) — it does NOT yield a
two-sided reach. Over a symmetric (bound) scaffold the mirror IS a true symmetry and the two sides match.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

Greedy = Callable[[np.ndarray], np.ndarray]
Mirror = Callable[[np.ndarray], np.ndarray]


def symmetrize(greedy: Greedy, mirror_obs: Mirror, mirror_act: Mirror) -> Greedy:
    """Wrap a greedy action fn into its exact mirror-equivariant Reynolds average.

    # Preconditions ``mirror_obs`` and ``mirror_act`` are involutions of the obs/action spaces of
    ``greedy`` (the mirror group is order 2). # Postconditions the returned fn ``a`` satisfies
    ``a(mirror_obs(s)) == mirror_act(a(s))`` up to float error (see :func:`equivariance_residual`).
    """

    def fn(obs: np.ndarray) -> np.ndarray:
        a = np.asarray(greedy(obs), np.float64)
        a_mir = np.asarray(mirror_act(greedy(mirror_obs(obs))), np.float64)
        return (0.5 * (a + a_mir)).astype(np.float32)

    return fn


def equivariance_residual(sym_greedy: Greedy, mirror_obs: Mirror, mirror_act: Mirror,
                          obs: np.ndarray) -> float:
    """Max-abs violation of ``a(g·s) == g·a(s)`` for the symmetrized policy at ``obs`` (should be ~0)."""
    lhs = np.asarray(sym_greedy(mirror_obs(obs)), np.float64)
    rhs = np.asarray(mirror_act(sym_greedy(obs)), np.float64)
    return float(np.max(np.abs(lhs - rhs)))
