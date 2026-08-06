"""In-loop hard mirror-equivariant SAC actor — trains a policy that is equivariant BY CONSTRUCTION.

Phase B symmetrized a *trained* policy post-hoc (worked for the MLP's active crab, not the HSiKAN's null
residual). This wraps the base squashed-Gaussian actor so the policy is exactly mirror-equivariant
*during training*: the pre-squash mean is Reynolds-averaged over the order-2 mirror group,

    mu_sym(s) = 1/2 ( mu(s) + G_pre( mu( g_obs(s) ) ) )

with ``G_pre`` the pre-squash action mirror (``tanh`` is odd, so ``G_pre`` = the same permute+sign that the
post-squash action mirror uses, and ``action_scale·tanh(mu_sym)`` is exactly equivariant). The mirror
``(g_obs, G_pre)`` is supplied by the caller — the hand-validated flat mirror, or the one READ FROM THE
HYMEKO STRUCTURE (:mod:`scenarios.aibo.structural_symmetry`). Duck-typed over the base's internals so it
needs no change to the shared SAC module.
"""

from __future__ import annotations

from typing import Callable

import torch

from hymeko_rl.train.sac import _SquashedGaussianActorBase, _squashed_mean, _squashed_sample

TensorMap = Callable[[torch.Tensor], torch.Tensor]

# flat 9-D obs mirror [dist,cos,sin,vx,vy,wz,sinph,cosph,up]: flip sin(herr),vy,wz,sin(ph),cos(ph)
_FLAT_FLIP = torch.tensor([1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0])
# the leg-slot permutation of the omni action [fl,fr,bl,br] -> [fr,fl,br,bl] (== structural action_perm)
_ACT_PERM = (1, 0, 3, 2)


def mirror_obs_flat(s: torch.Tensor) -> torch.Tensor:
    """The validated flat-obs left-right mirror as a batched tensor op (involution)."""
    return s * _FLAT_FLIP.to(dtype=s.dtype, device=s.device)


def mirror_pre_act(mu: torch.Tensor, perm: "tuple[int, ...]" = _ACT_PERM) -> torch.Tensor:
    """Pre-squash action mirror: swap left/right abduction slots + flip sign. Involution; commutes with
    ``tanh`` so it induces the post-squash action mirror ``-a[[perm]]``."""
    return -mu[..., list(perm)]


class MirrorEquivariantActor(_SquashedGaussianActorBase):
    """Wrap a squashed-Gaussian actor into an exactly mirror-equivariant one (mean symmetrized).

    # Preconditions ``mirror_obs`` and ``mirror_pre_act`` are involutions of the base's obs / pre-squash
    action spaces. # Postconditions ``action_mean(g_obs(s)) == g_act(action_mean(s))`` to float precision.
    """

    def __init__(self, base: _SquashedGaussianActorBase, mirror_obs: TensorMap,
                 mirror_pre_act: TensorMap) -> None:  # noqa: D401
        super().__init__()
        self.base = base
        self.action_scale = base.action_scale
        self.action_dim = base.action_dim
        self._mobs = mirror_obs
        self._mpre = mirror_pre_act

    def _feat(self, obs: torch.Tensor) -> torch.Tensor:
        # per-node backbone exposes node_activations; flat backbone is called directly
        b = self.base
        return b._node_acts(obs) if hasattr(b, "_node_acts") else b.backbone(obs)  # type: ignore[attr-defined]

    def _mu_sym(self, obs: torch.Tensor) -> torch.Tensor:
        mu = self.base.mu(self._feat(obs))                     # type: ignore[attr-defined]
        mu_mir = self._mpre(self.base.mu(self._feat(self._mobs(obs))))   # type: ignore[attr-defined]
        return 0.5 * (mu + mu_mir)

    def sample(self, obs: torch.Tensor) -> "tuple[torch.Tensor, torch.Tensor]":
        log_std = self.base.log_std(self._feat(obs))           # type: ignore[attr-defined]
        return _squashed_sample(self._mu_sym(obs), log_std, self.action_scale)

    def action_mean(self, obs: torch.Tensor) -> torch.Tensor:
        return _squashed_mean(self._mu_sym(obs), self.action_scale)


def equivariance_residual(actor: MirrorEquivariantActor, mirror_obs: TensorMap,
                          mirror_act: TensorMap, obs: torch.Tensor) -> float:
    """Max-abs violation of ``a(g·s) == g·a(s)`` for the (post-squash) deterministic policy (~0)."""
    with torch.no_grad():
        lhs = actor.action_mean(mirror_obs(obs))
        rhs = mirror_act(actor.action_mean(obs))
        return float((lhs - rhs).abs().max())
