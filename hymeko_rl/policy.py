"""Shared-backbone Gaussian actor-critic — the actor+critic unification in code.

One ``backbone`` (``obs -> features``) feeds **both** the actor (action mean) and
the critic (state value). The HSiKAN/Gömb policy and the MLP baseline differ *only*
in the backbone, so the architecture ablation is a backbone swap under one shared PPO
(plan: fix the algorithm, ablate the architecture). Continuous control: a diagonal
Gaussian with a state-independent learned ``log_std`` (the standard MuJoCo-control
parameterisation).
"""
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from signed_kan import SignedKANBackbone
# Re-exported so ``hymeko_rl.policy._catmull_rom`` / ``CatmullRomActivation`` keep working (the Triton
# ``cr_kernel`` + the CR parity tests import them); the implementation lives once, in the core.
from signed_kan.splines import CatmullRomActivation, catmull_rom as _catmull_rom  # noqa: F401

if TYPE_CHECKING:
    from hymeko_rl.hypergraph_state import HypergraphState

POLICY_KINDS = ("mlp", "hsikan", "signedkan")


class ActorCritic(nn.Module):
    """Diagonal-Gaussian actor-critic over a shared backbone.

    **Separate actor and critic backbones** (two networks, not a shared trunk): the
    value-loss gradient updates only the critic, the policy gradient only the actor — so
    the value loss cannot corrupt the actor's features (the PPO-degradation suspect). Both
    read the same observation; swapping the backbone *type* (HSiKAN vs MLP) is the ablation.

    # Preconditions
    each backbone maps ``obs`` to ``(B, feat_dim)``; ``feat_dim >= 1``; ``action_dim >= 1``.
    # Postconditions
    ``act`` returns ``(action (B, action_dim), log_prob (B,), value (B,))`` sampled from the
    current policy; ``evaluate`` returns ``(log_prob (B,), entropy (B,), value (B,))``.
    # Invariants
    ``log_std`` is a free parameter (per action dim); ``std = exp(log_std) > 0`` always.
    """

    def __init__(self, actor_backbone: nn.Module, critic_backbone: nn.Module,
                 feat_dim: int, action_dim: int, *, log_std_init: float = -1.6) -> None:
        super().__init__()
        if feat_dim < 1 or action_dim < 1:
            raise ValueError(
                f"feat_dim and action_dim must be >= 1; got {feat_dim}, {action_dim}")
        self.actor_backbone = actor_backbone
        self.critic_backbone = critic_backbone
        self.actor_mean = nn.Linear(feat_dim, action_dim)
        self.critic = nn.Linear(feat_dim, 1)
        self.log_std = nn.Parameter(torch.full((action_dim,), float(log_std_init)))

    def action_mean(self, obs: torch.Tensor) -> torch.Tensor:
        """The deterministic policy mean (for BC / evaluation)."""
        mean: torch.Tensor = self.actor_mean(self.actor_backbone(obs))
        return mean

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        """The critic's state value ``(B,)`` (separate critic backbone)."""
        v: torch.Tensor = self.critic(self.critic_backbone(obs)).squeeze(-1)
        return v

    # ``Any`` for the distribution: torch.distributions is only partially typed, so a
    # precise annotation would force per-call ``# type: ignore`` on sample/log_prob.
    def _dist_value(self, obs: torch.Tensor) -> tuple[Any, torch.Tensor]:
        mean = self.action_mean(obs)
        std = self.log_std.exp().expand_as(mean)
        return torch.distributions.Normal(mean, std), self.value(obs)

    @torch.no_grad()
    def act(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an action; returns ``(action, log_prob, value)`` (no grad — rollout)."""
        dist, value = self._dist_value(obs)
        action = dist.sample()
        return action, dist.log_prob(action).sum(-1), value

    def evaluate(self, obs: torch.Tensor,
                 action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Score a given ``action`` under the current policy (grad-enabled — the PPO
        update). Returns ``(log_prob, entropy, value)``."""
        dist, value = self._dist_value(obs)
        return dist.log_prob(action).sum(-1), dist.entropy().sum(-1), value

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def mlp_backbone(obs_dim: int, *, hidden: int = 64, depth: int = 2,
                 ) -> tuple[nn.Module, int]:
    """A plain ``ReLU`` MLP backbone (the ablation baseline). Returns ``(module, feat_dim)``.

    Leads with ``Flatten(start_dim=1)`` so it consumes the *same* node-feature obs
    ``(B, N, feat)`` the HSiKAN backbone reads (here ``obs_dim == N * feat``) — the
    ablation is then a pure backbone swap on identical observations. A 2-D ``(B, obs_dim)``
    obs flattens to itself, so flat-vector callers are unaffected.

    # Preconditions ``obs_dim >= 1``, ``hidden >= 1``, ``depth >= 1``."""
    if obs_dim < 1 or hidden < 1 or depth < 1:
        raise ValueError(f"obs_dim/hidden/depth must be >= 1; got {obs_dim}/{hidden}/{depth}")
    layers: list[nn.Module] = [nn.Flatten(start_dim=1)]
    d = obs_dim
    for _ in range(depth):
        layers += [nn.Linear(d, hidden), nn.ReLU()]
        d = hidden
    return nn.Sequential(*layers), hidden


class HSiKANBackbone(SignedKANBackbone):
    """HSiKAN (Highway Signed KAN) backbone for the RL line — the kinematic-``hg_state`` adapter over the
    shared :class:`signed_kan.SignedKANBackbone` core. ``hg_state`` (the MJCF-derived signed adjacency) and
    the dense/batched aggregation are the RL specifics; the signed-KAN layer, incidence modes, skip/highway,
    and pooling all live once, in :mod:`signed_kan`. Input is per-vertex features ``(B, N, in_feat)``; output
    is the pooled graph embedding ``(B, hidden)``. Inherits ``a_pos``/``a_neg``/``w_pos_arc``/``_effective_adj``
    /``node_activations``/``n_vertices`` from the core.

    # Preconditions ``in_feat, hidden, n_layers >= 1``; ``hg_state`` exposes ``dense_signed_adj`` +
    ``n_vertices``; ``incidence`` in :data:`signed_kan.INCIDENCE_MODES`; ``skip`` in :data:`signed_kan.SKIP_MODES`.
    """

    def __init__(self, in_feat: int, hidden: int, n_layers: int,
                 hg_state: "HypergraphState", device: torch.device | str = "cpu", *,
                 incidence: str = "fixed", activation: str = "cr", skip: str = "none") -> None:
        a_pos, a_neg = hg_state.dense_signed_adj(device)
        super().__init__(in_feat, hidden, n_layers, a_pos, a_neg,
                         incidence=incidence, activation=activation, skip=skip)


def hsikan_backbone(in_feat: int, *, hg_state: "HypergraphState", hidden: int = 64,
                    n_layers: int = 2, device: torch.device | str = "cpu",
                    incidence: str = "fixed", activation: str = "cr", skip: str = "none",
                    ) -> tuple[nn.Module, int]:
    """The HSiKAN (Highway Signed KAN) backbone over the kinematic hypergraph. Returns ``(module, feat_dim)``.

    ``incidence`` selects how the signed incidence A± carries its arc weights:
    ``"fixed"`` (default — the binary kinematic structure), ``"learned"`` (the ``signedkan`` variant: the
    full A± is trainable), or ``"weighted"`` (free real-valued weights on the *fixed* structural arcs,
    init 1.0 = parity with ``"fixed"``) — the signed-hypergraph premise of real arc weights without
    discarding the kinematic sparsity prior. ``skip`` selects the per-layer skip: ``"none"`` (default,
    parity with a plain signed-conv), ``"residual"``, or ``"highway"`` (the Schmidhuber gate — the H in
    HSiKAN). ``activation`` selects the edge nonlinearity: ``"cr"`` (the Catmull-Rom KAN spline), or
    ``"relu"``/``"tanh"`` for ablation.

    # Preconditions ``hg_state`` exposes ``dense_signed_adj`` + ``n_vertices``; ``incidence`` in
    :data:`signed_kan.INCIDENCE_MODES`; ``skip`` in :data:`signed_kan.SKIP_MODES`.
    """
    return HSiKANBackbone(in_feat, hidden, n_layers, hg_state, device,
                          incidence=incidence, activation=activation, skip=skip), hidden


def signedkan_backbone(in_feat: int, **kw: object) -> tuple[nn.Module, int]:
    """HSiKAN with a fully *learned* signed incidence — the trained (dense) weights are the star edges.
    For real weights on the *fixed* structural arcs instead (keeping the sparsity prior), pass
    ``incidence="weighted"`` to :func:`hsikan_backbone` / ``build_policy("hsikan", …)``."""
    return hsikan_backbone(in_feat, incidence="learned", **kw)  # type: ignore[arg-type]


# Backbone Strategy registry (§6.5 #1/#9: one dispatch, no per-kind wrappers). The
# HSiKAN/Gömb backbone reads the kinematic hypergraph; the MLP baseline reads a flat obs;
# ``signedkan`` is HSiKAN with the incidence itself learned.
_BACKBONES: dict[str, Callable[..., tuple[nn.Module, int]]] = {
    "mlp": mlp_backbone, "hsikan": hsikan_backbone, "signedkan": signedkan_backbone}


def build_policy(kind: str, obs_dim: int, action_dim: int, *, log_std_init: float = -1.6,
                 **backbone_kw: object) -> ActorCritic:
    """Construct an ``ActorCritic`` with the requested backbone (the ablation switch).

    ``mlp`` takes a flat ``obs_dim``; ``hsikan`` takes the per-vertex feature dim plus a
    required ``hg_state=`` (the kinematic hypergraph). The two share the actor+critic heads.
    ``log_std_init`` sets the initial action-noise scale (the exploration tactic; ``std =
    exp(log_std_init)``).

    # Preconditions ``kind in POLICY_KINDS``.
    # Errors ``ValueError`` on an unknown kind; ``TypeError`` if ``hsikan`` is built
    without ``hg_state`` (the structure is mandatory — it is the whole point).
    """
    if kind not in _BACKBONES:
        raise ValueError(f"unknown policy kind {kind!r}; expected one of {POLICY_KINDS}")
    actor_backbone, feat_dim = _BACKBONES[kind](obs_dim, **backbone_kw)
    critic_backbone, _ = _BACKBONES[kind](obs_dim, **backbone_kw)   # independent network
    return ActorCritic(actor_backbone, critic_backbone, feat_dim, action_dim,
                       log_std_init=log_std_init)
