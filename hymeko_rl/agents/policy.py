"""Shared-backbone Gaussian actor-critic — the actor+critic unification in code.

One ``backbone`` (``obs -> features``) feeds **both** the actor (action mean) and
the critic (state value). The HSiKAN/Gömb policy and the MLP baseline differ *only*
in the backbone, so the architecture ablation is a backbone swap under one shared PPO
(plan: fix the algorithm, ablate the architecture). Continuous control: a diagonal
Gaussian with a state-independent learned ``log_std`` (the standard MuJoCo-control
parameterisation).
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from signed_kan import SignedKANBackbone
# Re-exported so ``hymeko_rl.agents.policy._catmull_rom`` / ``CatmullRomActivation`` keep working (the Triton
# ``cr_kernel`` + the CR parity tests import them); the implementation lives once, in the core.
from signed_kan.splines import (  # noqa: F401  (CatmullRomActivation/_catmull_rom re-exported for the CR parity tests)
    CatmullRomActivation,
    catmull_rom as _catmull_rom,
    make_activation,
    set_deploy_mode,
)

if TYPE_CHECKING:
    from hymeko_rl.agents.hypergraph_state import HypergraphState

POLICY_KINDS = ("mlp", "hsikan", "signedkan", "sa_hsikan", "mixture")


def deploy_policy(model: nn.Module, probe: torch.Tensor, *, tol: float = 5e-2) -> float:
    """Switch a trained policy's ``cr_cheby`` cells to the Chebyshev **deploy fast path** (measured ~2.2× forward
    at B=1: 845→379 µs; no gather) for **inference / real-robot control / GIF rendering** — *not* for metric
    evaluation, where the Chebyshev *approximation* of the CR spline would silently shift the measured numbers
    (§3 evaluation-metric integrity).

    The switch is **validated**: the max ``|Chebyshev − CR|`` output error on ``probe`` must stay ``< tol``, else
    the model is reverted to the exact CR path and ``ValueError`` is raised — the caller never unknowingly ships a
    too-loose approximation. Reuses :func:`signed_kan.splines.set_deploy_mode` (§6.1; no re-implementation).

    # Preconditions ``probe`` is a valid forward input for ``model``; ``tol > 0``.
    # Postconditions on success every ``cr_cheby`` cell is in deploy mode and the returned error ``< tol``; on
      failure (or a raise) the cells are left in the exact CR path. Returns ``0.0`` for models with no ``cr_cheby``
      cell (``cr`` / ``mlp`` — nothing to approximate, nothing switched).
    # Errors ``ValueError`` if ``tol <= 0`` or the deploy approximation error exceeds ``tol``."""
    if tol <= 0:
        raise ValueError(f"tol must be > 0; got {tol}")
    set_deploy_mode(model, False)                       # ensure the exact CR reference
    with torch.no_grad():
        ref = model(probe)
    n = set_deploy_mode(model, True)                    # switch cr_cheby cells to the Chebyshev fast path
    if n == 0:
        return 0.0                                      # no cr_cheby cells (cr / mlp) → nothing approximated
    with torch.no_grad():
        err = float((model(probe) - ref).abs().max())
    if err > tol:
        set_deploy_mode(model, False)                   # too loose — revert to the exact CR path
        raise ValueError(f"deploy Chebyshev approx error {err:.3g} > tol {tol:.3g}; kept the CR path")
    return err


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


class PerNodeActionHead(nn.Module):
    """A per-actuator readout over a backbone's per-vertex activations — the non-collapsing actor head.

    Each actuated joint's scalar comes from **its own** message-passed node embedding concatenated with the
    broadcast global pool: ``out[a] = Linear([h[:, v(a)] ; mean_v h])``. This keeps the joint↔node
    correspondence a mean-pool readout destroys; the supervised probe measured collapse-then-expand at 3723×
    worse than this shape, and a per-node head beating a params-matched MLP (``reports/2026-06-26-multidim-readout``).

    # Preconditions ``act_vertices`` are valid vertex indices ``[0, N)``, one per actuator.
    # Postconditions ``forward((B, N, hidden)) -> (B, A)`` finite; ``A == len(act_vertices)``.
    """

    act_vertices: torch.Tensor   # declared so mypy uses Tensor (not nn.Module.__getattr__'s union) for indexing

    def __init__(self, hidden: int, act_vertices: Sequence[int]) -> None:
        super().__init__()
        verts = torch.as_tensor([int(v) for v in act_vertices], dtype=torch.long)
        if verts.ndim != 1 or verts.numel() < 1:
            raise ValueError("act_vertices must be a non-empty 1-D index array")
        self.register_buffer("act_vertices", verts)             # travels with .to(device)
        self.head = nn.Linear(2 * hidden, 1)
        self.action_dim = int(verts.numel())

    def forward(self, node_acts: torch.Tensor) -> torch.Tensor:
        if node_acts.dim() != 3:
            raise ValueError(f"expected per-vertex activations (B, N, H); got {tuple(node_acts.shape)}")
        pool = node_acts.mean(dim=1, keepdim=True)              # (B, 1, H) — global context
        gathered = node_acts[:, self.act_vertices, :]           # (B, A, H) — each actuator's own vertex
        ctx = pool.expand(-1, gathered.shape[1], -1)            # (B, A, H)
        out: torch.Tensor = self.head(torch.cat([gathered, ctx], dim=-1)).squeeze(-1)   # (B, A)
        return out


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


class StructuralActorBackbone(nn.Module):
    """SA-HSiKAN (Structural-Actor HSiKAN) backbone — HSiKAN's iterated signed message-pass COLLAPSED to one
    precomputed holonomy matmul. Over a FIXED topology the signed L-hop transport is a constant operator
    ``Bᴸ = matrix_power(A₊ − A₋, walk_len)`` (the walk sign-products = the Z₂ holonomy), so the forward is one
    ``Bᴸx`` einsum + one HSiKAN cell (Catmull-Rom) readout — a handful of large ops instead of L layers of
    spline-per-edge message-passing (the launch-bound fix; ``reports/2026-06-28-hsikan-launch-bound-bug.md``).
    Same structural prior, a fraction of the dispatch cost. Exposes the HSiKAN backbone interface so both actor
    heads drive it: ``forward -> (B, hidden)`` pooled, ``node_activations -> (B, N, hidden)`` per-vertex.

    # Preconditions ``in_feat, hidden >= 1``; ``hg_state`` exposes ``dense_signed_adj`` + ``n_vertices``.
    # Postconditions ``forward((B, N, in_feat)) -> (B, hidden)``; ``node_activations -> (B, N, hidden)``."""

    bl: torch.Tensor

    def __init__(self, in_feat: int, hidden: int, hg_state: "HypergraphState", walk_len: int = 2,
                 device: torch.device | str = "cpu", *, activation: str = "cr", pool: str = "mean") -> None:
        super().__init__()
        if in_feat < 1 or hidden < 1:
            raise ValueError(f"in_feat/hidden must be >= 1; got {in_feat}/{hidden}")
        a_pos, a_neg = hg_state.dense_signed_adj(device)
        bl = torch.matrix_power(a_pos - a_neg, walk_len)          # Bᴸ: the signed L-hop holonomy operator
        self.register_buffer("bl", bl)
        self.n_vertices = int(bl.shape[0])
        self.pool = pool
        act = make_activation(activation, hidden)        # the HSiKAN cell (cr / cr_cheby / tanh) over the holonomy
        self.node_head = nn.Sequential(nn.Linear(in_feat, hidden), act)

    def node_activations(self, x: torch.Tensor) -> torch.Tensor:
        """Per-vertex activations ``(B, N, hidden)`` — one ``Bᴸx`` gather then the HSiKAN-cell readout."""
        if x.dim() != 3 or x.shape[1] != self.n_vertices:
            raise ValueError(f"expected (B, {self.n_vertices}, in_feat); got {tuple(x.shape)}")
        node_agg = torch.einsum("ne,bef->bnf", self.bl, x)       # (B, N, F) = Bᴸx in ONE precomputed matmul
        h: torch.Tensor = self.node_head(node_agg)
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.node_activations(x)
        pooled: torch.Tensor = h.mean(dim=1) if self.pool == "mean" else h.sum(dim=1)
        return pooled


def sa_hsikan_backbone(in_feat: int, *, hg_state: "HypergraphState", hidden: int = 64, walk_len: int = 2,
                       device: torch.device | str = "cpu", activation: str = "cr_cheby",
                       **_ignored: object) -> tuple[nn.Module, int]:
    """SA-HSiKAN backbone over the fixed kinematic hypergraph — the Bᴸ-collapse of HSiKAN. Defaults to the
    ``cr_cheby`` cell (band-limited Chebyshev control points; the deploy-Chebyshev fast path applies). Returns
    ``(module, feat_dim)``. Ignores HSiKAN-only kwargs (``skip``/``incidence``): the holonomy operator is a
    structural constant, not a learned/gated message-pass."""
    return StructuralActorBackbone(in_feat, hidden, hg_state, walk_len=walk_len, device=device,
                                   activation=activation), hidden


class MixtureBackbone(nn.Module):
    """HSiKAN + MLP **mixture-of-experts**: a per-state learned gate ``g∈(0,1)`` blends the structural HSiKAN
    expert with the structure-blind MLP expert — ``g·HSiKAN(x) + (1−g)·MLP(x)``. The gate (a linear read of the
    flat obs → sigmoid) lets one policy lean on structure where it helps and on the dense MLP where it does not
    — the deliberative/reflexive hybrid in a single network. Both experts read the same ``(B, N, in_feat)`` obs;
    output is the pooled ``(B, hidden)`` embedding.

    # Preconditions ``in_feat, hidden >= 1``; ``hg_state`` exposes ``dense_signed_adj`` + ``n_vertices``.
    # Postconditions ``forward((B, N, in_feat)) -> (B, hidden)``; ``gate_value`` is in ``(0, 1)``."""

    def __init__(self, in_feat: int, hidden: int, hg_state: "HypergraphState",
                 *, n_layers: int = 2, device: torch.device | str = "cpu") -> None:
        super().__init__()
        if in_feat < 1 or hidden < 1:
            raise ValueError(f"in_feat/hidden must be >= 1; got {in_feat}/{hidden}")
        self.n_vertices, self.in_feat = hg_state.n_vertices, in_feat
        self.hsikan, _ = hsikan_backbone(in_feat, hg_state=hg_state, hidden=hidden, n_layers=n_layers,
                                         device=device)
        self.mlp, _ = mlp_backbone(self.n_vertices * in_feat, hidden=hidden)
        self.gate = nn.Linear(self.n_vertices * in_feat, 1)        # per-state HSiKAN-vs-MLP mixing weight

    def _as_3d(self, x: torch.Tensor) -> torch.Tensor:
        return x if x.dim() == 3 else x.view(x.shape[0], self.n_vertices, self.in_feat)

    def gate_value(self, x: torch.Tensor) -> torch.Tensor:
        """The per-sample mixing weight ``g∈(0,1)`` (1 = all HSiKAN, 0 = all MLP). # Postconditions ``(B, 1)``."""
        x3 = self._as_3d(x)
        return torch.sigmoid(self.gate(x3.reshape(x3.shape[0], -1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x3 = self._as_3d(x)
        g = self.gate_value(x3)
        return g * self.hsikan(x3) + (1.0 - g) * self.mlp(x3)


def mixture_backbone(in_feat: int, *, hg_state: "HypergraphState", hidden: int = 64, n_layers: int = 2,
                     device: torch.device | str = "cpu", **_ignored: object) -> tuple[nn.Module, int]:
    """HSiKAN + MLP gated mixture-of-experts over the kinematic hypergraph (same call signature as
    :func:`hsikan_backbone`). Returns ``(module, feat_dim)``."""
    return MixtureBackbone(in_feat, hidden, hg_state, n_layers=n_layers, device=device), hidden


# Backbone Strategy registry (§6.5 #1/#9: one dispatch, no per-kind wrappers). The
# HSiKAN/Gömb backbone reads the kinematic hypergraph; the MLP baseline reads a flat obs;
# ``signedkan`` is HSiKAN with the incidence itself learned; ``mixture`` gates HSiKAN against MLP.
_BACKBONES: dict[str, Callable[..., tuple[nn.Module, int]]] = {
    "mlp": mlp_backbone, "hsikan": hsikan_backbone, "signedkan": signedkan_backbone,
    "sa_hsikan": sa_hsikan_backbone, "mixture": mixture_backbone}


def build_policy(kind: str, obs_dim: int, action_dim: int, *, log_std_init: float = -1.6,
                 **backbone_kw: object) -> ActorCritic:
    """Construct an ``ActorCritic`` with the requested backbone (the ablation switch).

    ``mlp`` takes a flat ``obs_dim``; ``hsikan`` takes the per-vertex feature dim plus a
    required ``hg_state=`` (the kinematic hypergraph). The two share the actor+critic heads.
    ``log_std_init`` sets the initial action-noise scale (the exploration tactic; ``std =
    exp(log_std_init)``).

    GPU speedup: these models are tiny and launch-bound on GPU, so ``torch.compile(mode="reduce-overhead")``
    is a large win (~10x measured on a 3070). Apply it at the **call site**, not here — e.g.
    ``fast = torch.compile(ac.action_mean, mode="reduce-overhead")`` — so the saved model stays eager and
    checkpoints round-trip. Compiling the backbone *in place* prefixes state-dict keys with ``_orig_mod.`` and
    breaks checkpoint loading (verified), so ``build_policy`` deliberately does not bake compile into the model.

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
