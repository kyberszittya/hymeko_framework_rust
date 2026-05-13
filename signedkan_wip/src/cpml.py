"""Concentric-Pyramid Multi-Layer (CPML) signed-graph link-prediction
architecture (2026-05-11).

Plan:
    docs/plans/2026-05-11-cpml-xhc-architectures/plan.{tex,pdf,tikz,mmd}

Handbook (route maths + Highway / Capsule / KAN unification):
    docs/book/src/research/cpml-routing-highway-capsule-kan.md

This module is the **feasibility scaffold** — tier stratification
(degree percentiles) plus two readouts:

  * ``topology="route"`` (default): each aggregator reads **corners from
    the same base** ``X_0``; final ``concat(X_0, H_0, …, H_{L-1})`` feeds the
    edge head — **no widening concat between tiers**.

  * ``tier_organization`` (default ``structural``): tier ``ℓ`` uses cycles
    that **touch** a vertex with ``tier_of[v]=ℓ`` (hard incidence). With
    ``capsule_soft`` (**``topology="route"`` only**), learned **softmax**
    over tiers scales every cycle for every tier; routing logits come from
    ``capsule_soft_router`` (MLP pool, **hypergraph convolution** on cycles as
    hyperedges, or EM-style agreement — see ``CPMLConfig``).

  * ``topology="pyramid"``: legacy **inward pyramid** — tier ``ℓ`` consumes
    widening ``concat(X_0, H_0, …, H_{ℓ-1})`` on the corner path.

Design reading (``structural`` + ``route``): **Highway-like** carry of ``X_0``,
**Capsule-like** structural routing, **KAN-like** corners when
``aggregator_kind="hsikan"``.

Aggregators: MLP stub, HSiKAN (spline), or Clifford-FIR on cycle corners
(tier filtering per ``tier_organization``).

Unit tests in `signedkan_wip/tests/test_cpml.py` validate:

  * stratification correctness (tier sizes match percentile cuts)
  * forward shapes at L ∈ {2, 3, 4} for topology × tier_organization combos
  * backward gradient flow (no NaN)
  * L=1 reduces to a flat aggregator (sanity)
  * under ``pyramid``, innermost-tier routing still reflects outer-tier
    contributions (widening-concat funnel)

Once the feasibility gate is green, the simplified per-layer block
will be swapped for the full `MixedAritySignedKAN` and the model will
be wired into `run_final_cell.py` via `--model CPML`.

The degree-tier metaphor mirrors V1 → V2 → V4 → IT (Felleman & Van Essen
1991): periphery vs hubs — **routing** decides which cycles feed which
aggregator; **pyramid** additionally stacks widening representations
tier-over-tier.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Tier stratification ────────────────────────────────────────────


@dataclass
class TierSpec:
    """Tier boundaries on degree-percentile space.

    Example for L=3 (periphery / mid / centre):
        TierSpec(cuts=(0.0, 0.2, 0.8, 1.0))
    → T_1 covers percentiles (0.0, 0.2] (bottom 20%, leaves)
    → T_2 covers percentiles (0.2, 0.8] (middle 60%)
    → T_3 covers percentiles (0.8, 1.0] (top 20%, hubs)
    """

    cuts: tuple[float, ...] = (0.0, 0.2, 0.8, 1.0)

    @property
    def L(self) -> int:
        return len(self.cuts) - 1

    def assign(self, degrees: np.ndarray) -> np.ndarray:
        """Map each vertex to its tier index in {0, ..., L-1}.

        Degree-percentile is computed as rank/N (ties broken by stable
        sort).  Tier 0 = outermost (lowest degree); tier L-1 = innermost
        (highest degree).
        """
        n = degrees.shape[0]
        if n == 0:
            return np.array([], dtype=np.int64)
        # Rank: rank[i] = position of degrees[i] when sorted ascending.
        order = np.argsort(degrees, kind="stable")
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = np.arange(n) / max(1, n - 1) if n > 1 else 0.0
        # Bin by cuts.  cuts[0]=0.0, cuts[L]=1.0; closed-right intervals.
        tiers = np.zeros(n, dtype=np.int64)
        for ell in range(self.L):
            lo = self.cuts[ell]
            hi = self.cuts[ell + 1]
            if ell == 0:
                # Closed-left, closed-right for the first interval to
                # capture rank=0.
                mask = (ranks >= lo) & (ranks <= hi)
            else:
                mask = (ranks > lo) & (ranks <= hi)
            tiers[mask] = ell
        return tiers


# ─── Tier-restricted cycle pool ─────────────────────────────────────


def restrict_cycles_to_tier(
    cycles: np.ndarray, tier_of: np.ndarray, ell: int,
) -> np.ndarray:
    """Return the subset of `cycles` (shape (M, k)) that touches at
    least one vertex of tier `ell`.

    `tier_of[v]` is the tier index of vertex v in {0, ..., L-1}.
    """
    if cycles.size == 0:
        return cycles
    # cycles is (M, k); for each row, check if any vertex's tier == ell.
    cycle_tiers = tier_of[cycles]                      # (M, k)
    mask = (cycle_tiers == ell).any(axis=1)             # (M,)
    return cycles[mask]


# ─── Per-tier aggregator (feasibility-stub: small KAN-style MLP) ────


class ClifFIRTierAggregator(nn.Module):
    """Clifford-FIR per-tier aggregator.

    PyTorch counterpart of the Rust `hymeko_graph::spine::CliffordFIR`.
    Operates as the Cl(0,1) ≅ ℂ unified-filter signed-cycle FIR:

        k_i = a_i + i b_i ∈ Cl(0,1)
        proj_σ(k_i) = a_i if σ=+1 else b_i
        out_c[j] = Σ_i proj_{σ_i}(k_i) · X[v_i][j]

    Has only `2 k + bias` learnable parameters (one (a_i, b_i)
    Clifford coefficient per position + per-channel bias) — far
    fewer than the MLP stub's `(d_in × d_hidden + d_hidden × d_out)`.
    The whole point: a tiny, signed-structure-aware filter that
    isolates the contribution of the cycle's sign pattern.

    Forward signature:
        cycle_vertex_features: (M, k, d_in)
        cycle_signs:           (M, k) int8 / float
        return:                (M, d_in)        — self-map width

    The trailing optional `signs` arg lets CPML's dispatch route
    signs through without changing the per-tier aggregator API.
    """

    def __init__(self, d_in: int, k_arity: int):
        super().__init__()
        self.k_arity = k_arity
        self.d_in = d_in
        # Per-position scalar / pseudoscalar Clifford coefficients.
        # Initialise at 1/k (mean-pool baseline).
        init = torch.full((k_arity,), 1.0 / k_arity)
        self.coef_a = nn.Parameter(init.clone())
        self.coef_b = nn.Parameter(-init.clone())  # signed-mean init
        self.bias = nn.Parameter(torch.zeros(d_in))

    def forward(self, cycle_vertex_features: torch.Tensor,
                 cycle_signs: torch.Tensor | None = None) -> torch.Tensor:
        # cycle_vertex_features: (M, k, d_in)
        # cycle_signs: (M, k)  — if None, treat all as σ=+1 (degrades
        #   to a single scalar filter; useful for ablation).
        if cycle_signs is None:
            cycle_signs = torch.ones(
                cycle_vertex_features.shape[:2],
                device=cycle_vertex_features.device,
                dtype=cycle_vertex_features.dtype,
            )
        # (M, k) sign-aware coefficient.
        pos_mask = (cycle_signs > 0).to(cycle_vertex_features.dtype)
        coef = pos_mask * self.coef_a + (1.0 - pos_mask) * self.coef_b
        # (M, k, 1) · (M, k, d_in) → (M, d_in)
        return (coef.unsqueeze(-1) * cycle_vertex_features).sum(dim=1) + self.bias


class TierAggregator(nn.Module):
    """Simplified per-tier aggregator (feasibility stub).

    Operates on a tier-restricted cycle pool:
        agg(M_ell, X) = pool_over_cycles(MLP(corner_features))

    Used in unit tests + the smoke runner; swap to
    `SignedKANTierAggregator` for the real run (CPML × HSiKAN).

    Forward signature (note: NO signs — the MLP doesn't use signed
    incidence; the SignedKAN variant does):
        cycle_vertex_features : (M, k, d_in)
        return                 : (M, d_out)
    """

    def __init__(self, d_in: int, d_hidden: int, d_out: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, cycle_vertex_features: torch.Tensor) -> torch.Tensor:
        per_vertex = self.proj(cycle_vertex_features)
        return per_vertex.mean(dim=1)


class SignedKANTierAggregator(nn.Module):
    """**Real HSiKAN-flavoured** per-tier aggregator.

    Replaces `TierAggregator`'s MLP stub with a `SignedKANLayer` (the
    Catmull-Rom / B-spline signed-cycle aggregation kernel used in
    the Slashdot SOTA recipe).

    Forward signature differs from the MLP stub: it requires the
    full **per-vertex features** + **cycle vertex indices** + **cycle
    edge signs**, because the spline aggregation needs to gather
    features by index and σ-mask per branch.

    Forward signature:
        x        : (N, d_in)        per-vertex features
        cycles   : (M, k)           cycle vertex indices
        signs    : (M, k)           ±1 boundary-edge signs
        return   : (M, d_in)        per-cycle embeddings

    Output dim equals input dim because SignedKANLayer is a self-map.
    CPML adapts via projection if d_layer ≠ d_in (handled in the
    CPML wrapper, not here).
    """

    def __init__(self, n_nodes: int, hidden_dim: int = 32,
                 grid: int = 5, k_arity: int = 3,
                 spline_kind: str = "catmull_rom"):
        super().__init__()
        from .signedkan import SignedKANConfig, SignedKANLayer
        self.k_arity = k_arity
        cfg = SignedKANConfig(
            n_nodes=n_nodes,
            hidden_dim=hidden_dim,
            grid=grid,
            k=k_arity,
            use_minus_branch=True,
            use_zero_branch=False,
            spline_kind=spline_kind,
            init_scale=0.1,
            inner_skip="none",
            outer_skip="none",
        )
        self.layer = SignedKANLayer(cfg)

    def forward(
        self, x: torch.Tensor,                 # (N, d_in)
        cycles: torch.Tensor,                  # (M, k) long
        signs: torch.Tensor,                   # (M, k) int8
    ) -> torch.Tensor:
        # SignedKANLayer expects int64 cycles and float sigma.
        triad_v = cycles.to(torch.long)
        triad_sigma = signs.to(x.dtype)
        return self.layer(x, triad_v, triad_sigma)  # (M, d_hidden)


# ─── Capsule hypergraph routing (torch.compile-friendly submodule) ──


class CapsuleHypergraphRouter(nn.Module):
    """One signed HGNN-style step on the cycle hypergraph → routing logits ``(M, L)``.

    Vertex degrees ``d_v`` / ``dv_inv_sqrt`` are supplied by ``CPML`` (so
    ``capsule_hg_cache_degrees`` stays on the parent) — this module only runs
    the differentiable gather / MLP / scatter / readout path, which is safe
    to wrap with ``torch.compile``.
    """

    def __init__(self, d_in: int, d_hidden: int, n_tiers: int) -> None:
        super().__init__()
        self.vertex_proj = nn.Linear(d_in, d_hidden)
        self.edge_mlp = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
        )
        self.route_head = nn.Linear(2 * d_hidden, n_tiers)

    def forward(
        self,
        x0: torch.Tensor,
        cycles: torch.Tensor,
        sigma: torch.Tensor,
        d_v: torch.Tensor,
        dv_inv_sqrt: torch.Tensor,
    ) -> torch.Tensor:
        M, k = cycles.shape
        device, dtype = x0.device, x0.dtype
        n_vertices = int(x0.shape[0])
        flat_idx = cycles.reshape(-1)

        x_p = self.vertex_proj(x0)
        dv_c = dv_inv_sqrt[cycles].unsqueeze(-1)
        corner = x_p[cycles] * dv_c
        sig_exp = sigma.unsqueeze(-1)
        corner_signed = corner * sig_exp
        h_e = corner_signed.mean(dim=1)
        h_e = self.edge_mlp(h_e)

        d_h = x_p.shape[-1]
        out = torch.zeros(n_vertices, d_h, device=device, dtype=dtype)
        h_flat = (h_e.unsqueeze(1) * sig_exp).reshape(M * k, d_h)
        out.index_add_(0, flat_idx, h_flat)
        out = out / d_v.unsqueeze(-1)
        out = dv_inv_sqrt.unsqueeze(-1) * out

        v_mix = (out[cycles] * sig_exp).mean(dim=1)
        z = torch.cat([h_e, v_mix], dim=-1)
        return self.route_head(z)


# ─── CPML model ──────────────────────────────────────────────────────


@dataclass
class CPMLConfig:
    """CPML configuration.

    Default L=3 follows the periphery/mid/centre split tested in
    Felleman--Van Essen-inspired hierarchies.

    The `aggregator_kind` knob makes HSiKAN-vs-MLP an **orthogonal
    dimension** to the CPML-vs-Flat topology dimension:
        - "mlp"   : 2-layer GELU MLP (the feasibility stub)
        - "hsikan": SignedKANLayer (the Slashdot SOTA aggregator)
    Combined with `tier_spec` having L=1 (flat) or L>=3 (multi-tier),
    this yields a clean 2x2 factorial design:
        Flat-MLP / Flat-HSiKAN / CPML-MLP / CPML-HSiKAN

    ``topology``:
        - ``route`` (default): each tier index routes cycles to its own
          aggregator reading **base** ``d_in`` features; outputs are
          concatenated once (parallel routes, no widening concat between tiers).
        - ``pyramid``: legacy inward pyramid — each tier consumes the
          widening concat of all prior route outputs.

    ``tier_organization`` (orthogonal to ``topology`` for ``route``):
        - ``structural`` (default): cycle ``c`` is assigned to tier ``ℓ`` iff
          some vertex on ``c`` has ``tier_of[v]=ℓ`` (hard incidence routing).
        - ``capsule_soft``: **route topology only** — every cycle runs through
          every tier's aggregator; a learned **softmax router** on pooled
          corner features yields weights ``R_{c,ℓ}`` (CapsNet-style *routing
          coefficients*), and tier ``ℓ`` receives ``R_{c,ℓ} · Agg_ℓ(c)`` before
          scatter-mean. ``topology="pyramid"`` is rejected for this mode.

    ``capsule_soft_router`` (only ``tier_organization="capsule_soft"``):
        How cycle→tier logits are produced before ``softmax``:

        - ``"auto"`` (default): ``capsule_routing_iterations==1`` → MLP pool
          on corners (**``mlp_softmax``**); ``>1`` → **``em_agreement``**.
        - ``"mlp_softmax"``: one MLP on mean-pooled corner features (requires
          ``capsule_routing_iterations==1``).
        - ``"hypergraph_conv"``: **single-pass** HGNN-style message passing on
          the **signed** cycle hypergraph (``cycle_signs`` scale node→edge and
          edge→node incidence), then readout to ``L`` logits per cycle
          (requires ``capsule_routing_iterations==1``). No iterative EM loop.
        - ``"em_agreement"``: routing-by-agreement iterations (requires
          ``capsule_routing_iterations>=2``).

    ``capsule_routing_iterations`` (only ``tier_organization="capsule_soft"``):
        Must be ``>= 1``. Meaning depends on ``capsule_soft_router`` (see above).

    ``capsule_hg_hidden``: hidden width for ``hypergraph_conv`` routing only.

    ``capsule_hg_cache_degrees``: when True (default), ``hypergraph_conv`` caches
    per-vertex degree / ``D_v^{-1/2}`` tensors keyed by ``cycles`` storage
    identity so repeated forwards with the **same** ``cycles`` object skip the
    ``index_add`` degree pass (typical in static-graph training). Set False to
    always recompute (debug / if ``cycles`` is mutated in place without
    changing storage).

    ``torch_compile_hypergraph``: when True, wraps ``CapsuleHypergraphRouter``
    with ``torch.compile`` (requires PyTorch 2.x). First calls may compile;
    use fixed ``N``, ``M`` for predictable behaviour in benchmarks.
    """

    tier_spec: TierSpec = field(default_factory=TierSpec)
    d_in: int = 16                       # initial vertex feature dim
    d_layer: int = 16                    # per-layer hidden / output dim
    d_predictor_hidden: int = 32         # MLP head for edge scoring
    pool_to_vertex: str = "mean"         # "mean" or "sum"
    aggregator_kind: str = "mlp"         # "mlp" | "hsikan" | "clifford_fir"
    hsikan_grid: int = 5
    hsikan_spline_kind: str = "catmull_rom"
    n_nodes: int | None = None           # required for "hsikan"
    cycle_k: int = 3                     # K = cycle arity, used by "clifford_fir"
    topology: Literal["route", "pyramid"] = "route"
    tier_organization: Literal["structural", "capsule_soft"] = "structural"
    #: Hidden width for ``capsule_soft_router="mlp_softmax"`` MLP router.
    capsule_route_hidden: int = 64
    #: Capsule routing steps / EM depth; interpreted with ``capsule_soft_router``.
    capsule_routing_iterations: int = 1
    capsule_soft_router: Literal[
        "auto",
        "mlp_softmax",
        "hypergraph_conv",
        "em_agreement",
    ] = "auto"
    #: Hidden dim for ``capsule_soft_router="hypergraph_conv"`` (vertex/edge).
    capsule_hg_hidden: int = 64
    #: Cache ``D_v`` / ``D_v^{-1/2}`` for HG routing when ``cycles`` is reused.
    capsule_hg_cache_degrees: bool = True
    #: ``torch.compile`` the hypergraph routing submodule (PyTorch 2.x).
    torch_compile_hypergraph: bool = False


class CPML(nn.Module):
    """CPML signed-graph link predictor.

    ``topology="route"`` (default): each degree tier **routes** its
    cycle-restricted pool through a dedicated aggregator that always reads
    the **base** vertex features ``X_0``; route outputs are concatenated
    once with ``X_0`` for the edge head (width ``d_in + L·d_layer``).

    ``topology="pyramid"``: legacy **inward pyramid** — tier ℓ consumes
    the widening concat ``concat(X_0, H_1, …, H_ℓ)``.

    ``tier_organization`` (``route`` only for ``capsule_soft``):
        ``structural`` — hard masks from ``tier_of`` (default).
        ``capsule_soft`` — learned softmax routing over tiers; logits from
        ``capsule_soft_router`` (MLP on corners, **hypergraph conv** on cycles
        as hyperedges, or EM agreement — see ``CPMLConfig``).

    Forward signature:
        node_features : (N, d_in)
        cycles        : (M, k) int64
        cycle_signs   : (M, k) int8
        tier_of       : (N,)   int64
        edges_to_score: (E, 2) int64

    Returns:
        scores : (E,) float
    """

    def __init__(self, cfg: CPMLConfig) -> None:
        super().__init__()
        topo = cfg.topology
        if topo not in ("route", "pyramid"):
            raise ValueError(
                f"topology must be 'route' or 'pyramid', got {topo!r}",
            )
        self.cfg = cfg
        L = cfg.tier_spec.L
        self.L = L
        org = cfg.tier_organization
        if org not in ("structural", "capsule_soft"):
            raise ValueError(
                f"tier_organization must be 'structural' or 'capsule_soft', "
                f"got {org!r}",
            )
        if org == "capsule_soft" and topo != "route":
            raise ValueError(
                "tier_organization='capsule_soft' requires topology='route' "
                f"(pyramid widening is incompatible); got topology={topo!r}",
            )
        in_dims: list[int] = [cfg.d_in]
        for _ in range(L):
            in_dims.append(in_dims[-1] + cfg.d_layer)
        self.in_dims = in_dims
        self.aggregator_kind = cfg.aggregator_kind
        self.aggregators = nn.ModuleList()
        self.pre_projs = nn.ModuleList()
        self.post_projs = nn.ModuleList()
        for ell in range(L):
            agg_d_in = cfg.d_in if topo == "route" else in_dims[ell]
            if cfg.aggregator_kind == "mlp":
                self.aggregators.append(TierAggregator(
                    d_in=agg_d_in,
                    d_hidden=cfg.d_layer * 2,
                    d_out=cfg.d_layer,
                ))
                self.pre_projs.append(nn.Identity())
                self.post_projs.append(nn.Identity())
            elif cfg.aggregator_kind == "hsikan":
                if cfg.n_nodes is None:
                    raise ValueError(
                        "CPMLConfig.n_nodes must be set for aggregator_kind='hsikan'",
                    )
                self.pre_projs.append(nn.Linear(agg_d_in, cfg.d_layer))
                self.aggregators.append(SignedKANTierAggregator(
                    n_nodes=cfg.n_nodes,
                    hidden_dim=cfg.d_layer,
                    grid=cfg.hsikan_grid,
                    spline_kind=cfg.hsikan_spline_kind,
                ))
                self.post_projs.append(nn.Identity())
            elif cfg.aggregator_kind == "clifford_fir":
                self.pre_projs.append(nn.Linear(agg_d_in, cfg.d_layer))
                self.aggregators.append(ClifFIRTierAggregator(
                    d_in=cfg.d_layer, k_arity=cfg.cycle_k,
                ))
                self.post_projs.append(nn.Identity())
            else:
                raise ValueError(
                    f"unknown aggregator_kind {cfg.aggregator_kind!r}; "
                    "valid: 'mlp', 'hsikan', 'clifford_fir'",
                )
        final_dim = in_dims[L]
        self.head = nn.Sequential(
            nn.Linear(2 * final_dim, cfg.d_predictor_hidden),
            nn.GELU(),
            nn.Linear(cfg.d_predictor_hidden, 1),
        )
        if org == "capsule_soft":
            nit = int(cfg.capsule_routing_iterations)
            if nit < 1:
                raise ValueError(
                    f"capsule_routing_iterations must be >= 1, got {nit}",
                )
            router_kind = cfg.capsule_soft_router
            if router_kind == "auto":
                resolved = "mlp_softmax" if nit == 1 else "em_agreement"
            elif router_kind == "mlp_softmax":
                if nit != 1:
                    raise ValueError(
                        "capsule_soft_router='mlp_softmax' requires "
                        f"capsule_routing_iterations==1; got {nit}. "
                        "Use 'auto' or 'em_agreement' for multi-step routing.",
                    )
                resolved = "mlp_softmax"
            elif router_kind == "em_agreement":
                if nit < 2:
                    raise ValueError(
                        "capsule_soft_router='em_agreement' requires "
                        f"capsule_routing_iterations>=2; got {nit}",
                    )
                resolved = "em_agreement"
            elif router_kind == "hypergraph_conv":
                if nit != 1:
                    raise ValueError(
                        "capsule_soft_router='hypergraph_conv' requires "
                        f"capsule_routing_iterations==1; got {nit}",
                    )
                resolved = "hypergraph_conv"
            else:
                raise ValueError(
                    f"unknown capsule_soft_router {router_kind!r}",
                )
            self._capsule_soft_router_resolved = resolved
            d_hg = max(1, int(cfg.capsule_hg_hidden))
            if resolved == "mlp_softmax":
                h = max(1, int(cfg.capsule_route_hidden))
                self.capsule_router = nn.Sequential(
                    nn.Linear(cfg.d_in, h),
                    nn.GELU(),
                    nn.Linear(h, L),
                )
                self.capsule_init_logits = None
                self.capsule_hg_block = None
            elif resolved == "em_agreement":
                self.capsule_router = None
                self.capsule_init_logits = nn.Linear(cfg.d_in, L)
                self.capsule_hg_block = None
            else:
                assert resolved == "hypergraph_conv"
                self.capsule_router = None
                self.capsule_init_logits = None
                inner = CapsuleHypergraphRouter(cfg.d_in, d_hg, L)
                if cfg.torch_compile_hypergraph:
                    if not hasattr(torch, "compile"):
                        raise ValueError(
                            "torch_compile_hypergraph=True requires PyTorch 2.x "
                            "with torch.compile",
                        )
                    self.capsule_hg_block = torch.compile(
                        inner,
                        mode="reduce-overhead",
                        fullgraph=False,
                        dynamic=True,
                    )
                else:
                    self.capsule_hg_block = inner
        else:
            self.capsule_router = None
            self.capsule_init_logits = None
            self.capsule_hg_block = None
            self._capsule_soft_router_resolved = ""

        # Runtime cache for hypergraph_conv vertex degrees (not in state_dict).
        self._capsule_hg_deg_cache_key: object | None = None
        self._capsule_hg_d_v_cached: torch.Tensor | None = None
        self._capsule_hg_dv_inv_sqrt_cached: torch.Tensor | None = None

    def _unwrap_capsule_hg_router(self) -> CapsuleHypergraphRouter | None:
        block = getattr(self, "capsule_hg_block", None)
        if block is None:
            return None
        inner = getattr(block, "_orig_mod", block)
        return cast(CapsuleHypergraphRouter, inner)

    @property
    def capsule_hg_vertex_proj(self) -> nn.Linear | None:
        inner = self._unwrap_capsule_hg_router()
        return inner.vertex_proj if inner is not None else None

    @property
    def capsule_hg_edge_mlp(self) -> nn.Sequential | None:
        inner = self._unwrap_capsule_hg_router()
        return inner.edge_mlp if inner is not None else None

    @property
    def capsule_hg_route_head(self) -> nn.Linear | None:
        inner = self._unwrap_capsule_hg_router()
        return inner.route_head if inner is not None else None

    @staticmethod
    def _scatter_mean(
        per_cycle_features: torch.Tensor,    # (M, d)
        cycle_membership: torch.Tensor,      # (M, k) int64
        n_vertices: int,
    ) -> torch.Tensor:
        """Scatter-mean per-cycle features back to vertices.

        Each cycle's vertices receive an equal share of the cycle's
        feature.  Returns (N, d); vertices not in any cycle get zeros.
        """
        M, k = cycle_membership.shape
        d = per_cycle_features.shape[-1]
        # Broadcast per-cycle feature across its k member vertices.
        broadcast = per_cycle_features.unsqueeze(1).expand(M, k, d)
        # Flatten (M, k) to (M*k,) for index_add.
        vidx = cycle_membership.reshape(-1)
        vals = broadcast.reshape(-1, d)
        out = torch.zeros(n_vertices, d, device=per_cycle_features.device,
                          dtype=per_cycle_features.dtype)
        out.index_add_(0, vidx, vals)
        # Mean: divide by per-vertex contribution count.
        counts = torch.zeros(n_vertices, device=per_cycle_features.device,
                             dtype=per_cycle_features.dtype)
        ones = torch.ones_like(vidx, dtype=per_cycle_features.dtype)
        counts.index_add_(0, vidx, ones)
        counts = counts.clamp_min(1.0).unsqueeze(-1)
        return out / counts

    def _edge_logits(self, x_final: torch.Tensor,
                     edges_to_score: torch.Tensor) -> torch.Tensor:
        u = x_final[edges_to_score[:, 0]]
        v = x_final[edges_to_score[:, 1]]
        pair = torch.cat([u, v], dim=-1)
        return self.head(pair).squeeze(-1)

    def _tier_cycle_subset(
        self,
        cycles: torch.Tensor,
        cycle_signs: torch.Tensor,
        tier_of: torch.Tensor,
        ell: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cycle_tiers = tier_of[cycles]
        tier_mask = (cycle_tiers == ell).any(dim=1)
        return cycles[tier_mask], cycle_signs[tier_mask]

    def _capsule_aggregator_per_cycle(
        self,
        ell: int,
        x0: torch.Tensor,
        cv_feats: torch.Tensor,
        cycles: torch.Tensor,
        cycle_signs: torch.Tensor,
    ) -> torch.Tensor:
        """Per-cycle tier-``ell`` outputs (M, d_layer) for capsule routing."""
        if self.aggregator_kind == "mlp":
            return self.aggregators[ell](cv_feats)
        if self.aggregator_kind == "hsikan":
            x_proj = self.pre_projs[ell](x0)
            return self.aggregators[ell](x_proj, cycles, cycle_signs)
        x_proj = self.pre_projs[ell](x0)
        cv_p = x_proj[cycles]
        return self.aggregators[ell](cv_p, cycle_signs.float())

    def _capsule_hypergraph_vertex_degrees(
        self,
        cycles: torch.Tensor,
        n_vertices: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Unsigned corner counts ``D_v`` and ``D_v^{-1/2}`` (optionally cached)."""
        M, k = cycles.shape
        flat_idx = cycles.reshape(-1)
        if self.cfg.capsule_hg_cache_degrees:
            key = (cycles.data_ptr(), cycles.shape, device, dtype)
            if self._capsule_hg_deg_cache_key == key and self._capsule_hg_d_v_cached is not None:
                assert self._capsule_hg_dv_inv_sqrt_cached is not None
                return self._capsule_hg_d_v_cached, self._capsule_hg_dv_inv_sqrt_cached
        d_v = torch.zeros(n_vertices, device=device, dtype=dtype)
        d_v.index_add_(
            0, flat_idx, torch.ones(M * k, device=device, dtype=dtype),
        )
        d_v = d_v.clamp(min=1.0)
        dv_inv_sqrt = d_v.pow(-0.5)
        if self.cfg.capsule_hg_cache_degrees:
            self._capsule_hg_deg_cache_key = (
                cycles.data_ptr(), cycles.shape, device, dtype,
            )
            self._capsule_hg_d_v_cached = d_v
            self._capsule_hg_dv_inv_sqrt_cached = dv_inv_sqrt
        return d_v, dv_inv_sqrt

    def _capsule_hypergraph_routing_logits(
        self,
        x0: torch.Tensor,
        cycles: torch.Tensor,
        cycle_signs: torch.Tensor,
    ) -> torch.Tensor:
        """HGNN-style one step on the **signed** cycle hypergraph → logits (M, L).

        Vertices are graph nodes; each row of ``cycles`` is one hyperedge.
        Corner signs ``σ_{c,i} ∈ {±1}`` scale **both** the node→edge pool and
        the edge→node scatter (signed incidence on the star expansion), while
        vertex degrees ``D_v`` use **unsigned** corner counts for stable
        normalisation.

        Flow: gather ``x_p`` at corners, multiply by ``D_v^{-1/2}`` **there**
        (avoids materialising a full ``N × d`` left-normalised tensor), apply
        ``σ`` for the signed pool to ``h_e`` → edge MLP → signed scatter back →
        ``/ D_v`` → ``D_v^{-1/2}`` → readout ``concat(h_e, signed-mean of
        vertex states on corners)``.

        When ``torch_compile_hypergraph`` is True, ``capsule_hg_block`` may be
        a compiled wrapper; degrees are still computed on ``CPML`` (cacheable).
        """
        block = self.capsule_hg_block
        assert block is not None

        N = int(x0.shape[0])
        M, k = cycles.shape
        device, dtype = x0.device, x0.dtype
        sigma = cycle_signs.to(dtype=dtype)
        if sigma.shape != (M, k):
            raise ValueError(
                f"cycle_signs must be (M, k)=({M}, {k}), got {tuple(sigma.shape)}",
            )

        d_v, dv_inv_sqrt = self._capsule_hypergraph_vertex_degrees(
            cycles, N, device=device, dtype=dtype,
        )
        return block(x0, cycles, sigma, d_v, dv_inv_sqrt)

    def _forward_route_structural(
        self,
        node_features: torch.Tensor,
        cycles: torch.Tensor,
        cycle_signs: torch.Tensor,
        tier_of: torch.Tensor,
        edges_to_score: torch.Tensor,
    ) -> torch.Tensor:
        N = node_features.shape[0]
        x0 = node_features
        h_parts: list[torch.Tensor] = []
        for ell in range(self.L):
            cycles_ell, signs_ell = self._tier_cycle_subset(
                cycles, cycle_signs, tier_of, ell,
            )
            if cycles_ell.shape[0] == 0:
                h_ell = torch.zeros(
                    N, self.cfg.d_layer, device=x0.device, dtype=x0.dtype,
                )
            else:
                if self.aggregator_kind == "mlp":
                    cv_feats = x0[cycles_ell]
                    per_cycle = self.aggregators[ell](cv_feats)
                elif self.aggregator_kind == "hsikan":
                    x_proj = self.pre_projs[ell](x0)
                    per_cycle = self.aggregators[ell](
                        x_proj, cycles_ell, signs_ell,
                    )
                else:
                    x_proj = self.pre_projs[ell](x0)
                    cv_feats = x_proj[cycles_ell]
                    per_cycle = self.aggregators[ell](
                        cv_feats, signs_ell.float(),
                    )
                h_ell = self._scatter_mean(per_cycle, cycles_ell, N)
            h_parts.append(h_ell)
        x_final = torch.cat([x0, torch.cat(h_parts, dim=-1)], dim=-1)
        return self._edge_logits(x_final, edges_to_score)

    def _forward_route_capsule_soft(
        self,
        node_features: torch.Tensor,
        cycles: torch.Tensor,
        cycle_signs: torch.Tensor,
        tier_of: torch.Tensor,
        edges_to_score: torch.Tensor,
    ) -> torch.Tensor:
        """Route mode with learned cycle→tier softmax (CapsNet-style).

        ``tier_of`` is ignored; kept for API parity with structural routing.

        Resolved mode (``self._capsule_soft_router_resolved``):

        * ``mlp_softmax``: one MLP on mean-pooled corner features → softmax.
        * ``hypergraph_conv``: one HGNN-style **signed** pass on
          cycles-as-hyperedges → softmax (single pass over ``M``, no EM loop).
        * ``em_agreement``: iterative routing-by-agreement (``nit >= 2``).
        """
        del tier_of
        N = node_features.shape[0]
        x0 = node_features
        M = int(cycles.shape[0])
        h_parts: list[torch.Tensor] = []
        if M == 0:
            for _ in range(self.L):
                h_parts.append(torch.zeros(
                    N, self.cfg.d_layer, device=x0.device, dtype=x0.dtype,
                ))
            x_final = torch.cat([x0, torch.cat(h_parts, dim=-1)], dim=-1)
            return self._edge_logits(x_final, edges_to_score)

        cv_feats = x0[cycles]
        proto = cv_feats.mean(dim=1)
        nit = int(self.cfg.capsule_routing_iterations)
        mode = self._capsule_soft_router_resolved

        if mode != "em_agreement":
            if mode == "mlp_softmax":
                router = self.capsule_router
                assert router is not None
                route_w = F.softmax(router(proto), dim=-1)
            else:
                assert mode == "hypergraph_conv"
                route_w = F.softmax(
                    self._capsule_hypergraph_routing_logits(
                        x0, cycles, cycle_signs,
                    ),
                    dim=-1,
                )
            for ell in range(self.L):
                per_cycle = self._capsule_aggregator_per_cycle(
                    ell, x0, cv_feats, cycles, cycle_signs,
                )
                w = route_w[:, ell : ell + 1]
                h_ell = self._scatter_mean(per_cycle * w, cycles, N)
                h_parts.append(h_ell)
            x_final = torch.cat([x0, torch.cat(h_parts, dim=-1)], dim=-1)
            return self._edge_logits(x_final, edges_to_score)

        init = self.capsule_init_logits
        assert init is not None
        b_logits = init(proto)
        h_parts_final: list[torch.Tensor] = []
        for t in range(nit):
            route_w = F.softmax(b_logits, dim=-1)
            h_step: list[torch.Tensor] = []
            per_cycles: list[torch.Tensor] = []
            for ell in range(self.L):
                pc = self._capsule_aggregator_per_cycle(
                    ell, x0, cv_feats, cycles, cycle_signs,
                )
                per_cycles.append(pc)
                h_ell = self._scatter_mean(
                    pc * route_w[:, ell : ell + 1], cycles, N,
                )
                h_step.append(h_ell)
            if t == nit - 1:
                h_parts_final = h_step
                break
            agreement = torch.empty(
                M, self.L, device=x0.device, dtype=x0.dtype,
            )
            for ell in range(self.L):
                vote = h_step[ell][cycles].mean(dim=1)
                agreement[:, ell] = (per_cycles[ell] * vote).sum(dim=-1)
            b_logits = b_logits + agreement
        x_final = torch.cat([x0, torch.cat(h_parts_final, dim=-1)], dim=-1)
        return self._edge_logits(x_final, edges_to_score)

    def _forward_route(
        self,
        node_features: torch.Tensor,
        cycles: torch.Tensor,
        cycle_signs: torch.Tensor,
        tier_of: torch.Tensor,
        edges_to_score: torch.Tensor,
    ) -> torch.Tensor:
        if self.cfg.tier_organization == "capsule_soft":
            return self._forward_route_capsule_soft(
                node_features, cycles, cycle_signs, tier_of, edges_to_score,
            )
        return self._forward_route_structural(
            node_features, cycles, cycle_signs, tier_of, edges_to_score,
        )

    def _forward_pyramid(
        self,
        node_features: torch.Tensor,
        cycles: torch.Tensor,
        cycle_signs: torch.Tensor,
        tier_of: torch.Tensor,
        edges_to_score: torch.Tensor,
    ) -> torch.Tensor:
        x = node_features
        n_vertices = x.shape[0]
        for ell in range(self.L):
            cycles_ell, signs_ell = self._tier_cycle_subset(
                cycles, cycle_signs, tier_of, ell,
            )
            if cycles_ell.shape[0] == 0:
                h_ell = torch.zeros(
                    n_vertices, self.cfg.d_layer, device=x.device, dtype=x.dtype,
                )
            else:
                if self.aggregator_kind == "mlp":
                    cv_feats = x[cycles_ell]
                    per_cycle = self.aggregators[ell](cv_feats)
                elif self.aggregator_kind == "hsikan":
                    x_proj = self.pre_projs[ell](x)
                    per_cycle = self.aggregators[ell](
                        x_proj, cycles_ell, signs_ell,
                    )
                else:
                    x_proj = self.pre_projs[ell](x)
                    cv_feats = x_proj[cycles_ell]
                    per_cycle = self.aggregators[ell](
                        cv_feats, signs_ell.float(),
                    )
                h_ell = self._scatter_mean(
                    per_cycle, cycles_ell, n_vertices,
                )
            x = torch.cat([x, h_ell], dim=-1)
        return self._edge_logits(x, edges_to_score)

    def forward(
        self,
        node_features: torch.Tensor,
        cycles: torch.Tensor,
        cycle_signs: torch.Tensor,
        tier_of: torch.Tensor,
        edges_to_score: torch.Tensor,
    ) -> torch.Tensor:
        if self.cfg.topology == "route":
            return self._forward_route(
                node_features, cycles, cycle_signs, tier_of, edges_to_score,
            )
        return self._forward_pyramid(
            node_features, cycles, cycle_signs, tier_of, edges_to_score,
        )


__all__ = [
    "TierSpec",
    "TierAggregator",
    "CapsuleHypergraphRouter",
    "CPMLConfig",
    "CPML",
    "restrict_cycles_to_tier",
]
