"""Concentric-Pyramid Multi-Layer (CPML) signed-graph link-prediction
architecture (2026-05-11).

Plan:
    docs/plans/2026-05-11-cpml-xhc-architectures/plan.{tex,pdf,tikz,mmd}

This module is the **feasibility scaffold** — it implements the
tier stratification + inward-funnelling topology with a simplified
per-layer aggregator (small MLP over tier-restricted cycle features).
Unit tests in `signedkan_wip/tests/test_cpml.py` validate:

  * stratification correctness (tier sizes match percentile cuts)
  * forward shapes at L ∈ {2, 3, 4}
  * backward gradient flow (no NaN)
  * L=1 reduces to a flat aggregator (sanity)
  * innermost-tier embedding has received contributions from outer
    tiers (inward funnelling)

Once the feasibility gate is green, the simplified per-layer block
will be swapped for the full `MixedAritySignedKAN` and the model will
be wired into `run_final_cell.py` via `--model CPML`.

The architecture mirrors V1 → V2 → V4 → IT (Felleman & Van Essen 1991):
each tier increases receptive-field size and abstraction, with outer
tiers providing fine-grained signal that inner tiers integrate.
"""
from __future__ import annotations

from dataclasses import dataclass, field

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


class CPML(nn.Module):
    """Concentric-Pyramid Multi-Layer signed-graph link predictor.

    Forward signature:
        node_features : (N, d_in)            initial vertex features
        cycles        : (M, k) int64         all simple cycles in G
        cycle_signs   : (M, k) int8          ±1 edge sign of each
                                             cycle's k boundary edges
        tier_of       : (N,)   int64         tier index per vertex
        edges_to_score: (E, 2) int64         (u, v) pairs to score

    Returns:
        scores : (E,) float                  edge sign predictions

    Layer ℓ ∈ {0, ..., L-1}:
        cycles_ℓ = cycles touching at least one vertex of T_ℓ
        per_vertex_ℓ = scatter-mean over cycles_ℓ of TierAggregator_ℓ
                       (vertex features at each cycle's vertices)
        X_{ℓ+1}[v] = concat(X_ℓ[v], per_vertex_ℓ[v])

    The final per-vertex embedding X_L is consumed by the MLP head.
    """

    def __init__(self, cfg: CPMLConfig):
        super().__init__()
        self.cfg = cfg
        L = cfg.tier_spec.L
        self.L = L
        # Layer ℓ takes the running concat of (X_0, H_1, ..., H_ℓ) as
        # input.  Track the input dim per layer.
        in_dims: list[int] = [cfg.d_in]
        for ell in range(L):
            in_dims.append(in_dims[-1] + cfg.d_layer)
        self.in_dims = in_dims
        # Per-tier aggregators.  Factory choice = orthogonal aggregator
        # dimension (MLP vs HSiKAN).  For HSiKAN, output dim equals
        # input dim (SignedKANLayer is a self-map), so we project to
        # cfg.d_layer afterwards via the post-projection.
        self.aggregator_kind = cfg.aggregator_kind
        self.aggregators = nn.ModuleList()
        # For HSiKAN aggregators we need an explicit pre-projection to
        # bring the running concat into the SignedKANLayer's hidden_dim,
        # and a post-projection back to cfg.d_layer for the next concat.
        self.pre_projs = nn.ModuleList()
        self.post_projs = nn.ModuleList()
        for ell in range(L):
            if cfg.aggregator_kind == "mlp":
                self.aggregators.append(TierAggregator(
                    d_in=in_dims[ell],
                    d_hidden=cfg.d_layer * 2,
                    d_out=cfg.d_layer,
                ))
                self.pre_projs.append(nn.Identity())
                self.post_projs.append(nn.Identity())
            elif cfg.aggregator_kind == "hsikan":
                if cfg.n_nodes is None:
                    raise ValueError(
                        "CPMLConfig.n_nodes must be set for aggregator_kind='hsikan'"
                    )
                # Pre-project running concat (in_dims[ell]) → d_layer
                # so SignedKANLayer's self-map output is at d_layer.
                self.pre_projs.append(nn.Linear(in_dims[ell], cfg.d_layer))
                self.aggregators.append(SignedKANTierAggregator(
                    n_nodes=cfg.n_nodes,
                    hidden_dim=cfg.d_layer,
                    grid=cfg.hsikan_grid,
                    spline_kind=cfg.hsikan_spline_kind,
                ))
                self.post_projs.append(nn.Identity())
            elif cfg.aggregator_kind == "clifford_fir":
                # Clifford-FIR: tiny 2K+d-parameter signed FIR.  Same
                # math as the Rust hymeko_graph::spine::CliffordFIR.
                # Operates as a self-map at width d_layer; pre-project
                # the running concat (in_dims[ell]) → d_layer.
                self.pre_projs.append(nn.Linear(in_dims[ell], cfg.d_layer))
                self.aggregators.append(ClifFIRTierAggregator(
                    d_in=cfg.d_layer, k_arity=cfg.cycle_k,
                ))
                self.post_projs.append(nn.Identity())
            else:
                raise ValueError(
                    f"unknown aggregator_kind {cfg.aggregator_kind!r}; "
                    "valid: 'mlp', 'hsikan', 'clifford_fir'"
                )
        # Edge predictor head.
        final_dim = in_dims[L]   # = d_in + L * d_layer
        self.head = nn.Sequential(
            nn.Linear(2 * final_dim, cfg.d_predictor_hidden),
            nn.GELU(),
            nn.Linear(cfg.d_predictor_hidden, 1),
        )

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

    def forward(
        self,
        node_features: torch.Tensor,         # (N, d_in)
        cycles: torch.Tensor,                # (M, k) long
        cycle_signs: torch.Tensor,           # (M, k) — unused here, real
                                             # HSiKAN will use them
        tier_of: torch.Tensor,               # (N,) long
        edges_to_score: torch.Tensor,        # (E, 2) long
    ) -> torch.Tensor:
        N, d_in = node_features.shape
        X = node_features                     # rolling concat across tiers
        for ell in range(self.L):
            # Tier-restriction: keep cycles touching T_ℓ.
            cycle_tiers = tier_of[cycles]            # (M, k)
            tier_mask = (cycle_tiers == ell).any(dim=1)   # (M,)
            cycles_ell = cycles[tier_mask]                # (M_ℓ, k)
            signs_ell = cycle_signs[tier_mask]            # (M_ℓ, k)
            if cycles_ell.shape[0] == 0:
                H_ell = torch.zeros(N, self.cfg.d_layer,
                                     device=X.device, dtype=X.dtype)
            else:
                if self.aggregator_kind == "mlp":
                    # MLP path: gather vertex features at cycle indices,
                    # apply per-cycle MLP, pool to per-cycle.
                    cv_feats = X[cycles_ell]            # (M_ℓ, k, d_in_ℓ)
                    per_cycle = self.aggregators[ell](cv_feats)
                elif self.aggregator_kind == "hsikan":
                    # HSiKAN path: pre-project X to layer width, hand
                    # to SignedKANLayer with cycles + signs.  Output
                    # is (M_ℓ, d_layer) per-cycle (== per-hyperedge).
                    x_proj = self.pre_projs[ell](X)     # (N, d_layer)
                    per_cycle = self.aggregators[ell](
                        x_proj, cycles_ell, signs_ell,
                    )
                else:  # "clifford_fir"
                    # Clifford-FIR path: pre-project, gather vertex
                    # features at cycle positions, apply sign-branched
                    # FIR with closed-form Clifford coefficients.
                    x_proj = self.pre_projs[ell](X)     # (N, d_layer)
                    cv_feats = x_proj[cycles_ell]      # (M_ℓ, k, d_layer)
                    per_cycle = self.aggregators[ell](
                        cv_feats, signs_ell.float(),
                    )
                H_ell = self._scatter_mean(per_cycle, cycles_ell, N)
            X = torch.cat([X, H_ell], dim=-1)        # widening concat

        # Edge predictor over the final concat.
        u = X[edges_to_score[:, 0]]                  # (E, d_final)
        v = X[edges_to_score[:, 1]]
        pair = torch.cat([u, v], dim=-1)             # (E, 2*d_final)
        return self.head(pair).squeeze(-1)           # (E,)


__all__ = [
    "TierSpec",
    "TierAggregator",
    "CPMLConfig",
    "CPML",
    "restrict_cycles_to_tier",
]
