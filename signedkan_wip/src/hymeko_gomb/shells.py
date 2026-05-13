"""Shells of the HymeKo-Gömb cascade — the three role-distinct primitives.

OuterFIRShell (volume, V1-analogue), MiddleHSiKAN (V4-analogue), and
InnerCPMLCore (IT-analogue). See `hymeko_gomb/__init__.py` for the
package docstring; see `docs/plans/2026-05-11-hymeko-gomb-sphere/`
for the architectural plan.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import torch
import torch.nn as nn

from ..cpml import (
    ClifFIRTierAggregator, CPML, CPMLConfig, SignedKANTierAggregator,
    TierSpec,
)


# ─── Shell 1: Outer FIR volume ──────────────────────────────────────


class OuterFIRShell(nn.Module):
    """Volume of M parallel Clifford-FIR filter banks.

    Each bank is an independent ClifFIRTierAggregator with its own
    learnable Cl(0,1) coefficients (a, b) and bias. The shell concats
    the M per-bank per-vertex aggregated features into a single
    M·d_layer-wide vertex representation.

    Args:
        d_in:       input per-vertex feature dim
        d_layer:    output dim per FIR bank
        M:          number of parallel FIR banks
        cycle_k:    cycle arity (FIR filter length)

    Forward signature:
        x        : (N, d_in)   per-vertex features
        cycles   : (M_c, k)    cycle vertex indices (M_c = number of cycles)
        signs    : (M_c, k)    cycle edge signs
        return   : (N, M·d_layer) outer-shell per-vertex features
    """

    def __init__(self, d_in: int, d_layer: int, M: int = 8, cycle_k: int = 3):
        super().__init__()
        self.M = M
        self.d_layer = d_layer
        self.d_in = d_in
        self.cycle_k = cycle_k
        self.pre_projs = nn.ModuleList([
            nn.Linear(d_in, d_layer) for _ in range(M)
        ])
        self.banks = nn.ModuleList([
            ClifFIRTierAggregator(d_in=d_layer, k_arity=cycle_k)
            for _ in range(M)
        ])
        # Stagger initial Clifford coefficients across banks so each
        # bank starts at a slightly different filter pattern.
        with torch.no_grad():
            for m, bank in enumerate(self.banks):
                phase = (m + 1) / (M + 1)
                bank.coef_a.fill_(phase / cycle_k)
                bank.coef_b.fill_(-(1.0 - phase) / cycle_k)

    def forward(
        self,
        x: torch.Tensor,           # (N, d_in)
        cycles: torch.Tensor,      # (M_c, k) long
        signs: torch.Tensor,       # (M_c, k) int8 / float
    ) -> torch.Tensor:
        N = x.shape[0]
        if cycles.shape[0] == 0:
            return torch.zeros(
                N, self.M * self.d_layer, device=x.device, dtype=x.dtype,
            )
        # One fused matmul for all M pre-projections: (N,d_in) with
        # stacked weights (M,d_layer,d_in) → (M,N,d_layer).  Per-bank
        # gather+FIR+scatter stays sequential so we never materialise
        # (M, M_c, k, d) (memory hazard on large M_c).
        cycles_l = cycles.long()
        signs_f = signs.float()
        w_stack = torch.stack([p.weight for p in self.pre_projs], dim=0)
        b_stack = torch.stack([p.bias for p in self.pre_projs], dim=0)
        x_all = torch.einsum("ni,mji->mnj", x, w_stack) + b_stack.unsqueeze(1)
        bank_outputs: list[torch.Tensor] = []
        for m in range(self.M):
            cv_feats = x_all[m][cycles_l]
            per_cycle = self.banks[m](cv_feats, signs_f)
            bank_outputs.append(scatter_mean(per_cycle, cycles_l, N))
        return torch.cat(bank_outputs, dim=-1)


def scatter_mean(
    per_cycle: torch.Tensor,
    cycles: torch.Tensor,
    n_vertices: int,
) -> torch.Tensor:
    """Per-vertex mean of incident-cycle features.

    Preconditions:
        ``cycles`` has shape ``(M_c, k)`` with entries in
        ``[0, n_vertices)``.
        ``per_cycle`` is either ``(M_c, d)`` or ``(B, M_c, d)``.  The
        batched form uses the **same** ``cycles`` for every batch slice
        (parallel banks in ``OuterFIRShell``).

    Postconditions:
        Returns ``(n_vertices, d)`` or ``(B, n_vertices, d)`` matching
        the leading dim of ``per_cycle``.
    """
    c = cycles.long()
    if per_cycle.dim() == 2:
        return _scatter_mean_flat(per_cycle, c, n_vertices)
    if per_cycle.dim() == 3:
        return _scatter_mean_batched(per_cycle, c, n_vertices)
    raise ValueError(
        "scatter_mean expects per_cycle with shape (M_c, d) or (B, M_c, d); "
        f"got dim {per_cycle.dim()}"
    )


def _scatter_mean_flat(
    per_cycle: torch.Tensor,
    cycles: torch.Tensor,
    n_vertices: int,
) -> torch.Tensor:
    """Scatter-mean for ``(M_c, d)`` — single ``index_add`` over flattened corners."""
    m_c, k = cycles.shape
    d = per_cycle.shape[-1]
    device = per_cycle.device
    dtype = per_cycle.dtype
    vidx = cycles.reshape(-1)
    contrib = per_cycle.unsqueeze(1).expand(-1, k, -1).reshape(m_c * k, d)
    out = torch.zeros(n_vertices, d, device=device, dtype=dtype)
    counts = torch.zeros(n_vertices, device=device, dtype=dtype)
    out.index_add_(0, vidx, contrib)
    counts.index_add_(
        0, vidx, torch.ones(m_c * k, device=device, dtype=dtype),
    )
    return out / counts.clamp_min(1.0).unsqueeze(-1)


def _scatter_mean_batched(
    per_cycle: torch.Tensor,
    cycles: torch.Tensor,
    n_vertices: int,
) -> torch.Tensor:
    """Scatter-mean for ``(B, M_c, d)`` with shared ``cycles`` (B = bank count)."""
    b, m_c, d_dim = per_cycle.shape
    k = cycles.shape[1]
    device = per_cycle.device
    dtype = per_cycle.dtype
    cyc_flat = cycles.reshape(-1)
    mk = m_c * k
    flat_idx = (
        torch.arange(b, device=device, dtype=torch.long).view(b, 1) * n_vertices
        + cyc_flat.view(1, mk)
    )
    contrib = (
        per_cycle.unsqueeze(2).expand(-1, -1, k, -1).reshape(b, mk, d_dim)
    )
    flat_idx_flat = flat_idx.reshape(-1)
    contrib_flat = contrib.reshape(b * mk, d_dim)
    out_flat = torch.zeros(b * n_vertices, d_dim, device=device, dtype=dtype)
    counts_flat = torch.zeros(b * n_vertices, device=device, dtype=dtype)
    out_flat.index_add_(0, flat_idx_flat, contrib_flat)
    counts_flat.index_add_(
        0, flat_idx_flat, torch.ones(b * mk, device=device, dtype=dtype),
    )
    out = out_flat.view(b, n_vertices, d_dim)
    counts = counts_flat.view(b, n_vertices)
    return out / counts.clamp_min(1.0).unsqueeze(-1)


# ─── Shell 2: Middle HSiKAN (CR-spline) ─────────────────────────────


class MiddleHSiKAN(nn.Module):
    """Single Catmull-Rom signed-spline aggregator.

    Wraps `SignedKANTierAggregator` (the per-tier wrapper around the
    SignedKANLayer with CR spline kind). The middle shell provides
    the nonlinearity the outer FIR shell lacks.

    Forward signature:
        x_outer   : (N, d_in)        outer-shell output
        cycles    : (M_c, k)
        signs     : (M_c, k)
        return    : (N, d_layer)     refined per-vertex features
    """

    def __init__(
        self, n_nodes: int, d_in: int, d_layer: int,
        cycle_k: int = 3, grid: int = 5,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.d_in = d_in
        self.d_layer = d_layer
        self.cycle_k = cycle_k
        self.pre_proj = nn.Linear(d_in, d_layer)
        self.agg = SignedKANTierAggregator(
            n_nodes=n_nodes, hidden_dim=d_layer, grid=grid,
            k_arity=cycle_k, spline_kind="catmull_rom",
        )

    def forward(
        self,
        x_outer: torch.Tensor,
        cycles: torch.Tensor,
        signs: torch.Tensor,
    ) -> torch.Tensor:
        N = x_outer.shape[0]
        x_proj = self.pre_proj(x_outer)                  # (N, d_layer)
        if cycles.shape[0] == 0:
            return torch.zeros_like(x_proj)
        per_cycle = self.agg(x_proj, cycles, signs)      # (M_c, d_layer)
        return scatter_mean(per_cycle, cycles, N)        # (N, d_layer)


# ─── Shell 3: Inner CPML core ───────────────────────────────────────


class InnerCPMLCore(nn.Module):
    """CPML tier-stratified multi-layer aggregation at the core.

    Default L=3 (periphery → mid → centre). Uses the MLP per-tier
    aggregator since the spheres already host the specialised
    aggregators (Clifford-FIR outside, HSiKAN-CR middle).

    ``topology="route"`` (default): each tier routes its cycle pool through
    an aggregator that reads **base** ``d_in`` features (no widening
    concat between tiers inside the CPML stack).

    ``topology="pyramid"``: legacy widening-concat inward stack (larger
    aggregators and activation footprint from tier 1 onward).

    ``tier_organization`` (``CPMLConfig``): ``structural`` (default) uses
    hard cycle→tier masks from ``tier_of``; ``capsule_soft`` adds a learned
    softmax router over tiers (**route** topology only).
    """

    def __init__(
        self,
        d_in: int,
        d_layer: int,
        n_tiers: int = 3,
        cycle_k: int = 3,
        *,
        topology: Literal["route", "pyramid"] = "route",
        tier_organization: Literal["structural", "capsule_soft"] = "structural",
        capsule_route_hidden: int = 64,
        capsule_routing_iterations: int = 1,
        capsule_soft_router: Literal[
            "auto", "mlp_softmax", "hypergraph_conv", "em_agreement",
        ] = "auto",
        capsule_hg_hidden: int = 64,
        capsule_hg_cache_degrees: bool = True,
        torch_compile_hypergraph: bool = False,
    ):
        super().__init__()
        cuts = tuple(np.linspace(0.0, 1.0, n_tiers + 1).tolist())
        self.cfg = CPMLConfig(
            tier_spec=TierSpec(cuts=cuts),
            d_in=d_in, d_layer=d_layer,
            aggregator_kind="mlp",
            cycle_k=cycle_k,
            topology=topology,
            tier_organization=tier_organization,
            capsule_route_hidden=capsule_route_hidden,
            capsule_routing_iterations=capsule_routing_iterations,
            capsule_soft_router=capsule_soft_router,
            capsule_hg_hidden=capsule_hg_hidden,
            capsule_hg_cache_degrees=capsule_hg_cache_degrees,
            torch_compile_hypergraph=torch_compile_hypergraph,
        )
        self.cpml = CPML(self.cfg)

    @property
    def final_dim(self) -> int:
        return self.cpml.in_dims[-1]

    def forward(
        self,
        x_middle: torch.Tensor,        # (N, d_in)
        cycles: torch.Tensor,          # (M_c, k)
        signs: torch.Tensor,           # (M_c, k)
        tier_of: torch.Tensor,         # (N,) long
        edges_to_score: torch.Tensor,  # (E, 2)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.cpml(
            x_middle, cycles, signs, tier_of, edges_to_score,
        )
        return scores, x_middle


__all__ = ["OuterFIRShell", "MiddleHSiKAN", "InnerCPMLCore", "scatter_mean"]
