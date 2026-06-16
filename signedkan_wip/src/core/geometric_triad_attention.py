"""Geometric (quaternion + Clifford) attention pool of signed triads → vertices.

The leakage-free rotor-HSiKAN trails SiGAT by ~0.04 AUROC, and the gap is "partly
attention, not cycles": SiGAT's only edge is an attention readout. This module
gives HSiKAN a *stronger* readout — a signed-triad attention pool whose score is
a learned mix of two geometric channels over the same projected features:

  * **quaternion** channel — Hamilton-product real part
    ``Re(q ⊗ k) = q0 k0 − q1 k1 − q2 k2 − q3 k3`` (signature (+,−,−,−));
  * **Clifford** channel — Cl(2,0) geometric-product *scalar* part
    ``⟨q k⟩_0 = q0 k0 + q1 k1 + q2 k2 − q12 k12`` (signature (+,+,+,−)),

reusing ``signedkan_wip.src.sequence.clifford.geometric_product``. The two carry
*different* metric signatures, so the gate ``g = σ(γ)`` mixes genuinely distinct
inner products rather than two copies of one. The score is ``tanh`` (not softmax):
balanced triads vote ``+``, unbalanced ``−`` — the same signed-voting readout as
``core/attention.py``, generalised to a geometric score.

Plan: docs/plans/2026-06-17-geometric-attention-head/.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..sequence.clifford import CL2_DIM, geometric_product

# Hamilton-product real-part coefficients: Re(q⊗k) = Σ coeff_c · q_c · k_c.
_HAMILTON_REAL = (1.0, -1.0, -1.0, -1.0)


class GeometricTriadAttentionPool(nn.Module):
    """Pool per-triad embeddings into per-vertex embeddings by geometric attention.

    # Preconditions
    ``hidden % 4 == 0`` (multivector blocks of 4); ``channel`` in
    ``{"both", "quaternion", "clifford"}``; ``inc_vertex`` and ``inc_triad`` are
    1-D, equal-length ``long`` tensors listing every (vertex ∈ triad) incidence.
    # Postconditions
    ``forward`` returns ``(n_nodes, d_out)``; vertices with no incident triad get
    a zero row. Magnitude-normalised (``Σ w·v / Σ|w|``) so the pool is bounded and
    invariant to a vertex's triad count, while staying signed.
    # Invariants
    Parameter count is independent of graph size (projections are per-feature).
    """

    def __init__(self, d_node: int, d_triad: int, hidden: int, d_out: int,
                 channel: str = "both", temperature: float = 1.0,
                 eps: float = 1e-6) -> None:
        super().__init__()
        if hidden % CL2_DIM != 0:
            raise ValueError(f"hidden must be % {CL2_DIM} == 0; got {hidden}")
        if channel not in {"both", "quaternion", "clifford"}:
            raise ValueError(
                f"channel must be both|quaternion|clifford; got {channel!r}")
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0; got {temperature}")
        self.hidden = int(hidden)
        self.n_blocks = hidden // CL2_DIM
        self.channel = channel
        self.temperature = float(temperature)
        self.eps = float(eps)
        # Shared projections: both channels reinterpret the SAME (hidden) features
        # as n_blocks 4-tuples — as quaternions or as Cl(2,0) multivectors.
        self.W_q = nn.Linear(d_node, hidden, bias=False)
        self.W_k = nn.Linear(d_triad, hidden, bias=False)
        self.W_v = nn.Linear(d_triad, d_out, bias=False)
        self.gate = nn.Parameter(torch.zeros(()))   # σ(0)=0.5 → equal channels
        with torch.no_grad():
            self.W_q.weight.mul_(0.1)
            self.W_k.weight.mul_(0.1)

    # ── geometric scores (per incidence) ────────────────────────────────
    def _quaternion_score(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Re(q ⊗ k) summed over blocks. ``q``,``k``: (P, n_blocks, 4)."""
        coeff = torch.tensor(_HAMILTON_REAL, device=q.device, dtype=q.dtype)
        return ((q * coeff) * k).sum(dim=(-1, -2))

    def _clifford_score(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Cl(2,0) scalar part ⟨q k⟩_0 summed over blocks. (P, n_blocks, 4)."""
        return geometric_product(q, k)[..., 0].sum(dim=-1)

    def _score(self, q_flat: torch.Tensor, k_flat: torch.Tensor) -> torch.Tensor:
        """Mixed geometric score per incidence. ``q_flat``,``k_flat``: (P, hidden)."""
        P = q_flat.shape[0]
        q = q_flat.view(P, self.n_blocks, CL2_DIM)
        k = k_flat.view(P, self.n_blocks, CL2_DIM)
        scale = self.n_blocks ** -0.5
        if self.channel == "quaternion":
            s = self._quaternion_score(q, k)
        elif self.channel == "clifford":
            s = self._clifford_score(q, k)
        else:
            g = torch.sigmoid(self.gate)
            s = g * self._quaternion_score(q, k) + (1.0 - g) * self._clifford_score(q, k)
        return torch.tanh(s * scale / self.temperature)

    def forward(self, h_node: torch.Tensor, h_triad: torch.Tensor,
                inc_vertex: torch.Tensor, inc_triad: torch.Tensor) -> torch.Tensor:
        """``h_node``: (V, d_node) queries; ``h_triad``: (T, d_triad) keys/values;
        ``inc_vertex``/``inc_triad``: (P,) incidence index pairs.
        Returns ``(V, d_out)``."""
        assert inc_vertex.shape == inc_triad.shape, "incidence arrays must align"
        n_nodes = h_node.shape[0]
        q_flat = self.W_q(h_node)[inc_vertex]          # (P, hidden)
        k_flat = self.W_k(h_triad)[inc_triad]          # (P, hidden)
        w = self._score(q_flat, k_flat)                # (P,) signed weights
        v = self.W_v(h_triad)[inc_triad]               # (P, d_out)
        num = torch.zeros(n_nodes, v.shape[1], device=v.device, dtype=v.dtype)
        num.index_add_(0, inc_vertex, w.unsqueeze(-1) * v)
        den = torch.zeros(n_nodes, device=v.device, dtype=v.dtype)
        den.index_add_(0, inc_vertex, w.abs())
        return num / (den.unsqueeze(-1) + self.eps)


def build_vertex_triad_pairs(triad_v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten a ``(T, k)`` triad-vertex table into incidence index pairs.

    Returns ``(inc_vertex, inc_triad)`` each of shape ``(T*k,)``: for triad ``t``
    with vertices ``triad_v[t]``, emit one ``(v, t)`` pair per vertex. Mirrors the
    incidence ``build_vertex_triad_incidence`` builds as a sparse matrix, but as
    explicit pairs for the scatter-attention pool.
    """
    n_triads, k = triad_v.shape
    inc_vertex = triad_v.reshape(-1)
    inc_triad = torch.arange(n_triads, device=triad_v.device).repeat_interleave(k)
    return inc_vertex, inc_triad
