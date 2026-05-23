"""
ehk_torch_stub — minimal placeholder runtime for the HyMeKo torch_dataflow
codegen path.

This module defines just enough of the eventual `ehk_torch` API surface
that PyTorch source files emitted from HyMeKo descriptions can be
imported, instantiated, and forward-passed end-to-end. The math is
placeholder (a plain `nn.Linear` masquerading as `HypergraphConv`); the
real entropy stage, GGK kernel, and signed sparse ops are part of the
full `ehk_torch` package which lives outside this workspace.

Use this stub only for codegen-roundtrip tests. Do not train models
with it — the layers are not what they advertise.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class GGKSpec:
    """Stub GGK specification carrying just the parameters HyMeKo emits.

    The real `ehk_torch.kernels.ggk.GGKSpec` will be a structured
    dataclass plus an `nn.Module` basis evaluator. This stub stores the
    parameters as an opaque record so emitted `GGKSpec(basis="bspline",
    degree=3, n_knots=8)` calls construct without error.
    """
    basis: str
    degree: Optional[int] = None
    n_knots: Optional[int] = None
    n_centres: Optional[int] = None


class HypergraphConv(nn.Module):
    """Stub `HypergraphConv` — a plain `nn.Linear(d_in, d_out)`
    underneath. Accepts a `ggk_spec` argument and stores it for
    inspection; ignores it during forward.

    The real layer applies a signed-incidence matmul + GGK-parameterised
    activation; this stub just runs the linear map so emitted networks
    have a well-typed forward path for round-trip testing.
    """
    def __init__(self, d_in: int, d_out: int, ggk_spec: Optional[GGKSpec] = None):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.ggk_spec = ggk_spec
        self.linear = nn.Linear(d_in, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class ResidualBlock(nn.Module):
    """Tier-2 composite block: y = x + Linear(ReLU(Linear(x))).
    `hidden` sets the input/output dimension (must match for the
    additive skip). Shipped here so HyMeKo descriptions referencing
    `residual_block` instantiate cleanly via the codegen path.
    """
    def __init__(self, hidden: int):
        super().__init__()
        self.hidden = hidden
        self.l0 = nn.Linear(hidden, hidden)
        self.l1 = nn.Linear(hidden, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.l0(x))
        return x + self.l1(h)


class HighwayBlock(nn.Module):
    """Tier-2 composite block: highway network unit.
    y = T(x) ⊙ F(x) + (1 − T(x)) ⊙ x  with T learnt sigmoid gate
    and F a single Linear+ReLU.
    """
    def __init__(self, hidden: int):
        super().__init__()
        self.hidden = hidden
        self.transform = nn.Linear(hidden, hidden)
        self.gate = nn.Linear(hidden, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = torch.sigmoid(self.gate(x))
        f = torch.relu(self.transform(x))
        return t * f + (1.0 - t) * x


def build_incidence(*args, **kwargs) -> torch.Tensor:
    """Stub for the factor-view incidence builder.

    The factor view's emitted `__init__` calls `build_incidence(...)` to
    construct the sparse signed-incidence buffer `B`. This stub returns
    a zero tensor of the right shape if `shape` is given, otherwise a
    1×1 placeholder. Real version lives in `ehk_torch.ops.sparse_signed`.
    """
    shape = kwargs.get("shape", (1, 1))
    return torch.zeros(*shape)


# ─── Tier-3: signed-cycle KAN primitives (HSiKAN) ─────────────────────


def _try_load_real_signedkan():
    """Lazy import of the real Option-C SignedKANLayer.

    Returned on success: tuple (SignedKANLayer, SignedKANConfig).
    Returns None if signedkan_wip is not on the import path — in which
    case SignedKANLayer below falls back to its stub linear+tanh+linear
    math, preserving the codegen smoke test's "importable" guarantee.
    """
    try:
        from signedkan_wip.src.core.signedkan import (
            SignedKANLayer as _Real,
            SignedKANConfig as _Cfg,
        )
        return _Real, _Cfg
    except Exception:
        return None


class SignedKANLayer(nn.Module):
    """Tier-3 ``signedkan_layer`` — delegates to the real signedkan_wip
    Option-C signed-incidence layer when its inputs are present.

    Forward signatures:
      * ``forward(x, triad_v, triad_sigma)`` — real path.  ``triad_v``
        is ``(n_cycles, k)`` of vertex IDs and ``triad_sigma`` is
        ``(n_cycles, k)`` ∈ {+1, -1}.  Output: ``(n_cycles, hidden)``.
        Mirrors signedkan_wip.src.core.signedkan.SignedKANLayer.forward.
      * ``forward(x)`` — stub fallback for codegen smoke tests.

    The real layer's per-sign Option-C aggregation is:
    $h_c = \\sum_{s \\in \\{+,-\\}} \\phi_e^s(
              \\sum_{i: \\sigma_i = s} \\phi_v^s(h_{v_i}))$
    with batched Catmull--Rom / B-spline / Kochanek--Bartels splines
    on the inner $\\phi_v^s$ and a diagonal-fused outer $\\phi_e^s$.
    """
    def __init__(self, hidden: int, arity: int,
                 spline_kind: str = "catmull_rom", grid: int = 5):
        super().__init__()
        self.hidden = hidden
        self.arity = arity
        self.spline_kind = spline_kind
        self.grid = grid

        loaded = _try_load_real_signedkan()
        if loaded is not None:
            _Real, _Cfg = loaded
            # base SignedKANLayer ignores cfg.n_nodes; placeholder is fine.
            cfg = _Cfg(
                n_nodes=1, hidden_dim=hidden, k=arity, grid=grid,
                spline_kind=spline_kind,
            )
            self._real = _Real(cfg)
        else:
            self._real = None

        # Stub fallback parameters — also kept around so spectral_weights
        # has something to reference when the real path isn't taken.
        self.inner = nn.Linear(hidden, hidden)
        self.outer = nn.Linear(hidden, hidden)

    def forward(self, x: torch.Tensor,
                triad_v: torch.Tensor | None = None,
                triad_sigma: torch.Tensor | None = None) -> torch.Tensor:
        if self._real is not None and triad_v is not None and triad_sigma is not None:
            return self._real(x, triad_v, triad_sigma)
        return self.outer(torch.tanh(self.inner(x)))


class WalkLayer(nn.Module):
    """Stub Tier-3 ``walk_layer`` — open-walk sibling of SignedKANLayer.

    Same Option-C signed-aggregation structure as SignedKANLayer but
    consumes length-`walk_len` simple walks (open paths) instead of
    closed cycles.  Stub math is identical to SignedKANLayer (linear
    + tanh + linear); the real layer would apply the open-walk-aware
    σ-mask aggregation via `hymeko.enumerate_k_walks_rs` ouputs.
    """
    def __init__(self, hidden: int, walk_len: int,
                 spline_kind: str = "catmull_rom", grid: int = 5):
        super().__init__()
        self.hidden = hidden
        self.walk_len = walk_len
        self.spline_kind = spline_kind
        self.grid = grid
        self.inner = nn.Linear(hidden, hidden)
        self.outer = nn.Linear(hidden, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.outer(torch.tanh(self.inner(x)))


class ArityMixer(nn.Module):
    """Tier-3 ``arity_mixer`` — αₖ-weighted sparse-incidence fusion.

    The real mixer applies sparse signed-incidence matrices $M_e^{(k)}$
    to per-arity cycle embeddings, weighted by softmax-normalised
    learnable $\\alpha_k$:
    $h_e = \\sum_{k=1}^{K} \\mathrm{softmax}(\\alpha)_k \\cdot
            M_e^{(k)} h_c^{(k)}$.

    Forward signatures (selected by argument count):
      * ``forward(cyc_emb_k0, ..., cyc_emb_kK-1, M_e_k0, ..., M_e_kK-1)``
        — real path.  ``cyc_emb_kK`` is ``(n_cycles_kK, hidden)``;
        ``M_e_kK`` is ``(n_test_edges, n_cycles_kK)`` (sparse or dense).
        Output: ``(n_test_edges, hidden)``.
      * ``forward(cyc_emb_k0, ..., cyc_emb_kK-1)`` — stub fallback for
        codegen smoke tests; weights and projects without M_e.
    """
    def __init__(self, hidden: int, mix_K: int):
        super().__init__()
        self.hidden = hidden
        self.mix_K = mix_K
        self.alpha = nn.Parameter(torch.zeros(mix_K))
        self.proj = nn.Linear(hidden, hidden)

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        if not args:
            raise ValueError("ArityMixer.forward needs >=1 input")
        if len(args) == 2 * self.mix_K:
            # Real path: per-arity (cyc_emb, M_e) pairs.
            cyc_embs = args[: self.mix_K]
            M_es = args[self.mix_K:]
            alpha = torch.softmax(self.alpha, dim=-1)
            edge_emb = None
            for k in range(self.mix_K):
                M_e_k, h_k = M_es[k], cyc_embs[k]
                contrib = alpha[k] * (
                    torch.sparse.mm(M_e_k, h_k)
                    if M_e_k.is_sparse else (M_e_k @ h_k)
                )
                edge_emb = contrib if edge_emb is None else edge_emb + contrib
            return edge_emb
        # Stub fallback: cyc_emb tensors only, no M_e — weight + project.
        K = len(args)
        w = torch.softmax(self.alpha[:K], dim=-1)
        h = sum(w[i] * x for i, x in enumerate(args))
        return self.proj(torch.tanh(h))


class SignedClassifier(nn.Module):
    """Stub Tier-3 ``signed_classifier``.

    Linear classifier head over edge embeddings.  Identical semantics
    to `nn.Linear(d_in, d_out)`; named separately so the dataflow
    walker can distinguish "head" Linears from "body" ones for
    spectral-regulariser inclusion lists.
    """
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.linear = nn.Linear(d_in, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


from .proposal import SplitProposal, load_proposal, ClusterTag
from .hotswap import (
    TransferReport,
    transfer_compatible_weights,
    reinfer_structure_and_rebuild,
)

__all__ = [
    "GGKSpec", "HypergraphConv", "build_incidence",
    "ResidualBlock", "HighwayBlock",
    "SignedKANLayer", "WalkLayer", "ArityMixer", "SignedClassifier",
    "SplitProposal", "load_proposal", "ClusterTag",
    "TransferReport", "transfer_compatible_weights",
    "reinfer_structure_and_rebuild",
]
