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


class SignedKANLayer(nn.Module):
    """Stub Tier-3 ``signedkan_layer``.

    The real layer realises Option-C signed-incidence aggregation
    over $k$-cycle signatures: per signed cycle $c=(v_1,\\ldots,v_k)$
    with signs $\\sigma$,
    $h_c = \\sum_{s \\in \\{+,-,\\sim 0\\}}
              \\phi_e^s(\\sum_{i: \\sigma_i = s} \\phi_v^s(h_{v_i}))$
    with per-channel Catmull--Rom (or B-spline / Kochanek--Bartels)
    activations $\\phi_v^s, \\phi_e^s$.

    This stub is just two `nn.Linear`s wrapping the input — same field
    surface (hidden / arity / spline_kind / grid) so emitted networks
    construct without error.  Round-trip parity testing requires the
    real layer from `signedkan_wip.src.signedkan`; this stub keeps the
    codegen path importable.
    """
    def __init__(self, hidden: int, arity: int,
                 spline_kind: str = "catmull_rom", grid: int = 5):
        super().__init__()
        self.hidden = hidden
        self.arity = arity
        self.spline_kind = spline_kind
        self.grid = grid
        self.inner = nn.Linear(hidden, hidden)
        self.outer = nn.Linear(hidden, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    """Stub Tier-3 ``arity_mixer``.

    The real mixer applies sparse signed-incidence matrices $M_e^{(k)}$
    to per-arity cycle embeddings, weighted by softmax-normalised
    learnable $\\alpha_k$:
    $h_e = \\sum_{k=1}^{K} \\mathrm{softmax}(\\alpha)_k \\cdot
            M_e^{(k)} h_c^{(k)}$.

    Stub semantics: maintains the $K$-vector $\\alpha$ and a tied
    nn.Linear projection.  `forward(*args)` accepts any of the
    per-arity inputs (the dataflow walker emits `mixer(cyc_k_emb)`
    once per arity, fanning into the same sink); the stub sums them
    after a per-arity weight and projects.
    """
    def __init__(self, hidden: int, mix_K: int):
        super().__init__()
        self.hidden = hidden
        self.mix_K = mix_K
        self.alpha = nn.Parameter(torch.zeros(mix_K))
        self.proj = nn.Linear(hidden, hidden)

    def forward(self, *xs: torch.Tensor) -> torch.Tensor:
        # Multi-input fan-in: the dataflow emitter calls
        # `mixer(cyc_2_emb, cyc_3_emb, ..., cyc_K_emb)` once per
        # forward pass with all per-arity cycle embeddings as
        # positional arguments.  Single-input call (one arity) also
        # works.  Real mixer applies the per-arity sparse-incidence
        # mm M_e^{(k)} before the alpha-weighted sum; this stub
        # weights the inputs directly by softmax(alpha).
        if not xs:
            raise ValueError("ArityMixer.forward needs >=1 input")
        w = torch.softmax(self.alpha[:len(xs)], dim=-1)
        h = sum(w[i] * x for i, x in enumerate(xs))
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
