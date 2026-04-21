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


def build_incidence(*args, **kwargs) -> torch.Tensor:
    """Stub for the factor-view incidence builder.

    The factor view's emitted `__init__` calls `build_incidence(...)` to
    construct the sparse signed-incidence buffer `B`. This stub returns
    a zero tensor of the right shape if `shape` is given, otherwise a
    1×1 placeholder. Real version lives in `ehk_torch.ops.sparse_signed`.
    """
    shape = kwargs.get("shape", (1, 1))
    return torch.zeros(*shape)


__all__ = ["GGKSpec", "HypergraphConv", "build_incidence"]
