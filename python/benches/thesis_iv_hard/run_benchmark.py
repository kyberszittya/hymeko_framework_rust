"""Thesis IV on real vision datasets — MNIST + CIFAR-10.

Extends the Iris / synthetic reproduction in `python/benches/thesis_iv/`
to harder datasets where the spectral-entropy regularizer has more
room to show effect, *and* adds the KL-between-consecutive-steps
regularizer (Eq 6.3 / 6.4 of the thesis) which is what the thesis's
subthesis 4.3 actually proposes — "used between consecutive
optimization runs to improve the learning process."

## Three arms

- `baseline`          — cross-entropy only.
- `scalar_entropy`    — `J + λ · I(H)`. Direct spectral-entropy
                        minimization. Simpler than the thesis form
                        but matches the direction of Figure 6.5a
                        (entropy decreases during training).
- `kl_trajectory`     — `J + λ · D_KL(λ̂_prev ‖ λ̂_curr)`. Thesis
                        Eq 6.3/6.4: penalize step-to-step change in
                        the spectral distribution of the aggregated
                        normalized Laplacian. Forward KL pulls the
                        current architecture toward the previous
                        step's spectrum, smoothing the trajectory.

## Spectral entropy recap

For an MLP with weight matrices `W_l`:
- Build block-tridiagonal neuron-level adjacency `A_ij = |W_l[j,i]|`.
- Laplacian `L = D - A`, where `D[i] = Σ_j A[i,j]`.
- Aggregated normalization `L̂ = L / Σ D`. Since `trace(L) = Σ D`,
  we get `trace(L̂) = 1`, so eigenvalues `λ̂_i` form a valid
  probability distribution.
- Algebraic entropy `I(H) = -Σ λ̂_i log₂ λ̂_i` (Eq 6.2).
- KL between two hypergraph states (treating sorted eigenvalues as
  matched distributions): `D_KL(λ̂^t ‖ λ̂^{t+1}) = Σ λ̂_i^t log(λ̂_i^t / λ̂_i^{t+1})`.

## Compute strategy

Eigendecomposition of the full neuron adjacency matrix is O(N³). For
MNIST's 784→256→128→64→10 MLP that's N=1242, so ~2e9 FLOPs per call —
tractable on GPU. For CIFAR-10 (3072→…) the full-input neuron count
would make eigvalsh too expensive per step, so the CIFAR-10 model
uses a compact MLP with a fixed flatten-and-project first layer; the
benchmark runs the spectral regularizer on the spectral part of the
network only (ignoring the fixed projector).

The regularizer is updated every `--reg-every-n-batches` batches (not
every batch) to keep overhead bounded. The thesis's "between
consecutive optimization runs" supports any stride; `reg_every_n=1`
reproduces the strict step-by-step form at higher cost.
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ─── Spectral entropy (shared with simpler benchmark) ───────────────


def build_adjacency(weights: list[torch.Tensor]) -> torch.Tensor:
    """Block-tridiagonal symmetric adjacency from MLP layer weights —
    the **dataflow view** / **bipartite view** per thesis §6.2.1.

    Neurons are nodes; edges exist only between adjacent layers via
    star-expansion (each weight is an arc from input-neuron to output-
    neuron). No within-layer edges. Captures *layer-to-layer
    information flow* in the network's spectrum."""
    device, dtype = weights[0].device, weights[0].dtype
    layer_sizes = [weights[0].shape[1]] + [w.shape[0] for w in weights]
    total = sum(layer_sizes)
    offsets = [0]
    for s in layer_sizes:
        offsets.append(offsets[-1] + s)

    A = torch.zeros(total, total, device=device, dtype=dtype)
    for l, W in enumerate(weights):
        block = W.abs().t()  # (n_l, n_{l+1})
        r0, r1 = offsets[l], offsets[l + 1]
        c0, c1 = offsets[l + 1], offsets[l + 2]
        A[r0:r1, c0:c1] = block
        A[c0:c1, r0:r1] = block.t()
    return A


def build_adjacency_factor_view(weights: list[torch.Tensor]) -> torch.Tensor:
    """**Factor view** per thesis §6.1.2, via clique expansion.

    Each output neuron defines a factor: a hyperedge containing all
    its input neurons (from the previous layer) plus itself. Clique
    expansion of this hyperedge creates pairwise edges:

    - Between inputs (co-contributing to the same output): weight =
      |W[j,i₁]| · |W[j,i₂]|, summed over all output neurons j. In
      closed form: `|W|ᵀ @ |W|` for each layer's weight matrix,
      producing within-layer symmetric blocks.
    - Between input and output (same as dataflow view): |W[j,i]|.
    - Between outputs: none (each factor has one output; outputs do
      not directly co-occur in the *same* factor).

    This view captures *correlation structure* — inputs that feed the
    same output become spectrally connected. For layer ℓ's weight
    matrix `W_ℓ`, the within-layer-(ℓ-1) block gets += `|W_ℓ|ᵀ @ |W_ℓ|`
    with the diagonal zeroed out (self-loops excluded).

    Note: a layer ℓ's neurons receive within-layer connectivity from
    the factors of layer ℓ+1 (where they serve as inputs), not from
    layer ℓ's own factors (where they are unique outputs).
    """
    device, dtype = weights[0].device, weights[0].dtype
    layer_sizes = [weights[0].shape[1]] + [w.shape[0] for w in weights]
    total = sum(layer_sizes)
    offsets = [0]
    for s in layer_sizes:
        offsets.append(offsets[-1] + s)

    A = torch.zeros(total, total, device=device, dtype=dtype)
    for l, W in enumerate(weights):
        absW = W.abs()
        block = absW.t()  # (n_l, n_{l+1}) — same input→output as dataflow

        # Between-layer (dataflow) contribution.
        r0, r1 = offsets[l], offsets[l + 1]
        c0, c1 = offsets[l + 1], offsets[l + 2]
        A[r0:r1, c0:c1] = block
        A[c0:c1, r0:r1] = block.t()

        # Within-layer-ℓ clique contribution from factors in layer ℓ+1.
        # Pair (i₁, i₂) both in layer ℓ gets Σ_j |W[j,i₁]|·|W[j,i₂]|
        # = (|W|ᵀ @ |W|)[i₁, i₂]. Zero out diagonal (self-loops).
        within = absW.t() @ absW  # (n_l, n_l)
        within = within - torch.diag(torch.diag(within))
        A[r0:r1, r0:r1] += within

    return A


def normalized_laplacian_eigvals(
    weights_or_adj, eps: float = 1e-8,
) -> Optional[torch.Tensor]:
    """Eigenvalues of L̂ = (D − A) / Σ D. Returns a 1-D tensor of
    eigenvalues sorted ascending (eigvalsh's convention), or None if
    the total degree is numerically zero.

    Accepts either a list of weight matrices (plain MLP) or a
    pre-built adjacency tensor (for architectures with skip
    connections — ResNet, Highway — that need custom adjacency)."""
    if isinstance(weights_or_adj, torch.Tensor):
        A = weights_or_adj
    else:
        A = build_adjacency(weights_or_adj)
    D = A.sum(dim=1)
    total_degree = D.sum()
    if total_degree.item() < eps:
        return None
    L = torch.diag(D) - A
    L_hat = L / total_degree
    return torch.linalg.eigvalsh(L_hat)


def spectral_metrics(W: torch.Tensor) -> dict[str, float]:
    """Weight-matrix-level capacity proxies, computed from the SVD:
    - stable_rank: ||W||_F² / ||W||_2² — Bartlett-Foster-Telgarsky
      "effective dimension" used in their margin bound.
    - spectral_norm: ||W||_2 — largest singular value.
    - participation_ratio: 1 / Σ σᵢ⁴ (normalized) — inverse Rényi-2
      spread over singular values; another effective-rank proxy.
    All returned as plain floats for CSV/aggregation."""
    with torch.no_grad():
        sv = torch.linalg.svdvals(W.detach())
        sv2 = sv * sv
        fro2 = float(sv2.sum().item())
        smax2 = float(sv2.max().item())
        stable_rank = fro2 / smax2 if smax2 > 1e-12 else 0.0
        norm = sv.max().item()
        p = sv2 / sv2.sum().clamp_min(1e-12)
        pr = 1.0 / float((p * p).sum().item()) if float((p * p).sum().item()) > 0 else 0.0
    return {
        "stable_rank": stable_rank,
        "spectral_norm": norm,
        "participation_ratio": pr,
    }


def aggregate_spectral_metrics(weights: list[torch.Tensor]) -> dict[str, float]:
    """Per-layer metrics averaged into a single dict for flat CSV export."""
    metrics_per_layer = [spectral_metrics(W) for W in weights]
    out = {}
    for key in ("stable_rank", "spectral_norm", "participation_ratio"):
        vals = [m[key] for m in metrics_per_layer]
        out[f"{key}_mean"] = sum(vals) / len(vals) if vals else 0.0
        out[f"{key}_max"] = max(vals) if vals else 0.0
        # Product of spectral norms enters Bartlett's bound as a factor.
        if key == "spectral_norm":
            prod = 1.0
            for v in vals:
                prod *= v
            out["spectral_norm_product"] = prod
    return out


def spectral_entropy_bits(eigvals: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """I(H) = -Σ λ̂_i log₂ λ̂_i over non-zero eigenvalues. Differentiable."""
    mask = eigvals > eps
    lam = eigvals[mask]
    entropy_nats = -(lam * torch.log(lam)).sum()
    return entropy_nats / torch.log(torch.tensor(2.0, device=eigvals.device))


def kl_spectra(
    lam_prev: torch.Tensor, lam_curr: torch.Tensor, eps: float = 1e-8,
) -> torch.Tensor:
    """KL(prev ‖ curr) on matched sorted eigenvalue distributions.

    Both inputs are assumed sorted ascending and to sum to ~1 (which
    the aggregated normalization guarantees). Only entries where
    `lam_prev > eps` contribute. The `curr` side is clamped at `eps`
    to avoid log(0) — this adds at most `lam_prev * log(1/eps)` bias,
    small for reasonable eps."""
    mask = lam_prev > eps
    if not mask.any():
        return torch.zeros((), device=lam_prev.device, dtype=lam_prev.dtype)
    p = lam_prev[mask]
    q = lam_curr[mask].clamp_min(eps)
    return (p * (p.log() - q.log())).sum()


# ─── Models ─────────────────────────────────────────────────────────


class MNISTNetSmallNClass(nn.Module):
    """Thesis-scale MNIST MLP with configurable output dim.
    Used for EMNIST Letters (26 classes) at the same 784→16→8 backbone
    as the MNIST plain-MLP baseline."""

    def __init__(self, n_classes: int = 26):
        super().__init__()
        self.l0 = nn.Linear(784, 16)
        self.l1 = nn.Linear(16, 8)
        self.l2 = nn.Linear(8, n_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.l0(x))
        x = F.relu(self.l1(x))
        return self.l2(x)

    def spectral_weights(self):
        return [self.l0.weight, self.l1.weight, self.l2.weight]

    def spectral_adjacency(self, view="dataflow"):
        W = self.spectral_weights()
        return build_adjacency_factor_view(W) if view == "factor" else build_adjacency(W)


def _make_mnist_nclass(n_classes: int):
    def f():
        return MNISTNetSmallNClass(n_classes=n_classes)
    return f


class CapsMLP(nn.Module):
    """Capsule-network-lite: flat 784 → n_primary capsules of d_primary-D
    → n_classes digit capsules of d_out-D with dynamic routing-by-
    agreement (3 iterations). Length of each digit capsule ∝ class score.

    This is a simplification of Sabour et al. (2017): we skip the
    convolutional feature extractor and route directly from primary-caps
    (a single Linear layer reshaped into capsules) to digit-caps. The
    routing tensor W has shape (n_primary, n_classes, d_out, d_primary);
    spectral_weights() returns the primary Linear plus W flattened to 2-D
    so the entropy regulariser can hook in through the existing
    adjacency builder.

    For cross-entropy compatibility, forward() returns digit-capsule
    lengths scaled by 10; the scaling keeps the effective logits in a
    numerically reasonable range without changing argmax behaviour.
    """

    def __init__(self, in_dim: int = 784, n_primary: int = 32,
                 d_primary: int = 8, n_classes: int = 10,
                 d_out: int = 16, n_routing: int = 3,
                 apply_bn: bool = False, apply_dropout_p: float = 0.0):
        super().__init__()
        self.in_dim = in_dim
        self.n_primary = n_primary
        self.d_primary = d_primary
        self.n_classes = n_classes
        self.d_out = d_out
        self.n_routing = n_routing
        self.apply_bn = apply_bn
        self.apply_dropout_p = apply_dropout_p

        self.primary = nn.Linear(in_dim, n_primary * d_primary)
        # Routing transformation: (n_primary, n_classes, d_out, d_primary)
        # Glorot-style init scaled by 0.01 for stable early routing.
        self.W = nn.Parameter(
            0.01 * torch.randn(n_primary, n_classes, d_out, d_primary)
        )
        # Phase-9 optional BN on the primary linear output
        # (BatchNorm1d on the flattened pre-capsule features).
        if apply_bn:
            self.bn_primary = nn.BatchNorm1d(n_primary * d_primary)
        if apply_dropout_p > 0:
            self.drop = nn.Dropout(apply_dropout_p)

    @staticmethod
    def _squash(s, dim=-1):
        norm2 = (s ** 2).sum(dim=dim, keepdim=True)
        scale = norm2 / (1.0 + norm2)
        return scale * s / torch.sqrt(norm2 + 1e-8)

    def forward(self, x):
        B = x.size(0)
        x = x.view(B, -1)
        # Primary linear projection (optional BN before reshape).
        h = self.primary(x)
        if self.apply_bn:
            h = self.bn_primary(h)
        # Primary capsules: (B, n_primary, d_primary)
        u = h.view(B, self.n_primary, self.d_primary)
        u = self._squash(u, dim=-1)
        # Optional dropout on the primary capsule outputs (per-capsule mask).
        if self.apply_dropout_p > 0:
            u = self.drop(u)
        # Predicted output vectors u_hat: (B, n_primary, n_classes, d_out)
        u_hat = torch.einsum('bpd, pjod -> bpjo', u, self.W)
        # Dynamic routing
        b = torch.zeros(B, self.n_primary, self.n_classes, device=x.device)
        v = None
        for _ in range(self.n_routing):
            c = F.softmax(b, dim=2)
            s = (c.unsqueeze(-1) * u_hat).sum(dim=1)        # (B, n_classes, d_out)
            v = self._squash(s, dim=-1)
            agreement = (u_hat * v.unsqueeze(1)).sum(dim=-1)  # (B, n_primary, n_classes)
            b = b + agreement
        lengths = v.norm(dim=-1)                              # (B, n_classes)
        return lengths * 10.0                                 # scaled for CE loss

    def spectral_weights(self):
        # Re-lay the routing tensor W (n_primary, n_classes, d_out, d_primary)
        # as a single dataflow matrix mapping 256 primary-caps-neurons →
        # 160 digit-caps-neurons.
        #
        # Row index:  digit j × d_out component k  →  j*d_out + k
        # Col index:  primary p × d_primary comp l →  p*d_primary + l
        # Value: W[p, j, k, l]  (the per-pair 16×8 routing block)
        w_matrix = self.W.permute(1, 2, 0, 3).reshape(
            self.n_classes * self.d_out,
            self.n_primary * self.d_primary,
        )
        return [self.primary.weight, w_matrix]

    def spectral_adjacency(self, view: str = "dataflow"):
        W = self.spectral_weights()
        return build_adjacency_factor_view(W) if view == "factor" else build_adjacency(W)


class SVHNNet(nn.Module):
    """Compact MLP for SVHN: 3072 → 64 → 32 → 10 flat; fixed projection
    first layer (matching CIFAR-10 style — spectral regulariser skips
    the projection to keep eigendecomp cost manageable)."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(3072, 64)
        self.l0 = nn.Linear(64, 32)
        self.l1 = nn.Linear(32, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.proj(x))
        x = F.relu(self.l0(x))
        return self.l1(x)

    def spectral_weights(self):
        # Skip the 3072→64 projection (would dominate eigendecomp), in
        # line with CIFAR10Net's approach.
        return [self.l0.weight, self.l1.weight]

    def spectral_adjacency(self, view="dataflow"):
        W = self.spectral_weights()
        return build_adjacency_factor_view(W) if view == "factor" else build_adjacency(W)


class TabularMLP(nn.Module):
    """Parameterised MLP for small tabular datasets.
    input_dim → h0 → h1 → n_classes with ReLU.
    Used for iris (4→n=3), wine (13→n=3), breast_cancer (30→n=2),
    and digits (64→n=10)."""

    def __init__(self, input_dim: int, n_classes: int,
                 h0: int = 32, h1: int = 16):
        super().__init__()
        self.l0 = nn.Linear(input_dim, h0)
        self.l1 = nn.Linear(h0, h1)
        self.l2 = nn.Linear(h1, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.l0(x))
        x = F.relu(self.l1(x))
        return self.l2(x)

    def spectral_weights(self) -> list[torch.Tensor]:
        return [self.l0.weight, self.l1.weight, self.l2.weight]

    def spectral_adjacency(self, view: str = "dataflow") -> torch.Tensor:
        W = self.spectral_weights()
        return build_adjacency_factor_view(W) if view == "factor" else build_adjacency(W)


def _make_tabular_model(input_dim: int, n_classes: int):
    """Factory usable as the `model_class` slot in DATASETS."""
    def f():
        return TabularMLP(input_dim, n_classes)
    return f


class SyntheticMLP3(nn.Module):
    """2D → 32 → 16 → 3 MLP for Gaussian-quantiles (3-class)."""

    def __init__(self):
        super().__init__()
        self.l0 = nn.Linear(2, 32)
        self.l1 = nn.Linear(32, 16)
        self.l2 = nn.Linear(16, 3)

    def forward(self, x):
        x = F.relu(self.l0(x))
        x = F.relu(self.l1(x))
        return self.l2(x)

    def spectral_weights(self):
        return [self.l0.weight, self.l1.weight, self.l2.weight]

    def spectral_adjacency(self, view="dataflow"):
        W = self.spectral_weights()
        return build_adjacency_factor_view(W) if view == "factor" else build_adjacency(W)


class SyntheticMLP(nn.Module):
    """2D → 32 → 16 → 2 MLP for the Two Moons / Spirals / Circles
    benchmarks.

    Phase-9 BN/dropout switches via `apply_bn` and `apply_dropout_p`
    (defaults OFF — existing experiments byte-identical).
    Matches the thesis scale pattern (hidden 32, 16) but on a 2-dim
    input so training is seconds-per-seed and entropy regularization can
    be tested at very high seed counts."""

    def __init__(self, apply_bn: bool = False, apply_dropout_p: float = 0.0):
        super().__init__()
        self.l0 = nn.Linear(2, 32)
        self.l1 = nn.Linear(32, 16)
        self.l2 = nn.Linear(16, 2)
        self.apply_bn = apply_bn
        self.apply_dropout_p = apply_dropout_p
        if apply_bn:
            self.bn0 = nn.BatchNorm1d(32)
            self.bn1 = nn.BatchNorm1d(16)
        if apply_dropout_p > 0:
            self.drop = nn.Dropout(apply_dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.l0(x)
        if self.apply_bn:
            x = self.bn0(x)
        x = F.relu(x)
        if self.apply_dropout_p > 0:
            x = self.drop(x)
        x = self.l1(x)
        if self.apply_bn:
            x = self.bn1(x)
        x = F.relu(x)
        if self.apply_dropout_p > 0:
            x = self.drop(x)
        return self.l2(x)

    def spectral_weights(self) -> list[torch.Tensor]:
        return [self.l0.weight, self.l1.weight, self.l2.weight]

    def spectral_adjacency(self, view: str = "dataflow") -> torch.Tensor:
        W = self.spectral_weights()
        return build_adjacency_factor_view(W) if view == "factor" else build_adjacency(W)


class MNISTNet(nn.Module):
    """Flattened-pixel MLP for MNIST: 784 → 256 → 128 → 64 → 10.
    Total spectral-regularized neuron count: 1242."""

    def __init__(self):
        super().__init__()
        self.l0 = nn.Linear(784, 256)
        self.l1 = nn.Linear(256, 128)
        self.l2 = nn.Linear(128, 64)
        self.l3 = nn.Linear(64, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = F.relu(self.l0(x))
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.l3(x)

    def spectral_weights(self) -> list[torch.Tensor]:
        return [self.l0.weight, self.l1.weight, self.l2.weight, self.l3.weight]

    def spectral_adjacency(self, view: str = "dataflow") -> torch.Tensor:
        W = self.spectral_weights()
        return build_adjacency_factor_view(W) if view == "factor" else build_adjacency(W)


class MNISTNetSmall(nn.Module):
    """Thesis-scale MNIST MLP: 784 → 16 → 8 → 10.
    Total spectral-regularized neuron count: 818.
    Matches the thesis's 4→16→8→3 architecture pattern (hidden 16, 8).

    Optional `apply_bn` and `apply_dropout_p` switches add per-layer
    BatchNorm1d (post-linear, pre-activation) and Dropout (post-
    activation) respectively, used by phase-9 composability runs.
    Defaults are OFF so existing experiments are byte-identical."""

    def __init__(self, apply_bn: bool = False, apply_dropout_p: float = 0.0):
        super().__init__()
        self.l0 = nn.Linear(784, 16)
        self.l1 = nn.Linear(16, 8)
        self.l2 = nn.Linear(8, 10)
        self.apply_bn = apply_bn
        self.apply_dropout_p = apply_dropout_p
        if apply_bn:
            self.bn0 = nn.BatchNorm1d(16)
            self.bn1 = nn.BatchNorm1d(8)
        if apply_dropout_p > 0:
            self.drop = nn.Dropout(apply_dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = self.l0(x)
        if self.apply_bn:
            x = self.bn0(x)
        x = F.relu(x)
        if self.apply_dropout_p > 0:
            x = self.drop(x)
        x = self.l1(x)
        if self.apply_bn:
            x = self.bn1(x)
        x = F.relu(x)
        if self.apply_dropout_p > 0:
            x = self.drop(x)
        return self.l2(x)

    def spectral_weights(self) -> list[torch.Tensor]:
        return [self.l0.weight, self.l1.weight, self.l2.weight]

    def spectral_adjacency(self, view: str = "dataflow") -> torch.Tensor:
        W = self.spectral_weights()
        return build_adjacency_factor_view(W) if view == "factor" else build_adjacency(W)


class MNISTScaled(nn.Module):
    """MNIST MLP with parameterized hidden widths.

    Architecture: 784 → (h0) → (h1) → 10 with ReLU activations.
    Used by the architecture-scale sweep to characterize how the
    spectral-entropy regularizer's effect varies with network size.

    The thesis scale is (h0=16, h1=8) → 818 spectral neurons total.
    Doubling hidden widths grows the hidden subgraph quadratically
    while the input (784) stays fixed; the resulting neuron count
    and spectrum dimension both scale."""

    def __init__(self, h0: int, h1: int):
        super().__init__()
        self.h0, self.h1 = h0, h1
        self.l0 = nn.Linear(784, h0)
        self.l1 = nn.Linear(h0, h1)
        self.l2 = nn.Linear(h1, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = F.relu(self.l0(x))
        x = F.relu(self.l1(x))
        return self.l2(x)

    def spectral_weights(self) -> list[torch.Tensor]:
        return [self.l0.weight, self.l1.weight, self.l2.weight]

    def spectral_adjacency(self, view: str = "dataflow") -> torch.Tensor:
        W = self.spectral_weights()
        return build_adjacency_factor_view(W) if view == "factor" else build_adjacency(W)


def make_scaled_mnist_factory(h0: int, h1: int):
    """Create a factory for MNISTScaled at given widths, usable as a
    drop-in replacement for the class in DATASETS."""
    def factory():
        return MNISTScaled(h0, h1)
    factory.h0, factory.h1 = h0, h1  # annotate for introspection
    return factory


class ResMLP(nn.Module):
    """ResNet-style MLP for MNIST with a fixed projection + K residual
    blocks at hidden dim H. Each block: y = x + F(x) where F = Linear(H,H)
    + ReLU + Linear(H,H). A final Linear(H → 10) projects to classes.

    The spectral adjacency includes both the learned F weights *and*
    the identity skip connection (with magnitude 1) from the block's
    input to its output. This reflects the true information pathway:
    an entropy penalty on this adjacency sees skip-dominated blocks
    (where F is small relative to identity) as having strong single-
    path structure, naturally."""

    def __init__(self, hidden: int = 16, n_blocks: int = 3):
        super().__init__()
        self.hidden = hidden
        self.n_blocks = n_blocks
        self.proj = nn.Linear(784, hidden)
        self.block_weights = nn.ModuleList([
            nn.ModuleDict({
                "f1": nn.Linear(hidden, hidden),
                "f2": nn.Linear(hidden, hidden),
            })
            for _ in range(n_blocks)
        ])
        self.out = nn.Linear(hidden, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = F.relu(self.proj(x))
        for block in self.block_weights:
            h = F.relu(block["f1"](x))
            x = x + block["f2"](h)  # residual
        return self.out(x)

    def spectral_adjacency(self, view: str = "dataflow") -> torch.Tensor:
        """Build the adjacency for the spectral-regularized section
        (skipping the proj layer for fair comparison with MNISTScaled).

        Factor view on ResMLP adds within-layer clique contributions
        from the Linear blocks' `f1`, `f2`, and output weights — skip
        connections (identity) contribute no within-layer edges
        (1-to-1 mapping has no clique).

        Layout of neurons in order: proj_out (H) → block_0_mid (H) →
        block_0_out (H) → block_1_mid (H) → block_1_out (H) → ... →
        final_out (10). Each block contributes two weight blocks
        (f1, f2) and an identity skip (magnitude 1) from its input
        to its output.
        """
        device = self.proj.weight.device
        dtype = self.proj.weight.dtype
        H = self.hidden
        # Layer sizes: [H, H, H, H, H, ..., H, 10]
        # Where block k has mid (H) and out (H).
        layer_sizes = [H] + [H] * (2 * self.n_blocks) + [10]
        total = sum(layer_sizes)
        offsets = [0]
        for s in layer_sizes:
            offsets.append(offsets[-1] + s)

        A = torch.zeros(total, total, device=device, dtype=dtype)

        # For each block: f1 (block input → mid), f2 (mid → block output),
        # plus identity skip (block input → block output, weight 1).
        for k, block in enumerate(self.block_weights):
            in_idx = 2 * k       # block input is output of previous block (or proj)
            mid_idx = 2 * k + 1  # mid is the ReLU after f1
            out_idx = 2 * k + 2  # block output is after residual add

            # f1: in → mid
            absW1 = block["f1"].weight.abs()
            W1 = absW1.t()
            r0, r1 = offsets[in_idx], offsets[in_idx + 1]
            c0, c1 = offsets[mid_idx], offsets[mid_idx + 1]
            A[r0:r1, c0:c1] += W1
            A[c0:c1, r0:r1] += W1.t()
            if view == "factor":
                # Within-layer clique on f1's inputs.
                w = absW1.t() @ absW1
                w = w - torch.diag(torch.diag(w))
                A[r0:r1, r0:r1] += w

            # f2: mid → out
            absW2 = block["f2"].weight.abs()
            W2 = absW2.t()
            r0, r1 = offsets[mid_idx], offsets[mid_idx + 1]
            c0, c1 = offsets[out_idx], offsets[out_idx + 1]
            A[r0:r1, c0:c1] += W2
            A[c0:c1, r0:r1] += W2.t()
            if view == "factor":
                # Within-layer clique on f2's inputs (mid neurons).
                w = absW2.t() @ absW2
                w = w - torch.diag(torch.diag(w))
                A[r0:r1, r0:r1] += w

            # Identity skip: in → out, weight 1 per matched neuron.
            # Skip is 1-to-1 so no clique contribution in factor view.
            I = torch.eye(H, device=device, dtype=dtype)
            r0, r1 = offsets[in_idx], offsets[in_idx + 1]
            c0, c1 = offsets[out_idx], offsets[out_idx + 1]
            A[r0:r1, c0:c1] += I
            A[c0:c1, r0:r1] += I.t()

        # Final output layer: last block output → 10 classes.
        last_block_out = 2 * self.n_blocks
        final = 2 * self.n_blocks + 1
        absWout = self.out.weight.abs()
        Wout = absWout.t()
        r0, r1 = offsets[last_block_out], offsets[last_block_out + 1]
        c0, c1 = offsets[final], offsets[final + 1]
        A[r0:r1, c0:c1] += Wout
        A[c0:c1, r0:r1] += Wout.t()
        if view == "factor":
            w = absWout.t() @ absWout
            w = w - torch.diag(torch.diag(w))
            A[r0:r1, r0:r1] += w

        return A

    def spectral_weights(self) -> list[torch.Tensor]:
        """For the effective-rank metrics. Only the learned weights —
        skip connections don't have trainable parameters to measure."""
        out = []
        for block in self.block_weights:
            out.append(block["f1"].weight)
            out.append(block["f2"].weight)
        out.append(self.out.weight)
        return out


class HighwayMLP(nn.Module):
    """Highway-network-style MLP: each block has a learned gate T(x)
    modulating a residual path. y = T(x) * H(x) + (1 - T(x)) * x.

    The spectral adjacency includes the skip with a weight equal to
    the running mean of (1 - T(x)) measured at construction time —
    i.e., the effective skip strength at current parameters. Since
    T is input-dependent we use its initialization-time expected
    value (gate weights start so that mean(T) ≈ 0.5 after sigmoid)."""

    def __init__(self, hidden: int = 16, n_blocks: int = 3):
        super().__init__()
        self.hidden = hidden
        self.n_blocks = n_blocks
        self.proj = nn.Linear(784, hidden)
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "h": nn.Linear(hidden, hidden),  # transform
                "t": nn.Linear(hidden, hidden),  # gate
            })
            for _ in range(n_blocks)
        ])
        # Highway convention: bias gate toward skip at init so training
        # can open the transform gradually.
        for block in self.blocks:
            nn.init.constant_(block["t"].bias, -2.0)  # sigmoid(-2) ≈ 0.12
        self.out = nn.Linear(hidden, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = F.relu(self.proj(x))
        for block in self.blocks:
            T = torch.sigmoid(block["t"](x))
            H = F.relu(block["h"](x))
            x = T * H + (1.0 - T) * x
        return self.out(x)

    def spectral_adjacency(self, view: str = "dataflow") -> torch.Tensor:
        """Build adjacency weighted by the expected gate value.

        At initialization `sigmoid(bias=-2) ≈ 0.12`, so skip weight is
        (1 - 0.12) = 0.88 and transform weight is 0.12. We use these
        as proxies; a more faithful version would observe the mean
        gate value from a forward pass on the actual data — follow-up
        if needed. Factor view adds within-layer clique contributions
        from the transform weights (scaled by gate_mean²)."""
        device = self.proj.weight.device
        dtype = self.proj.weight.dtype
        H = self.hidden
        layer_sizes = [H] * (self.n_blocks + 1) + [10]  # proj_out → block_0 → ... → final_block_out → 10
        total = sum(layer_sizes)
        offsets = [0]
        for s in layer_sizes:
            offsets.append(offsets[-1] + s)

        A = torch.zeros(total, total, device=device, dtype=dtype)
        eye = torch.eye(H, device=device, dtype=dtype)

        # Approximate gate expectation — compute from gate biases alone.
        for k, block in enumerate(self.blocks):
            gate_mean = torch.sigmoid(block["t"].bias.mean()).item()
            transform_scale = gate_mean
            skip_scale = 1.0 - gate_mean

            absWh = block["h"].weight.abs()
            Wh = absWh.t()
            r0, r1 = offsets[k], offsets[k + 1]
            c0, c1 = offsets[k + 1], offsets[k + 2]
            A[r0:r1, c0:c1] += transform_scale * Wh
            A[c0:c1, r0:r1] += transform_scale * Wh.t()
            if view == "factor":
                # Within-layer clique from transform factor, scaled by
                # gate² (since each pair co-appears in the factor with
                # scaled incidence).
                w = (transform_scale ** 2) * (absWh.t() @ absWh)
                w = w - torch.diag(torch.diag(w))
                A[r0:r1, r0:r1] += w

            # Skip with scale (1 - gate_mean)
            A[r0:r1, c0:c1] += skip_scale * eye
            A[c0:c1, r0:r1] += skip_scale * eye.t()

        absWout = self.out.weight.abs()
        Wout = absWout.t()
        last_block_out = self.n_blocks
        final = self.n_blocks + 1
        r0, r1 = offsets[last_block_out], offsets[last_block_out + 1]
        c0, c1 = offsets[final], offsets[final + 1]
        A[r0:r1, c0:c1] += Wout
        A[c0:c1, r0:r1] += Wout.t()
        if view == "factor":
            w = absWout.t() @ absWout
            w = w - torch.diag(torch.diag(w))
            A[r0:r1, r0:r1] += w

        return A

    def spectral_weights(self) -> list[torch.Tensor]:
        """For effective-rank metrics — learned weights only."""
        out = []
        for block in self.blocks:
            out.append(block["h"].weight)
            out.append(block["t"].weight)
        out.append(self.out.weight)
        return out


class CIFAR10Net(nn.Module):
    """Compact MLP for CIFAR-10 with a fixed flatten + projection first
    layer so the spectral-regularized section stays small. The thesis's
    formalism applies to any connected MLP; we regularize only the
    trainable-spectral portion (128 → 64 → 32 → 10).

    The first linear `proj` (3072 → 128) is trained normally but its
    weights are *not* fed into the spectral regularizer — the 3072-dim
    adjacency would dominate eigendecomp cost. This is a practical
    compromise, documented honestly in RESULTS.md."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(3072, 128)  # not spectrally regularized
        self.l0 = nn.Linear(128, 64)
        self.l1 = nn.Linear(64, 32)
        self.l2 = nn.Linear(32, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = F.relu(self.proj(x))
        x = F.relu(self.l0(x))
        x = F.relu(self.l1(x))
        return self.l2(x)

    def spectral_weights(self) -> list[torch.Tensor]:
        return [self.l0.weight, self.l1.weight, self.l2.weight]

    def spectral_adjacency(self, view: str = "dataflow") -> torch.Tensor:
        W = self.spectral_weights()
        return build_adjacency_factor_view(W) if view == "factor" else build_adjacency(W)


# ─── Datasets ───────────────────────────────────────────────────────


# Benchmark datasets live outside the HyMeKo `data/` tree (which is
# reserved for HyMeKo example/user content). Downloaded image datasets
# go under `datasets/torchvision/`; KMNIST (HF parquet mirror) under
# `datasets/kmnist/`.
DATA_ROOT = Path("datasets/torchvision")
KMNIST_ROOT = Path("datasets/kmnist")


def mnist_loaders(seed: int, batch_size: int = 128) -> tuple[DataLoader, DataLoader]:
    from torchvision import datasets, transforms
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    tr = datasets.MNIST(DATA_ROOT, train=True, download=True, transform=tfm)
    va = datasets.MNIST(DATA_ROOT, train=False, download=True, transform=tfm)
    g = torch.Generator().manual_seed(seed)
    tr_loader = DataLoader(tr, batch_size=batch_size, shuffle=True, generator=g, num_workers=2)
    va_loader = DataLoader(va, batch_size=512, shuffle=False, num_workers=2)
    return tr_loader, va_loader


def fashion_mnist_loaders(seed: int, batch_size: int = 128) -> tuple[DataLoader, DataLoader]:
    """FashionMNIST — 28×28 grayscale, 10 classes, MNIST-compatible shape.
    Sibling-dataset check: does the MNIST plain-MLP entropy effect survive
    on a visually different 784-dim input?"""
    from torchvision import datasets, transforms
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])
    tr = datasets.FashionMNIST(DATA_ROOT, train=True, download=True, transform=tfm)
    va = datasets.FashionMNIST(DATA_ROOT, train=False, download=True, transform=tfm)
    g = torch.Generator().manual_seed(seed)
    tr_loader = DataLoader(tr, batch_size=batch_size, shuffle=True, generator=g, num_workers=2)
    va_loader = DataLoader(va, batch_size=512, shuffle=False, num_workers=2)
    return tr_loader, va_loader


# KMNIST: torchvision's bundled URL (codh.rois.ac.jp) is dead as of
# 2026-04. We load from the HuggingFace mirror tanganke/kmnist (public
# parquets at data/kmnist/) instead, decode PNG bytes once, and cache
# decoded uint8 arrays to data/kmnist/kmnist_decoded.npz so subsequent
# seeds load instantly.
_KMNIST_ARRAYS: tuple | None = None


def _load_kmnist_arrays() -> tuple:
    """Return (train_x, train_y, test_x, test_y) as numpy uint8/int64
    arrays. Decodes parquet on first call, then caches .npz and an
    in-process memo."""
    global _KMNIST_ARRAYS
    if _KMNIST_ARRAYS is not None:
        return _KMNIST_ARRAYS

    import numpy as np

    cache = KMNIST_ROOT / "kmnist_decoded.npz"
    if cache.exists():
        z = np.load(cache)
        _KMNIST_ARRAYS = (z["train_x"], z["train_y"], z["test_x"], z["test_y"])
        return _KMNIST_ARRAYS

    # First-time decode from parquet.
    import io
    import pandas as pd
    from PIL import Image

    parquet_dir = KMNIST_ROOT
    train_pq = parquet_dir / "train-00000-of-00001.parquet"
    test_pq  = parquet_dir / "test-00000-of-00001.parquet"
    if not (train_pq.exists() and test_pq.exists()):
        raise RuntimeError(
            f"KMNIST parquet files not found in {parquet_dir}. "
            "Fetch them from https://huggingface.co/datasets/tanganke/kmnist "
            "into data/kmnist/ before running KMNIST experiments."
        )

    def _decode_df(df) -> tuple:
        n = len(df)
        x = np.empty((n, 28, 28), dtype=np.uint8)
        y = np.empty(n, dtype=np.int64)
        for i, (img_cell, label) in enumerate(zip(df["image"], df["label"])):
            buf = img_cell["bytes"] if isinstance(img_cell, dict) else img_cell
            x[i] = np.array(Image.open(io.BytesIO(buf)), dtype=np.uint8)
            y[i] = int(label)
        return x, y

    print(f"  decoding {train_pq.name} (first-time cache build) …", flush=True)
    train_x, train_y = _decode_df(pd.read_parquet(train_pq))
    print(f"  decoding {test_pq.name} …", flush=True)
    test_x, test_y = _decode_df(pd.read_parquet(test_pq))

    np.savez_compressed(cache, train_x=train_x, train_y=train_y,
                        test_x=test_x, test_y=test_y)
    _KMNIST_ARRAYS = (train_x, train_y, test_x, test_y)
    return _KMNIST_ARRAYS


def emnist_letters_loaders(seed: int, batch_size: int = 128) -> tuple[DataLoader, DataLoader]:
    """EMNIST Letters — 26-class handwritten letter classification.
    Labels are 1..26 in the raw dataset; we subtract 1 to get 0..25.
    Matches MNIST's 28×28 shape, so MNIST models drop in by overriding
    the final linear's out_features."""
    from torchvision import datasets, transforms

    class _ShiftLabel:
        """EMNIST Letters labels are 1..26; shift to 0..25 for CE loss."""
        def __call__(self, y):
            return int(y) - 1

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    tr = datasets.EMNIST(DATA_ROOT, split="letters", train=True,
                         download=True, transform=tfm,
                         target_transform=_ShiftLabel())
    va = datasets.EMNIST(DATA_ROOT, split="letters", train=False,
                         download=True, transform=tfm,
                         target_transform=_ShiftLabel())
    g = torch.Generator().manual_seed(seed)
    return (DataLoader(tr, batch_size=batch_size, shuffle=True, generator=g, num_workers=2),
            DataLoader(va, batch_size=512, shuffle=False, num_workers=2))


def svhn_loaders(seed: int, batch_size: int = 128) -> tuple[DataLoader, DataLoader]:
    """SVHN — Street View House Numbers, 32×32 color, 10 classes.
    Natural sibling to CIFAR-10 but easier (crisper class boundaries,
    digit-typing structure); useful for a second 3072-dim image test."""
    from torchvision import datasets, transforms
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4377, 0.4438, 0.4728),
                             (0.1980, 0.2010, 0.1970)),
    ])
    tr = datasets.SVHN(DATA_ROOT, split="train", download=True, transform=tfm)
    va = datasets.SVHN(DATA_ROOT, split="test",  download=True, transform=tfm)
    g = torch.Generator().manual_seed(seed)
    return (DataLoader(tr, batch_size=batch_size, shuffle=True, generator=g, num_workers=2),
            DataLoader(va, batch_size=512, shuffle=False, num_workers=2))


def kmnist_loaders(seed: int, batch_size: int = 128) -> tuple[DataLoader, DataLoader]:
    """KMNIST — Japanese hiragana, 28×28 grayscale, 10 classes.
    Second MNIST-style sibling: different script family, different pixel
    statistics (higher stroke density than Latin digits).

    Backed by the HF `tanganke/kmnist` parquet mirror — the original
    codh.rois.ac.jp host that torchvision's built-in loader points at is
    dead. The loader decodes parquets once, caches the decoded arrays
    as an .npz, and hands out channel-first tensors shaped (N, 1, 28, 28)
    to match the torchvision MNIST/FashionMNIST convention used by
    MNISTNet/MNISTNetSmall."""
    train_x, train_y, test_x, test_y = _load_kmnist_arrays()

    mean, std = 0.1918, 0.3483  # canonical KMNIST stats
    def _to_tensor(x_u8, y_i64):
        x = torch.from_numpy(x_u8).float().div_(255.0).sub_(mean).div_(std)
        x = x.unsqueeze(1)  # (N, 1, 28, 28) — matches torchvision ToTensor
        return x, torch.from_numpy(y_i64).long()

    x_tr, y_tr = _to_tensor(train_x, train_y)
    x_va, y_va = _to_tensor(test_x,  test_y)

    g = torch.Generator().manual_seed(seed)
    tr_loader = DataLoader(TensorDataset(x_tr, y_tr),
                           batch_size=batch_size, shuffle=True, generator=g)
    va_loader = DataLoader(TensorDataset(x_va, y_va),
                           batch_size=512, shuffle=False)
    return tr_loader, va_loader


# ------------------------------------------------------------------- #
# Synthetic 2D classification datasets                                 #
# ------------------------------------------------------------------- #

def _make_moons(n: int, noise: float, rng) -> tuple["np.ndarray", "np.ndarray"]:
    import numpy as np
    n_out = n // 2
    n_in = n - n_out
    outer_theta = np.linspace(0, np.pi, n_out)
    outer = np.stack([np.cos(outer_theta), np.sin(outer_theta)], axis=1)
    inner_theta = np.linspace(0, np.pi, n_in)
    inner = np.stack([1.0 - np.cos(inner_theta), 0.5 - np.sin(inner_theta)], axis=1)
    X = np.vstack([outer, inner]).astype(np.float32)
    y = np.hstack([np.zeros(n_out), np.ones(n_in)]).astype(np.int64)
    X += rng.normal(0.0, noise, X.shape).astype(np.float32)
    return X, y


def _make_spirals(n: int, noise: float, rng) -> tuple["np.ndarray", "np.ndarray"]:
    import numpy as np
    n_per = n // 2
    theta = np.sqrt(rng.uniform(0.0, 1.0, n_per)) * 2.0 * np.pi
    r = 2.0 * theta + np.pi
    d1 = np.stack([-np.cos(theta) * r,  np.sin(theta) * r], axis=1)
    d2 = np.stack([ np.cos(theta) * r, -np.sin(theta) * r], axis=1)
    X = np.vstack([d1, d2]).astype(np.float32)
    y = np.hstack([np.zeros(n_per), np.ones(n_per)]).astype(np.int64)
    X += rng.normal(0.0, noise, X.shape).astype(np.float32)
    X /= 20.0  # into roughly [-1, 1]
    return X, y


def _make_gaussian_quantiles(n: int, noise: float, rng) -> tuple["np.ndarray", "np.ndarray"]:
    """3-class 2-D Gaussian quantiles — samples drawn from a single 2-D
    Gaussian and partitioned by radius into 3 concentric classes.
    Noise parameter here controls the covariance scale."""
    import numpy as np
    from sklearn.datasets import make_gaussian_quantiles as _gq
    X, y = _gq(n_samples=n, n_features=2, n_classes=3,
               cov=max(noise, 0.01), random_state=int(rng.integers(0, 2**31 - 1)))
    return X.astype(np.float32), y.astype(np.int64)


def _make_circles(n: int, noise: float, rng) -> tuple["np.ndarray", "np.ndarray"]:
    """Two concentric circles — topologically distinct from moons/spirals:
    the decision boundary is a closed curve. Inner radius 0.5, outer 1.0."""
    import numpy as np
    n_out = n // 2
    n_in  = n - n_out
    theta_out = rng.uniform(0.0, 2.0 * np.pi, n_out)
    theta_in  = rng.uniform(0.0, 2.0 * np.pi, n_in)
    outer = np.stack([np.cos(theta_out), np.sin(theta_out)], axis=1)
    inner = 0.5 * np.stack([np.cos(theta_in), np.sin(theta_in)], axis=1)
    X = np.vstack([outer, inner]).astype(np.float32)
    y = np.hstack([np.zeros(n_out), np.ones(n_in)]).astype(np.int64)
    X += rng.normal(0.0, noise, X.shape).astype(np.float32)
    return X, y


# ------------------------------------------------------------------- #
# sklearn tabular datasets — iris, wine, breast_cancer, digits         #
# ------------------------------------------------------------------- #

_SKLEARN_CACHE: dict = {}


def _sklearn_arrays(kind: str) -> tuple:
    """Return (X_train, y_train, X_val, y_val) standardized on the train
    split. Caches per-process."""
    if kind in _SKLEARN_CACHE:
        return _SKLEARN_CACHE[kind]
    import numpy as np
    from sklearn import datasets as sk_ds
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    loaders = {
        "iris":          sk_ds.load_iris,
        "wine":          sk_ds.load_wine,
        "breast_cancer": sk_ds.load_breast_cancer,
        "digits":        sk_ds.load_digits,
    }
    d = loaders[kind]()
    X, y = d.data.astype(np.float32), d.target.astype(np.int64)
    # Fixed split per dataset (same for every seed) — seed-level variation
    # comes from weight init + shuffler, not from split shuffling.
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y,
    )
    scaler = StandardScaler().fit(X_tr)
    X_tr = scaler.transform(X_tr).astype(np.float32)
    X_va = scaler.transform(X_va).astype(np.float32)
    _SKLEARN_CACHE[kind] = (X_tr, y_tr, X_va, y_va)
    return _SKLEARN_CACHE[kind]


def _tabular_loaders_factory(kind: str) -> Callable:
    """Build DataLoader factory for a sklearn tabular dataset."""
    def loaders(seed: int, batch_size: int = 64) -> tuple[DataLoader, DataLoader]:
        X_tr, y_tr, X_va, y_va = _sklearn_arrays(kind)
        ds_tr = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
        ds_va = TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va))
        g = torch.Generator().manual_seed(seed)
        return (
            DataLoader(ds_tr, batch_size=batch_size, shuffle=True, generator=g),
            DataLoader(ds_va, batch_size=256, shuffle=False),
        )
    return loaders


def _synthetic_loaders_factory(kind: str, n_train: int = 2000, n_val: int = 500,
                                noise: float = 0.2) -> Callable:
    """Build loader factory for a 2D synthetic classification dataset.
    Train and val splits use disjoint RNG streams derived from `seed`
    so per-seed variation is captured while splits remain independent."""
    gen = {
        "moons":           _make_moons,
        "spirals":         _make_spirals,
        "circles":         _make_circles,
        "gaussian_quants": _make_gaussian_quantiles,
    }[kind]

    def loaders(seed: int, batch_size: int = 128) -> tuple[DataLoader, DataLoader]:
        import numpy as np
        rng_tr = np.random.default_rng(seed * 2 + 0)
        rng_va = np.random.default_rng(seed * 2 + 1)
        X_tr, y_tr = gen(n_train, noise, rng_tr)
        X_va, y_va = gen(n_val,   noise, rng_va)
        ds_tr = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
        ds_va = TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va))
        g = torch.Generator().manual_seed(seed)
        return (
            DataLoader(ds_tr, batch_size=batch_size, shuffle=True, generator=g),
            DataLoader(ds_va, batch_size=256, shuffle=False),
        )

    return loaders


def cifar_loaders(seed: int, batch_size: int = 128) -> tuple[DataLoader, DataLoader]:
    from torchvision import datasets, transforms
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    tr = datasets.CIFAR10(DATA_ROOT, train=True, download=True, transform=tfm)
    va = datasets.CIFAR10(DATA_ROOT, train=False, download=True, transform=tfm)
    g = torch.Generator().manual_seed(seed)
    tr_loader = DataLoader(tr, batch_size=batch_size, shuffle=True, generator=g, num_workers=2)
    va_loader = DataLoader(va, batch_size=512, shuffle=False, num_workers=2)
    return tr_loader, va_loader


DATASETS: dict[str, tuple[Callable, Callable]] = {
    "mnist": (mnist_loaders, MNISTNet),
    "mnist_small": (mnist_loaders, MNISTNetSmall),
    # Sibling MNIST-shape datasets: reuse MNISTNetSmall so they're
    # directly comparable to the 33-seed plain-MLP baseline.
    "fashion_mnist": (fashion_mnist_loaders, MNISTNetSmall),
    "kmnist":        (kmnist_loaders,        MNISTNetSmall),
    # Synthetic 2D classification — seconds per seed, cheap at 100+ seeds.
    "two_moons": (_synthetic_loaders_factory("moons"),   SyntheticMLP),
    "spirals":   (_synthetic_loaders_factory("spirals"), SyntheticMLP),
    "circles":   (_synthetic_loaders_factory("circles"), SyntheticMLP),
    # 3-class 2-D Gaussian quantiles — multi-class synthetic.
    "gaussian_quants": (_synthetic_loaders_factory("gaussian_quants", noise=0.5),
                         SyntheticMLP3),
    # Image siblings/extensions
    "emnist_letters":  (emnist_letters_loaders, _make_mnist_nclass(26)),
    "svhn":            (svhn_loaders, SVHNNet),
    # CapsMLP on MNIST / FashionMNIST — new architecture class
    "mnist_capsnet":         (mnist_loaders,         lambda: CapsMLP(n_classes=10)),
    "fashion_mnist_capsnet": (fashion_mnist_loaders, lambda: CapsMLP(n_classes=10)),
    # ─── Phase 9 architectural variants (BN / dropout / both) ───
    # Used by the composability study against scalar_entropy_normalized.
    "mnist_small_drop": (mnist_loaders,
                         lambda: MNISTNetSmall(apply_dropout_p=0.5)),
    "mnist_small_bn":   (mnist_loaders,
                         lambda: MNISTNetSmall(apply_bn=True)),
    "mnist_small_full": (mnist_loaders,
                         lambda: MNISTNetSmall(apply_bn=True, apply_dropout_p=0.5)),
    "spirals_drop":     (_synthetic_loaders_factory("spirals"),
                         lambda: SyntheticMLP(apply_dropout_p=0.5)),
    "spirals_bn":       (_synthetic_loaders_factory("spirals"),
                         lambda: SyntheticMLP(apply_bn=True)),
    "spirals_full":     (_synthetic_loaders_factory("spirals"),
                         lambda: SyntheticMLP(apply_bn=True, apply_dropout_p=0.5)),
    "circles_drop":     (_synthetic_loaders_factory("circles"),
                         lambda: SyntheticMLP(apply_dropout_p=0.5)),
    "circles_bn":       (_synthetic_loaders_factory("circles"),
                         lambda: SyntheticMLP(apply_bn=True)),
    "circles_full":     (_synthetic_loaders_factory("circles"),
                         lambda: SyntheticMLP(apply_bn=True, apply_dropout_p=0.5)),
    "mnist_capsnet_drop": (mnist_loaders,
                           lambda: CapsMLP(n_classes=10, apply_dropout_p=0.5)),
    "mnist_capsnet_bn":   (mnist_loaders,
                           lambda: CapsMLP(n_classes=10, apply_bn=True)),
    "mnist_capsnet_full": (mnist_loaders,
                           lambda: CapsMLP(n_classes=10, apply_bn=True, apply_dropout_p=0.5)),
    # Tabular sklearn datasets — TabularMLP is (input_dim → 32 → 16 → n_classes)
    "iris":          (_tabular_loaders_factory("iris"),          _make_tabular_model(4,  3)),
    "wine":          (_tabular_loaders_factory("wine"),          _make_tabular_model(13, 3)),
    "breast_cancer": (_tabular_loaders_factory("breast_cancer"), _make_tabular_model(30, 2)),
    "digits":        (_tabular_loaders_factory("digits"),        _make_tabular_model(64, 10)),
    # Highway MLP on FashionMNIST — sibling-dataset depth check
    "fashion_mnist_highway_10": (fashion_mnist_loaders, lambda: HighwayMLP(hidden=16, n_blocks=10)),
    "fashion_mnist_highway_20": (fashion_mnist_loaders, lambda: HighwayMLP(hidden=16, n_blocks=20)),
    "fashion_mnist_resnet_20":  (fashion_mnist_loaders, lambda: ResMLP(hidden=16, n_blocks=20)),
    # Depth sweep variants for ResMLP + Highway — same width, more blocks.
    # The 3-block versions are the original "toy" tests; 10 and 20 are
    # depths where real ResNets start living.
    "mnist_resnet": (mnist_loaders, lambda: ResMLP(hidden=16, n_blocks=3)),
    "mnist_resnet_10": (mnist_loaders, lambda: ResMLP(hidden=16, n_blocks=10)),
    "mnist_resnet_20": (mnist_loaders, lambda: ResMLP(hidden=16, n_blocks=20)),
    "mnist_resnet_40": (mnist_loaders, lambda: ResMLP(hidden=16, n_blocks=40)),
    "mnist_highway": (mnist_loaders, lambda: HighwayMLP(hidden=16, n_blocks=3)),
    "mnist_highway_10": (mnist_loaders, lambda: HighwayMLP(hidden=16, n_blocks=10)),
    "mnist_highway_20": (mnist_loaders, lambda: HighwayMLP(hidden=16, n_blocks=20)),
    "cifar10": (cifar_loaders, CIFAR10Net),
    "cifar10_small": (cifar_loaders, lambda: CIFAR10ResMLP(hidden=16, n_blocks=1)),
    "cifar10_resnet_3": (cifar_loaders, lambda: CIFAR10ResMLP(hidden=16, n_blocks=3)),
    "cifar10_resnet_10": (cifar_loaders, lambda: CIFAR10ResMLP(hidden=16, n_blocks=10)),
    "cifar10_resnet_20": (cifar_loaders, lambda: CIFAR10ResMLP(hidden=16, n_blocks=20)),
}


# ─── HyMeKo-generated networks (drop-in replacements via the
# torch_dataflow backend; see data/nn/*.hymeko sources and
# python/benches/thesis_iv_hard/generated_nets/ for the auto-emitted
# Python). The emitted classes already expose `spectral_weights()` via
# the patched torch_dataflow template; we add the matching
# `spectral_adjacency(view)` here so they slot into the bench harness
# without further modification. The registered dataset entries
# `*_hymeko` reuse the `mnist_loaders` data pipeline (the architectures
# are MNIST-shaped with 784→hidden→10 dataflow).
def _wrap_hymeko_net(emitted_cls):
    """Add spectral_adjacency(view) and an MNIST-shape flatten to a
    HyMeKo-emitted nn.Module class. The HyMeKo .hymeko sources declare
    input shape [784], so the emitted forward expects a flat (B, 784)
    tensor; mnist_loaders returns (B, 28, 28) unflattened. We monkey-
    patch forward to flatten on entry, preserving the canonical IR
    semantics. The dataflow / factor adjacency builders are the same
    module-level functions used by the hand-written ResMLP /
    HighwayMLP, so the spectral observables are fully comparable
    between the two paths."""
    def spectral_adjacency(self, view: str = "dataflow"):
        W = self.spectral_weights()
        return build_adjacency_factor_view(W) if view == "factor" else build_adjacency(W)
    emitted_cls.spectral_adjacency = spectral_adjacency

    _orig_forward = emitted_cls.forward
    def forward(self, x):
        if x.dim() > 2:
            x = x.flatten(start_dim=1)
        return _orig_forward(self, x)
    emitted_cls.forward = forward
    return emitted_cls


import os as _os
_HYMEKO_GEN_DIR = _os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), "generated_nets"
)
if _os.path.isdir(_HYMEKO_GEN_DIR):
    import sys as _sys
    if _HYMEKO_GEN_DIR not in _sys.path:
        _sys.path.insert(0, _HYMEKO_GEN_DIR)
    try:
        import mnist_highway_3   as _hg3
        import mnist_highway_10  as _hg10
        import mnist_highway_20  as _hg20
        import mnist_resmlp_10   as _rm10
        import mnist_resmlp_20   as _rm20
        import mnist_resmlp_40   as _rm40
        DATASETS["mnist_highway_3_hymeko"]  = (mnist_loaders,
            _wrap_hymeko_net(_hg3.HighwayMLP3FromHymeko))
        DATASETS["mnist_highway_10_hymeko"] = (mnist_loaders,
            _wrap_hymeko_net(_hg10.HighwayMLP10FromHymeko))
        DATASETS["mnist_highway_20_hymeko"] = (mnist_loaders,
            _wrap_hymeko_net(_hg20.HighwayMLP20FromHymeko))
        DATASETS["mnist_resmlp_10_hymeko"]  = (mnist_loaders,
            _wrap_hymeko_net(_rm10.ResMLP10FromHymeko))
        DATASETS["mnist_resmlp_20_hymeko"]  = (mnist_loaders,
            _wrap_hymeko_net(_rm20.ResMLP20FromHymeko))
        DATASETS["mnist_resmlp_40_hymeko"]  = (mnist_loaders,
            _wrap_hymeko_net(_rm40.ResMLP40FromHymeko))
    except Exception as _e:
        print(f"  [WARN] HyMeKo-generated nets not loaded: {_e}", flush=True)


class CIFAR10ResMLP(nn.Module):
    """CIFAR-10 ResNet-style MLP: fixed projection 3072 → 16, then N
    residual blocks at hidden=16, then linear head → 10. Like ResMLP
    but with CIFAR's 3072-dim input flattened and projected.

    Spectral adjacency omits the 3072→16 projection (same reasoning as
    CIFAR10Net — the projection would dominate eigendecomp cost).
    When `n_blocks=1` this behaves like a tiny MLP with a single
    residual refinement."""

    def __init__(self, hidden: int = 16, n_blocks: int = 3):
        super().__init__()
        self.hidden = hidden
        self.n_blocks = n_blocks
        self.proj = nn.Linear(3072, hidden)
        self.block_weights = nn.ModuleList([
            nn.ModuleDict({
                "f1": nn.Linear(hidden, hidden),
                "f2": nn.Linear(hidden, hidden),
            })
            for _ in range(n_blocks)
        ])
        self.out = nn.Linear(hidden, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = F.relu(self.proj(x))
        for block in self.block_weights:
            h = F.relu(block["f1"](x))
            x = x + block["f2"](h)
        return self.out(x)

    def spectral_adjacency(self, view: str = "dataflow") -> torch.Tensor:
        # Same construction as ResMLP.spectral_adjacency — reuse.
        device = self.proj.weight.device
        dtype = self.proj.weight.dtype
        H = self.hidden
        layer_sizes = [H] + [H] * (2 * self.n_blocks) + [10]
        total = sum(layer_sizes)
        offsets = [0]
        for s in layer_sizes:
            offsets.append(offsets[-1] + s)

        A = torch.zeros(total, total, device=device, dtype=dtype)
        for k, block in enumerate(self.block_weights):
            in_idx = 2 * k
            mid_idx = 2 * k + 1
            out_idx = 2 * k + 2

            absW1 = block["f1"].weight.abs()
            W1 = absW1.t()
            r0, r1 = offsets[in_idx], offsets[in_idx + 1]
            c0, c1 = offsets[mid_idx], offsets[mid_idx + 1]
            A[r0:r1, c0:c1] += W1
            A[c0:c1, r0:r1] += W1.t()
            if view == "factor":
                w = absW1.t() @ absW1
                w = w - torch.diag(torch.diag(w))
                A[r0:r1, r0:r1] += w

            absW2 = block["f2"].weight.abs()
            W2 = absW2.t()
            r0, r1 = offsets[mid_idx], offsets[mid_idx + 1]
            c0, c1 = offsets[out_idx], offsets[out_idx + 1]
            A[r0:r1, c0:c1] += W2
            A[c0:c1, r0:r1] += W2.t()
            if view == "factor":
                w = absW2.t() @ absW2
                w = w - torch.diag(torch.diag(w))
                A[r0:r1, r0:r1] += w

            I = torch.eye(H, device=device, dtype=dtype)
            r0, r1 = offsets[in_idx], offsets[in_idx + 1]
            c0, c1 = offsets[out_idx], offsets[out_idx + 1]
            A[r0:r1, c0:c1] += I
            A[c0:c1, r0:r1] += I.t()

        last_block_out = 2 * self.n_blocks
        final = 2 * self.n_blocks + 1
        absWout = self.out.weight.abs()
        Wout = absWout.t()
        r0, r1 = offsets[last_block_out], offsets[last_block_out + 1]
        c0, c1 = offsets[final], offsets[final + 1]
        A[r0:r1, c0:c1] += Wout
        A[c0:c1, r0:r1] += Wout.t()
        if view == "factor":
            w = absWout.t() @ absWout
            w = w - torch.diag(torch.diag(w))
            A[r0:r1, r0:r1] += w

        return A

    def spectral_weights(self) -> list[torch.Tensor]:
        out = []
        for block in self.block_weights:
            out.append(block["f1"].weight)
            out.append(block["f2"].weight)
        out.append(self.out.weight)
        return out


# ─── Training loop ──────────────────────────────────────────────────


@dataclass
class RunResult:
    dataset: str
    arm: str
    seed: int
    final_val_acc: float
    final_train_loss: float
    final_entropy: float
    final_kl: float
    wall_seconds: float
    # Effective-rank metrics (post-training snapshot)
    stable_rank_mean: float = 0.0
    stable_rank_max: float = 0.0
    spectral_norm_mean: float = 0.0
    spectral_norm_max: float = 0.0
    spectral_norm_product: float = 0.0
    participation_ratio_mean: float = 0.0
    # Per-epoch val accuracy trajectory (for convergence-speed analysis).
    # Stored as a list; serialised to CSV as a semicolon-separated string.
    val_acc_per_epoch: list[float] = field(default_factory=list)
    view: str = "dataflow"


def train_one_run(
    dataset: str, arm: str, seed: int,
    *, epochs: int, lr: float, lam: float, batch_size: int,
    reg_every_n: int, device: torch.device,
    view: str = "dataflow",
    target_entropy: float = 0.5,
    unified_beta: float = 1.0,
    adaptive_eta: float = 5.0,
    weight_decay: float = 0.0,
    lyapunov_eta: float = 5.0,
    lam_a: float = 1.0,
    lam_b: float = 1.0,
    tc_momentum_beta: float = 0.9,
    tc_variance_mode: str = "mix",
) -> RunResult:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    loaders_fn, net_cls = DATASETS[dataset]
    tr_loader, va_loader = loaders_fn(seed, batch_size=batch_size)
    model = net_cls().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    prev_eigvals: Optional[torch.Tensor] = None
    last_entropy = 0.0
    last_kl = 0.0
    last_task_loss = 0.0
    val_per_epoch: list[float] = []
    # For entropy_adaptive: H_norm trajectory window for second-derivative
    # feedback. Three values are enough for v→a; we keep five to smooth.
    h_norm_history: list[float] = []
    # For cross_layer_mi: live activations captured via forward hooks
    # on the Linear modules whose .weight is in spectral_weights().
    # Populated on every forward; consumed during reg-term computation.
    cross_layer_activations: list[torch.Tensor] = []
    cross_layer_hooks: list = []
    # For total_correlation_mi (Path I): KL-of-joint-Gram-spectrum and
    # EMA of joint-spectrum variance — combined into a damping λ_eff.
    prev_joint_eigvals: Optional[torch.Tensor] = None
    joint_var_ema: float = 0.0
    if arm in ("cross_layer_mi", "total_correlation_mi"):
        spec_ws = model.spectral_weights()
        weight_ids = [id(w) for w in spec_ws]
        target_modules = []
        for m in model.modules():
            if hasattr(m, "weight") and id(m.weight) in weight_ids:
                target_modules.append(m)
        target_modules.sort(key=lambda mm: weight_ids.index(id(mm.weight)))
        # Architectures whose spectral_weights() includes synthesised
        # tensors (e.g., CapsMLP's routing matrix `W.permute(...).reshape(...)`)
        # don't have an `nn.Linear` whose `.weight is W`. Path F / Path I
        # need at least two hookable modules to compute a pairwise / joint
        # MI; if fewer are found, the regulariser silently no-ops and the
        # arm collapses to baseline. Make this LOUD so a re-run with a
        # different model fixes it.
        if len(target_modules) < 2:
            print(
                f"  [WARN] {arm} on {dataset}: only {len(target_modules)} "
                f"hookable modules found out of {len(spec_ws)} spectral_weights "
                f"entries. Activation-side regulariser will not fire — "
                f"results will be byte-identical to baseline. Likely cause: "
                f"model.spectral_weights() includes synthesised tensors not "
                f"backed by an nn.Linear (e.g., CapsMLP routing W).",
                flush=True,
            )
        # `cross_layer_activations` is reset on every forward by clearing
        # in-place from inside the first hook.
        first_id = id(target_modules[0]) if target_modules else None
        for mod in target_modules:
            def _hook(_m, _i, out, *,
                      store=cross_layer_activations,
                      is_first=(id(mod) == first_id)):
                if is_first:
                    store.clear()
                # Keep the live graph (no .detach()) so gradient can flow.
                store.append(out)
            cross_layer_hooks.append(mod.register_forward_hook(_hook))

    t0 = time.time()
    for _epoch in range(epochs):
        model.train()
        for batch_idx, (x, y) in enumerate(tr_loader):
            x, y = x.to(device), y.to(device)
            optim.zero_grad()

            logits = model(x)
            task_loss = F.cross_entropy(logits, y)
            reg_term: Optional[torch.Tensor] = None

            if arm == "l2_weight_decay":
                l2 = sum((w.pow(2).sum() for w in model.spectral_weights()))
                reg_term = l2
            elif arm != "baseline" and (batch_idx % reg_every_n == 0):
                adj = model.spectral_adjacency(view=view)
                eigs = normalized_laplacian_eigvals(adj)
                if eigs is not None:
                    if arm == "scalar_entropy":
                        reg_term = spectral_entropy_bits(eigs)
                        last_entropy = float(reg_term.item())
                    elif arm == "kl_trajectory":
                        if prev_eigvals is not None:
                            reg_term = kl_spectra(prev_eigvals, eigs)
                            last_kl = float(reg_term.item())
                        prev_eigvals = eigs.detach()
                        last_entropy = float(
                            spectral_entropy_bits(eigs).item()
                        )
                    # ----- Path B: scale-invariant entropy -----
                    # Normalize H(A) ∈ [0, log2(rank)] down to [0, 1] by
                    # dividing by the maximal entropy for the spectrum
                    # dimension. Removes the architecture-dependent
                    # scale that makes a fixed λ non-universal.
                    # Theoretical hook: normalized entropy = effective-
                    # rank ratio, which bounds pseudodimension (VC) and
                    # matches the Kolmogorov-Arnold capacity lower
                    # bound.
                    elif arm == "scalar_entropy_normalized":
                        H = spectral_entropy_bits(eigs)
                        H_max = torch.log2(
                            torch.tensor(float(eigs.numel()),
                                          device=eigs.device)
                        ).clamp(min=1.0)
                        reg_term = H / H_max   # in [0, 1]
                        last_entropy = float(H.item())
                    # ----- Path A: target entropy (information bottleneck) -----
                    # Penalise squared deviation from a task-specific
                    # target H*. At H*=0 recovers the current behaviour.
                    # Allows "keep entropy at 0.5" for tasks where
                    # dropping below hurts (CapsMLP, circles).
                    elif arm == "entropy_target":
                        H = spectral_entropy_bits(eigs)
                        H_max = torch.log2(
                            torch.tensor(float(eigs.numel()),
                                          device=eigs.device)
                        ).clamp(min=1.0)
                        H_norm = H / H_max
                        target = torch.tensor(target_entropy,
                                               device=eigs.device,
                                               dtype=H_norm.dtype)
                        reg_term = (H_norm - target).pow(2)
                        last_entropy = float(H.item())
                    # ----- Path C: composite VC-style bound -----
                    # Regularise the Bartlett-Foster-Telgarsky-style
                    # generalisation bound rather than one component:
                    #   max( H_norm,
                    #        spectral_radius_ratio,
                    #        1 - stable_rank_ratio )
                    # All three components live in [0, 1] so the `max`
                    # has a natural scale and the `λ` is shared.
                    elif arm == "structural_composite":
                        H = spectral_entropy_bits(eigs)
                        rank_f = float(eigs.numel())
                        H_max = torch.log2(
                            torch.tensor(rank_f, device=eigs.device)
                        ).clamp(min=1.0)
                        H_norm = H / H_max                         # entropy term
                        # normalised Laplacian eigvals live in [0, 2]
                        max_eig = eigs.max()
                        spread = max_eig / 2.0                       # spectral radius term
                        # stable-rank proxy: (Σ λ²) / (max λ)² ∈ (0, rank]
                        eig_sq_sum = (eigs * eigs).sum()
                        sr = eig_sq_sum / max_eig.pow(2).clamp(min=1e-8)
                        sr_ratio = (sr / rank_f).clamp(max=1.0)
                        inv_sr = 1.0 - sr_ratio                      # low-rank penalty
                        reg_term = torch.stack(
                            [H_norm, spread, inv_sr]
                        ).max()
                        last_entropy = float(H.item())
                    # ----- entropy_unified: A + B blend -----
                    # L = (H_norm − H*)² + β · H_norm
                    # Combines target-attractor (Path A) with linear pull
                    # (Path B). β is the relative weight of B vs A's
                    # quadratic well; β=0 ⇒ pure A, β→∞ ⇒ pure B.
                    elif arm == "entropy_unified":
                        H = spectral_entropy_bits(eigs)
                        H_max = torch.log2(
                            torch.tensor(float(eigs.numel()),
                                          device=eigs.device)
                        ).clamp(min=1.0)
                        H_norm = H / H_max
                        target = torch.tensor(target_entropy,
                                               device=eigs.device,
                                               dtype=H_norm.dtype)
                        a_part = (H_norm - target).pow(2)
                        b_part = H_norm
                        reg_term = a_part + unified_beta * b_part
                        last_entropy = float(H.item())
                    # ----- entropy_adaptive: λ modulated by H acceleration -----
                    # Track normalised entropy across reg-update steps;
                    # if entropy is collapsing too fast (large negative
                    # acceleration) the regulariser is over-pushing →
                    # damp it. If stable / accelerating upward → boost.
                    # Multiplicative modulator on the standard B term.
                    elif arm == "entropy_adaptive":
                        H = spectral_entropy_bits(eigs)
                        H_max = torch.log2(
                            torch.tensor(float(eigs.numel()),
                                          device=eigs.device)
                        ).clamp(min=1.0)
                        H_norm = H / H_max
                        h_norm_history.append(float(H_norm.item()))
                        if len(h_norm_history) > 5:
                            h_norm_history.pop(0)
                        modulator = 1.0
                        if len(h_norm_history) >= 3:
                            v1 = h_norm_history[-1] - h_norm_history[-2]
                            v2 = h_norm_history[-2] - h_norm_history[-3]
                            accel = v1 - v2
                            # accel < 0 means entropy is dropping faster
                            # over time → over-regularising; reduce λ.
                            # accel > 0 means stabilising/recovering →
                            # safe to push more.
                            modulator = float(
                                math.exp(adaptive_eta * accel)
                            )
                            modulator = max(0.1, min(modulator, 10.0))
                        reg_term = modulator * H_norm
                        last_entropy = float(H.item())
                    # ----- cross_layer_mi: Sanchez-Giraldo Renyi-2 layer MI -----
                    # Activation-side regulariser. Uses the trace-1
                    # Gaussian-RBF Gram of each layer's post-linear
                    # output on the current mini-batch and penalises
                    # the sum of pairwise (i, j) Renyi-2 mutual
                    # informations across all layer pairs. Math doc:
                    # reports/sanchez_giraldo_framework.{md,pdf}.
                    elif arm == "cross_layer_mi":
                        if len(cross_layer_activations) >= 2:
                            mi_terms = []
                            # Build Gaussian-RBF Grams for every layer.
                            K_list = []
                            for a in cross_layer_activations:
                                a_flat = a.reshape(a.shape[0], -1)
                                # Median-bandwidth Gaussian kernel.
                                sq_norm = a_flat.pow(2).sum(dim=1, keepdim=True)
                                pair_sq = (sq_norm + sq_norm.t()
                                           - 2 * a_flat @ a_flat.t()).clamp(min=0)
                                pos = pair_sq[pair_sq > 0]
                                if pos.numel() == 0:
                                    continue
                                med = pos.median().clamp(min=1e-8)
                                sigma_sq = (med / 2.0).clamp(min=1e-8)
                                K = torch.exp(-pair_sq / (2.0 * sigma_sq))
                                K = K / K.trace().clamp(min=1e-12)
                                K_list.append(K)
                            # All-pairs I_2 (sum, normalised by # pairs).
                            n_pairs = 0
                            for i in range(len(K_list)):
                                for j in range(i + 1, len(K_list)):
                                    K_j = K_list[i] * K_list[j]
                                    K_j = K_j / K_j.trace().clamp(min=1e-12)
                                    H_i = -torch.log(
                                        (K_list[i] * K_list[i]).sum().clamp(min=1e-12))
                                    H_j = -torch.log(
                                        (K_list[j] * K_list[j]).sum().clamp(min=1e-12))
                                    H_join = -torch.log(
                                        (K_j * K_j).sum().clamp(min=1e-12))
                                    mi_terms.append(H_i + H_j - H_join)
                                    n_pairs += 1
                            if mi_terms:
                                reg_term = torch.stack(mi_terms).sum() / max(n_pairs, 1)
                                last_entropy = float(reg_term.item())
                    # ----- total_correlation_mi (Path I): L-way joint TC,
                    #       KL-feedback λ, variance-momentum λ -----------
                    # Three-way generalisation of cross_layer_mi (Path F):
                    #   (a) L-way joint Gram (renormalised after each
                    #       Hadamard) → multi-information / total
                    #       correlation
                    #         TC_2 = Σ_l H_2(K_l) − H_2(K_join_all_layers)
                    #       Subsumes the pairwise sum: pairwise misses
                    #       3-way (and higher) redundancy.
                    #   (b) KL-as-entropy-change feedback on λ:
                    #         lam_factor = exp(−η · KL(p_{t-1} ‖ p_t))
                    #       on the eigenvalue distribution of the joint
                    #       Gram (matched-sort, like kl_spectra). Same
                    #       Lyapunov-safe schedule as entropy_lyapunov.
                    #   (c) Variance-momentum damping on λ:
                    #         μ_t = β μ_{t-1} + (1−β) σ²(p_t)
                    #         inertia = μ_t / (μ_t + ε) ∈ (0, 1)
                    #       High variance ⇒ inertia ≈ 1 ⇒ λ damped
                    #       (be quiet while spectrum is still settling);
                    #       low variance ⇒ inertia → 0 ⇒ λ recovers.
                    elif arm == "total_correlation_mi":
                        if len(cross_layer_activations) >= 2:
                            K_list = []
                            for a in cross_layer_activations:
                                a_flat = a.reshape(a.shape[0], -1)
                                sq_norm = a_flat.pow(2).sum(dim=1, keepdim=True)
                                pair_sq = (sq_norm + sq_norm.t()
                                           - 2 * a_flat @ a_flat.t()).clamp(min=0)
                                pos = pair_sq[pair_sq > 0]
                                if pos.numel() == 0:
                                    continue
                                med = pos.median().clamp(min=1e-8)
                                sigma_sq = (med / 2.0).clamp(min=1e-8)
                                K = torch.exp(-pair_sq / (2.0 * sigma_sq))
                                K = K / K.trace().clamp(min=1e-12)
                                K_list.append(K)
                            if len(K_list) >= 2:
                                # (a) L-way joint Gram with stepwise
                                # renormalisation to keep entries from
                                # underflowing as L grows.
                                K_join = K_list[0]
                                for K_next in K_list[1:]:
                                    K_join = K_join * K_next
                                    K_join = K_join / K_join.trace().clamp(min=1e-12)
                                H_join = -torch.log(
                                    (K_join * K_join).sum().clamp(min=1e-12))
                                H_sum_terms = [
                                    -torch.log((K * K).sum().clamp(min=1e-12))
                                    for K in K_list
                                ]
                                H_sum = torch.stack(H_sum_terms).sum()
                                TC = H_sum - H_join
                                # (b) KL feedback. Eigenvalue distribution
                                # of the joint Gram → p_t. Matched-sort
                                # KL between p_{t-1} and p_t, mirroring
                                # kl_spectra's protocol.
                                joint_eigs = torch.linalg.eigvalsh(
                                    K_join.detach()
                                ).clamp(min=1e-12)
                                joint_eigs = joint_eigs / joint_eigs.sum()
                                kl_step = 0.0
                                if prev_joint_eigvals is not None and \
                                   prev_joint_eigvals.shape == joint_eigs.shape:
                                    p_prev = prev_joint_eigvals
                                    p_curr = joint_eigs
                                    kl_step = float(
                                        (p_prev * (p_prev / p_curr).log()).sum().item()
                                    )
                                prev_joint_eigvals = joint_eigs
                                # (c) variance-momentum.
                                sigma2 = float(
                                    (joint_eigs - joint_eigs.mean()).pow(2).sum().item()
                                )
                                joint_var_ema = (
                                    tc_momentum_beta * joint_var_ema
                                    + (1.0 - tc_momentum_beta) * sigma2
                                )
                                inertia = joint_var_ema / (joint_var_ema + 1e-3)
                                # KL-feedback λ factor (Lyapunov-style).
                                lam_factor = math.exp(-lyapunov_eta * kl_step)
                                lam_factor = max(0.1, min(10.0, lam_factor))
                                # Variance-mode mixer:
                                #   damp:    var_factor = (1 - inertia)
                                #            (transient-conservative —
                                #            quiet during high-spread
                                #            regime)
                                #   amplify: var_factor = inertia
                                #            (push harder once joint
                                #            spectrum has settled wide)
                                #   mix:     stage-aware blend driven by
                                #            the stability indicator
                                #            w = exp(-η·KL) clipped to
                                #            [0, 1]. KL large (early /
                                #            transients) ⇒ w small ⇒
                                #            blend → damp. KL→0 (steady)
                                #            ⇒ w→1 ⇒ blend → amplify.
                                #            Smoothly hands off control.
                                if tc_variance_mode == "amplify":
                                    var_factor = inertia
                                elif tc_variance_mode == "mix":
                                    w_raw = math.exp(-lyapunov_eta * kl_step)
                                    w = max(0.0, min(1.0, w_raw))
                                    var_factor = w * inertia + (1.0 - w) * (1.0 - inertia)
                                else:                          # "damp" (default)
                                    var_factor = 1.0 - inertia
                                reg_term = lam_factor * var_factor * TC
                                last_entropy = float(TC.item())
                                last_kl = kl_step
                    # ----- entropy_lyapunov: closed-loop controller -----
                    # Solves three framework gaps at once:
                    #
                    #   (1) Per-term lambda — `lam_a` weights the
                    #       quadratic-target term A, `lam_b` weights the
                    #       linear pull term B. Multi-objective made
                    #       explicit.
                    #   (2) KL as feedback — KL(p_{t-1} ‖ p_t) modulates
                    #       lambda_eff via lam_eff = lam_0 · exp(-eta · KL).
                    #       The Lyapunov-safe schedule: as the regulariser
                    #       starts winning (large KL = spectrum moving
                    #       fast), lambda damps. When stable (KL→0),
                    #       lambda recovers. Closed-loop control derived
                    #       from V̇ < 0 condition with V = task loss.
                    #   (3) Distribution-moment decay — H_norm running
                    #       window's mean appears as the linear-pull
                    #       term (Adam-like, but only first moment for
                    #       simplicity; full m_t/√v_t variant deferred).
                    #
                    # NOTE: this arm uses prev_eigvals (the kl_trajectory
                    # state) for the feedback signal — connects two
                    # previously-independent regularisers.
                    elif arm == "entropy_lyapunov":
                        H = spectral_entropy_bits(eigs)
                        rank_f = float(eigs.numel())
                        H_max = torch.log2(
                            torch.tensor(rank_f, device=eigs.device)
                        ).clamp(min=1.0)
                        H_norm = H / H_max

                        # KL trajectory signal (in bits per step)
                        if prev_eigvals is not None:
                            kl_t = float(kl_spectra(prev_eigvals, eigs).item())
                            last_kl = kl_t
                        else:
                            kl_t = 0.0
                        prev_eigvals = eigs.detach()

                        # Lyapunov-safe λ multiplier: exp(−η · KL)
                        # (large KL → reduce; small KL → grow up to clamp).
                        lam_factor = float(math.exp(-lyapunov_eta * kl_t))
                        lam_factor = max(0.1, min(10.0, lam_factor))

                        # Per-term weighted: A (quadratic-target) + B (linear)
                        target = torch.tensor(
                            target_entropy, device=eigs.device,
                            dtype=H_norm.dtype,
                        )
                        a_part = (H_norm - target).pow(2)
                        b_part = H_norm
                        reg_term = lam_factor * (
                            lam_a * a_part + lam_b * b_part
                        )

                        # Track moments (informational; no Adam re-scale yet).
                        h_norm_history.append(float(H_norm.item()))
                        if len(h_norm_history) > 20:
                            h_norm_history.pop(0)

                        last_entropy = float(H.item())
                    # ----- entropy_target_ka: KA-derived per-dataset H* -----
                    # Hecht-Nielsen / Kolmogorov-Arnold: any continuous
                    # f: ℝⁿ → ℝ representable by a network with effective
                    # rank ≥ 2n+1. Below that the network is
                    # representationally insufficient. So:
                    #   H*_KA = log₂(2n+1) / log₂(rank(A))
                    #   H*    = max(0.5, H*_KA)              # never below safe default
                    # The 0.5 floor protects against KA over-tightness on
                    # overparameterised image tasks where empirically a
                    # lower H* still helps.
                    elif arm == "entropy_target_ka":
                        H = spectral_entropy_bits(eigs)
                        rank_f = float(eigs.numel())
                        H_max = torch.log2(
                            torch.tensor(rank_f, device=eigs.device)
                        ).clamp(min=1.0)
                        H_norm = H / H_max
                        # Read input dim from the model's first weight.
                        n_input = float(model.spectral_weights()[0].shape[1])
                        h_ka = math.log2(2.0 * n_input + 1.0) / math.log2(rank_f if rank_f > 1 else 2.0)
                        h_star = max(0.5, min(1.0, h_ka))
                        target = torch.tensor(h_star,
                                               device=eigs.device,
                                               dtype=H_norm.dtype)
                        reg_term = (H_norm - target).pow(2)
                        last_entropy = float(H.item())
                    # ----- entropy_telgarsky: depth-weighted per-layer entropy -----
                    # Compute a per-layer adjacency block (each layer's
                    # weight matrix's normalised Laplacian) and apply a
                    # depth-decay weight to its entropy term. Earlier
                    # layers get penalised more (≈compression); later
                    # layers carry less penalty (≈information
                    # preservation). Operationalises Telgarsky's
                    # depth-vs-width tradeoff: deep networks have
                    # representational headroom in late layers.
                    #   weight_l = (1 - l/L)^p     # p > 0; default p=1
                    #   L_reg = Σ_l weight_l · H_norm(layer_l)²
                    elif arm == "entropy_telgarsky":
                        ws = model.spectral_weights()
                        L_layers = len(ws)
                        per_layer_terms: list[torch.Tensor] = []
                        for l, W in enumerate(ws):
                            # Per-layer adjacency: bipartite (W^T W) on
                            # input nodes ∪ (W W^T) on output nodes;
                            # cheaper proxy is just |W|^T |W| diagonal
                            # / spectrum. Use eigvals of W W^T (symmetric).
                            Wn = W.detach().abs()
                            gram = Wn @ Wn.t()
                            # Make stable; use eigvalsh on the
                            # symmetrised gram matrix.
                            try:
                                lay_eigs = torch.linalg.eigvalsh(
                                    gram + 1e-8 * torch.eye(
                                        gram.shape[0], device=gram.device,
                                        dtype=gram.dtype)
                                )
                            except Exception:
                                continue
                            # Normalise into a probability distribution.
                            lay_eigs = lay_eigs.clamp(min=0)
                            total = lay_eigs.sum().clamp(min=1e-8)
                            p_dist = (lay_eigs / total) + 1e-12
                            H_l = -(p_dist * torch.log2(p_dist)).sum()
                            H_l_max = math.log2(float(p_dist.numel()))
                            if H_l_max <= 0:
                                continue
                            H_norm_l = H_l / H_l_max
                            # We want to compute it differentiably from
                            # the actual W (re-build with grad)
                            Wg = W.abs()
                            Gg = Wg @ Wg.t()
                            # use a smooth entropy proxy: −Σ(σ_i² log σ_i²)
                            # via singular values of W is differentiable
                            sv = torch.linalg.svdvals(W)
                            sv2 = (sv * sv).clamp(min=1e-8)
                            ptot = sv2.sum().clamp(min=1e-8)
                            pq = sv2 / ptot
                            H_diff = -(pq * torch.log2(pq + 1e-12)).sum()
                            H_diff_max = math.log2(float(sv.numel()))
                            if H_diff_max <= 0:
                                continue
                            H_norm_diff = H_diff / H_diff_max
                            depth_weight = (1.0 - l / max(L_layers - 1, 1)) ** 1.0
                            per_layer_terms.append(depth_weight * H_norm_diff)
                            if l == 0:
                                last_entropy = float(H_diff.item())
                        if per_layer_terms:
                            reg_term = torch.stack(per_layer_terms).sum() / max(len(per_layer_terms), 1)
                    # ----- total_combined: A + B + C summed -----
                    # Literal sum of the three universality candidates.
                    # H_norm appears in all three terms (A's quadratic
                    # well, B's linear pull, C's max), so the entropy
                    # signal is triple-counted; non-entropy structural
                    # bound terms (spread, inv_sr) come only from C.
                    elif arm == "total_combined":
                        H = spectral_entropy_bits(eigs)
                        rank_f = float(eigs.numel())
                        H_max = torch.log2(
                            torch.tensor(rank_f, device=eigs.device)
                        ).clamp(min=1.0)
                        H_norm = H / H_max
                        target = torch.tensor(target_entropy,
                                               device=eigs.device,
                                               dtype=H_norm.dtype)
                        # A
                        a_part = (H_norm - target).pow(2)
                        # B
                        b_part = H_norm
                        # C
                        max_eig = eigs.max()
                        spread = max_eig / 2.0
                        eig_sq_sum = (eigs * eigs).sum()
                        sr = eig_sq_sum / max_eig.pow(2).clamp(min=1e-8)
                        sr_ratio = (sr / rank_f).clamp(max=1.0)
                        inv_sr = 1.0 - sr_ratio
                        c_part = torch.stack([H_norm, spread, inv_sr]).max()
                        reg_term = a_part + b_part + c_part
                        last_entropy = float(H.item())

            loss = task_loss if reg_term is None else task_loss + lam * reg_term
            loss.backward()
            optim.step()
            last_task_loss = float(task_loss.item())

        # End-of-epoch val accuracy for convergence-speed tracking.
        model.eval()
        ep_correct, ep_total = 0, 0
        with torch.no_grad():
            for x, y in va_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(dim=1)
                ep_correct += int((pred == y).sum().item())
                ep_total += y.size(0)
        val_per_epoch.append(ep_correct / ep_total)

    val_acc = val_per_epoch[-1] if val_per_epoch else 0.0

    # Remove forward hooks (avoid leaks across runs sharing the model).
    for h in cross_layer_hooks:
        h.remove()
    cross_layer_hooks.clear()

    # Final entropy + effective-rank snapshot.
    with torch.no_grad():
        eigs = normalized_laplacian_eigvals(model.spectral_adjacency(view=view))
        if eigs is not None:
            last_entropy = float(spectral_entropy_bits(eigs).item())
        rank_metrics = aggregate_spectral_metrics(model.spectral_weights())

    return RunResult(
        dataset=dataset, arm=arm, seed=seed,
        final_val_acc=val_acc, final_train_loss=last_task_loss,
        final_entropy=last_entropy, final_kl=last_kl,
        wall_seconds=time.time() - t0,
        stable_rank_mean=rank_metrics["stable_rank_mean"],
        stable_rank_max=rank_metrics["stable_rank_max"],
        spectral_norm_mean=rank_metrics["spectral_norm_mean"],
        spectral_norm_max=rank_metrics["spectral_norm_max"],
        spectral_norm_product=rank_metrics["spectral_norm_product"],
        participation_ratio_mean=rank_metrics["participation_ratio_mean"],
        val_acc_per_epoch=val_per_epoch,
        view=view,
    )


# ─── Orchestration ──────────────────────────────────────────────────


@dataclass
class BenchConfig:
    datasets: tuple[str, ...]
    arms: tuple[str, ...]
    seeds: int
    epochs: int
    lr: float
    lam: float
    batch_size: int
    reg_every_n: int
    out_dir: Path
    view: str = "dataflow"
    target_entropy: float = 0.5
    unified_beta: float = 1.0
    adaptive_eta: float = 5.0
    weight_decay: float = 0.0
    lyapunov_eta: float = 5.0
    lam_a: float = 1.0
    lam_b: float = 1.0
    tc_momentum_beta: float = 0.9
    tc_variance_mode: str = "mix"


def run_all(cfg: BenchConfig, device: torch.device) -> list[RunResult]:
    results: list[RunResult] = []
    total = len(cfg.datasets) * len(cfg.arms) * cfg.seeds
    done = 0
    for dataset in cfg.datasets:
        for arm in cfg.arms:
            for seed in range(cfg.seeds):
                r = train_one_run(
                    dataset, arm, seed,
                    epochs=cfg.epochs, lr=cfg.lr, lam=cfg.lam,
                    batch_size=cfg.batch_size,
                    reg_every_n=cfg.reg_every_n, device=device,
                    view=cfg.view,
                    target_entropy=cfg.target_entropy,
                    unified_beta=cfg.unified_beta,
                    adaptive_eta=cfg.adaptive_eta,
                    weight_decay=cfg.weight_decay,
                    lyapunov_eta=cfg.lyapunov_eta,
                    lam_a=cfg.lam_a,
                    lam_b=cfg.lam_b,
                    tc_momentum_beta=cfg.tc_momentum_beta,
                    tc_variance_mode=cfg.tc_variance_mode,
                )
                results.append(r)
                done += 1
                print(f"  [{done}/{total}] {dataset}/{arm}/view={cfg.view}/seed={seed} "
                      f"acc={r.final_val_acc:.4f} H={r.final_entropy:.3f} "
                      f"KL={r.final_kl:.4f} {r.wall_seconds:.1f}s",
                      flush=True)
    return results


def summarise(results: list[RunResult], cfg: BenchConfig) -> None:
    print()
    print(f"Seeds per arm: {cfg.seeds}  Epochs: {cfg.epochs}  λ: {cfg.lam}")
    print(f"Regularizer update: every {cfg.reg_every_n} batches")
    print()
    for dataset in cfg.datasets:
        print(f"=== {dataset} ===")
        print(f"  {'arm':<18}  {'min':>8}  {'avg':>8}  {'max':>8}  "
              f"{'stdev':>10}  {'final_H':>8}  {'final_KL':>10}")
        for arm in cfg.arms:
            accs = [r.final_val_acc for r in results
                    if r.dataset == dataset and r.arm == arm]
            ents = [r.final_entropy for r in results
                    if r.dataset == dataset and r.arm == arm]
            kls = [r.final_kl for r in results
                    if r.dataset == dataset and r.arm == arm]
            if not accs:
                continue
            mn, mx = min(accs), max(accs)
            av = statistics.mean(accs)
            sd = statistics.stdev(accs) if len(accs) > 1 else 0.0
            me = statistics.mean(ents) if ents else 0.0
            mk = statistics.mean(kls) if kls else 0.0
            print(f"  {arm:<18}  {mn:>8.4f}  {av:>8.4f}  {mx:>8.4f}  "
                  f"{sd:>10.5f}  {me:>8.3f}  {mk:>10.5f}")

        # Paired comparison vs baseline
        for arm in cfg.arms:
            if arm == "baseline":
                continue
            deltas = []
            for seed in range(cfg.seeds):
                b = [r for r in results if r.dataset == dataset and r.arm == "baseline" and r.seed == seed]
                a = [r for r in results if r.dataset == dataset and r.arm == arm and r.seed == seed]
                if b and a:
                    deltas.append(a[0].final_val_acc - b[0].final_val_acc)
            if not deltas:
                continue
            mean_d = statistics.mean(deltas)
            sd_d = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
            se = sd_d / (len(deltas) ** 0.5) if sd_d > 0 else 1.0
            t = mean_d / se if se > 0 else 0.0
            w = sum(1 for d in deltas if d > 0)
            l = sum(1 for d in deltas if d < 0)
            tie = sum(1 for d in deltas if d == 0)
            print(f"  paired Δ ({arm} − baseline): "
                  f"mean={mean_d:+.5f}  sd={sd_d:.5f}  t={t:+.2f}  "
                  f"(W/L/T={w}/{l}/{tie})")
        print()


def write_csv(results: list[RunResult], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "arm", "view", "seed", "final_val_acc",
                    "final_train_loss", "final_entropy", "final_kl",
                    "wall_seconds",
                    "stable_rank_mean", "stable_rank_max",
                    "spectral_norm_mean", "spectral_norm_max",
                    "spectral_norm_product", "participation_ratio_mean",
                    "val_acc_per_epoch"])
        for r in results:
            # Per-epoch trajectory is serialised as semicolon-separated
            # floats so the CSV stays parseable in one pass; consumers
            # split on ';' and convert to float.
            traj = ";".join(f"{a:.6f}" for a in r.val_acc_per_epoch)
            w.writerow([r.dataset, r.arm, r.view, r.seed,
                        f"{r.final_val_acc:.6f}",
                        f"{r.final_train_loss:.6f}",
                        f"{r.final_entropy:.6f}",
                        f"{r.final_kl:.6f}",
                        f"{r.wall_seconds:.2f}",
                        f"{r.stable_rank_mean:.4f}",
                        f"{r.stable_rank_max:.4f}",
                        f"{r.spectral_norm_mean:.4f}",
                        f"{r.spectral_norm_max:.4f}",
                        f"{r.spectral_norm_product:.4f}",
                        f"{r.participation_ratio_mean:.4f}",
                        traj])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--datasets", nargs="+", default=["mnist"],
                    choices=list(DATASETS.keys()))
    ap.add_argument("--arms", nargs="+",
                    default=["baseline", "scalar_entropy", "kl_trajectory"],
                    choices=["baseline", "scalar_entropy", "kl_trajectory",
                             "l2_weight_decay",
                             "scalar_entropy_normalized",
                             "entropy_target",
                             "structural_composite",
                             "entropy_unified",
                             "entropy_adaptive",
                             "total_combined",
                             "entropy_target_ka",
                             "entropy_telgarsky",
                             "cross_layer_mi",
                             "entropy_lyapunov",
                             "total_correlation_mi"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lam", type=float, default=1e-2)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--reg-every-n", type=int, default=50,
                    help="Update spectral regularizer every N batches.")
    ap.add_argument("--view", default="dataflow", choices=["dataflow", "factor"],
                    help="Which hypergraph view's spectrum to regularize: "
                         "`dataflow` = bipartite layer-to-layer adjacency "
                         "(star expansion); `factor` = clique expansion "
                         "adding within-layer correlation terms. Thesis §6.1.")
    ap.add_argument("--target-entropy", type=float, default=0.5,
                    help="Target H* for the entropy_target arm; "
                         "units are normalised entropy in [0, 1].")
    ap.add_argument("--unified-beta", type=float, default=1.0,
                    help="entropy_unified mix weight: relative weight of "
                         "B (linear) vs A (quadratic-target). β=0 ⇒ pure A.")
    ap.add_argument("--adaptive-eta", type=float, default=5.0,
                    help="entropy_adaptive sensitivity to H acceleration. "
                         "Higher η = stronger λ modulation per H second-derivative.")
    ap.add_argument("--weight-decay", type=float, default=0.0,
                    help="Adam optimizer weight_decay (L2 weight decay applied "
                         "by the optimizer itself; phase-9 composability test).")
    ap.add_argument("--lyapunov-eta", type=float, default=5.0,
                    help="entropy_lyapunov KL-feedback strength: "
                         "lam_eff = lam_0 · exp(-eta · KL_step). Higher η = "
                         "tighter coupling between KL trajectory and lambda.")
    ap.add_argument("--lam-a", type=float, default=1.0,
                    help="Per-term weight on the (H_norm - H*)² (Path A) "
                         "component of entropy_lyapunov. Default 1.0.")
    ap.add_argument("--lam-b", type=float, default=1.0,
                    help="Per-term weight on the H_norm (Path B) component "
                         "of entropy_lyapunov. Default 1.0.")
    ap.add_argument("--tc-momentum-beta", type=float, default=0.9,
                    help="EMA decay for the variance-momentum term in "
                         "total_correlation_mi (Path I). Higher β = more "
                         "inertia, slower λ updates. Default 0.9.")
    ap.add_argument("--tc-variance-mode", default="mix",
                    choices=["damp", "amplify", "mix"],
                    help="How variance-momentum modulates λ in "
                         "total_correlation_mi (Path I). "
                         "`damp`: λ ∝ (1−inertia) — quiet when joint "
                         "spectrum has high spread (transient-conservative). "
                         "`amplify`: λ ∝ inertia — push harder when "
                         "spectrum is settled and spread (steady-state). "
                         "`mix` (default): stage-aware blend, weight = "
                         "exp(−η·KL) ∈ [0, 1] — damp during transients, "
                         "amplify in steady state. Default 'mix'.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dir", type=Path, default=Path("data/benchmarks"))
    ap.add_argument("--no-csv", action="store_true")
    args = ap.parse_args()

    cfg = BenchConfig(
        datasets=tuple(args.datasets), arms=tuple(args.arms),
        seeds=args.seeds, epochs=args.epochs, lr=args.lr, lam=args.lam,
        batch_size=args.batch_size, reg_every_n=args.reg_every_n,
        out_dir=args.out_dir, view=args.view,
        target_entropy=args.target_entropy,
        unified_beta=args.unified_beta,
        adaptive_eta=args.adaptive_eta,
        weight_decay=args.weight_decay,
        lyapunov_eta=args.lyapunov_eta,
        lam_a=args.lam_a,
        lam_b=args.lam_b,
        tc_momentum_beta=args.tc_momentum_beta,
        tc_variance_mode=args.tc_variance_mode,
    )

    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Config: {cfg}")
    print()

    t0 = time.time()
    results = run_all(cfg, device)
    elapsed = time.time() - t0

    if not args.no_csv:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        csv_path = cfg.out_dir / f"thesis_iv_hard_{stamp}.csv"
        write_csv(results, csv_path)
        print(f"\nWrote {len(results)} records to {csv_path}")

    summarise(results, cfg)
    print(f"Total elapsed: {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
