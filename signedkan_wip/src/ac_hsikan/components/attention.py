"""SignAttention strategies — per-pair sign computation.

Each strategy implements:
    compute_anchor_signs(x, local) -> (B, L, K_total)
        Signs anchor -> each candidate.
    compute_consecutive_signs(x, local, k) -> (B, L, k-1)
        Signs n_{t-1} -> n_t for the (k-1) chain edges.
    compute_pair_signs(x, src, dst) -> (B, L)
        Signs at (src[i], dst[i]) pairs (used by the closing-edge of a cycle).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from ..config import AcHsikanConfig


class SignAttention(nn.Module, ABC):
    """Strategy ABC for sign computation between position pairs.

    All concrete strategies return values in $[-1, +1]$ (tanh-clamped
    or hard-sign + STE) and produce them with linear cost in the
    candidate-set size ``K_total`` rather than quadratic in the
    sequence length ``L``.
    """

    @abstractmethod
    def compute_anchor_signs(
        self, x: torch.Tensor, local: torch.Tensor,
    ) -> torch.Tensor:
        """``x``: (B, L, d_model); ``local``: (L, K_total).  Returns
        (B, L, K_total) signs from anchor i to candidate ``local[i, k]``.
        """

    @abstractmethod
    def compute_consecutive_signs(
        self, x: torch.Tensor, local: torch.Tensor, k: int,
    ) -> torch.Tensor:
        """Signs along the chain ``n_{t-1} -> n_t`` for hops
        ``t in [1, k)``. Returns shape (B, L, k-1).
        """

    @abstractmethod
    def compute_pair_signs(
        self, x: torch.Tensor, src_idx: torch.Tensor, dst_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Signs at (src_idx[i], dst_idx[i]) pairs.  Returns (B, L)."""


# ── Bilinear sparse ──────────────────────────────────────────────────

class BilinearSparseAttention(SignAttention):
    """Cheap bilinear $W_q x_i \\cdot W_k x_j$ scoring; no L² scan."""

    def __init__(self, d_model: int, hidden: int, temperature: float = 1.0):
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0; got {temperature}")
        self.hidden = int(hidden)
        self.temperature = float(temperature)
        self.W_q = nn.Linear(d_model, hidden, bias=False)
        self.W_k = nn.Linear(d_model, hidden, bias=False)
        with torch.no_grad():
            self.W_q.weight.mul_(0.1)
            self.W_k.weight.mul_(0.1)

    def _qk(self, x: torch.Tensor):
        return self.W_q(x), self.W_k(x)

    def compute_anchor_signs(self, x, local):
        q, k_ = self._qk(x)
        k_g = k_[:, local, :]
        s = (q.unsqueeze(2) * k_g).sum(-1) / (self.hidden ** 0.5)
        return torch.tanh(s / self.temperature)

    def compute_consecutive_signs(self, x, local, k):
        q, k_ = self._qk(x)
        src = local[:, :k - 1]; dst = local[:, 1:k]
        s = (q[:, src, :] * k_[:, dst, :]).sum(-1) / (self.hidden ** 0.5)
        return torch.tanh(s / self.temperature)

    def compute_pair_signs(self, x, src_idx, dst_idx):
        q, k_ = self._qk(x)
        s = (q[:, src_idx, :] * k_[:, dst_idx, :]).sum(-1) / (self.hidden ** 0.5)
        return torch.tanh(s / self.temperature)


# ── Quaternion sparse ────────────────────────────────────────────────

class QuaternionSparseAttention(SignAttention):
    """Hamilton-product real-part scoring; single head.

    Bakes the $\\{+1, -1, -1, -1\\}$ coefficients into the q projection
    so the per-pair compute collapses to a sum-product.
    """
    _COEFFS = (1.0, -1.0, -1.0, -1.0)

    def __init__(self, d_model: int, hidden: int, temperature: float = 1.0):
        super().__init__()
        if hidden % 4 != 0:
            raise ValueError(f"hidden must be % 4 == 0; got {hidden}")
        self.hidden = int(hidden); self.n_quat = hidden // 4
        self.temperature = float(temperature)
        self.W_q = nn.Linear(d_model, hidden, bias=False)
        self.W_k = nn.Linear(d_model, hidden, bias=False)
        with torch.no_grad():
            self.W_q.weight.mul_(0.1)
            self.W_k.weight.mul_(0.1)
        self.scale = self.n_quat ** -0.5

    def _qk(self, x: torch.Tensor):
        B, L, _ = x.shape
        q = self.W_q(x).view(B, L, self.n_quat, 4)
        k_ = self.W_k(x).view(B, L, self.n_quat, 4)
        coeffs = torch.tensor(self._COEFFS, device=x.device, dtype=x.dtype)
        # Pre-sign the q via Hamilton-product coefficients so subsequent
        # sums are correct without a separate stacked-coeff einsum.
        q = (q * coeffs).reshape(B, L, self.hidden)
        k_ = k_.reshape(B, L, self.hidden)
        return q, k_

    def compute_anchor_signs(self, x, local):
        q, k_ = self._qk(x)
        k_g = k_[:, local, :]
        s = (q.unsqueeze(2) * k_g).sum(-1) * self.scale / self.temperature
        return torch.tanh(s)

    def compute_consecutive_signs(self, x, local, k):
        q, k_ = self._qk(x)
        src = local[:, :k - 1]; dst = local[:, 1:k]
        s = (q[:, src, :] * k_[:, dst, :]).sum(-1) * self.scale / self.temperature
        return torch.tanh(s)

    def compute_pair_signs(self, x, src_idx, dst_idx):
        q, k_ = self._qk(x)
        s = (q[:, src_idx, :] * k_[:, dst_idx, :]).sum(-1) * self.scale / self.temperature
        return torch.tanh(s)


# ── Multi-head quaternion sparse (speedup #1) ────────────────────────

class MultiHeadQuaternionSparseAttention(SignAttention):
    """Multi-head Hamilton-product attention; heads computed in parallel
    via a single batched einsum (no Python loop over heads).

    Per-head: ``hidden_per_head = hidden / n_heads`` features, of which
    ``hidden_per_head / 4`` are independent quaternions. The H heads'
    scores are averaged at the end (could also be concatenated;
    averaging matches the production graph-HSiKAN attention).
    """
    _COEFFS = (1.0, -1.0, -1.0, -1.0)

    def __init__(
        self, d_model: int, hidden: int, n_heads: int,
        temperature: float = 1.0,
    ):
        super().__init__()
        if hidden % n_heads != 0:
            raise ValueError(
                f"hidden ({hidden}) must be divisible by n_heads ({n_heads})"
            )
        per_head = hidden // n_heads
        if per_head % 4 != 0:
            raise ValueError(
                f"hidden_per_head ({per_head}) must be % 4 == 0; "
                f"adjust hidden or n_heads"
            )
        self.hidden = int(hidden); self.n_heads = int(n_heads)
        self.per_head = per_head; self.n_quat_per_head = per_head // 4
        self.temperature = float(temperature)
        self.W_q = nn.Linear(d_model, hidden, bias=False)
        self.W_k = nn.Linear(d_model, hidden, bias=False)
        with torch.no_grad():
            self.W_q.weight.mul_(0.1)
            self.W_k.weight.mul_(0.1)
        self.scale = self.n_quat_per_head ** -0.5

    def _qk(self, x: torch.Tensor):
        B, L, _ = x.shape
        q = self.W_q(x).view(B, L, self.n_heads, self.n_quat_per_head, 4)
        k_ = self.W_k(x).view(B, L, self.n_heads, self.n_quat_per_head, 4)
        coeffs = torch.tensor(self._COEFFS, device=x.device, dtype=x.dtype)
        q = (q * coeffs).reshape(B, L, self.n_heads, self.per_head)
        k_ = k_.reshape(B, L, self.n_heads, self.per_head)
        return q, k_

    def compute_anchor_signs(self, x, local):
        q, k_ = self._qk(x)                         # (B, L, H, ph)
        # Gather along L dim per anchor: (B, L, K_total, H, ph)
        # Using index_select-style: k_[:, local, :, :]
        k_g = k_[:, local, :, :]                    # (B, L, K_total, H, ph)
        # einsum: "blhd, blkhd -> blkh"
        s = torch.einsum("blhd,blkhd->blkh", q, k_g)
        s = s * self.scale / self.temperature
        # Average heads -> (B, L, K_total)
        s = s.mean(dim=-1)
        return torch.tanh(s)

    def compute_consecutive_signs(self, x, local, k):
        q, k_ = self._qk(x)
        src = local[:, :k - 1]; dst = local[:, 1:k]
        q_s = q[:, src, :, :]; k_d = k_[:, dst, :, :]  # (B, L, k-1, H, ph)
        s = torch.einsum("blthd,blthd->blth", q_s, k_d)
        s = (s * self.scale / self.temperature).mean(dim=-1)
        return torch.tanh(s)

    def compute_pair_signs(self, x, src_idx, dst_idx):
        q, k_ = self._qk(x)
        q_s = q[:, src_idx, :, :]; k_d = k_[:, dst_idx, :, :]
        s = torch.einsum("blhd,blhd->blh", q_s, k_d)
        s = (s * self.scale / self.temperature).mean(dim=-1)
        return torch.tanh(s)


# ── Dense adapter (back-compat for v1 / v1.3) ────────────────────────

class DenseAttentionAdapter(SignAttention):
    """Wraps a dense SignHead so it satisfies the SignAttention contract.

    For back-compat with the v1 path; computes the full ``(B, L, L)``
    tensor and then gathers the requested subset. Useful at small L
    where the dense L² is cheaper than the gather overhead.
    """
    def __init__(self, dense_head: nn.Module):
        super().__init__()
        self.dense_head = dense_head

    def _dense(self, x: torch.Tensor):
        return self.dense_head(x)

    def compute_anchor_signs(self, x, local):
        signs = self._dense(x)
        B, L, _ = x.shape
        K_total = local.shape[1]
        return torch.gather(
            signs, dim=2,
            index=local.unsqueeze(0).expand(B, L, K_total),
        )

    def compute_consecutive_signs(self, x, local, k):
        signs = self._dense(x)
        B, L, _ = x.shape
        src = local[:, :k - 1]; dst = local[:, 1:k]
        sig_rows = signs[:, src, :]                  # (B, L, k-1, L)
        dst_exp = dst.unsqueeze(0).unsqueeze(-1).expand(B, L, k - 1, 1)
        return sig_rows.gather(dim=-1, index=dst_exp).squeeze(-1)

    def compute_pair_signs(self, x, src_idx, dst_idx):
        signs = self._dense(x)
        return signs[:, src_idx, dst_idx]


# ── Factory ──────────────────────────────────────────────────────────

def build_sign_attention(cfg: "AcHsikanConfig") -> SignAttention:
    """Pick the right concrete SignAttention from the config flags."""
    if not cfg.sparse_sign_head:
        # Dense path: wrap the legacy SignHead for compat.
        from ..sign_head import build_sign_head
        head = build_sign_head(
            kind=cfg.sign_head_kind, d_model=cfg.d_model,
            hidden=cfg.sign_head_hidden, temperature=cfg.sign_temperature,
            use_ste=cfg.sign_ste,
        )
        return DenseAttentionAdapter(head)
    # Sparse paths.
    n_heads = getattr(cfg, "n_sign_heads", 1)
    if cfg.sign_head_kind == "quaternion":
        if n_heads > 1:
            return MultiHeadQuaternionSparseAttention(
                d_model=cfg.d_model, hidden=cfg.sign_head_hidden,
                n_heads=n_heads, temperature=cfg.sign_temperature,
            )
        return QuaternionSparseAttention(
            d_model=cfg.d_model, hidden=cfg.sign_head_hidden,
            temperature=cfg.sign_temperature,
        )
    if cfg.sign_head_kind == "bilinear":
        return BilinearSparseAttention(
            d_model=cfg.d_model, hidden=cfg.sign_head_hidden,
            temperature=cfg.sign_temperature,
        )
    raise ValueError(f"unknown sign_head_kind: {cfg.sign_head_kind!r}")
