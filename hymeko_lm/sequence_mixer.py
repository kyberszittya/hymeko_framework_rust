"""Fiber-Spike-Rotor sequence mixer — replaces softmax self-attention.

The mixing of token $i$ is a causal, signed, rotor-transported, spike-gated gather
over the past:

    mixed_i = (1 / |{j<=i}|) * sum_{j<=i}  gate(i,j) * sign[i-j] * R[i-j] · h_j

where, per 3-block of the hidden vector,
  * **R[i-j]** is a rotor (unit quaternion, Cl(0,2)+) keyed by the *relative offset*
    i-j — the learned-connection generalisation of RoPE (RoPE = the abelian, fixed-angle
    special case). Reuses ``cayley_to_unit_quat`` / ``quat_rotate``.
  * **sign[i-j]** is a learned soft sign in (-1,1) per offset per block — the (relaxed)
    Z_2 gauge structure of the signed graph.
  * **gate(i,j)** is a content-dependent spike gate (low-rank sigmoid score) — the
    dynamic routing that a static signed graph lacks; sparsifiable to hard spikes later.

At walk length 1 (this Phase-0 mixer) the holonomy is exactly the per-edge rotor; deeper
walks compose rotors via ``quat_mul`` (Phase 2), mirroring ``structural_actor``'s signed B^L.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from hymeko_neuro.graph.embeddings.cayley_rotor import cayley_to_unit_quat, quat_rotate

from hymeko_lm.config import GateMode


class FiberSpikeRotorMixer(nn.Module):
    """Causal rotor-transported spike-gated token mixer. ``(B,T,d) -> (B,T,d)``, ``d = 3*n_blocks``.

    # Preconditions ``T <= max_seq_len``; ``h.shape[-1] == 3*n_blocks``.
    # Postconditions returns the same shape; output of position ``i`` depends only on ``j <= i``.
    """

    def __init__(self, n_blocks: int, max_seq_len: int, gate_rank: int,
                 gate_mode: GateMode = GateMode.SOFTMAX, spike_k: int | None = None) -> None:
        super().__init__()
        self.n_blocks = n_blocks
        self.d_model = 3 * n_blocks
        self.max_seq_len = max_seq_len
        self.gate_rank = gate_rank
        self.gate_mode = gate_mode
        self.spike_k = spike_k          # None = dense O(T^2); int = hard top-k spike, O(T*k)
        # Rotor + sign keyed by relative offset. Bivector init 0 => identity rotor
        # (the mixer starts as a pure signed gather; it *learns* the connection).
        self.offset_bivec = nn.Parameter(torch.zeros(max_seq_len, n_blocks, 3))
        self.offset_sign = nn.Parameter(torch.ones(max_seq_len, n_blocks))
        # The spike gate is position- AND content-dependent: a learned per-offset bias (which
        # relative offsets the token graph connects) plus a low-rank content score. Without the
        # positional term the gate cannot route by offset on content-free tasks (Phase-0 finding).
        self.offset_gate_bias = nn.Parameter(torch.zeros(max_seq_len))
        self.to_q = nn.Linear(self.d_model, gate_rank)
        self.to_k = nn.Linear(self.d_model, gate_rank)
        # The transported "fiber" is a learned value view of the source (not the raw hidden) — the
        # rotor parallel-transports this value fiber. Without it the mixer cannot form the 2-layer
        # induction circuit (bind-then-recall), so content-addressed memory fails (Phase-0 finding).
        self.to_v = nn.Linear(self.d_model, self.d_model)
        self.out = nn.Linear(self.d_model, self.d_model)

    def _select(self, score: torch.Tensor, causal_bool: torch.Tensor) -> torch.Tensor:
        """Spike-gate selection over past positions ``j`` (dim 2), causal-masked.

        SOFTMAX sharply selects the matching key (content-addressed recall); SIGMOID is the
        diffuse soft-gate ablation, normalised by its own routing mass.
        """
        if self.gate_mode is GateMode.SOFTMAX:
            masked = score.masked_fill(~causal_bool.unsqueeze(0), float("-inf"))
            return F.softmax(masked, dim=2)
        gate = torch.sigmoid(score) * causal_bool.to(score.dtype)
        return gate / gate.sum(dim=2, keepdim=True).clamp_min(1e-6)

    def _dense_mix(self, vb: torch.Tensor, score: torch.Tensor, causal_bool: torch.Tensor,
                   offset_c: torch.Tensor, rotor: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
        """Dense O(T^2) aggregation: transport every causal pair, then gate-weight and sum."""
        b, t = score.shape[0], score.shape[1]
        nb = self.n_blocks
        rotor_ij = rotor[offset_c].unsqueeze(0).expand(b, t, t, nb, 4)
        vb_j = vb.unsqueeze(1).expand(b, t, t, nb, 3)
        transported = quat_rotate(rotor_ij, vb_j)                         # (B,T,T,nb,3)
        gate = self._select(score, causal_bool)                          # (B,T,T)
        weight = gate.unsqueeze(-1) * signs[offset_c].unsqueeze(0)        # (B,T,T,nb)
        return (weight.unsqueeze(-1) * transported).sum(dim=2)           # (B,T,nb,3)

    def _sparse_mix(self, vb: torch.Tensor, score: torch.Tensor, causal_bool: torch.Tensor,
                    rotor: torch.Tensor, signs: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """Hard top-k spike gate: keep only the ``k`` highest-scoring past positions per query, so
        the (expensive) rotor transport runs on ``(B,T,k,...)`` not ``(B,T,T,...)`` — O(T*k). Selection
        is sharp (softmax over the kept scores); future positions get ``-inf`` and never enter."""
        b, t = score.shape[0], score.shape[1]
        k = min(int(self.spike_k or t), t)
        masked = score.masked_fill(~causal_bool.unsqueeze(0), float("-inf"))
        topv, topi = masked.topk(k, dim=2)                               # (B,T,k) scores, source j
        w = F.softmax(topv, dim=2)                                       # (B,T,k); -inf -> 0 weight
        bar = torch.arange(b, device=score.device)[:, None, None]
        v_sel = vb[bar, topi]                                            # (B,T,k,nb,3) gather values
        off_sel = (idx[None, :, None] - topi).clamp(min=0)              # (B,T,k) relative offsets
        transported = quat_rotate(rotor[off_sel], v_sel)                # (B,T,k,nb,3) rotate only k
        weight = w.unsqueeze(-1) * signs[off_sel]                        # (B,T,k,nb)
        return (weight.unsqueeze(-1) * transported).sum(dim=2)          # (B,T,nb,3)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        b, t, d = h.shape
        if t > self.max_seq_len:
            raise ValueError(f"sequence length {t} exceeds max_seq_len {self.max_seq_len}")
        if d != self.d_model:
            raise ValueError(f"expected hidden {self.d_model}; got {d}")
        nb = self.n_blocks
        vb = self.to_v(h).view(b, t, nb, 3)             # learned value fiber per source position
        idx = torch.arange(t, device=h.device)
        offset = idx[:, None] - idx[None, :]            # (T,T) = i-j
        causal_bool = offset >= 0
        offset_c = offset.clamp(min=0)
        rotor = cayley_to_unit_quat(self.offset_bivec[:t])      # (T,nb,4)
        signs = torch.tanh(self.offset_sign[:t])               # (T,nb) soft sign in (-1,1)
        content = torch.einsum("bir,bjr->bij", self.to_q(h), self.to_k(h)) * (self.gate_rank ** -0.5)
        score = content + self.offset_gate_bias[:t][offset_c].unsqueeze(0)   # position + content
        if self.spike_k is not None and self.spike_k < t:
            mixed = self._sparse_mix(vb, score, causal_bool, rotor, signs, idx)
        else:
            mixed = self._dense_mix(vb, score, causal_bool, offset_c, rotor, signs)
        out: torch.Tensor = self.out(mixed.reshape(b, t, d))
        return out
