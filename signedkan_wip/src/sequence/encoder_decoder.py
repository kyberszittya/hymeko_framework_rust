"""Full encoder-decoder model: Sequential HSiKAN + CliffordFIR with
cross-attention dual-path.

Plan: docs/plans/2026-05-17-text-encoder-decoder-contest/.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .clifford import CL2_DIM
from .dual_path_model import _STESign, ste_sign
from .dual_router import DualPathSeqBlock
from .cross_dual_path import CrossDualPathBlock
from .text_embedding import TokenMultivectorEmbedding


class DualPathDecoderBlock(nn.Module):
    """One decoder block: causal self-attn (dual-path) + cross-attn
    (dual-path) + position-wise FF (dual-path, K=1).

    v2 (2026-05-17): accepts an ``n_channels`` kwarg threaded into
    all three sub-blocks. ``n_channels = 1`` recovers v1 behavior
    (with the C=1 byte-identity caveats from the sub-module docs).
    """

    def __init__(self, K: int = 4, n_channels: int = 1) -> None:
        super().__init__()
        self.n_channels = int(n_channels)
        self.self_attn = DualPathSeqBlock(K=K, n_channels=n_channels)
        self.cross_attn = CrossDualPathBlock(n_channels=n_channels)
        self.ff = DualPathSeqBlock(K=1, n_channels=n_channels)

    def forward(
        self,
        x: torch.Tensor, sigma_x: torch.Tensor,
        h_enc: torch.Tensor, sigma_enc: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Causal self-attention (CliffordFIR + HSiKAN are causal by
        # construction).
        x, sigma_x = self.self_attn(x, sigma_x)
        # Cross-attention to the encoder context.
        cross_out = self.cross_attn(x, h_enc, sigma_x, sigma_enc)
        x = x + cross_out
        # Position-wise feed-forward (K=1 dual-path block).
        x, sigma_x = self.ff(x, sigma_x)
        return x, sigma_x


class DualPathEncoderDecoder(nn.Module):
    """Token-level encoder-decoder over Cl(2,0) multivectors.

    Parameters
    ----------
    vocab_size : int
        Shared source/target vocabulary size.
    enc_depth : int, default 3
    dec_depth : int, default 3
    K : int, default 4
        Window length for dual-path blocks.
    max_len : int, default 1024
        Maximum sequence length (for positional encoding cache).
    tie_embeddings : bool, default True
        If True, the output projection shares weights with the input
        token embedding.

    The model exposes a v1 single-channel-per-layer architecture:
    each layer's multivector stream has shape (B, L, 4). The token
    embedding may have multiple channels (squashed by flattening),
    but per-layer processing is single-channel. Multi-channel
    per-layer is a follow-up.

    Forward signatures:
        encode(src_ids)            → (h_enc, sigma_enc)
        decode_step(tgt_ids_so_far, h_enc, sigma_enc) → logits at last position
        forward(src_ids, tgt_ids)  → full logits over tgt_ids (training)
    """

    def __init__(
        self,
        vocab_size: int,
        enc_depth: int = 3,
        dec_depth: int = 3,
        K: int = 4,
        max_len: int = 1024,
        tie_embeddings: bool = True,
        n_channels: int = 1,
    ) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.K = K
        self.n_channels = int(n_channels)
        self.embed = TokenMultivectorEmbedding(
            vocab_size, n_channels=n_channels, max_len=max_len, positional=True,
        )
        # Sign-projection head: takes flattened multivectors (C*4 dims)
        # and projects to a single scalar.
        self.sign_head = nn.Linear(n_channels * CL2_DIM, 1)
        nn.init.normal_(self.sign_head.weight, std=0.5)
        self.enc_blocks = nn.ModuleList([
            DualPathSeqBlock(K=K, n_channels=n_channels) for _ in range(enc_depth)
        ])
        self.dec_blocks = nn.ModuleList([
            DualPathDecoderBlock(K=K, n_channels=n_channels) for _ in range(dec_depth)
        ])
        self.tie_embeddings = bool(tie_embeddings)
        if not tie_embeddings:
            self.output_proj = nn.Linear(n_channels * CL2_DIM, vocab_size)
        else:
            self.output_proj = None  # use embed.embed.weight at forward

    def _flatten_mv(self, x: torch.Tensor) -> torch.Tensor:
        # (B, L, C=1, 4) → (B, L, 4) at C=1; keep (B, L, C, 4) at C>1.
        if x.dim() == 4 and x.shape[2] == 1 and self.n_channels == 1:
            return x.squeeze(2)
        return x

    def _signs_from(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, 4) at C=1 or (B, L, C, 4) at C>1. Returns (B, L)."""
        if x.dim() == 4:
            # (B, L, C, 4) → flatten last two dims → (B, L, C*4)
            B, L, _, _ = x.shape
            x_flat = x.reshape(B, L, -1)
        else:
            x_flat = x
        s = self.sign_head(x_flat).squeeze(-1)    # (B, L)
        return ste_sign(torch.tanh(s))

    def encode(
        self, src_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self._flatten_mv(self.embed(src_ids))     # (B, L_src, 4)
        sigma = self._signs_from(x)
        for block in self.enc_blocks:
            x, sigma = block(x, sigma)
        return x, sigma

    def _decode_stack(
        self,
        tgt_ids: torch.Tensor,
        h_enc: torch.Tensor, sigma_enc: torch.Tensor,
    ) -> torch.Tensor:
        x = self._flatten_mv(self.embed(tgt_ids))    # (B, L_tgt, 4)
        sigma = self._signs_from(x)
        for block in self.dec_blocks:
            x, sigma = block(x, sigma, h_enc, sigma_enc)
        return x

    def _project_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Project to vocab. x is (B, L, 4) at C=1 or (B, L, C, 4) at C>1.
        Returns (B, L, V) logits."""
        if x.dim() == 4:
            B, L, _, _ = x.shape
            x_flat = x.reshape(B, L, -1)  # (B, L, C*4)
        else:
            x_flat = x
        if self.tie_embeddings:
            # embed.embed.weight: (V, C*4). Logits via x_flat @ W.t().
            W = self.embed.embed.weight  # (V, C*4)
            return x_flat @ W.t()
        else:
            return self.output_proj(x_flat)

    def forward(
        self, src_ids: torch.Tensor, tgt_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Teacher-forced training forward. Returns (B, L_tgt, V) logits."""
        h_enc, sigma_enc = self.encode(src_ids)
        x_dec = self._decode_stack(tgt_ids, h_enc, sigma_enc)
        return self._project_logits(x_dec)

    def forward_scheduled_sampling(
        self,
        src_ids: torch.Tensor,
        tgt_ids: torch.Tensor,
        epsilon: float,
        sos_token_id: int = 0,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Scheduled-sampling training forward (Bengio et al. 2015).

        At each decoder position $t$, with probability $\\epsilon$ the
        decoder input is the model's own argmax at position $t-1$; with
        probability $1-\\epsilon$ it is the ground-truth token at $t-1$.

        At $\\epsilon = 0$ this is identical to teacher-forced
        ``forward``. At $\\epsilon = 1$ it is pure free-running
        autoregressive training (no teacher signal).

        The decoder must be applied position-by-position so the model
        can see its own previous output. Cost is $O(L)$ decoder passes
        vs the teacher-forced $O(1)$, so use only when exposure bias
        is the bottleneck.

        Returns
        -------
        logits : (B, L_tgt, V) tensor — same shape as ``forward``.
        """
        if epsilon <= 0.0:
            return self.forward(src_ids, tgt_ids)
        device = src_ids.device
        B, L_tgt = tgt_ids.shape
        h_enc, sigma_enc = self.encode(src_ids)
        # Build the decoder input stream position-by-position.
        # Start with the SOS token, then for t = 1..L_tgt-1 decide
        # per-batch whether to use GT or model's own argmax for the
        # token at position t-1 (which fed into producing logits at t).
        sos = torch.full(
            (B, 1), sos_token_id, dtype=tgt_ids.dtype, device=device,
        )
        dec_in = sos  # (B, 1)
        all_logits = []
        for t in range(L_tgt):
            x_dec = self._decode_stack(dec_in, h_enc, sigma_enc)
            logits_t = self._project_logits(x_dec[:, -1:, :])  # (B, 1, V)
            all_logits.append(logits_t)
            if t == L_tgt - 1:
                break
            # Decide next decoder input per-batch.
            with torch.no_grad():
                model_pred = logits_t.argmax(dim=-1)  # (B, 1)
            gt_next = tgt_ids[:, t : t + 1]  # (B, 1) — GT at position t
            # Bernoulli mask per batch element: 1 = use model_pred.
            if generator is not None:
                u = torch.rand(B, 1, generator=generator, device=device)
            else:
                u = torch.rand(B, 1, device=device)
            use_model = (u < epsilon).to(model_pred.dtype)
            next_in = use_model * model_pred + (1 - use_model) * gt_next
            dec_in = torch.cat([dec_in, next_in], dim=1)
        return torch.cat(all_logits, dim=1)  # (B, L_tgt, V)

    @torch.no_grad()
    def generate(
        self,
        src_ids: torch.Tensor,
        max_new_tokens: int = 16,
        sos_token_id: int = 0,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """Autoregressive generation. Returns (B, max_new_tokens)
        of generated token IDs (greedy decoding)."""
        device = src_ids.device
        B = src_ids.shape[0]
        h_enc, sigma_enc = self.encode(src_ids)
        # Start with the SOS token.
        out = torch.full((B, 1), sos_token_id, dtype=torch.long, device=device)
        for _ in range(max_new_tokens):
            x_dec = self._decode_stack(out, h_enc, sigma_enc)
            logits = self._project_logits(x_dec[:, -1:, :])  # (B, 1, V)
            next_tok = logits.argmax(dim=-1)                  # (B, 1)
            out = torch.cat([out, next_tok], dim=1)
            if eos_token_id is not None:
                if (next_tok == eos_token_id).all():
                    break
        return out[:, 1:]  # drop the SOS column

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
