"""Masked-language-model pretraining on the IMDB unsupervised split.

The aclImdb tarball ships ``train/unsup/*.txt`` — 50 k unlabeled
movie reviews, ~10-15 M tokens — designed by Maas et al. (2011)
for unsupervised pretraining followed by labeled fine-tuning.

This module implements BERT-recipe MLM pretraining at our scale:
  * 15% of non-padding tokens are masked per sequence.
  * Of the masked tokens, 80% become a ``<mask>`` sentinel, 10%
    are replaced with a random vocab token, 10% are left unchanged.
  * Loss is cross-entropy over the masked positions only.

Compatible with both :class:`text_classifier.IMDBClassifier`
(HSiKAN Sequential) and
:class:`iso_param_transformer.IMDBTransformerBaseline` via duck-typed
``encode()`` + tied embedding projection.

Plan: docs/plans/2026-05-17-sequential-hsikan-imdb-benchmark/
(MLM pretrain extension, 2026-05-18).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .imdb_dataset import (
    PAD_ID, UNK_ID, _iter_split, build_imdb_vocab, download_imdb,
    encode_tokens,
)


# We reuse UNK_ID as the MLM mask sentinel to keep the vocabulary
# compatible across from-scratch and pretrain-then-fine-tune runs
# (no vocab-pinning required at fine-tune time). The cost is that
# the model can't distinguish "masked" from "out-of-vocab" at
# pretrain time — well-precedented in early MLM literature.
MASK_ID = UNK_ID


def materialise_unsup(
    aclimdb_dir: Path,
    vocab: dict[str, int],
    L_max: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Read every file in ``train/unsup/`` and encode to (ids, mask).

    The unsup split is positioned under ``train/`` but has neither
    pos nor neg labels — instead a single ``unsup/`` subdir with
    50 000 .txt reviews.
    """
    from .imdb_dataset import _SPLIT_RE, _HTML_RE
    import re
    unsup_dir = aclimdb_dir / "train" / "unsup"
    if not unsup_dir.is_dir():
        raise FileNotFoundError(f"unsup split missing: {unsup_dir}")
    ids_rows: list[np.ndarray] = []
    mask_rows: list[np.ndarray] = []
    for p in sorted(unsup_dir.iterdir()):
        if p.suffix != ".txt":
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        text = re.sub(r"<[^>]+>", " ", text).lower().strip()
        tokens = [t for t in re.split(r"\s+", text) if t]
        ids, mask = encode_tokens(tokens, vocab, L_max=L_max)
        ids_rows.append(ids)
        mask_rows.append(mask)
    return (
        np.stack(ids_rows, axis=0),
        np.stack(mask_rows, axis=0),
    )


def apply_mlm_masking(
    ids: torch.Tensor,
    mask: torch.Tensor,
    vocab_size: int,
    mask_prob: float = 0.15,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply BERT-recipe MLM masking in-place over a fresh clone.

    Parameters
    ----------
    ids : (B, L) int64 token ids.
    mask : (B, L) bool — True for real positions (eligible for masking).
    vocab_size : int — for random-token replacement.
    mask_prob : float — fraction of REAL positions to mark for MLM.

    Returns
    -------
    masked_ids : (B, L) int64 — input to the model (with masking applied).
    labels     : (B, L) int64 — original token at masked positions,
                                ``-100`` elsewhere (PyTorch CE ignore).
    masked_pos : (B, L) bool — True at positions whose loss contributes.
    """
    device = ids.device
    rand_shape = ids.shape
    # Per-token random in [0,1); positions with r < mask_prob are
    # candidates, but the candidacy is only valid where mask==True.
    rand = torch.rand(rand_shape, device=device, generator=generator)
    masked_pos = (rand < mask_prob) & mask
    labels = torch.where(masked_pos, ids, torch.full_like(ids, -100))
    # Of the masked positions: 80% -> MASK_ID, 10% -> random vocab,
    # 10% -> unchanged (which still contributes loss).
    masked_ids = ids.clone()
    rand2 = torch.rand(rand_shape, device=device, generator=generator)
    do_mask = masked_pos & (rand2 < 0.80)
    do_rand = masked_pos & (rand2 >= 0.80) & (rand2 < 0.90)
    masked_ids = torch.where(
        do_mask, torch.full_like(ids, MASK_ID), masked_ids,
    )
    if do_rand.any():
        rand_tokens = torch.randint(
            2, vocab_size, ids.shape, device=device, generator=generator,
        )  # skip the 2 reserved IDs (pad, unk)
        masked_ids = torch.where(do_rand, rand_tokens, masked_ids)
    return masked_ids, labels, masked_pos


class MLMHead(nn.Module):
    """Project encoder hidden states to vocab logits via tied embeddings.

    Works with both ``IMDBClassifier`` (encoder output is the v2
    multichannel Cl(2,0) stream flattened to ``n_channels * 4``) and
    ``IMDBTransformerBaseline`` (encoder output is ``d_model``). The
    tied-embedding projection ``logits = h @ W^T`` uses whichever
    embedding the upstream model owns.
    """

    def __init__(self, embedding: nn.Embedding) -> None:
        super().__init__()
        self.embedding = embedding  # share weights, no copy

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h : (B, L, d_emb) — d_emb must equal embedding.weight.shape[1]
        W = self.embedding.weight  # (V, d_emb)
        return h @ W.t()            # (B, L, V)


def _encode_hsikan(model, ids: torch.Tensor) -> torch.Tensor:
    """Run HSiKAN IMDBClassifier's encoder + flatten the multivector dim
    to a single (B, L, d_emb) tensor matching the embedding row width."""
    h, _sigma = model.encoder.encode(ids)
    if h.dim() == 4:
        B, L, C, D = h.shape
        return h.reshape(B, L, C * D)
    return h


def _encode_transformer(model, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return model.encode(ids, mask=mask)


def pretrain_mlm(
    model: nn.Module,
    embedding: nn.Embedding,
    unsup_ids: torch.Tensor,
    unsup_mask: torch.Tensor,
    vocab_size: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    mask_prob: float = 0.15,
    seed: int = 0,
    log_every: int = 100,
    is_hsikan: bool = True,
) -> dict:
    """Run MLM pretraining; mutates ``model`` in place.

    Returns a dict with per-epoch loss curve and wall time.
    """
    model.train()
    head = MLMHead(embedding).to(device)
    # Optimizer covers BOTH encoder params and the (tied) embedding
    # only once — the MLMHead has no own params.
    opt = torch.optim.AdamW(
        list(model.parameters()), lr=lr, weight_decay=weight_decay,
    )
    gen = torch.Generator(device=device).manual_seed(seed)
    n = unsup_ids.shape[0]
    t0 = time.perf_counter()
    losses_per_epoch: list[float] = []
    for ep in range(epochs):
        perm = torch.randperm(n, device=device, generator=gen)
        ep_losses: list[float] = []
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            ids_b = unsup_ids[idx]
            mask_b = unsup_mask[idx]
            masked_ids, labels, masked_pos = apply_mlm_masking(
                ids_b, mask_b, vocab_size=vocab_size,
                mask_prob=mask_prob, generator=gen,
            )
            if is_hsikan:
                h = _encode_hsikan(model, masked_ids)
            else:
                h = _encode_transformer(model, masked_ids, mask=mask_b)
            logits = head(h)              # (B, L, V)
            loss = F.cross_entropy(
                logits.reshape(-1, vocab_size),
                labels.reshape(-1),
                ignore_index=-100,
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            ep_losses.append(float(loss.detach()))
        mean_l = sum(ep_losses) / max(1, len(ep_losses))
        losses_per_epoch.append(mean_l)
        elapsed = time.perf_counter() - t0
        print(f"  [pretrain ep {ep:3d}] loss={mean_l:.4f}  "
              f"n_batches={len(ep_losses)}  elapsed={elapsed:.0f}s",
              flush=True)
    return {
        "losses_per_epoch": losses_per_epoch,
        "wall_s": time.perf_counter() - t0,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "mask_prob": mask_prob,
    }
