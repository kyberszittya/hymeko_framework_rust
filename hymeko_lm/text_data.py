"""Byte-level text corpus for the Phase-1 language A/B.

Loads a UTF-8/byte file, holds out a contiguous validation tail (no leakage — train and val do not
overlap), and serves random fixed-length crops. Vocabulary is the 256 byte values, so the model is
tokenizer-free and the loss is directly bits-per-byte (CE / ln 2).
"""
from __future__ import annotations

from pathlib import Path

import torch

BYTE_VOCAB = 256


class ByteCorpus:
    """A byte tensor split into a train prefix and a held-out validation tail.

    # Preconditions ``path`` exists and is non-empty; ``0 < val_frac < 0.5``.
    # Postconditions ``train`` and ``val`` are disjoint 1-D ``long`` tensors of byte values.
    """

    def __init__(self, path: str | Path, *, val_frac: float = 0.1) -> None:
        raw = Path(path).read_bytes()
        if not raw:
            raise ValueError(f"empty corpus: {path}")
        if not 0.0 < val_frac < 0.5:
            raise ValueError(f"val_frac must be in (0, 0.5); got {val_frac}")
        data = torch.frombuffer(bytearray(raw), dtype=torch.uint8).long()
        n_val = int(len(data) * val_frac)
        self.train = data[:-n_val].contiguous()
        self.val = data[-n_val:].contiguous()

    def batch(self, batch: int, seq_len: int, split: str, generator: torch.Generator
              ) -> tuple[torch.Tensor, torch.Tensor]:
        """Random crops from ``split`` (``"train"`` or ``"val"``).

        # Preconditions ``split in {"train","val"}``; the chosen split is longer than ``seq_len+1``.
        # Postconditions ``(ids, targets)`` each ``(batch, seq_len)`` ``long``; targets are ids shifted +1.
        """
        src = self.train if split == "train" else self.val
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val'; got {split!r}")
        if len(src) <= seq_len + 1:
            raise ValueError(f"{split} split too short ({len(src)}) for seq_len {seq_len}")
        starts = torch.randint(0, len(src) - seq_len - 1, (batch,), generator=generator)
        idx = starts[:, None] + torch.arange(seq_len + 1)[None, :]
        crop = src[idx]
        return crop[:, :-1].contiguous(), crop[:, 1:].contiguous()
