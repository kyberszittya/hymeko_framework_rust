"""Phase 1 synthetic copy-task generator for the text encoder-decoder.

Toy seq2seq smoke benchmark: input is a length-L random sequence
over vocab {1, …, V-1}; target is the same sequence shifted right
by ``shift`` positions, with PAD (0) in the leading positions.

Designed to be solvable trivially by a Transformer-tiny baseline,
so failing this smoke implies our enc-dec architecture is broken.

Plan: docs/plans/2026-05-17-text-encoder-decoder-contest/.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class CopyTaskConfig:
    n_samples: int = 1024
    L_src: int = 16
    vocab_size: int = 20   # 0 = PAD, 1 = SOS, 2..V-1 = actual tokens
    shift: int = 3
    seed: int = 0
    sos_token: int = 1
    pad_token: int = 0


def make_copy_dataset(
    cfg: CopyTaskConfig = CopyTaskConfig(),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate ``n_samples`` (src, tgt) pairs.

    src: (N, L_src) tokens drawn uniformly from {2, …, V-1}.
    tgt: (N, L_src) "shifted copy": tgt[t] = src[t - shift] for
         t >= shift, else PAD.

    Returns
    -------
    src : LongTensor (N, L_src)
    tgt : LongTensor (N, L_src)
    """
    gen = torch.Generator().manual_seed(cfg.seed)
    src = torch.randint(
        low=2, high=cfg.vocab_size, size=(cfg.n_samples, cfg.L_src),
        generator=gen,
    )
    tgt = torch.full_like(src, fill_value=cfg.pad_token)
    if cfg.shift > 0:
        tgt[:, cfg.shift:] = src[:, : cfg.L_src - cfg.shift]
    else:
        tgt = src.clone()
    return src, tgt


def make_decoder_inputs(
    tgt: torch.Tensor, sos_token: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert target tokens into (decoder-input, decoder-target) for
    teacher-forced training.

    decoder_input: [SOS, tgt[0], tgt[1], …, tgt[L-2]]
    decoder_target: [tgt[0], tgt[1], …, tgt[L-1]]
    """
    N, L = tgt.shape
    sos = torch.full((N, 1), sos_token, dtype=tgt.dtype, device=tgt.device)
    decoder_input = torch.cat([sos, tgt[:, :-1]], dim=1)
    decoder_target = tgt
    return decoder_input, decoder_target
