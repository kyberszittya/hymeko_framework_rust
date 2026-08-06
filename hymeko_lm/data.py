"""Toy data for the Phase-0 smoke: the fixed-lag copy task.

``x[t] = x[t-lag]`` for ``t >= lag`` (the first ``lag`` tokens are random). Predicting the
next token therefore requires *routing* the token at relative offset ``lag`` into the
present — the minimal discriminator for the Fiber-Spike-Rotor sequence mixer. The uniform
(no-routing) baseline loss is ``ln(vocab_size)``; a working mixer drives it toward 0 on the
``t >= lag`` positions.
"""
from __future__ import annotations

import torch


def make_lag_copy_batch(
    batch: int, seq_len: int, vocab_size: int, lag: int, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """One batch of the lag-copy task.

    # Preconditions ``1 <= lag < seq_len``; ``vocab_size >= 2``; ``batch, seq_len >= 1``.
    # Postconditions returns ``(ids, targets)`` each ``(batch, seq_len)`` ``long``; ``targets``
    is ``ids`` shifted left by one (next-token).
    """
    if not (1 <= lag < seq_len):
        raise ValueError(f"need 1 <= lag < seq_len; got lag={lag}, seq_len={seq_len}")
    if vocab_size < 2:
        raise ValueError(f"vocab_size must be >= 2; got {vocab_size}")
    base = torch.randint(0, vocab_size, (batch, seq_len + 1), generator=generator)
    for t in range(lag, seq_len + 1):
        base[:, t] = base[:, t - lag]
    return base[:, :seq_len].contiguous(), base[:, 1:].contiguous()


def make_associative_recall_batch(
    batch: int, n_pairs: int, key_vocab: int, val_vocab: int, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """In-context associative recall (the memory discriminator).

    A stream of distinct (key, value) pairs ``k1 v1 k2 v2 … kK vK`` followed by a query
    ``kq vq``, where ``vq`` is the value that was paired with ``kq``. The key→value map is
    **random per sequence**, so it cannot be memorised in the weights — predicting ``vq``
    at the query requires reading it back from context (content-addressed memory, the
    capability attention provides). Keys live in ``[0, key_vocab)``, values in
    ``[key_vocab, key_vocab+val_vocab)`` (disjoint, so the model can tell them apart).

    # Preconditions ``key_vocab >= n_pairs >= 1``; ``val_vocab >= 1``; ``batch >= 1``.
    # Postconditions returns ``(ids, targets, query_pos)`` with ``ids, targets`` of shape
    ``(batch, 2*n_pairs+1)``; ``targets[:, query_pos]`` is the recalled value ``vq``.
    """
    if not (key_vocab >= n_pairs >= 1) or val_vocab < 1 or batch < 1:
        raise ValueError(f"need key_vocab>=n_pairs>=1, val_vocab>=1, batch>=1; "
                         f"got key_vocab={key_vocab}, n_pairs={n_pairs}, val_vocab={val_vocab}")
    keys = torch.rand(batch, key_vocab, generator=generator).argsort(dim=1)[:, :n_pairs]   # distinct
    values = torch.randint(0, val_vocab, (batch, n_pairs), generator=generator) + key_vocab
    query = torch.randint(0, n_pairs, (batch,), generator=generator)
    rows = torch.arange(batch)
    stream = torch.empty(batch, 2 * n_pairs + 2, dtype=torch.long)
    stream[:, 0 : 2 * n_pairs : 2] = keys
    stream[:, 1 : 2 * n_pairs : 2] = values
    stream[:, 2 * n_pairs] = keys[rows, query]              # query key
    stream[:, 2 * n_pairs + 1] = values[rows, query]        # the value to recall (target)
    return stream[:, :-1].contiguous(), stream[:, 1:].contiguous(), 2 * n_pairs
