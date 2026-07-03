"""Parity + behavior tests for the vectorized segment-attention primitive.

The correctness oracle is a naive per-node softmax reference (not the old
by-length-loop code): the vectorized `segment_attention` must match it to fp
tolerance, with and without the sign bias. Plus empty-node and determinism.

Run: ``pytest -p no:randomly hymeko_neuro/tests/test_vectorized_attention.py``
"""
from __future__ import annotations

import math

import torch

from hymeko_neuro.baselines._attention import build_csr, segment_attention


def _random_buckets(n: int, seed: int, signed: bool = False):
    g = torch.Generator().manual_seed(seed)
    buckets, signs = [], []
    for _ in range(n):
        k = int(torch.randint(0, 6, (1,), generator=g).item())
        nb = torch.randint(0, n, (k,), generator=g).tolist()
        buckets.append(nb)
        if signed:
            signs.append([1 if torch.rand(1, generator=g).item() < 0.6 else -1 for _ in nb])
    return (buckets, signs) if signed else buckets


def _naive(Q, K, V, buckets, signs=None, bias_pos=None, bias_neg=None):
    n, H, D = Q.shape
    out = torch.zeros(n, H, D)
    scale = 1.0 / math.sqrt(D)
    for v in range(n):
        nb = buckets[v]
        if not nb:
            continue
        idx = torch.tensor(nb, dtype=torch.long)
        for h in range(H):
            scores = (K[idx, h] @ Q[v, h]) * scale          # (L,)
            if signs is not None:
                sg = torch.tensor(signs[v], dtype=torch.float32)
                scores = scores + torch.where(sg > 0, bias_pos[h], bias_neg[h])
            attn = torch.softmax(scores, dim=0)
            out[v, h] = (attn.unsqueeze(-1) * V[idx, h]).sum(0)
    return out


def _qkv(n=20, H=4, D=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    shape = (n, H, D)
    return (torch.randn(shape, generator=g),
            torch.randn(shape, generator=g),
            torch.randn(shape, generator=g))


def test_parity_no_bias() -> None:
    n = 20
    Q, K, V = _qkv(n)
    buckets = _random_buckets(n, seed=1)
    seg, nbr, _ = build_csr(buckets, torch.device("cpu"))
    got = segment_attention(Q, K, V, seg, nbr, n)
    ref = _naive(Q, K, V, buckets)
    assert torch.allclose(got, ref, atol=1e-5), (got - ref).abs().max().item()


def test_parity_with_sign_bias() -> None:
    n = 20
    Q, K, V = _qkv(n, seed=2)
    buckets, signs = _random_buckets(n, seed=3, signed=True)
    bias_pos = torch.randn(4)
    bias_neg = torch.randn(4)
    seg, nbr, sgn = build_csr(buckets, torch.device("cpu"), signs=signs)
    bias = None
    if sgn.numel() > 0:
        pos = (sgn > 0).float().unsqueeze(-1)
        neg = (sgn < 0).float().unsqueeze(-1)
        bias = pos * bias_pos.unsqueeze(0) + neg * bias_neg.unsqueeze(0)
    got = segment_attention(Q, K, V, seg, nbr, n, bias_per_inc=bias)
    ref = _naive(Q, K, V, buckets, signs=signs, bias_pos=bias_pos, bias_neg=bias_neg)
    assert torch.allclose(got, ref, atol=1e-5), (got - ref).abs().max().item()


def test_empty_neighbours_yield_zero_rows() -> None:
    n = 5
    Q, K, V = _qkv(n, seed=4)
    buckets = [[], [0, 1], [], [2], []]  # nodes 0,2,4 isolated
    seg, nbr, _ = build_csr(buckets, torch.device("cpu"))
    out = segment_attention(Q, K, V, seg, nbr, n)
    for v in (0, 2, 4):
        assert torch.count_nonzero(out[v]) == 0
    assert torch.count_nonzero(out[1]) > 0


def test_all_empty_returns_zeros() -> None:
    n = 4
    Q, K, V = _qkv(n, seed=5)
    seg, nbr, _ = build_csr([[], [], [], []], torch.device("cpu"))
    out = segment_attention(Q, K, V, seg, nbr, n)
    assert torch.count_nonzero(out) == 0


def test_determinism() -> None:
    n = 16
    Q, K, V = _qkv(n, seed=6)
    buckets = _random_buckets(n, seed=7)
    seg, nbr, _ = build_csr(buckets, torch.device("cpu"))
    a = segment_attention(Q, K, V, seg, nbr, n)
    b = segment_attention(Q, K, V, seg, nbr, n)
    assert torch.equal(a, b)
