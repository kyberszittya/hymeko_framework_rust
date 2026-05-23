"""Tests for attention / alpha auxiliary entropy wiring."""

from __future__ import annotations

import torch
import torch.nn as nn

from signedkan_wip.experiments.runs.run_final_cell import (
    _aux_entropy_attention_alpha,
    _shannon_entropy_discrete,
)


def test_shannon_entropy_two_class_uniform():
    p = torch.tensor([0.5, 0.5], dtype=torch.float64)
    h = _shannon_entropy_discrete(p)
    assert abs(h.item() - 0.69314718056) < 1e-5


def test_aux_alpha_entropy_only():
    class Fake:
        def __init__(self) -> None:
            self.p = nn.Parameter(torch.tensor([0.2, 0.8], dtype=torch.float32))

        def parameters(self):
            yield self.p

        def alpha(self) -> torch.Tensor:
            return torch.softmax(self.p, dim=0)

    m = Fake()
    aux = _aux_entropy_attention_alpha(
        m,
        alpha_entropy_lambda=1.0,
        attn_entropy_lambda=0.0,
    )
    assert aux.ndim == 0
    assert aux.item() < 0.0
    aux.backward()
    assert m.p.grad is not None


def test_runtime_parses_aux_entropy_env(monkeypatch):
    monkeypatch.setenv("HSIKAN_ALPHA_ENTROPY_LAMBDA", "0.02")
    monkeypatch.setenv("HSIKAN_ATTN_ENTROPY_LAMBDA", "0.03")
    from signedkan_wip.src.runtime_config import get_runtime

    t = get_runtime().training
    assert t.alpha_entropy_lambda == 0.02
    assert t.attn_entropy_lambda == 0.03
