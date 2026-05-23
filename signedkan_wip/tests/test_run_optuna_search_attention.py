"""``run_optuna_search._attention_kind_candidates`` (no Optuna runs)."""

from __future__ import annotations

import os

import pytest

import signedkan_wip.experiments.runs.run_optuna_search as ros


def test_attention_kinds_explicit_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HSIKAN_OPTUNA_ATTENTION_KINDS", "none,dot")
    monkeypatch.delenv("HSIKAN_OPTUNA_SKIP_EXPENSIVE_ATTENTION", raising=False)
    assert ros._attention_kind_candidates() == ["none", "dot"]


def test_attention_kinds_skip_expensive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HSIKAN_OPTUNA_SKIP_EXPENSIVE_ATTENTION", "1")
    monkeypatch.delenv("HSIKAN_OPTUNA_ATTENTION_KINDS", raising=False)
    assert ros._attention_kind_candidates() == ["none"]


def test_attention_kinds_default_full_when_no_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HSIKAN_OPTUNA_ATTENTION_KINDS", raising=False)
    monkeypatch.delenv("HSIKAN_OPTUNA_SKIP_EXPENSIVE_ATTENTION", raising=False)
    monkeypatch.setenv("HSIKAN_OPTUNA_ATTENTION_VRAM_GIB_MIN", "0")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA for VRAM threshold branch")
    assert ros._attention_kind_candidates() == ["none", "dot", "quaternion"]


def test_attention_kinds_huge_threshold_forces_none_on_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    monkeypatch.delenv("HSIKAN_OPTUNA_ATTENTION_KINDS", raising=False)
    monkeypatch.delenv("HSIKAN_OPTUNA_SKIP_EXPENSIVE_ATTENTION", raising=False)
    monkeypatch.setenv("HSIKAN_OPTUNA_ATTENTION_VRAM_GIB_MIN", "999999")
    assert ros._attention_kind_candidates() == ["none"]
