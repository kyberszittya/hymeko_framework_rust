"""Tests for ``HSIKAN_DEVICE`` resolution in ``run_final_cell``."""
from __future__ import annotations

import pytest

from signedkan_wip.src.hsikan_device_env import resolve_hsikan_device


def test_resolve_hsikan_device_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HSIKAN_DEVICE", "cpu")
    assert str(resolve_hsikan_device()) == "cpu"


def test_resolve_hsikan_device_auto_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HSIKAN_DEVICE", raising=False)
    d = resolve_hsikan_device()
    assert d.type in ("cpu", "cuda")


def test_resolve_hsikan_device_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HSIKAN_DEVICE", "mps")
    with pytest.raises(ValueError, match="HSIKAN_DEVICE"):
        resolve_hsikan_device()
