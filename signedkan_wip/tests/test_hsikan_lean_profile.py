"""Unit tests for HSiKAN lean-profile subprocess env builder."""
from __future__ import annotations

import pytest

from signedkan_wip.experiments.runs.run_hsikan_lean_profile import (
    PROFILE_ENV,
    _run_cell_subprocess,
    build_child_env,
)


def test_profile_env_keys_are_complete() -> None:
    for name, d in PROFILE_ENV.items():
        assert "HSIKAN_TOPK_MODE" in d, name
        assert "HSIKAN_TOPK_K" in d, name
        assert "HSIKAN_USE_PER_VERTEX_ABB" in d, name
        assert "HSIKAN_VERTEX_FILTER" in d, name


def test_build_child_env_scrubs_hsikan_and_applies_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HSIKAN_TOPK_K", "999")
    monkeypatch.setenv("HSIKAN_JUNK_TEST", "x")
    child = build_child_env(
        {"HSIKAN_TOPK_K": "64", "HSIKAN_TOPK_MODE": "per_vertex"}
    )
    assert child["HSIKAN_TOPK_K"] == "64"
    assert child["HSIKAN_TOPK_MODE"] == "per_vertex"
    assert "HSIKAN_JUNK_TEST" not in child


def test_build_child_env_preserves_torch_compile_from_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HSIKAN_TORCH_COMPILE", "1")
    monkeypatch.setenv("HSIKAN_TOPK_K", "77")
    child = build_child_env({"HSIKAN_TOPK_MODE": "per_vertex", "HSIKAN_TOPK_K": "8"})
    assert child["HSIKAN_TORCH_COMPILE"] == "1"
    assert child["HSIKAN_TOPK_K"] == "8"


def test_run_cell_subprocess_injects_hsikan_device(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_run(*_a: object, **kwargs: object) -> object:
        captured["env"] = kwargs["env"]

        class _Proc:
            returncode = 0
            stdout = '{"dataset":"bitcoin_alpha","auc":0.5}\n'
            stderr = ""

        return _Proc()

    monkeypatch.setenv("HOME", "/tmp")
    monkeypatch.setattr(
        "signedkan_wip.experiments.runs.run_hsikan_lean_profile.subprocess.run",
        fake_run,
    )
    _run_cell_subprocess(
        py="python3",
        dataset="bitcoin_alpha",
        seed=0,
        hidden=8,
        n_epochs=1,
        max_k4=100,
        profile_vars=dict(PROFILE_ENV["clean_baseline"]),
        timeout_s=60,
        device="cpu",
    )
    env = captured["env"]
    assert isinstance(env, dict)
    assert env.get("HSIKAN_DEVICE") == "cpu"

    _run_cell_subprocess(
        py="python3",
        dataset="bitcoin_alpha",
        seed=0,
        hidden=8,
        n_epochs=1,
        max_k4=100,
        profile_vars=dict(PROFILE_ENV["clean_baseline"]),
        timeout_s=60,
        device="auto",
    )
    env2 = captured["env"]
    assert "HSIKAN_DEVICE" not in env2
