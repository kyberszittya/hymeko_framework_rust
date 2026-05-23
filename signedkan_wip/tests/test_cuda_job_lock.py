"""Tests for ``benchmarks.cuda_job_lock`` (no GPU work)."""

from __future__ import annotations

import fcntl

import pytest

from signedkan_wip.src.benchmarks.cuda_job_lock import cuda_job_lock


def test_cuda_job_lock_respects_disable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HYMEKO_CUDA_DISABLE_JOB_LOCK", "1")
    with cuda_job_lock():
        pass


def test_cuda_job_lock_uses_custom_path(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock = tmp_path / "only.lock"
    monkeypatch.setenv("HYMEKO_CUDA_JOB_LOCK", str(lock))
    with cuda_job_lock():
        assert lock.is_file()


def test_cuda_job_lock_nonblock_raises_when_busy(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock = tmp_path / "busy.lock"
    monkeypatch.setenv("HYMEKO_CUDA_JOB_LOCK", str(lock))
    monkeypatch.delenv("HYMEKO_CUDA_DISABLE_JOB_LOCK", raising=False)
    hold = open(lock, "a+", encoding="utf-8")  # noqa: SIM115
    try:
        fcntl.flock(hold.fileno(), fcntl.LOCK_EX)
        monkeypatch.setenv("HYMEKO_CUDA_LOCK_NONBLOCK", "1")
        with pytest.raises(RuntimeError, match="CUDA job lock busy"):
            with cuda_job_lock():
                pass
    finally:
        fcntl.flock(hold.fileno(), fcntl.LOCK_UN)
        hold.close()
