"""Serialize CUDA-heavy SignedKAN drivers on one machine.

Multiple terminals (Optuna chase, SOTA gate, smoke scripts) each spawn
``run_final_cell`` and can exhaust VRAM if they overlap.  This module takes an
**exclusive** ``flock(2)`` on a repo-local lock file so at most one *driver*
process holds the GPU-critical section at a time.

* **Default path:** ``signedkan_wip/experiments/results/.cuda_job_serial.lock``
* **Override path:** env ``HYMEKO_CUDA_JOB_LOCK`` (absolute or relative cwd).
* **Disable:** env ``HYMEKO_CUDA_DISABLE_JOB_LOCK=1`` (tests / power users).
* **Fail fast instead of block:** env ``HYMEKO_CUDA_LOCK_NONBLOCK=1`` — raises
  ``RuntimeError`` if the lock is busy.

Child processes started by the holder (e.g. ``subprocess.run`` for
``run_final_cell``) do **not** acquire this lock; only the parent driver does,
which is enough to prevent two Optuna / gate parents from overlapping.
"""
from __future__ import annotations

import fcntl
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


def _default_lock_path() -> Path:
    # .../signedkan_wip/src/benchmarks/cuda_job_lock.py → signedkan_wip/
    signedkan_wip = Path(__file__).resolve().parent.parent.parent
    return signedkan_wip / "experiments" / "results" / ".cuda_job_serial.lock"


def _disabled() -> bool:
    v = os.environ.get("HYMEKO_CUDA_DISABLE_JOB_LOCK", "")
    return v.strip().lower() in ("1", "true", "yes")


@contextmanager
def cuda_job_lock(*, blocking: bool | None = None) -> Iterator[None]:
    """Exclusive flock on the CUDA job lock file."""
    if _disabled():
        yield
        return

    if blocking is None:
        nb = os.environ.get("HYMEKO_CUDA_LOCK_NONBLOCK", "").strip().lower()
        blocking = nb not in ("1", "true", "yes")

    lock_path = Path(os.environ.get("HYMEKO_CUDA_JOB_LOCK", _default_lock_path()))
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = open(lock_path, "a+", encoding="utf-8")  # noqa: SIM115 — closed in finally
    try:
        flags = fcntl.LOCK_EX
        if not blocking:
            flags |= fcntl.LOCK_NB
        try:
            fcntl.flock(fd.fileno(), flags)
        except BlockingIOError as e:
            raise RuntimeError(
                f"CUDA job lock busy ({lock_path}); another driver holds it "
                "or set HYMEKO_CUDA_DISABLE_JOB_LOCK=1."
            ) from e
        try:
            fd.seek(0, os.SEEK_END)
            fd.write(f"pid={os.getpid()} t={time.time():.3f}\n")
            fd.flush()
        except OSError:
            pass
        yield
    finally:
        try:
            fcntl.flock(fd.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        fd.close()
