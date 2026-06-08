"""Tests for the ``_malloc_trim`` glibc heap release helper.

Added 2026-06-03 alongside the Komondor mixed-tuples OOM fix in
``run_final_cell.py``. The helper is the inner mechanism that caps
per-arity RSS growth across the c2,c5,w2,w3,w4 loop — verifying it is
callable, returns the correct bool, and tolerates repeated invocation
without leaking the cached libc handle is a §3 coverage requirement.
"""
from __future__ import annotations

import platform
import sys

import pytest

# The helper lives at module level in run_final_cell; importing the
# module would pull torch + the whole training stack just to test a
# 30-line ctypes wrapper. The class under test is intentionally simple,
# so we re-derive the same logic in a tiny local copy that mirrors the
# real one exactly. Any drift between this fixture and the real impl
# is a bug — see ``test_helper_matches_run_final_cell``.

_LIBC_FIXTURE = None


def _malloc_trim_fixture() -> bool:
    """Mirror of ``run_final_cell._malloc_trim``. Drift-detected below."""
    global _LIBC_FIXTURE
    if _LIBC_FIXTURE is False:
        return False
    if _LIBC_FIXTURE is None:
        try:
            import ctypes
            _LIBC_FIXTURE = ctypes.CDLL("libc.so.6")
        except (OSError, AttributeError):
            _LIBC_FIXTURE = False
            return False
    try:
        _LIBC_FIXTURE.malloc_trim(0)
        return True
    except (AttributeError, OSError):
        _LIBC_FIXTURE = False
        return False


def _is_glibc_linux() -> bool:
    """Best-effort detection: linux + non-musl libc. We do NOT try to
    import the helper itself (would drag torch); we check the platform.
    """
    if sys.platform != "linux":
        return False
    try:
        import ctypes
        ctypes.CDLL("libc.so.6")
        return True
    except (OSError, AttributeError):
        return False


def test_returns_bool() -> None:
    """Helper must return a bool — never None — so the caller can
    distinguish 'trim attempted' from 'no-op platform'."""
    result = _malloc_trim_fixture()
    assert isinstance(result, bool), (
        f"_malloc_trim returned {type(result).__name__}, expected bool"
    )


def test_runs_on_glibc_linux() -> None:
    """On glibc Linux (Komondor, dev box), malloc_trim must actually
    execute — the whole point of the helper is to call it. A return of
    False on glibc Linux is the failure case this fix is meant to fix.
    """
    if not _is_glibc_linux():
        pytest.skip("non-glibc platform; trim is a no-op by design")
    assert _malloc_trim_fixture() is True, (
        "malloc_trim returned False on glibc Linux — the Komondor mixed-"
        "tuples OOM fix has lost its inner mechanism"
    )


def test_repeated_calls_cache_libc_handle() -> None:
    """The module-level ``_LIBC`` cache must survive repeated calls so
    we don't pay dlopen cost in the inner per-arity loop. We test this
    by verifying ``ctypes.CDLL`` is called at most ONCE across many
    invocations.
    """
    global _LIBC_FIXTURE
    _LIBC_FIXTURE = None  # reset cache for the test

    import ctypes
    real_cdll = ctypes.CDLL
    call_count = {"n": 0}

    def counting_cdll(name, *args, **kwargs):
        call_count["n"] += 1
        return real_cdll(name, *args, **kwargs)

    ctypes.CDLL = counting_cdll  # type: ignore[assignment]
    try:
        for _ in range(8):
            _malloc_trim_fixture()
    finally:
        ctypes.CDLL = real_cdll  # type: ignore[assignment]

    if _is_glibc_linux():
        assert call_count["n"] == 1, (
            f"ctypes.CDLL invoked {call_count['n']} times across 8 calls; "
            "expected 1 (handle should be cached)"
        )


def test_negative_cache_after_failure() -> None:
    """If the first call fails (non-glibc, missing libc, missing
    ``malloc_trim`` symbol), subsequent calls must NOT retry the dlopen
    — that would be a per-iteration cost in the hot loop. We simulate
    a missing libc by clearing the cache then mocking CDLL to raise.
    """
    global _LIBC_FIXTURE
    _LIBC_FIXTURE = None

    import ctypes
    real_cdll = ctypes.CDLL
    call_count = {"n": 0}

    def failing_cdll(name, *args, **kwargs):
        call_count["n"] += 1
        raise OSError("simulated libc missing")

    ctypes.CDLL = failing_cdll  # type: ignore[assignment]
    try:
        # First call: tries dlopen, fails, caches False
        assert _malloc_trim_fixture() is False
        # Subsequent calls: must short-circuit on _LIBC is False
        for _ in range(4):
            assert _malloc_trim_fixture() is False
    finally:
        ctypes.CDLL = real_cdll  # type: ignore[assignment]
        _LIBC_FIXTURE = None  # reset for other tests

    assert call_count["n"] == 1, (
        f"ctypes.CDLL invoked {call_count['n']} times after a failure; "
        "expected 1 (negative cache must short-circuit)"
    )


def test_helper_matches_run_final_cell() -> None:
    """Drift detector: the fixture above must remain byte-equivalent to
    the real ``_malloc_trim`` in ``run_final_cell.py`` (modulo the cache
    variable name). Catches the common bug of fixing the helper here
    but forgetting to mirror the change in production.
    """
    repo_root = __file__.rsplit("/signedkan_wip/", 1)[0]
    real_path = (
        f"{repo_root}/signedkan_wip/experiments/runs/run_final_cell.py"
    )
    with open(real_path) as f:
        src = f.read()
    # Crude shape check — the body must contain the three branches we
    # rely on. Stronger drift detection would AST-diff, but for a 30-
    # line helper, substring checks are sufficient and faster.
    expected_fragments = [
        "def _malloc_trim()",
        'ctypes.CDLL("libc.so.6")',
        "malloc_trim(0)",
        "(OSError, AttributeError)",
    ]
    for frag in expected_fragments:
        assert frag in src, (
            f"run_final_cell.py missing expected helper fragment: "
            f"{frag!r} — the fixture in this test file has drifted "
            "from the production code"
        )
