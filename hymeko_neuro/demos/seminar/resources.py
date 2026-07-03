"""Cross-platform peak-RSS measurement, dependency-free.

CLAUDE.md §4 requires every demo to report **peak resident set size** and
assert it stays under the 16 GB cap. ``resource.getrusage`` is POSIX-only, so
this module provides one ``peak_rss_bytes()`` that uses the Win32
``GetProcessMemoryInfo`` peak working set on Windows and ``ru_maxrss`` on
POSIX. No third-party dependency (adding ``psutil`` would be a CORE.YAML
dependency decision, CLAUDE.md §1).

Preconditions: none.
Postconditions: returns peak resident bytes for the current process as a
non-negative int, or ``None`` if the platform exposes no peak counter.
"""
from __future__ import annotations

import sys


def _peak_rss_windows() -> int | None:
    """Win32 PeakWorkingSetSize for the current process, in bytes."""
    import ctypes
    from ctypes import wintypes

    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    kernel32 = ctypes.windll.kernel32
    psapi = ctypes.windll.psapi
    # Without explicit signatures ctypes truncates the 64-bit HANDLE to 32-bit,
    # the call gets an invalid handle, and returns 0 (peak = 0).
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), wintypes.DWORD,
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

    counters = PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
    ok = psapi.GetProcessMemoryInfo(
        kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb,
    )
    if not ok:
        return None
    return int(counters.PeakWorkingSetSize)


def _peak_rss_posix() -> int | None:
    """``ru_maxrss`` normalised to bytes (KiB on Linux, bytes on macOS)."""
    import resource  # POSIX-only; this function is never called on Windows.

    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # type: ignore[attr-defined]
    # Linux reports KiB; macOS/BSD report bytes.
    factor = 1 if sys.platform == "darwin" else 1024
    return int(raw) * factor


def peak_rss_bytes() -> int | None:
    """Peak resident set size of the current process, in bytes.

    Returns ``None`` only on a platform with no peak counter — callers treat
    that as "cap not enforceable" and must say so, not silently pass.
    """
    if sys.platform.startswith("win"):
        return _peak_rss_windows()
    return _peak_rss_posix()


def fmt_bytes(n: int | None) -> str:
    """Human-readable size; ``"unknown"`` for ``None``."""
    if n is None:
        return "unknown"
    gb = n / (1024 ** 3)
    if gb >= 1.0:
        return f"{gb:.2f} GB"
    return f"{n / (1024 ** 2):.1f} MB"


__all__ = ["peak_rss_bytes", "fmt_bytes"]
