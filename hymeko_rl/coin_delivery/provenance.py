"""Run provenance — the minimal identity needed to reconstruct a scenario run."""
from __future__ import annotations

import subprocess


def git_commit(short: bool = True) -> str:
    """The current git commit id (best-effort; ``"unknown"`` outside a repo). # Postconditions never raises."""
    args = ["git", "rev-parse", "--short", "HEAD"] if short else ["git", "rev-parse", "HEAD"]
    try:
        out = subprocess.run(args, capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"
