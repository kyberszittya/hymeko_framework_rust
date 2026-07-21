"""Isolated state-identity + run provenance for the coin-delivery corpus (RECOVERY-BASELINE-0 §3).

``git_commit`` lives here (not in a sibling ``provenance.py`` module) because a package directory shadows a
same-named module file on import — the file form was dead/unreachable and broke the runners' import."""
from __future__ import annotations

import subprocess


def git_commit(short: bool = True) -> str:
    """The current git commit id (best-effort; ``"unknown"`` outside a repo). # Postconditions never raises."""
    args = ["git", "rev-parse", "--short", "HEAD"] if short else ["git", "rev-parse", "HEAD"]
    try:
        out = subprocess.run(args, capture_output=True, text=True, timeout=5)  # noqa: S603,S607 (trusted argv)
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"
