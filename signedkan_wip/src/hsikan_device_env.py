"""HSiKAN training device selection (no sklearn / graph imports)."""
from __future__ import annotations

import os

import torch


def resolve_hsikan_device() -> torch.device:
    """Select ``torch.device`` from ``HSIKAN_DEVICE`` env.

    Values: ``auto`` (default), ``cpu``, ``cuda``. Unknown values raise
    ``ValueError`` so typos fail fast in batch scripts.
    """
    raw = os.environ.get("HSIKAN_DEVICE", "auto").strip().lower()
    if raw in ("", "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if raw == "cpu":
        return torch.device("cpu")
    if raw == "cuda":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    raise ValueError(
        "HSIKAN_DEVICE must be one of: auto, cpu, cuda "
        f"(got {os.environ.get('HSIKAN_DEVICE')!r})"
    )
