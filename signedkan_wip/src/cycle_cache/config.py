"""Auto-split from cycle_cache.py 2026-05-11 (CLAUDE.md §6.5 #4)."""
from __future__ import annotations
import hashlib
import json
import os
import pathlib
from dataclasses import dataclass
from typing import Any
import numpy as np

from ..runtime_config import get_runtime

def _import_n_tuples():
    from .. import n_tuples
    return n_tuples


def _import_walks():
    from . import walks
    return walks


# ─── Cache directory ────────────────────────────────────────────────




def _cache_dir() -> pathlib.Path:
    base = os.environ.get(
        "HYMEKO_CYCLE_CACHE_DIR",
        str(pathlib.Path.home() / ".cache" / "hymeko" / "cycles_v1"),
    )
    p = pathlib.Path(base)
    p.mkdir(parents=True, exist_ok=True)
    return p


def cache_enabled() -> bool:
    return get_runtime().cycle_cache.enabled


def _enum_seed() -> int:
    """Sentinel seed used for enumeration sampling.  Decoupled from the
    model seed so caching can amortise across all seeds in an
    ablation."""
    return get_runtime().cycle_cache.enum_seed


def _topk_fingerprint() -> dict[str, str]:
    """Cache-key fingerprint over every TopK knob that affects which
    cycles are enumerated. Delegates to `TopKConfig.fingerprint()` so
    adding a new knob is a single-site change.

    Critical correctness rule: the fingerprint MUST include every
    field that can change the cycle output. Adding a knob to the
    enumerator without extending `TopKConfig` is a silent
    correctness bug — see `feedback_cycle_cache_fingerprint.md`.
    """
    return get_runtime().topk.fingerprint()


