"""HyMeKo seminar demo program — single inference-only entry point.

Run:  ``python -m signedkan_wip.demos.seminar <demo> --device auto --seed 0``

One package, one CLI, one demo per mode (CLAUDE.md §6.5 #13). See
``SEMINAR_DEMO_OUTLINE.md`` and ``signedkan_wip/demos/SEMINAR_DEMOS.md``.
"""
from __future__ import annotations

from .base import (
    DemoContext, DemoResult, DemoRunner, SeminarDemo,
)
from .cli import build_parser, main
from .demos import DEMOS

__all__ = [
    "DemoContext", "DemoResult", "DemoRunner", "SeminarDemo",
    "DEMOS", "build_parser", "main",
]
