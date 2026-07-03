"""Demo registry: name -> SeminarDemo instance.

Adding a demo = import its class and add one entry here. The CLI builds its
subparsers from this dict (no per-demo if/elif ladder, CLAUDE.md §6.5 #1).
"""
from __future__ import annotations

from ..base import SeminarDemo
from .balance import BalanceDemo
from .hive import HiveDemo
from .latency import LatencyDemo
from .link import LinkDemo

DEMOS: dict[str, SeminarDemo] = {
    d.name: d for d in (
        BalanceDemo(),
        LinkDemo(),
        HiveDemo(),
        LatencyDemo(),
    )
}

__all__ = ["DEMOS"]
