"""Shared contract + runner for the seminar demos.

One ``SeminarDemo`` Protocol, one ``DemoRunner`` that wraps every demo with
the cross-cutting concerns CLAUDE.md mandates for the seminar program:
deterministic seeding, ``--device auto`` resolution, an output directory per
demo, and a peak-RSS + wall-time report with the 16 GB cap asserted on exit.

Adding a demo means writing one object that satisfies ``SeminarDemo`` and
registering it (see ``demos/__init__.py``). No demo re-implements seeding,
device handling, or resource reporting — they are enforced here once, so every
demo registered later inherits the 16 GB contract by construction
(plan.tex §"Risk anticipation").
"""
from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .resources import fmt_bytes, peak_rss_bytes

# Repo root: hymeko_neuro/demos/seminar/base.py -> parents[4].
REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUT_DIR = REPO_ROOT / "demo_out"
DEFAULT_RSS_CAP_GB = 16.0


@dataclass
class DemoContext:
    """Resolved run context handed to every demo's ``run``.

    Invariants: ``out_dir`` exists; ``seed`` has already been applied to
    ``random``/``numpy``/``torch`` global RNGs by the runner.
    """

    device: str
    seed: int
    out_dir: Path
    quick: bool = False


@dataclass
class DemoResult:
    """What a demo returns. ``provenance`` maps each reported number to the
    checkpoint/fixture path it came from (CLAUDE.md honest-reporting rule)."""

    name: str
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: list[Path] = field(default_factory=list)
    provenance: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    # Filled in by DemoRunner:
    wall_s: float | None = None
    peak_rss_bytes: int | None = None


@runtime_checkable
class SeminarDemo(Protocol):
    """A single seminar demo. Stateless: all run state flows through args/ctx."""

    name: str
    help: str

    def add_args(self, parser: argparse.ArgumentParser) -> None: ...

    def run(self, args: argparse.Namespace, ctx: DemoContext) -> DemoResult: ...


def resolve_device(requested: str) -> str:
    """``auto`` -> ``cuda`` when torch reports a CUDA device, else ``cpu``.

    Preconditions: ``requested`` in {auto, cpu, cuda}. Postconditions: returns
    a concrete device string. Falls back to ``cpu`` if torch is unavailable so
    that pure-Rust demos (e.g. HIVE) still run.
    """
    assert requested in {"auto", "cpu", "cuda"}, f"bad device {requested!r}"
    if requested == "cpu":
        return "cpu"
    try:
        import torch
    except ImportError:
        if requested == "cuda":
            raise RuntimeError(
                "device 'cuda' requested but torch is not importable. "
                "Run inference demos under `uv run --group ml`."
            )
        return "cpu"
    has_cuda = torch.cuda.is_available()
    if requested == "cuda":
        if not has_cuda:
            raise RuntimeError("device 'cuda' requested but no CUDA device is available.")
        return "cuda"
    # requested == "auto"
    return "cuda" if has_cuda else "cpu"


def set_global_seed(seed: int) -> None:
    """Seed every RNG a demo might touch. Torch is best-effort (absent for
    pure-Rust demos). Sets deterministic flags so reruns match (CLAUDE.md §3)."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


class DemoRunner:
    """Runs one ``SeminarDemo`` with the shared cross-cutting contract."""

    def __init__(
        self,
        demo: SeminarDemo,
        *,
        out_root: Path = DEFAULT_OUT_DIR,
        rss_cap_gb: float = DEFAULT_RSS_CAP_GB,
    ) -> None:
        self.demo = demo
        self.out_root = Path(out_root)
        self.rss_cap_gb = rss_cap_gb

    def run(self, args: argparse.Namespace) -> DemoResult:
        """Execute the demo end to end.

        Postconditions: ``demo_out/<name>/`` exists; the returned result has
        ``wall_s`` and ``peak_rss_bytes`` populated; raises ``MemoryError`` if
        peak RSS exceeded the cap (CLAUDE.md §4 — abort, do not raise the cap).
        """
        device = resolve_device(getattr(args, "device", "auto"))
        seed = int(getattr(args, "seed", 0))
        set_global_seed(seed)
        out_dir = self.out_root / self.demo.name
        out_dir.mkdir(parents=True, exist_ok=True)
        ctx = DemoContext(
            device=device, seed=seed, out_dir=out_dir,
            quick=bool(getattr(args, "quick", False)),
        )

        print(f"[seminar:{self.demo.name}] device={device} seed={seed} "
              f"out={out_dir}")
        t0 = time.perf_counter()
        result = self.demo.run(args, ctx)
        result.wall_s = time.perf_counter() - t0
        result.peak_rss_bytes = peak_rss_bytes()

        self._report(result)
        self._enforce_cap(result)
        return result

    def _report(self, result: DemoResult) -> None:
        print(f"\n[seminar:{result.name}] --- results ---")
        for k, v in result.metrics.items():
            src = result.provenance.get(k)
            tail = f"   (from {src})" if src else ""
            print(f"  {k}: {v}{tail}")
        for note in result.notes:
            print(f"  · {note}")
        if result.artifacts:
            print("  artifacts:")
            for p in result.artifacts:
                print(f"    {p}")
        print(f"[seminar:{result.name}] wall={result.wall_s:.2f}s  "
              f"peak_rss={fmt_bytes(result.peak_rss_bytes)}")

    def _enforce_cap(self, result: DemoResult) -> None:
        peak = result.peak_rss_bytes
        if peak is None:
            print(f"[seminar:{result.name}] WARNING: peak RSS unavailable on "
                  f"this platform — 16 GB cap NOT enforced.")
            return
        cap = int(self.rss_cap_gb * (1024 ** 3))
        if peak > cap:
            raise MemoryError(
                f"peak RSS {fmt_bytes(peak)} exceeded the "
                f"{self.rss_cap_gb:.0f} GB cap. Abort and redesign "
                f"(CLAUDE.md §4) — do not raise the cap."
            )


__all__ = [
    "DemoContext",
    "DemoResult",
    "SeminarDemo",
    "DemoRunner",
    "resolve_device",
    "set_global_seed",
    "REPO_ROOT",
    "DEFAULT_OUT_DIR",
    "DEFAULT_RSS_CAP_GB",
]
