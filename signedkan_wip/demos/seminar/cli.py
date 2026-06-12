"""CLI for the seminar demo program.

Single entry: ``python -m signedkan_wip.demos.seminar <demo> [--device auto]
[--seed 0] [--quick]``. Subparsers are generated from the ``DEMOS`` registry;
the chosen demo runs through ``DemoRunner`` so seed / device / peak-RSS / the
16 GB cap are enforced identically for every demo.
"""
from __future__ import annotations

import argparse
import sys

from .base import DEFAULT_OUT_DIR, DEFAULT_RSS_CAP_GB, DemoRunner
from .demos import DEMOS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m signedkan_wip.demos.seminar",
        description="HyMeKo seminar demos (inference-only).",
    )
    sub = parser.add_subparsers(dest="demo", required=True, metavar="<demo>")
    for name, demo in DEMOS.items():
        sp = sub.add_parser(name, help=demo.help)
        sp.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
        sp.add_argument("--seed", type=int, default=0)
        sp.add_argument("--out-root", default=str(DEFAULT_OUT_DIR),
                        help="root output dir; demo writes to <out-root>/<demo>/.")
        sp.add_argument("--rss-cap-gb", type=float, default=DEFAULT_RSS_CAP_GB)
        sp.add_argument("--quick", action="store_true",
                        help="rehearsal smoke path (fewer repeats / smaller work).")
        demo.add_args(sp)
    return parser


def _force_utf8_console() -> None:
    """Stage machines are Windows (cp125x console); αₖ and ≈ must not crash a
    demo. Re-encode stdout/stderr as UTF-8, replacing unmappable glyphs."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="backslashreplace")


def main(argv: list[str] | None = None) -> int:
    _force_utf8_console()
    args = build_parser().parse_args(argv)
    demo = DEMOS[args.demo]
    runner = DemoRunner(demo, out_root=args.out_root, rss_cap_gb=args.rss_cap_gb)
    try:
        runner.run(args)
    except MemoryError as err:  # cap exceeded — surfaced, not swallowed.
        print(f"[seminar:{args.demo}] ABORT: {err}", file=sys.stderr)
        return 2
    return 0


__all__ = ["build_parser", "main"]
