#!/usr/bin/env python3
"""Run the Friedler/Orosz/Pimentel P-graph book examples through the HyMeKo
engine (MSG / ABB) and check the canonical results.

Pure stdlib (subprocess + json) — no torch/numpy. Shells out to the Rust
``hymeko_pgraph_dump`` binary, parses its JSON, and compares MSG unit counts +
ABB optima against the values published in the book.

Usage (from repo root or anywhere):
    python scripts/pgraph/run_examples.py            # book conformance table
    python scripts/pgraph/run_examples.py --regimes  # also show regime effects
    python scripts/pgraph/run_examples.py --build     # force-rebuild the binary

Exit code 0 iff every example matches its expected canonical value (so this is
CI-able), 1 on any mismatch, 2 on a build/run error.

Note: the 3465 / 19 *solution-structure* counts (Examples 3.3 / 3.2) come from
the decision-mapping SSG and are checked by the Rust suite
(`cargo test -p hymeko_pgraph --test ssg_decision_mapping` / `book_validation`),
not here — the CLI's brute SSG cannot enumerate 2^29 subsets.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "pgraph"


def dump_bin() -> Path:
    # Use the debug build (what `build()` produces). Avoid preferring a possibly
    # stale `release/` binary that may predate newer CLI flags like `--regime`.
    exe = "hymeko_pgraph_dump" + (".exe" if sys.platform == "win32" else "")
    return REPO / "target" / "debug" / exe


def build() -> None:
    print("building hymeko_pgraph_dump (incremental) ...", flush=True)
    rc = subprocess.run(
        ["cargo", "build", "-p", "hymeko_pgraph", "--bin", "hymeko_pgraph_dump"],
        cwd=REPO,
    ).returncode
    if rc != 0:
        sys.exit(2)


def run(path: Path, algorithm: str = "abb", regime: str | None = None) -> dict:
    cmd = [str(dump_bin()), str(path), "--algorithm", algorithm]
    if regime:
        cmd += ["--regime", regime]
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    if proc.returncode not in (0, 2) or not proc.stdout.strip():
        print(f"  ERROR running {path.name}: rc={proc.returncode}\n{proc.stderr[-500:]}")
        sys.exit(2)
    return json.loads(proc.stdout)


# Canonical expectations (book values; see docs/plans/2026-05-27-msg-abb-verification).
# (relative path, expected MSG units, expected ABB cost, note)
EXAMPLES = [
    ("Chapter3/example3_2.hymeko", 7, 0.0, "Ex 3.2: maximal 7 (SSG=19 in tests)"),
    ("Chapter4/example4_1.hymeko", 7, 13.0, "Ex 4.1: maximal 7; ABB {u2,u4,u8}=13"),
    ("Chapter4/example4_3.hymeko", 29, 0.0, "Ex 3.3: maximal 29 (SSG=3465 in tests)"),
    ("Chapter5/example5_1.hymeko", 6, 0.0, "Ex 5.1: maximal 6 (structural)"),
    ("Chapter6/example6_1.hymeko", 7, 9.0, "Ex 6.1: maximal 7; ABB {O2,O5,O7}=9"),
    ("book/example14_1.hymeko", 12, 16.0, "Ex 14.1: maximal 12; ABB {u1,u4,u8,u11}=16"),
    ("hda.hymeko", 3, 350.0, "HDA: maximal 3; ABB {Mixer,Reactor}=350"),
    ("methanol_synthesis.hymeko", 8, 2940.0, "methanol: maximal 8; ABB scalar=2940"),
]


def conformance() -> int:
    print(f"\nCanonical book conformance (engine = {dump_bin().name})\n" + "-" * 78)
    print(f"{'example':<28}{'MSG':>5}{'exp':>5}{'ABB cost':>11}{'exp':>9}  ok")
    failures = 0
    for rel, exp_msg, exp_cost, _note in EXAMPLES:
        path = DATA / rel
        if not path.exists():
            print(f"{rel:<28}  (missing fixture — skipped)")
            continue
        d = run(path)
        msg = len(d.get("msg_units") or [])
        abb = d.get("abb")
        cost = abb["cost"] if abb else float("nan")
        ok = (msg == exp_msg) and (abb is not None) and abs(cost - exp_cost) < 1e-6
        failures += 0 if ok else 1
        print(f"{rel:<28}{msg:>5}{exp_msg:>5}{cost:>11.1f}{exp_cost:>9.1f}  {'OK' if ok else 'XX'}")
    print("-" * 78)
    print("ALL CANONICAL VALUES MATCH" if failures == 0 else f"{failures} MISMATCH(ES)")
    return 1 if failures else 0


def regime_demo() -> None:
    """Show how the regimes change the selected architecture on the HSiKAN
    byproduct fixture (canonical vs no-excess vs cost-dominance vs composite)."""
    fix = DATA.parent / "hsikan" / "sweep_msg_byproduct_dominated.hymeko"
    if not fix.exists():
        return
    print("\nRegime effects on data/hsikan/sweep_msg_byproduct_dominated.hymeko\n" + "-" * 78)
    print(f"{'--regime':<30}{'MSG':>5}  ABB optimum")
    for spec in ("canonical", "no-excess", "cost-dominance", "cost-dominance+no-excess"):
        d = run(fix, "abb", regime=spec)
        abb = d.get("abb")
        units = sorted(abb["units"]) if abb else None
        cost = abb["cost"] if abb else None
        print(f"{spec:<30}{len(d.get('msg_units') or []):>5}  {units} cost={cost}")


def main() -> int:
    args = sys.argv[1:]
    # Always (incrementally) build so the binary matches the current source —
    # near-instant when up to date, and avoids stale-binary surprises.
    build()
    rc = conformance()
    if "--regimes" in args:
        regime_demo()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
