"""Smoke test for the hymeko Python wheel.

Exercises the public surface expected in hymeko_py/README.md against the
canonical paper example, checks against numbers in PAPER_INTEGRATION_REPORT.md.
"""
import sys
from pathlib import Path

import hymeko

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "examples" / "paper" / "hymeko_robot.hymeko"

print("=" * 60)
print(f"hymeko module: {hymeko.__file__}")
print(f"attrs: {[a for a in dir(hymeko) if not a.startswith('_')]}")
print("=" * 60)

engine = hymeko.PyHypergraphEngine()
ir = engine.load_file(str(SRC))

print(f"IR:     {ir}")
print(f"  nodes: {ir.node_count}")
print(f"  edges: {ir.edge_count}")
print(f"  arcs:  {ir.arc_count}")

# Paper predicate counts (PAPER_INTEGRATION_REPORT.md §4.6, measured).
expected = {
    "P1  KIND(joint)":                                              4,
    "P2  KIND(joint) AND HASARCREF(+1, INHERITS(link))":            4,
    "P3  KIND(sensor) AND HASARCREF(+1, KIND(joint))":              3,
    "P4  INHERITS(aggregation) AND HASARCREF(-1, ANY)":             8,
    "P5  KIND(constraint) AND HASARCREF(+1, SCOPEDIN(context))":    1,
}

print("-" * 60)
print("Predicate queries (expected → measured):")
all_ok = True
for line, want in expected.items():
    label, pred = line.split(None, 1)
    got = ir.query_count(pred)
    ok = got == want
    all_ok &= ok
    mark = "✓" if ok else "✗"
    print(f"  {mark} {label:4} {pred:<58} {want} → {got}")

print("-" * 60)
urdf = ir.to_urdf("mini_arm")
sdf = ir.to_sdf("mini_arm")
print(f"URDF emitted: {len(urdf)} bytes, {urdf.count('<link')} links, {urdf.count('<joint')} joints")
print(f"SDF  emitted: {len(sdf)} bytes, {sdf.count('<link')} links, {sdf.count('<joint')} joints")

# Tensor export round-trip: star expansion → Arrow → indices/values.
star = engine.compile_star_expansion(ir)
print(f"Star expansion: shape={star.shape} nnz={star.nnz}")

print("=" * 60)
print("PASS" if all_ok else "FAIL")
sys.exit(0 if all_ok else 1)
