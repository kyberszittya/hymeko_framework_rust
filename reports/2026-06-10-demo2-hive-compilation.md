# Seminar demo program — build-item 3: Demo 2 (HIVE compilation)

**Date:** 2026-06-10 · **Plan:** `docs/plans/2026-06-10-seminar-demo-program/`
· **Spec:** `hymeko_neuro/demos/SEMINAR_DEMOS.md` §2

## Summary
Added the `hive` demo to the seminar package: surface `.hymeko` → IR →
star COO / clique matrix encodings + the **canonicalisation** punchline. Runs as
`python -m hymeko_neuro.demos.seminar hive --src <file>`. Pure transform, no
model, no torch (0.03 s compute). Reuses the `canonical_hash` getter added in
build-item 2.

The canonicalisation proof is built on a controlled signed chain via
`parse_dsl` and asserts: declaration-order permutations (node order, edge order)
hash **equal**; a one-edge structural change hashes **different** → emits
`canonicalisation: PASS`.

## Files touched
| LOC | File | |
|---:|---|---|
| 200 | `hymeko_neuro/demos/seminar/demos/hive.py` | new — `HiveDemo` |
| 79 | `hymeko_neuro/tests/test_seminar_hive.py` | new — tests |
| +2 | `hymeko_neuro/demos/seminar/demos/__init__.py` | register `HiveDemo` |

## CORE.YAML items touched
**None.** Reuses the engine via `hymeko` and the (build-item-2, non-core)
`canonical_hash` getter.

## Test results
`PYTHONPATH=. .venv/Scripts/python.exe -m pytest -p no:randomly` (engine live):

| Layer | Tests | Result |
|---|---|---|
| Integration — registry, fano counts, canonicalisation PASS, report JSON | 4 | pass |
| **All new seminar tests, one run** (harness 13 · link 3 · star-export 5 · hive 4) | **25** | **pass** |

## Acceptance gate
SEMINAR_DEMOS §2: declaration-order-permuted inputs → identical fingerprint; a
one-edge change → different fingerprint. **PASS** — `order_invariant=True`,
`structural_sensitive=True`. Fano compile: 8 nodes, 7 edges, star COO nnz 42,
clique nnz 42, `canonical_hash=blake3:b88715ed…`.

## Honest framing (corrected vs the original spec)
The spec said "isomorphic-but-differently-written → same hash." Verified
behaviour (build-item 2 finding) is narrower: the hash is invariant to
**declaration order**, not to relabeling or within-edge member order. The demo
claims and demonstrates exactly the declaration-order property and says so in
its output; the stronger isomorphism-invariance is a deferred plan
(`docs/plans/2026-06-10-canonical-hash-iso-invariance/`). The star-vs-clique
O(d) vs O(d²) gap is stated as a per-edge formula and shown visually in the
star-expansion viewer (build-item 2), not over-claimed here.

## Static analysis
- `ruff check`: clean (whole seminar package).
- `mypy --strict` (hive.py): `Success: no issues found`.
- No §6.5 anti-patterns: `HiveDemo` is one `SeminarDemo` registered in the
  dict (#1/#13); no algorithm logic re-implemented (#2); small static helpers,
  all < 40 LOC (#6.2).

## Known limitation
The robot fixtures (`data/robotics/robot_4wh.hymeko` …) use `@"…"` includes that
this engine build resolves relative to cwd and currently fail to load (or report
cyclic includes). The demo therefore defaults to the self-contained
`fano_graph.hymeko`; the dramatic NNZ gap (the "1,498 vs 10,991" talking point)
is best shown in the viewer on a high-arity graph. Fixing the include resolver
is out of scope (engine-internal) and not required for the demo.

## Provenance
- Git SHA `af803ee` (dirty). Python 3.12.13; `hymeko` 0.1.0 editable (maturin).
- Deterministic; no RNG; fixed `parse_dsl` strings.
- Artifact: `demo_out/hive/hive_report.json` (surface text + IR snapshot + DOT +
  both encodings + the canonicalisation proof).

## Open issues / follow-ups
- Remaining build items: latency bench (4), Demo 1 balance (5), Demo 4 mesh +
  Sinkhorn (6), Demo 5 bridge (7).
