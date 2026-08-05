# Hypergraph zoo → HyMeKo → native tensor round-trip

**Date:** 2026-08-05 · **Worktree:** `hymeko_humanoid` (head at start `fb3941c3`) · follow-up (1) of the zoo.

## Summary
Bridged the hypergraph zoo to the native HyMeKo tensor path. `hypergraph_hymeko.py`:
- **`to_hymeko(hg, signs=…)`** emits a zoo `Hypergraph` as HyMeKo source — a `node` per vertex, an
  `@`-hyperedge per edge with **signed incidence** (`+ v` / `− v`), the Nagare signed-incidence form.
- **`round_trip(hg)`** parses it with the native `PyHypergraphEngine` → `PyHypergraphIR`, compiles the
  **star-expansion sparse incidence tensor** (`PySparseMatrix2D`), and verifies the emit is faithful.
- **`signed_graph_cycles(...)`** feeds the zoo's graph families to `enumerate_cycles_rs` (the Rust top-`m`
  signed-cycle / holonomy engine).

## Verified round-trip (incidence preserved)
| hypergraph | IR edges | star nnz = incidence | tensor shape |
|---|---|---|---|
| Fano `PG(2,2)` | 7 | 21 (7×3) | (7, 17, 17) |
| `PG(2,3)` | 13 | 52 (13×4) | (13, 29, 29) |
| loose 3-cycle(4) | 4 | 12 | (4, 15, 15) |
| `STS(13)` | 26 | 78 | (26, 42, 42) |

The star tensor's `nnz` equals the hypergraph's total incidence and the IR edge count is preserved for every
family; the IR carries a `blake3` `canonical_hash`; the native signed-cycle enumerator runs on the graph families.
So the zoo now feeds the **doctoral canonical sparse tensor** and the **native cycle engine**.

## Files
`scenarios/hypergraph_hymeko.py` (+70), `tests/test_hypergraph_hymeko.py` (+55, 4 tests, `importorskip` for
headless), this report. numpy + the built `hymeko` pyo3 extension (no dependency change, no §1).

## Tests
`pytest tests/test_hypergraph_hymeko.py` → 4 passed; `ruff check` clean.

## Follow-up
- **Signed / holonomy experiments** on the Fano + cycle families through `enumerate_cycles_rs` (score/pruner
  options) — the actual Nagare payoff the round-trip enables.
- Emit each zoo family to a saved `.hymeko` and `hymeko validate` in CI.

## Provenance
Git SHA at start `fb3941c3`. Env: `.venv` (Python 3.11, NumPy 2, `hymeko` pyo3 ext), macOS. Deterministic.
