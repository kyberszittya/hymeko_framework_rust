# Seminar demo program — build-item 2: star-expansion viewer (Phases 1–3)

**Date:** 2026-06-10 · **Plan:** `docs/plans/2026-06-10-seminar-demo-program/`
· **Spec:** `demo_web/STAR_EXPANSION_VIEWER_OUTLINE.md`

## Summary
Turned the synthetic Phase-0 prototype into a HyMeKo-fed viewer.
- **Phase 1 (data contract):** committed `demo_web/star_expansion_data.example.json`
  (Fano plane; counts hand-verifiable: 7 arity-3 edges → star incidence 21,
  clique edges 21).
- **Phase 2 (exporter):** `demo_web/export_star_expansion.py`, mirroring
  `export_kinematic_data.py` — engine-primary (parse `.hymeko` → IR → snapshot +
  `compile_star/clique_expansion` + canonical hash) with a dependency-free
  literal-grammar fallback. Emits `star_expansion_data.{json,js}`.
- **Phase 3 (page):** wired `star_expansion_viewer.html` to a "HyMeKo source"
  preset that loads the exported data (`window.SXV_DATA` for `file://`, else
  `fetch`); displays `source` + `canonical_hash` + engine counts and verifies
  the JS-derived counts agree (no drift); the synthetic generators remain as a
  labelled "Sandbox".

To support this (and Demo 2), built the `hymeko` PyO3 module and added a small
**additive** `canonical_hash` getter to the binding crate (see below).

## Files touched
**New:**
| LOC | File |
|---:|---|
| 268 | `demo_web/export_star_expansion.py` — exporter (engine + fallback Strategy) |
| 137 | `hymeko_neuro/tests/test_star_expansion_export.py` — tests |
| 19 | `demo_web/star_expansion_data.example.json` — Phase-1 schema example |
| — | `demo_web/star_expansion_data.{json,js}` — generated (Fano) |

**Modified:**
- `demo_web/star_expansion_viewer.html` — Phase-3 data wiring (preset, boot,
  provenance display, data `.js` include).
- `hymeko_py/src/interface_python/api.rs` — **+15 lines**, additive
  `PyHypergraphIR.canonical_hash` getter returning `blake3:<hex>`.

## CORE.YAML items touched
**None.** `hymeko_py` is the PyO3 *binding* crate and is **not** in CORE.YAML
(CORE protects `hymeko_core`, `hymeko_query`, `hymeko_client`, `hymeko_daemon`,
`parser`). The getter is additive public API on a non-core crate and reads the
already-computed `compiled.canon_hash` field — no core logic changed, no
dependency added. Building `hymeko` (compiling) is not a core edit.

## Test results
Engine-backed tests require the built `hymeko`; run with the venv interpreter
(`uv run --group` re-syncs and drops the editable install):

    PYTHONPATH=. .venv/Scripts/python.exe -m pytest -p no:randomly \
        hymeko_neuro/tests/test_star_expansion_export.py

| Layer | Tests | Result |
|---|---|---|
| Unit — fallback counts, incidence-COO↔members, dual JSON/JS output | 3 | pass |
| Integration — engine vs fallback count agreement; canonical-hash invariance | 2 | pass |
| **Item-2 total** | **5** | **pass** |
| Combined with item-1 suite (one run, engine live) | **21** | **pass** |

## Key finding — canonical-hash invariance (matters for Demo 2)
Empirically probed (2026-06-10) the engine canonical hash is:
- **invariant** to node- and edge-**declaration order**; deterministic;
- **sensitive** to a structural edit (changed connection);
- **NOT** invariant to relabeling edges, nor to within-edge member order
  (those carry signed-arc meaning).

So Demo 2's "isomorphic-but-differently-written → same fingerprint" claim is
true specifically for **declaration-order** permutations (the demonstrable,
honest framing); it is not a full graph-isomorphism canonicalisation. The
viewer only displays the hash as provenance, so this does not affect Phase 1–3.
Flagged for build-item 3 (Demo 2).

## Counts verified (Fano plane fixture)
`star.incidence_nnz = Σ|e| = 21`, `clique.edge_count = Σ C(|e|,2) = 21`;
engine raw COO nnz = 42 for both (the `~`→sign-0 convention pushes incidences
symmetrically, and the clique matrix stores directed entries). The exporter
asserts the JS/engine agreement (`_cross_check_engine`). `canonical_hash =
blake3:b88715ed…c86dc5`. Fallback path reproduces the same counts dependency-free.

## Static analysis
- `ruff check`: clean (exporter + test).
- `mypy --strict` (exporter): `Success: no issues found`.
- `cargo clippy -p hymeko_py`: the getter introduces **no** new warning
  (pre-existing "too many arguments" warnings on existing pyfunctions remain,
  untouched).
- No §6.5 anti-patterns: exporter uses a runtime Strategy (engine/fallback),
  not duplicated count logic; reuses `export_kinematic_data.py`'s dual JSON/JS
  convention rather than forking it (#6.1); single file with `--src/--out`.

## Honest-presentation notes (carry to the talk)
- 3D layout is **force-directed for legibility, not geometric ground truth**.
- The edge-count arithmetic is exact and engine-sourced; the page shows the
  engine numbers and asserts the JS-derived ones agree.

## Reproducibility / provenance
- Build the engine once: `.venv/Scripts/maturin.exe develop --manifest-path
  hymeko_py/Cargo.toml` (≈50 s incremental; ≈3 min cold). Installed
  `hymeko-0.1.0` editable into `.venv`.
- Regenerate viewer data: `python demo_web/export_star_expansion.py --src
  data/typical_graphs/fano_graph.hymeko --out demo_web/star_expansion_data.json`.
- Git SHA `af803ee` (dirty). Rust 1.93.1; maturin 1.13.3; Python 3.12.13.
- Deterministic (no RNG in the exporter; layout RNG is browser-side, cosmetic).

## Open issues / follow-ups
- **Browser render not auto-verified** — the page logic is wired and the data
  loads/parses, but I cannot drive a browser here; visually confirm the
  "HyMeKo source" preset renders the Fano graph and the provenance line shows
  the engine counts with the ✓.
- Phase 4 (live WebSocket bridge) deferred — optional, build-item 7.
- The `canonical_hash` getter wants a Rust-side unit test in `hymeko_py`; it is
  currently covered by the Python invariance test. Add a `#[test]` if the
  binding crate grows a test module.
