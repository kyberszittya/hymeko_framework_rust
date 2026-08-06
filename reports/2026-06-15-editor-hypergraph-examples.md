# Editor gallery: classic hypergraph examples (Fano, sunflower, K₄³, generic)

**Date:** 2026-06-15
**Plan:** `docs/plans/2026-06-15-editor-hypergraph-examples/` (tex/pdf/tikz/mmd)
**Branch:** `feature/ac-hsikan` · **Base SHA:** `9684f09` (working tree dirty — see provenance)

## Summary

Added a built-in **example gallery** to the WASM editor (`docs/editor`) so the
classic combinatorial hypergraphs can be loaded and inspected live in the
existing **Hypergraph 3D** view (star / clique / **prism** modes): the **Fano
plane** S(2,3,7), a **sunflower / Δ-system**, the complete 3-uniform hypergraph
**K₄⁽³⁾**, and a **generic mixed-arity** hypergraph.

The discovery pass (CLAUDE.md §6.1/§6.5 #12) found the canonical sources
**already existed** as committed fixtures under `data/typical_graphs/*.hymeko`
(AST-tested in `hymeko_core`). So this was **wiring, not generation**: surface
those fixtures in the editor UI and route a selection through the existing
compile → snapshot → 3D-view pipeline. No generators were written, no new
fixtures created, no WASM rebuilt.

Mechanism: a data-driven catalog (`views/examples.js`) embeds the sources (the
editor serves from `docs/editor/`, so the repo `data/` tree is not
`fetch`-reachable — embedding matches the existing `EXAMPLE` precedent). A
consistency test pins **embed ≡ fixture**; a Rust test pins that the fixtures
**compile through the editor's exact pipeline** and produce the intended
hypergraph. The two inline example constants previously hard-coded in
`editor.js` were consolidated into the same catalog (de-duplication), and the
two `Example`/`Req-trace` buttons were replaced by one `Examples` dropdown.

## Files touched

New:
- `docs/editor/views/examples.js` (208 LOC) — catalog `EXAMPLES`, `exampleById`,
  `FIXTURE_OF`; embeds the 4 hypergraph sources + the existing kinematic /
  req-trace examples.
- `docs/editor/views/examples.test.mjs` (66 LOC) — `node --test`: catalog
  integrity + embed≡fixture consistency.

Edited:
- `docs/editor/editor.js` — import the catalog; build the dropdown; wire
  select → load → recompile → (hypergraphs) `showView("hyper3d")`; removed the
  ~75 LOC of inline `EXAMPLE`/`EXAMPLE_TRACE` constants and their two button
  handlers (now catalog-driven); fixed one stale comment reference.
- `docs/editor/index.html` — replaced the two example buttons with
  `<select id="exampleSelect">`; bumped `editor.js?v=19 → ?v=20` (cache-bust).
- `docs/editor/editor.css` — `.example-pick` toolbar rule (~12 LOC).
- `hymeko_wasm/tests/test_compile.rs` — new `typical_graph_examples` test module
  (compile-smoke + perf guard). `rustfmt` also reflowed two pre-existing tests
  in this file so the file now passes `fmt --check`.

> Note: `git diff --stat` against HEAD also shows changes in
> `views/{geometry3d,hypergraph3d,kinematic}.js`, `sysml.*`, etc. — these are
> **pre-existing uncommitted** edits from prior session work (SysML lens, etc.),
> not part of this task.

## CORE.YAML items touched

**None.** `docs/editor/**` and `hymeko_wasm/**` are outside every CORE.YAML
crate / file / glob; `tests/**` is allowlisted. The Fano AST tests in
`hymeko_core` (`lockdown: full`) were not touched. No dependency changes.

## Test results

- **JS unit (`node --test`, all editor `views/*.test.mjs`):** 38 passed / 0
  failed (5 new in `examples.test.mjs`: id uniqueness, label/source/view
  validity, `exampleById`, the four hypergraphs jump to `hyper3d`, embed ≡
  fixture). ~0.35 s.
- **Rust integration + perf (`cargo test -p hymeko_wasm --test test_compile`):**
  6 passed / 0 failed. New: `typical_graphs_compile_to_expected_hypergraphs`
  (counts + arities) and `typical_graphs_compile_within_budget`. ~0.02 s.
  - Verified shapes (nodes incl. the enclosing block decl): Fano 8 nodes / 7
    edges / arities `[3×7]`; K₄³ 5 / 4 / `[3×4]`; sunflower 9 / 3 / `[4×3]`;
    generic 8 / 4 / `[2,3,3,3]`.

## Performance results

Compile median (of 5, after warm-up) on the dev host, per fixture:

| fixture | median |
|---|---|
| fano_graph | 1371 µs |
| k4_3uniform | 547 µs |
| sunflower_delta_system | 606 µs |
| generic_hypergraph | 555 µs |

All well under the planned 20 ms budget; the test guard ceiling is 250 ms
(gross-regression only). Peak RSS negligible (tiny fixtures; far under the
16 GB cap). Render path is the unchanged 3D loop.

**§3 deviation (declared):** no `criterion` micro-benchmark was added. There is
no algorithmic hot path in scope — the change adds static example data — so a
median-of-5 wall-time *guard* in the integration test is used instead of a
reported benchmark. (CLAUDE.md §3 forbids defensive/ceremonial optimization;
a criterion bench over a 7-node compile would be exactly that.)

## Static analysis / health

- `cargo clippy -p hymeko_wasm --tests -- -D warnings`: clean.
- `rustfmt --check` on `test_compile.rs`: clean (post-format).
- No new `unwrap`/`expect` in non-test code (test-only; allowed). No new
  `#[allow]`. No globals. **No §6.5 anti-patterns introduced** — in fact the
  change *removes* a duplication (inline example consts → single catalog) and is
  data-driven (#1/#8 honored).

> Crate-wide `cargo fmt -p hymeko_wasm --check` flags `test_snapshot_relationships.rs`,
> a **pre-existing** unrelated file not touched here.

## New / removed dependencies

None.

## Open issues / follow-ups

- Optional future hypergraphs (if wanted): Steiner triple S(2,3,9) / affine
  plane AG(2,3), Pasch configuration, Möbius–Kantor, complete K₅⁽³⁾. Each is one
  fixture + one catalog line; the wiring already scales.
- A headless-browser end-to-end click test was **not** run (no driver
  configured). Coverage is transitive: native `compile_source` ≡ browser WASM,
  embed ≡ fixture (JS), and `snapshotToHyperedges` is unit-tested
  (`adapters.test.mjs`).
- The enclosing block decl (`fano`, `sunflower`, …) renders as an extra vertex
  in the 3D view — identical to how `robot`/`kit` already appear; the scope
  layer links it. Not a regression.

## Experiment provenance

Not an experiment (no data/models produced). Working tree was dirty at start
(prior-session editor edits, signedkan runs, reports) — unrelated to this task.
Toolchain: node v24.14.0, cargo (stable), MiKTeX pdflatex for the plan PDF.
