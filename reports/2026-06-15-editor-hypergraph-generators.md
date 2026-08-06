# Editor "Generate" tab: parametric hypergraph generators

**Date:** 2026-06-15
**Plan:** `docs/plans/2026-06-15-editor-hypergraph-generators/` (tex/pdf/tikz/mmd)
**Branch:** `feature/ac-hsikan` · **Base SHA:** `9684f09` (working tree dirty)

## Summary

Added a **Generate** tab to the WASM editor that builds classic hypergraphs from
parameters on demand and loads them into the editor (source-as-truth →
recompile → Hypergraph 3D). Follows the 2026-06-15 example gallery; this adds
parametric **construction**. Generator families:

- **Steiner triple system** S(2,3,n) for n ∈ {7,9,13,15,19,21,25} — Bose
  construction for n ≡ 3 (mod 6); MRV backtracking for n ≡ 1. (Fano = S(2,3,7);
  STS(9) comes out as the affine plane AG(2,3).)
- **Sunflower / Δ-system** — petals k, core size c, petal size p.
- **Complete r-uniform** Kₙ⁽ʳ⁾ — all C(n,r) subsets.

The pure construction logic is fully separated from the DOM (the persona's
trait/strategy + pure-core preference): a `GENERATORS` registry (Strategy: each
family = `{params, validate, build, summary}`) drives a **data-driven form** —
adding a generator needs no UI code. A **progress bar** + "Generating…" state
shows during the (deferred) build, per the user's request.

## Files touched

New (all in `docs/editor/views/`):
- `generators.js` (pure, ~280 LOC) — `GENERATORS`, `generatorById`,
  `generateSource`, `specToHymeko`, `stsTriples` (Bose + MRV backtracking),
  `combinations`, `binom`.
- `generators.test.mjs` (~190 LOC) — property tests (STS pair-exactly-once,
  sunflower intersection = core, complete = C(n,r) distinct r-sets), serialiser
  shape/identifier tests, and a build perf budget that **gates** the offered
  STS list.
- `generator_view.js` (~150 LOC) — `createGeneratorView(loadSource)`: the
  data-driven form + progress bar. View contract; `render` is a no-op (it emits
  source, doesn't consume a snapshot).

Edited:
- `editor.js` — import the view; `loadGenerated(source)` callback (set source →
  recompile → `showView("hyper3d")`); registered in `VIEWS`.
- `index.html` — `Generate` tab + `view-generate` pane; `editor.js?v=20 → v=21`.
- `editor.css` — `.gen-form` form styling + indeterminate `.gen-progress` bar.
- `hymeko_wasm/tests/test_compile.rs` — `generated_sources_compile_to_expected_hypergraphs`:
  three verbatim Generate-tab outputs compiled through the editor's pipeline.

## CORE.YAML items touched

**None.** `docs/editor/**` and `hymeko_wasm/**` are outside every CORE.YAML item;
`tests/**` allowlisted. No dependency, grammar, or WASM-source change — generated
sources are ordinary `.hymeko` text the existing bundle compiles. **No WASM
rebuild needed.**

## Test results

- **JS (`node --test`, all `views/*.test.mjs`):** 48 pass / 0 fail (10 new in
  `generators.test.mjs`). ~0.37 s.
- **Rust (`cargo test -p hymeko_wasm --test test_compile`):** 7 pass / 0 fail
  (new `generated_sources_compile_to_expected_hypergraphs`: STS(9) → 10 nodes /
  12 edges / arity 3; sunflower → 9 / 3 / 4; K₅⁽³⁾ → 6 / 10 / 3).
- **Browser:** headless-Chrome screenshot of `?view=generate` — the form renders
  with the live summary "→ 9 points, 12 triples" and the Generate button.

## Performance results

STS build median (of 5) is well under the 100 ms interactive budget for all
offered n after the MRV fix. The **naive smallest-pair backtracker measured
639 ms for STS(19)** (over budget); replacing it with most-constrained-pair
selection + forward checking made the search effectively backtrack-free, so
n = 19/25 stay in the offered list. The perf test gates the list — any n that
regressed past 100 ms would be dropped, not silently shipped.

**§3 deviation (declared):** the JS perf test is a median-of-5 budget guard, not
a `criterion`/`pytest-benchmark` run (those tools don't apply to browser JS, and
there is no Rust hot path here). It still asserts a numerical budget per §3.

## Static analysis / health

- `cargo clippy -p hymeko_wasm --tests -- -D warnings`: clean. `rustfmt --check`
  on `test_compile.rs`: clean.
- **No §6.5 anti-patterns.** Strategy registry (not a per-variant Cartesian
  dump, #1/#5); pure core split from DOM view (#4); string params coerced at the
  boundary; no globals (#11). The data-driven form means new generators need no
  new UI code.
- No new deps. No new `unwrap` in non-test code.

## Open issues / follow-ups

- STS for n ≡ 1 (mod 6) uses bounded search; a closed-form Skolem construction
  would remove the (now-irrelevant) search cost and lift the cap beyond 25.
- The floating source panel overlaps the right edge of the form (it's
  draggable); a dedicated layout for non-canvas views (Generate) could give the
  form full width.
- Possible additions: S(2,4,n), Pasch/Möbius–Kantor, projective planes PG(2,q) —
  each is one registry entry.

## Experiment provenance

Not an experiment. Toolchain: node v24.14.0, cargo stable, Chrome (headless
screenshot), MiKTeX pdflatex (plan PDF). Working tree dirty from prior session
work unrelated to this change.
