# Machine-verified cross-view consistency for the HyMeKo T-SMC article

**Date:** 2026-06-28 (JST) · **Author:** Aiko (Claude Code), for Dr. Csaba Hajdu
**Plan:** `docs/plans/2026-06-28-cross-view-consistency/` · **Companion:** `reports/2026-06-28-proposition-verification.md` (P1–P4)

## Summary

The ChatGPT review's #1 strengthening point is that HyMeKo's defensible scientific kernel is **cross-view
consistency** — one authoritative IR projects without drift into heterogeneous views sharing non-trivial
invariants. The article *asserts* this (§codegen: "a cross-view discrepancy is impossible") but pins it on
**Proposition 3**, which is only about **cost factorization** (compile-once-emit-many) and is explicitly "not a
commuting diagram (the codomains differ)". The **value-level** claim — that the invariants you read *back out* of
each emitted view agree — was never formalized or machine-checked.

This work closes that gap with an **extraction-function commuting square**. For each format `f` we define an
extractor `X_f` that parses the *emitted* text back into a structural invariant, and verify

- `X_f(ε_f(H)) = X_g(ε_g(H))` for every view pair (**mutual**; each `X_f` parses a *different* concrete syntax, so
  agreement is not a shared-parser artifact), and
- `X_f(ε_f(H)) = Q(H)` anchored to the IR query where obtainable.

Unlike Prop. 3 this square has a **common codomain** (the invariant `Q`), so it genuinely commutes.

**Result.** Across **16 robot fixtures × 5 independently-parsed views** (URDF, SDF, MJCF, DOT, Mermaid) from the
**real CLI emitters**:

- **16/16 EXACT** — the data-interchange formats (URDF/SDF/MJCF) agree on the **full numeric invariant** (link
  set, per-link mass, actuated-joint set, parent/child polarity, signed joint axis);
- **16/16 TOPOLOGICAL** — all five views agree on (links, actuated joints, parent/child).

The exercise also produced **two genuine findings** and one **flagged regression** (below), exactly the reviewer
attack surface the review predicted — and now answered.

## What was built (all new, non-core)

| file | role | LOC |
|---|---|---|
| `verification/cross_view_consistency/extract.py` | `ViewExtractor` ABC + Strategy impls (URDF/SDF/MJCF/DOT/Mermaid) → frozen `KinematicInvariant` | 247 |
| `verification/cross_view_consistency/cross_view.py` | drive CLI emit × fixture, apply `X_f`, test the square at two strengths; JSON out | 196 |
| `verification/cross_view_consistency/trace_witness.py` | non-robotics witness: `X_sysml = X_dot = Q` (requirements traceability) | 96 |
| `verification/cross_view_consistency/commute_z3.py` | Z3 proof: shared-query ⇒ agreement; untethered view can drift | 110 |
| `verification/cross_view_consistency/storage_regime.py` | sympy reframe of Prop. 4 (controlled overhead, vanishing for high arity) | 73 |
| `verification/cross_view_consistency/plot.py` | report figure (consistency grid + storage-regime curve) | 87 |
| `verification/cross_view_consistency/tests/test_cross_view.py` | unit / integration / performance / proof tests | 196 |
| `verification/cross_view_consistency/README.md` | how-to | — |
| `docs/plans/2026-06-28-cross-view-consistency/plan.{tex,pdf,tikz,mmd}` | plan (4 artifacts) | — |
| `reports/figures/cross_view_consistency.{png,svg}`, `reports/cross_view_consistency.json` | artifacts | — |

**CORE.YAML items touched:** none. `hymeko_formats`/`hymeko_cli`/`verification/` are non-core; the CLI is driven
as a black box. `reports/` and `docs/plans/` are allowlisted.

## Substrate decision (measured)

The Python binding (`hymeko.to_urdf/to_sdf/to_dot`) resolves only **one** import level — 26/35 robot fixtures
fail to load, and the loadable ones extract **zero** joints. The **CLI** (`target/release/hymeko emit`) uses
`ModuleStore` (transitive imports) and the model-based emitter (joints, axes, origins, limits) across all six
formats; it is the emitter the article benchmarks. The verification drives the CLI. Measured on `fanuc_lrmate`:
URDF 8 links / 7 joints, SDF 7 / 7, MJCF 6 joints / 6 axes (nested bodies, no `<link name=`).

## Two normalized conventions (named — these are conventions, not drift)

- **(W)** URDF emits a synthetic `world` ground anchor as a `<link>` with no inertial → the comparable link set is
  the **mass-bearing** links.
- **(F)** the fixed root weld (world→base) is an explicit joint in URDF/SDF/DOT/Mermaid but **implicit** in MJCF
  (a jointless root body is welded to the world) → the comparable joint set is the **actuated** joints; fixed
  welds are reported separately.

## Two genuine findings (diagram views are lossy *by design*)

1. **DOT** rounds the mass label to **1 decimal** (`0.02 kg → 0.0`) and encodes the joint axis as an **unsigned
   letter** `(Z)` — so `robot_4wh`'s real `(0,0,-1)` axis reads as `(0,0,1)`. **Mermaid** uses 2 decimals.
2. Consequently the precise numeric invariants live in the **data-interchange** formats (URDF/SDF/MJCF), which are
   **exactly** consistent (16/16), while the **diagram** formats (DOT/Mermaid) carry a *topologically faithful*
   projection whose numeric labels are a documented, reduced-resolution rendering (16/16 topological). This
   two-tier statement is the honest, reviewer-proof form of "cross-view consistency".

These findings were surfaced by the verification itself; before normalization the harness reported them as
disagreements (the run log is in the git history of this report's development). Two of my initial extractor
mismatches were *my* bugs (ElementTree `find(...) or default` dropping a childless `<parent>`; MJCF root-body
implicit world parent) — fixed and regression-tested; the remaining two are the genuine, documented projections
above.

## Z3 proof (logical half)

- **T1 (positive):** under the hub-and-spoke axiom (every view renders from the **same** query `Q`), pairwise
  agreement holds — Z3 reports the negation **unsat** (proved by construction).
- **T2 (negative, falsifiable):** drop the shared-query tie for one view (the per-format-converter anti-pattern
  the article argues against) — Z3 finds a **sat** drift model.

So cross-view consistency is **entailed by the architecture**; the only empirical obligation is **per-view
faithfulness**, which `cross_view.py` falsification-tests over 16×5 instances against the real emitters.

## Storage-overhead reframe (review #2)

Reusing the symbolic Prop. 4 core: `ρ = 1 + 2/d̄` (n=m, c=1). The honest regime table:

| regime | d̄ | ρ |
|---|---|---|
| robotics (binary joints) | 2 | **2.00** |
| low-arity (tri/quad) | 3 | 1.67 |
| mixed | 6 | 1.33 |
| high-arity relations | 20 | 1.10 |
| very high arity | 200 | 1.01 |

`lim_{d̄→∞} ρ = 1`, monotone decreasing. The article should say **"controlled overhead, asymptotically vanishing
for high-arity relations"** — robotics is the binary-joint regime where ρ≈2 (a small constant), **not** ρ→1.

## Non-robotics witness (review #3)

Requirements traceability (`data/paper/traceability_smc.*`): `X_sysml(ε_sysml) = X_dot(ε_dot)` — both views recover
**4 requirements, 4 blocks, 5 satisfy + 1 allocate + 1 derive (7 trace edges)**, matching the article's stated
`Q`. A second domain where one IR makes cross-view invariants checkable.

## Fixed regression — `requirements_sysml`/`requirements_dot` (non-core, `hymeko_cli`)

The live `requirements_sysml` / `requirements_dot` CLI emitters returned **"model extraction failed"** on
`data/paper/traceability_smc.hymeko`. **Diagnosed and fixed this session** (the user authorized a core edit if
needed; none was required — the fix is in non-core `hymeko_cli`).

**Root cause.** The CLI `Emit` path dispatched `if t.accepts()==ModelKind::Kinematic { emit } else { template }`.
The two transforms are **template-only** (`emit()` returns `None`, they render from `transforms/<fmt>/`) but
declared `accepts()==ModelKind::Kinematic`, so they hit the emit branch → `None` → the error. `ModelKind` lives in
core `hymeko_query`, so I did **not** add a variant; instead the dispatch in `hymeko_cli/src/main.rs` now falls
through to the template path when `emit()` yields nothing **and** the transform declares a `template_dir()` — a
kinematic transform with neither output nor a template is still a genuine failure (error preserved).

**Verified.** Live `requirements_sysml` output is **byte-identical** to the committed
`data/paper/traceability_smc.sysml`; `requirements_dot` emits; kinematic formats unaffected (urdf 7 joints, dot
works). Two regression tests added (`test_regression_template_only_emitters_route_to_template_path`), both would
have failed pre-fix. `cargo clippy -p hymeko_cli` clean. The non-robotics witness now drives the **live** emitter
(with committed-file fallback). A Windows-codec UTF-8 fix (`encoding="utf-8"` on the subprocess calls) was needed
for the guillemets in the SysML output.

## Test results

`pytest -p no:randomly verification/cross_view_consistency/tests/` — **17 passed in 13.6 s**.

- **Unit (CLI-free):** helper normal/boundary/failure; each extractor recovers the invariant; the square on
  hand-written fixtures; empty-model boundary; malformed-XML and empty-text failure paths (typed `ValueError`).
- **Integration (real CLI):** the square over all 16 kinematic fixtures — 16/16 exact + topological.
- **Proofs:** Z3 T1/T2; storage regime.
- **Performance (≥5 iters after warm-up, median/IQR/worst):** single-fixture five-format extraction
  **median 282 ms, IQR 42 ms, worst 346 ms** (budget < 1000 ms ✓); **peak RSS 109 MB** (budget < 512 MB ✓, well
  under the 16 GB cap).

## Static analysis (§6.3)

- `ruff check verification/cross_view_consistency/` — **clean**.
- `radon cc` — max cyclomatic after refactor: `run` 11 (warn), all others ≤ 5; no function over the 15 fail
  ceiling. `check_fixture` was refactored 19→5 by extracting `_emit_invariants` / `_exact_disagreements` /
  `_topo_disagreements`. `Sdf/Urdf` extractors sit at 12/10 (warn level, inherent parse branching).
- No new error-handling waivers; no §6.5 anti-patterns introduced (Strategy/ABC for the per-format axis, no
  Cartesian function dump, no globals, no v-suffix files).

## Article edits (companion, in `01_analysis/.../smc_02`) — applied, compiles clean

- **Title** retargeted: *HyMeKo: A Canonical Hypergraph Intermediate Representation for Cross-View Consistency in
  Cyber-Physical Model Generation* (+ matching `\markboth`).
- **Abstract** updated: "Five algebraic properties … machine-verified against the reference implementation …
  property tests of the extraction-function commuting square `X_f(ε_f(H))=Q(H)`".
- New **Proposition 4 (cross-view consistency)** `prop:crossview` + proof appendix `A5_proof_cross_view.tex`: the
  extraction-function square, machine-verified (16×5), distinguished from the cost-factorization Prop. 3 (now
  Prop. 3; storage is Prop. 5).
- **Storage wording** softened to "controlled overhead, asymptotically vanishing for high-arity relations".
- §codegen "cross-view discrepancy is impossible" re-anchored from `prop:commute` to `prop:crossview`; the SysML
  witness paragraph now cites the machine-verified `X_sysml = X_dot = Q`.

## Length trim + positioning (article, applied)

The review flagged the paper as over-weighted toward the robotics benchmark (and Table IV reveals HyMeKo as a
*general* canonical-hypergraph framework, robotics being only the witness). Rebalanced 15 → **14 pages**, shifting
weight from "fast robot generator" to "general IR with proved cross-view consistency":

- **`07_eval_scaling.tex`** compressed 16 KB → ~5 KB; the head-to-head and morphology **tables moved to
  supplementary** (key numbers kept inline; the honest architectural-vs-representational attribution retained).
  Tables renumber I/II/III (was I–V).
- **Table V (`tab:sota_quant`)**: dropped the attackable `Approx. impl. surface [LoC/fmt]` column (review: "not a
  controlled measurement"); caption now states the latency win is architectural, not the central claim.
- **Table IV (`tab:sota_qual`)**: "algebraic argument" redefined as *machine-verified proofs*.
- **Conclusion** reframed: HyMeKo stated as "a general canonical signed-typed hypergraph description framework
  with proved cross-view consistency, not a robot-description generator"; five properties, two witnesses
  (robotics + SysML); pointer added to the `docs/proofs` consolidated proof document.
- **New evaluation subsection `Cross-view consistency, measured` (`sec:eval_crossview`)**: writes the actual
  `X_f` square result (robotics 16/16 exact + 16/16 topo, the z3 entailment) — previously the `prop:crossview`
  references forward-pointed to `sec:eval_parity`, which only covered alias invariance — and **promotes the SysML
  witness to a load-bearing second-domain result** (`X_sysml = X_dot = Q`, 4/4/7, coverage 3/4). The §codegen
  paragraph is trimmed to introduce the listing and forward-reference the measured result (no duplication). This
  makes the "general framework" claim load-bearing rather than asserted.
- **Section consolidated + reframed (#3):** §VII renamed "Case Study: Robotic Description Generation" → **"Evaluation"**;
  opening rewritten to lead with cross-view consistency (the central claim, two domains) and frame performance as
  *practicality evidence, not the claim*. Subsection order now: VII-A cross-view → VII-B throughput → VII-C alias →
  VII-D determinism → VII-E end-to-end → VII-F scaling → VII-G threats. Article 14 pages, compiles clean, no
  undefined/duplicate refs.

## Prior-art search + missing-citation fix (article, applied)

A focused search (WebSearch + paper_search) on the *specific* contribution (machine-verified cross-view
consistency over a signed canonical hypergraph IR) found **no concrete precedent**; the cross-view-consistency
literature is all ML (multi-view representation), and canonical-model generators (EMF-style) are typed-graph,
no-proof. Two must-cite related works were missing from Related Work and are now added (`sources.bib` +
`08_related.tex`): (1) the **precursor** Hajdu \& Hegyi 2025, *Modeling Kinematic and Dynamic Structures with
Hypergraph-Based Formalism* (Applied Mechanics) — the signed-hypergraph robot model + star-expansion, but
modelling/expressiveness, **not** proved consistency; (2) **HyperGraphOS** (Ceravola et al. 2024) — hypergraph
DSL codegen without a content-addressable canonical IR or proved invariants. Both differentiated in one sentence
each; converts a latent "didn't survey / self-overlap" reviewer attack into a strength. Compiles clean, 14 pp,
citations resolve. (Process: added a `CLAUDE.md` operating principle — *search before claiming novelty/prior art*
— after an on-record novelty-assessment miss this session.)

## Supplementary document + language-cleanup (article, applied)

- **`supplementary.tex`** created (2 pp, compiles): renders the head-to-head and realistic-morphology scaling
  tables (moved out of the body during the trim) and the full alias-invariance proof (Prop. 1). Uses `xr`
  (`\externaldocument`) so section/proposition cross-references resolve to the main paper's numbers.
- **Removed Rust/Python *code* idioms** from the body (per user scope: design/memory/perf details and named
  algorithms stay; only language-specific code goes): `apply_usings`→"name resolution",
  `prop1_alias`/`prop3_commute`/`cross_view_consistency` test tokens → descriptive names, "Rust string-builder" →
  "direct string-builder". **Kept** (contributions): AVX2/SIMD lexer, COO assembly, Arrow/DLPack interop,
  release-build/single-threaded measurement conditions, the module store.
- **BLAKE3 kept and cited**: added `@misc{blake3}` (the official spec) to `sources.bib`; reverted the earlier
  genericization — naming a published, citable primitive is a design choice, not an implementation leak.

## Proofs bundle (`docs/proofs/`)

- **`hymeko_machine_verified_proofs.pdf`** — 25-page master document: all five propositions, proofs, methods,
  verbatim machine outputs, the Z3 reduction, the storage regime, the dispatch-bug fix, and full script listings
  (Appendix A via `\lstinputlisting`, read from canonical sources — no duplication).
- `MANIFEST.md` (script locations + PDFs), `results/*.txt` (captured machine outputs), the figure, and a copy of
  the earlier `proposition-verification.pdf`.

## Provenance

- Git SHA `ec98095` (working tree dirty — pre-existing `M` files unrelated to this task; new files listed above).
- Python 3.12.13; sympy 1.14.0, z3 4.16.0, matplotlib 3.11.0, numpy 2.4.6; pytest 8.x (`-p no:randomly`).
- CLI: `target/release/hymeko.exe` (built `cargo build --release -p hymeko_cli`, this session).
- Host: Windows 11, the project `.venv`. No GPU, no network. Deterministic (no RNG in the verification path).

## Open issues / follow-up

1. `requirements_sysml`/`requirements_dot` "model extraction failed" regression (above) — bisect.
2. Optional: extend `X_f` to **gazebo** (world wrapper) and **torch_dataflow** (the projection π) for a 7-view
   square; gazebo wraps SDF, torch_dataflow is the tensor projection (different codomain — belongs with Prop. 3,
   not this square).
3. Optional: wire `cross_view.py` into CI as a regression gate on the emitters.
