# SMC #5 — Requirements-trace witness: HyMeKo → SysML v2 + DOT (Phase 1)

**Date:** 2026-06-14 · **Plan:** `docs/plans/2026-06-14-sysml-requirements-trace/`
(four-format) · **Reviewer item:** `smc_02/review_1` #5 (minimal non-robotics witness)

## Summary
Built the requirements-traceability witness reviewer item 5 asks for: a single
`.hymeko` model of a pick-and-place cell (requirements ↔ components, signed trace
hyperedges) that projects — through **template-only transforms** — to a faithful
SysML v2 requirements model and a DOT traceability graph. **No core edit, no
`emit_sysml.rs` change.** The discovery finding that the template path
(`render_from_templates` → `execute_transform`) runs off the raw `Ir` and bypasses
the core `ModelKind`/`ModelView` enums is what makes this fully non-core; the
existing `NameResolver` trait (Interner + StringTable) is reused, not written.

This is Phase 1 (the paper artifact). Phase 2 (the editor SysML lens reusing these
templates via `include_str!`) is not started.

## Files touched
**New (non-core):**
| LOC | File | |
|---:|---|---|
| 22 | `data/paper/meta_sysml_trace.hymeko` | vocabulary: node types `requirement`/`block`; `@satisfies`/`@derives`/`@allocated_to` hyperedge types |
| 35 | `data/paper/traceability_smc.hymeko` | the witness: 4 requirements (1 derived), 4 components, signed trace edges |
| 11 | `transforms/requirements_sysml/queries.hymeko` | selects req/block + trace edges |
| 40 | `transforms/requirements_sysml/template.sysml` | SysML v2: `requirement def`/`part def`/`satisfy`/`allocate`/derive connection |
| 9 | `transforms/requirements_dot/queries.hymeko` | same context |
| 38 | `transforms/requirements_dot/template.dot` | DOT traceability graph, stereotyped edges |
| +49 | `hymeko_formats/src/transforms.rs` | `RequirementsSysmlTransform` + `RequirementsDotTransform` (template-only `DomainTransform`s) |
| +4/−2 | `hymeko_formats/src/lib.rs` | register both; `pub use`; doc-comment accuracy |
| +66 | `hymeko_query/tests/test_transform_ecosystem.rs` | `requirements_trace` test module (3 tests) |

**Generated artifacts (committed for the paper):**
`data/paper/traceability_smc.sysml` (2152 B), `data/paper/traceability_smc.dot` (1722 B).

## CORE.YAML items touched
**None.** `hymeko_formats`, `transforms/`, `data/` are non-core. The template path
bypasses the core `ModelKind`/`ModelView`; `hymeko_query::{traits::NameResolver,
rewrite::template::execute_transform, transforms::DomainTransform}` are *used*, not
modified. `emit_sysml.rs` is untouched.

## Test results
| Layer | Tests | Result | Notes |
|---|---|---|---|
| Unit (Rust, `hymeko_query` integration) | 3 | pass | registry has both transforms + no registration regression; SysML emits `requirement def`/`satisfy`/`allocate`/derive; DOT emits trace nodes + `«satisfy»`/`«allocate»`/`«deriveReqt»` edges |
| Regression (Rust, full ecosystem suite) | 209 | pass (1 ignored) | kinematics SysML/DOT byte-unchanged — new transforms don't perturb the existing six |
| Validate (CLI) | 1 | ✅ | `hymeko validate data/paper/traceability_smc.hymeko` — parse + resolve + topology |

Gates: `cargo clippy -p hymeko_formats --all-targets` clean; changed Rust files
`fmt`-clean (pre-existing `codegen.rs` drift is unrelated, not touched).

## Performance
Template render is sub-millisecond for the witness (no benchmark warranted at this
scale); full ecosystem suite 4.28 s. No new hot loop; no RSS concern (well under
16 GB).

## What the artifacts contain
- **SysML v2:** `requirement def R{1,2,3,2a}` with `doc`, `part def` per component,
  `satisfy R by Component;` (R2 and R3 each show two realizers as two statements —
  the binary projection of the trace), `allocate R2a_repeatability to Arm;`, and a
  `«deriveReqt»` connection for R2a ← R2.
- **DOT:** requirement boxes + component nodes + colour/style-coded `«satisfy»`
  (green dashed), `«allocate»` (pink), `«deriveReqt»` (purple dotted) edges.

## Open issues / follow-ups
- **DOT → PDF figure** needs graphviz (`dot`), absent on this machine: the `.dot`
  text generates and is tested; `dot -Tpdf` renders on a graphviz host. (Plan risk
  item, stated not skipped.)
- **Paper wiring — DONE 2026-06-14.** Wired into `smc_02` (the TSMC journal build
  `bare_jrnl_new_sample4.tex`): a new §V subsection *"A non-robotics witness:
  requirements traceability"* (`sec:codegen-trace`) with a 10-line `hymeko` listing
  (`lst:trace`); the §IX limitation softened to cite it (`Section~\ref{sec:codegen-trace},
  Listing~\ref{lst:trace}`). Figure moved to the supplement (prose-referenced, like
  `fig_crate_overview`); `figures/traceability_smc.pdf` copied in for it. Compiles
  clean (refs resolve). **Page count 14 → 15** — accepted by the user; the reviewer's
  own item 4 (shorten §VI to ~0.5 pg) reclaims it. `dot` (graphviz 15.0.0) now
  installed; `.svg`/`.pdf`/`.png` rendered into `data/paper/`.
- **Editor viewing aid (added 2026-06-14):** a **Req-trace** button in `docs/editor/`
  (`index.html` + `EXAMPLE_TRACE` in `editor.js`, `?v=19`) loads a self-contained
  inline variant of the witness (validated) so it renders in the existing Hypergraph
  3D view (vertices + signed hyperedge cubes) without a file include. This is the
  hypergraph form, not the styled SysML diagram (that's Phase 2).
- **Phase 2 — editor SysML lens:** `CompiledDoc::to_sysml`/`to_trace_dot` via
  `include_str!` of these same templates + `execute_transform` (no filesystem in
  WASM), rebuild `pkg/`, `views/sysml.js`. Reuses Phase 1 with no new emission logic.
- **n-ary note:** the witness uses binary trace edges (the conventional MBSE form and
  what the template DSL renders completely — it can't fan out arc members). The
  signed n-ary generality is the IR's formal property, exercised by the robotics
  witness, not this one.

## Provenance
- Witness validated + artifacts generated with the workspace `hymeko_cli` (debug),
  this working tree. Deterministic (template render is pure over the IR).
- No new dependencies. No `#[allow]`/`unwrap` waivers in non-test code (the transforms
  return `None` for the unused Rust `emit()` path; tests use `.expect` with messages).
