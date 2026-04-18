# Project Changelog — 2026-04-18

## `hymeko_hre` Crate Extraction

- Created new workspace crate `hymeko_hre` (Hypergraph Rewriting Engine) and registered it in the top-level `Cargo.toml` workspace members alongside the existing seven crates.
- Moved `hymeko_core/src/engine/` (module file, `hypergraphengine.rs`, `hypergraphengine_impl.rs`, `hymeko_subscriber.rs`) into `hymeko_hre/src/engine/`, rewriting `crate::{ir,tensor,resolution,traversal}::...` paths to `hymeko::...` against the new `hymeko_core` dependency. `lib.rs` re-exports `HypergraphEngine` at the crate root.
- Removed `pub mod engine;` from `hymeko_core/src/lib.rs` and deleted `hymeko_core/tests/engine/`; stripped the `mod engine;` line from `hymeko_core/tests/mod.rs`.
- Relocated the engine integration test to `hymeko_hre/tests/test_hypergraphengine.rs`, inlining its harness so it no longer depends on the crate-level `test_helpers` module.
- Mirrored the `ipc` and `arrow-schema` feature flags: `hymeko_hre/ipc` transitively enables `hymeko_core/ipc` plus its own `iceoryx2` dep, matching the subscriber's `shared_state` dependencies.
- Updated `hymeko_py`: added `hymeko_hre = { path = "../hymeko_hre", features = ["ipc"] }` and rewrote `use hymeko::engine::hypergraphengine::HypergraphEngine;` to `use hymeko_hre::HypergraphEngine;` in `hymeko_py/src/interface_python/api.rs`. `PyHypergraphEngine` continues to wrap the same underlying type — no behavioral change for Python users.
- Verified `cargo check --workspace --all-features`, `cargo test -p hymeko_core` (133/133 passing), `cargo test -p hymeko_hre` (2/2 passing), and `cargo test -p hymeko_query` (93/93 passing, 3 doc-tests ignored as before).
- Decision record: traversal (`HyperGraphView`, `GraphView`, `DeclTreeView`) stays in `hymeko_core` for now. Pulling it into `hymeko_hre` would create a cyclic dependency because ~10 files in `hymeko_core::tensor` (conv, mesh_nn, message_passing, representations) depend on `HyperGraphView`. A follow-up `hymeko_hnn` extraction is the right place to unwind that tangle — tracked in `docs/plans/05_hre_extraction/plan.md § Follow-up`.

## Architecture Diagram Refresh

- Added `architecture/overview_crates.mermaid` — workspace-level crate dependency diagram reflecting the new `hymeko_hre` layer between `hymeko_core` and downstream consumers (daemon, CLI, Python bindings, client).
- Updated `architecture/hre_rewriting_engine/architecture.mermaid`: Layer 1 now distinguishes the compilation-side `hymeko_hre` node from the mutation-side `hymeko_query` rewrite node, with the IR and Blake3 hasher labelled by their owning crate.
- Extended `architecture/README.md` with a new "Crate Dependency Overview" section (pre-Layered view) that inlines the Mermaid render and points readers to `docs/plans/05_hre_extraction/plan.md` for the rationale.

## Planning Artefacts

- `docs/plans/05_hre_extraction/plan.md` — seven-phase plan covering scope, cycle-avoidance rationale, per-crate consumer updates, verification, and commit strategy.
- `docs/plans/05_hre_extraction/features.md` — feature table (F1–F8) for the new crate's public surface, seven code examples covering manual graph building, IR compilation, star/clique expansions, raw-buffer streaming, the subscriber loop, and CLI integration.
- `docs/plans/06_wasm_editor/outline.md` — index of the WASM editor + MCP server spec dropped in `steps/20260418/hymeko_claude_code_spec.md`, with a per-crate status table and the eight-step implementation order.
- `docs/examples/visualizations.md` — six worked visualizations (Mermaid hypergraphs, Graphviz DOT, URDF, tensor shape) for `mini_arm` and `anthropomorphic_arm` fixtures, seeding the visual language the WASM canvas must reproduce.
- `docs/examples/hymeko_to_sysmlv2.md` — end-to-end T2M workflow with a hand-authored ground-truth `mini_arm.sysml` that the future `hymeko_emitter::emit_sysml` must reproduce, plus a metadata-profile encoding table and round-trip invariant.

## Namespace-Alias (`using ... as`) Audit and Test Coverage

- Audited the alias feature across parser, lowering, and resolution: grammar rule (`parser/src/hymeko.lalrpop` line 89), lexer tokens (`Using`, `As`), AST node (`UsingStmt` in `parser/src/ast.rs`), intern pass (`lower_using` in `hymeko_core/src/resolution/intern_pass.rs`), and `apply_usings` integration at `ModuleStore::compile()` step 6b are all present and functional. Fixtures `data/robotics/anthropomorphic_arm_using.hymeko` and `data/robotics/robot_4wh_using.hymeko` compile end-to-end and produce valid URDF/SDF/MJCF/DOT output.
- Identified that the 2026-04-07 changelog's claim of "alias-parity scenarios" in `hymeko_query/tests/test_transform_ecosystem.rs` was aspirational — the fixtures were never referenced in test code.
- Added `mod alias_parity` (11 tests) inside `hymeko_query/tests/test_transform_ecosystem.rs`: per-fixture link-count parity, joint-count parity, link-name set equality, joint-name set equality, URDF `<link>`/`<joint>` count parity, and DOT edge-count parity for both `anthropomorphic_arm_using.hymeko` vs `anthropomorphic_arm.hymeko` and `robot_4wh_using.hymeko` vs `robot_4wh.hymeko`. All 11 pass.
- Added `parser/tests/using_alias.rs` (4 tests) guarding the grammar rule directly: single-alias capture, multi-alias ordering, coexistence with imports, and aliased-path inheritance. All 4 pass.

## Query-Variable `?` Token (T10 slice 1)

- Promoted the `QueryVar` rule in `parser/src/hymeko.lalrpop` to `pub` and added `parse_query_var()` in `parser/src/lib.rs` so the `?name` syntax is now an exercisable parser entry point.
- Added `parser/tests/query_variable.rs` (7 tests): simple parse, underscored / mixed-case idents, rejection of missing `?`, rejection of lone `?`, whitespace tolerance between `?` and the ident, and single-shot rejection of batched `?x ?y`.
- Published `docs/examples/query_variables.md` with five worked examples covering today's standalone parsing, intended future query patterns, rewrite-template integration, and a punch list for the next T10 slice (lift `?x` into `Ref::Var`, thread through `interpret` + engine + template layers).
- Lexer and grammar wiring for `?` was already present (see `parser/src/lexer/common.rs:260` and the extern-token block in the grammar); the dead `QueryVar` production is now reachable and tested rather than decoration.

## State Snapshot

- Added `docs/STATE.md` — point-in-time map of workspace crates, test counts, integrated vs planned features, in-flight uncommitted work, and the prioritised backlog. Intended to be regenerated at each major phase boundary.

## Demo Shell Scripts and Expanded Test Coverage

- Added `scripts/` directory with five executable walk-throughs:
  - `demo_state.sh` — workspace crate list + per-crate test counts + recent changelogs + uncommitted status.
  - `demo_alias_parity.sh` — runs the parser grammar suite + end-to-end parity suite + structural diff between aliased and baseline fixtures.
  - `demo_query_variable.sh` — builds the parser, runs the `?`-token suite, drives `parse_query_var` from a throwaway binary over seven inputs.
  - `demo_hre_extraction.sh` — verifies `hymeko_hre` compiles, runs its tests, confirms core no longer owns `engine`, checks `hymeko_py` imports.
  - `demo_visualizations.sh` — emits all registered transforms for a robotics fixture, optionally rendering DOT → SVG via Graphviz.
  - `scripts/README.md` — index + example execution runs (stdout captures) for each script.
- Expanded `parser/tests/using_alias.rs` from 4 → 11 tests: rejection cases (missing `;`, missing `as`, missing alias ident, reserved-word alias), single-segment and deep-path capture, comment-between-aliases parse.
- Expanded `parser/tests/query_variable.rs` from 7 → 15 tests: digits in name, leading-digit rejection, leading-underscore, line/block comments before `?`, reserved-word rejection, single-char vars, trailing-content rejection.
- Expanded `hymeko_query` alias parity from 11 → 16 tests: added SDF link-count parity, MJCF body-count parity, MJCF hinge-count parity, cross-format SDF parity for `diff_robot`, and a per-link-mass map equality stronger than name-set equality.
- Rewrote `parser/src/lib.rs` for IDE-friendliness: replaced `lalrpop_mod!(pub hymeko)` with an explicit `pub mod hymeko { include!(concat!(env!("OUT_DIR"), "/hymeko.rs")); }` so rust-analyzer can follow the include without needing procedural-macro expansion. Applied lifetime elision on `parse_description` and `parse_query_var`, shortened `std::io::Result` → `io::Result`, split SIMD dispatch into `parse_description_inner` / `parse_query_var_inner` helpers, and added module-level docs plus a runnable doc-test for `parse_query_var`.

## Workspace test tally

`cargo test --workspace`: **347 tests passing** (was 304 pre-changes: +16 alias-parity, +7 parser using-alias, +15 query-variable, +1 doc-test, +4 across existing suites counted separately after test-binary splits), 0 failures, 3 ignored doc-tests (pre-existing).
