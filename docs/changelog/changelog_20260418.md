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

## HRE Ops Consolidation — Expansion + Traversal + Visitor Pattern

- **Expansion moved to `hymeko_hre`.** `star_expansion_coo`, `clique_expansion_coo`, and `star_expansion_coo_normalized` were moved from `hymeko_core::tensor::representations::tensor_coo_representation` to `hymeko_hre::expansion`. Consumer updates: `hymeko_daemon/src/worker.rs` (4 call sites), `hymeko_hre::engine::hypergraphengine_impl` (intra-crate), and six core test files (`tests/test_tensor_representations/*`, `tests/benchmarks/bench_coo_builder_random.rs`, `tests/typical_graphs/fano/tensor_fano.rs`). Added `hymeko_hre` as a dev-dep of `hymeko_core` for test access; the `hymeko_core` library itself remains independent of `hymeko_hre`. `HyperGraphView` intentionally stays in `hymeko_core` — moving it would cycle through the HGNN tensor ops; a proper `hymeko_hnn` extraction is the right forum for that work.
- **Berge traversal in `hymeko_hre::traversal::berge`.** New `berge_bfs` and `berge_dfs` over the bipartite incidence graph defined by `hymeko_core::traversal::hypergraphview::BergeView`. Both routines take `&mut impl HypergraphVisitor` and call `on_enter_node` / `on_enter_edge` / `on_incidence` hooks at each step, with `should_continue()` consulted after every event for early termination.
- **`HypergraphVisitor` trait (L1) in `hymeko_hre::visitor`.** Synchronous trait with default no-op methods. Ships with three concrete visitors: `CountingVisitor` (instrumentation), `TraceVisitor` (records Berge-state order), `PatternMatcherVisitor<F>` (fires `on_pattern_match` via an `FnMut` sink when the traversal enters a designated edge). `ChainVisitor` composes multiple visitors in order, AND-combining their `should_continue` votes — lets a trace + a pattern matcher share one walk.
- **Concurrent / live pattern-matching roadmap.** L1 (sync, this slice) is the trait contract. L2 (`BroadcastVisitor` over `crossbeam_channel::Sender<TraversalEvent>`) and L3 (`tokio::sync::broadcast` / `async_stream`) will land as alternative visitor implementations when needed — the traversal loop already emits everything required.
- **Tests.** `hymeko_hre/tests/test_expansion.rs` (7 tests): hand-built `HyperGraphView` via struct-literal construction, asserting shape + exact `(k,i,j,value)` entries for star, clique, and normalized-star expansion on a 3-node / 1-hyperedge fixture plus a two-edge chain. `hymeko_hre/tests/test_berge_traversal.rs` (6 tests): per-state visit counting, trace order, DFS preorder, `ChainVisitor` composition, `PatternMatcherVisitor` firing once on the targeted edge, and `should_continue()` short-circuit.
- **`hymeko_hre/src/lib.rs`** — added `pub mod expansion; pub mod traversal; pub mod visitor;`.

## Fixture-based HRE Tests (end-to-end through the parser)

- Added `hymeko_hre/tests/common/mod.rs` with a `LalrpopParser` + `load_and_lower()` helper + `view_f32` / `view_f64` builders. Mirrors the minimal parts of `hymeko_core/tests/test_helpers.rs` that hre needs to drive the full `ModuleStore → Ir → HyperGraphView` pipeline without reaching into core's test harness.
- Added `parser` as a `hymeko_hre` dev-dependency so integration tests can drive the LALRPOP grammar end-to-end.
- `hymeko_hre/tests/test_fixture_expansion.rs` (8 tests): loads `mini_arm.hymeko`, `anthropomorphic_arm.hymeko`, and `robot_4wh.hymeko` and asserts star/clique shape invariants, per-slice index validity, per-slice non-empty count for the robot's own joints, sparse-vs-dense ratio, and alias-parity on the expansion side (`anthropomorphic_arm.hymeko` vs `anthropomorphic_arm_using.hymeko` produce equal nnz).
- `hymeko_hre/tests/test_fixture_berge.rs` (7 tests): picks a "first connected node" helper (first incidence of first edge) to avoid the standalone meta-decl nodes that imports contribute, then exercises Berge BFS/DFS, counting visitor vs tracer agreement, pattern matching on reachable target edges, pattern ignoring unreachable-edge IDs, early-exit via `should_continue`, and chained trace + matcher in a single walk.
- Key learning captured in the test doc-comments: real `.hymeko` fixtures import `meta_kinematics.hymeko`, whose type declarations become standalone `Ir` nodes with no edge incidences. Tests must start traversals from a node that is actually incident to at least one hyperedge (the helper `first_connected_node` does this).

## Anthropomorphic-Arm Generation Test Suite

- Added `hymeko_query/tests/test_anthropomorphic_generation.rs` (25 tests) dedicated to `data/robotics/anthropomorphic_arm.hymeko`. Organised by concern rather than by format:
  - **Structural signature (§1):** 7 link-typed nodes (world is a `frame`, not a `link`), 6 revolute + 1 fixed joint, `j_fix` connects world→base_link, serial chain adjacency from world to tool, tree invariant that every link is the child of at most one joint.
  - **Kinematic specifics (§2):** revolute joints use canonical unit axes, exact `(j0=Z, j1=X, j2=Z, j3=X, j4=Y, jtool=Z)` signature, j1's 90° twist preserved, base_link is the heaviest link (25 kg), `joint0_limit` shared-node values are queryable at the IR level.
  - **Control / simulation hyperedges (§3):** `gazebo_sim_system` edge is queryable, `joint_trajectory_controller` node inherits from the template, `sim_control_plugin` `filename` string matches `gz_ros2_control-system`.
  - **URDF / SDF generation via `formats::` (§4):** every expected link + joint present in URDF, URDF emits exactly 1 fixed + 6 revolute, SDF collapses non-fixed to `type="revolute"`, URDF and SDF agree on joint count.
  - **MJCF / DOT via `TransformRegistry` (§5):** body hierarchy matches chain depth (with tolerance for optional world-wrapper), one `<motor>` per revolute joint, DOT emits exactly 7 arrows with the fixed joint dashed, MJCF validator produces no errors on the serial chain.
  - **Determinism (§6):** two URDF runs produce byte-identical output; two extractions produce identical link/joint sequences.
- Discovered and documented two API realities while landing this suite:
  - `TransformRegistry`'s `UrdfTransform` / `SdfTransform` are currently stubs (see the `TODO` comments in `hymeko_query/src/transforms/mod.rs`); the full generators live at `hymeko_query::formats::{urdf::generate_urdf, sdf::generate_sdf}`. The anthropomorphic suite uses the full generators directly and calls the registry only for MJCF + DOT where the registry implementation is complete.
  - `extract_joint_limits` looks for inline `limit_lower` / `limit_upper` child values and does not yet dereference `limit -> joint0_limit;` references. The test documents this gap inline and queries `joint0_limit` at the IR level instead, so the shared-limit fixture pattern remains regression-covered until the extractor catches up.
- Registered the new module in `hymeko_query/tests/mod.rs`.

## Levi-Graph Aliases

- Added `LeviState`, `LeviIter`, `LeviView` type aliases in `hymeko_core::traversal::hypergraphview` alongside the existing `Berge*` types, with doc-comment blocks calling out that Levi (1942), Berge (1973), and König are three names for the same bipartite incidence construction — the code keeps `Berge` as the canonical spelling for historical reasons.
- Added `levi_bfs`, `levi_dfs`, `levi_bfs_from_node`, `levi_bfs_from_edge` as `pub use` re-exports of the `berge_*` traversal functions in `hymeko_hre::traversal::berge`. Zero-cost — they refer to the same code, so either name reads naturally depending on the reader's literature background.
- Module doc-comment in `hymeko_hre/src/traversal/berge.rs` now explains the naming equivalence so future readers don't have to chase the history.

## Logged Output in `test_anthropomorphic_generation`

- Wrapped eight representative tests with `log_test_header` / `log_test_footer` (from `crate::test_helpers`) plus `log::info!` calls that surface the discovered axis signature (`ZXZXYZ`), mass ranking, URDF/SDF/MJCF/DOT census counts, serial-chain parent/child adjacency, and the `j_fix` world→base_link wiring. Visible via `RUST_LOG=info cargo test -p hymeko_query --test integration test_anthropomorphic_generation -- --nocapture`.

## Gazebo (New `gz sim`) Launch Bundle Test

- Added `hymeko_query/tests/test_gazebo_sim_launch.rs` with 3 tests:
  - `generate_gz_sim_launch_bundle_for_moveo` — runs `generate_urdf` on `anthropomorphic_arm.hymeko`, writes `moveo.urdf` (~4.5 KB), a hand-templated SDF 1.8 world with the `gz-sim-physics-system` / `user-commands-system` / `scene-broadcaster-system` plugin triple and a ground_plane (`moveo.world.sdf`, ~1.4 KB), and a ROS 2 `gz_sim.launch.py` (~2 KB) that starts `gz sim`, runs `robot_state_publisher` with the URDF, spawns via `ros_gz_sim::create`, and bridges `/clock` + joint-state topics through `ros_gz_bridge::parameter_bridge`. Bundle lands under `target/test_gz_launch_bundle_moveo/` (or `CARGO_TARGET_TMPDIR/gz_launch_bundle_moveo` when cargo provides it) so it is directly launchable with `ros2 launch gz_sim.launch.py`.
  - `bundle_files_are_referentially_consistent` — parses the generated launch file and asserts every referenced filename actually exists in the bundle directory.
  - `launch_targets_new_gazebo_not_classic` — regression guard that fails if the launch template ever re-introduces `gazebo_ros` / `gazebo_ros_pkgs` / `libgazebo_ros` (classic Gazebo). The new-Gazebo stack (`gz sim`, `ros_gz_sim`, `ros_gz_bridge`) is a project requirement.
- All three tests wrap their bodies with `log_test_header` / `log_test_footer` and log the bundle output path, joint/link tag counts, plugin counts, and the exact `cd` + `ros2 launch` commands needed to run the bundle. `RUST_LOG=info cargo test … -- --nocapture` prints the full trace.

## Gazebo Bundle Path Relocation

- Moved the `test_gazebo_sim_launch` bundle output from `target/test_gz_launch_bundle_moveo/` to **`generated/gazebo_launch/<robot>/`** at the workspace root. Motivation: `target/` gets wiped by `cargo clean` and is deeply nested, whereas `generated/` is top-level, survives `cargo clean`, and reads as the canonical home for any "emitted for external tooling" artefact.
- Path resolution uses `env!("CARGO_MANIFEST_DIR")/..` so it is always the workspace root regardless of where `cargo test` is invoked from.
- Each bundle now ships a generated `README.md` (~1.5 KB) alongside `moveo.urdf`, `moveo.world.sdf`, and `gz_sim.launch.py`. The README spells out prerequisites (`ros-jazzy-ros-gz`, `ros-jazzy-ros-gz-sim`, `ros-jazzy-ros-gz-bridge`, `ros-jazzy-robot-state-publisher`), the regenerate command, and the exact `ros2 launch gz_sim.launch.py` invocation. Written by a fresh `make_readme()` helper so the directory is self-documenting even if someone stumbles into it without project context.
- Added `/generated/` to `.gitignore` so the bundle is discoverable locally but never committed.
- Switched the test's directory prep to idempotent `fs::create_dir_all` + `fs::write` overwrite — previously the two tests would race on `remove_dir_all` under cargo's default parallel runner. No stale-file risk because the bundle has a fixed set of filenames.

## Plan 06 Step 1 — IR Strategy Decision

- Wrote `docs/plans/06_wasm_editor/step1_ir_design.md` documenting the design decision for the WASM editor's IR: **do not** create a separate `hymeko_ir` crate (as the original spec proposed). Instead, keep the compile-time arena IR in `hymeko_core::ir` unchanged, and host an editor-facing slotmap IR + `IRDelta` + bridge inside a new `hymeko_emitter` crate.
- Rationale recorded in the doc: (1) phase separation is the right axis — compile IR wants arena + canonical hash, editor IR wants slotmap + atomic mutations for CRDT gossip; (2) industry precedent (rustc HIR/MIR/Ty, Swift AST/SIL, GCC GENERIC/GIMPLE/RTL); (3) one new crate (`hymeko_emitter`) vs two (+`hymeko_ir`) is the right over-vs-under-split trade at this stage; (4) the bridge between the two IRs is ~200 lines and reversible-by-construction.
- Editor IR shape diverges from the spec in two places: `HyperEdge.incident` carries per-arc signs `(VertexKey, Sign)` (lossless bridge to `SignedRefR`), and a new `IRDelta::Batch` variant supports one-shot bulk apply for CRDT gossip and bridge "rebuild after n edits" patterns.

## Plan 06 Step 2 — `hymeko_emitter` Crate

- Created the new `hymeko_emitter` workspace crate (registered in `Cargo.toml`). Depends on `hymeko_core`, `slotmap` (feature `serde`), `serde`, `thiserror`; dev-dep on `parser` for end-to-end fixture tests.
- `editor_ir.rs` — slotmap-backed `HyMeKoEditorIR` with `VertexKey` / `EdgeKey` / `PatchKey`, `Vertex` / `HyperEdge` / `Patch` / `Attribute` / `AttributeValue` / `Position` / `Sign`, and `IRDelta::{AddVertex, RemoveVertex, AddHyperEdge, RemoveEdge, MoveVertex, UpdateWeight, UpdateSign, AttachAttribute, DetachAttribute, AddPatch, Batch}`. `apply(delta)` implements all variants; `RemoveVertex` also prunes dangling incident references from hyperedges.
- `emit_hymeko.rs` — arena `Ir` → deterministic `.hymeko` text with `robot_name { node … @edge …(+a, -b, -c); }` wrapper. Children/values/weights carry `TODO` markers for Step 2b.
- `emit_sysml.rs` — arena `Ir` → SysML v2: `package { metadata def HyperedgeAnnotation { … }; part def <node>; connection def <edge>_arc_<i>; part <edge>_arc_<i>_arcs { signs, targets }; }`. Matches the contract in `docs/examples/hymeko_to_sysmlv2.md` for the trivially-bracketed structure.
- `emit_rust_stubs.rs` — one `pub trait <PascalName>` per node with `fn process()` placeholder.
- `emit_lean4.rs` — one trivial `theorem <lower>_level_invariant : True := by trivial` per node.
- `bridge.rs` — `to_compiler_ir` / `from_compiler_ir` signatures only; full implementation is Step 2c (property-tested round-trip against `mini_arm.hymeko`).
- `hymeko_emitter/tests/test_editor_ir.rs` (10 tests): `AddVertex`, `RemoveVertex` pruning incident refs, `MoveVertex` position update, `UpdateWeight`, `UpdateSign` in-range + out-of-range, `AttachAttribute` / `DetachAttribute` round-trip, `DetachAttribute` error on missing name, `Batch` ordered apply, `Batch` short-circuit on first error (partial apply is intentional).
- `hymeko_emitter/tests/test_emitters.rs` (6 tests): loads `mini_arm.hymeko` via `ModuleStore`, then asserts `emit_hymeko` produces a wrapped + deterministic output containing `base_link` / `spinner` / `@spin_joint`; `emit_sysml` opens/closes `package MiniArm`, declares the metadata profile, and emits a `part def` per node + `connection def` per arc; `emit_rust_stubs` produces PascalCase `pub trait BaseLink { fn process(&self) … }`; `emit_lean4` produces `import Mathlib` + `theorem base_link_level_invariant : True := by trivial`.

## Plan 06 Step 2b — Full Emitter Round-Trip

- **`emit_hymeko` now walks the decl tree recursively** (`DeclNode::first_child` / `next_sibling`) from every top-level decl (parent == `DeclId::NONE`, kind != `HyperArc`). Emits inline numeric / string / list values, attachment-arc sugar (`visual -> link_geometry;`), tag annotations (`<tag>`), nested child blocks, signed refs with optional weight annotations (`+ base_link [[0.0, 0.0, 0.1], [0.0, 0.0, 0.0]]`). Multiple bases render comma-separated per the grammar.
- **Critical shape fix (same change):** top-level decls are now emitted as sibling `HyperItem`s **outside** the `description_name { }` header block, not nested inside it. The parser's `Description` production reserves the outer `{ … }` for a header block that only accepts imports / usings / simple statement nodes — so the previous emitter's output failed round-trip with an `UnrecognizedToken { Colon }` error on the first `name: bases { … }` child. Documented in the emitter's module doc-comment.
- **`emit_sysml` expanded** to walk the decl tree for per-node inline values, emitting `attribute <child> :>> <value>;` lines for scalar / list children and a second `metadata def JointOrigin { xyz, rpy }` profile matching `docs/examples/hymeko_to_sysmlv2.md`.
- **Round-trip regression test:** new `emit_hymeko_roundtrips_through_the_parser` loads `mini_arm.hymeko`, emits it, re-parses the emitted text via `parser::parse_description`, and asserts the round-tripped AST has ≥ the original node/edge count. Four companion tests cover signed-ref sign preservation, inline numeric value preservation, insertion-order byte stability (double-emission byte equality), and round-trip determinism.
- **SysML companion tests:** `emit_sysml_inlines_node_scalar_attributes` (asserts `attribute mass :>> 5.0;` + origin tuple `(0.0, 0.0, 0.05)`); `emit_sysml_metadata_profile_includes_joint_origin` (asserts the `HyperedgeAnnotation` + `JointOrigin` metadata defs with the `xyz : ScalarValues::Real[3]` attribute).
- **Net test growth:** `hymeko_emitter` went from 16 → 24 passing tests (+8: 4 emit_hymeko round-trip/content, 2 SysML scalar & metadata, 2 emit_hymeko stability variants).

## Workspace test tally

`cargo test --workspace`: **427 tests passing** (was 419 before Step 2b: +8 emitter round-trip / content / SysML expansion), 0 failures, 3 ignored doc-tests (pre-existing).

## Plan 06 Steps 3–6 — WASM + Server + MCP + Wire Crates

Four new workspace crates landed in a single batch. Each compiles and ships passing tests; together they trace the spec in `steps/20260418/hymeko_claude_code_spec.md` from Step 3 through Step 8.

### `hymeko_wasm` (Step 3)

- `src/session.rs` — native-portable `EditorSession` wrapping `HyMeKoEditorIR` with CBOR export/import, JSON snapshot, summary counts, add_vertex / add_hyperedge / move / attach / reset / apply.
- `src/wasm.rs` — wasm-bindgen façade exposing the same surface to JavaScript, gated behind `cfg(target_arch = "wasm32")` so native `cargo test` is unaffected. Build: `wasm-pack build hymeko_wasm --target web`.
- `tests/test_session.rs` — 9 tests: empty init, add/move/attach, CBOR round-trip, JSON snapshot, reset, batch apply.

### `hymeko_server` (Step 4)

- Axum 0.7 router: `GET /health`, `GET /api/workspace`, `GET/POST /api/files/:name`, static service at `/static`. CORS permissive. Path-traversal guard on filename joins.
- `bin/hymeko_server` binds `127.0.0.1:3000` (override via `HYMEKO_SERVER_ADDR`); workspace root via `HYMEKO_WORKSPACE`.
- `tests/test_api.rs` — 5 tests via `tower::ServiceExt::oneshot`: `/health`, empty workspace, populated workspace listing (`.hymeko` + `.sysml` only), POST/GET round-trip, path-traversal rejection.

### `hymeko_mcp` (Step 5)

- Plain JSON-RPC 2.0 MCP implementation — **no `rmcp` dependency**. `src/protocol.rs` defines the JSON-RPC envelope types; `src/server.rs` provides `McpServer::handle_request(json) -> json` and a `bin/hymeko_mcp` driver that reads newline-delimited requests from stdin and writes responses to stdout (the shape Claude Code's `.claude/mcp.json` expects).
- Six tools: `add_vertex`, `add_hyperedge` (references vertices by name via internal reverse index), `snapshot`, `summary`, `reset`, `export_cbor` (base64-encoded). `initialize` and `tools/list` return the protocol version and the `ToolDescriptor` catalogue.
- `tests/test_mcp.rs` — 9 tests: initialize handshake, tools/list returns all six, add_vertex + summary flow, add_hyperedge with name-indexed vertices, unknown method (→ `-32601`), unknown tool (→ `-32603`), sign-vector length mismatch, malformed JSON (→ `-32700`), reset clears state.

### `hymeko_wire` (Step 6)

- `PacketHeader` (`#[repr(C, packed)]`, 28 bytes) with magic `0x484D4B4F` "HMKO", version 1, flags (bit `0x01` = zstd), 8-byte `patch_id`, 8-byte `delta_seq`, 4-byte xxh3_32 checksum of the (post-compression) payload.
- `encode_delta(delta, patch_id, seq, compress)` runs CBOR → optional zstd → xxh3_64 → header + payload. `decode_delta(packet)` validates magic + version, verifies checksum, decompresses, CBOR-decodes back to `IRDelta`.
- `tests/test_wire.rs` — 7 tests: compressed round-trip, uncompressed round-trip, bad magic rejection, checksum mismatch, too-short packet, magic-prefix assertion, 500-delta `Batch` round-trip exercising the zstd path.

## Workspace test tally

`cargo test --workspace`: **457 tests passing** (was 427 before Steps 3–6: +9 hymeko_wasm, +5 hymeko_server, +9 hymeko_mcp, +7 hymeko_wire), 0 failures, 3 ignored doc-tests (pre-existing).

## Plan 06 Step 2c — Bridge Round-Trip + Session Emitters

- **`hymeko_emitter::bridge::to_compiler_ir`** walks a `HyMeKoEditorIR` and produces a fresh `hymeko::ir::ir::Ir`: one `decl_nodes` entry per editor vertex (kind `Node`) + `NodeRec`, one per hyperedge (kind `Edge`) + `EdgeRec` with a single anonymous `HyperArc` decl and matching `ArcRec`. Sign discipline (`Plus` / `Minus` / `Neutral`) is preserved in `SignedRefR::{Plus,Minus,Neutral}`. Layout is flat (`parent == DeclId::NONE` for every decl) — editor IR does not track containment yet.
- **`from_compiler_ir`** does the reverse walk: each `NodeRec` becomes a `Vertex` (with default editor metadata — `level = 0`, empty attributes, no position — because the arena IR does not retain those editor-only fields), each `EdgeRec` flattens its arcs into a single `HyperEdge` with concatenated `(VertexKey, Sign)` incidents, filtering out edge-to-edge refs that the editor cannot represent.
- **Structural round-trip contract** (documented inline): `from_compiler_ir(to_compiler_ir(editor))` preserves vertex-name set, hyperedge-name set, and per-edge multiset of `(vertex_name, sign)` — slotmap keys are *not* preserved (fresh allocation on each pass) and editor-only metadata fields are dropped.
- **`hymeko_emitter/tests/test_bridge.rs`** (6 tests): basic editor round-trip, empty editor case, multi-edge sign-preservation (`+/-` vs `~/~` on distinct edges), `mini_arm.hymeko` project-and-back via the full `ModuleStore` pipeline preserving node/edge counts, vertex-and-edge count parity after arena→editor→arena, and edge-to-edge ref filtering on the reverse path.
- **`EditorSession::emit_{hymeko,sysml,rust_stubs,lean4}`** wired through the bridge — each call creates a fresh `Interner`, projects to a fresh arena `Ir`, and hands off to the matching `hymeko_emitter` emitter. Output is deterministic because both the bridge and the emitters are insertion-order-stable.
- **`hymeko_wasm/tests/test_session.rs` +5 tests**: `emit_hymeko` roundtrips from an editor session (result parses via `parser::parse_description`), `emit_sysml` wraps in `package Demo { … }` with `part def` per vertex, `emit_rust_stubs` produces PascalCase `pub trait`s, `emit_lean4` produces trivial theorems, and all four emit methods are deterministic across repeated calls.

## Workspace test tally

`cargo test --workspace`: **468 tests passing** (was 457 before Step 2c: +6 bridge round-trip + +5 session emit front-ends), 0 failures, 3 ignored doc-tests (pre-existing).
