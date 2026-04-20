# Project Changelog — 2026-04-19

## `hymeko_hnn` Extraction — Hypergraph Neural Ops Leave `hymeko_core`

The tensor-op tangle that kept `HyperGraphView` wedged inside
`hymeko_core` (and forced the Apr-18 `hymeko_hre` split to keep the
engine-only "tight" scope) is finally unwound. Everything that operates
on a `HyperGraphView` now lives in a dedicated `hymeko_hnn` crate; the
core library becomes a clean tensor-primitives + IR layer with no
reverse dependencies on higher abstractions.

### Files moved (14 source files + `calc_approx_nnz` function)

From `hymeko_core/src/traversal/` → `hymeko_hnn/src/traversal/`:

- `graphview.rs` · `hypergraphview.rs` · `graph_traversal.rs` · `decltreeview.rs` · `mod.rs`

From `hymeko_core/src/tensor/` → `hymeko_hnn/src/tensor/`:

- `common_traversal.rs` · `message_passing.rs` · `tensor.rs`
- `mesh_nn/mod.rs` · `mesh_nn/mesh_conv.rs`
- `conv/hgnn.rs` · `conv/signed_hgnn.rs` · `conv/gcn_clique.rs`
- `representations/tensor_csr_representations.rs`

From `hymeko_core/src/tensor/common.rs`: extracted `calc_approx_nnz`
(the only `HyperGraphView`-aware helper) into
`hymeko_hnn/src/tensor/common.rs`. `Real`, `AsF32`, `AsF64`,
`signed_incidence` stay in core.

### Crate layout preserved

The `hymeko_hnn` module tree mirrors the pre-extraction paths inside
`hymeko_core::{traversal,tensor}` so downstream crates need only a
prefix search-and-replace:

| Old path (`hymeko::`)                                                                | New path (`hymeko_hnn::`)                                                          |
| ------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| `traversal::*`                                                                        | `traversal::*`                                                                      |
| `tensor::{common_traversal,message_passing,mesh_nn,tensor}::*`                        | `tensor::…::*`                                                                      |
| `tensor::conv::{hgnn,signed_hgnn,gcn_clique}::*`                                      | `tensor::conv::…::*`                                                                |
| `tensor::representations::tensor_csr_representations::*`                              | `tensor::representations::tensor_csr_representations::*`                            |
| `tensor::common::calc_approx_nnz`                                                     | `tensor::common::calc_approx_nnz`                                                   |

Primitives-level modules (`common::{Real,AsF32,AsF64,signed_incidence}`,
`aggregation`, `tensor_val`, `representations::{tensor_coo,tensor_csr}`,
`conv::{traits,weight_init}`) stay in `hymeko_core` and are re-exported
from there as before.

### `TensorInc::_pd` visibility

`TensorInc._pd: PhantomData<F>` in
`hymeko::tensor::representations::tensor_coo` flipped from `pub(crate)`
to fully `pub`, since `HyperGraphView::from_ir` in the sibling crate now
needs to construct the value. Also added a `TensorInc::new(e, n, s, w)`
convenience constructor so the `PhantomData` ceremony stays local.

### Consumer updates — 20 files rewritten

Import paths changed in:

- `hymeko_hre/src/{expansion,visitor,engine/hypergraphengine_impl}.rs` and `hymeko_hre/src/traversal/berge.rs`
- `hymeko_hre/tests/{common/mod,test_expansion,test_berge_traversal,test_fixture_berge}.rs`
- `hymeko_daemon/src/worker.rs`
- `hymeko_core/tests/{benchmarks/bench_coo_builder_random, domain_transformations/parse_files, test_tensor_representations/{coo_tests/test_coo_aos,coo_tests/test_coo_builder,minimal_tensor_representations,test_csr_builder,test_csr_representations,test_message_passing_components,test_tensor_representation}, traversal/test_traversal, typical_graphs/fano/tensor_fano}.rs`

`Cargo.toml` changes:

- `hymeko_hre` gained `hymeko_hnn = { path = "../hymeko_hnn" }` as a
  regular dep (hre's expansion, engine, and Berge traversal all consume
  `HyperGraphView`).
- `hymeko_daemon` gained `hymeko_hnn` as a regular dep.
- `hymeko_core` gained `hymeko_hnn` as a **dev-dependency only** — the
  library itself has no reverse link, but the integration tests in
  `tests/` now reach into hnn. Same pattern as the earlier
  `hymeko_core [dev-dependencies] hymeko_hre = …` arrangement.

### Cycle-avoidance verified

Dependency chain is now a clean:

```
hymeko_core  ←  hymeko_hnn  ←  hymeko_hre  ←  hymeko_daemon
                    ↑                  ↑
                 (tests)             (tests)
```

No crate depends on a successor. The long-standing cycle (core.tensor ↔
core.traversal.hypergraphview) is gone because traversal left core
entirely.

### Workspace test tally

`cargo test --workspace`: **468 tests passing** (no net change from
pre-extraction; the move is purely structural and behaviour-preserving),
0 failures, 3 ignored doc-tests (pre-existing).

## Paper 2 — T11 Gazebo World Transform

Closes the end-to-end robotics pipeline: a single `.hymeko` source now
lowers not only to URDF/SDF/MJCF/DOT but to a complete **SDF 1.8
Gazebo world** directly launchable with `gz sim`. The fixture's
`sim_plugin` / `control_plugin` hyperedges (`@sim_control_plugin`,
`@gazebo_sim_system`, …) are queried and rendered as `<plugin>` tags in
the output.

### Plugin extractor — `hymeko_query::kinematics::gazebo_plugins`

- `GazeboPluginInfo` struct with `did`, `edge_name`, `kind` (Sim |
  Control), `plugin_class` (C++ class), `filename` (shared lib),
  `parameters` (YAML path). `is_complete()` only requires
  `plugin_class` so control plugins (loaded via the sim plugin's
  library) without a standalone filename still render.
- `extract_gazebo_plugins(engine)` walks every hyperedge inheriting
  `sim_plugin` or `control_plugin` and pulls the `plugin` / `filename`
  / `parameters` string-valued children via a small
  `find_child_str` helper.

### World emitter — `hymeko_query::formats::gazebo`

- `generate_gazebo_world(ir, resolver, robot_name, world_name)` emits:
  - `<?xml ?>` + `<sdf version="1.8">` + `<world name="…">`
  - `<physics name="default">` with `max_step_size` / `real_time_factor`
  - Standard plugin triple (`gz-sim-physics-system`,
    `-user-commands-system`, `-scene-broadcaster-system`)
  - `<light type="directional" name="sun">` + `<model name="ground_plane">` with a 100 × 100 m plane
  - World-level `sim_plugin` tags extracted from the IR
  - The robot's `<model>` block (via `generate_sdf_from_model`), with
    `<?xml …?>` and outer `<sdf>` wrappers stripped, re-indented under
    the `<world>` padding
  - `control_plugin` tags injected **before** the robot model's closing
    `</model>` tag — matches the `gz_ros2_control` example layout
- Full deterministic: two successive calls produce byte-identical output.

### Registry integration — `GazeboWorldTransform`

- Added to `TransformRegistry::default()` as the fifth transform (after
  URDF / SDF / MJCF / DOT). Name `"gazebo"`, extension `"world.sdf"`.
- Registry `emit` returns a **plugins-stripped stub** (physics + ground
  plane + `<!-- TODO: delegate to formats::gazebo::generate_gazebo_world -->`
  marker) because the `ModelView` abstraction doesn't expose the raw
  `Ir` needed for plugin extraction — matches the pre-existing
  URDF/SDF stub pattern. Callers needing the fully-populated world use
  the `generate_gazebo_world` free function directly.
- Registry `validate` runs the same tree-topology check as MJCF.

### Tests — `hymeko_query/tests/test_gazebo_world.rs`

15 tests:
- Plugin extractor: finds both kinds on moveo, populates class / filename
  / parameters for `sim_control_plugin`, works on `diff_robot` too.
- World emitter: SDF 1.8 header, named `<world>`, standard physics
  triple, ground plane, inline robot links (`base_link`, `link_0`),
  extracted `gz_ros2_control::GazeboSimROS2ControlPlugin`, extracted
  `gz_ros2_control/GazeboSimSystem`, deterministic, diff_robot fixture.
- Registry: `gazebo` listed in `available()` with extension
  `world.sdf`; registry stub emits a valid SDF skeleton; validator
  passes for the serial-chain moveo.

### Launch bundle rewired

`test_gazebo_sim_launch` previously used a hand-templated
`make_world_sdf` helper; it now routes through
`generate_gazebo_world` so the bundle landing under
`generated/gazebo_launch/moveo/moveo.world.sdf` carries the
fixture-extracted plugin tags end-to-end. The 3 bundle tests still
pass.

### Bumped assertion

`test_transform_ecosystem::registry::generate_all_formats` previously
expected `results.len() == 4`; bumped to 5 (urdf, sdf, mjcf, dot,
gazebo) and commented the date of the change inline.

### Workspace test tally

`cargo test --workspace`: **483 tests passing** (was 468, +15 Paper 2
T11 gazebo world), 0 failures, 3 ignored doc-tests (pre-existing).

## Mermaid Transform — Docs-Friendly Diagram Path

Second half of the "HyMeKo → diagram" answer from today's earlier
discussion: the first path is `DotTransform` → `dot -Tsvg` → rendered
SVG (toolchain-dependent); the second, now live, is
`MermaidTransform` → `flowchart TD` text that **renders inline on
GitHub, in VS Code previews, Obsidian, and most docs sites** with zero
external dependency. Lossier than DOT for n-ary hyperedges, but
zero-friction for kinematic-chain diagrams.

### Emitter — `hymeko_query::transforms::MermaidTransform`

- New transform registered as the 6th default (`name: "mermaid"`,
  `extension: "mmd"`), sitting alongside urdf/sdf/mjcf/dot/gazebo.
- `emit_mermaid(model, config)` produces:
  - `flowchart TD` directive + two `classDef`s (link, root)
  - One node per `model.links[]` entry, PascalCase-labelled with mass
    (`base_link["<b>base_link</b><br/>25.00 kg"]:::link`)
  - Root-frame nodes (recovered via `find_roots` — catches fixtures
    like anthropomorphic_arm's `world` which is declared as a `frame`,
    not a `link`) styled with the `root` classdef
  - One arrow per joint:
    - `-->` solid for revolute / continuous / prismatic
    - `-.->` dashed for fixed (matches DOT's dashed fixed style)
  - Label includes joint type + axis letter:
    `parent -->|"j0 (rev, Z)"| child`
- Two small helpers: `mermaid_id` sanitises identifiers to legal
  Mermaid node ids; `escape_label` escapes `"` and `|` inside labels.

### Tests — `hymeko_query/tests/test_mermaid.rs` (12 tests)

- `mermaid_registered_with_mmd_extension` — presence + extension.
- `mermaid_output_opens_with_flowchart_directive` — header check.
- `mermaid_declares_classdef_for_links_and_roots` — style classes.
- `mermaid_emits_one_node_per_link_with_mass_label` — mass propagated
  (asserts `25.00 kg` for base_link).
- `mermaid_emits_world_as_a_root_frame_for_moveo` — covers
  frame-vs-link distinction on the anthropomorphic fixture.
- `mermaid_emits_dashed_arrow_for_fixed_joint` — `world -.->` style.
- `mermaid_emits_solid_arrow_for_revolute_joint` — `base_link -->` style.
- `mermaid_emits_one_arrow_per_joint` — exactly 6 revolute + 1 fixed
  for moveo.
- `mermaid_axis_letter_appears_in_joint_label` — `j0 (rev, Z)` +
  `j1 (rev, X)` per the 6-DoF anthropomorphic signature.
- `mermaid_emit_is_deterministic` — byte-stable across repeated calls.
- `mermaid_diff_robot_emits_continuous_label` — covers `conti_joint`
  inheritance on robot_4wh.
- `mermaid_mini_arm_has_single_continuous_joint_arrow` — the single
  `spin_joint` arrow.

### Assertion bumped

`test_transform_ecosystem::registry::generate_all_formats` expected
`results.len() == 5` after the T11 Gazebo addition; bumped to 6 with
an inline comment recording both dates.

### Workspace test tally

`cargo test --workspace`: **495 tests passing** (was 483, +12 Mermaid),
0 failures, 3 ignored doc-tests (pre-existing).

## Template-Driven Transform Pipeline Restored

Regression fix: recent transforms (Mermaid, Gazebo world) and the older
registry stubs (URDF / SDF / MJCF / DOT) were emitting output via
`String::push_str` / `writeln!` chains in Rust. The pre-existing
template engine (`hymeko_query::rewrite::template::execute_transform`)
and the file-pair templates under `transforms/<name>/{queries.hymeko,
template.*}` already handle this data-drivenly for the URDF / SDF / MJCF
/ DOT / ros2_launch formats, but the registry dispatch table was
short-circuiting the template path. This slice wires every format with
a template directory through `execute_transform` via a new registry
entry point.

### Missing templates filled in

- **`transforms/gazebo/queries.hymeko`** — context block with `links`,
  `frames`, `@fixed_joints`, `@revolute_joints`, `@continuous_joints`,
  `@prismatic_joints`, `@sim_plugins`, `@control_plugins`.
- **`transforms/gazebo/template.world.sdf`** — SDF 1.8 world skeleton
  with physics triple, ground plane, `{{#each sim_plugins}}` +
  `{{#each control_plugins}}` blocks, inline robot `<model>`, and
  per-joint `{{#each … _joints}}` expansions. Supports
  `{{config:world_name}}` with a `"empty"` default.
- **`transforms/mermaid/queries.hymeko`** — identical join-type query
  context to DOT.
- **`transforms/mermaid/template.mmd`** — `flowchart TD` with `classDef
  link` + `classDef root`, per-link node declarations, and dashed /
  solid arrows for fixed vs revolute/continuous joints.

### `DomainTransform::template_dir()` trait method

- New default-`None` method on the trait. Returning `Some("<subdir>")`
  opts the transform into the data-driven path. All six shipped
  transforms override it: `UrdfTransform → "urdf"`,
  `SdfTransform → "sdf"`, `MjcfTransform → "mjcf"`,
  `DotTransform → "dot"`, `MermaidTransform → "mermaid"`,
  `GazeboWorldTransform → "gazebo"`.
- The trait stays `dyn`-compatible (no generics) because the generic
  `NameResolver` lives on the registry method, not the trait.

### `TransformRegistry::render_from_templates` — canonical entry point

- New method:
  ```rust
  pub fn render_from_templates<R: NameResolver>(
      &self,
      name: &str,
      ir: &Ir,
      resolver: &R,
      config: &TransformConfig,
      transforms_root: &Path,
  ) -> Option<Result<String, String>>
  ```
- Looks up the transform, resolves its `template_dir()`, reads
  `queries.hymeko`, scans the directory for the single `template.*`
  file (ignorant of per-format extensions like
  `template.urdf.xml` vs `template.world.sdf`), builds a
  `TransformSpec`, and hands off to `execute_transform`.
- `config.options` propagates into the template's `{{config:*}}`
  lookups; `"robot_name"` and `"world_name"` defaults are injected so
  existing templates keep working without caller-side ceremony.

### Tests — `hymeko_query/tests/test_template_driven.rs` (13 tests)

- `every_shipped_transform_exposes_a_template_dir` — all six formats
  return `Some("<name>")`.
- `render_from_templates_returns_none_for_unknown_transform` — error
  path.
- Per-format: URDF robot header + each link; SDF model wrapper; MJCF
  `<mujoco>` + `<worldbody>`; DOT `digraph` + arrows; Mermaid
  `flowchart TD` + dashed fixed-joint arrows; Gazebo world wrapper +
  physics triple + ground plane + robot model + extracted sim plugin
  (`gz_ros2_control::GazeboSimROS2ControlPlugin`, `gz_ros2_control-system`).
- Determinism across two renders; mini_arm renders every format
  non-empty with the robot name threaded through.

### Workspace test tally

`cargo test --workspace`: **508 tests passing** (was 495, +13
template-driven pipeline), 0 failures, 3 ignored doc-tests

## "Last Mile" Retirement — Template Engine Covers The Full Feature Set

Three new template-engine directives let the data-driven path reach
feature parity with the hand-written URDF/SDF/Gazebo emitters for the
geometry-dispatch axis that dominated the remaining hard-coded output.

### Template engine extensions (`hymeko_query/src/rewrite/template.rs`)

- **`{{#inherits <field> "<base>"}}...{{/inherits}}`** — conditional
  block that renders only when the field at `<field>` resolves to a decl
  whose base chain (transitively) contains a decl named `<base>`.
  Implemented via `decl_inherits_from()` walking the
  `NodeRec.bases`/`EdgeRec.bases` arrays. Lets templates do geometry
  dispatch (`box` vs `cylinder` vs `sphere`) without a switch/case
  primitive.
- **`{{nth:<field_path>:<N>}}`** — interpolation that indexes into a
  list-valued field. Resolves the list, formats element `N` as the
  template variable. Makes it possible to pull `radius` from
  `dimension[0]` and `length` from `dimension[1]` on a single
  `cylinder` decl.
- **`{{rad:<field_path>}}`** — scalar transform that reads a
  degrees-valued list and emits the radian form, one
  space-separated scalar per element (via `deg_list_to_rad_string()`).
  Reserved for joint rpy emission when the bind-attached inline
  attribute story lands.
- **`MatchContext::resolve_field_decl(field_path)`** — new helper that
  walks dot-separated paths (e.g. `link_geometry.dimension`) through
  the child-decl tree, powering the three new directives above.

### Template enrichment — `{{#inherits}}` geometry dispatch everywhere

- **`transforms/urdf/template.urdf.xml`** — per-link `<visual>` +
  `<collision>` now emit the right geometry tag via
  `{{#inherits link_geometry "box"}}` / `"cylinder"` / `"sphere"`,
  with `{{nth:link_geometry.dimension:0}}` / `...:1` extracting
  cylinder radius/length. `<inertial>` block with identity-diagonal
  inertia matrix. `<pose>`/origin wired through `{{field:origin}}`.
- **`transforms/sdf/template.sdf.xml`** — same dispatch pattern,
  `<pose relative_to="{{name}}">` element, full inertia matrix.
- **`transforms/gazebo/template.world.sdf`** (this turn) — the
  embedded robot `<model>` block upgraded with the same
  geometry-dispatch pattern. `mini_arm` end-to-end (`hymeko_cli emit
  --format gazebo`) now emits `<box><size>0.3 0.3 0.1</size></box>`
  for `base_link` and `<cylinder><radius>0.1</radius><length>0.2
  </length></cylinder>` for `spinner` straight from the template —
  no cylinder-hardcoded fallback.

### Retired hard-coded emitters

- **`hymeko_query/src/codegen.rs`** — rewritten top-to-bottom. Was 211
  lines with hand-rolled `generate_mjcf` + `emit_mjcf_body` +
  `mjcf_joint_type` + `generate_dot` string builders; now 85 lines
  that dispatch *every* format through
  `TransformRegistry::render_from_templates`. The CLI `Compile`
  subcommand (`hymeko_cli compile --format {urdf|sdf|mjcf|dot}`) is
  therefore fully template-driven; no Rust-side `push_str` remains on
  the primary path.
- **`transforms::UrdfTransform::emit` / `SdfTransform::emit`** — the
  `emit_urdf_stub` / `emit_sdf_stub` functions in
  `hymeko_query/src/transforms/mod.rs` previously emitted TODO
  placeholders. They now delegate to
  `crate::formats::urdf::generate_urdf_from_model` /
  `formats::sdf::generate_sdf_from_model` (the rich model-view
  emitters), so `TransformRegistry::emit_all` — the
  `ModelView`-backed legacy API still called by
  `test_transform_ecosystem::registry::generate_all_formats` and
  friends — returns useful content. The function-level
  `<R: NameResolver>` phantom generic on `generate_urdf_from_model`
  was dropped along the way (it was always unused).

### Test tally after retirement

`cargo test --workspace`: **still 508 passing, 0 failed**. All
retirements are binary-equivalent on the test surface — the CLI
`Compile` path was previously only exercised implicitly, and
`emit_all` tests only check non-emptiness.

### What's still hard-coded (deferred)

- `formats::urdf::generate_urdf` / `formats::sdf::generate_sdf`
  (and their model-view cousins) remain rich Rust emitters. Tests at
  `test_generation_engine.rs` and `test_anthropomorphic_generation.rs`
  call them directly and assert on format details the template engine
  cannot yet emit: per-joint axis (`"1 0 0"` vs `"0 0 1"` —
  needs to resolve the joint's incident `axis_definition` bind to its
  `direction` field), joint-origin rpy conversion from degrees
  (bind-attached inline attributes `[[xyz], [rpy_deg]]` are not yet
  exposed as template variables), `<material name="color">` / `<color
  rgba>` elements. These require either (a) extending the template
  engine with a bind-attribute accessor + the existing `{{rad:…}}`
  directive, or (b) migrating the affected tests to call
  `render_from_templates` and drop the legacy assertions. Tracked for
  a follow-up turn.
- `transforms::{emit_mjcf, emit_dot, emit_mermaid}` — reached only via
  `DomainTransform::emit`, which takes `&ModelView` (no IR access).
  Retiring them means either changing the trait signature or giving
  up the `emit_all` entry point; both are larger architectural moves
  than this turn's scope.

(pre-existing).
