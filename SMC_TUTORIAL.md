# HyMeKo — SMC 2026 Companion Tutorial

This document is the reproducibility and reader's companion for the paper

> *HyMeKo: A Canonical-Hypergraph IR Pipeline for Multi-Target Code Generation*,
> submitted to IEEE SMC 2026.

It walks through the artefacts the paper refers to (fixtures, templates, the
benchmark harness, the emitted Gazebo bundle) and points to the code paths
that realize the constructions described in the paper.

## 1. What the paper points to in this repository

| Paper section | Repository location |
|---|---|
| Definition 1 (HyMeKo structure) | `hymeko_core/src/` — the typed, signed hypergraph IR |
| Pipeline: compile / project / emit | `hymeko_cli/` (dispatch), `hymeko_core/`, `hymeko_emitter/` |
| Template dispatcher | `hymeko_emitter/` |
| Per-format templates | `hymeko_emitter/templates/` |
| Named queries (Section V) | `hymeko_query/` |
| Robot fixtures | `hymeko_query/tests/fixtures/` and `data/` |
| Benchmark harness | `hymeko_query/tests/bench_workflow.rs` |
| Raw benchmark timings | `paper/smc2026/data/workflow_benchmark.csv` |
| End-to-end Gazebo bundle | `generated/` (generated artefacts) |

## 2. Environment the paper measurements were taken on

- CPU: AMD Ryzen 7 3700X (8 cores, boost 4.2 GHz), single-threaded
- OS: Linux 6.17 x86_64
- Toolchain: `rustc 1.92` (stable), release profile, AVX2 lexer back-end
- Build: `cargo build --release`
- Repository revision used for the paper: `17e51a8`

Any modern x86_64 or Apple Silicon machine with a recent stable `rustc` will
reproduce the qualitative claims of the paper (sub-millisecond end-to-end
generation, ~70% compile share, 40–350 MiB/s emitter throughput). Absolute
timings are cache-sensitive; the paper reports medians over 30 iterations
per fixture and frames the measurements as a feasibility demonstration.

## 3. Reproducing the workflow benchmark (Table I, Figure 4)

The benchmark harness lives at `hymeko_query/tests/bench_workflow.rs`. It:

1. Loads each fixture (`mini_arm`, `anthropomorphic_arm`, `robot_4wh`, plus
   two alias-variants).
2. Runs `compile` to produce the canonical IR `H`.
3. Applies every emitter `ε_f` for `f ∈ {URDF, SDF, Gazebo-world, MJCF, DOT, Mermaid}`.
4. Records per-stage wall-clock times.
5. Repeats 30 times per fixture.

The raw output lands in `paper/smc2026/data/workflow_benchmark.csv` (150
rows: 30 iterations × 5 fixtures). Tables and figures in the paper are
rendered from this file.

## 4. Reproducing the end-to-end Gazebo demonstration (Section VI-D)

From the `anthropomorphic_arm` source, HyMeKo emits a complete ROS 2 launch
bundle against `gz sim`:

- a URDF file,
- an SDF world with the `gz-sim-physics-system`, `gz-sim-user-commands-system`,
  and `gz-sim-scene-broadcaster-system` plugin triple,
- a Python launch script wired through `ros_gz_sim` and `ros_gz_bridge`.

A regression guard in `hymeko_query/tests/test_gazebo_sim_launch.rs` fails
the build if the launch template ever reintroduces the legacy `gazebo_ros`
stack.

## 5. Reading the code as the paper describes it

- **Canonical hash.** The content-addressable Blake3 digest (Proposition 2)
  is computed over a canonical left-to-right traversal of the IR; see
  `hymeko_core/src/` for the traversal and digest.
- **Template language.** The three constructs discussed in Section IV-D
  (`repeat q`, `inherits q { … }`, attribute interpolation) are implemented
  by the single dispatcher in `hymeko_emitter/`. A new target format is a
  new template in `hymeko_emitter/templates/`, not a new program.
- **Named queries.** The predicate algebra used by templates (is-link,
  is-joint, inherits-from, has-tag, has-child, has-ref) lives in
  `hymeko_query/`. The kinematic-link extractor worked example in
  Section V-A is the concrete witness.

## 6. Extending HyMeKo

To add a new target format *f*:

1. Author a template `f.hymeko-template` in `hymeko_emitter/templates/`
   using `repeat`, `inherits`, and attribute interpolation over the named
   queries.
2. Register *f* in the emitter dispatch table.
3. Add a fixture-level invariance test asserting the format-specific
   invariants (e.g. link count, joint count) under the shared query bundle.

No dispatcher change is required; Propositions 1–3 lift automatically.

## 6.1  Compile-time constants and arithmetic (Tier B, 2026-04-20)

A description header may declare numeric constants, optionally with
arithmetic expressions, that resolve to `f64` literals before the IR is
built. Constants subsume `xacro:property` and `${expr}` for the
numerical-parameterization case:

```hymeko
mini_arm_description {
    @"meta_kinematics.hymeko";
    using kinematics.elements as el;
    using kinematics.geometry as geo;
    using kinematics.axes as ax;

    const RADIUS      = 0.05;
    const LINK_HEIGHT = 0.1;
    const LENGTH      = LINK_HEIGHT * 2.0;     // arithmetic in expr
    const ORIGIN_Z    = LINK_HEIGHT / 2.0;
}

mini_arm: el, geo, ax {
    base: el.link {
        mass 5.0;
        link_geometry: geo.cylinder { dimension [RADIUS, LENGTH]; }
        visual    -> link_geometry;
        collision -> link_geometry;
        origin [0.0, 0.0, ORIGIN_Z];
    }
    spinner: el.link {
        mass 1.0;
        link_geometry: geo.cylinder {
            dimension [(RADIUS * 0.5), (LENGTH / 2.0)];   // expr in list
        }
        visual    -> link_geometry;
        collision -> link_geometry;
        origin [0.0, 0.0, ORIGIN_Z];
    }
}
```

**Surface rules.**
- `const NAME = <expr>;` — bindings live in the description header
  block (alongside `using`/`@"…"`).
- A bare `NAME` in numeric value position (e.g.\ `mass NAME;`,
  `origin [NAME, 0.0, 0.0]`) refers to the binding. No syntax change
  for the simple case.
- Compound expressions go in parentheses: `(NAME * 2.0)`,
  `(LEN + 0.05)`. Operators: `+ - * /`, unary `-`. Parens nest.
- Builtins: `pi` (constant) and `exp(<expr>)` (natural exponential).
  More builtins are easy to add as needed.
- Forward references between consts are allowed and resolved
  topologically; cycles are reported at compile time.

**Errors caught at compile time.**
- Undefined identifier — `const A = UNKNOWN;` fails with
  `undefined const reference 'UNKNOWN'`.
- Cycle — `const A = B; const B = A;` fails with `cyclic const
  definitions: A → B → A`.
- Division by zero — `const A = 1.0 / 0.0;` fails with `division by
  zero in const expression`.

**What it does not change.**
- The IR is byte-identical to a description that inlines the resolved
  numbers. Verified by the `test_const_resolve` integration suite
  (`hymeko_query/tests/test_const_resolve.rs`): a fixture using
  consts emits byte-equal URDF and SDF to a hand-written
  literal-only equivalent.
- Compile-time cost is one extra pass over the AST — an `O(|consts|
  + |values|)` walk with `HashMap` lookups. Bench numbers do not
  change measurably.
- `const` is the only new keyword. `*`, `/`, `=` are new tokens but
  used only inside parenthesised numeric expressions and `const` decl
  RHSes — existing fixtures parse unchanged.

## 7. Contact

For questions about the artefact, open an issue on the repository or
contact the first author (see the paper).
