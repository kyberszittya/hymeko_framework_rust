# Demo scripts

Executable walk-throughs of the functionality landed around 2026-04-18 in the
HRE extraction + alias-parity + T10 `?` slice. Each script is self-contained
and prints what it is doing before it does it, so you can copy the commands
out if you prefer to run them one-by-one.

All scripts assume you run them from the workspace root:

```bash
bash scripts/<name>.sh
```

| Script | What it shows | Expected runtime |
|--------|---------------|------------------|
| `demo_state.sh` | Workspace crate list, per-crate test counts, recent changelogs, uncommitted status. Good "morning check". | ~1 min on a warm cache |
| `demo_alias_parity.sh` | Runs the 11-test parser grammar suite + the 16-test end-to-end alias-parity suite; diffs the `_using.hymeko` fixtures against their non-alias counterparts. | <10s on a warm cache |
| `demo_query_variable.sh` | Builds the parser, runs the 15-test `?` regression suite, then drives `parse_query_var` from a throwaway binary over seven representative inputs (3 accepted, 4 rejected). | ~15s first run |
| `demo_hre_extraction.sh` | Verifies `hymeko_hre` compiles standalone, runs its 2 integration tests, confirms `hymeko_core::engine` is gone, and that `hymeko_py` imports `HypergraphEngine` from `hymeko_hre`. | ~20s first run |
| `demo_visualizations.sh` | Emits URDF/SDF/MJCF/DOT + ROS2 launch for a fixture (default: `mini_arm.hymeko`); renders DOT → SVG if Graphviz is installed. Takes an optional `.hymeko` path. | depends on CLI build state |

## Example execution runs

### `demo_alias_parity.sh`

```text
$ bash scripts/demo_alias_parity.sh
==> Parser-level grammar tests (parser/tests/using_alias.rs — 11 tests)
test result: ok. 11 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out

==> End-to-end alias-parity tests (hymeko_query — 16 tests in mod alias_parity)
test result: ok. 16 passed; 0 failed; 0 ignored; 0 measured; 90 filtered out

==> Structural diff between baseline and aliased fixtures
    -- anthropomorphic_arm.hymeko vs anthropomorphic_arm_using.hymeko --
  176 data/robotics/anthropomorphic_arm.hymeko
  179 data/robotics/anthropomorphic_arm_using.hymeko
    using-statements in aliased source:
      using kinematics.elements as el;
      using kinematics.geometry as geo;
      using kinematics.axes as ax;

    -- robot_4wh.hymeko vs robot_4wh_using.hymeko --
  166 data/robotics/robot_4wh.hymeko
  171 data/robotics/robot_4wh_using.hymeko
    using-statements in aliased source:
      using kinematics.elements as el;
      using kinematics.geometry as geo;
      using kinematics.axes as ax;
      using kinematics.sensors as sens;
      using kinematics.controllers as ctrl;

==> Done. Same topology via either spelling.
```

### `demo_query_variable.sh`

```text
$ bash scripts/demo_query_variable.sh
==> Build parser (populates OUT_DIR for the LALRPOP-generated module)

==> Run the `?`-token regression suite (15 tests)
test result: ok. 15 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out

==> Example 1: parse_query_var("?x") via an inline Rust snippet
  "?x"            ->  binds `x`
  "?link_name"    ->  binds `link_name`
  "?MyVar"        ->  binds `MyVar`
  "? spaced"      ->  binds `spaced`
  "x"             ->  rejected (expected)
  "?"             ->  rejected (expected)
  "?x ?y"         ->  rejected (expected)
```

### `demo_hre_extraction.sh`

```text
$ bash scripts/demo_hre_extraction.sh
==> hymeko_hre crate layout
    hymeko_hre/Cargo.toml
    hymeko_hre/src/engine/hymeko_subscriber.rs
    hymeko_hre/src/engine/hypergraphengine.rs
    hymeko_hre/src/engine/hypergraphengine_impl.rs
    hymeko_hre/src/engine/mod.rs
    hymeko_hre/src/lib.rs
    hymeko_hre/tests/test_hypergraphengine.rs

==> cargo test — hymeko_hre (integration: 2 tests)
    running 2 tests
    test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured

    ok — hymeko_core::engine removed

==> hymeko_py import (should reference hymeko_hre now)
    14:use hymeko_hre::HypergraphEngine;

==> Done.
```

## Related documentation

- `docs/STATE.md` — overall framework snapshot.
- `docs/plans/05_hre_extraction/{plan.md,features.md}` — HRE extraction plan.
- `docs/plans/06_wasm_editor/outline.md` — future WASM editor/MCP server plan.
- `docs/examples/visualizations.md` — hand-authored hypergraph / DOT / URDF renders.
- `docs/examples/hymeko_to_sysmlv2.md` — T2M workflow with SysML v2 ground truth.
- `docs/examples/query_variables.md` — `?name` syntax examples + integration roadmap.
