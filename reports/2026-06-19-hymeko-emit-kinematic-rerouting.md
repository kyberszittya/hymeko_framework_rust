# Report — `hymeko emit` kinematic formats now produce a loadable, articulated robot (B-004/B-005)

**Date:** 2026-06-19
**Author:** Aiko (agent), for Dr. Csaba Hajdu
**Status:** ✅ complete — `hymeko emit -f {mjcf,urdf,sdf} <arm>` emits the correct mixed-axis,
articulated arm directly from `.hymeko`. Tests green, clippy clean.

## Summary
Building the `hymeko_rl` robot-RL line (the Kato collaboration) surfaced that a robot emitted
from `.hymeko` was kinematically **degenerate** — every joint on Z, end-effector workspace zero.
The investigation pinned the cause precisely and it was **not** where BUGS.md first guessed:

- The **model extractor** (`hymeko_query::kinematics::extract_kinematic_model`, CORE) was *always
  correct* — it reads each joint's `AXIS_*` arc. Proven by a new regression test that **passes**.
- The bug was the **CLI emit path**: `emit`/`compile` route kinematic formats through static
  `transforms/<fmt>/` templates that **hardcode** `<axis xyz="0 0 1"/>` (B-005) and, for MJCF,
  emit flat bodies with **no `<joint>` at all** (B-004). A correct model-based emitter existed but
  the CLI never used it.

Fix (all **non-CORE**): route kinematic formats to the model-based emitter, plus two `emit_mjcf`
fixes required to make the MJCF actually load in MuJoCo. The CORE extractor is **byte-unchanged**.

## Root-cause chain (measured, not assumed)
1. **Measured:** emitted URDF *and* SDF show all axes `0 0 1`, though the source declares
   Z/X/Z/X/Y/Z. → not MJCF-specific; a shared upstream defect.
2. **Measured:** instrumenting `extract_kinematic_model` showed its axis branch *never fired* for
   the CLI path → the CLI does not call the model extractor at all.
3. **Inferred → confirmed:** the CLI uses `render_from_templates` over `transforms/<fmt>/`; the
   templates literally contain `<axis xyz="0 0 1"/>`. The model path (`extract_kinematic` → registered
   `emit()`) is correct and unused by the CLI.
4. **Measured (regression test):** the model extractor returns j0=Z, j1=X, j2=Z, j3=X, j4=Y,
   jtool=Z — correct. So the fix is dispatch, not extraction.
5. **Measured (MJCF load):** after rerouting, MuJoCo rejected the scene for (a) a `<body name="world">`
   colliding with the implicit `<worldbody>`, then (b) `<inertial>` missing the required `pos`. Both
   fixed in `emit_mjcf`. Final scene loads (nq=6, nu=6) and articulates (EE spread ≈ 1.22 × 0.89 × 0.79 m).

## Changes
- **`hymeko_cli/src/main.rs`** — `Commands::Emit` dispatch: kinematic formats (`accepts() ==
  Kinematic`) now always use the model-based emitter; non-kinematic formats keep the template path.
  Removes the `--rich` opt-in requirement (flag retained as a backward-compat no-op) and
  de-duplicates the two `render_from_templates` blocks.
- **`hymeko_formats/src/transforms.rs`** — `emit_mjcf`: (1) a root `world` frame maps onto MuJoCo's
  implicit `<worldbody>` (emit its children as top-level bodies, preserving their fixed-joint origin)
  instead of a colliding `<body name="world">`; (2) `<inertial>` now emits the required `pos="0 0 0"`.
- **`hymeko_query/tests/test_anthropomorphic_generation.rs`** — two regression tests (below).
- **`docs/BUGS.md`** — B-004 and B-005 marked RESOLVED with the pinned root cause; minor open
  follow-up recorded (joint limits via `limit -> joint_rev_limit` ref not yet followed).

## Files touched
| File | +/- |
|---|---|
| `hymeko_cli/src/main.rs` | +30 / −28 (net logic: reroute + de-dup) |
| `hymeko_formats/src/transforms.rs` | +18 / −6 |
| `hymeko_query/tests/test_anthropomorphic_generation.rs` | +82 / −0 |
| `docs/BUGS.md` | +~120 / −10 |

**CORE.YAML items touched:** **none.** `hymeko_query/src/kinematics/kinematic.rs` (CORE) was
instrumented during diagnosis and then fully reverted — confirmed **byte-unchanged** via `git diff`.
The CORE-edit token granted for `fix-b005-joint-axis-extraction` was therefore **not used** (the bug
was non-CORE). Tests live under `hymeko_query/tests/` (allowlisted, not CORE).

## Test results
Layer: **integration** (`cargo test -p hymeko_query --test integration`).
- `test_anthropomorphic_generation` module: **27 passed, 0 failed** (0.10 s), including:
  - `per_joint_axes_match_the_source_b005` — model extractor returns the mixed axes (guards the
    CORE extraction contract; documents the "per-joint axis pattern" the file's own header promised).
  - `mjcf_emit_is_loadable_and_articulable_b004_b005` — exercises the exact CLI path (registry
    `emit()` over `ModelView::Kinematic`); asserts 6 hinge joints present (B-004), mixed axes
    `1 0 0` + `0 1 0` survive (B-005), no `<body name="world">` collision, every `<inertial>` has `pos`.

End-to-end (manual, MuJoCo 3.9.0): `hymeko emit -f mjcf data/robotics/anthropomorphic_arm.hymeko`
→ loads, nq=6/nu=6, axes `[Z,X,Z,X,Y,Z]`, EE workspace spread ≈ `[1.217, 0.891, 0.789]` m.
`emit -f urdf` and `emit -f sdf` → mixed axes confirmed.

## Static analysis
- `cargo clippy -p hymeko_cli -p hymeko_formats --all-targets -- -D warnings` → **clean**.
- `cargo fmt`: my edited regions are clean. Two pre-existing drift sites remain on lines **outside
  this diff** (`main.rs:415` `use` block, `transforms.rs:808` SysML macro) — not reformatted, per
  scope discipline (git confirms my main.rs changes are only lines 424–474).
- **§6.5 anti-patterns:** none introduced; the change *removes* a duplicate `render_from_templates`
  block and *removes* the dead `--rich` branch divergence (anti-pattern #1/#9 reduction).

## Performance
Not a performance change — correctness fix on a one-shot emit (sub-second; not on any hot path). No
budget declared or needed. No `criterion`/perf assertion applicable.

## Dependencies
None added or removed.

## Open issues / follow-ups
1. **Joint limits (minor, open).** Emitted joints are unlimited: `extract_joint_limits` reads direct
   `limit_lower/upper` children but the fixture uses `limit -> joint_rev_limit` (a ref). Following the
   ref would add `range=` to the emitted joints. Articulation works without it. Recorded in BUGS.md.
2. **B-003 (open).** PyO3 `load_file` import resolver — unaffected here; `AgentSpec.from_hymeko`
   still routes through the CLI until fixed.
3. **`transforms/mjcf|urdf|sdf/` static templates** are now unused by the CLI for robots. Candidates
   for removal in a later cleanup (left in place now to avoid scope creep).

## Provenance
- Git SHA at start: `7d16ad0` (working tree dirty — pre-existing session changes; this task touched
  the 4 files listed above plus the reverted-clean CORE file).
- Platform: Windows 11, MuJoCo 3.9.0 (Python), `cargo` dev profile.
- Fixture: `data/robotics/anthropomorphic_arm.hymeko` (7 links, 1 fixed + 6 revolute joints,
  source axes Z/X/Z/X/Y/Z). Robot name `moveo` in the test.
- Random seeds: articulation probe used `numpy` default_rng seeds 0–11 over `qpos ∈ [-1.5, 1.5]` rad.
