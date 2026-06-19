# Report — 6-DOF arm joint-axis correction (j3 and the elbow/wrist axes)

**Date:** 2026-06-19 · **Author:** Aiko (agent), for Dr.\ Csaba Hajdu
**Status:** ✅ **Fixed and verified.** The `anthropomorphic_arm` (Moveo) joint axes now form the
canonical anthropomorphic pattern (shoulder ∥ elbow); world axes confirmed **Z, Y, Y, Z, Y, Z**.
Full `hymeko_query` integration suite green (212), source↔generated-URDF consistency restored.

## What was wrong (measured, not assumed)
The user flagged j3's axis. Computing the **world-frame** joint axes at the home pose (the only
correct frame to judge in, since j1 carries a 90°-about-Z twist that propagates to link_1..tool):

| joint | role | world axis (before) | world axis (after) |
|---|---|---|---|
| j0 | waist | Z | Z |
| j1 | shoulder | Y | Y |
| j2 | elbow | **Z (roll)** | **Y (pitch, ∥ shoulder)** |
| j3 | forearm | **Y (pitch)** | **Z (roll)** |
| j4 | wrist | **X** | **Y (pitch)** |
| jtool | tool | Z | Z |

The anthropomorphic/Moveo signature is that **shoulder and elbow are parallel pitches**. The arm
was `Z,Y,Z,Y,X,Z` — the elbow pitch and forearm roll were swapped (j2/j3), and the wrist (j4) sat
on X. The corrected arm is the textbook `Z,Y,Y,Z,Y,Z`. **User-approved** (AskUserQuestion: "full
canonical fix").

## The local-vs-world subtlety
Because link_1..tool inherit j1's 90°-about-Z frame twist, the **local** `AXIS_*` token is not the
world axis: local **X reads as world Y** for j2/j4, local **Z stays world Z** for j3. So the source
edits are local **j2: Z→X, j3: X→Z, j4: Y→X** (local pattern `ZXXZXZ`), which I verified by
recomputing the world axes (`Z,Y,Y,Z,Y,Z`, exact match) rather than trusting the algebra.

## Files touched
| file | change |
|---|---|
| `data/robotics/anthropomorphic_arm.hymeko` | j2→`AXIS_X`, j3→`AXIS_Z`, j4→`AXIS_X` |
| `data/robotics/anthropomorphic_arm_using.hymeko` | same 3 (the alias fixture — must match for alias-invariance) |
| `hymeko_query/tests/test_anthropomorphic_generation.rs` | `per_joint_axes_b005` (j2/j3/j4), `six_dof_axis_signature` (→`ZXXZXZ`), B-004/005 emit assertion (`0 1 0`→`0 0 1`), comments |
| `hymeko_query/tests/test_generation_engine.rs` | `moveo_joint_axes` expected; URDF axis assertion (no local `0 1 0`) |
| `hymeko_query/tests/test_transform_ecosystem.rs` | `moveo_dot_axis_labels` (no `(Y)` label) |
| `reports/2026-06-19-6dof-fixed-axes-frame.png` | render evidence |

## Root cause of the test churn (a real pre-existing inconsistency)
The committed `generated/gazebo_launch/moveo/moveo.urdf` (modified in the working tree **before**
this session) already had the *corrected* axes, while the `.hymeko` **source** still had the old
ones. A prior session fixed the generated artifact but not the source — leaving them inconsistent.
This fix corrects the source; a fresh `hymeko emit -f urdf` is now **byte-identical** (joints+axes)
to the committed URDF. The alias fixture had to change in lockstep (the `prop1_alias` property test
asserts both sources emit identical URDF).

## Tests
- `cargo test -p hymeko_query --test integration` — **212 passed, 0 failed, 1 ignored** (3.6 s).
  Caught and fixed 5 assertion sites + 1 alias-fixture divergence across 3 test files.
- `hymeko_rl` reach/scene_style tests unaffected (they use the 4-DOF reach arm / structural
  invariants); not re-run beyond the earlier green pass.

## CORE.YAML
**No `hymeko_query` implementation touched** — only its **test files** (allowlist `tests/**`) and
the two non-core `data/robotics/*.hymeko` fixtures. The extractor/emitter logic was already correct
(it faithfully propagated whatever axes the source declared); the bug was in the source data, not
the core query layer. Flagging the test-in-core-crate edits for visibility.

## Performance
N/A (data + test correction). Render smoke: corrected arm, position control, 81 frames.

## Follow-up
- The `_consistency/moveo.urdf` variant was likewise pre-modified; it matches the corrected axes.
- The render default (`render_reach.py`) now shows the corrected arm; the GIF/still artifacts from
  the earlier prettier-sim task predate the axis fix and can be re-rendered if needed for the deck.
