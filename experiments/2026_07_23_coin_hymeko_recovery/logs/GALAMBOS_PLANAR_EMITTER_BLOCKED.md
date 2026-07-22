# GALAMBOS_PLANAR_EMITTER_BLOCKED — core IR cannot carry capsule geometry or collision masks

Halted per CLAUDE.md §1 / §11 and your explicit instruction ("if such an IR limitation is encountered, stop with
exact evidence"). No CORE crate was edited.

## Exact failing gate

Robot-emitter implementation (§1) — cannot emit the golden robot's `type="capsule"` structural links **or** the
ARM_LEGALITY collision masks. Both are required for `GALAMBOS_PLANAR_KINEMATIC_PARITY_PASS` /
`GALAMBOS_PLANAR_CONTACT_PARITY_PASS`.

## Exact source / spec location (the IR limitation)

- `hymeko_query/src/kinematics/kinematic.rs:15` — `pub enum GeometryShape { Box, Cylinder, Sphere }` — **no `Capsule`**.
- `hymeko_query/src/kinematics/kinematic.rs:246-250` — shape-name parse: `"box"/"cylinder"/"sphere" => Some(...)`,
  **`_ => None`** — a `capsule` in the spec parses to `None` and is **silently dropped** (the `if let Some(shape)`
  guard skips it → the link emits no geom).
- `hymeko_query/src/kinematics/kinematic.rs:22` — `pub struct GeometryInfo { shape, dimensions }` and `:27`
  `LinkInfo { did, name, mass, geometry, origin, color }` — **no collision fields**; `grep contype|conaffinity|
  collision` over the core kinematic IR returns nothing. Collision masks cannot be carried.
- `hymeko_query` is CORE.YAML-protected (`CORE.YAML:44-47`, `lockdown: implementation`, "parser tables frozen").

The geometry shape and the collision mask are **typed IR fields**, not generic attributes — so the existing generic
attribute transport provably cannot carry them (verified: `capsule` → `None` → dropped).

## Expected vs actual

- Expected (golden `make_planar_arms_mjcf`): link1/link2 `<geom type="capsule" fromto="0 0 0 0 L 0" size="r">` with
  `contype="1" conaffinity="3"` (ARM_LEGALITY); fingertip sphere with FINGERTIP mask.
- Actual through the current IR: `capsule` is dropped (`None`); no collision-mask field exists. Cylinder is **not** a
  valid substitute — MJCF cylinder ≠ capsule (flat vs hemispherical caps → different contact geometry → the §4/§6
  parity gates would fail on geom type and contact normal). So parity is unreachable without the IR variants.

## Why compute cannot help

It is a typed-IR representational gap in a frozen core crate, not a runtime/training issue.

## Minimal required decision (CORE edit to `hymeko_query`, needs approval)

Bounded, additive, backward-compatible (no grammar/.lalrpop change; the generic bundle grammar already parses the
attributes — only the typed IR reader needs the new variant/fields):

1. `kinematic.rs:15` — add `Capsule` to `GeometryShape`.
2. `kinematic.rs:249` — add `"capsule" => Some(GeometryShape::Capsule)`; capsule `dimensions = [length, radius]`
   (via the existing `dimension` list, same as box/cylinder).
3. `kinematic.rs:22` — add `pub collision: Option<(i64, i64)>` (contype, conaffinity) to `GeometryInfo`, parsed from
   an optional `collision { contype N; conaffinity M; }` sub-node of `link_geometry` (generic attribute read; the
   grammar already accepts nested bundles — this is a typed reader addition, not a grammar change).

Then NON-CORE (no approval needed, done immediately after the CORE variants land):
4. `hymeko_formats/src/{transforms,sdf,urdf}.rs` — add the `GeometryShape::Capsule` match arms (the matches are
   exhaustive; a new variant requires them regardless). `transforms.rs` emits `type="capsule" fromto="0 0 0 0 L 0"
   size="r"` and appends `contype`/`conaffinity` from the collision field; box/cylinder/sphere unchanged.
5. `galambos_planar_v2.hymeko` (spec, non-core) + `robot_source="hymeko_spec"` wiring + the MjModel parity tests +
   the kinematic/contact/sentinel gates.

## Status of the rest of the recovery

Blocked here (§1 halt). Already committed on `recovery/coin-hymeko-bundle-and-results` and NOT affected: reward
load-bearing (`3054572`), K=6 canonical + v3 + `COIN_REWARD_EVALUATION_ALIGNMENT_PASS` (`df04a8e`), the scene
integration (§8, mechanical `EnvSpec.from_hymeko`) is independent and could proceed, but the **complete bundle gate
cannot pass** until the robot spec is load-bearing, which requires the CORE decision above. Scene work alone would
leave `MULTIPLE_HYMEKO_SPECS_IGNORED` for the robot.
