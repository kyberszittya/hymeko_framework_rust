# Robot-emitter extension plan — galambos_planar_v2 load-bearing (§6 decision, 2026-07-22)

## CORE-boundary determination (checked before touching anything)

Non-core. The change is confined to:
- `hymeko_formats/src/codegen.rs` (+ `transforms.rs`) — the MJCF emitter (crate NOT in CORE.YAML: 0 mentions).
- `data/robotics/meta_kinematics.hymeko` (or a new `meta_kinematics_v2`) — geometry-type / collision-mask declarations (spec file, non-core).
- `data/robotics/galambos_planar_v2.hymeko` — new versioned robot spec (non-core).
- rebuild `cargo build -p hymeko_cli` (build, not a core edit).

NOT touched: `parser/src/hymeko.lalrpop` (CORE, frozen) — the grammar is a GENERIC object/attribute language; geometry types (`box`/`cylinder`/`sphere`) are declared in `meta_kinematics.hymeko`, not hardcoded in the grammar, so adding `capsule` + mask attributes needs no grammar change. `hymeko_core` IR carries generic attributes. **No CORE approval required.** (If, during implementation, the IR proves unable to carry a needed attribute and a `hymeko_core` change is required, halt per §1.)

## Golden reference — `make_planar_arms_mjcf` (planar_grasp_env.py:41-82), the exact contract to reproduce

- 2 arms, bases at **x = ±0.14**, y = −0.02, z = _PLANE_Z; each yawed so home points +Y.
- Per arm: base hub `<geom type="cylinder" size="0.022 0.012" ARM_LEGALITY>` → link1 `<geom type="capsule" fromto="0 0 0 0 0.16 0" size="0.012" ARM_LEGALITY>` (hinge `j1` axis z, range −4..4) → link2 at `pos="0 0.16 0"` `<geom type="capsule" fromto="0 0 0 0 0.14 0" size="0.01" ARM_LEGALITY>` (hinge `j2` axis z, range −4..4) → `<geom name="fingertip_{side}" type="sphere" size="0.014" pos="0 0.14 0" FINGERTIP>` + `<site tip_{side}>`.
- Collision: **ARM_LEGALITY = contype 1 / conaffinity 3** on every structural geom; **FINGERTIP** on the fingertips; floor conaffinity ANY.
- Actuators: 4 × `<position kp="40" kv="4.0" ctrlrange="-4.0 4.0">` (`j1_left,j2_left,j1_right,j2_right`).
- `<default><joint damping="1.5"/><geom friction="1 0.05 0.001"/></default>`; `<option Physics.option_attrs()>`.
- Embodiment variants: POINT (one sphere fingertip, r=0.014) / RING (`with_fingertip_clamp`, 12-sphere ring).

## Current emitter output (`hymeko emit -f mjcf galambos_planar`) — the gap

Connected rods emit correctly (base→upper `pos 0 0 0`→lower `pos 0.16 0 0`; box `origin`=mid-point honored). Gap vs golden:
1. **geometry** `type="box"` → needs `type="capsule"` (fromto/size). — emitter + spec
2. **collision masks** none → needs `contype="1" conaffinity="3"` (ARM_LEGALITY) + fingertip mask. — emitter + spec
3. **fingertip** absent → needs the terminal sphere geom + POINT/RING variant. — emitter + spec
4. **bases** ±0.18 → ±0.14; **joint range** ±π → ±4.0; **actuators** kp/kv/ctrlrange. — v2 spec values

## Bounded implementation steps (next, with tests per §3)

1. `meta_kinematics.hymeko` (or `_v2`): add `capsule {}` to `geometry`, and a `collision { contype; conaffinity; }` attribute on link geometry (+ a terminal `fingertip` element or reuse `sphere`).
2. `hymeko_formats/codegen.rs`: emit `type="capsule"` with `fromto`/`size` from a capsule geometry; emit `contype`/`conaffinity` from the collision attribute; emit the terminal fingertip sphere. Keep box/cylinder/sphere unchanged.
3. `galambos_planar_v2.hymeko`: bases ±0.14, capsule links (l1=0.16 r=0.012, l2=0.14 r=0.01), cylinder hub r=0.022 h=0.012, ARM_LEGALITY masks, hinge range ±4.0, fingertip sphere r=0.014 FINGERTIP, position actuators kp40/kv4.
4. Wire `make_coin_env` / `PlanarGraspEnv` to emit from `galambos_planar_v2` (via `emit_arm_mjcf`), NOT `robot=None`; keep `make_planar_arms_mjcf` as the golden reference for the parity test only.
5. **Exact-reproduction test** (§3): parse both MJCFs to `MjModel`; assert per-geom type/size/fromto/pos, per-joint axis/range, contype/conaffinity, actuator kp/kv/ctrlrange, body tree, masses all match make_planar_arms_mjcf within tol. Sentinel: a change to v2's link length / mask reaches the outer env.

## Status

CLI built (`target/debug/hymeko`, exit 0). Golden reference extracted. Gap pinned. CORE boundary cleared (non-core).
Implementation is the next bounded step; it is a substantial Rust change (emitter + spec + parity proof) and is
scoped here per §2 before editing `hymeko_formats`.
