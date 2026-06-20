# MJCF emitter: parent→child contact exclusions (proper fix for frozen joints)

*2026-06-20 · Aiko (Claude Code) for Dr. Csaba Hajdu*
*Plan: [docs/plans/2026-06-20-mjcf-parent-contact-excludes/](../docs/plans/2026-06-20-mjcf-parent-contact-excludes/)*
*Supersedes the env-level workaround in [2026-06-20-galambos-reward-shaping.md](2026-06-20-galambos-reward-shaping.md).*

## Summary

The earlier Galambos fix unfroze the shoulder with an env-level contact exclusion. That
treated the symptom in one consumer. The **proper** fix is in the emitter: `emit_mjcf`
now emits an explicit `<contact><exclude>` for every parent→child link pair, so *every*
consumer of `hymeko emit -f mjcf` gets a physically self-consistent scene. The env
workaround is removed.

## Root cause (recap)

Adjacent links meet at their shared joint, so their collision geoms overlap there
(`base_left`/`upper_left` share an origin, −0.022 m penetration). The self-contact force
pins the joint between them. The emitter emitted **no `<contact>` block at all**, and
MuJoCo's default `filterparent` did not remove the overlap on these scenes.

## Change

[hymeko_formats/src/transforms.rs](../hymeko_formats/src/transforms.rs) `emit_mjcf`: after
`</worldbody>`, emit one `<exclude body1=parent body2=child/>` per joint, deduplicated,
skipping `world`-rooted joints (`world` is the implicit worldbody, not an emitted body).
Emitted output for `galambos_planar`:

```xml
<contact>
  <exclude body1="base_left" body2="upper_left"/>
  <exclude body1="upper_left" body2="lower_left"/>
  <exclude body1="base_right" body2="upper_right"/>
  <exclude body1="upper_right" body2="lower_right"/>
</contact>
```

[hymeko_rl/env/planar_grasp_env.py](../hymeko_rl/env/planar_grasp_env.py): removed
`adjacent_link_excludes` and its call (the emitter now supplies the exclusions; keeping
both would emit duplicate `<exclude>` and MuJoCo would reject the model).

## Files touched

| File | Δ | Note |
|------|---|------|
| `hymeko_formats/src/transforms.rs` | +~22 (incl. 2 tests) | parent→child `<contact><exclude>` emission |
| `hymeko_rl/env/planar_grasp_env.py` | −~30 | removed the env workaround |

## CORE.YAML / dependencies

**None.** `hymeko_formats` is non-core. No dependency change.

## Validation

- **Shoulder now tracks exactly.** Holding `[1.2, −1.6, −1.2, 1.6]` with **no env
  workaround**: final qpos = `[1.2, −1.6, −1.2, 1.6]` (error ~0, vs 0.16 rad with the
  narrower env-only exclude — the emitter excludes *all* adjacent pairs, not just the
  shoulder).
- **Rust unit:** `mjcf_emits_parent_child_contact_excludes` (one exclude per parent→child
  joint), `mjcf_skips_world_rooted_contact_exclude` (no `world` exclude). `cargo test -p
  hymeko_formats mjcf` — 3 passed (incl. the deep-chain body-count test, unchanged).
- **Rust integration:** `hymeko_query` integration — **214 passed**, 1 ignored. WAM /
  DRC-Hubo MJCF emit, generation, and transform-ecosystem suites unaffected (the excludes
  are additive; the `</body>` count is unchanged).
- **Python:** full `hymeko_rl` suite — **110 passed**, including the shoulder-mobility
  regression now validated *through the emitter*.
- **Gates:** `cargo clippy -p hymeko_formats --all-targets -D warnings` — clean;
  `ruff check` — clean.

## Effect beyond Galambos

Every emitted multi-link robot now ships with adjacent-link contact filtering — the
6-DOF arm, `reach_arm`, WAM, DRC-Hubo. This is the general, correct form of the
self-collision handling the RL line had been patching per-env.

## Open / follow-up

- The URDF/SDF emitters express adjacent-link collision filtering differently (URDF has
  no MuJoCo `<contact>`; SDF uses `<collide_bitmask>` / self-collision flags). If those
  emitters feed a physics consumer that exhibits the same freeze, apply the analogous
  filtering there. Not triggered by any current test.

## Provenance

Git branch `soma-vision`; tree dirty (pre-existing unrelated changes). Determinism: all
fixtures committed; no RNG in the emitter path.
