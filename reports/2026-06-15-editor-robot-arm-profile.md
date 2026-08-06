# Editor: robot-arm hero-cell profile (imported kinematics)

**Date:** 2026-06-15
**Branch:** `feature/ac-hsikan` · **Base SHA:** `9684f09` (working tree dirty)
**Extends:** the profiles/imports feature (`docs/plans/2026-06-15-editor-profiles-imports/`)
and ties the hero demo into the live editor.

## Summary

Added a **Robot arm (imported kinematics)** profile to the editor: a small arm
(two links with box/cylinder geometry + mass, one continuous joint whose arc
carries an origin transform) whose link/joint **vocabulary is imported** from a
separate `meta_kinematics.hymeko` in the compile space — the same multi-file
mechanism the real `data/robotics` fixtures use. This puts the hero-demo
kinematic cell *live in the editor*: viewable in Graph / Hypergraph 3D,
editable arc values, and exportable to URDF / SDF / MJCF.

It connects the session's threads — imports, profiles, the data-driven palette,
arc-value editing, and the hero demo — in one selectable profile.

## Files touched

New fixtures (canonical, compiled + embedded):
- `data/profiles/meta_kinematics.hymeko` — compact kinematics vocabulary
  (elements/link, `@joint` + 4 joint edge-types, geometry, axes).
- `data/profiles/robot_arm.hymeko` — the arm root; `@"meta_kinematics.hymeko"`.

Edited:
- `docs/editor/views/profiles.js` — `robot_arm` profile (files + root embedded
  verbatim from the fixtures; kinematics palette with `meta_kinematics.*` bases);
  added both to `EMBEDDED_FILES` (consistency-tested).
- `hymeko_wasm/tests/test_compile.rs` — `multi_file` test now also compiles
  `robot_arm.hymeko` + `meta_kinematics.hymeko`.
- `docs/editor/editor.js` + `index.html` — `profiles.js?v=25`, `editor.js?v=25`.

## CORE.YAML items touched

None. `data/**`, `docs/editor/**`, `tests/**`. No WASM rebuild (no binding
change — the multi-file path shipped earlier today). No dependency.

## Test results

- **JS (`profiles.test.mjs` + full suite):** pass — embedded `meta_kinematics`/
  `robot_arm` match the `data/profiles` fixtures; palette entries well-formed;
  every profile root imports the meta it ships.
- **Rust (`cargo test -p hymeko_wasm --test test_compile`):** 11 passed —
  including the new `robot_arm + meta_kinematics` compile case. `rustfmt --check`
  clean.
- **Browser (`?profile=robot_arm&select=spin_joint`):** the arm compiles (31
  nodes / 6 edges) with the imported vocabulary; palette shows Link + 4 joint
  kinds; the joint's ARC-REFS editor shows the editable origin transform
  `[[0.0, 0.0, 0.1], …]`. CLI sanity: `validate` clean, `emit -f urdf` produces
  `<robot name="robot_arm">`.

## Static analysis / health

- ruff/mypy not applicable (no Python). rustfmt clean. **No §6.5 anti-patterns:**
  reuses the data-driven profile registry (a profile is one entry); compact meta
  authored as a canonical fixture + consistency-tested embed (no drift, no
  byte-fragile transcription of the 91-line real meta); not a new plan dir
  (extension of an already-planned feature — avoids #13 file proliferation).

## Open issues / follow-ups

- The hero demo (CLI) and this editor profile now share the "robot arm + imported
  vocabulary" idea; they remain separate surfaces (CLI orchestrator vs live
  editor) by design.
- Remaining hero roadmap: Phase 3 (Gömb/Soma perception — needs Soma vision
  round-tripped through `.hymeko`; a design fork, left for a scoping pass).

## Experiment provenance

Not a measurement experiment. Toolchain: `hymeko_cli` (cargo, stable), node
v24.14.0, Chrome (headless verify). Working tree dirty from prior session work
unrelated to this change.
