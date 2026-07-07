---
name: project-engine-transitive-imports
description: "Engine resolves only ONE level of @\"…\" imports — a composite profile imported into another loses its own imports (UnresolvedRef); blocks multi-reference scenario .hymeko from being inspect-clean; fix = core edit (approval-gated)"
metadata: 
  node_type: memory
  type: project
  originSessionId: 413f6759-7b59-4979-b07c-39a8de633fc8
---

**Limitation (measured, Stage 1, 2026-06-23).** The HyMeKo engine (`hymeko_core`/`parser`) follows only **one
level** of `@"…"` imports. When a *composite* profile is itself imported into another profile, the composite's
own imports are NOT transitively followed:
- `data/robotics/arm_gripper_fanuc_import.hymeko` (imports `meta_kinematics.hymeko` + `fanuc_lrmate.hymeko`)
  `hymeko inspect`s **clean on its own**.
- `data/robotics/pick_place_scenario.hymeko` imports that composite (+ `pick_place_task.hymeko`,
  `meta_scenario.hymeko`) and **fails**: `UnresolvedRef { from "…arm_gripper", target
  "meta_kinematics.kinematics.elements" }` — scenario → robot → meta_kinematics is two levels.

**Why it matters.** It blocks the declarative RL **scenario** `.hymeko` (the "HyMeKo describes structure + obs +
reward + scenario" thesis, [[project-rl-algorithm-roadmap]] Stage 1, [[project-fsm-structured-rl]]) from being
`hymeko inspect`-clean — every other profile in the line validates, only the multi-reference instance doesn't.

**Current workaround (works today).** The Python bridge readers in `hymeko_rl/env/_profile.py`
(`read_bundle`, `read_scene_fields`, `read_imports`) DO follow imports transitively (regex `_gather_decls`), so
`ScenarioSpec.from_hymeko` builds the correct env and the Stage-1 parity test passes. Engine-side validation is
what's missing, not the Python path.

**How to apply (when addressing later).** Fix = transitive (>1-level) import resolution in the engine — a
`hymeko_core`/`parser` change → **CORE.YAML §1, approval-gated** (request `APPROVED-CORE-EDIT: <slug>` first;
this builds on the prior `xprofile-instance-refs` core edit that gave 1-level cross-profile resolution). Verify
with `target/debug/hymeko.exe inspect data/robotics/pick_place_scenario.hymeko` (must stop erroring on
`meta_kinematics.kinematics.elements`). Report: `reports/2026-06-23-stage1-hymeko-rl-scenario.md`.
