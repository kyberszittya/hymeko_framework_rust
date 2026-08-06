---
name: project-xprofile-instance-refs
description: HyMeKo cross-profile instance references — CORE enabler done (approved+tested); downstream shared-models + AgentSpec.from_hymeko is phase 2
metadata: 
  node_type: memory
  type: project
  originSessionId: 2592f2cd-e7fd-4d3d-ad05-8a5823bf1628
---

**Goal (user, 2026-06-19):** shared reward/observation models reused across agent descriptions
(kill the reach-reward dup between arm_reach_task and arm_reach_safe_task). Requires referencing one
profile's *instance* decls from another.

**Key finding:** HyMeKo's `@"…"` import shares **meta vocabulary** (kinds), not instance decls.
Cross-profile instance refs failed (`UnresolvedRef`) because `compile()` applied `using` aliases for
the **root** AST only — an imported *profile*'s own `using … as el` was never applied, so its
arcs/bases couldn't lower. The grammar **requires** the `_description` wrapper (imports can't sit in
a content node — parse error), so a dep's decls live at `[ns, content, decl]`; the importer reaches
them by full path or `using <desc>.<content> as arr; (+ arr.decl)`.

**CORE change DONE (approved + verified):** `APPROVED-CORE-EDIT: xprofile-instance-refs` (2026-06-19).
`hymeko_core/src/module_store/module_store.rs` (`lockdown: full`) — added a best-effort, per-statement
loop applying **each imported profile's `using` aliases** after the root's. Strictly additive
(meta-only imports = no-op; no hash drift). Verified: new core test
`check_xprofile_instance_ref` (+ fixtures `data/minimal_examples/import_examples/xprofile_{shared,importer}.hymeko`),
full suites green (hymeko_core 133, hymeko_query 212, hash-parity intact). Report
`reports/2026-06-19-xprofile-instance-refs.md`, plan `docs/plans/2026-06-19-xprofile-instance-refs/`.

**✅ PHASE 2 DONE (2026-06-19, `reports/2026-06-19-shared-agent-models.md`):**
1. `read_bundle` is import-aware (`_gather_decls` follows `@"…"`, cycle-guarded, local shadows
   imported; member last-segment = decl name, so `arr.dist`→`dist`). Matches the compiler.
2. Shared `data/robotics/arm_reach_reward.hymeko` created; arm_reach_task + arm_reach_safe_task
   rewired to `using ...arm_reach_reward as arr; (+ arr.dist[, +penalties])`. **Phase-1 constraint:
   the importer must ALSO import the meta vocab the shared profile uses** (no transitive indexing) —
   both rewired tasks now import meta_kinematics/meta_task/meta_reward. `hymeko inspect` validates;
   dist decl hash unchanged (semantics preserved); Python specs identical (regression).
3. `AgentSpec.from_hymeko(robot, obs_profile, task_profile)` composes obs_dim + n_vertices +
   action bounds + reward; parity test vs ArmReachEnv passes. 97 hymeko_rl tests green.
Observation was already shared (arm_reach_observation.hymeko as obs_profile) — no change needed.

**⏭ Still open:** mixin/bare-name reuse + transitive-import indexing + cleaner `shared.dist` path
(drop the `_description` segment) — all future CORE edits, separate approval, "when performance
requires it" (user).

Observation is ALREADY a shared standalone model (`arm_reach_observation.hymeko`, used as obs_profile).
Ties to [[project-kato-collaboration-grasping]].

**NEW (2026-06-23): cross-model KINEMATIC attachment works too** (not just reward refs). A joint in an
importing robot can attach to an *imported* robot's link: `data/robotics/arm_gripper_import.hymeko` does
`@"anthropomorphic_arm.hymeko"; using robot as arm;` then `@grip_l: prismatic_joint { (+ arm.tool [[...]], -
finger_l, - AXIS_X); }`. `hymeko emit -f mjcf` produces the merged arm+gripper MJCF (nbody 10, joints
j0..jtool+grip_l/r, fingers verified children of `tool`), MuJoCo-valid. So the no-duplication arm+gripper
composite for the Kato pick-and-place is real (the self-contained `arm_gripper.hymeko` that restates the arm is
now just a fallback). Same phase-1 constraint: importer must also import the meta vocab. PickPlaceEnv defaults
to the import composite.
