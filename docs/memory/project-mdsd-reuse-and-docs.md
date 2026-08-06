---
name: project-mdsd-reuse-and-docs
description: "HyMeKo MDSD single-source-of-truth — reused ROS2 scenario (import vs inline), SYSTEM_ENGINEERING_VIEW manifesto + HYMEKO_ECO_CAP ledger; import path was already wired"
metadata: 
  node_type: memory
  type: project
  originSessionId: e049ea12-7387-4a59-87f4-051966d7cfcb
---

**Done 2026-06-20** (`reports/2026-06-20-mdsd-reuse-scenario.md`, plan `docs/plans/2026-06-20-mdsd-reuse-scenario/`):

**Premise correction:** "wire the import path" needed NO framework work. Cross-file import
resolution already flows through emission — `data/robotics_imported/wam/wam.hymeko` imports
`meta_kinematics.hymeko` and `hymeko_query/tests/test_imported_real.rs` proves extract + 6-format
emit. The `xprofile-instance-refs` CORE enabler landed 2026-06-19 ([[project-xprofile-instance-refs]]).
The ROS2 scenario `hymeko_ros2_demo/.../scenarios/hymeko_robot.hymeko` (verbatim paper Listing A.6.1)
just predated the pattern, re-declaring link/joint/axis/context/signal/aggregation as bare types.

**What was built:**
1. `data/robotics/meta_context.hymeko` — NEW shared context-state vocabulary (companion to
   meta_kinematics): context/signal/counter/mode/reference/tool/payload/component/grasp_mode +
   @interpretation/@aggregation/@constraint. (meta_hri is a DIFFERENT domain — §6.1 confirmed absent.)
2. `scenarios/hymeko_robot_reuse.hymeko` — reused variant importing meta_kinematics + meta_context.
   **Baseline kept as frozen control** (user: selling point = measure length decrease).
3. `hymeko_query/tests/test_mdsd_reuse.rs` — 2 tests green (5 links/4 rev joints/unit axes/URDF
   `<joint>`/LOC guard). Full integration target 214 pass, clippy clean. Test target name is
   **`integration`** (`[[test]] name="integration" path="tests/mod.rs"`), NOT `mod`/`lib`.

**Measured:** 189→126 lines (−33%), 108→76 code lines (−30%) for ONE robot; vocab (~55 lines)
amortised. Semantic win: bare `joint` type emits 0 URDF joints (not one of extractor's 4 typed
kinds); typed `rev_joint` emits 4 with axes. Reuse = difference between "projects to a working
robot" and "does not".

**Docs created (user asked to collect these "somewhere"):**
- `docs/architecture/SYSTEM_ENGINEERING_VIEW.md` — MDSD manifesto ("one source, many views";
  vocabulary vs instance; boundary rules; SysML is a view not a rival).
- `docs/HYMEKO_ECO_CAP.md` — capability-evidence ledger (PROVEN/SHIPPED/RESEARCH per row, each with
  a verifiable anchor). Keep updated when capabilities change status.

**Open:** live demo still consumes baseline by design; switching it needs a context-eval regression
for the imported form first. Transitive-import indexing + cleaner `shared.dist` path still future
CORE edits. Ties to [[project-seminar-demos-and-hymeyolo-plan]].
