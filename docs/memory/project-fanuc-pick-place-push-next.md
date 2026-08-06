---
name: project-fanuc-pick-place-push-next
description: "2026-07-05 corrected: the condition for porting the Galambos refine pattern to FANUC was NOT met; Galambos TD3+BC refine degraded. Keep the declarative controller/monitor idea as suspect framework evidence, but do not port the failed refine loop as a success pattern."
metadata: 
  node_type: memory
  type: project
  originSessionId: d2ccb45c-9c6f-4422-a725-08dd14fe9109
---

Standing order given at bedtime 2026-07-05 03:10: *"if the results are good, go with the FANUC pick-and-place.
Don't reinvent the wheel, extend the framework."*

**Condition status:** not met. The Galambos TD3+BC/refine path degraded the BC floor, so diagnose the
Q-term/critic/update mechanism before porting any refine loop to FANUC. Keep the controller/profile/monitor
shape as an idea, not as a validated production recipe.

**How to apply (the extend-don't-reinvent list):**
- Vocabulary + reader already exist: `data/robotics/meta_controller.hymeko`, `hymeko_rl/control/controller_spec.py`,
  `PushDemonstrator` machinery (robustness GUARDS, LAWS registries, law injection, `last_targets`).
- FANUC/pick-place scaffolding already exists: `data/robotics/fanuc_lrmate.hymeko`,
  `arm_gripper_fanuc_import.hymeko`, `pick_place_task.hymeko`/`pick_place_scenario.hymeko`, `pick_place_env`,
  `hymeko_rl/experiments/pick_place_bc.py` (+ checkpoints `checkpoints/fanuc_pick_*.pt`). §6.1 discovery pass
  before ANY new file.
- Port = author a `fanuc_pick_place.hymeko` controller profile (phases approach→descend→grasp→lift→transport→
  place as `fsm_phase` nodes) + bind pick-place guards/laws in a registries module; reuse
  `collect_*_demos`-style dwell-consistent collection and the BC/TD3+BC pipeline.
- Known FANUC history: warm-start bridge collapse (fix = TD3+BC + ≥1e5 steps, [[project-fanuc-offpolicy-collapse]]);
  pick-place "win" was an explosion artifact — real ≈0.125, unsolved, divergence guard needed in metric
  ([[project-pick-place-explosion-artifact]]).

Related: [[project-galambos-reward-fixed-rl-below-demo]] (the push controller precedent + overnight run).
