---
title: Pick-place capstone — the honest capability + generalized lessons
date: 2026-07-16
scope: does pick-place work, on the task that requires it, and what generalizes from the RL/causal arc
status: capstone (synthesis + assimilation)
core_touched: none
---

# Pick-place capstone: yes, we can pick and place — measured on the task that requires it

## The honest capability (far-spawn split, N=48 @ seed0=20000, place_radius 0.07 m)

~42 % of benchmark episodes spawn the box **already within place_radius of the target** — no pick-place needed, and a
good policy correctly leaves them alone. Measuring skill on the **far-spawn** episodes (58 %, which require a real
grasp→lift→place):

| policy | far-spawn grasp | far-spawn placed | note |
|---|---|---|---|
| **scripted v3 expert** | **1.00** | **0.89** | the deployable controller — genuinely picks and places |
| **FF-DAgger base** (learned) | 0.96 | **0.75** | a real 75 % learned pick-place policy |
| LSTM / residual-SAC clones | 0.00 | 0.00 | fail the real task (never grasp) |
| idle (zero action) | 0.00 | 0.00 (1.00 on near-spawn only) | the ~0.458 "floor" = trivial episodes |

**Correction to F-PP-013:** the ~0.458 idle floor is **benchmark triviality** (spawn-at-target episodes), *not*
inflation of a failing policy. The base is not "0.42 skill" in a bad sense — on the episodes that need pick-place it
succeeds 75 % (grasp 96 %). Read pick-place skill on the far-spawn subset (or `grasp∧place`), not the raw metric.

## What generalizes (the framework-level lessons from the RL + causal arc)

1. **Metric integrity — split by task difficulty.** A benchmark can carry a large do-nothing floor (here 42 %
   trivial episodes). Always split by whether the task is non-trivial and report skill on that subset; a headline
   number that a zero-action policy scores 0.458 on is not a skill number. (F-PP-013, corrected.)
2. **Two structural walls bound learned control over a good demonstrator** (F-PP-009/010/011/012):
   *setpoint wall* — a residual on a frozen base only holds it, never beats it; *exploration wall* — from-scratch RL
   never discovers a contact grasp by random exploration (PPO + SAC, optimizer-agnostic). **Imitation→RL is the only
   viable learned path**, and it *is* the deployed base (75 % on the real task).
3. **A good scripted/model-based controller beats learned policies on contact manipulation** (expert 89 % vs base
   75 %), and imitation clones it to ~75 %. For best performance, deploy the scripted controller; use imitation for a
   differentiable policy; use residual RL only to *hold* the base safely.
4. **Causal attribution needs an intervention, not just aggregates** (F-PP-014/015). The signed-hypergraph
   DirectLiNGAM is a strong *diagnostic* — it surfaced the metric confound and formally showed only the DAgger'd base
   has a real grasp→success chain — but cross-policy / 2-variable LiNGAM is under-identified for edge *direction*.
   A within-policy **intervention** (training progression / ablation) decides it: this confirmed F-PP-009 causally
   (*deviation from the base causes grasp loss*, corr −0.48). "DirectLiNGAM proposes; the ablation decides."
5. **An ensemble needs a genuinely different second policy.** A causal gate over {FF, LSTM, SAC} is pointless because
   only the base has real skill; clones/RL are ~0 and DAgger does not rescue them (F-PP-015). A worthwhile gate needs
   a second *complementary* controller (e.g. model-based/planning), not another clone.

## Deployable entry points (the working pick-place, already in the tree)
- **Scripted expert (89 %):** `fanuc_pick_env(expert_version=3, require_settle=True)` + `expert_action_fn()`; run it
  in the GUI (`hymeko_rl.gui.qt_pick_place` / `pick_place_scene`).
- **Learned base (75 %):** `experiments/hybrid_dagger_gif/policies/hybrid_dagger_hsikan_s0_best.pt` via
  `gripper_pick_bc.load_pick_policy`.
- **Regression guard:** `hymeko_rl/tests/test_residual_abstraction.py` (zero-residual==base; reactive-target≥base).

## Provenance
- Branch `integration/fanuc-pick-place-canonical`; findings F-PP-009…016. Verification `scratchpad/` split-by-spawn
  probe (N=48). No CORE. No kato15. No new repo code.
