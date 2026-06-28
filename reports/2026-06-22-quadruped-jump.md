# Proper jumping quadruped + per-scenario sanity tests

**Date:** 2026-06-22
**Branch:** soma-vision
**Scope:** Replace the box-cluster scaling-fixture "quadruped" with a real four-legged HyMeKo robot that
**jumps**; make the world attachment an **explicit declaration** (not an auto-injected freejoint); add
**per-scenario sanity tests** (geometry well-formed + capable of moving).

## Summary

The earlier "quadruped" was `quadruped_d3_t0.hymeko`, a graph-scaling fixture (uniform box links). Worse, its
freejoint was added by a body-name regex that **silently never matched** — the fixture's root link merges into
`<worldbody>`, so that robot trained **bolted to the world**. This change ships a purpose-built robot and a
clean, declared base attachment.

- **Robot** — `data/robotics/quadruped.hymeko`: torso (box) + four two-link legs (revolute hip + knee, both
  about AXIS_Y), 8 actuated DOF, 9 kinematic-hypergraph vertices.
- **World fixation, declared (not automatic)** — the base joint `@base` is declared in the `.hymeko` at the
  rest height. `QuadrupedJumpEnv(base="free"|"fixed")` reads it **by name** and promotes it: `free` →
  `<freejoint>` (floating base, can jump), `fixed` → welded. No body-name guessing, no hidden injection.
  - The merging behaviour of a *fixed* root is why the carrier joint is declared non-fixed (`conti_joint`):
    a non-fixed world joint keeps the torso a real `<body>`, so the free-joint promotion is a clean,
    single-element rewrite. Native `free_joint` support would be a **CORE** edit to `hymeko_query`'s
    `JointType` (lockdown: implementation) — deferred, flagged for approval.
- **Jump** — reward = `height_w·max(0, z−rest) + rise_w·max(0, vz) + alive_w − ctrl_w·‖a‖²`, terminate on
  tumble. RL **learns to jump** after two fixes (below).

## What made it actually jump (two non-obvious fixes)

The first PPO smoke **failed** (rose +0.009 m, returns *fell* 41→19). Diagnosed, not guessed:

1. **Standing was a reward trap.** `alive_w=0.2` paid ~40/episode for merely not falling, so the policy
   converged to standing rather than risk a dynamic jump. → cut `alive_w` to 0.05 and made height/launch the
   dominant terms.
2. **Exploration was starved.** Actions were ±50 N·m but the Gaussian explored with std≈1, so the coordinated
   push the geometry needs was never sampled. → **normalised the action space to ±1** (scaled to torque
   internally) so unit-variance exploration covers the whole range.

After both: **`LEARNED=True`** — eval peak jump mean **0.599 m** / best **0.688 m** (rest 0.420), rose
**+0.18 m** above standing, returns 118→125 (max 151) over 80 PPO iters, single seed.

Open-loop de-risk independently confirms the geometry is jump-capable: a mirrored front/back squat-extend
leaves the ground by **+0.46 m** (front and back legs push oppositely so the thrust is vertical, not a pitch).

## Files touched

| File | Change |
|---|---|
| `data/robotics/quadruped.hymeko` | **new** — torso + 4 two-link legs, declared `@base` world joint (≈63 LOC) |
| `hymeko_rl/env/quadruped_env.py` | **new** — `QuadrupedJumpEnv`, `set_base_mode`, `JumpReward` (≈190 LOC) |
| `hymeko_rl/tests/test_scenario_sanity.py` | **new** — parametrized geometry+motion sanity, 3 scenarios (≈100 LOC) |
| `hymeko_rl/tests/test_quadruped_env.py` | **new** — base modes, stands, jumps, flip, perf (≈120 LOC) |
| `scripts/compare_backbones_gif.py` | quad task now the jumping robot; height HUD; removed box-cluster env |

**CORE.YAML items touched:** none. (Native free-joint = a flagged future `hymeko_query` edit, **not** done.)

## Tests

- **Sanity (`test_scenario_sanity.py`)** — parametrized over cart-pole / coin-grasp / quadruped:
  `geometry_well_formed` (DOF + hypergraph counts, finite mass/inertia), `observation_contract`,
  `capable_of_moving` (actuation moves the per-vertex state and stays finite — catches dead actuators / a
  bolted base), `reset_deterministic`.
- **Quadruped (`test_quadruped_env.py`)** — `set_base_mode` free/fixed/missing, DOF counts, stands-under-
  gravity, **geometry-can-jump**, flip-terminates, fixed-base-never-flips, + a perf budget (200-step median
  < 2 s, tracked peak < 256 MB).
- **Result:** `21 passed` (12 sanity + 9 quadruped). All prior env/render tests unaffected.

Gates: `ruff` clean, `mypy --strict` clean on changed modules.

## Performance

Single-seed smoke, CPU, `torch.set_num_threads(1)`, vectorized PPO `n_envs=8`. Per-step env throughput well
within the perf-test budget (200 steps median < 2 s). Peak RSS far under the 16 GB cap (toy model, 10 bodies).

## Open issues / follow-ups

- **Native `free_joint` in HyMeKo** — would let the floating base be fully described in the `.hymeko` instead
  of promoted in the env. Requires a CORE edit to `hymeko_query` `JointType` (+ emitter free-joint support).
  Flagged for `APPROVED-CORE-EDIT`.
- Single-seed result; a multi-seed run would firm up the jump-height number.
- **HSiKAN-vs-MLP on the jump task (first visible structure edge):** HSiKAN (26.7k params) eval mean peak
  **0.718 m** (rose +0.298 m), best 0.834 m; matched MLP (23.2k) mean peak **0.557 m** (+0.137 m), best
  0.680 m — single seed, 80 iters. HSiKAN jumps ~2× higher above standing here, the first task (vs cart-pole
  tie / coin-grasp tie / old-quad param-competitive) where structure shows a visible advantage. **Caveats:**
  single seed, high RL variance, MLP slightly fewer params; needs multi-seed before crediting structure (same
  discipline as every prior task — do not over-claim on one seed). Gif:
  `reports/gifs/compare/quadruped_jump_hsikan_vs_mlp.gif`. Checkpoints:
  `checkpoints/quadruped_jump_{hsikan,mlp}.pt`.
- Leg damping/armature are env-side sim tuning (the `.hymeko` geometry can't carry them); fine, documented.
