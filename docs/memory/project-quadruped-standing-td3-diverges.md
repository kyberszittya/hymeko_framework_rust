---
name: project-quadruped-standing-td3-diverges
description: Quadruped standing (Rung-2 postural) NOT solved by pure-TD3 at any budget (Q-overestimation divergence); GPU compile path fixed; BC warm-start is the next lever
metadata: 
  node_type: memory
  type: project
  originSessionId: d8544f20-a9dc-4c9c-9180-6d1373e0ede0
---

2026-07-03 (Hajdu, "fix GPU first" then "launch"): two linked outcomes on the quadruped **standing** task
(`base=free, task=stand`, DwellMetric held ≥200/250, SA-HSiKAN TD3 + LayerNorm twin critics, vec-8).

**1. GPU `torch.compile` CUDA-graph crash FIXED.** `train_offpolicy` compiles `_critic_loss` + `_actor_loss`
as two `reduce-overhead` CUDA graphs sharing one memory pool; interleaving them (actor re-invokes `critics[0]`)
overwrote buffers the critic graph's autograd held → "accessing tensor output of CUDAGraphs overwritten by a
subsequent run" at `a_loss.backward()`. Fix = `torch.compiler.cudagraph_mark_step_begin()` per `_update_once`
(guarded by `cudagraph_step = compile and dev==cuda`; inert on CPU). NOT quad-specific — reproduces on cartpole
under pure-TD3+cuda+compile whenever the actor update fires. GPU now ~185 steps/s vs ~41 CPU (~4.5–5×). Regression
test `test_cuda_compile_interleaved_graphs_no_overwrite` (CUDA-gated) verified to fail pre-fix. `ddpg.py` is
non-core. Report `reports/2026-07-03-quadruped-gpu-cudagraph-fix.md`.

**2. Pure-TD3 does NOT solve standing — at 60k OR 150k.** stand-rate median 0.000 both; the 60k seed-0 0.24 did
NOT reproduce at 150k (→0.0; rates [0.0,0.02,0.0]). Actor loss climbs monotonically -7→-44 (Q inflating) with
late critic spikes = **Q-overestimation** (measured signature; LayerNorm bounds it only partially). More steps
made it worse — the 0.24 was pre-divergence, not a foothold.

**3. UPDATE 2026-07-03 — the OBJECTIVE was mis-specified (Hajdu caught it), not just Q-overestimation.** Old
STAND_REWARD paid an UNCONDITIONAL `alive` +0.5 + upright, so a crouched/collapsed-but-not-inverted robot scored
≈ a standing one (flip_cos=-0.2 rarely terminates) and height was dominated → the reward did NOT require standing,
so stand_rate stayed ~0 while the policy "learned" (evaluation-metric-integrity failure). FIXED: dominant term is
now `standing` (+1 iff upright>stand_cos AND |z-h|<tol = the exact DwellMetric), `alive` REMOVED, torso_height
up-weighted; step() reordered so the term reads the current pose. Regression test: standing beats a crouch by >4.0
(old: ~0.3, would fail). **BUT fixed reward + pure TD3 still 0.0 at 100k** — objective is necessary, NOT
sufficient; pure TD3 can't DISCOVER balancing from a free-falling base. **How to apply:** BC warm-start
(PD-hold-q0 demo) → TD3+BC on the FIXED reward is the lever ([[project-fanuc-offpolicy-collapse]],
[[project-ditch-ppo-offpolicy-sahsikan]]). Artifacts `experiments/2026_07_03_14_57_quadruped_standing/`,
`.../19_19_quad_stand_fixedreward_smoke/`. Distinct from goal-reach [[project-quadruped-collaboration-derisk]].
