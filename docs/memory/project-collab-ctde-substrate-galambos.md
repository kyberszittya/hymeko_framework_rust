---
name: project-collab-ctde-substrate-galambos
description: "2026-07-03 — collab CTDE beats joint on coin-toss (0.40>0.34), HyMeKo-declarative-MDP substrate PoC works (0.96), Galambos env changes (fingertip-only contact + two-arm-force) implemented"
metadata: 
  node_type: memory
  type: project
  originSessionId: d8544f20-a9dc-4c9c-9180-6d1373e0ede0
---

2026-07-03 session (Hajdu + Galambos-sensei feedback). Report `reports/2026-07-03-collab-ctde-substrate-galambos.md`.

**1. Off-policy collaborative CTDE BEATS joint on the coin-toss.** Built `DeterministicMultiChannelActor`
(2 per-arm backbones coordinated by the existing `MultiTreeChannel` — arms talk through the coin/zone couplings;
per-arm output heads over a shared backbone are algebraically a single head, so coordination must live in the
*reasoning*) + `build_collaborative_offpolicy` (centralized twin LayerNorm QCritics). `train_offpolicy`
generalized additively (`action_dim` property + backbone-optional shared-trunk guard). 3 seeds × 200k
best-checkpoint: **peak delivery median 0.40 [0.20,0.40,0.44] vs joint baseline 0.34** — coordination helps
(NOT param-matched — collab has 2 actor backbones; follow-up = vs widened joint). Late Q-collapse persists.
Extends [[project-kato-dual-discriminator-plan]] / [[project-actor-critic-shared-reasoning]].

**2. HyMeKo declarative control-substrate PoC WORKS.** `data/robotics/toy_reach.hymeko` declares a point-mass
reach MDP END-TO-END (state dims, observed channels, dynamics params, target, `reward_spec` reusing
reach_distance+action_cost). `hymeko_rl/env/hymeko_mdp.py::HymekoReachEnv.from_hymeko` reads it via existing
`read_scene_fields`/`RewardSpec.from_hymeko` (no new parser). Same off-policy TD3 as robots → reach median 0.96
(untrained 0.1, P-ctrl floor 0.55). The [[project-hymeko-as-control-substrate]] thesis in miniature — one .hymeko
= state+obs+dynamics+objective, algorithm-agnostic backend.

**3. Galambos-sensei env changes (from the first collab GIF).** (a) *only the yellow fingertip contacts the coin,
the arm can't* — DONE via collision bitmasks (coin bit 2; arm links default 1/1 can't touch; fingertip geom
conaffinity 3 can; floor opened to 3). Both arm paths (hand-authored + emitted `with_fingertip_sites`). Test
`test_only_fingertip_can_touch_the_coin`. (b) *two robots' force to move the coin* — MECHANISM done:
`coin_frictionloss` = dry Coulomb friction on the slide joints (a real force threshold, not damping's linear drag),
opt-in. **FINDING: (a) BREAKS the old scripted demonstrator** (it herded with the arm bodies; fingertip-only makes
its strategy mis-aimed → moves coin wrong way). So the task is now genuine fingertip manipulation; demonstrator +
policies need re-tuning, and (b)'s threshold can't be dialed until a working fingertip controller exists.
**PHYSICS route (arm literally can't touch coin) is the principled fix to knock/collision** — the reward-penalty
route (armcol 2.0, A/B) shows the historical suppression (delivery collapses 0.14→0 while crashes drop). Reward
work: 2 new terms `arm_body_collision` (excludes the pinch) + `finger_contact` (graded per fingertip) in
galambos_task.hymeko [[feedback-reward-definition-in-hymeko]]. Also: GPU cudagraph fix
[[project-quadruped-standing-td3-diverges]]; page-file constraint resolved (overlapping torch runs now OK).

**4. Coordination-reward A/B FALSIFIED (2026-07-04).** Added `both_approach = -max(left,right)` fingertip
distance (penalise the LAGGING arm — the simultaneity gradient the compensable mean `grasp_approach` lacks),
in `galambos_task_coord.hymeko`; A/B vs baseline, same collab off-policy driver, 3 seeds × 200k. Verdict:
**coord 0.12 ≤ baseline 0.16 (no help), both_contact stayed ~0.002 in BOTH** — reward-term shaping is NOT the
coordination lever (matches the reward-rebalancing-is-second-order finding). Report
`reports/2026-07-04-galambos-coord-ab.md`. TWO lessons: (i) **harness-calibration caveat** — my new
Campaign-based `exp_galambos_coord_ab.py` reproduced baseline at 0.16 not the recorded 0.40 (env did NOT
regress: masks + 0.40 are the same commit f8a5b57, constants change kept MJCF byte-identical) → the A/B driver
under-reproduces the 0.40 setup's BC/TD3+BC hypers; trust the WITHIN-driver delta, calibrate before more term
runs. (ii) **both_contact≈0 corroborates the broken demonstrator** (point 3: fingertip-only masks broke the old
arm-herding demo) → the real lever is a FIXED two-fingertip BC demonstrator or a `coin_frictionloss` curriculum,
NOT term weights. Did NOT reflexively re-tweak terms (the just-run test predicts failure; don't chase a phantom).
Also this session: hymeko_neuro merge (signed_kan+signedkan_wip, two distinct cores), env constants ontology,
humanoid.hymeko authored+validated (13-DOF) + [[project-quadruped-standing-td3-diverges]] locomotion plan.
