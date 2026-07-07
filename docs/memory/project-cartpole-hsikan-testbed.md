---
name: project-cartpole-hsikan-testbed
description: "HyMeKo cart-pole = canonical HSiKAN actor-critic validation testbed; emitter gotchas (slide-range deg→rad, actuates-every-joint) handled in env"
metadata: 
  node_type: memory
  type: project
  originSessionId: 3060e292-680f-4645-82c1-156ce78e537c
---

Built 2026-06-21: `data/robotics/inverted_pendulum.hymeko` + `hymeko_rl/env/inverted_pendulum_env.py` +
`hymeko_rl/train_inverted_pendulum.py` — the canonical **inverted pendulum** as a clean testbed that
isolates the HSiKAN actor-critic from the hard Galambos grasp task. `hg_state` comes from the `.hymeko`
(2-vertex cart/pole signed graph). Validated: 120-iter PPO smoke, init_return 25.8→152.3, untrained
upright-steps 27.8→trained 144.0/200 — it learns *and* the task discriminates. Report:
`reports/2026-06-21-cartpole-hsikan-wirein.md`.

**Two emitter gotchas (non-obvious, both handled in the env layer, NOT the `.hymeko`):**
1. The MJCF emitter actuates **every** non-fixed joint → for an under-actuated robot, strip the passive
   joint's `<motor>` via the new `arm_world.strip_actuators(mjcf, joint_names)` (here `["hinge"]` → nu=1).
2. `hymeko_formats/src/transforms.rs:619` applies a deg→rad factor to **slide (prismatic)** joint ranges —
   wrong for a translational DOF. Avoid by leaving the slide unlimited (the cart bound is an env
   termination). A future *limited*-rail cart-pole would need the emitter fixed or the range stripped.

**Wiring:** `ppo.py::train_ppo` now depends on a `RolloutEnv` Protocol (not concretely `ArmReachEnv`), so
the same PPO loop trains reach or balance — fix the algorithm, vary task/architecture. Backbone swap via
`build_policy("hsikan"|"mlp")` unchanged.

**Why:** the user asked to validate the HSiKAN actor-critic on a known benchmark because "the policies
could be more efficient" (see [[project-hymeko-rl-phase2-debug]], [[project-galambos-reward-shaping]]).
**How to apply:** for any new MuJoCo RL env from a `.hymeko`, reuse `emit_arm_mjcf` + `strip_actuators` +
`HypergraphState.from_mjcf` + the `RolloutEnv` Protocol; do NOT re-add the rotor cos/sin joint encoding
([[project-rotor-joint-encoding-falsified]] — falsified). Open: reward still in code, not `meta_reward`;
HSiKAN-vs-MLP at parity on a 2-vertex graph likely shows no structure benefit (record honestly).

**Policy efficiency — MEASURED + acted on (2026-06-21):** the batch-1 policy forward `ac.act` is **87%**
of a single-env rollout (phase timing); the forward is **dispatch-bound** (per-call ~flat to B=8, per-sample
2.19ms→0.087ms B=1→32), NOT FLOPs (2-vertex graph). Fix shipped: **vectorized rollout** `ppo._collect_vec`
+ `train_ppo(n_envs>1, make_env=…)` — batch the forward over N lock-step envs → **3.1× faster wall** (vec
N=16: 147s vs 452s/120it) with learning preserved (161 vs 144 upright). Single-env `_collect` path kept
untouched (reach reproducibility + the truncation-bootstrap fix). Plus: torch `set_num_threads(1)` at the
CLI entry (~10%, tiny-tensor case) and emit MJCF once shared across workers (`InvertedPendulumEnv(mjcf=…)`,
`emit_cartpole_mjcf`) to avoid N CLI subprocess calls. Report:
`reports/2026-06-21-vectorized-ppo-rollout.md`. **Gotcha:** vec `final_return` under-reports (per-iter
horizon n_steps/N < episode length); use `upright_steps` eval as the true metric. **HSiKAN vs MLP** (2-vtx, 5-seed): structure is **NOT load-bearing on cart-pole — capacity is** (control
overturned the first read). HSiKAN 192±15 (5/5); but a **params-matched MLP (26.7k) ties it: 195±8, 5/5**;
over-param MLP (135k) = 200±0. The "MLP fails 2/5" was an UNDER-PARAMETERIZED baseline (9k, hidden=64), not
absent structure. Artifacts: `reports/2026-06-21-cartpole-{multiseed,controls}.jsonl`. **Lesson: always run
the params-matched control before crediting structure** (same trap as 2026-06-18 rotor-vs-MLP-embed). Caveat:
2-vtx graph has no topology → cart-pole can't test the architecture; fair test = 6-DOF arm or Galambos. PPO
*algorithm* baseline stands; *architecture* claim doesn't.

**REAL-TOPOLOGY TEST DONE (2026-06-22, Galambos coin-grasp, 6-vtx):** HSiKAN (28745p) vs params-matched MLP-96
(28521p), single seed, 300-iter PPO. **STILL no structure advantage** — comparable within noise, MLP higher
peaks (best 28.9 vs 10.0), HSiKAN final marginally better (-6.4 vs -9.6). So on BOTH cart-pole AND the
6-vtx coin task, signed-hypergraph structure ≈ matched MLP. CAVEAT: single seed, huge variance (need
multi-seed); structure may be too thin (2 simple 2-link arms ≈ tree); fixed-incidence HSiKAN (signedkan /
structural-entropy-feedback UNTESTED — the "exploit further" line). `reports/2026-06-22-galambos-structure-vs-capacity.md`.
Galambos PlanarGraspEnv: 6-vtx hg, obs (6,8), 4 actions, RolloutEnv+HSiKAN compatible; train_planar_grasp
(PPO+strategy from galambos_strategy.hymeko). mlp baseline needs obs_dim=n_vtx*feat=48 (NOT per-vertex 8).

**QUADRUPED (2026-06-22, 14-vtx richest topology):** scaling fixture `quadruped_d3_t0.hymeko` emits a usable
MuJoCo robot (nbody 15, 13 joints); add a `<freejoint>` to the torso (else fixed base, can't fall) + floor →
floats & stands. Minimal QuadEnv (forward-velocity reward, 14-vtx obs (14,2)) one-off smoke 40-iter PPO:
HSiKAN 27k params ties MLP-112 (33k, over-param) on best (11.2 vs 11.1) — param-competitive, HINT of
param-efficiency on rich topology, but 40-iter smoke / single seed / imperfect match → NOT a win.
**THREE-TASK VERDICT: HSiKAN never worse, never decisively better than matched MLP** (cart-pole tie / coin-grasp
tie / quad param-competitive). Decisive test = longer, properly-matched, MULTI-SEED quad run (does the
param-efficiency edge widen?). Env not formalized (one-off script). `reports/2026-06-22-galambos-structure-vs-capacity.md`. Fusing the 3 `_SignedConv` linears remains a forbidden micro-opt without a profile.

**PROPER JUMPING QUADRUPED (2026-06-22)** — the box-cluster `quadruped_d3_t0` is NOT a real quadruped, and its
freejoint was added by a body-name regex that **silently never matched** (root link merges into `<worldbody>`),
so that "locomotion" smoke ran **bolted to the world**. Replaced by `data/robotics/quadruped.hymeko` (torso + 4
two-link legs, 8 DOF, 9 vtx) + `hymeko_rl/env/quadruped_env.py` (`QuadrupedJumpEnv`). **World fixation is
DECLARED in the .hymeko** (`@base` joint) and promoted by name in the env — `base="free"` → `<freejoint>`,
`base="fixed"` → welded — NOT auto-injected (user directive). Carrier is a non-fixed `conti_joint` because a
fixed root MERGES into worldbody; non-fixed keeps torso a real `<body>` for a clean freejoint swap. Native
`free_joint` = a CORE `hymeko_query::JointType` edit (deferred, needs APPROVED-CORE-EDIT). **RL learns to jump**
(peak 0.6–0.69 m, +0.18 m above standing) ONLY after two fixes: (1) cut `alive_w` 0.2→0.05 (standing was a
reward trap, returns fell 41→19), (2) **normalise action space to ±1** scaled to ±50 N·m internally (std≈1 was
starving exploration on the raw ±50 range). Open-loop mirrored squat-extend de-risk = +0.46 m (front/back legs
push oppositely → vertical thrust). Per-scenario **sanity tests** `test_scenario_sanity.py` (parametrized
geometry-well-formed + capable-of-moving over all 3 envs) + `test_quadruped_env.py` = 21 passing.
`reports/2026-06-22-quadruped-jump.md`. **Lesson: `capable_of_moving` sanity (actuation changes per-vertex obs)
would have caught the bolted box-cluster immediately** — geometry that emits is not geometry that moves.
