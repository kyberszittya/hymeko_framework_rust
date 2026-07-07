---
name: project-galambos-reward-shaping
description: "Galambos planar grasp \"not moving\" root-caused to a FROZEN SHOULDER (emitter base/first-link geom overlap pins j1), not reward/training; fixed via adjacent-link contact excludes → 5/8 goals"
metadata: 
  node_type: memory
  type: project
  originSessionId: e049ea12-7387-4a59-87f4-051966d7cfcb
---

**Done 2026-06-20** (`reports/2026-06-20-galambos-reward-shaping.md`, plan
`docs/plans/2026-06-20-galambos-reward-shaping/`). User: "the RL scenarios are not moving."

**Root cause (found by a CHAIN of discriminating tests, not a guess):**
1. NOT the PPO truncation bug — already fixed in `ppo.py:108-118` (see [[project-hymeko-rl-phase2-debug]]).
2. Diagnostic rollout (`hymeko_rl/diagnose_planar_grasp.py`, NEW, read-only): trained policy makes
   **0/160 both-finger contact**, coin moves 0.004 m, reward ≈100% the near-flat disk→zone `pull`.
3. Added a dense `grasp_approach` reward term + ent_coef, retrained → **STILL 0 contact** (measurement
   contradicted the plan → stopped, didn't stack hacks).
4. Constant-target test: elbows (j2) track, **shoulders (j1) frozen** (tracking err 1.19 rad, stuck at 0).
5. Contact listing: `base_left↔upper_left` + `base_right↔upper_right` ACTIVE with −0.022 m penetration;
   both bodies share the SAME origin `[-0.14,-0.02,0.04]` — the emitter places the first link ON the
   base hub, and that self-contact pins the shoulder. MuJoCo `filterparent` did NOT remove it.
6. CONFIRMED: inject `<contact><exclude base↔upper>` → shoulder err 1.19 → **0.16 rad, shoulders move**.

**Fix:** `planar_grasp_env.py::adjacent_link_excludes(arm_mjcf)` — compiles the arm alone, reads exact
parent→child body topology, emits `<contact><exclude>` for every adjacent-link pair (robust to emitted
`upper/lower` vs hand-authored `link1/link2`). Injected in `__init__`. Regression test
`test_shoulder_joint_is_not_frozen_by_self_contact` (holds target, asserts shoulders rotate).

**Result (retrain shoulder-freed + approach reward, 150 it):** return **−45.6 → +21.2**; coin disp
0.004 → **0.055 m**; **goals 0/8 → 5/8 episodes**. DECISIVE VARIABLE = the shoulder unfreeze (same
reward on the frozen arm = 0 goals). **Honest nuance:** `both_contact` stays 0 — the policy *pushes*
the coin (asymmetric/single-arm), not a two-sided pinch; valid for "coin→zone", but a true grasp needs
contact-specific shaping. The dense `grasp_approach` term is kept (sound, but necessary-not-sufficient).
110 hymeko_rl tests green, ruff+mypy clean.

**✅ PROPER EMITTER FIX DONE same day** (`reports/2026-06-20-mjcf-parent-contact-excludes.md`):
`hymeko_formats/src/transforms.rs::emit_mjcf` now emits `<contact><exclude body1=parent body2=child>`
per non-`world` joint, so EVERY emitted multi-link robot (6-DOF, reach_arm, WAM, DRC-Hubo) gets
adjacent-link contact filtering. The env-level `adjacent_link_excludes` workaround was REMOVED.
Shoulder now tracks the target EXACTLY (error ~0, vs 0.16 with the env-only exclude). 3 hymeko_formats
mjcf tests + 214 hymeko_query integration + 110 hymeko_rl all green; clippy/ruff clean.

**⚠️ CURRICULUM = NULL (2026-06-20, `reports/2026-06-20-galambos-curriculum.md`):** reverse start-state
curriculum (coin near zone → anneal out; `PlanarGraspEnv.difficulty`, general `train_ppo(on_iteration=)`
hook, `--curriculum-iters`) did NOT move the goal rate — still **5/8, the SAME 5 episodes**. Corrected
diagnosis: the 3 misses are **control-precision** (ep4 overshoot, ep2 wrong-direction, ep1 undershoot),
NOT start-state — curriculum can't fix push precision. Code KEPT (tested infra, `difficulty=1`=identity,
hook is general). Don't re-try curriculum here. **Next lever = settle/overshoot reward shaping** (coin-
velocity penalty near zone + directional term), or more compute, or a true two-sided pinch.

**⚠️ SETTLE (overshoot-brake) = NEGATIVE (2026-06-20, `reports/2026-06-20-galambos-settle-and-gifs.md`):**
both gatings (near-zone w=0.3 AND in-zone) scored 4/8 < 5/8 — trades overshoot for undershoot. REVERTED
out of `galambos_task.hymeko`; kept as opt-in vocab (`meta_reward.@settle` + extractor + `disk_speed`
metric, tested). So 5/8 is robust to BOTH knobs tried (curriculum null + settle negative); the real
lever for >5/8 is STRUCTURAL = a true two-sided pinch (`both_contact` still 0 — solved by pushing).

**GIFs: `hymeko_rl/render_planar_gifs.py`** (reuses `evaluate.render_episode_gif` + top-down camera) →
GIFs per run in `reports/gifs/<run>/`. Rendered best runs: `galambos_freed/` + `galambos_curriculum/`
(5 goal-seed GIFs each). Offscreen MuJoCo render WORKS on this Windows box.

**USER LIKES GALAMBOS as the "hello world" problem (2026-06-20)** — good showcase morphology.

**✅ HARDER TASK + DECLARATIVE ENV DONE (2026-06-20, `reports/2026-06-20-galambos-declarative-env.md`):**
De-degenerated per user: (a) SMALL zone (half 0.04) RANDOMIZED each episode in both-arm reach
(`model.site_pos` moved at reset; observed via coin→zone obs channel), (b) coin spawns over the
reachable TABLE incl. OUTSIDE the arm band, (c) wider stance ±0.14→±0.18 in `galambos_planar.hymeko`
(reach centres read from model). **THE ENV IS NOW A `.hymeko`** (user's idea, THREE-PATH form):
`meta_env.hymeko` (vocab) + `galambos_env.hymeko` (scene: zone/coin/workspace/success as config terms
in an `env_spec` bundle) + `hymeko_rl/env/env_spec.py::EnvSpec.from_hymeko` (reads via the same
`_profile.read_bundle` as RewardSpec) + `PlanarGraspEnv.from_hymeko(robot=, env=, task=)`. So the WHOLE
MDP (robot+scene+reward) is from .hymeko; a new test case = a new `galambos_env.hymeko`, no Python.
**Elegance:** `__init__` 16→6 args by passing the `EnvSpec` config struct (not unpacked kwargs).
**Re-baseline = 1/8** on the harder task (was 5/8 easy) — much harder (small moving zone + far coins).

**✅ PRECISION + ANTI-STALL (2026-06-20, `reports/2026-06-20-galambos-exploration-precision.md`):** user
directed: drop `action_cost` (it rewarded stationarity), add `center_bonus` (graded +1@centre→0@edge,
precision) + `arm_motion` (anti-stall, −max(0,v_min−arm_speed)). Result **1/8→2/8**. HONEST attribution:
**center_bonus WORKS** (goals centre tightly, dz→0.034 inside 0.04 zone); **anti-stall does NOT** —
stationary timeouts went UP (1→4), all on far-coin spawns where the arm approaches but won't engage then
freezes. So the user's "timeout = no exploration" hypothesis is NOT binding here; freezes are
unreachable-far-coin give-ups. Real lever = budget / reach-out curriculum / structural two-sided pinch
(`both_contact` STILL 0 — pushing, not grasping). New metric `PlanarGraspMetrics.arm_speed`; new terms
`arm_motion`/`center_bonus` (opt-in vocab; both in canonical task now). 123 hymeko_rl tests green.
GIFs `reports/gifs/galambos_{freed,curriculum,harder,explore}/`. Checkpoints `ppo_explore.pt` (2/8 best
on harder task), `ppo_harder.pt` (1/8). Open: center-only ablation to confirm attribution.

**✅ IT WORKS ON THE HARD TASK — 5/8 (2026-06-20, `reports/2026-06-20-galambos-strategy-and-disk.md`):**
Three user asks done: (a) SMALLER DECLARATIVE DISK (`EnvSpec.disk_radius` 0.035→0.02, in
`galambos_env.hymeko @disk{radius}`); (b) proximity confirmed disk-CENTRE-based; (c) **EXPLORE/EXPLOIT
STRATEGY AS .hymeko** (user idea) — `meta_strategy.hymeko` + `galambos_strategy.hymeko` +
`hymeko_rl/strategy_spec.py::StrategySpec.from_hymeko` → PPOConfig + `log_std_init` + curriculum;
`build_policy(log_std_init=)`; `train_planar_grasp` reads it. WHOLE pipeline (robot+env+reward+strategy)
now data. **Result: WIDER exploration (`log_std_init -0.5`, std~0.6) → 5/8 on the HARDER task** (small
disk + randomized small zone + far spawns), return −65.9→−3.6, and the arms NOW CONTACT (ep0 7/ep1 39/
ep7 85 contact-steps) and pull FAR disks to dead-centre (ep5 0.172→0.012). My worry "noise kills
precision" was WRONG — noise drove engagement, dense reward kept centring. 3 failures = OVER-pushing now
(opposite of freezing) → fix by annealing log_std DOWN late (explore→exploit schedule, declarative).
Checkpoint `ppo_strategy.pt`. GIFs `reports/gifs/galambos_strategy/`.

**⚠️ 100-SEED EVAL + ARM-ARM PENALTY = KEY HONEST FINDING (2026-06-20,
`reports/2026-06-20-galambos-arm-collision-and-eval.md`):** `hymeko_rl/eval_planar_grasp.py` (N-seed
goal rate + Wilson CI). The 5/8 was NOISE — real rate **25%** (CI 17.5-34.3%) over 100 held-out seeds.
THEN user asked to penalize arm-arm contact → declarative `@arm_collision` term (-1 on left↔right geom
contact; new `PlanarGraspMetrics.arm_self_contact`). It EXPOSED that the 25% policy was DEGENERATE:
arms in mutual contact **72.5%** of steps — it MASHED the two arms together and shoved the disk as a
clump, not a two-finger grasp. Penalty → **0% clash** (works perfectly) but honest two-arm score drops
to **13%** (+20 deaths, separated arms over-push). PENALTY KEPT — it converts a clumped-pusher hack into
a legitimate (weak, undertrained) two-arm task; the drop is honest difficulty. NEXT: retrain the no-clash
setup with MORE budget (150 iters too few for separated arms). Checkpoints `ppo_noclash.pt` (13%,
legitimate), `ppo_strategy.pt` (25%, clumped). 128 tests green.
**400-ITER NO-CLASH: 17% goals BUT 67% DEATHS** (strategy n_iters 150→400 via .hymeko edit; was 20
deaths). More training → separated-arm policy SHOVES the disk BALLISTICALLY, knocks it OUT 2/3 of the
time. ROOT: NO reward penalty for death/out-of-bounds (death only terminates). PATTERN across all
iterations = freeze→push→clump→knock-out; every reward patch reveals a new degenerate shortcut. = THE
ARGUMENT for the attractor-field direction [[project-gsphf-attractor-planning-integration]] (explicit
attractor flow structurally can't knock the disk out). Checkpoint `ppo_noclash_long.pt`.
**⭐ SCALE CORRECTION (user 2026-06-21): my runs were ~1/40 of proper PPO scale.** At n_steps=512,
1000 iters ≈ 0.5M timesteps; real PPO continuous-manipulation needs ~**40000 iters ≈ 20M timesteps**.
So the low goal rates were UNDERTRAINING, not a hard task. TRAJECTORY (legitimate no-clash setup, 100-seed
eval): goals **13%@150 → 17%@400 → 28%@1000** (RISING), deaths **67%@400 → 53%@1000** (FALLING) = learning
a valid policy, NOT stuck. User was right; my "plateaued / pivot to attractor" call was premature. LAUNCHED
**40000-iter checkpointed run** (`ppo_40k`, `--checkpoint-every 1000` → `ppo_40k.<i>.pt`, resumable via
`--resume`; strategy n_iters=40000; ~33 CPU-h @ 3s/iter). Trainer now supports `--checkpoint-every` +
`--resume` (via the on_iteration hook). Eval intermediate `ppo_40k.<i>.pt` to track the curve. The attractor
controller is now the COMPLEMENT/warm-start ([[project-gsphf-attractor-planning-integration]]), not a rescue.
Also made RL architecture diagrams
(`docs/architecture/RL_SETUP.md` + `Diagrams/rl_{pipeline,agent}.{tex,pdf,png}`). HSiKAN = BOTH actor +
critic (separate nets, `build_policy` builds two HSiKANBackbones). **Open:** (1) settle-shaping for the 3 precision misses. (2) two-sided pinch (`both_contact`
stays 0 — solved by pushing). (3) URDF/SDF emitters express adjacent-link filtering differently if needed.
Checkpoints: `ppo_freed.pt` (5/8 baseline), `ppo_curriculum.pt` (5/8, null), `ppo_shaped.pt` (frozen, 0),
`ppo.pt` (original).

**⚠️ KNOCK-NOT-GRASP CONFIRMED + QUANTIFIED (2026-06-27, `hymeko_rl/diag_contact.py`, user asked "did the
fingertips ever contact the coin?"):** MLP @25k on galambos, 50 greedy eps: delivered_grasp **1**, delivered_knock
**17**, death 3, timeout 29 → **grasp_fraction_of_deliveries 0.056** (94% of deliveries are KNOCKS), TOTAL 4
fingertip-contact-steps over 50 eps. So `both_contact≈0` (the persistent issue since 2026-06-20) PERSISTS even
after the 2026-06-26 fingertip-reward fix (that fixed the APPROACH measurement, not the grasp). Explains the
negative return at "45% delivery": the +10 in_zone fires for knocks; both_contact(+3) never fires. CAVEAT: 25k
steps ≈ 1/800 of proper PPO scale (undertrained) — but history shows grasping doesn't emerge at higher budgets
either → structural. **THE UNTRIED LEVER = GRASP-GATE SUCCESS:** make in_zone(+10) REQUIRE both_contact /
env._ever_grasped so a knock earns NOTHING → forces a real pinch. History added penalties (clash/oob) but NEVER
gated success on contact. Fix = galambos_task.hymeko edit + a new `grasp_deliver` term in the (non-core)
hymeko_rl/env/reward.py registry. Reframes Galambos work: "make deliveries be GRASPS", not "push delivery %".
Diag found grasp_fraction via the env step info["both_contact"] (evaluate only logs goal/death/timeout, saves no
policy → had to retrain). Still the standing argument for the attractor-field direction
[[project-gsphf-attractor-planning-integration]] (flow can't knock the disk out).

**⛔ GRASP-GATE FALSIFIES REWARD-SHAPING-FOR-GRASPING (2026-06-27, the decisive negative):** built the one
untried reward lever — `grasp_deliver` term (reward.py, non-core) = +1 only if in_zone AND env._ever_grasped;
galambos_task.hymeko gated: in_zone 10→1 (knock token) + grasp_deliver 12 (real grasp), so a KNOCK earns +1 not
+10. 3-layer wired (reward.py term + meta_reward.hymeko vocab + galambos_task.hymeko bundle), 14 reward tests green.
RESULT (MLP gated@100k vs ungated@25k, diag_contact): grasp_fraction **0.056 → 0.056 (IDENTICAL)**, contact-steps
4→2, but deaths **3→28** + timeouts 29→4 — removing the knock reward + 4× budget made the policy MORE aggressive
(shoves harder → knocks coin OFF table), NOT grasping. CONCLUSION: **grasping is STRUCTURAL/exploration, not a
reward-weight problem** — reward-shaping is now FALSIFIED as the path to a two-finger pinch (consistent with
both_contact≈0 across ALL historical patches). KEEP the gate (honest reward design — knocks shouldn't earn the big
bonus) but it does NOT solve grasping. CAVEAT: 100k ≈ 1/200 of proper scale, but the gate should help
DIRECTIONALLY if reward were the issue; it didn't. REAL LEVERS now = structural: attractor-field
[[project-gsphf-attractor-planning-integration]] (flow can't knock out), grasp PRIMITIVE / coin-as-grasp-hyperedge
[[project-galambos-hsikan-tie-rootcause]], demonstration/BC of a pinch, or 20M-scale. Reward grind for grasping =
DEAD; don't re-tune weights to chase a grasp. Move to the DTC rung-2 quadruped (bigger prize) or a structural grasp.

**CONVERGENCE PLOT (2026-06-21):** `hymeko_rl/plot_runs_overlay.py` → `reports/gifs/galambos_convergence.png`
(reward-trace overlay of the 150/400/1000 runs + goal-rate-vs-budget panel). Fresh **common-test** eval
(100 seeds, `--seed0 3000`, CURRENT env, `eval_planar_grasp`): **150→20%, 400→13%, 1000→27%** — NON-monotonic,
but the Wilson CIs overlap heavily (13.3-28.9 / 7.8-21.0 / 19.3-36.4), so the 400-dip is suggestive not
significant; only 1000>400 is near-clear. These are INDEPENDENT runs on an evolving env, not one checkpointed
trajectory. The 40000-iter run is the real convergence test (still the open item). Editor side: the whole MDP
is now browsable as a multi-file project → [[project-editor-mdp-project]].
