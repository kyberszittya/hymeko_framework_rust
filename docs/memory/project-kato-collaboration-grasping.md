---
name: project-kato-collaboration-grasping
description: "Kato-sensei lab collaboration on learned grasping via the tensorised robot hypergraph; MuJoCo RL demo plan, blocked on dep approval"
metadata: 
  node_type: memory
  type: project
  originSessionId: 5fd53482-77c4-4262-be8f-95b4bcaaa64f
---

Collaboration opening with **Kato Shohei (加藤昇平), kato_shohei** — after Csaba's
2026-06-18 talk. Kato's ask: the kinematic robot is a simple hypergraph; can the *same*
hypergraph become a tensor and then **learn movement** (reaching, grasping, popping)?
Csaba replied yes, with an honest boundary: the hypergraph→IR→star-expansion→PyTorch
(Arrow+DLPack, topology-hash-gated, zero-copy) bridge is **real**, and kinematics-from-
topology to ~5 cm is **real**, but a **learned control head** on that tensorised
hypergraph (imitation or RL in the MuJoCo loop) is the **natural next step, not a
finished result**. He proposed exploring it jointly — "strong fit with your lab."

**Plan:** `docs/plans/2026-06-18-mujoco-rl-grasping/` (4 artifacts). Staged: Ph0 wire the
existing bridge into a state encoder; Ph1 REACHING via behaviour cloning (MVP, no reward,
shown milestone first); Ph2 GRASPING via in-repo PPO (new gripper+object MJCF); Ph3
deploy-tiny-policy (Nagare/embedded) + warm-start from the kinematics-from-topology prior.

**Why:** real external collaboration, and the demo is the concrete home for the rotor/
HSiKAN + embedded line — the distinctive claim is that the policy reads the *compiled*
hypergraph tensor, not raw joints (a unit test must assert this, else it's a generic RL
grasp).

**2026-06-19 — pipeline is now end-to-end from one `.hymeko` source.** Building the env
surfaced + fixed three emit bugs (all NON-CORE; CORE extractor was correct — proven by a
regression test): B-005 (CLI emitted via static templates hardcoding `axis="0 0 1"`) and
B-004 (MJCF template had no `<joint>`) fixed by rerouting `emit -f {mjcf,urdf,sdf}` to the
model-based emitter in `hymeko_cli`; plus two `emit_mjcf` fixes (root `world`→implicit
worldbody, `<inertial pos>`) so the MJCF loads. `hymeko emit -f mjcf <arm>` now yields a
loadable, articulated 6-DOF arm (EE workspace ~1.2×0.9×0.8 m). Then wired the RL env to it:
`ArmReachEnv.from_hymeko(robot, obs_profile=...)` builds scene + kinematic hypergraph +
**declarative observation** from one source — `node_features` is now assembled from an
`ObservationSpec` (`hymeko_rl/env/observation.py`) read from the obs profile, AND the reward
from a `RewardSpec` (`hymeko_rl/env/reward.py`) read from a task profile — so the **whole
reaching MDP (robot + obs + reward) now comes from `.hymeko`** (`from_hymeko(robot,
obs_profile=, task_profile=)`). New vocab `data/robotics/meta_reward.hymeko` + profile
`arm_reach_task.hymeko`; the obs/reward readers share `env/_profile.py::read_bundle` (narrow
parse, B-003 bridge). Both specs are bit-identical to the former procedural code (obs
`array_equal`, reward `== −dist`). Reports: `2026-06-19-hymeko-emit-kinematic-rerouting.md`,
`-hymeko-rl-env-from-hymeko.md`, `-hymeko-rl-declarative-observation.md`,
`-hymeko-rl-declarative-reward.md`. 55/55 hymeko_rl tests green (incl. the once-flaky
`test_ppo_improves_return`: fixed by `PPOConfig.seed` + `train_ppo` seeding `env.reset(seed=)`
— ppo.py:163 was unseeded → §3 entropy reliance; runs now reproducible, 3/3). **Open:**
(b) `AgentSpec.from_hymeko` can now compose obs_dim + reward +
vertex count into a full MDP spec. (c) profiles declare effector/target vertex bindings the
env doesn't yet consume (uses procedural ee/target).

**2026-06-19 (cont) — architecture comparison (HSiKAN vs MLP) attempted; blocked on torque-BC
setup.** Built the comparison harness `hymeko_rl/reach_arch_compare.py` (BC, both backbones ×
seeds, median/IQR). Smoke-gated (§3) and HALTED twice on measured contradictions: the emitted
6-DOF arm's torque expert wanted ~2600 N·m vs ±25 clip (80% saturated). Fixed the **joint-limits
CORE gap** (token `extract-joint-limits-ref`): `extract_joint_limits` now follows the `limit ->`
ref (own + inherited) → emitted MJCF/URDF/SDF carry declared range + effort (j0/j1 ±500, j2-jtool
±50). 212 hymeko_query + reach tests green. BUT the comparison is **still not delivered**: needs
(a) realistic masses (placeholder 2-25 kg → ~0.5-2 kg; ripples to `anthropomorphic_arm_using.hymeko`
+ 5 fixture tests) AND (b) **action normalization** (±500 torque targets unregressable by a ±1-init
net → BC loss ~18k) — a broad action-interface change. Then attempted the full effort ("A"):
re-applied masses + implemented **action normalization** (Box(±1) action space; expert returns
torque÷ctrlrange, step rescales). Normalization WORKED for fitting (BC loss 18000→0.003) — a real,
correct env improvement — BUT hit a 4th, fundamental layer: **BC compounding** — the clone fits the
demos yet doesn't reach (hsikan 0.703 / mlp 0.714 vs expert 0.058 over 80 steps on the redundant
6-DOF arm). That's a BC limitation, not a bug; needs PPO (on-policy, robust to compounding) or
DAgger. PPO-on-torque is itself known-hard (existing PPO test uses position). So the canonical
6-DOF **torque** comparison is a multi-session research effort, not a bounded task. **Reverted
masses + normalization + kp back to the clean limits fix** (212 hymeko_query + 58 hymeko_rl green);
the normalization is documented to re-apply for the PPO effort. Then (user chose) **authored a tractable
canonical arm**: `data/robotics/reach_arm.hymeko` — a 4-DOF base(Z)-shoulder(Y)-elbow(Y)-wrist(Z)
arm, light masses, arm_world-equivalent kinematics (~0.64m workspace), **position-controlled**
(emit_arm_mjcf gained `control_mode=` that retargets the emitted torque motors → `<position>`
servos + joint damping; from_hymeko + emitted_arm_factory now take control_mode/ee_body). **BC
GATE PASSES**: both backbones learn on the canonical reach arm (hsikan 0.464→0.240, mlp 0.454→0.226,
loss ~0.002). So the canonical comparison is now viable. 59 hymeko_rl tests green; reach_arm.hymeko
emits 4 joints, axes Z/Y/Y/Z, loads + articulates.

**✅ 2026-06-19 — 5-seed comparison DELIVERED** (`reports/2026-06-19-reach-arch-result.{md,json}`).
On the 4-DOF position-controlled canonical `reach_arm.hymeko`: hsikan reach 0.2401 m (IQR 0.0199,
28.7k params) vs mlp 0.2262 m (IQR 0.0208, 13.9k params), floor ~0.46 m. **MLP marginally ahead at
HALF the params; the 1.4 cm gap is well inside both IQRs → statistically indistinguishable.** No
HSiKAN advantage on a serial chain — the EXPECTED control outcome; the structural prior is meant to
pay off on redundant/branched morphologies, not a 4-DOF chain. Reported honestly, not dressed up.
Also fixed a stale report-label bug (main() hardcoded "anthropomorphic 6-DOF" while the factory runs
reach_arm 4-DOF). 14 reach tests green; wall 277 s.

**⏭ NEXT RL TRACK (user-chosen 2026-06-19): the Galambos planar grasping scenario.** Spec in
`docs/demo/galambos_scenario/` (gitignored — collaborator sketch + private notes). Galambos-sensei's
"hello world": two PLANAR elbow manipulators (thumb/index), a disk spawned random per episode, both
arms pull it into a fixed target zone between them; MuJoCo, Hymeko as central model. He explicitly
called the 6-DOF arm too complex. This is the showcase morphology (two-arm/branched) the serial-arm
result above could not be. NEW MuJoCo env on the existing hymeko_rl env framework (arm_reach_env /
observation / reward / ppo) — **plan-gated (§2, 4 artifacts) + discovery pass (§6.1) BEFORE code.**
**2026-06-19 — reach safety/config penalties built** (`reports/2026-06-19-reach-safety-penalties.md`,
plan `docs/plans/2026-06-19-reach-safety-penalties/`). New `hymeko_rl/env/safety.py` (`SafetyState`
+ `compute_safety`); `ArmReachEnv` gained **opt-in `enable_safety`** (default off — preserves the BC
comparison + emitted-arm tests): floor injection, death-terminate on ground-contact ∨ self-collision,
target reject-sampled `≥ reach_min_radius` from base (outside the robot). 4 model-declared reward
terms (`ground_penalty`/`self_collision_penalty`/`joint_limit_penalty`/`below_ground_penalty`) in
`meta_reward.hymeko` + new `arm_reach_safe_task.hymeko` (BOUNDED weights, ground/self 5.0 — NOT
unbounded, per the Phase 2 critic-corruption finding [[project-hymeko-rl-phase2-debug]]). Also fixed
`_profile.py` to read `.hymeko` as UTF-8 (Windows cp1250 bug). 95 hymeko_rl tests green. **6-DOF blocker RESOLVED same session** (the spurious self-collision was
NOT just fat geoms — the emitted scene doesn't honour MuJoCo `filterparent`, so adjacent links
touching at their joint counted as self-collision): fixed by (a) `compute_safety` excluding
parent/child + same-body geom pairs explicitly, (b) `slim_arm_collision` (×0.4 cylinder radii when
safety on), (c) compile-time floor seating via `_arm_home_floor_z` (runtime `model.geom_pos` does
NOT move a plane's collision surface — verified the hard way). 6-DOF home now contact-free, 0 instant
deaths. **Termination is now model-declared too**: `meta_task.hymeko` gained a `termination` vocab +
`termination_spec`; `hymeko_rl/env/termination.py::TerminationSpec` (kind→predicate, from_hymeko);
`arm_reach_safe_task.hymeko` declares `@dies_when` and `ArmReachEnv.from_hymeko` reads it (fallback
`DEATH_ON_CONTACT`). So reward AND death predicate both come from the .hymeko. Viewer/render keep
`enable_safety=False` (target-outside via reach_min_radius=0.2 still applies). Open: PPO retrain
under arm_reach_safe_task (now viable on the 6-DOF arm).

**✅ 2026-06-20 (late) — EMITTER FIXED so HyMeKo describes PROPER robots + verify tool**
(`reports/2026-06-20-robot-generation-status-and-next-steps.md`). The robot-generation defect the
user flagged: the MJCF emitter (`hymeko_formats/src/transforms.rs`, NON-CORE) emitted every geom at
the body origin (joint) and COM at 0 0 0 — links were disconnected stubs, mass off-centre. FIXED:
geom `pos` AND `<inertial>` COM now = the link's geometry `origin` → connected rods + centroid mass.
212 generation tests still green. `galambos_planar.hymeko` rebuilt as a proper top-down planar 2-arm
(Z-hinges, box rods w/ origin midpoints → connected, reachable). New `hymeko_rl/env/verify_arm.py`
(`verify_arm`→ArmReport: loads/articulates[pos+roll]/connected/reach) — galambos/reach/anthropomorphic
all verify [OK]. `PlanarGraspEnv` uses the HyMeKo-emitted arm; obs = per-vertex hypergraph + link→coin
+ coin→zone. 107 hymeko_rl tests green. **PPO still 0 goals (env correct, policy untrained — needs
training+curriculum).** NEXT: (1) train Galambos to goals (curriculum); (2) apply the same origin fix
to URDF/SDF emitters (still broken there); (3) 6-DOF self-collision-during-MOTION blocks the safety
scoreboard; (4) generalize bc._make_policy to size from observation_space; (5) verify_arm as a
`hymeko verify` CLI verb; (6) commit viewer/safety/j3/scene_style (still uncommitted).

**⚠️ 2026-06-20 — Galambos CORRECTED to TOP-DOWN TABLE** (`reports/2026-06-20-galambos-topdown-correction.md`).
First build misread it (coin FALLING under gravity in a VERTICAL plane). User corrected: MuJoCo +
planar robots, **coin PLACED in reach on the 2D plane, not dropped**; chose **top-down horizontal
table**. Rebuilt: `galambos_planar.hymeko` now AXIS_Z joints (chain sweeps in XY — verified planar) +
BOX links (horizontal rods; cylinders point out of plane). `PlanarGraspEnv` coin = planar table body
(slide-x/slide-y/hinge-z, confined to plane, damping≈friction), PLACED at random reachable (x,y) at
reset, pulled in-plane to a centre zone; death = coin knocked out of workspace. 5 tests updated+green.
Re-trained 100it: return −15.2→−13.3, PPO mean −15.4 vs random −30.9, **still 0 goals** (hard task;
needs more training/curriculum). Lesson for future: the Galambos coin is PLACED in-reach, NOT falling;
plane is the top-down table.

**(superseded) 2026-06-20 — Galambos planar env first build** (`reports/2026-06-20-galambos-planar-env.md`,
the vertical-falling misinterpretation).
`hymeko_rl/env/planar_grasp_env.py`: `compose_planar_scene` (injects a 3-DOF planar disk
[slide-x/z, hinge-y] + zone site + floor into the emitted two-arm MJCF, appended so arm body indices
stay aligned), `PlanarGraspMetrics`/`compute_planar_metrics` (disk_to_zone, per-finger contact,
in_zone), `PlanarGraspEnv` (obs = per-vertex (6,6) on the TWO-ARM hypergraph — the distinctive claim;
disk spawns OUTSIDE the zone; success-termination). Declarative reward `galambos_task.hymeko` (dense
pull REUSES reach_distance with disk_to_zone; + both_contact 0.5 + in_zone +10 + action_cost); 2 new
term kinds in meta_reward + reward.py (RewardSpec.evaluate env→Any, duck-typed across both envs).
5 planar tests + full hymeko_rl 102 green; ruff/mypy clean; **PPO smoke (2 iters) finite, 6.7s** —
trainable by the in-repo ppo.py. **Open:** committed PPO runner (generalize bc._make_policy to size
from observation_space, not obs_spec) + full multi-hundred-iter training to a learned pull + render.
NOT committed yet (uncommitted increment on 73ee5a6).

No existing planar/two-finger env in hymeko_rl as of 2026-06-19. **Plan written + compiles**
(`docs/plans/2026-06-19-galambos-planar-grasping/`, 4 artifacts). User decisions: pure PPO +
shaping, full-task milestone, **kept as a SEPARATE track from grasp-ball-reward** (single-arm jaw
lift, oracle — still unimplemented). **DE-RISK CLEARED 2026-06-19:** authored
`data/robotics/galambos_planar.hymeko` (two 2-link planar fingers as ONE branching kinematic tree,
all axes Y → XZ plane); `hymeko emit -f mjcf` walks the two-chain branch off `world` cleanly →
loads in MuJoCo (nu=4, 7 bodies), hypergraph spans 6 vertices across both fingers. **So the whole
Galambos build is NON-CORE — no hymeko_query escalation needed.** Disk + target zone are scene
objects injected by the env (emitter has no free-/planar-joint). Next: PlanarGraspEnv + reward
terms + PPO runner + tests per the plan.

**⭐ 2026-06-22 — AFFECTIVE / INTERACTION MODEL as a HyMeKo model (user flagged "could be a big hit for
Kato").** Idea: external **human OR agent** input shapes the reward **at runtime** — a valence `v∈[-1,1]`
from a pluggable `AffectiveSource` (neutral/human/agent), coupled two ways (additive `affective` reward term
+ runtime modulator `(1+gain·v)·Σw·term`). User's key refinement: **define the affective/interaction model AS
a HyMeKo model** (`meta_affect.hymeko` vocabulary + `affect_spec` bundle in a task `.hymeko` + `AffectSpec.from_hymeko`
reader) — so the interaction protocol is a first-class declarative artifact, parallel to meta_reward/meta_kinematics,
and thus **auditable** (the steering channel is declared, not hidden glue). The pitch to Kato: structurally
accountable AND human-/agent-steerable grasping head; an agentic source = ANOTHER agent scoring this one →
ties to the shared-agent-reward-model line ([[project-xprofile-instance-refs]]) + the fuzzy-defuzzification
line ([[project-fuzzy-defuzzification-heads]]) (valence = fuzzy membership defuzzified into a reward nudge).
**Status: PLAN-ONLY per user ("just plan for now") + §2.** Plan (4 artifacts, pdf built):
`docs/plans/2026-06-22-affective-interaction-model/`. Runtime core PROTOTYPED `hymeko_rl/env/affect.py`
(AffectiveSource Protocol + Neutral/Human/Agent + AffectiveModulator; lint/mypy clean; **NOT wired in**).
Implementation (meta_affect.hymeko + reader + env poll-hook + `affective` term + tests + demo) is OPEN —
BACKLOG **P1**. Non-core (meta_*.hymeko + hymeko_rl). Also this session: generic fast-and-smooth reward terms
(`time_penalty`/`joint_velocity`/`joint_acceleration` bounded-Δq̇ jerk) DONE in meta_reward+registry (coin
joint speed 13.9→4.9 rad/s); proper jumping quadruped (`quadruped.hymeko`, declared `@base` world-fixation)
DONE; but quadruped GOAL-reaching **locomotion FAILS** (jump-geometry is a poor walker, 0% reached — BACKLOG P3).

**✅ 2026-06-23 — ARM TOP-DOWN GRASP SOLVED via the FANUC LR Mate config** (`reports/2026-06-23-fanuc-lrmate-top-down-grasp.md`).
Pick-and-place line: Phase 0 (faithful arm+gripper+box+plane) + Phase 1 path B (floating-gripper expert→BC,
HSiKAN 0→75%, GIFs sent) done earlier. Built `arm_gripper_import.hymeko` proving **HyMeKo cross-model KINEMATIC
attachment works** (`@`-import arm + `using robot as arm` + joint on `arm.tool`) — see [[project-xprofile-instance-refs]].
Then path A (real IK): `hymeko_rl/env/ik.py::DampedPoseIK` (iterative DLS pose-IK, tested). **Diagnostic
(falsification): the `anthropomorphic_arm.hymeko` STRUCTURALLY cannot grasp top-down** — kinematically valid
down-poses ALWAYS self-collide (`link_0↔link_3`, `link_3↔tool`: fat r=0.075 links + Z-X-Z-twist wrist fold onto
themselves). Axes/pedestal/radius FK-probes were misleading (ignored collision). User's instinct (robot/axes
wrong) was right. **Fix = `data/robotics/fanuc_lrmate.hymeko`** (FANUC LR Mate axis config **Z,Y,Y,Z,Y,Z** = base
yaw + Y shoulder/elbow + **Z-Y-Z spherical wrist**; slim r≈0.03–0.045 collision cylinders — meta has no capsule)
+ `arm_gripper_fanuc_import.hymeko`. Result: **collision-free top-down grasps at r∈[0.20,0.40], down-ness 1.00,
NO pedestal** (spherical wrist points tool down without folding). Scripted expert **grasps 8/8, lifts** (seed 2:
25 cm, GIF `reports/gifs/fanuc_pick.gif` sent). PickPlaceEnv now parameterised (mount_height/obj_radius/arm_home
[non-singular ready pose] + gravity-comp `body_gravcomp` + grip-settle dwell + straight-up-then-transport).
Runnable: `python -m hymeko_rl.render_pick_place`. Arch confirmed with user: **HyMeKo = base description (robot
now; reward/obs still Python for pick-place, .hymeko in reach), HSiKAN = actor-critic**.

**2026-06-23 (cont) — RELIABILITY PASS done + BC port hits the compounding wall.** Reliability fixes (each
trace-isolated): wide-open fingers on descent (`_OPEN=-0.014`, grip joints unlimited) so the box isn't knocked;
capture grasp xy ONCE (`_lift_xy`) so the straight-up lift doesn't chase the drifting tool; committed-phase
rate 0.28 (not 0.4) for a holdable lift; heavier box (`box_mass=0.15`). **Expert now grasp 10/10, lift 9/10,
PLACE 9/10** over the random workspace (full pick-and-place; GIF re-sent). New env params box_mass/obj_angle/
n_actions. **BC ported** (generalised `gripper_pick_bc` helpers to a `PickEnv` Protocol + `only_success` demo
filter; `hymeko_rl/pick_place_bc.py`; `python -m hymeko_rl.pick_place_bc`). **BC clone FITS the demos
near-perfectly (loss 3.6e-4, t0 action matches expert <1°) but picks 0% — AND 0% in-distribution → pure
BC-compounding/covariate-shift over the 620-step horizon**, not a fit failure (same wall the 6-DOF arm BC hit;
expert's hidden phase state — grip-settle counter + captured grasp xy, not in obs — compounds it). Floating
gripper BC worked because short-horizon + direct-Cartesian. **NEXT for a LEARNED arm pick: DAgger or PPO (BC is
compounding-bound), not more BC**; Phase 2 (reward/obs as `pick_place_task.hymeko`) also adds task-phase obs
features that shrink the non-Markovian gap. 15 tests green; report `reports/2026-06-23-fanuc-lrmate-top-down-grasp.md`.

**2026-06-23 (cont) — PPO (BC warm-start) ran, PARTIAL.** `hymeko_rl/pick_place_ppo.py` (runnable; reuses generic
`train_ppo`); 70 iters/143k steps HSiKAN, BC-warm-started, ~32 min; checkpoint `checkpoints/fanuc_pick_ppo_hsikan.pt`,
curve `reports/figures/fanuc_ppo_return.png`. **Return −32→+83** (noisy, upward). Greedy eval: **learned to
approach + finger-box contact 7/8** (real gain over BC's 0) **but only nudges the box (max_lift ~1.2 cm) — full
grasp-lift did NOT converge** at this budget (the firm-grip-then-lift is the hard-exploration step). PPO is the
right tool + shows clear learning on the HyMeKo structure, but a reliably-picking learned policy needs more steps
+ grasp-lift reward shaping/curriculum, THEN the HSiKAN-vs-MLP ablation. Reliable pick today = the scripted IK
expert (10/10 grasp, 9/10 place).

**2026-06-23 (cont) — ground/table collision handling + "EVIL" env generator.** PickPlaceEnv now: a real TABLE
(`compose_pick_place_scene(table_top=)`) the box rests on + the arm on a matching PEDESTAL (`mount_height==table_top`)
so it works ABOVE the floor (0 ground contact); sky-gradient skybox + checker floor + lights (render no longer
black — `reports/gifs/fanuc_pick.gif`); grasped-box gravity-comp (`body_gravcomp` toggled when held → no sag/slip
→ place 10/10). **Approach-vs-grasp contact distinguished**: `info["approach_contact"]` = manip-surface contact
while NOT over the object (tool_horiz>0.06) → penalised; contact while over/grasping the object is allowed (user:
"no problem to contact the table during grasp, but not during approach"). The scripted expert at the LEVEL mount
still drags low on approach (can't hover high at the reach radius); a higher mount fixes the drag but breaks
grasp centering (reach-limit tradeoff) — the env-level penalty is the real fix (shapes a LEARNED policy);
perfecting the scripted demonstrator is a follow-up. **Evil-env generator `hymeko_rl/evil_pick.py`** (user's idea):
one `difficulty`∈[0,1] knob perturbs the SAME HyMeKo structure adversarially (object→reach edge + wide angles,
heavy, small, slippery fingers via new `finger_friction` param); `make_evil_env`, `robustness_sweep` → the
scripted expert's place rate degrades 0.8→0.0 easy→evil. Framings: domain randomization / curriculum /
**adversarial accountability** (deliberately break the policy, report where — the honest counterpart to a
cherry-picked success; great Kato angle). New params on PickPlaceEnv: box_z_half, finger_friction, ground_penalty,
obj_angle, table_top, mount_height. 15 pick/evil/ik tests green.

**2026-06-23 (cont) — Phase 2 declarative reward + curriculum PPO + HSiKAN article skeleton & ablation plan.**
Phase 2 DONE: `data/robotics/pick_place_task.hymeko` + `meta_reward.hymeko` pick terms + `PickMetrics` cache +
`PickPlaceEnv(reward_profile=)` using `RewardSpec` — **parity test PASSES** (declarative reward == procedural,
bit-for-bit); 16 tests green. So the whole pick MDP (robot+state+action+reward) now comes from `.hymeko`.
**WMI hang root cause = concurrent instances (resource contention), NOT WMI itself** (user clarified); mitigation
when contended = run RL CPU-only (`CUDA_VISIBLE_DEVICES=-1`, dodges GPU enumeration); else GPU is fine. **GPU is
little help for these RL jobs — rollout-bound (MuJoCo CPU), small HSiKAN nets.** Curriculum PPO (`set_difficulty`
knob + on_iteration ramp; `hymeko_rl/pick_place_ppo.py --curriculum`): return **−321→+93 at diff 0.87, final
−89** (curriculum works) BUT **greedy place 0% at all difficulties — on-policy PPO does NOT crack the full
pick-place** (learns approach+contact only; same wall as plain PPO). → learned-control needs **off-policy
(DDPG/SAC, ~250× sample-eff — already in repo)**, not PPO. Scripted IK expert stays the reliable pick.
**HSiKAN article skeleton** `docs/drafts/hsikan-article-skeleton.md` (honest why-better/when-worse/trade-offs;
in-hand vs to-run split) + **4-artifact ablation plan** `docs/plans/2026-06-23-hsikan-ablations/` (A/B#1 HSiKAN-vs-
KAN-vs-MLP params@iso-acc; A/B#2 serial-vs-branched morphology on OFF-POLICY). Honest pub take: systems/accountability
paper submittable NOW (parity + evil-env); the "better" ML claim rides on A/B#2 (branched). See [[project-rl-algorithm-roadmap]]. **Kato meeting framing: structural-unification thesis (one HyMeKo model →
kinematic structure + state-domain/hypergraph + action interface; policy reads the same structure) is
solid+demonstrable; full-declarative MDP proven on reach / in-progress on manipulation; manipulation = POC
(scripted pick reliable, learned pick converging).**

**How to apply:** (1) dep approval for mujoco+gymnasium was granted (the RL line runs;
algo in-repo, no stable-baselines3). (2) Keep the framing to Kato as "imitation reaching +
RL grasping POC", never "solved manipulation" — Csaba's reply was well-calibrated; don't let
the demo writeup outrun it. (3) The workflow is `source → kinematic model (→ MJCF + hypergraph)
→ obs/state/reward over the hypergraph → HSiKAN actor-critic → action → MuJoCo`; the obs half
is wired, reward is the open piece. Ties to [[project-hero-demo]], [[project-ur-sim-setup]],
[[project-seminar-demos-and-hymeyolo-plan]].
