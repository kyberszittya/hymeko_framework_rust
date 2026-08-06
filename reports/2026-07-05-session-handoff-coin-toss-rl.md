# Session handoff — k-arm coin toss RL (2026-07-05, 06:05 JST)

Complete state for continuation by any agent or human. Every number below has a disk artifact; paths given.

## Cleanup Note (Codex, 2026-07-05)

Stage labels are now quarantined in `reports/2026-07-05-galambos-stage-ledger.md`.

Read every number below through this lens:

- `scripted_controller`: hand-designed push/plow controller, useful but not learned policy evidence.
- `bc_clone`: behaviour cloning from the scripted controller; this is the source of the 0.44-0.52 learned artifact.
- `rl_refined`: TD3+BC/SAC continuation after BC; measured runs degrade the clone.
- `framework_substrate`: the intended dataflow event + FSM + monitor machinery. A Galambos-specific FSM is only a scenario-local prototype, not proof that the framework substrate exists.

## Bottom line

- **Task:** two planar arms deliver a coin (cylinder) into a target zone; metric = dwell delivery rate
  (`DwellMetric(in_zone, success_steps)`, 50 eps, eval seed 9000, difficulty 0.3, 300-step horizon).
- **Scripted push controller (hand-designed, declared in HyMeKo): 0.84.** The best delivering controller.
- **Best current learned artifact is the BC clone: 0.52** (3 seeds: 0.44/0.52/0.52, all step-0 floors,
  `experiments/2026_07_05_03_29_galambos_coord_ab_deliver/`).
- **Every RL continuation tested tonight made the clone WORSE, never better** (details below). The open
  levers that do not use a critic: DAgger (driver ready, below) and demo scale-up (folded into it).
- User verdict on the arc: on the raw delivery metric, hand engineering beat the learning stack. The
  learning line's remaining justification is amortization across tasks (k-arms, FANUC, humanoid) — untested.

## What is measured (all 2026-07-05, this machine, CPU MuJoCo)

| policy / stage | delivery | artifact |
|---|---|---|
| OLD pinch-carry teacher | 0.205 (pooled 200 eps) | reports/2026-07-05-galambos-bc-only-localization.md |
| push controller (new teacher) | 0.80–0.84 (3×50 eps; press 0.012) | reports/2026-07-05-push-controller-demonstrator-hybrid-fsm.md (historical filename) |
| BC clone, 27,475-sample demo set, b128×200 epochs | **0.52** (0.44/0.52/0.52); best current learned artifact, not RL-refined improvement | experiments/2026_07_05_03_29_galambos_coord_ab_deliver/results.json |
| BC clone, b512×100 epochs | 0.34 median | experiments/2026_07_05_02_12_galambos_bc_only/results.json |
| hybrid per-mode learned laws (declared FSM + tiny nets) | 0.28 — TIED with flat BC | experiments/2026_07_05_02_48_galambos_hybrid_modewise/ |
| TD3+BC (certified deliver reward, 3×200k) | peak = its own BC floor, all seeds | …03_29_galambos_coord_ab_deliver/run.log |
| TD3+BC repair cells (50k each, floor 0.32 at b512): σ=0.01 / bc_coef=10 / adaptive_bc | ALL FAIL (0.02–0.18 end) | experiments/2026_07_05_05_1*_galambos_coord_ab_dx_*/ |
| TD3+BC repair cell: critic_warmup=20k | verdict pending at handoff time | task output brz796nsz / its experiment dir |
| SAC + BC warm-start, 100k (joint arch) | in flight; curve so far 0/0.42/0/0/0 at 10–50k | task output b2b104zg3; checkpoint checkpoints/galambos/sac_bc_sahsikan_s0.pt when done |

**Mechanism table (why the off-policy stage fails):** noise exonerated (collapse at σ=0.01); fixed anchor
only slows decay (bc_coef 10: 0.18@25k→0.02); adaptive anchor fails too. The Q-maximization term's update
direction reduces delivery under every coefficient tried; root cause (critic mis-fit vs replay shift vs
reward scale) NOT yet isolated — do not assert one without a discriminating test.

## In-flight processes at handoff (self-terminating; kill by PID if the machine is needed)

- Diagnosis cells: PID 20744 (4th cell `latewarm`, ends ≈06:15), log → its `experiments/..._dx_latewarm/`.
- SAC: PID 36628 (≈45 min remaining), prints `[sac]` lines; saves checkpoint + GIF dir `experiments/2026_07_05_sac_galambos`.

## The next lever, ready to run (NOT yet started)

`scratchpad/dagger_coin_toss.py` (session scratchpad; copy is self-contained — also reproduce from this
description): DAgger — roll the CLONE, label every visited state with the closed-loop scripted controller
(`PushDemonstrator` recomputes from current state, so it labels any state), aggregate, retrain, repeat;
rounds: 400 teacher eps base (2× prior), 3 DAgger rounds × 100 clone eps; b128×200-epoch BC (the 0.52-quality
recipe); keeps best across rounds. Expected: closes part of the 0.52→0.84 compounding-error gap. ~50 min wall.

## The declarative stack (single source of truth — all working, all tested)

- **Controller:** `data/robotics/galambos_push.hymeko` (phases approach→push→hold, `on <event> to <phase>`
  transitions, gait scalars) + vocabulary `meta_controller.hymeko`; walked by
  `hymeko_rl/control/controller_spec.py::ControllerSpec`; Python binds named GUARDS (STL-robustness margins,
  ρ>0 fires, margins logged on events) and LAWS in `hymeko_rl/experiments/galambos_demo.py::PushDemonstrator`
  (k-arm general: rotation-group slot fan, permutation assignment; law injection for learned per-mode laws).
- **Reward:** `galambos_task_deliver.hymeko` — oracle-certified delivering (de-annuitized, terminal_deliver 30).
  Baseline `galambos_task.hymeko` is certified NON-delivering (farming) — never train against it.
- **Experiment:** `galambos_ab_deliver.hymeko` (budget/arms/@offpolicy overrides) + `meta_experiment.hymeko`;
  read by `exp_galambos_coord_ab.py::AbConfig.from_hymeko`; CLI = `python -m hymeko_rl.experiments.exp_galambos_coord_ab <profile> [--smoke]`.
- **Machine-enforced gates:** pre-queue reward certification (`_certify_variants` raises on uncertified;
  waiver = `uncertified_waiver`); smoke CAPS profile budgets (`resolve_budget`); Campaign evaluates the
  step-0 BC floor (warm start always in the best-checkpoint race). All regression-tested
  (`hymeko_rl/tests/test_galambos_demo.py`, 18 tests; `test_bc.py`, `test_sac.py`).
- **SAC on this task:** `galambos_bc.py --algo sac` (env-agnostic `train_sac`, live `[sac]` logging added).

## Standing rules learned/enforced this session (in CLAUDE.md + memory — read before acting)

oracle-certify the TRAINING reward before any queue; measurements are cached facts (grep reports/results.json
before running; re-measure ONLY the changed cell); consult the user before coining any name (technical or
Japanese-Hungarian inspiration only); scenario is called "k-arm coin toss"; no per-monitor-event chatter;
batch work into few turns; RL/imitation vocabulary (demonstration set, replay buffer — not "corpus").

## User work queue (standing order, 04:26, in memory `project-work-queue-2026-07-05`)

1. Max coin-toss delivery without regression (current front: DAgger; then root-cause the Q-term failure with
   a discriminating test, or manifold optimization over reward weights per docs/plans/2026-07-04-reward-shape-optimization/).
2. Increasing k (controller ready; blockers: k-arm scene emission loop, k-vector contact metrics, reward terms
   reading left/right, actor channel count).
3. FANUC pick-and-place (plan compiled: docs/plans/2026-07-05-fanuc-pick-place-controller/; discovery map in
   memory `project-fanuc-pick-place-push-next`; env's `expert_action` + `DampedPoseIK` exist; success ONLY via
   divergence-guarded `LiftPlaceMetric`).
4. Kato CIP-LiNGAM PoC over coin-toss rollouts (memory `project-kato-lingam-cip-hymeko`).
5. Humanoid (humanoid.hymeko 13-DOF validated; BC warm-start → TD3+BC precedent).
6. Niitsuma RAPPORT (collaboration groundwork; k-6DOF arms + interaction hypergraphs).

## Deadline context

User deadline: arms delivering by RL + HyMeKo by 11:00 JST 2026-07-05. State at 06:05: BC-cloned learned delivery 0.52
(exists, renderable); no RL-refined improvement over the clone yet; DAgger is the highest-probability path to raise
the learned number before then. Working tree is UNCOMMITTED on `hymeko-neuro-migration` (user has not
requested commits; diff spans hymeko_rl/{experiments,train,control,tests}, data/robotics/*.hymeko, docs/plans,
reports).
