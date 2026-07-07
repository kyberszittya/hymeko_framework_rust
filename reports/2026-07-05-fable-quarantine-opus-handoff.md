# Fable Quarantine / Opus Handoff

**Date:** 2026-07-05 JST  
**Audience:** Opus, Codex, or any future agent touching Galambos / k-arm coin-toss / framework-control work  
**Status:** binding local operating note for this thread of work

## Executive Rule

Treat the July 5 Fable/Claude burst as an **untrusted import batch**.

Do not use it as architecture truth. Use it only as a set of artifacts to classify and salvage.

This is not a claim that sabotage was proven. The safety issue is enough without proving intent: Fable made a large, fast intervention around a wrong interpretation of the user's architecture goal.

Additional observed failure signals:

- It introduced or propagated encoding damage: multiple modified Python files acquired UTF-8 BOMs, and several July 5 reports display mojibake in PowerShell/codepage contexts. Encoding hygiene was not treated as a first-class correctness issue.
- It did not start in the required Aiko register and appeared not to apply `CLAUDE.md` from turn one. For this project, ignoring the operating contract is not cosmetic; it predicts the same class of layer/architecture drift seen in the Galambos work.
- When RL failed to improve, it drifted toward "something is wrong with the scenario" instead of first auditing the learning implementation. That reversed the proper diagnostic order and encouraged scenario/controller redesign before the RL defect was isolated.

## The Misunderstood Requirement

The user wanted:

> A framework controlled through dataflow event machinery, FSM runtime, and a monitoring framework.

That means HyMeKo itself owns:

- event flow,
- dataflow graph semantics,
- FSM state transitions,
- monitor evaluation,
- controller/reward/observation/action orchestration,
- scenario-independent execution machinery.

The Galambos scenario is only one scenario plugged into this framework.

What Fable mostly did instead:

> Made the Galambos scenario controller into a local FSM.

That can be a useful worked example. It is not the framework substrate.

## Stage Ledger

Use these labels. Do not invent softer wording.

| Label | Meaning | Current Galambos status |
|---|---|---|
| `scripted_controller` | Hand-designed push/plow controller | useful, about 0.80-0.84 delivery |
| `bc_clone` | Behaviour clone of scripted controller | best current learned artifact, about 0.44-0.52 |
| `rl_refined` | TD3+BC/SAC/off-policy continuation after BC | measured worse than BC |
| `framework_substrate` | General HyMeKo dataflow event + FSM + monitor runtime | not implemented by the Galambos FSM |

Forbidden shorthand:

- "RL achieved 0.52."
- "The framework is now FSM/dataflow based."
- "The Galambos FSM satisfies the framework architecture."
- "Gymnasium is the control substrate."

Correct shorthand:

- "Scripted controller delivers around 0.8."
- "BC clone reaches around 0.5."
- "RL refinement currently degrades the clone."
- "The Galambos FSM is a scenario-local prototype."
- "Gymnasium is only a physics/RL adapter below the HyMeKo substrate."

## Codex Intervention: Trainer Bug Fixed, Not Yet a Win Claim

2026-07-05 07:45 JST: Codex patched one implementation defect in `hymeko_rl/train/ddpg.py`.

In the non-heterogeneous TD3/DDPG path, target critics were Polyak-updated only inside the actor-update branch.
During `critic_warmup`, this meant the critic could train against stale target critics while the actor was
deliberately frozen to protect a BC clone. That is an implementation defect on the exact path where Galambos
RL refinement degraded the clone.

Patch:

- non-hetero critic targets now Polyak-track after every critic optimizer step,
- actor target Polyak remains gated by actor updates,
- hetero critic targets keep their existing per-critic cadence,
- regressions added:
  - `test_critic_targets_update_during_actor_warmup`: target critics track even while actor warmup blocks actor updates,
  - `test_critic_warmup_freezes_actor_but_updates_critic`: toy invariant that warmup preserves the cloned actor while critic parameters learn.

Verification:

- `.venv\Scripts\python.exe -m pytest hymeko_rl\tests\test_ddpg.py hymeko_rl\tests\test_critic_probe.py -q`
- result: `28 passed`
- after adding the toy warmup invariant: `.venv\Scripts\python.exe -m pytest hymeko_rl\tests\test_ddpg.py -q`
- result: `10 passed`

Do **not** convert this into "RL is fixed" or "RL beats BC." It only removes one concrete trainer bug. A bounded
post-fix rerun/probe is still required before any performance claim.

## Cross-Scenario Health Check

2026-07-05 08:00 JST: Codex ran a focused cross-scenario construction/behavior suite:

```powershell
.venv\Scripts\python.exe -m pytest `
  hymeko_rl\tests\test_scenario_sanity.py `
  hymeko_rl\tests\test_tasks.py `
  hymeko_rl\tests\test_galambos_demo.py `
  hymeko_rl\tests\test_galambos_task_graph.py `
  hymeko_rl\tests\test_fanuc_pick.py `
  hymeko_rl\tests\test_pick_place_env.py `
  hymeko_rl\tests\test_pick_place_task.py `
  hymeko_rl\tests\test_quadruped_env.py `
  hymeko_rl\tests\test_quadruped_standing.py `
  hymeko_rl\tests\test_humanoid_model.py `
  hymeko_rl\tests\test_planar_grasp_env.py `
  hymeko_rl\tests\test_inverted_pendulum_env.py `
  hymeko_rl\tests\test_reach_bc.py `
  hymeko_rl\tests\test_reach_arch_compare.py -q
```

Result: `139 passed in 46.47s`.

One plumbing bug was found and fixed before the clean run: `hymeko_rl/experiments/reach_arch_compare.py`
resolved `_REPO` to `hymeko_rl/` and therefore looked for `hymeko_rl/data/robotics/reach_arm.hymeko`. The
canonical file is under repo-root `data/robotics/`; `_REPO` now uses `Path(__file__).resolve().parents[2]`.

This verifies scenario construction/basic behavior across coin-toss/Galambos, FANUC pick-place, quadruped,
humanoid model, inverted pendulum, and reach-arm comparison. It does **not** certify long RL training success.

## Plain-Python Reward Isolation

2026-07-05 08:21 JST: Codex added `hymeko_rl/experiments/galambos_plain_reward.py` and
`hymeko_rl/tests/test_galambos_plain_reward.py`.

Purpose: test whether Galambos TD3+BC collapse is caused specifically by `.hymeko` reward parsing/terms.

Isolation properties:

- `robot=None`: uses the hand-authored planar MJCF path, not a robot emitted from `.hymeko`,
- `reward_spec=PlainPythonDeliverReward(...)`: reward is a Python object with `evaluate(env, dist, action)`,
- `env=DEFAULT_ENV`: environment geometry comes from code defaults, not `EnvSpec.from_hymeko`,
- same MuJoCo physics class, same push-controller demo collection, same BC and TD3+BC trainer.

Verification:

- `.venv\Scripts\python.exe -m pytest hymeko_rl\tests\test_galambos_plain_reward.py -q`
- result: `2 passed`
- `.venv\Scripts\python.exe -m pytest hymeko_rl\tests\test_galambos_plain_reward.py hymeko_rl\tests\test_ddpg.py -q`
- result: `12 passed`

Bounded smoke run:

- `experiments/2026_07_05_14_07_galambos_plain_dense_td3bc_5000_s0`
- `bc_step0`: delivery `0.30`, both_contact `0.0459`
- `rl_refined @2500`: delivery `0.30`, both_contact `0.0459`
- `rl_refined @5000`: delivery `0.00`, both_contact `0.00`

Interpretation: `.hymeko` reward parsing/terms are **not sufficient** to explain the collapse. A no-HyMeKo
plain dense reward still loses contact by 5k. Keep investigating actor/Q-gradient/contact-manifold dynamics
before blaming reward profiles.

## Cross-Scenario TD3+BC Collapse Check

2026-07-05 08:30 JST: Codex ran two small non-Galambos checks to see whether BC→TD3+BC degradation is universal.

| Scenario | Budget | BC floor | Post-refine | Interpretation |
|---|---:|---:|---:|---|
| Reach arm (`ArmReachEnv`, MLP, final distance lower is better) | 12 demos, 40 BC epochs, 2k TD3+BC steps | `0.3717 m` | `0.3408 m` | TD3+BC improved the BC clone |
| FANUC pick-place (`fanuc_pick_env`, MLP, place success) | 6 demos, 20 BC epochs, 1k TD3 steps | `0.0` | `0.0` | inconclusive tiny-budget cell; no success floor to degrade |

Reach-arm command was an inline Python check using existing helpers:

- `collect_demos`, `behaviour_clone`, `eval_reach`,
- `build_offpolicy(..., n_critics=2)`,
- `td3_bc_config(total_steps=2000, critic_warmup=200, noise_scale=0.05)`.

Result: the Galambos/contact collapse is **not** a universal TD3+BC failure. TD3+BC can improve a BC clone in a
simpler non-contact reach scenario. The active suspect is therefore the interaction between deterministic
off-policy actor updates and the Galambos contact manifold/replay distribution, not "TD3 never improves BC."

## Residual Probe And Eval-Harness Leak

2026-07-05 14:27 JST: Codex verified the latest Claude handoff as evidence, not as trusted testimony.

Confirmed code-level issue:

- Galambos campaign measurement created fresh MuJoCo envs for delivery/contact eval and did not close them.
- The same risk existed in the plain no-HyMeKo diagnostic measure path.
- `PlanarGraspEnv`, `ResidualControllerEnv`, and `VectorizedEnv` now expose/forward `close()`.
- The Galambos measure paths now close their temporary eval envs.
- Focused verification: `.venv\Scripts\python.exe -m pytest hymeko_rl\tests\test_residual.py hymeko_rl\tests\test_campaign.py hymeko_rl\tests\test_ddpg.py hymeko_rl\tests\test_galambos_plain_reward.py -q`
- Result: `25 passed, 14 warnings`.

New reproducible diagnostic:

- `hymeko_rl/experiments/galambos_residual_probe.py`
- Artifact: `experiments/2026_07_05_14_25_galambos_residual_probe_6000_s0`
- Command: `.venv\Scripts\python.exe -m hymeko_rl.experiments.galambos_residual_probe --steps 6000 --n-eval 25 --n-envs 4 --hidden 32`

Results:

| Cell | Pre delivery | Post delivery | Pre both_contact | Post both_contact | Read |
|---|---:|---:|---:|---:|---|
| frozen zero residual | `0.88` | `0.88` | `0.1536` | `0.1536` | loop/wrapper preserves scripted floor |
| tiny trained residual | `0.88` | `0.64` | `0.1536` | `0.0932` | actor update still degrades the contact behavior |

Interpretation:

- Claude's claim that the frozen residual loop itself does not corrupt the base behavior is supported.
- The eval-harness leak explanation is plausible for the exit-127 crashes, and the repository now closes the
  relevant temporary envs in the checked paths.
- The open scientific question is no longer "does a tiny trained correction hold or collapse?" for this first
  6k cell: it **degrades** from `0.88` to `0.64`, but does not collapse to zero.
- Do **not** continue by adding a new RL algorithm or raw-action residual ablation.

## Phase-Parameter Controller Gate

2026-07-06 JST: Codex implemented the ChatGPT hint as a concrete code interface, not another ablation.

New/updated code:

- `PushControllerParams`: the only learning-facing interface; exactly five bounded high-level parameters:
  `contact_offset`, `push_gain`, `direction_correction`, `brake_threshold`, `release_threshold`.
- `PhasePushController`: explicit five-phase controller: `APPROACH`, `CONTACT`, `PUSH`, `BRAKE`, `DONE`.
- `DeliveryFloorCriterion`: accepts learned controllers only if delivery stays at or above scripted baseline minus
  tolerance; with `0.84 - 0.03`, the floor is `0.81`.
- Safety arbiter: in `PUSH`/`BRAKE`, candidate parameter targets fall back to safe scripted targets when the
  candidate is not behind the coin, the coin is receding from the target, or braking risks overshoot.

Policy:

- The neural component may output controller parameters only.
- It must not output raw motor/joint actions for the Galambos refinement path.
- It must not bypass the phase controller or safety arbiter.
- Any training result below `0.81` delivery is rejected, not analyzed as an improvement.

Verification:

- `.venv\Scripts\python.exe -m pytest hymeko_rl\tests\test_arbiter_push.py hymeko_rl\tests\test_galambos_demo.py hymeko_rl\tests\test_affordance.py -q`
- Result: `29 passed`; pytest warned that `.pytest_cache` could not be written under the current sandbox.

## Artifact Classification

Before building on any July 5 Fable-era artifact, classify it as one of:

- `keep`: measured useful, tested, and aligned with the real architecture direction.
- `suspect`: useful-looking but wrong layer, ambiguous reporting, or not independently verified.
- `generated`: experiment/checkpoint/report output; preserve if referenced, but do not treat as design.
- `unknown`: do not build on it yet.

Initial classification:

| Artifact | Classification | Reason |
|---|---|---|
| `PushDemonstrator` / push-plow behavior | `keep` as scenario example | Direct rollout and tests show useful scripted control |
| Galambos FSM profile | `keep` as scenario-local prototype | Useful example, not framework substrate |
| `ControllerSpec` | `suspect/keep candidate` | May become a framework piece, but currently shaped by Galambos |
| TD3+BC July 5 results | `keep` as negative evidence | Shows refinement degrades BC |
| SAC additions | `suspect` | Not yet a clean positive result |
| residual wrapper | `suspect` | Conceptually plausible, but not the framework substrate and not fully result-proven |
| reports claiming "learned policy 0.52" | `suspect` unless read through stage ledger | The number is BC clone, not RL-refined success |
| Gymnasium wrappers | `adapter only` | Useful boundary API, not HyMeKo control model |
| Fable-modified files with encoding anomalies | `suspect until checked` | BOMs/mojibake indicate contract and tooling hygiene were not respected |

## Correct Next Architecture Direction

Do not start by adding another Galambos-specific controller layer.

Do not blame the scenario first when RL fails. The diagnostic order is:

1. Verify the implementation:
   - action scaling,
   - observation mapping,
   - reward evaluation timing,
   - termination/truncation handling,
   - replay data distribution,
   - critic target construction,
   - BC anchor application,
   - checkpoint/eval stage labeling.
2. Verify the measured teacher / BC floor.
3. Run a discriminating implementation test.
4. Only then consider whether the scenario itself is ill-posed or needs redesign.

### Concrete July-5 RL Failure Hypothesis To Isolate

The decisive code site is `hymeko_rl/train/ddpg.py::_actor_loss`:

```python
al = -critics[0](s, actor(s)).mean()
if bo is not None and ba is not None:
    al = al + cfg.bc_coef * bc_scale * F.mse_loss(actor(bo), ba)
```

This optimizes Q on replay states `s`, while the BC anchor is evaluated on demonstration states `bo`. The observed failure signature (delivery and `both_contact` both decaying toward zero) is consistent with:

> The Q-gradient pulls the actor off the contact manifold while the BC anchor only constrains the demo manifold.

This is still a hypothesis until isolated. The nearby critic comment already mentions actor-chasing-inflated-Q, but that was never separated from alternatives.

Discriminating tests should compare, at minimum:

- **BC-only frozen actor:** no Q actor update; verifies the floor.
- **BC-anchor-only actor update:** actor update with Q term disabled; should preserve/improve imitation if anchor path is correct.
- **Q-only actor update from the BC clone:** isolates whether Q immediately moves actions away from demo/contact states.
- **Same-state anchor test:** apply BC penalty on replay states relabeled by the scripted controller, not only original demo states; distinguishes "anchor manifold mismatch" from generic critic failure.
- **Actor drift probes:** log `||mu(s_demo)-a_demo||`, `||mu(s_replay)-teacher(s_replay)||`, Q value, delivery, and `both_contact` at every eval.

If Q-only or mixed Q+BC reduces contact while BC-only preserves it, the implementation-level actor objective is implicated. Scenario redesign is not justified before this test.

Start by defining the framework substrate:

1. Dataflow event model:
   - typed event records,
   - event sources/sinks,
   - graph edges between observation, monitor, controller, reward, and action nodes.

2. FSM runtime:
   - scenario-independent state machine representation,
   - transition guards as monitor predicates,
   - deterministic step semantics,
   - trace/provenance output.

3. Monitor framework:
   - temporal predicates,
   - robustness/margin values,
   - sliding-window observation,
   - shared use by success metrics, guards, rewards, and diagnostics.

4. Scenario adapter:
   - Galambos plugs into the substrate,
   - at least one second scenario plugs in too,
   - Gymnasium/MuJoCo sits below the adapter boundary only.

Acceptance condition:

> The same framework-level dataflow/FSM/monitor machinery runs Galambos and at least one non-Galambos scenario.

If only Galambos runs, the architecture is not proven.

## Current Useful Galambos State

Preserve these facts:

- Old pinch/carry controller is weak under fingertip-only physics.
- Push/plow scripted controller is useful and should remain as a reference scenario behavior.
- BC clone is the best current learned artifact.
- Off-policy refinement has not improved the clone in measured July 5 runs.
- Campaign outputs now label curve stages as `bc_step0`, `rl_refined`, or `rl_from_scratch`.

Relevant reports:

- `reports/2026-07-05-galambos-stage-ledger.md`
- `reports/2026-07-05-rl-scenario-assumption-audit.md`
- `reports/2026-07-05-session-handoff-coin-toss-rl.md`
- `reports/2026-07-05-postmortem-coin-toss-rl-session.md`

## Operational Guidance For Opus

1. Read this report first.
2. Read `reports/2026-07-05-galambos-stage-ledger.md`.
3. Do not accept Fable's July 5 architecture framing.
4. Do not run new RL campaigns until the stage ledger and reward certificate rules are satisfied.
5. If asked to continue architecture work, build the general substrate first and use Galambos only as one test scenario.
6. If asked to continue Galambos performance work, label every output by stage and protect the push controller as scripted reference behavior.

## Verification From Cleanup Pass

Cleanup performed after the audit:

- Removed UTF-8 BOMs from modified RL Python files.
- Added stage labels to `Campaign` curve outputs.
- Added regression test for `bc_step0` vs `rl_refined`.
- Patched July 5 reports with cleanup notes.
- Added `reports/2026-07-05-galambos-stage-ledger.md`.

Scoped verification:

```text
59 passed, 14 warnings
BOM files: []
```

The warnings are Torch JIT deprecation warnings from the local environment, not failures.
