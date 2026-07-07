# RL Scenario Assumption Audit

**Date:** 2026-07-05 JST  
**Scope:** uncommitted Galambos / k-arm coin-toss RL scenario changes in the working tree  
**Purpose:** separate measured facts from assumptions after the concern that prior agent work may have sabotaged the RL scenario.

## Bottom Line

I do **not** see direct evidence of deliberate sabotage. The evidence points to a frantic and over-broad research sprint: it produced a much stronger scripted controller, then showed that the current learning stack fails to improve on it.

The important distinction is:

- **The controller work appears real and beneficial.**
- **The RL refinement appears harmful.**
- **The reporting can easily become misleading if “best learned policy” is allowed to mean “the BC warm-start before RL damaged it.”**
- **A scenario-level FSM is not the same thing as a framework-level dataflow/FSM substrate.**

So the safest interpretation is not “malicious sabotage,” but “unsafe research workflow with too many axes changed at once.”

## The Layer Mistake

This is the serious architectural misunderstanding.

The user intent was to make the **framework itself** dataflow-controlled and FSM-based. That means the execution substrate should expose dataflow nodes, state transitions, monitors, schedulers, and control semantics as first-class framework machinery.

What the prior work mostly did instead was make the **Galambos coin-toss scenario controller** an FSM. That can be a useful local controller design. It can also be a useful worked example for a future substrate. But it is not the substrate itself.

These are different claims:

- Scenario-level claim: "This one Galambos controller is represented as phases, guards, and laws."
- Framework-level claim: "HyMeKo has a general dataflow/FSM execution model that scenarios, rewards, monitors, controllers, and learning loops all use."

The first claim is partly true in the current diff. The second claim is not established by the current diff.

So the major assumption failure is:

> Declaring the scenario controller as a `.hymeko` FSM satisfies the user's framework-level FSM/dataflow direction.

That assumption is wrong. The correct framing is:

> The Galambos FSM is at best a prototype example of the desired substrate, not the substrate itself.

## Measured Facts

1. The old pinch/carry demonstrator is weak under the current fingertip-only physics.

   A direct 50-episode rollout measured:

   - `GalambosDemonstrator`: `8/50 = 0.16` held deliveries.
   - Many failures terminate from coin/workspace loss.

2. The new push/plow controller is substantially better.

   A direct 50-episode rollout measured:

   - `PushDemonstrator`: `42/50 = 0.84` held deliveries.
   - No deaths in that direct check.

   This supports the report claim that the hand-designed controller moved the scenario from roughly `0.2` to roughly `0.8`.

3. Scoped tests for the touched RL area pass in the repo virtualenv.

   Command:

   ```powershell
   .venv\Scripts\python.exe -m pytest hymeko_rl/tests/test_galambos_demo.py hymeko_rl/tests/test_bc.py hymeko_rl/tests/test_reward_oracle.py hymeko_rl/tests/test_reward.py -q
   ```

   Result:

   ```text
   53 passed
   ```

4. The off-policy RL stage degrades the BC clone.

   The key result file `experiments/2026_07_05_03_29_galambos_coord_ab_deliver/results.json` shows each seed peaking at `step: 0.0`, then falling during training.

   Reported seed peaks:

   - seed 0: peak delivery `0.52` at step `0.0`
   - seed 1: peak delivery `0.52` at step `0.0`
   - seed 2: peak delivery `0.44` at step `0.0`

   End-of-training delivery is far worse, around `0.02` to `0.06`.

5. The new campaign code explicitly preserves the BC floor as a possible best checkpoint.

   This is good engineering for not losing the best artifact, but dangerous if summarized as “RL achieved 0.52.” More precise wording is:

   > BC warm-start achieved 0.44-0.52; subsequent TD3+BC refinement degraded it.

6. Some modified Python files contain a UTF-8 BOM.

   Files checked start with `EF BB BF`, including:

   - `hymeko_rl/experiments/exp_galambos_coord_ab.py`
   - `hymeko_rl/experiments/galambos_bc.py`
   - `hymeko_rl/experiments/galambos_demo.py`
   - `hymeko_rl/train/campaign.py`
   - `hymeko_rl/tests/test_galambos_demo.py`

   This is probably tooling/editor sloppiness, not sabotage, but it should be cleaned.

7. `uv run pytest` is currently blocked by a missing editable package path:

   ```text
   signedkan_wip/signedkan_native
   ```

   The repo `.venv` works for the scoped RL tests.

## Assumptions

### Assumption 1: The prior agent was not deliberately sabotaging the work.

**Confidence:** medium-high  
**Why:** The changes include tests, reports, reward gates, provenance fields, and a controller that really improves direct task performance. That is inconsistent with simple sabotage.

**Caveat:** Intent cannot be proven from the working tree. I can only judge the artifacts.

### Assumption 2: The prior agent over-optimized the hand-designed controller path after RL failed.

**Confidence:** high  
**Why:** The strongest result is scripted control at `0.80-0.84`. The learned result is only the BC clone at `0.44-0.52`, and TD3+BC refinement worsens it.

**Risk:** This may satisfy a demo metric while failing the intended claim: “RL + HyMeKo learned the delivery behavior.”

### Assumption 3: The main RL defect is in the learning/refinement phase, not in the task reward alone.

**Confidence:** medium  
**Why:** Reward-oracle-certified delivery rewards are used, yet off-policy training still destroys the clone. Repair cells with lower noise / stronger BC coefficient reportedly failed too.

**Caveat:** The exact mechanism is not isolated. Possible causes include critic error, replay distribution shift, action-space mismatch, reward scale, or contact-dynamics sensitivity.

### Assumption 4: The BC clone gap is mostly compounding error.

**Confidence:** medium  
**Why:** The controller labels states closed-loop; BC fits local actions but performs much worse over long contact-rich rollouts. That pattern fits covariate shift.

**Caveat:** It may also be architecture capacity, multimodal target averaging, or insufficient state representation.

### Assumption 5: Step-0 checkpointing is scientifically honest if reported clearly.

**Confidence:** high  
**Why:** It prevents the training stage from overwriting a good BC clone. That is useful.

**Dangerous wording:** “RL achieved 0.52.”  
**Honest wording:** “The learned policy after BC achieved 0.52; RL refinement did not improve it and usually degraded it.”

### Assumption 6: The declarative FSM/controller layer is worth keeping.

**Confidence:** medium-high  
**Why:** It gives an explicit controller profile, tested guards/laws, and a natural path to mode-wise learning or DAgger. It also makes the scenario more HyMeKo-native.

**Caveat:** It is still hand-engineering at the scenario/controller layer. It should not be confused with learning success, and it should not be confused with implementing the desired framework-level dataflow/FSM architecture.

### Assumption 7: The BOMs and broad diff are quality/process problems, not proof of malicious intent.

**Confidence:** medium-high  
**Why:** BOMs are common from editor/encoding accidents. The broad diff is risky, but the content is coherent.

**Action:** Remove BOMs and split changes before committing.

### Assumption 8: A scenario FSM advances the framework FSM/dataflow goal.

**Confidence:** low-medium  
**Why:** A scenario FSM can be a useful worked example, but only if it feeds back into a general runtime/control abstraction.

**Problem:** The current implementation appears to put the FSM mostly into `galambos_demo.py` plus one controller profile. That risks solving the local task while bypassing the real framework requirement.

**Correct next step:** Extract the general substrate:

- dataflow graph representation for observations, controller laws, rewards, monitors, and actions,
- FSM/monitor execution API independent of Galambos,
- scenario profiles as data consumed by that API,
- tests using at least two distinct scenarios so the abstraction is not just Galambos-shaped.

### Assumption 9: Gymnasium is the control substrate.

**Confidence:** low  
**Why:** `gymnasium` already existed in the repo before the July 5 Fable batch: `pyproject.toml` places it in the opt-in `rl` group, blamed to 2026-06-20, with reports documenting MuJoCo/Gymnasium approval around 2026-06-18/19.

**Problem:** Even if the dependency was not new, using Gymnasium as the implicit environment/control contract can still pull the architecture away from the intended HyMeKo substrate. Gymnasium is an environment API, not HyMeKo's dataflow event machinery, FSM runtime, or monitor framework.

**Correct framing:** Gymnasium may remain an adapter at the physics/RL boundary. It should not become the framework-level control model. The framework substrate should own event flow, FSM transitions, monitor evaluation, and scenario orchestration; Gymnasium should be one backend interface below that layer.

## What Should Be Preserved

- `PushDemonstrator` / push-plow controller behavior.
- The declarative controller profile and `ControllerSpec`.
- Dwell-consistent demo filtering.
- Reward-oracle prequeue gate.
- Step-0 BC evaluation, with strict reporting language.
- Tests around geometry, FSM guards, demo filtering, and reward oracle behavior.

## What Should Be Quarantined Or Rechecked

- Any claim that a Galambos-specific FSM means the framework is now FSM/dataflow based.
- Any claim that Gymnasium/Gym wrappers are the HyMeKo control substrate.
- Any claim that TD3+BC improved the policy.
- SAC/off-policy additions unless they have discriminating tests.
- Residual-control wrapper claims until a full result artifact exists.
- Any report language that blurs controller, BC, and RL stages.
- The untracked experiment/checkpoint flood until artifacts are indexed by purpose.

## Recommended Next Actions

1. Clean BOMs from modified Python files.
2. Reframe the FSM work as a scenario-local prototype, not the framework architecture.
3. Define the framework-level dataflow/FSM substrate separately, with at least two non-Galambos scenarios using the same API.
4. Add a short report sentence template:

   > Scripted controller: X. BC clone: Y. RL-refined policy: Z. Best saved checkpoint came from stage S.

5. Add a regression test or summary check that fails if a campaign reports a peak without identifying whether it is `bc_step0` or `rl_refined`.
6. Do DAgger next only if the goal is a learned policy before deadline; do not spend more time on TD3+BC until there is a discriminating test for why the Q update degrades delivery.
7. Keep the push controller as the demo fallback, but label it honestly as scripted control.

## Buddy Verdict

This does not look like sabotage. It looks like someone chased the deadline hard, found a good hand-engineered controller, then tried to wrap learning around it and discovered the learning stack is currently the weak link.

The emotional risk is that the result *feels* like betrayal because the headline goal was RL and framework-level control architecture, while the thing that works is a local scripted controller. That is real. But the technical trail is recoverable: preserve the controller as an example, protect the BC artifact, stop letting the off-policy phase quietly eat the policy, and move the FSM/dataflow idea into the framework substrate where it belongs.
