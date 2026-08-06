# Runtime-Monitor Spec → Implementation Audit

**Date:** 2026-07-07
**Author:** Aiko (agent)
**Type:** read-only mapping audit. No code changed, no training run.

Maps the specification bundle in `docs/plans/2026-07-04-hymeko-code-agent/` onto the
existing monitor implementation in `hymeko_rl/eval/task_monitor/` to determine what
exists, what is partial, and what is missing.

**Files inspected**

- Spec: `docs/plans/2026-07-04-hymeko-code-agent/{runtime_success_monitor,
  hymeko_to_cip,metaworld_task_descriptions,dagger_for_hymeko_tasks,
  hymeko_code_agent}.md`, `README.md`, `plan.md`.
- Implementation: `hymeko_rl/eval/task_monitor/{__init__,contract,context,
  submonitors,root,consistency,pipeline}.py`, `hymeko_rl/tests/test_task_monitor.py`.
- Adjacent (referenced, not part of the monitor package):
  `hymeko_rl/eval/pick_clearance.py`, `hymeko_rl/eval/tasks.py`,
  `hymeko_rl/train/ddpg.py`, `hymeko_rl/train/dagger.py`.

---

## A. Existing implementation map

Reward-independence is already the design invariant: `record_trajectory`
(`context.py:17`) records **positions / velocities / contacts / distances only — no
reward, no policy internals**. Every submonitor reads that top-down view.

| File · symbol | What it checks | Spec category |
| --- | --- | --- |
| `contract.py` · `MonitorContract(.from_env)` | zone geometry, dwell/success rule, near-coin, body/progress tolerances read from the env's HyMeKo contract | contract generation (feeds all) |
| `contract.py` · `TensorContract(.from_env)` | obs / privileged-z / node-feature dims + privileged field order | **TensorContract** |
| `context.py` · `record_trajectory` | rolls one episode; records position-based tensor, no reward | reward-independent substrate |
| `context.py` · `MonitorContext.build` | SoA of derived per-step arrays: `toward`, `ft_prog_step`, `body_prog_step`, `dwell`, engagement timing | substrate (feeds all) |
| `context.py` · `contiguous_runs` | half-open `[start,end)` evidence slices | verdict evidence |
| `submonitors.py` · `TrajectoryMonitor` (ABC) | Strategy base for one aspect | — |
| `submonitors.py` · `GeometryMonitor` | coin–target distance, tip–coin distance, zone membership (base facts) | **Success** (reported base layer) |
| `submonitors.py` · `ApproachMonitor` | fingertips approach before displacement; coin not pushed away; approach precedes displacement | **Progress** (+ Scenario order) |
| `submonitors.py` · `ContactMonitor` | left/right contact, both-fingertip engagement, contact duration/timing | **Progress** (contact established) (+ Scenario) |
| `submonitors.py` · `ProgressMonitor` | distance decreasing; fingertip- vs body-attributed progress | **Progress** |
| `submonitors.py` · `DeliveryMonitor` | enters zone, holds k steps (stable), final distance | **Success** |
| `submonitors.py` · `AntiExploitMonitor` | body-driven / body-assisted / no-engagement delivery, arm-body shove | **AntiExploit / Violation** |
| `submonitors.py` · `SubVerdict` | per-aspect: `passed`, signed `score`, `violations`, `time_indices`, `slices`, `as_dict` | verdict contract |
| `submonitors.py` · `default_submonitors` | canonical ordered hierarchy | Scenario (order) |
| `root.py` · `TaskMonitor` (Composite, `from_env`, `evaluate`) | composes gating submonitors → `monitor_pass` conjunction + mean `monitor_score`; gating order `approach→contact→progress→delivery→anti_exploit` | **Scenario** + Success roll-up |
| `root.py` · `TaskMonitor.evaluate_policy` | aggregates over n episodes: `monitor_pass_rate`, per-aspect means, violation histogram, `top_violation` | evaluation harness |
| `root.py` · `TaskVerdict(.as_dict)` | task-level verdict + dict export | verdict contract + JSON (partial) |
| `consistency.py` · `RewardConsistencyMonitor` | `check_reward_alignment` (reward-vs-monitor rank inversions), `check_critic_alignment` (Q-vs-monitor inversions) | **Learning / RewardConsistency** |
| `consistency.py` · `TensorContractMonitor` | field-order hash, tensor signature, `check_env`, `verify_stages` | **TensorContract** |
| `pipeline.py` · `PipelineSchemaLedger` (+ `TransitionSchema`, `TransitionField`, `flat_dim`) | live per-stage schema guard across rollout→serialize→load→critic→eval; wired into `train_offpolicy` via `ddpg.py:36` | **TensorContract** (live sibling) |
| `__init__.py` · `monitor_policy` | back-compat aggregator over n episodes | evaluation harness |
| `tests/test_task_monitor.py` | unit coverage: empty traj, pass-on-good-delivery, fail-on-exploit, fail-on-no-delivery, reward/critic alignment | test |

**Category coverage summary:** Success ✔, Progress ✔ (minus stagnation), Violation ◐
(AntiExploit only), Scenario ◐ (order is implicit in gating, no explicit
phase-ordering check), Learning/RewardConsistency ✔, TensorContract ✔, AntiExploit ✔.

**Scope note:** every submonitor's contract is **coin-delivery-specific**
(`MonitorContract.from_env` reads coin/zone/fingertip fields). The task-agnostic,
HyMeKo-generated contract the spec calls for is not yet present.

---

## B. Spec coverage table

| Spec file | Already implemented | Partially implemented | Missing | Notes |
| --- | --- | --- | --- | --- |
| `runtime_success_monitor.md` | Success, Progress (core), Learning/RewardConsistency, TensorContract, AntiExploit; reward-independent substrate; verdict + evidence | Violation (AntiExploit only); Scenario (order implicit, not explicit); verdict `outcome_class` field; task-agnostic contract generation | `StagnationMonitor`; clearance/workspace/action-bound/unsafe-transition submonitors | Concept is realized for the coin task; generalisation to arbitrary HyMeKo tasks is the outstanding work |
| `hymeko_to_cip.md` | — | raw disagreement is computed (`RewardConsistencyMonitor`) | **CIP variable export bridge**; the 8 named CIP variables | Nothing converts a `TaskVerdict` into CIP variables; no CIP consumer exists |
| `metaworld_task_descriptions.md` | — | — | coffee-push monitor; dial-turn monitor; MetaWorld `TaskSpec` entries | `tasks.py` has no coffee/dial task; no angular-progress/overshoot submonitor exists |
| `dagger_for_hymeko_tasks.md` | DAgger dispatch (`train/dagger.py`); `evaluate_policy` monitor aggregation exists | — | **monitor-success as the DAgger selection/eval metric** | `dagger.py` does not import the monitor; selection is on reward/val, not `monitor_pass_rate` |
| `hymeko_code_agent.md` | — | — | code-agent monitor loop (step 3 judge) | Conceptual/plan only; `plan.md` build not started |

Legend: ✔ done · ◐ partial · — none.

---

## C. Gap list

### 1. `StagnationMonitor` — MISSING
- Should produce `stagnation_duration`; detect no-progress windows; be
  reward-independent.
- **Feasibility:** cheap. `MonitorContext` already holds the per-step `toward` array
  (`context.py:74`) — a no-progress window is `toward <= progress_eps` over a sliding
  window. No new observation is needed; add a submonitor + the `stagnation` outcome
  class. Lowest-cost gap.

### 2. Clearance / workspace / action-bound / unsafe-transition — MISSING (as monitor submonitors)
- `clearance_min` — the *measurement* primitive exists in `pick_clearance.py`
  (`min_clearance_min`, `pick_clearance.py:217`) but as a separate eval, **not** a
  reward-independent trajectory submonitor. Reuse it; do not re-derive.
- `forbidden_contact_count` — `AntiExploitMonitor` + the trajectory's
  `arm_body_contact` flag detect illegal body contact, but there is no explicit
  count variable.
- `phase_transition_failure` — privileged phase fields exist (`phase0/1/2` in
  `TensorContract.privileged_fields`) but there is no phase-ordering violation check.
- `workspace_violation` — absent.
- `action_saturation` — absent, **and blocked**: `record_trajectory` records
  positions/contacts but **not the action** (`context.py:27-35`). This variable
  requires adding the action to the monitor trajectory tensor first — a substrate
  change, flagged as a precondition.

### 3. MetaWorld monitor templates — MISSING (both)
- No coffee-push monitor, no dial-turn monitor. dial-turn additionally needs an
  angular-progress + overshoot submonitor that does not exist in the coin hierarchy.

### 4. CIP-export bridge — MISSING
- No conversion of monitor outputs into the 8 CIP variables; no CIP consumer. The raw
  reward/critic disagreement is computed by `RewardConsistencyMonitor` and would feed
  the bridge, but the bridge itself is absent.

### 5. DAgger evaluation bridge — MISSING / partial
- `evaluate_policy` already yields `monitor_pass_rate` and per-aspect means, so the
  metric is *computable*; it is simply **not wired** as the DAgger selection/early-stop
  metric. `train/dagger.py` does not import `task_monitor`.

---

## D. Minimal build order (proposed — NOT to be implemented from this report)

Smallest sensible sequence; each step is independently testable and reward-independent.

1. **`StagnationMonitor`** — reuse `MonitorContext.toward`; emit `stagnation_duration`
   and the `stagnation` outcome class. Cheapest; unblocks a CIP variable.
2. **Generic `MonitorRecord` / JSON export** — formalise per-episode persistence.
   *Partial today:* `TaskVerdict.as_dict` / `SubVerdict.as_dict` and the
   `evaluate_policy` aggregate dict exist (the scratchpad harness already writes
   `reports/figures/task_monitor/monitor_eval.json`); this step promotes that to a
   declared record schema the downstream bridges read.
3. **CIP variable export** — map a `MonitorRecord` → the 8 CIP variables
   (`success_monitor_pass`, `progress_slope`, `stagnation_duration`,
   `forbidden_contact_count`, `clearance_min`, `phase_transition_failure`,
   `reward_progress_disagreement`, `expert_vs_policy_monitor_gap`).
4. **MetaWorld coffee-push monitor stub** — task-agnostic contract + a push/displacement
   submonitor; register a `TaskSpec` entry. First proof the generation generalises off
   the coin task.
5. **DAgger monitor-success evaluation hook** — wire `evaluate_policy`'s
   `monitor_pass_rate` as the DAgger selection/early-stop metric (reward/BC-loss stay
   reported, not deciding).

Deferred (out of the minimal order, flagged): `action_saturation` needs the action
added to the trajectory tensor first; clearance/workspace/unsafe-transition submonitors
follow once the CIP bridge proves the variable path end-to-end.

---

## E. Safety rule (explicit)

- **Reward can guide learning.**
- **Runtime monitors judge scenario evolution.**
- **CIP prioritizes monitor/reward disagreement.**
- **Training should not launch unless the monitor contract exists.**

---

## Constraints honoured

No training run; no monitor code modified; FANUC v2 untouched; coin-collab v2b
untouched; `CORE.YAML` untouched. Report only.

---

## Update — 2026-07-07 18:20: build-order item #1 (`StagnationMonitor`) implemented

Gap C.1 is now closed. `StagnationMonitor` shipped as a reward-independent,
composable submonitor.

**Changed files**
- `hymeko_rl/eval/task_monitor/submonitors.py` — `+StagnationVerdict(SubVerdict)`
  (adds `stagnation_duration`, `stagnated`, serialising `as_dict`);
  `+StagnationMonitor(TrajectoryMonitor)`; a clarifying note on `default_submonitors`.
- `hymeko_rl/eval/task_monitor/__init__.py` — exported `StagnationMonitor`,
  `StagnationVerdict`.
- `hymeko_rl/tests/test_task_monitor.py` — +6 tests (params, improving, flat,
  insufficient-steps, oscillatory, root-composition).

**Exact semantics**
- Signal: **net** window displacement `dist[t-window] - dist[t]` over the
  reward-independent distance-to-target series (`MonitorContext.dist`) — *not* a
  one-sided `toward` sum, which cannot see oscillatory no-net-progress.
- Decision: a step `t ≥ max(window, min_steps)` is stagnating iff its net window
  progress `< eps`; `stagnation_duration` = longest contiguous stagnating run;
  `stagnated = duration > 0`; a rollout shorter than `max(window, min_steps)` never
  flags (insufficient evidence, not a pass claim).
- Defaults `window=20, eps=0.005, min_steps=0`; `score` `>0` when progressing,
  `<0` when stagnating (`tanh`-squashed, sibling convention). Invalid params raise
  `ValueError`.

**Not in `default_submonitors()`** — kept opt-in/composable, so the coin-delivery
task verdict is byte-unchanged.

**Test results** — `pytest -p no:randomly hymeko_rl/tests/test_task_monitor.py`:
**28 passed** (22 pre-existing unchanged → existing verdicts did not change; 6 new).
ruff clean; mypy clean on changed code; radon `evaluate` CC=4 (A).

**Existing verdicts changed?** No — proven by the 22 unchanged pre-existing tests
plus the root-composition test (`monitor_pass` / `monitor_score` identical with the
monitor composed in).

**`stagnation_duration` as a future CIP variable** — yes, now available.
`StagnationVerdict.as_dict` serialises `stagnation_duration` (and `stagnated`), and
it survives into `TaskVerdict.as_dict` when composed, so the CIP-export bridge
(gap C.4, still not implemented) can read it directly. Build-order item #1 done;
items #2–#5 unchanged.

---

## Update — 2026-07-07 18:55: build-order item #2 (CIP-export bridge) implemented

Gap C.4 is now closed. A generic, reward-independent bridge converts a monitor
verdict into named CIP scalar variables. CIP itself is **not** implemented.

**Changed files**
- `hymeko_rl/eval/task_monitor/cip_export.py` — new module:
  `export_cip_variables(verdict) -> CipExport`; `CipExport(variables, missing,
  source_keys)`; `CIP_VARIABLES`, `CIP_DEFAULTS`.
- `hymeko_rl/eval/task_monitor/__init__.py` — exported the four symbols.
- `hymeko_rl/tests/test_cip_export.py` — new file, 9 focused tests.

**Exported variable names** (the initial factored set, in order):
`success_monitor_pass`, `progress_score`, `stagnation_duration`, `stagnated`,
`forbidden_contact_count`, `clearance_min`, `phase_transition_failure`,
`reward_progress_disagreement`.

**Sourcing** — `success_monitor_pass ← monitor_pass` (task-level conjunctive
success); `progress_score ← TaskVerdict.progress_score`; `stagnation_duration` /
`stagnated ← sub_verdicts["stagnation"]` (present iff `StagnationMonitor` composed);
`reward_progress_disagreement ← 1 − concordance` of an attached
`RewardConsistencyMonitor` output (or a pre-computed scalar).

**Missing/default policy** — every variable always appears in `variables`; a value
that cannot be sourced gets a deterministic default (`CIP_DEFAULTS`: `0` / `0.0` /
`False`) and its name is listed in `missing`. Nothing is fabricated. The three
violation variables (`forbidden_contact_count`, `clearance_min`,
`phase_transition_failure`) are read opportunistically (top-level or any sub-verdict
field of that name) so the bridge will pick them up unchanged when the violation
submonitors (gap C.2) land — until then they are default+missing. Input is not
mutated. Robust to both a live `TaskVerdict` object and its `as_dict()` mapping via
one uniform accessor.

**Test results** — `pytest -p no:randomly test_cip_export.py test_task_monitor.py`:
**37 passed** (9 new + 28 monitor unchanged). ruff clean; mypy clean on the new
module; radon max CC = 6 (B, `_reward_disagreement`), rest A.

**Ready for CIP/DirectLiNGAM logging?** Yes for the sourced variables. A consumer
calls `export_cip_variables(verdict)` per rollout, drops the names in `.missing`
(and, for LiNGAM, keeps only continuous columns — `progress_score`,
`stagnation_duration`, `reward_progress_disagreement` — stratifying on the booleans),
and logs `.variables`. The violation columns remain reserved (default+missing) until
gap C.2. Build-order items #1–#2 done; #3–#5 unchanged.

---

## Update — 2026-07-07 19:35: build-order item #3 (MetaWorld coffee-push monitor) implemented

Gap C.3 (coffee-push half) is now closed as a **template stub** — the generalisation
proof that the monitor framework is no longer coin-delivery-only. MetaWorld is not
installed or run; the template is exercised on synthetic position trajectories.

**Changed files**
- `hymeko_rl/eval/task_monitor/metaworld.py` — new module: `CoffeePushMonitor`
  (Composite root), `CoffeePushProgressMonitor` / `CoffeePushSuccessMonitor`
  (Strategy submonitors), `CoffeePushSubmonitor` (ABC), `CoffeePushContext`,
  `CoffeePushVerdict`.
- `hymeko_rl/eval/task_monitor/context.py` — new `SupportsDistanceSeries` Protocol
  (`n` + `dist`) so a distance-only monitor is task-agnostic.
- `hymeko_rl/eval/task_monitor/submonitors.py` — `StagnationMonitor.evaluate` widened
  from `MonitorContext` to `SupportsDistanceSeries` (additive; `MonitorContext` still
  satisfies it — all 28 coin tests unchanged).
- `hymeko_rl/eval/task_monitor/__init__.py` — exported the coffee-push symbols +
  `SupportsDistanceSeries`.
- `hymeko_rl/tests/test_metaworld_monitors.py` — new file, 10 tests.

**Monitor API** — `CoffeePushMonitor(success_radius=0.05, min_progress=0.01,
stagnation: StagnationMonitor | None = None).evaluate(traj) -> CoffeePushVerdict`.
Gates on progress ∧ success; a composed `StagnationMonitor` is diagnostic (reported,
not gating), mirroring the coin design.

**Trajectory fields** (plain `list[dict]`, no MetaWorld types): `object_xy` (req),
`target_xy` (req; usually constant), `gripper_xy` (opt), `contact` (opt).

**Success/progress semantics** — distance `= ‖object_xy − target_xy‖`.
`object_moved_toward_target` iff net `delta = d0 − dfin > min_progress` (moving away
→ `object_moved_away_from_target`; flat → `object_did_not_move_toward_target`).
`target_reached` iff final distance `dfin ≤ success_radius`. `monitor_pass =
object_moved_toward_target ∧ target_reached` (progress-without-success is the partial
case: progress passes, success fails). Verdict exposes
`object_target_distance_{initial,final,delta}`, `progress_score`, `monitor_pass`,
`monitor_score`, `violation_reason`, `sub_verdicts` (+ `stagnation_duration` when
composed).

**Test results** — `pytest -p no:randomly` over the three monitor test files:
**47 passed** (10 new + 9 CIP + 28 coin, all unchanged). ruff clean; mypy clean on
changed code (the Protocol override typechecks — no Liskov error); radon max CC = 8 (B).

**Does this prove the framework is no longer coin-delivery-only?** Yes.
`CoffeePushMonitor` judges a task with no coin/fingertip/zone concepts using the same
`SubVerdict` / Strategy-submonitor / Composite-root style, its verdict is read
unchanged by `export_cip_variables`, and `StagnationMonitor` (item #1, unmodified)
evaluates a `CoffeePushContext` directly via `SupportsDistanceSeries`
(`test_stagnation_monitor_reused_on_coffee_context`). Build-order items #1–#3 done;
#4 (`dial-turn`) and #5 (DAgger monitor-success hook) remain and were not touched.

---

## Update — 2026-07-07 19:55: build-order item #4 (MetaWorld dial-turn monitor) implemented

Gap C.3 (dial-turn half) is now closed — a second MetaWorld template, this one with
**angular** progress semantics, proving the framework covers both distance-based and
rotational tasks. MetaWorld is not installed or run.

**Changed files**
- `hymeko_rl/eval/task_monitor/metaworld.py` — added `DialTurnMonitor` (Composite
  root), `DialTurnProgressMonitor` / `DialTurnSuccessMonitor` /
  `DialTurnOvershootMonitor` (Strategy submonitors), `DialTurnContext`,
  `DialTurnVerdict`, `_wrap`. Refactored the coffee-push submonitor ABC into a generic
  `MetaWorldSubmonitor[_Ctx]` base (one submonitor contract for both task families, not
  one ABC per family); renamed `_GATING → _COFFEE_GATING`.
- `hymeko_rl/eval/task_monitor/__init__.py` — exported the dial-turn symbols +
  `MetaWorldSubmonitor`.
- `hymeko_rl/tests/test_metaworld_monitors.py` — +12 dial-turn tests.

**Monitor API** — `DialTurnMonitor(success_tolerance=0.05, min_rotation=0.05,
overshoot_tolerance=0.10, direction: Literal["positive","negative","auto"]="auto",
stagnation: StagnationMonitor | None = None).evaluate(traj) -> DialTurnVerdict`.
Gates on rotate-toward ∧ reached ∧ ¬overshot.

**Trajectory fields** — `dial_angle` (req, scalar radians), `target_angle` (req,
scalar radians; usually constant); `gripper_xy` / `dial_xy` / `contact` / `time`
optional. No MetaWorld types.

**Angular semantics** — signed error `s = wrap(angle − target)` in (−π, π],
`target_error = |s|`. `rotated_toward_target` iff `err0 − errfin > min_rotation`
(reverse ⇒ `rotated_away_from_target`); `target_reached` iff `errfin ≤
success_tolerance`; `overshot` iff the dial crosses to the far side of the target by
more than `overshoot_tolerance` (direction fixed or inferred from the initial side);
`monitor_pass = rotated_toward ∧ reached ∧ ¬overshot`. **Angle wrapping is correct**:
a trajectory crossing the ±π seam (`3.14 → −3.13` toward a target near −π) reads as a
0.25 rad approach, not a 6.03 rad false failure (`test_dial_angle_wrapping_no_false_failure`).

**No `AngularStagnationMonitor` needed** — `DialTurnContext.dist` is the angular error
series, so the generic `StagnationMonitor` (item #1, unmodified) composes on angular
progress directly (`test_stagnation_monitor_reused_on_dial_context`).

**Test results** — `pytest -p no:randomly` over the three monitor test files:
**59 passed** (12 new + 47 prior, all unchanged). ruff clean; mypy clean on changed
code (generic ABC, `Literal`, and the cross-context `StagnationMonitor` reuse all
typecheck); radon max CC = 8 (B).

**Does this prove both distance-based and angular task progress are covered?** Yes.
`CoffeePushMonitor` (Euclidean distance) and `DialTurnMonitor` (wrapped angular error)
share one `SubVerdict` / generic-Strategy / Composite-root design, both feed
`export_cip_variables` unchanged, and the single `StagnationMonitor` composes on both a
Euclidean and an angular error series via `SupportsDistanceSeries`. Build-order items
#1–#4 done; #5 (DAgger monitor-success selection hook) remains and was not touched.
