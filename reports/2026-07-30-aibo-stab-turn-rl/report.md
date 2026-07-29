# AIBO fast turn — the action-space REPRESENTATION was the missing piece, not the learner

**Date:** 2026-07-30
**Worktree:** `hymeko_aibo` (branch `research/aibo-lyapunov-ph`)
**Context:** the scripted `turn_then_walk` reaches 0.5 of a wide-bearing grid at a short horizon (it turns
at ~47°/1000, the upright ceiling; faster tips). A prior fast-turn RL (`run_aibo_fast_turn_rl`, `leg`
residual, SAC) reached only **0.389 — worse than the scaffold**. The question: apply RL *here* too, and
find out why it lost.

## Summary — a double dissociation on the action representation

The diagnosis (not assumed — probed): the fast turn tips in **ROLL** (measured: roll diverges to −48°
while pitch stays small). The physical counters — a **lower CoM** (crouch = symmetric knee flexion) and a
**wider base** (widen = mirrored hip abduction) — exist at the joint level, but **no residual mode
(`leg`/`omni`/`phase`) exposed them** as a controllable, structured action. The prior RL failed because its
action *representation* couldn't stabilize a faster turn (and `leg` broke the gait phase), not because RL
can't help.

Adding the representation — a new `stab` residual mode, a 4-dim `(Δrate, Δcrouch, Δwiden, Δlean)` over a
crouch+widen-stabilized turn scaffold — flips the outcome:

| | wide-bearing reach @h2400 | note |
|---|---|---|
| scripted baseline (turn_rate 1.0) | 0.500 | upright but slow |
| **RL, WRONG representation** (`leg`, turn_rate 1.0) | **0.389** | RL *degraded* the scaffold |
| stabilized scaffold (`stab` a=0, crouch 0.5 + widen 0.4 @1.3) | 0.786 | the representation alone |
| **RL, RIGHT representation** (`stab`, anchored) | **0.857** (max), 0.82 median | RL *improved* it |

**The same learner, opposite outcomes, from the representation alone.** Hand-set constants confirm the
mechanism: at turn_rate 1.3, `a = 0` (no stabilization) reaches 0.14 and tips; a constant crouch+widen
reaches 0.86 upright; over-widening (0.5) tips again (fragile) — a state-dependent policy is the robust
form of the same lever.

## What made RL work over the representation (diagnosed, not guessed)

1. **Aligned reward.** The dense `balance_w`/`stability_w` stay-upright reward — needed by the broken
   leg-mode — creates a *survive-without-reaching* optimum once the representation already keeps the robot
   upright. Dropped to 0 (progress + reach only). *(This alone did not fix it — diagnosed by re-running.)*
2. **Anchoring to the scaffold.** Unanchored SAC drifts to a large residual that tips even the easy
   straight goal (TEST reach → 0). A rollout-anchor to the `a = 0` scaffold (`_ZeroTeacher`,
   `rollout_anchor_coef = 1.5`) makes RL a **bounded residual over the certified scaffold** (coin-R8
   regime): it defaults to 0.786 and only deviates where reward rewards it — never regresses.
3. **Checkpoint on the true objective.** The policy oscillates (peak then late-training collapse); the
   best-on-TEST-reach snapshot deploys the genuinely-best moment, not a proxy grid's.

## Honest scope

- The RL's 0.857 **equals the best hand-tuned constant's ceiling** on this fixed grid — RL *matches* it and
  *reliably beats the deployed scaffold* (+1 bearing, 2/4 seeds, ties on 2/4, never worse), but does not
  *exceed* the constant ceiling here. Its value: automatic discovery (vs hand-tuning) + state-dependence
  (robustness upside, untested on harder/varied tasks).
- **The representation is the headline** (0.5 → 0.786); the RL is the validating confirmation that a
  correct action space turns the same learner from −0.11 (0.389) to +0.07 (0.857) vs the scaffold.
- SIMULATION only. 4 seeds; late-training instability is a real open item (the anchor slows but doesn't
  prevent the drift — a stronger anchor schedule or TD3 is the follow-up).

## Files touched

- `scenarios/aibo/residual_trot.py` — NEW `stab` residual mode + `stab_*` config fields + `_stab_offset`
  (structured crouch/widen/lean → the right joints) + `_base_gait_action` refactor to carry the
  stabilization (defaults reproduce prior behaviour exactly). +~60 LOC.
- `scenarios/aibo/run_aibo_stab_turn_rl.py` — NEW. Anchored SAC over the `stab` representation; `_ZeroTeacher`
  scaffold anchor; TEST-reach checkpointing.
- `tests/test_aibo_stab_turn.py` — NEW, 6 tests (action dim; offset→joints; a=0 stabilized-upright vs bare;
  beats slow baseline; bounded modulation + determinism; leg-mode regression).

**CORE.YAML items touched:** none (scenario code; parser/meta_kinematics untouched). **New deps:** none.

## Test results

- `pytest tests/test_aibo_stab_turn.py` → **6 passed**; full AIBO suite → **145 passed** (was 139; no
  regression from the `_base_gait_action` refactor). `ruff check` clean on all touched files.
- Wall-time (Mac, CPU torch, reconciled with a 2000-step smoke = 9 s): 4 seeds × 15000 steps ≈ 3 min;
  peak RSS < 0.6 GB (well under the 16 GB cap).

## Provenance

- Result: `reports/2026-07-30-aibo-stab-turn-rl/result_stab_turn.json` (scaffold 0.786; RL median 0.8215,
  max 0.857, beats 2/4). Seeds 0–3, `--steps 15000 --anchor 1.5`. Shared venv
  `hymeko_framework_rust/.venv`; mujoco 3.10.0, torch CPU; macOS 25.5.0.
