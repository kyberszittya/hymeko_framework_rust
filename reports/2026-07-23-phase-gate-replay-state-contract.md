---
title: PHASE_GATE_REPLAY_STATE_CONTRACT_PASS — stored gate_tp1 controls the target, no fresh FSM
date: 2026-07-23
slug: phase-gate-replay-state-contract
task: coin_v3 delivery — phase-gated learned-residual TD3 (§5)
verdict: PHASE_GATE_REPLAY_STATE_CONTRACT_PASS
---

# Replay / controller-state contract V2 + TD3 target-action semantics

**Created-at:** 2026-07-23 04:05 CEST
**Decisive invariant:** a replay minibatch never reconstructs, advances, or queries a fresh phase FSM; the target
action for transition `t` uses the **stored** `gate_{t+1}`. No TD3 training this stage — target construction only.

## §5.1 schema — `PHASE_GATE_CONTROLLER_STATE_V2` (replay side)

`ReplayControllerStateV2` (frozen dataclass): `gate` (deployed multiplier, restricted to `{0,1}`), `mode`,
`bilateral_counter`, `uni_counter`, `uni_side`, `loss_counter`, `provenance` (coin/tip history, last mechanism,
comotion flag). Constructed from a live gate by value (`from_gate`, deep-copied provenance). `ResidualReplayBuffer`
stores every field **by value** (defensive copies; verified by `test_replay_stores_by_value_not_reference`).
Deterministic serialization; schema fingerprint **`0b11b60e`**.

## §5.2 collection ordering

read `controller_state_t` → compute `gate_t` → deployed composite action → step env → advance the online gate
**exactly once** → read `controller_state_{t+1}` → store the complete transition. `gate_t`/`gate_{t+1}` are never
recomputed after insertion; the FSM is never advanced during sampling.

## §5.3 target-action contract

    base_tp1     = clip(pi_0(obs_tp1), -4, 4)                                  # frozen, no smoothing, no_grad
    residual_tp1 = clip(0.25*tanh(residual_target(obs_tp1)) + clamp(eps), -0.25, 0.25)   # smoothing on residual only
    target_tp1   = clip(base_tp1 + gate_tp1 * residual_tp1, -4, 4)             # STORED gate_tp1

## §5.4 six transition-reference tests — all pass

`test_coin_residual_replay.py` (8 tests: the six + round-trip + by-value):

1. **Replay round-trip** — `gate_t`/`gate_tp1` returned bit-identical, dtype `float32`, shape `(B,)`.
2. **Stored gate controls target** — gate=1 row gets residual, gate=0 row == base.
3. **No fresh FSM** — `StableEngagementGate.__init__/update/reset` monkeypatched to raise; target construction on
   stored gates a fresh gate would not reproduce → **not called** (`calls == 0`).
4. **Gate zero preserves base** — residual + noise forced large/saturated/random, `gate_tp1==0` ⇒
   `torch.equal(target, clip(base))`; base hash unchanged.
5. **Gate one residual-only smoothing** — learner target == independent reference `clip(base + bounded_smoothed_
   residual)`, smoothed residual `≤ 0.25`; base branch unsmoothed.
6. **Terminal masking** — `done=1` ⇒ `y == reward` regardless of `q_next` (gate/residual/noise); non-terminal
   depends on `q_next`.

## §5.6 deterministic fixture (machine-readable `unilat`→`replay_fixture.json`)

| transition | g_t | g_tp1 | done | maxdiff | FSM? | g0=base | term mask |
|-----------|-----|-------|------|---------|------|---------|-----------|
| EARLY→EARLY | 0 | 0 | 0 | 0 | no | ✓ | — |
| EARLY→RESIDUAL | 0 | 1 | 0 | 0 | no | — | — |
| RESIDUAL→RESIDUAL | 1 | 1 | 0 | 0 | no | — | — |
| RESIDUAL→REACQUIRE | 1 | 0 | 0 | 0 | no | ✓ | — |
| terminal | 1 | 0 | 1 | 0 | no | ✓ | ✓ |
| truncated_reset | 1 | 0 | 0 | 0 | no | ✓ | — |

Every learner target matches the reference (maxdiff 0), **no FSM invoked** in any row, gate_tp1=0 rows equal the
base bit-identically, terminal bootstrap masked. Figure `reports/figures/coin_replay_state_contract.png`.

## §5.5 optimizer / gradient guards (`test_optimizer_and_gradient_guards`)

`assert_frozen_base` + `assert_base_absent_from_optimizer` pass; residual-only optimizer; **base gradients remain
`None`** through the target path; `gate_tp1==0` ⇒ zero residual gradient; `gate_tp1==1` ⇒ residual path active.

## §5.7 regression — unchanged

- Update-0: HL **3/9**, VAL **2/30**, grasp **9/9**, delivered **{1011,1447,1568}**, composite−base maxdiff **0.0**,
  π₀ hash unchanged.
- All prior tests green: **44/44** (12 phase_gate + 14 stable_engagement + 10 residual_controller + 8 residual_replay).

## Limitations / non-claims

No TD3 training was run; no critic was trained. This stage validates **only** replay storage + target-action
construction semantics. The composite-action critic (conditioned on encoded V2 state) is the next gate
`PHASE_GATED_RESIDUAL_CRITIC_PASS`; `td_target_scalar` here is validated for masking only, with a placeholder
`q_next`. No claim about learned policy quality.

## Files touched

- `hymeko_rl/coin_delivery/coin_residual_replay.py` (new, ~150 L).
- `hymeko_rl/tests/test_coin_residual_replay.py` (new, 8 tests).
- `experiments/2026_07_22_coin_v3_learning/rl_entry/coin_residual_replay_fixture.py` + `plot_replay_fixture.py`
  + `replay_fixture.json`.
- `reports/figures/coin_replay_state_contract.png`.

**CORE.YAML:** none. Frozen π₀ (`1902454c`) / reward / gamma / bundle / obs / gate thresholds / residual range
`[-0.25,0.25]` / action bounds `[-4,4]` unchanged. SAC quarantined. Final-test bank untouched. Mac; kato14 clean.
§20 commit 3.

## Next gate

`PHASE_GATED_RESIDUAL_CRITIC_PASS` — a composite-action critic conditioned on the encoded V2 controller state,
calibrated on a residual-specific held-out panel (transport/entry/settling/contact-retention rankings, no OOD
residual-boundary preference), disjoint train/authorization/final panels. The expected next stage is the first
controlled residual-only TD3 update test, not a training campaign.
