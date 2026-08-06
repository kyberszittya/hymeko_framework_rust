---
title: PHASE_GATED_RESIDUAL_UPDATE0_REPRODUCED + EARLY_PHASE_STRUCTURAL_PRESERVATION_PASS
date: 2026-07-23
slug: coin-residual-update0-structural
task: coin_v3 delivery — phase-gated learned-residual TD3 (§1-§5)
verdict: EARLY_PHASE_STRUCTURAL_PRESERVATION_PASS
prior_gate: PHASE_GATED_RESIDUAL_UPDATE0_REPRODUCED
---

# Frozen-base composite actor — update-0 reproduction + structural preservation

**Created-at:** 2026-07-23 03:35 CEST
**Accepted gate:** `STABLE_OBJECT_ENGAGEMENT_V1` (SHA `7633dd3c`). This stage builds the composite controller
(§1), freezes the base (§2), zero-inits the residual (§3), and proves early-phase structural preservation (§4).
Critic / replay / gradient / trust-region / smoke are subsequent turns (§6-§19).

## §1 composite controller (`coin_residual_controller.py`)

    base_t      = clip(pi_0(obs_t), -4, 4)                      # frozen
    residual_t  = 0.25 * tanh(residual_actor(obs_t))           # bounded [-0.25, 0.25], §5
    composite_t = clip(base_t + gate_t * residual_t, -4, 4)    # gate_t ∈ {0,1} from STABLE_OBJECT_ENGAGEMENT_V1

- `BoundedResidualTransform`: `0.25·tanh(raw)` — explicit bounded transform (NOT unbounded-then-env-clip, §5);
  slope `0.25` at 0 so the residual can learn; forward+gradient tested.
- `ZeroInitResidualActor`: 48→256→256→4 MLP, **last layer zero-initialized** ⇒ executed residual == 0 at init.
  Contract SHA prefix `25a820ea`. Sees ONLY `node_features` (48) — no phase/target/success/planner/traj/future.

## §2 frozen base

`pi_0` loaded from `frozen/pi0_shared_clip_actor.pt` (file-SHA **`1902454c`**) via `load_frozen_clip_actor`
(`requires_grad=False` on every param). Guards `assert_frozen_base` (raises if any base param requires grad) and
`assert_base_absent_from_optimizer` (raises if a base tensor is in any optimizer param group). Param hash unchanged
across the whole run (`pre == post`).

## §3 update-0 reproduction — PHASE_GATED_RESIDUAL_UPDATE0_REPRODUCED

| quantity | composite+gate | pi_0 direct | required |
|----------|----------------|-------------|----------|
| headline | **3/9** | 3/9 | 3/9 |
| validation | **2/30** | 2/30 | 2/30 |
| grasp | **9/9** | 9/9 | 9/9 |
| delivered | **{1011,1447,1568}** | — | {1011,1447,1568} |
| composite−base maxdiff (gate=0) | **0.0** | — | 0 |
| composite−base maxdiff (gate=1) | **0.0** | — | ~0 |
| pi_0 param hash | unchanged | — | unchanged |

The full composite+gate rollout from neutral reproduces `pi_0` **exactly** — including gate=1 states, because the
residual is identically 0 at init. Figure `reports/figures/coin_residual_update0.png` (all composite actions on the
`y=x` diagonal; residual output a single spike at 0).

## §4 structural preservation — EARLY_PHASE_STRUCTURAL_PRESERVATION_PASS

Unit tests (`test_coin_residual_controller.py`, **10/10 pass**, 1.44 s):

- **gate=0 ⇒ composite == base BIT-identical** even with the residual net forced to large / saturated / random
  outputs (`test_composite_gate0_equals_base_for_arbitrary_residual`, `torch.equal`).
- gate=1 at init ⇒ composite == base (residual 0).
- **residual gradient is exactly 0 when gate=0** (`test_residual_gradient_zero_when_gate_off`) — arbitrary residual
  optimizer steps cannot alter gate-off (approach/acquisition) actions.
- residual gradient nonzero when gate=1 (last layer learns from zero init).
- frozen-base assertions fire correctly; residual-only optimizer excludes `pi_0`.
- bounded transform range + slope; residual contract SHA.

Rollout-side: composite == `pi_0` at update 0 ⇒ approach and acquisition fingerprints are `pi_0`-identical; the gate
`reset()` returns to `EARLY_CONTROL` (tested in `test_coin_stable_engagement.py::test_reset_clears`).

## Files touched

- `hymeko_rl/coin_delivery/coin_residual_controller.py` (new, ~150 L).
- `hymeko_rl/tests/test_coin_residual_controller.py` (new, 10 tests).
- `experiments/2026_07_22_coin_v3_learning/rl_entry/coin_residual_update0.py` + `plot_update0.py` + `update0.json`.
- `reports/figures/coin_residual_update0.png`.

**CORE.YAML:** none. **Frozen base / reward / gamma / bundle / obs contract / gate thresholds unchanged.**
SAC quarantined. Final-test bank untouched. Mac; kato14 clean. §20 commits 1-2.

## Next (§6 onward)

`PHASE_GATE_REPLAY_STATE_CONTRACT_PASS` (V2 controller state in replay + TD3 target from `gate_{t+1}`; 6 transition
reference tests) → `PHASE_GATED_RESIDUAL_CRITIC_PASS` (composite-action critic conditioned on encoded V2 state) →
`LATE_PHASE_RESIDUAL_GRADIENT_CONTRACT_PASS` → hard late-phase trust region → guarded micro-smoke.
