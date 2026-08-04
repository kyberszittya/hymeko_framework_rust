# Gradient-based RL for run-stop via differentiable simulation (BPTT) — beats CEM and tuned-linear

**Date:** 2026-08-05
**Branch/worktree:** `research/humanoid-com-lyapunov` @ `hymeko_humanoid` (head at start `071329ec`)
**Follow-up (2 of the "both"):** the memory's preferred scaling axis — gradient-based RL on the same interface.

---

## Summary

The run-stop dynamics are differentiable, so the cleanest gradient method is **backprop-through-time (BPTT)**:
roll a torch policy through a torch copy of `runstop_step` and descend a differentiable surrogate of the objective.
Its 1-hidden-layer net is flattened back into the numpy policy, so the **same held-out `evaluate`** scores it
against CEM and tuned-linear (the flattening parity is pinned by a test).

## Result — and an honest lesson about the surrogate

| method | held-out stop-success | fall rate |
|---|---|---|
| **BPTT (differentiable-sim gradient RL)** | **0.996** | 0.004 |
| CEM (evolutionary) | 0.941 | 0.000 |
| best single linear gain | 0.750 | — |

**BPTT beats both CEM and tuned-linear** — but only after fixing the surrogate:

- **Naive surrogate (final-step `vx²`) → 0.83, below CEM.** A final-step-only speed cost lets the policy dip to
  zero speed for one instant and drift; on the true discontinuous metric (stop *and stay* stopped, never fall) it
  underperformed the direct method (CEM optimises the true metric).
- **Aligned surrogate (`vx²` over the whole episode *tail* + a smooth upright penalty) → 0.996.** Penalising the
  tail forces a genuine held stop, aligning the smooth gradient objective with the true one. This is the honest
  takeaway: **differentiable-sim RL is only as good as the surrogate's alignment with the true objective** — get
  that right and the analytic gradient beats the black-box search; get it wrong and it loses.

- **With the safety shield:** BPTT + shield stops 0.69 (fall 0.000) — the conservative shield costs this
  aggressive policy more; safe-but-lower, honestly reported.

## Files touched

| File | LOC | notes |
|---|---|---|
| `scenarios/humanoid/centroidal_runstop_bptt.py` | +95 (new) | torch `BpttPolicy`, differentiable `_torch_step`, tail-penalty `rollout_loss`, `train_bptt`, numpy-param export |
| `tests/test_centroidal_runstop_bptt.py` | +55 (new) | 3 tests (param-conversion parity, beats-tuned-linear held-out, determinism) |
| `reports/2026-08-05-runstop-bptt-gradient-rl.md` | new | this report |

## CORE.YAML items touched
None. Uses the already-pinned torch (`==2.12.0`) — no dependency change. The BPTT net shares `policy_actions`'
shape so it reuses the existing numpy evaluation (no forked metric).

## Test results
- `pytest tests/test_centroidal_runstop_bptt.py -p no:randomly` → **3 passed in 27.9 s** (the training fixture is the bulk).
- `ruff check` → clean.

## Performance
- BPTT training (500 Adam steps of BPTT over ~450-step rollouts, batch = mixed set, torch CPU): ~40 s. Held-out
  `evaluate`: < 1 s. Peak RSS negligible (small net, CPU).

## Method note (why BPTT here, TD3 elsewhere)
BPTT exploits a **known, differentiable** simulator — an exact analytic policy gradient, cheaper and more
sample-efficient than model-free RL when it applies. **TD3 / model-free policy-gradient** is the method when the
dynamics are a black box (real robot, non-differentiable contacts); it fits the same `policy_actions` residual
interface and is the drop-in for the MuJoCo cross-check. CEM remains the robust, gradient-free baseline that
established the honest held-out verdict without tuning confounds.

## Open issues / follow-up
- **Shield-aware BPTT:** differentiate *through* the shield (it is a smooth, differentiable map) so the gradient
  policy learns to work with the safety envelope rather than fight it — should recover the shielded stop-success.
- **TD3 on the residual interface** for the non-differentiable / embodied (MuJoCo) setting.

## Provenance
Git SHA at start `071329ec`. Env: HyMeKo `.venv` (Python 3.11, torch 2.12.0 CPU, NumPy 2), macOS (darwin 25.5),
4 CPU threads. Deterministic: `torch.manual_seed`; pinned `dt = 4 ms`; deterministic grids. No GPU, no dataset.
