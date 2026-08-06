# CANONICAL_DYNAMIC_EXPERT_BLOCKED — v2 robot mass/inertia mismatch breaks behavioral reproduction

**Created-at:** 2026-07-22 15:41 JST
**Branch:** recovery/coin-hymeko-bundle-and-results
**Bundle:** `388dd238c2546354` (graph/checkpoint/control/scene gates unaffected — this is purely a mass/inertia bug)

## Blocker

The dynamic expert EXISTS — the frozen neutral-delivery chain (E_valselect approach → handoff transport) delivers
**3/9** on the headline panel `_HEADLINE=(1011,1045,1164,1174,1202,1278,1358,1447,1568)` — but ONLY on the legacy
robot. On the canonical v2 robot it delivers **0/9** (grasp 3/9 both — acquisition works; transport+settle fails).
So `CANONICAL_DYNAMIC_EXPERT_PASS` cannot be reached until the v2 robot reproduces legacy dynamics.

## Exact mechanism — the parity gate missed mass/inertia

| body | legacy mass | v2 mass |
|---|---|---|
| base | 0.0365 | **0.4000** |
| link1 | 0.0796 | **0.2500** |
| link2 | 0.0597 | **0.2000** |
| fingertip | (in-link) | 0.0200 |
| **total** | **0.3516 kg** | **1.7400 kg (5×)** |

Root cause: `galambos_planar_v2.hymeko` carries arbitrary explicit `mass` fields (0.4/0.25/0.2/0.02) which the emitter
writes as `<inertial mass=… diaginertia=…/>` (with a crude `diaginertia` formula). The golden `make_planar_arms_mjcf`
carries NO `<inertial>`, so MuJoCo derives mass+inertia from the geometry (density 1000). The v2 arm is ~5× too heavy,
and `GALAMBOS_PLANAR_HYMEKO_EQUIVALENCE_PASS` checked geometry / kinematics (0.00 mm) / collision masks but NOT
mass/inertia — the gap that let this through. Step-zero actions are identical (Δ0.0); the full rollout diverges under
the wrong dynamics.

## Diagnostic (reproduction)

```
# canonical v2: 0/9
python -c "from hymeko_rl.experiments.coin_neutral_start import eval_composed,_HEADLINE,neutral_env; \
from hymeko_rl.train.sac import build_sac; import torch; \
tr=build_sac('mlp',obs_dim=41,flat_dim=41,action_dim=6,action_scale=1.0)[0]; \
tr.load_state_dict(torch.load('experiments/2026_07_21_coin_neutral_handoff/handoff_best.pt',weights_only=True)); \
print(eval_composed(tr,_HEADLINE,grasp_hold=3,env_cf=neutral_env(prefix_steps=0)))"
# legacy (monkeypatch make_coin_env → robot_source='legacy_python'): 3/9
```

## Attempted fix + why it is not a one-liner

Stripping the emitted `<inertial>` so MuJoCo density-computes it makes the TOTAL mass match legacy exactly (0.3516)
and the link masses match — BUT it redistributes the tip mass into the welded fingertip child bodies (v2 link2 0.048
+ fingertip 0.011 vs legacy link2 0.0597 all-in-link). Under the frozen chain this produces `QACC NaN` instability on
BOTH POINT and E0 within ~13 steps and breaks the discounted-alignment strict reference. A correct fix must give the
v2 bodies inertial properties that match legacy EXACTLY (link2 carries the full tip mass; the fingertip helper bodies
are effectively massless), then extend the parity gate to assert `body_mass` + `body_inertia` parity, then re-verify
3/9. Reverted for now to keep all recovered-bundle gates green (18/18).

## Minimal required fix (bounded robot-spec task — NOT a compute problem)

1. Make the v2 robot inertial properties match the golden body-for-body (mass + diaginertia), with the fingertip
   helper bodies massless and the tip mass folded into link2 — via the adapter or corrected spec masses.
2. Extend `test_galambos_planar_v2_parity` with a `body_mass` + `body_inertia` parity assertion (the missing gate).
3. Re-run the chain: expect 3/9 on canonical v2 == legacy → `CANONICAL_DYNAMIC_EXPERT_PASS`.

Additional KatoLab compute alone cannot fix this — it is a robot-specification inertia bug, not a search-budget or
training issue. Once fixed, the neutral chain IS the dynamic expert (3/9), and the BC/DAgger phase can proceed.

## Note on FROZEN_ARTIFACT_RUNTIME_COMPATIBILITY

This confirms the user's terminology correction: the §10 step-zero Δ0.0 result proved runtime compatibility, NOT
behavioral reproduction. The behavioral gap (3/9 → 0/9) is now explained and localized to the robot inertia.
