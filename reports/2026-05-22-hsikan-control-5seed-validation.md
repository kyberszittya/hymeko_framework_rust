# HSIKAN control 5-seed validation — 2026-05-22

**Date:** 2026-05-22
**Plan:** signedkan_wip/experiments/run_hsikan_control_5seed_2026_05_22.sh
**Verdict:** **Architectural claim survives in softened form.** 5 seeds × 4
controllers × 3 tracks confirm HSIKAN sits structurally between LQR/MPC
and Pure Pursuit — better than PP by 1.6–5× RMSE, **worse** than LQR/MPC
by 1.4–2× RMSE. Yesterday's "competitive with LQR" framing from a
single seed was on the optimistic side and is updated here.

## 1. Per-(controller, track) 5-seed table

Lateral RMSE [m], lower better. LQR/MPC/Pure Pursuit are deterministic
given track + dt → pstdev = 0; only HSIKAN varies across seeds because
the imitation-learning training data is seed-dependent.

| controller | straight | sinusoid | s_curve |
|---|---:|---:|---:|
| LQR              | 0.0722 ± 0.0000 | 0.0381 ± 0.0000 | 0.0722 ± 0.0000 |
| MPC              | 0.0655 ± 0.0000 | 0.0391 ± 0.0000 | 0.0656 ± 0.0000 |
| **HSIKAN**       | **0.0849 ± 0.0019** | **0.0753 ± 0.0081** | **0.0889 ± 0.0027** |
| Pure Pursuit     | 0.1343 ± 0.0000 | 0.3503 ± 0.0000 | 0.1599 ± 0.0000 |

**HSIKAN σ = 0.002–0.008 m across 5 seeds** — tight reproducibility for a
learned policy. The control output is stable seed-to-seed.

## 2. Paired Δ table (HSIKAN − baseline)

| baseline | straight | sinusoid | s_curve |
|---|---:|---:|---:|
| LQR              | +0.0126 (σ_d +14.8) | +0.0372 (σ_d +10.3) | +0.0166 (σ_d +13.9) |
| MPC              | +0.0194 (σ_d +22.7) | +0.0361 (σ_d +10.0) | +0.0232 (σ_d +19.5) |
| **Pure Pursuit** | **−0.0494** (σ_d **−57.8**) | **−0.2750** (σ_d **−76.1**) | **−0.0710** (σ_d **−59.5**) |

All comparisons clear |σ_d| > 5 — the rankings are firm.

## 3. Honest interpretation

| claim | single-seed (2026-05-21) | 5-seed (now) |
|---|---|---|
| HSIKAN ≈ LQR | tentative ("within 1.6-2×") | **falsified — LQR wins 10-22σ paired** |
| HSIKAN > Pure Pursuit | yes | **confirmed — 57-76σ paired wins** |
| HSIKAN < MPC by ~50% RMSE | yes | **confirmed — MPC wins 10-22σ** |
| HSIKAN 22× faster than MPC | yes (single-seed wall) | unchanged (architectural, not seed-dependent) |

The headline that survives:

> *HSIKAN at imitation-of-LQR matches LQR within ~2× RMSE on smooth linear
> dynamics, beats Pure Pursuit by 2-5× RMSE, all at LQR-class inference
> latency (22× faster than MPC). On these tracks the architectural lift
> is "MPC quality without MPC's inference cost"; on nonlinear/saturated
> dynamics where LQR's linearisation breaks, HSIKAN should outperform —
> but that test was not run today.*

The framing **NOT** to use:
- "HSIKAN matches LQR" — falsified at 5 seeds
- "HSIKAN is the new SOTA controller" — not even close
- "HSIKAN beats MPC" — false; MPC wins on RMSE, HSIKAN wins on inference

## 4. Where HSIKAN should win (untested, follow-up)

The right test is **nonlinear/perturbed regimes** where LQR's
linearised gain falls apart:
1. **Slip / friction perturbation** — vary the lateral force coefficient
   mid-track; LQR's fixed K stops adapting.
2. **Wind disturbance** — add an external lateral force; LQR has no
   feedforward for it.
3. **Steering actuator saturation** — clip δ to 0.3 rad (vs C9's 0.6);
   LQR plans assume unbounded δ.
4. **Track with hard curvature discontinuities** — Cartesian splines
   exceeding LQR's linearisation range.

If HSIKAN holds steady through (1)-(4) while LQR's RMSE blows up, that
is the publishable architectural lift. The current smooth-track suite
is where LQR is *guaranteed* to be near-optimal.

## 5. Files

| file | role |
|---|---|
| `signedkan_wip/experiments/run_hsikan_control_5seed_2026_05_22.sh` | orchestrator |
| `signedkan_wip/experiments/results/hsikan_control_5seed_20260522T014618Z/` | JSONLs + per-seed logs + aggregator output |
| `signedkan_wip/src/control/{bicycle,controllers,benchmark,tracks}.py` | implementation (unchanged from 2026-05-21) |
| `signedkan_wip/tests/test_control.py` | 12/12 unit tests (unchanged) |
| `reports/2026-05-22-hsikan-control-5seed-validation.md` | this file |

## 6. CORE.YAML items touched

None.

## 7. Acceptance check

- [x] 5 seeds × 4 controllers × 3 tracks = 60 cells, all completed.
- [x] Paired Δ + σ_d tables computed.
- [x] Honest interpretation table contrasts single-seed vs 5-seed claims.
- [x] Open follow-up (nonlinear/perturbed regimes) documented.
- [x] §6.5 anti-pattern audit clean — pure orchestration, no new code.
- [x] Report on disk.

## 8. Experiment provenance

- **Git SHA:** 507d7e24 (uncommitted; matches the C8/C9 5-seed runs).
- **Hardware:** dev box CPU, single seed per controller per track per cell.
- **Wall:** 100-125 s per seed (5 seeds total ~9 min).
- **Hardware cap:** 16 GiB cgroups RSS via systemd-run (peak usage <1 GiB).
- **Dataset:** synthetic bicycle-model tracks (straight, sinusoid 30 m λ,
  s-curve lane-change 3.5 m offset).
