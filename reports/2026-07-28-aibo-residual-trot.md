# Training the AIBO — bounded residual over the trot gait (multi-position goal-reaching)

**Date:** 2026-07-28 (JST) · **Branch:** `research/aibo-lyapunov-ph` (worktree `hymeko_aibo`)
**SIMULATION. Residual SAC (no from-scratch).** ·
**Verdict: `SAGITTAL_RESIDUAL_FAILS_BUT_RICHER_ABDUCTION_ACTION_SPACE_REACHES_OFF_AXIS` (4 designs, provisional 1-seed each).**

---

## Question

Can we *train* the AIBO (it was model-based only)? The user asked for a trained policy that reaches
**multiple goal positions**. The campaign-consistent path is a **bounded residual over a certified
scaffold** (coin-R8 / humanoid / PnP), so: does a bounded residual over the `SteeredTrotGait`
scaffold reach more goal positions than the gait alone, on a **held-out** (distance × bearing) grid?

## Setup

- **Scaffold** (`a = 0`): `SteeredTrotGait` + heading pursuit, under the motion contract. Measured
  baseline: reaches only **1/5 held-out goals** (min-dist 0.355 m mean) — it steers poorly off-axis
  and closes distance slowly — but **never falls** (upright ≈ 1.0), the coin-R8 safe-scaffold
  prerequisite.
- **Env** (`residual_trot.py`, tested): multi-goal reset (dist 0.5–0.75, bearing ±40°), compact 9-D
  obs (goal-relative + gait phase), reward = progress + heading + reach bonus − control, fall
  terminates. Three residual designs, all bounded (coin-R8), each trained 120k steps, best-val
  checkpoint, held-out test grid reported once.

## Result — three honest negatives, monotone and instructive

| design | what the residual acts on | verdict | reach (test) | mean min-dist (test) |
|---|---|---|---|---|
| scaffold (`a=0`) | — | baseline | 0.20 | 0.355 |
| **leg** | raw 12-D leg targets, every step | **`REGRESSES`** | 0.00 | **0.591** (worst) |
| **phase** | 12-D leg targets **phase-gated per leg** | **`REGRESSES`** | 0.00 | **0.524** (less) |
| **steer** | 2-D gait params (Δyaw, Δspeed) | **`MATCHES`** | 0.20 | 0.354 (≈) |
| **omni** | 4-D per-leg **ABDUCTION** (lateral crab) | **`REACHES MORE` ✅** | **0.40** | 0.379 |

The three designs form a **monotone disruption-vs-preservation trend**: the more the residual touches
the raw leg targets, the more it breaks the trot.

- **Leg residual REGRESSES (worst):** a raw per-step residual on the 12 leg targets **breaks the
  periodic trot limit cycle** (min-dist 0.355 → 0.591, val reach flat 0). Unlike PnP/humanoid
  (quasi-static/postural scaffolds), the AIBO scaffold is a **dynamic periodic gait**, and perturbing
  its output every step destroys the trot.
- **Phase-gated residual REGRESSES less:** gating each leg's residual by its own stride phase
  (`g_l = ½(1+sin(ph+DIAG_PHASE_l))`, so the residual pulses in sync with the gait) **does reduce the
  disruption** (0.591 → 0.524) — confirming the phase-locking mechanism — but it still slows the gait
  enough to regress, and reach stays 0. The smooth gate is nonzero across most of the cycle, so it
  never fully protects the trot.
- **Steer residual MATCHES:** modulating the gait's *parameters* (steering + speed) instead of its
  raw output **preserves the limit cycle** — the learned residual converges to ≈ the scaffold (reach
  0.20 = 0.20, min-dist 0.354 ≈ 0.355). The scaffold's off-axis failure is a **kinematic limit of the
  gait** (asymmetric turning, slow forward speed); a *bounded* correction within the gait's structure
  cannot overcome it, and the most gait-preserving design tops out at MATCH — never improvement.

## The resolution — a RICHER action space (abduction) breaks the wall ✅

The three sagittal designs all act on the *same* leg axes the trot already uses (hip-flex + knee). The
`omni` design opens the **unused axis**: a 4-D per-leg **abduction** residual, phase-locked, over the
forward trot — the AIBO's third leg DOF, which the sagittal gait leaves idle. It produces a **lateral
crab-walk** (measured ~0.1–0.2 m sideways, upright), so the learned policy reaches off-axis goals by
**side-stepping** instead of turning — bypassing the whole turn/stability wall.

**Measured (best-val, upright):** the omni residual reaches **2/5** held-out goals vs the scaffold's
**1/5** — it reaches the straight goal **and the +20° off-axis goal** (min-dist 0.12, upright 0.99),
the **first off-axis reach in the entire arc**. It is the only mode with a non-zero validation curve
(peaks 0.25 then the entropy anneal destabilises it — the best-val checkpoint captures it). Honest
limits: the crab is **asymmetric** (reaches +20° but not −20° — the persistent diagonal-gait bias) and
**limited range** (±40° still out of reach). So it's a **real but modest positive**: the richer action
space is the lever, and a learned controller uses it — but symmetric, wide-range omnidirectional
reaching needs more (a symmetric crab + range + training stability).

**Why omni works where the sagittal residuals didn't:** abduction is **orthogonal** to the fore-aft
trot, so the residual adds a lateral DOF *without* perturbing the limit cycle (the leg residual broke
it; the steer residual couldn't exceed the gait's sagittal reach). The deficit was the **missing
lateral action**, and opening it is what a residual can exploit.

## The campaign insight (cross-embodiment)

Bounded residual RL helps when the needed correction is **within the scaffold's capability envelope**
(humanoid *sagittal* balance 0→12/12; PnP *transport* 0.58→1.0 constant residual) — but **not** when
the deficit is a **structural / kinematic limitation** (humanoid *lateral step* 0/12; **AIBO off-axis
steering**). Those need a *different primitive* (a real step; a sharper turning gait), not a residual
over the existing one. The AIBO result adds a third embodiment to this dissociation, and a sharper
mechanism (monotone across three designs): **a raw residual over a periodic gait breaks its limit
cycle (leg=REGRESS 0.591); phase-gating the residual reduces the disruption (phase=REGRESS 0.524) but
not enough; a parameter residual preserves the gait but can't exceed its kinematic reach
(steer=MATCH 0.354).**

## Can we train the AIBO? — YES, with the RIGHT action space

The infrastructure trains cleanly (~470–500 steps/s, no OOM, best-val checkpoints). The lesson is
about the **action space**, not the algorithm: a residual on the axes the trot *already uses*
(sagittal) can't help (breaks or can't-exceed the gait); a residual on the **unused abduction axis**
(lateral) **does** — because it adds the missing DOF rather than perturbing the gait. Caveat: 1 seed
× 1 config per design; the omni win is +0.2 (0.2→0.4), modest and asymmetric.

## Files

```
scenarios/aibo/residual_trot.py       NEW  (ResidualTrotEnv: multi-goal, leg|steer|phase|omni residual, tested core)
scenarios/aibo/run_aibo_residual.py   NEW  (residual SAC + held-out multi-position eval; --mode leg|steer|phase|omni)
tests/test_aibo_residual_trot.py      NEW  14 tests
reports/2026-07-28-aibo-residual-trot/{result_leg,result_steer,result_phase,result_omni}.json  NEW
```

## Tests / perf / provenance

`ruff` clean. **14/14** residual-env tests (dims, goal-in-range, bounded residual ≤ scale, a=0 = pure
scaffold in all four modes, phase gates ∈ [0,1] summing to 2, **omni abduction produces lateral
motion**, scaffold never falls, progress reward); full AIBO suite green. Training: 120k steps each,
~4–5 min wall, **peak RSS < 2 GB** (under the 16 GB cap), CPU, seed 0, MuJoCo. Git SHA at commit;
`result_{leg,steer,phase,omni}.json` carry the numbers.

## Bottom line

We **can** train the AIBO to reach more positions — but only with the **right action space**. Three
residuals over the trot's *sagittal* axes fail (raw-leg REGRESS 0.591 breaks the gait; phase-gated
REGRESS 0.524 less; param steer MATCH 0.354 can't exceed the gait's reach) — the off-axis deficit is a
**missing lateral action**, not something the sagittal residual can fix. Opening the **unused
abduction DOF** (the `omni` mode: a learned phase-locked lateral crab) **doubles the reach (0.2 → 0.4)
and lands the first off-axis goal upright (+20°)** — the richer action space is the lever, and a
learned controller exploits it. Honest limits: modest (+0.2), asymmetric (+20° not −20°), ±40° still
out of range. The full AIBO multi-position arc resolves: the wall was the **action space**, and the
fix is to give the learned controller the AIBO's third (lateral) leg DOF.
