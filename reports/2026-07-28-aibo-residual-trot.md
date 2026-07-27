# Training the AIBO — bounded residual over the trot gait (multi-position goal-reaching)

**Date:** 2026-07-28 (JST) · **Branch:** `research/aibo-lyapunov-ph` (worktree `hymeko_aibo`)
**SIMULATION. Residual SAC (no from-scratch).** ·
**Verdict: `BOUNDED_RESIDUAL_OVER_TROT_GAIT_DOES_NOT_EXTEND_MULTI_POSITION_REACH` (2 designs, provisional 1-seed each).**

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
  terminates. Two residual designs, both bounded (coin-R8), both trained 120k steps, best-val
  checkpoint, held-out test grid reported once.

## Result — two honest negatives, one instructive

| design | what the residual acts on | verdict | reach (test) | mean min-dist (test) |
|---|---|---|---|---|
| scaffold (`a=0`) | — | baseline | 0.20 | 0.355 |
| **leg** | raw 12-D leg targets, every step | **`REGRESSES`** | 0.00 | **0.591** (worse) |
| **steer** | 2-D gait params (Δyaw, Δspeed) | **`MATCHES`** | 0.20 | 0.354 (≈) |

- **Leg residual REGRESSES:** a raw per-step residual on the 12 leg targets **breaks the periodic
  trot limit cycle** — it disrupts forward locomotion (min-dist 0.355 → 0.591), val reach flat 0
  across all 6 evals. This is the key mechanism: unlike PnP/humanoid (quasi-static/postural
  scaffolds), the AIBO scaffold is a **dynamic periodic gait**, and perturbing its output every step
  destroys the trot.
- **Steer residual MATCHES:** modulating the gait's *parameters* (steering + speed) instead of its
  raw output **preserves the limit cycle** (no regression) — but the learned residual converges to
  ≈ the scaffold (reach 0.20 = 0.20, min-dist 0.354 ≈ 0.355, 6 flat evals). The scaffold's off-axis
  failure is a **kinematic limit of the gait** (asymmetric turning, slow forward speed); a *bounded*
  parameter tweak within the gait's structure cannot overcome it.

## The campaign insight (cross-embodiment)

Bounded residual RL helps when the needed correction is **within the scaffold's capability envelope**
(humanoid *sagittal* balance 0→12/12; PnP *transport* 0.58→1.0 constant residual) — but **not** when
the deficit is a **structural / kinematic limitation** (humanoid *lateral step* 0/12; **AIBO off-axis
steering**). Those need a *different primitive* (a real step; a sharper turning gait), not a residual
over the existing one. The AIBO result adds a third embodiment to this dissociation, and a sharper
mechanism: **a raw residual over a periodic gait breaks its limit cycle (leg=REGRESS); a
parameter residual preserves it but can't exceed its kinematic reach (steer=MATCH).**

## Can we train the AIBO? — yes; this just isn't the lever

The infrastructure works and trains cleanly (563/497 steps/s leg/steer, no OOM, best-val checkpoints
saved). The honest finding is that **a bounded residual over *this* gait is not the lever** for
multi-position reaching. Untested next hypotheses (would change the *primitive*, not add a residual):
a learned/scripted **sharper turning gait** (fixing the gait's asymmetric steering directly), a
**phase-gated** residual, or a **reach-aligned** reward. Caveat: 1 seed × 1 residual-scale × 1 reward
per design — the verdicts are provisional on that config, though both eval curves are consistent.

## Files

```
scenarios/aibo/residual_trot.py       NEW  (ResidualTrotEnv: multi-goal, leg|steer residual, tested core)
scenarios/aibo/run_aibo_residual.py   NEW  (residual SAC + held-out multi-position eval; --mode leg|steer)
tests/test_aibo_residual_trot.py      NEW  9 tests
reports/2026-07-28-aibo-residual-trot/{result_leg,result_steer}.json  NEW
```

## Tests / perf / provenance

`ruff` clean. **9/9** residual-env tests (dims, goal-in-range, bounded residual ≤ scale, a=0 = pure
scaffold in both modes, scaffold never falls, progress reward); full AIBO suite **66/66**. Training:
120k steps each, ~4–5 min wall, **peak RSS < 2 GB** (well under the 16 GB cap), CPU, seed 0, MuJoCo.
Git SHA at commit; `result_{leg,steer}.json` carry the numbers.

## Bottom line

We **can** train the AIBO (residual SAC runs cleanly), but a **bounded residual over the trot gait
does not extend multi-position reach**: the raw-leg residual breaks the periodic gait (regresses),
the parameter residual preserves it but can't overcome the gait's kinematic steering limit (matches).
Honest two-design negative — consistent with the campaign's in-envelope-vs-structural dissociation.
The lever for multi-position reaching is a **richer turning primitive**, not a residual.
