# Galambos coordination-reward A/B — verdict

**Date:** 2026-07-04 08:06 JST · **Branch:** `hymeko-neuro-migration` · **Commit under test:** `6707bab`
**Run:** collab off-policy CTDE (sa_hsikan, TD3+BC, 200k, best-checkpoint), 3 seeds × 2 variants, `coin_frictionloss=0` (free coin), difficulty 0.3.

## Result

| Variant | delivery median | per-seed | peak `both_contact` (per seed) |
|---------|-----------------|----------|-------------------------------|
| baseline (`galambos_task.hymeko`) | **0.16** | 0.08 / 0.16 / 0.18 | 0.0003 / 0.0019 / 0.0063 |
| coord (`+ both_approach 4.0`)       | **0.12** | 0.06 / 0.12 / 0.20 | 0.0048 / 0.0 / 0.0016 |

## Verdict: the coordination term is FALSIFIED

`coord (0.12) ≤ baseline (0.16)` — within seed variance (overlapping ranges), i.e. **no improvement**. Crucially,
`both_contact` stayed near-zero (~0.001–0.006) in **both** variants — the coordination gradient
(`both_approach = -max(left,right)`, penalising the lagging arm) did **not** raise simultaneous fingertip
contact. The A/B is internally valid (identical driver, only the reward differs), so this conclusion is clean:
**reward-term shaping is not the lever for two-arm coordination here.** This is consistent with the prior
on-record finding that reward rebalancing on this task is second-order and physics is the blocker.

## Calibration caveat (why I did NOT "adjust the terms" and re-run)

The baseline came in at **0.16**, not the recorded **0.40** (`experiments/2026_07_03_17_15_collab_coin_offpolicy/`).
I checked whether the env regressed: the fingertip-only collision masks and the 0.40 result are in the **same
commit** (`f8a5b57`), so the 0.40 run already had the masks; and the constants extraction kept the emitted MJCF
byte-identical (golden test). So **the env did not regress** — the gap is the new `Campaign`-based A/B driver
(`exp_galambos_coord_ab.py`) being a *weaker reproduction* of the original 0.40 setup (different BC-epoch /
TD3+BC hyperparameters than whatever produced 0.40).

Per evaluation-integrity (a mis-calibrated harness sends the optimization loop chasing a phantom): tweaking
reward terms against an uncalibrated baseline that under-reproduces the known-good 0.40 would burn compute on
noise. The term-lever hypothesis was *just* tested and came back negative; re-tweaking terms is the experiment
this run predicts will fail. So I stopped rather than launch another 3-hour run.

## Recommended next steps (for the morning decision)

1. **Calibrate the harness first** — reconcile `exp_galambos_coord_ab.py` to reproduce the 0.40 baseline (match
   the original BC-epochs / TD3+BC config), so any further A/B runs on a trustworthy reference.
2. **The coordination lever is not the reward** — the evidence points to (a) **BC-demonstrator quality** (the
   teacher must actually achieve two-fingertip grasps, so the policy starts in a good basin — `both_contact≈0`
   suggests it never does), or (b) a **`coin_frictionloss` curriculum** (start low so single-arm pushes bootstrap
   contact, ramp up to force two-arm force). These are the physics/demonstrator route, not term weights.

## Artifacts

`experiments/2026_07_04_03_33_galambos_coord_ab_baseline/` and `…_05_47_galambos_coord_ab_coord/`
(results.json + curves + GIFs + run.log per variant). Code committed at `6707bab`; tests 19 passed.
