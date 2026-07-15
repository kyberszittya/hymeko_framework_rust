---
title: Recurrent / history residual-clone gap test (pick-place)
date: 2026-07-15
scope: does temporal/history info close the 0.792 feedforward residual-clone gap? (Step 1 after F-PP-009)
status: experiment (no TD3+BC, no kato15, no vanilla SAC — gate-first)
core_touched: none
---

# Recurrent / history residual-clone gap test

**Question (Step 1).** The abstraction audit (F-PP-009) proved the residual space/target/obs are correct: the
reactive teacher residual executed closed-loop scores **1.000**, yet a feedforward BC clone of it scores **0.792**.
Is that 0.79→1.0 gap due to **missing temporal/history information** (fixable by history/recurrence), or is it
**execution-hard** (a memoryless approximation error the task cannot tolerate)?

**Method.** Clone the reactive-teacher residual with four architectures, all executed as the *same* bounded residual
`clip(base + δ·r, lo, hi)`, differing ONLY in history access. Report **both** supervised residual-MSE **and**
closed-loop score (F-PP-009: low MSE ≠ high score). Reuses `_make_rnn`/`FlexLSTM` + the residual infra.

- **A. FF** — MLP(obs) → tanh (memoryless; reproduces the 0.792 reference)
- **B. History-K** — MLP(concat last K obs) → tanh, K∈{2,4,8}
- **C. GRU / LSTM** — RNN(obs sequence) → Linear → tanh (unbounded recurrent memory)

Targets: base 0.875, reactive-teacher oracle 1.000. Gate: median ≥0.875 → TD3+BC warm-start viable; ≥0.95 → gap
solved; ~0.79 or worse → not just history/recurrence.

## Single-seed pass (n_eval=24, seed0=20000)

| clone | residual-MSE | stable-place | place-err | exec-gap to expert | saturation | vs base |
|---|---|---|---|---|---|---|
| FF     | 0.00409 | 0.792 | 4.16 cm | 0.0909 | 0.005 | < base |
| Hist2  | 0.00399 | **0.875** | 3.55 cm | 0.0774 | 0.014 | = base |
| Hist4  | 0.00377 | 0.750 | 4.39 cm | 0.0797 | 0.010 | < base |
| Hist8  | 0.00362 | 0.833 | 4.61 cm | 0.0797 | 0.007 | < base |
| GRU    | 0.00727 | 0.667 | 3.78 cm | 0.0810 | 0.001 | < base |
| LSTM   | 0.00606 | 0.833 | 3.49 cm | 0.0743 | 0.001 | < base |

Three signals (single-seed, so treated as hypotheses until the multi-seed pass):
1. **No clone reaches 0.95** — the gap is not solved by any architecture tried.
2. **More memory is not monotonically better** — Hist4 (0.750) < Hist8 (0.833) < Hist2 (0.875); GRU worst (0.667). A
   genuine history deficit would improve with depth; it does not.
3. **MSE is decoupled from score** — Hist8 has the *lowest* MSE (0.00362) but 0.833; GRU has high MSE and low score.
   The residual is learnable; the closed-loop failure is not a fitting failure.

Together these point to **execution-hard**, not history-hard: a persistent ~0.08-rad executed-action gap to the
expert that the task cannot tolerate at critical (grasp/lift) moments, regardless of memory.

## Multi-seed pass (n_seeds=3, n_eval=24) — §3, single-seed is not a verdict

| clone | rate median | rate range | per-seed | err median | gate |
|---|---|---|---|---|---|
| FF     | 0.750 | [0.667, 0.792] | 0.667, 0.750, 0.792 | 4.16 cm | < base |
| Hist2  | 0.792 | [0.625, 0.833] | 0.625, 0.792, 0.833 | 4.92 cm | < base |
| Hist4  | **0.833** | [0.708, 0.833] | 0.833, 0.833, 0.708 | 3.78 cm | < base |
| Hist8  | 0.792 | [0.750, 0.875] | 0.875, 0.750, 0.792 | 3.78 cm | < base |
| LSTM   | 0.792 | [0.667, 0.875] | 0.667, 0.875, 0.792 | 5.25 cm | < base |

**No architecture reaches base (0.875) on the median.** The single-seed Hist2=0.875 was noise (multi-seed median
0.792). All five cluster at **0.75–0.83 median with heavily overlapping ranges** — history depth and recurrence are
statistically indistinguishable here, and none approaches 0.95. The best median is Hist4 at **0.833 < base**.

## Interpretation — execution-hard at the contact discontinuity, not history-hard

The failure-mode inspection (`scratchpad/clone_failure_inspect.py`) localizes it:

```text
(1) teacher residual by phase (oracle rollout):     |r| mean   step-jump |Δr|   MAX jump
      approach   n=1225                              0.060      0.022          0.440
      grasp      n=1597                              0.065      0.012          0.733   ← sharp spike at contact
      lift       n= 344                              0.095      0.027          0.561
(2) FF clone executed-gap to expert by phase:  approach=0.038   grasp=0.060 (largest)   lift=0.047
(3) FF clone failures: last contact phase = grasp (2/3, grabbed-then-dropped), approach (1/3, never grasped)
```

The teacher residual is small on average but has **near-full-range discontinuities at contact-mode switches**
(grasp max-jump 0.733). A smooth NN clone cannot reproduce those spikes, so its executed-action error is **largest at
grasp** (0.060 rad), and the dominant failure is **grasp-then-drop** — the clone grabs the box, then a small residual
error during the grasp/early-lift transition slips it, which is **irreversible**. The reactive oracle scores 1.000
only because it recomputes the exact residual (including the spike) from the live state every step.

This explains every observation: no clone reaches 0.95 (the spikes are un-clonable smoothly); more memory doesn't
help (the deficit is not temporal); MSE is decoupled from score (the average residual fits, the rare high-magnitude
contact spikes — which decide success — do not). **The 0.79→1.0 gap is execution-hard, not history-hard.**

One side signal worth noting (weak, single-seed): LSTM seed 1 reached **2.39 cm** place-error (base 4.69, expert
2.16) — recurrence may recover the expert's *precision* integral even where it does not lift the success *rate*
(consistent with the 2026-07-14 stateful-integral hypothesis). Precision is a different metric with headroom;
success-rate is not.

## Decision

- **Gate NOT met.** No history/recurrent clone reaches base 0.875 on the median (best 0.833), none reaches 0.95.
  Per your Step-2 condition ("only if the clone reaches or beats base"), **TD3+BC is NOT justified yet** and is not
  run. No kato15, no vanilla SAC.
- **Root cause identified:** the residual clone gap is execution-hard at the contact discontinuity — the teacher
  residual spikes (max-jump 0.733) at grasp, a smooth clone can't reproduce it, and the grasp-error is irreversible.
- **Why TD3+BC would not rescue it:** it would warm-start *below* base from a ~0.8 clone, hit the same contact-spike
  wall, and F-PP-009's critic over-estimation would drag it further off. The premise ("refine a clone that beats
  base") is false here.
- **Better-matched next levers** (candidates, not launched — for your call):
  1. **Mode-gated residual** (the residual env already supports `active_modes`): zero the residual during
     grasp/lift (let the base run the contact exactly) and learn it only in the smooth approach/place-align phases —
     the coin-toss "structured residual" regime; likely *preserves* base rather than beats it, but removes the
     grasp-spike failure the clone introduces.
  2. **Precision track**, not success track: if the LSTM's 2.39 cm precision replicates across seeds, target
     place-error (where headroom exists) rather than stable-place rate (where the base is near the learned ceiling).
  3. Accept the base (0.875) as the deployable policy; the oracle's 1.000 is a recompute-from-state controller, not
     a learnable one on this metric.

## Provenance
- Git: branch `integration/fanuc-pick-place-canonical`, audit checkpoint `6b90ca8`; base ckpt
  `experiments/hybrid_dagger_gif/policies/hybrid_dagger_hsikan_s0_best.pt`.
- Experiment: `scratchpad/residual_clone_gap.py` (reuses `pick_place_recurrent_clone._make_rnn`/`FlexLSTM` +
  `pick_place_residual_rl` residual infra); deterministic eval (fixed 24 tasks @ seed0=20000), divergence-guarded.
- Env: `fanuc_pick_env(expert_version=3, require_settle=True, max_steps=1000)`, δ=0.25/grip0.02, reward=settle.
- No CORE. No persistent state mutated. No kato15. No TD3+BC/SAC launched (gate-first).
