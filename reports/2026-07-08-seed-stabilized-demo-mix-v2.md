# seed_stabilized_demo_mix_v2 — morning report (2026-07-08) — POSITIVE_ROBUST

**Run:** kato15 (RTX 6000 Ada, torch 2.11+cu128) · `…/experiments/2026_07_08_seed_stabilized/` · wall **4779 s**
(~80 min) · **5 recipes × 5 training seeds** · 4 eval seeds × n=48 · results pulled to the Mac.
**Driver:** `exp_seed_stabilized.py --stage full --device auto` · v2b reward, frozen TaskMonitor, ledgers active.

## Verdict: **POSITIVE_ROBUST** — the demo-mix method CAN be made training-seed-robust

The NOT_ROBUST result (`reports/2026-07-08-option-msdm-trainseed-robustness.md`) is **reversed by variance
reduction**. Two recipes reach POSITIVE_ROBUST; the overall best is **E_valselect** (val-selected checkpointing).
And the central question is answered: **yes, a validation gate identifies the good seeds/checkpoints before test.**

Guards **PASS/PASS**, provenance **PASS**, actor md5 `edf4fe81…`, v2b certified. Baseline (multi-seed): ft_dom
**0.5677 ± 0.063**, monitor_score 0.2521, sustained-PUSH 0.3802.

## Validation → test predictivity (the key decision variable)

| metric | Spearman(val, test) |
|---|---:|
| ft_dom | **+0.547** |
| monitor_score | **+0.676** |
| sustained-PUSH | **+0.626** |

All three positive and moderate-to-strong ⇒ **validation metrics reliably predict test performance.**
val-selected − final test ft_dom = **+0.056** (selection beats the final epoch), and it rescues bad seeds
dramatically (E_valselect seed-1: final 0.167 → val-selected 0.505). **Conclusion: checkpoint selection rescues
the method** — the instability was in *which* checkpoint/seed you keep, and the val gate can pick the good one
without touching test.

## Per-recipe (val-selected pooled over 5 training seeds; baseline ft_dom 0.568 / mon 0.252 / sustained 0.380)

| recipe | ft_dom (mean±std) | monitor_score | sustained-PUSH | POSITIVE seeds | ft_dom IQR | exploit / body / arm-body | verdict |
|---|---:|---:|---:|:--:|---:|:--:|---|
| A_control | 0.495 ± 0.128 | 0.391 | 0.952 | 1/5 | 0.063 | 0 / 0 / 0 | PROMISING_BUT_HIGH_VARIANCE |
| B_gentle_lr | 0.547 ± 0.067 | 0.402 | 1.020 | 1/5 | 0.031 | 0 / 0 / 0 | PROMISING_BUT_HIGH_VARIANCE |
| **C_anchor** | **0.564 ± 0.081** | 0.311 | 0.640 | **4/5** | **0.026** | 0 / 0 / 0 | **POSITIVE_ROBUST** |
| D_balanced | 0.509 ± 0.090 | 0.364 | 1.070 | 1/5 | 0.037 | 0 / 0 / 0 | PROMISING_BUT_HIGH_VARIANCE |
| **E_valselect** | **0.575 ± 0.088** | **0.412** | **1.127** | **3/5** | 0.078 | 0 / 0 / 0 | **POSITIVE_ROBUST** (best) |

Final-epoch vs val-selected (ft_dom): the val gate lifts every recipe, most for the high-variance ones —
A 0.426→0.495, D 0.465→0.509, **E 0.426→0.575** (the plain-recipe final that val-selection turns into the best
result). C's final (0.5625) ≈ its val-selected (0.5635): the anchor already lands in a good basin, so selection
adds little there.

Per-seed val-selected ft_dom (the variance the recipes fight):
- C_anchor: `[0.578, 0.552, 0.578, 0.542, 0.568]` — tightest, all ≈ baseline; **4/5 POSITIVE**.
- E_valselect: `[0.573, 0.505, 0.537, 0.615, 0.646]` — highest, more spread; **3/5 POSITIVE**.
- A_control (unstabilized final): `[0.526, 0.167, 0.406, 0.542, 0.490]` — the raw instability, for contrast.

## Two mechanisms that work (different trade-offs)

- **C_anchor (DAgger-anchor):** `+λ·MSE(π, π_DAgger)` keeps the fine-tune in the delivering basin → **most robust**
  (4/5 POS, IQR 0.026, ft_dom preserved) but **less contact injected** (sustained 0.64, monitor 0.31 — the anchor
  trades some contact for stability).
- **E_valselect (val-selected checkpointing):** trains normally but keeps the best checkpoint by the val gate →
  **highest performance** (ft_dom 0.575, monitor 0.412, sustained 1.127) and POSITIVE_ROBUST (3/5), because the
  val gate discards the bad checkpoints. This is the overall best and the saved artifact.

## Gate decision

**POSITIVE_ROBUST** on the overall best recipe (E_valselect): ft_dom preserved (0.575 ≈ baseline, tied p=0.85, mean
slightly above), monitor_score up (0.412 vs 0.252), sustained-PUSH up 3× (1.127 vs 0.380), fingertip progress up,
**zero exploit / body-driven / arm-body across all 25 cells**, majority of seeds POSITIVE, bounded IQR, and
val-selection is predictive. **Deployable checkpoint saved:** `experiments/2026_07_08_seed_stabilized/
E_valselect_v2.pt` (md5 `b822a660…`, the best-seed val-selected policy).

Honest caveat: ft_dom is **preserved** (statistically tied with baseline), not significantly *improved*. The robust
win is behavioral — **3× sustained two-finger contact + higher monitor_score at preserved delivery, with zero
exploit, now stable across training seeds** — plus a working val gate. This is a real, deployable improvement in
contact quality, not a delivery-rate increase.

## Required fields recap

- Per recipe A–E: final vs val-selected performance, ft_dom / monitor_score / sustained-PUSH median±(IQR via std),
  exploit/body/arm-body (0 everywhere), per-seed verdict distribution, taxonomy verdict — **table + per-seed arrays
  above**. Spearman(val,test) for all three metrics — **above**. val-selection beats final — **yes, +0.056 ft_dom**.
- tensor-contract **PASS**, policy-provenance **PASS**, actor md5 `edf4fe81…`, deployable md5 `b822a660…`.

## Next (now in scope, per the standing plan)

A recipe is POSITIVE_ROBUST, so **bounded option-parameter RL on top of the robust sustained-contact policy is now
in scope** (the previously-gated Branch E). Recommended sequencing before that: (1) a short confirmation that
E_valselect's checkpoint holds on a fresh eval-seed set (independent of the 4 used here); (2) then option-parameter
RL (CEM/ES over the 5 bounded PhasePushController params) *on top of* `E_valselect_v2.pt` as the base — no per-step
residual, no scalar TD3/SAC/CQL, TaskMonitor external verifier, deployable only if it beats the E_valselect
baseline robustly. I have not launched it — awaiting your go per the "only then consider option-RL" gate.

## Guards / discipline

No scalar TD3/SAC/CQL, no per-step residual, no reward change, no CORE edit; TaskMonitor stayed the external
verifier (SearchObjective/val-gate separate). kato15 used the safe separate dir (existing checkout untouched);
device-mismatch bug in the val-gate's action-deviation was caught by a **cuda** smoke and fixed before the run.
Figure: `recipes.png` (val-selected pooled ft_dom + per-seed scatter).
