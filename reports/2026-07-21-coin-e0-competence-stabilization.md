---
campaign: COIN E0 learned-delivery competence stabilization
title: LEARNED_DELIVERY_POSITIVE — a learned E0 policy reproducibly delivers clear-start coins up to +0.070 that BC-init cannot
date: 2026-07-21
branch: exp/coin-e0-competence-stabilization
source_commit: 7e6c624
classification: LEARNED_DELIVERY_POSITIVE (learned E0 policy reproducibly certified clear-start delivery from clearance >= +0.030, incl. +0.0698 > preferred)
---

# E0 learned Coin-Delivery competence stabilization (§1–§12)

**Created-at:** 2026-07-21 15:10 JST. Objective: preserve and stabilize the learned competence already observed
(static-BC SAC peaked 6/9, collapsed to 0/9 as α annealed). No embodiment / certificate / grasp change; no
contact/relay/critic/replay/n-step reopen; no hyperparameter grid.

## §2 Checkpoint audit (per-state, 10 deterministic restores) — no deployable checkpoint had survived
| checkpoint | headline states ≥8/10 | per-state |
|---|---|---|
| `bc_init.pt` (825cec8d) | **2/9** | 1011 10/10, 1447 10/10, rest 0/10 |
| `sac_actor_best.pt` (9293c0a5, gated run) | **2/9** | same 2 states — the gated "best" added nothing over BC |
| `sac_actor_final.pt` (9a21debc) | **0/9** | collapsed |

The genuine 6/9 improvement was the **static** run's peak, whose weights were **overwritten** (only its metrics
survive in `run_static_bc.json`). So no deployable learned-superior-to-BC checkpoint existed on disk → must regenerate.
Best historical per-state 10-run result available on disk: 1011 & 1447 at 10/10 (both also delivered by BC-init).

## §3 Collapse signature (from the static `train.log`)
Competence (dev/headline coverage) was **sustained while α ∈ [0.0367, 0.076]** (steps 6k–18k, evals 5–6/9), then
**decayed as α → 0.005** (steps 20k–28k) to a full 0/9 collapse. **α at the 6/9 peaks: 0.076 (6k) → 0.060 (10k) →
0.0367 (16k, last 6/9); bc_coef held at 1.0 throughout.** The collapse coincides with α reaching the 0.005 floor →
supports the hypothesis that annealing below the competence-associated entropy contributes to the actor collapse.
(The shared trainer's own docstring flags the same mode, F-SAC-8/9/10.)

## §4 Warm-up NaN handling (fixed + 3 regression tests)
`train.sac` now logs `crit=N/A`/`act=N/A` **before** the first update (was `nan` — logging-only, `last_c` init), and
**aborts** on a genuinely non-finite optimized loss (real divergence ≠ warm-up). Numerical path unchanged. Tests:
warm-up logs N/A never nan; healthy run stays finite; a poisoned critic aborts with a clear error. Existing SAC
compiled-update tests still pass (no regression).

## §5 COMPETENCE_PRESERVING_BC_SAC (over the existing canonical SAC — no new trainer)
- **Persistent demonstration anchor** — `bc_coef` held constant at **1.0** (the value that produced the 6/9 peak), via
  `bc_coef_fn=lambda _: 1.0`; never decays, contributes every actor update.
- **Entropy floor** — `alpha_final = 0.0367` (α at the last reproducible 6/9 pre-collapse checkpoint); ANNEAL holds
  there instead of →0.005. Not an invented floor.
- **Dev-bank competence checkpointing** — a hash-disjoint dev bank (8 states) never used for gradients; checkpoint
  selection by COIN_DELIVERY_STRICT dev coverage; **retain the best** (the run ends at its best checkpoint, §5.4).
- 3-way disjoint split (`state_split.json` 045a4864): **headline 9** (locked eval) / **dev 8** (selection) / **train 17**
  (gradients + demos).

## §6 Matched campaign — CONTROL (α→0.005) vs STABILIZED (α floor), seeds 0–3 × 2 reps (16 runs)
| arm | headline ≥8/10 (deployed best) | peak dev | **final dev** | **degrade (peak−final)** |
|---|---|---|---|---|
| CONTROL | [2,3,4,5,5,5,5,6] **med 5** | med 6 | med **2.5** | med **4** |
| STABILIZED | [0,2,2,2,5,5,5,5] **med 3.5** | med 7 | med **4** | med **3.5** |

**Honest attribution:** the α floor **reduces the final-step collapse** (final dev med 4 vs 2.5; less degradation; higher
peak) — the collapse hypothesis holds for the *final* policy. **But on the deployed (best-checkpoint) headline coverage
the two arms are equivalent within noise** (CONTROL med 5 ≥ STABILIZED med 3.5, both high-variance). **The deployable win
came from best-checkpoint retention + dev selection — which both arms use — not the α floor specifically.** I do not
claim the α floor as the hero the data does not support; it is a retention aid, not the source of the deployable policy.

## §7/§8 Learned-delivery criterion — MET and reproducible
Per-state ≥8/10 coverage across all 16 runs (learned checkpoints):

| state | clearance | # of 16 runs ≥8/10 | BC-init |
|---|---|---|---|
| 1174 / 1202 / 1278 | **+0.0698** (> preferred +0.060) | **5/16 each** | **0/10 (learned-only)** |
| 1164 / 1358 / 1568 | +0.0367 | 6/16 each | 0/10 (learned-only) |
| 1045 | +0.0386 | 2/16 | 0/10 (learned-only) |
| 1011 / 1447 | +0.033 | 13/16 each | also BC-init |

**7 of 9 headline states are LEARNED-ONLY.** The three **+0.0698** states (above the *preferred* +0.060) are delivered
**10/10 by the learned policy and 0/10 by BC-init**, and 15/16 runs produce a deployable checkpoint with ≥1 headline
state at ≥8/10. The §7 per-state criterion (oracle 10/10 ∧ zero 0/10 ∧ BC-init <8/10 ∧ learned ≥8/10, complete
trajectory learned actions, push/coast valid, no force closure) is **satisfied** — strongest at the +0.0698 band.

## §9 Deployed learned checkpoint + canonical verification
- **Checkpoint:** `experiments/2026_07_21_coin_e0_stabilize/learned_delivery_positive.pt` (sha256 **8bd73d8cbea0**).
- **Headline state:** seed **1174**, initial signed clearance **+0.0699**.
- **10-run comparison on 1174:** scripted oracle **10/10** · zero-action **0/10** · BC-init **0/10** · **learned 10/10**
  (grasp 0/10 — push/coast, not a grasp).
- **Canonical deterministic command** (restores state, asserts clearance, loads checkpoint, runs learned actions, prints
  COIN_DELIVERY_STRICT, renders, exits nonzero on failure — verified exit 0):
  ```
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. \
  python -m hymeko_rl.experiments.coin_delivery_e0_stabilize verify \
    --ckpt experiments/2026_07_21_coin_e0_stabilize/learned_delivery_positive.pt --seed 1174 \
    --render-dir reports/figures/2026-07-21-coin-delivery-e0
  ```
- **Videos** (`reports/figures/2026-07-21-coin-delivery-e0/`): `coin_delivery_learned_clear_start.gif` /
  `.mp4` / `_clean.mp4`, `coin_delivery_oracle_vs_learned.gif` / `.mp4` — same initial-state hash (seed 1174) both sides.
- Plots: `e0_stabilization_campaign.png` (per-state coverage + arm comparison), `e0_delivery_causal.png`.

## §10 Classification: **LEARNED_DELIVERY_POSITIVE**
A learned E0 policy reproducibly achieves clear-start certified Coin Delivery from clearance ≥+0.030 (indeed at
**+0.0698**, above the preferred +0.060), on states BC-init cannot deliver, across multiple seeds. Not merely
STABILIZATION_POSITIVE (individual states DO reach ≥8/10).

## §11 Grasp result preserved
COIN_GRASP_DELIVERY_STRICT remains **not achieved** by the tested embodiments (force closure) — relevant to
pick-and-place / humanoid, and untouched by this delivery campaign.

## §12 Provenance
- Frozen prior work: commits **009ba71**, **7e6c624** (unamended); reports `2026-07-21-coin-delivery-e0-learned.md`,
  `2026-07-21-coin-certify-before-release.md`; `experiments/2026_07_21_coin_e0_learned/`.
- New code: `hymeko_rl/experiments/coin_delivery_e0_stabilize.py` (COMPETENCE_PRESERVING_BC_SAC + verify);
  `hymeko_rl/train/sac.py` (warm-up N/A logging + non-finite abort); `hymeko_rl/tests/test_sac_warmup_nan_guard.py`.
  Reuses `coin_delivery_e0_campaign`, `delivery_certificate`, `train.sac`. No CORE.YAML. No deps.
- Data: `experiments/2026_07_21_coin_e0_stabilize/` — `state_split.json` (045a4864), `campaign.json` (a156a260),
  per-run `{control,stabilized}_s*_best.pt`, deployed `learned_delivery_positive.pt` (8bd73d8c), `validate/`.
- Tests: 15 pass (3 NaN-guard + 8 certificate + 4 compiled-update). ruff clean.
- Host Apple M5 Pro; threads pinned OMP/MKL/OPENBLAS=1; ~400 steps/s; RSS < 1 GB. RL not bit-reproducible (BLAS) —
  every claim rests on multi-seed matched runs + deterministic per-state 10-restore eval, per the discipline.

## Honest scope
The learned positive is solid and multi-seed. The stabilization *mechanism* claim is deliberately narrow: the α floor
demonstrably reduces final-step collapse, but best-checkpoint retention is what makes a deployable policy — so I report
the α floor as a retention aid, not the deployable cause. High run-to-run variance (0–6 headline states) is inherent to
this off-policy setup; the deployable artifact is the *retained best* checkpoint, verified per-state.
