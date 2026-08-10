# R12.2-B — orientation-aware ranker: feasibility GO, but the ranker test is UNDERPOWERED

**Date:** 2026-08-10 · **Branch:** `feature/r11-4a-target-conditioned-delivery-teacher`
**Verdict:** `R12_2B_FEASIBILITY_GO_BUT_RANKER_UNDERPOWERED` — the physics has a real orientation×θ interaction
(feasibility GO), but the ranker test on the bounded orientation-varying dataset cannot resolve whether a structured
model exploits it: the critic is near chance, so the interaction Δ_task-HSiKAN − Δ_MLP is **inconclusive** (a power
limit — too few handoffs — NOT a "structure doesn't help" result).

**POWERED UPDATE (2026-08-10):** the acquisition WAS scaled (20 → **76 handoffs**, ~25 held-out), which resolved the
power problem: the critic now learns (**AUROC ~0.67**, was ~0.46–0.53 at chance) and the interaction CI halved. Powered
verdict — **with a sin-cos orientation encoding, explicit orientation gives NO model (flat or structured) a meaningful
ranking advantage** (Δ_AUROC(orient) ~0 for all; interaction Δ_task-HSiKAN − Δ_MLP = **+0.013 ± 0.045**, CI includes 0).
This is now a real (near-null) measurement, not an underpowered one. It is the clean **baseline for R12.3**: does a
quaternion/rotor orientation encoding beat sin-cos on the same powered dataset? See the "Powered test" section below.

## The chain

1. **Feasibility probe (GO, `e296918b`).** From certified handoff snapshots, a per-(yaw×target) CEM found delivering θ
   at all 4 yaws (dtz 5–17mm); a per-target yaw×yaw cross-transfer showed the delivering θ are **orientation-specific**
   (each yaw's θ delivers only at its yaw, fails at others — 28 such interactions; θ-spread 83% of the θ-box). The
   physics interaction R12.2 needs is real. (Fixed a v1 confound where the cross-transfer mixed target/position with
   yaw.)
2. **Orientation-aware θ bank.** Collected the CEM's top-k delivering θ per cell → **46 θ (42 K6), tuning-yaw
   {0:13,30:16,60:12,90:5}**. This resolves the pooled-bank degeneracy (which delivered only near yaw≈0).
3. **Orientation-varying dataset.** Applied the bank to certified handoffs (yaw{0,30,60,90}×3 targets×3 seeds),
   recording yaw → **920 pairs, 52 positives (5.7%)** from 20 certified handoffs. Non-degenerate (positives spread over
   yaw 0/30/60; sparse at yaw90), but **thin** — ~1/9 the R12.1 dataset, ~36 training positives.
4. **Ranker.** MLP / random-sparse / task-HSiKAN / Steiner / degree-matched, trained WITH vs WITHOUT the orientation
   feature ([sin,cos] yaw on the object_state node; incidence unchanged, R12.1 path bit-identical). 12 split-seeds ×
   80 ep.

## Ranker result (stratified handoff split, AUROC primary)

| model | AUROC without→with | Δ_AUROC(orient) | top-1 K6 Δ |
|---|---|---|---|
| A0 MLP | 0.526→0.545 | +0.019 ± 0.034 | −0.04 |
| A1 random-sparse | 0.507→0.479 | −0.028 ± 0.044 | −0.01 |
| A2 task-HSiKAN | 0.503→0.535 | +0.032 ± 0.060 | −0.00 |
| A3 Steiner | 0.484→0.512 | +0.028 ± 0.055 | +0.00 |
| A3c degree-matched | 0.516→0.578 | +0.062 ± 0.069 | −0.01 |

**Interaction Δ_task-HSiKAN − Δ_MLP = +0.013 ± 0.081 → INCONCLUSIVE.**

## Why underpowered (and a confound caught)

- **The critic barely learns.** Averaged over 12 seeds the without-orientation AUROC is ~0.50–0.53 — near chance. With
  only ~36 training positives (5.7% of 920), the 30k-param critic cannot robustly learn the handoff→delivering-θ
  mapping, so orientation's benefit is small/noisy and the structure-vs-flat interaction is buried. (A single-seed
  diagnostic hit AUROC 0.78 — a lucky draw, not the expectation.)
- **Split confound, fixed.** The first split held out a whole scenario/**target** for E1, but the θ are
  (yaw×target)-specific, so it measured θ-transfer-to-an-unseen-target (an R11.7B-style wall) → every model at chance,
  regardless of orientation. Switched to a **stratified handoff-level split** (yaws AND targets seen in training) so the
  test isolates the orientation-ranking question. Top-1 K6 is oracle-capped by the thin positives (oracle ~0.6), so
  AUROC is the informative metric.

## Honest verdict

The **feasibility GO is the durable result**: the rectangle's delivery has a genuine, orientation-specific θ structure
— the substrate R12.2 (and later the rotor/quaternion/Spike representations) needs. The **ranker interaction is not yet
resolvable**: at 52 positives the critic is near chance, so we cannot say whether a structured model exploits orientation
better than flat. Reporting "structure is a shared lever" here would be a false negative from an underpowered test.

## Widen-bank attempt (v3) — improved the dataset, did NOT power the ranker

Per the user's steer ("widen the bank first"), re-ran the CEM on **2 certified handoffs per (yaw×target) cell** (was 1),
keep-12 (was 8): bank **46 → 65 θ**, and the yaw90 coverage jumped **5 → 14** (well-balanced now, {0:18,30:16,60:17,
90:14}). Regenerated the dataset: **1300 pairs, 73 positives (+40% over v1's 52)**, yaw90 no longer starved.

Ranker on the v3 dataset (12 split-seeds): AUROC still **~0.44–0.49 (at/below chance)**; Δ_AUROC(orient) small/noisy
(+0.01..+0.05); interaction **Δ_task-HSiKAN − Δ_MLP = +0.034 ± 0.080 → still INCONCLUSIVE**. (A first re-run hit a
stale-file race — it read the v1 jsonl mid-write; corrected.) **Conclusion: widening the bank enriched the dataset but
did not lift the critic above chance.** 73 positives with ~3 delivering θ per handoff (of 65) is still ~30× below the
R12.1 scale (2234 positives, 26%) at which this critic learned (AUROC ~0.85). The K6 label is too sparse to learn a
handoff→θ ranker from, at bounded acquisition cost.

## Dense-signal ranker (dtz regression) — also underpowered

Per the user's steer, converted the ranker to regress the **dense normalized dtz** (present on all 1300 pairs) instead
of the sparse binary K6, ranking θ by predicted dtz. This gives the critic 1300 informative examples instead of 73.
Result (12 split-seeds): AUROC **~0.51–0.57** — marginally above the K6-classifier's chance but still near it; Δ_AUROC
(orient) noisy with mixed signs; interaction **Δ_task-HSiKAN − Δ_MLP = −0.063 ± 0.128 → still INCONCLUSIVE**.

**So both cheap levers — widen the bank (θ-density) and densify the target (dtz vs K6) — failed, for the SAME reason:
the bottleneck is not label sparsity, it is the number of HANDOFFS.** With only 20 certified handoffs (~14 train / 6
held-out), the held-out metric is too high-variance to resolve a Δ of order 0.03–0.06, whatever the target. (A
single-seed run can hit AUROC 0.78; the 12-seed mean sits near chance — that variance IS the diagnosis.) The cheap
in-place levers are exhausted.

## Verdict + next (decision, not assumed)

`R12_2B_FEASIBILITY_GO_BUT_RANKER_UNDERPOWERED` stands, now with the cause pinned: **too few handoffs**. The
durable result is the **feasibility GO** — the rectangle's delivery has genuine orientation-specific θ structure. The
modeling question (does a structured model exploit it better than flat) is **not resolvable at 20 handoffs** and no
θ-density / target-density trick fixes it.

- **R12.1-scale acquisition** — the only real path to a powered interaction test: many more handoffs (more scenarios ×
  seeds; the bank already spans yaws) → the held-out set is large enough to resolve Δ_HSiKAN − Δ_MLP. ~hours of
  acquisition (R12.1's 8484-pair scale was itself borderline).
- **Accept the feasibility GO** and defer the modeling/representation comparison (raw-yaw / sin-cos / quaternion / rotor
  — R12.3) to a dedicated larger effort. Clean, fully-committed state.

## Powered test (76 handoffs) — the resolving run

Per "mehet", scaled the acquisition on the diagnosed lever (handoffs). Made the dataset generator **checkpointed +
resumable** (per-handoff flush + an `attempted.tsv` sidecar) so the ~1 h run was safe, and **extended** the v3 base
(reused its 20 handoffs, added seeds 3–11 on the same 3 targets × 4 yaws with the existing 65-θ bank). Result:
**4940 pairs, 249 positives, 76 handoffs** (was 20), now yaw-balanced (yaw90: 5 → 27 handoffs, 50 positives).

Ranker (dtz-regression, 12 split-seeds, stratified handoff split, ~25 held-out):

| model | AUROC without→with | Δ_AUROC(orient) |
|---|---|---|
| A0 MLP | 0.669→0.667 | −0.002 ± 0.037 |
| A1 random-sparse | 0.656→0.666 | +0.010 ± 0.032 |
| A2 task-HSiKAN | 0.670→0.681 | +0.011 ± 0.020 |
| A3 Steiner | 0.669→0.656 | −0.013 ± 0.048 |
| A3c degree-matched | 0.653→0.659 | +0.006 ± 0.024 |

**Interaction Δ_task-HSiKAN − Δ_MLP = +0.013 ± 0.045** (CI includes 0).

**Findings:**
1. **Powering worked.** AUROC jumped from ~0.46–0.53 (chance, at 20 handoffs) to **~0.67** — the critic now learns the
   transportability ranking. The underpowering was, as diagnosed, purely handoff count. The interaction CI halved
   (±0.081 → ±0.045) as expected with ~4× the held-out units.
2. **A sin-cos orientation feature adds ~nothing** — Δ_AUROC(orient) ≈ 0 for *every* model. Since the flat MLP also gets
   sin-cos and also gains ~0, this is not a wiring failure of the HSiKAN's orientation node: the handoff descriptor
   (contact geometry) + θ already carry the predictive signal, so explicit sin-cos yaw is redundant.
3. **Structure ≈ flat on orientation** — the interaction is a tight near-zero. At this (now-learnable) scale, the
   physical contact-hypergraph extracts no more from orientation than a flat model. (Consistent with R12.1's static-
   ranker null — structure doesn't beat flat here either.)

**This is a genuine powered result, not an underpowered one.** It does NOT prove orientation is irrelevant to the
*physics* (the feasibility GO stands — orientation-specific θ exist); it shows a **sin-cos encoding + this structure**
don't turn that into ranker skill. That is exactly the **R12.3 baseline**: swap the orientation encoding to
**quaternion / rotor** on this same powered dataset (no new acquisition) and test whether a richer geometric
representation beats sin-cos. Now well-posed — the ranker is learnable and sin-cos is the measured baseline.

## Provenance

Env: Python 3.11.15, mujoco 3.10.0, torch 2.12.0 (MPS), numpy 2.4.6, macOS, `OMP_NUM_THREADS=1`. Deterministic.
`r12_2b_theta_feasibility.py` (GO, `b_theta_feasibility.json`), `r12_2b_orientation_bank.py` (`orientation_bank_O5-R.json`,
46 θ), `r12_2a_orientation_dataset.py` (`orientation_dataset_O5-R.jsonl`, 920 pairs), `r12_2b_ranker.py`
(`b_ranker.json`, 12 split-seeds). Critic orientation variant additive; R12.1 41-D path bit-identical (12 tests).
CORE.YAML touched: none.
