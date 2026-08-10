# R12.2-B — orientation-aware ranker: feasibility GO, but the ranker test is UNDERPOWERED

**Date:** 2026-08-10 · **Branch:** `feature/r11-4a-target-conditioned-delivery-teacher`
**Verdict:** `R12_2B_FEASIBILITY_GO_BUT_RANKER_UNDERPOWERED` — the physics has a real orientation×θ interaction
(feasibility GO), but the ranker test on the bounded orientation-varying dataset cannot resolve whether a structured
model exploits it: the critic is near chance on the thin data, so the interaction Δ_task-HSiKAN − Δ_MLP = +0.013±0.081
is **inconclusive**. This is a power limit, NOT a "structure doesn't help" result.

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

## Next (decision, not assumed)

- **Dense-signal ranker (cheap, recommended first).** Train the critic to regress **dtz** (how close each θ gets — a
  DENSE target present on all 1300 pairs) instead of the sparse binary K6, then rank θ by predicted dtz. This gives the
  critic 1300 informative examples instead of 73, so it can plausibly learn above chance and make Δ_HSiKAN − Δ_MLP
  resolvable — WITHOUT more acquisition. ~20 min (ranker change + re-run). Directly attacks the sparsity root cause.
- **R12.1-scale acquisition** (many more handoffs → thousands of positives): the brute-force powered test, but ~hours
  of acquisition (R12.1's 8484 pairs was itself borderline).
- **Accept the feasibility GO** and defer the modeling/representation comparison to a dedicated larger run.

## Provenance

Env: Python 3.11.15, mujoco 3.10.0, torch 2.12.0 (MPS), numpy 2.4.6, macOS, `OMP_NUM_THREADS=1`. Deterministic.
`r12_2b_theta_feasibility.py` (GO, `b_theta_feasibility.json`), `r12_2b_orientation_bank.py` (`orientation_bank_O5-R.json`,
46 θ), `r12_2a_orientation_dataset.py` (`orientation_dataset_O5-R.jsonl`, 920 pairs), `r12_2b_ranker.py`
(`b_ranker.json`, 12 split-seeds). Critic orientation variant additive; R12.1 41-D path bit-identical (12 tests).
CORE.YAML touched: none.
