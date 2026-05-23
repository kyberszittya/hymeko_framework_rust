# VOC2007 Hungarian-at-C9-recipe — architectural verdict

**Date:** 2026-05-22
**Plan:** [`docs/plans/2026-05-21-voc-d3-break-phase3/`](../docs/plans/2026-05-21-voc-d3-break-phase3/) (Phase 5 follows the same plan family)
**Verdict:** **Architectural claim survives.**  Applying the C9
recipe levers to the Hungarian head delivers no measurable lift
(0.0155 vs the unoptimised baseline 0.0153, +0.0002, 1.01×).  Nodelet
at the same recipe stays at **0.0790 ± 0.0105** (C9 5-seed).  Gap is
**0.0635 below the C9 band**, a **5.1× advantage** for the nodelet
head.  The K+1 softmax bottleneck is mechanically observable
(4 of 6 Hungarian cells with λ\_{no\_obj} ≥ 2.0 collapsed
``cls_acc → 0.000``).

## 1. Recipe vs. architecture decomposition

Two sweeps applied the *same* recipe levers (n_q, epochs, suppression
weight) to two heads:

| head | lever stack | best mAP$_{50}$ | over unoptimised |
|---|---|---:|---:|
| Nodelet (C9 5-seed) | n_q=6, 90 ep, λ_gate=2.0 | **0.0790** | **5.16×** |
| Hungarian (H1) | n_q=6, 90 ep, λ_no_obj=0.5 | 0.0155 | 1.01× |

**Conclusion:** the 5.16× lift on the nodelet side is **not** a recipe
artifact.  The same lever stack on Hungarian recovers essentially
nothing.  The architectural difference (per-query sigmoid gate vs.
softmax K+1 class) is the active mechanism.

## 2. Per-cell table

Phase 5 — six single-seed Hungarian cells at the C9 recipe with the
suppression weight (``--lam-no-obj``) swept:

| cell | n_q | ep | λ_no_obj | mAP$_{50}$ | mIoU | cls_acc | loss_end |
|---|---:|---:|---:|---:|---:|---:|---:|
| H1 | 6 | 90 | **0.5** | **0.0155** | 0.315 | **0.500** | 3.296 |
| H4 | **12** | 90 | 2.0 | 0.0139 | 0.245 | 0.125 | 4.046 |
| H6 | **4** | 90 | 2.0 | 0.0034 | 0.298 | 0.000 | 4.634 |
| H3 | 6 | 90 | **5.0** | 0.0007 | 0.239 | 0.000 | 5.702 |
| H5 | 6 | **60** | 2.0 | 0.0005 | 0.238 | 0.000 | 4.884 |
| H2 | 6 | 90 | 2.0 | 0.0004 | 0.245 | 0.000 | 4.923 |

**The `cls_acc` column is the smoking gun.**  Every Hungarian cell
with λ_no_obj ≥ 2.0 (H2/H3/H5/H6) shows **cls_acc = 0.000** — the
softmax forced every matched query into the "no-object" class.  H4
(n_q=12) partially absorbed the pressure via over-provisioning
(2 of 12 queries kept signal, hence 0.125 cls_acc).  Only H1
(λ_no_obj = 0.5, the lever the C9 sweep would have inherited as
*equivalent* to gate-balance) preserved cls_acc above zero.

## 3. Why the nodelet head escapes this trap

The nodelet head's per-query objectness gate is a **separate sigmoid**,
not a class within the cls softmax.  Increasing ``--lam-gate-neg``
applies gradient pressure to the *gate* without competing with the
*class* logits.  Hungarian's ``--lam-no-obj`` is structurally
different: K+1 classes share a single softmax, so any pressure on the
no-object slot drains probability mass from the K real classes.  This
is the mechanism the day-18 dossier diagnosed; Phase 5 is the
matched-recipe falsification test, and it passes.

## 4. Final summary

| metric | value |
|---|---|
| Best Hungarian @ C9 recipe (H1) | mAP$_{50}$ = 0.0155 |
| C9 nodelet 5-seed mean | mAP$_{50}$ = 0.0790 ± 0.0105 |
| Nodelet advantage | **+0.0635 (5.1×)** at matched recipe |
| Recipe-only lift on Hungarian | +0.0002 (1.01×, essentially zero) |
| Cells with cls_acc = 0 (Hungarian, λ ≥ 2.0) | 4 / 5 |

The 5.16× headline mAP lift from the 18-hour staged sweep
(D-3-bis → C8 → C9) is now firmly **architectural-and-recipe**, not
recipe alone.  The nodelet head is the load-bearing primitive.

## 5. Open follow-ups

- 5-seed of H1 (Hungarian at recipe) would tighten the paired
  comparison vs C9, but the single-seed gap (5.1×) is already
  outside any plausible n=5 noise band.
- Further nodelet lifts beyond C9 will come from the *next* axis
  (backbone size — input resolution is dead per Phase 4 OOM).
