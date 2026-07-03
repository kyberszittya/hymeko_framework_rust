# Holonomy-Group Ablation for Gömb-Soma Vision

**Date:** 2026-06-29
**Author:** Aiko (Claude Code) for Dr. Csaba Hajdu

## Motivation

The earlier vision holonomy test used only the smallest structure group, **Z₂**
(the brightness sign). The user's point: holonomy is defined for *any* group, so
Z₂ being falsified says nothing about richer connections. This ablation lifts the
*same* brightness connection from Z₂ to **U(1)** and compares, on the
position-preserving flatten readout (Phase 1 showed mean-pool caps every
connection, so only flatten gives the holonomy a fair test — the connection,
not the pooling, is the variable).

## Result (Cluttered-MNIST, canvas 48, flatten readout, 3 seeds × 5 ep × 3000)

| mode | connection | acc |
|---|---|---|
| none | unsigned pool (no connection) | 0.4933 ± 0.0205 |
| routing | Z₂ as a switch (dual sign banks) | **0.5293 ± 0.0408** |
| Z₂ | Z₂ as a connection (message × σ) | 0.3990 ± 0.0230 |
| **U(1)** | magnetic: phase α·tanh(Δbright), holonomy Σθ rotates feature pairs | 0.5103 ± 0.0231 |

Figure: `reports/figures/soma_holonomy_group_ablation_20260629.png`.

## Verdict — no holonomy group is the lever

- **U(1) ≈ none ≈ routing** (0.510 / 0.493 / 0.529), all within ~1σ. Lifting the
  group Z₂→U(1) unlocked nothing: a continuous connection does not beat
  no-connection.
- **Z₂-as-connection is the *worst*** (0.399) — the hard message×σ multiply
  actively hurts (consistent with the earlier MNIST holonomy result).
- **The U(1) connection is genuinely *used*, not ignored.** A discriminating
  check (the failure mode would be α→0, collapsing U(1) to none): the learned
  flux scale moved **α: 1.0 → 1.114** over training — it grew, the optimizer kept
  the phase rotation active. So U(1)≈none is *not* an "ignored-connection"
  artifact; an actively-learned continuous holonomy still gives no net benefit.
  (Measured: α trajectory on one seed; the 4-arm accuracies are the 3-seed means.)

## Interpretation

Across the whole Soma-vision investigation the lever has consistently been the
**readout (position)**, never the **connection**: sign-routing, Z₂-connection,
and now U(1)-magnetic all sit in the same band once the readout is fair, and the
discrete-sign connection is mildly harmful. The user's "try another holonomy" was
a well-motivated bet and the right thing to test; it came back negative — the
*group* of the connection is not what gates Gömb-Soma vision.

Caveat: absolute level (~0.49–0.53) is below the earlier `gomb_soma_flat` (0.617,
5000 train) because this harness uses 3000 train + a single-layer reimplementation;
the **relative** ranking across the 4 arms (same settings) is the result.

## Files
- `hymeko_neuro/models/hymeko_gomb/soma/vision/holonomy_walk.py` (new: Holonomy enum,
  `HolonomyWalkConv`, `HolonomyClassifier`; reuses `PatchGraphBuilder`).
- `hymeko_neuro/models/hymeko_gomb/soma/vision/train_mnist.py` (`holo_{none,routing,z2,u1}` arms).
- `hymeko_neuro/tests/test_holonomy_walk.py` (9 tests).
- jsonl `reports/soma_holonomy_group_ablation_20260629.jsonl`; figure as above.
- No CORE.YAML items; no new dependency.
