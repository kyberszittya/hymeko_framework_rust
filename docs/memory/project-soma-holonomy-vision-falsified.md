---
name: project-soma-holonomy-vision-falsified
description: "Holonomy (sign-as-connection) aggregation was tested on Gömb-Soma MNIST vision (2026-06-29) and came back NEGATIVE; don't re-propose holonomy-revives-walk-vision"
metadata: 
  node_type: memory
  type: project
  originSessionId: 3b1d3a94-913b-4c0f-9bc1-e8c767463815
---

2026-06-29: re-tested whether the new StructuralActor / gauge result revives
Gömb-Soma walk-vision. The 2026-06-15 falsification used sign-as-**routing**
(dual `W±` banks + sign-blind sum-pool); the holonomy result says sign is a
**connection** (σ-product Z₂ holonomy, `M_v(σ⊙m)` = signed Bᴸ) — never run on
vision. Added a `HOLONOMY` aggregation mode to GömbSoma `HypergraphConv`
(`signedkan_wip/.../soma/hg_conv.py`, enum + Strategy dispatch, no CORE) and
re-ran the exact 3-arm MNIST A/B.

**Result: falsification CONFIRMED & tightened.** holonomy **0.4888 ± 0.0093**
(1226p) ≤ routing **0.5186 ± 0.0204** (2010p) ≪ linear **0.9056 ± 0.0079**. Both
anchors reproduced 2026-06-15 to 4 decimals (same-machine). Testing the *right*
operator gives the *same* answer — signed walk-holonomy is **not load-bearing
for MNIST patch-graph vision**.

**Holonomy-GROUP ablation (2026-06-29, follow-up):** user correctly noted the
above tested only **Z₂** (smallest group); holonomy generalizes to any group.
Lifted the same brightness connection Z₂→**U(1)** (magnetic Laplacian: per-edge
phase α·tanh(Δbright), walk holonomy Σθ rotates feature pairs), on the fair
flatten readout. **No group is the lever:** none 0.493 ≈ U(1) 0.510 ≈ routing
0.529 (within ~1σ); Z₂-as-connection 0.399 is *worst*. Discriminating check: the
U(1) flux α grew 1.0→1.114 (connection actively used, NOT collapsed to 0/none) —
so an *actively-learned* continuous holonomy still gives no benefit. So the
holonomy GROUP is not the lever; the readout (position) is. Don't re-propose
"a richer holonomy group will revive Soma vision" — Z₂ and U(1) both tested
negative. `reports/2026-06-29-soma-holonomy-group-ablation.md`.

**Why / How to apply:** the user's intuition (holonomy might fit gömb-soma) was
a *reasonable* bet — the old ablation genuinely tested the wrong operator — and
the re-test was the correct move; it just landed negative (cf.
[[feedback-user-intuition-is-calibrated]]: bring data to certify, this time it
disconfirmed). **Do NOT re-propose "holonomy revives walk-vision" or escalate to
RicciStim-holonomy / soma_vision.hymeko round-trip on this basis** — closed,
like [[project-rotor-joint-encoding-falsified]]. The `HOLONOMY` mode stays in
the code (tested, reusable via `HypergraphConvConfig(aggregation=…)`) but is not
default/promoted. Holonomy's value remains on the control/RL side
([[project-structural-actor-walk-holonomy]], [[project-gauge-holonomy-signed-hsikan]]),
not vision. Report: `reports/2026-06-29-soma-holonomy-aggregation.md`.
