# R12 / HSiKAN-1 — Phase 0: transportability dataset

**Date:** 2026-08-07 · **Branch:** `feature/r11-4a-target-conditioned-delivery-teacher`
**Verdict:** dataset ready — 8,484 pairs, 26.3% positive, rich cross-family transfer structure. Phase-1 architecture
ablation is unblocked.

## What it is

The train-only θ×handoff supervision the structured critic learns from: `(s_handoff, o, g, θ_i) → (K6, dtz, safe)`.
For each object family, handoff snapshots were re-acquired on a 6-scenario × 5-seed grid (certifying band, center
excluded) and the **pooled θ-bank** (101 θ, reused from existing teacher θ: box dense 66 + each family's
characterization θ) applied to each. Every row carries `handoff_family`, `theta_family`, `scenario`, `seed` — so the
split is **scenario-level** (a scenario's *all* θ-pairs move together; no same-handoff leakage) and cross-family
transfer is labelled. `dataset_{O0,O1-L,O2-M,O4-S}.jsonl`, meta `dataset_meta_ALL.json`.

## Stats

| family (handoffs) | rows | positive | rate |
|---|---|---|---|
| O0 coin | 2020 | 438 | 21.7% |
| O1-L size | 2323 | 813 | 35.0% |
| O2-M mass | 1919 | 557 | 29.0% |
| O4-S box | 2222 | 426 | 19.2% |
| **total** | **8484** | **2234** | **26.3%** |

Healthy balance — no rare-positive pathology; light class weighting suffices. 72.7 min (1.45 s/pair; the pooled-θ
refinement avoided ~5 h of per-family dense bank-gen). Deterministic (fixed seeds 0–4).

## Cross-family transfer matrix (positive rate, row = handoff family, col = θ family)

|  | θ:O0 | θ:O1-L | θ:O2-M | θ:O4-S |
|---|---|---|---|---|
| handoff O0 | **48** | 12 | 34 | 18 |
| handoff O1-L | 53 | 34 | 34 | 33 |
| handoff O2-M | 46 | 18 | 30 | 28 |
| handoff O4-S | 28 | 12 | 12 | 20 |

**Structure (provisional — the substrate the ablation tests):**
- **Coin θ (col O0) are broadly transferable** — 46–53% onto coin/size/mass handoffs, but only 28% onto the box.
- **Size-variant θ (col O1-L) are specialized** — 12–34%, transfer poorly off the bigger disk.
- **O1-L handoffs are forgiving** (row 33–53% — many θ transport); **box handoffs are demanding** (row 12–28%).
- The matrix is **asymmetric and interaction-structured** — a flat descriptor averages it; a structured handoff×θ
  model (task contact-hypergraph / Steiner HSiKAN) is what could exploit it. This is the benchmark's central bet,
  visible directly in the supervision.

## Split plan (Phase 1)

- **Train** — a subset of (family, scenario) cells.
- **E1 (unseen scenario, seen family)** — hold out ≥1 scenario across all families.
- **E2 (unseen object family)** — hold out a whole family's handoffs (the real generalization test).
- A leakage test asserts no (family, scenario, seed) handoff appears in two splits.

## Next — Phase 1 architecture ablation

A0 MLP · A1 random-sparse HSiKAN · A2 task contact-hypergraph HSiKAN · A3 Steiner/block-design HSiKAN, matched on
input/params/pairs/optimizer/split/compute. Gate on **closed-loop top-1 physical K6**, not AUROC; Steiner
pre-registered against the degree-matched-random control. Substantial build; plugs into the committed R12 plan.

## Provenance

Env: Python 3.11.15, mujoco 3.10.0, numpy 2.4.6, macOS (Apple Silicon), `OMP_NUM_THREADS=1`. Harness
`hymeko_rl/experiments/r12_hsikan1_dataset.py`. Pooled θ sources: `bank_dense.json` (66, O4-S) +
`characterization_{family}.json` (O0:10, O1-L:9, O2-M:7). Scenarios: `bank_c2_+0.015_+0.000`, `bank_c2_+0.025_+0.000`,
`bank_c3_r6_a+15`, `bank_c3_r7_a-15`, `bank_c2_+0.015_-0.015`, `bank_c1_+0.01_+0.00`.
