# R12 / HSiKAN-1 — Phase 1: architecture ablation (offline ranking)

**Date:** 2026-08-08 · **Branch:** `feature/r11-4a-target-conditioned-delivery-teacher`
**Verdict:** `R12_HSIKAN1_STRUCTURE_DOES_NOT_BEAT_FLAT` (Phase-1, offline, this design/scale) — **but learning breaks the
retrieval wall**, and cross-family generalization (E2) is the identified frontier.

## Setup

Five models at **matched budget** (6097–6241 params, ±20%): A0 flat MLP; A1 random-sparse / A2 task-contact-hypergraph
/ A3 Steiner-block-design / A3c degree-matched HSiKAN (shared edge function → param count independent of the
incidence, so only *structure* varies). Trained on the 8,484-pair transportability dataset (class-weighted BCE, 80
epochs, 5 seeds), scenario-level split with a **leakage assertion** (0 leaked handoffs). Metric that matters: per-handoff
**top-1 K6** (rank the handoff's candidate θ by predicted P, take top-1, read its dataset K6) — the offline proxy for
the closed-loop physical gate — plus AUROC + oracle regret. `ablation_sweep.json`.

## Results (mean ± 95% CI)

**E1 — unseen scenario, seen family** (oracle top-1 K6 = 0.86):

| model | AUROC | top-1 K6 | regret |
|---|---|---|---|
| A0 MLP | 0.853 ± 0.008 | 0.643 ± 0.044 | 0.214 |
| A1 random-sparse | 0.823 ± 0.038 | 0.686 ± 0.056 | 0.171 |
| A2 task-HSiKAN | 0.849 ± 0.015 | **0.729 ± 0.052** | 0.129 |
| A3 Steiner-HSiKAN | 0.852 ± 0.025 | 0.700 ± 0.082 | 0.157 |
| A3c degree-matched | 0.845 ± 0.013 | 0.671 ± 0.095 | 0.186 |

**E2 — unseen object family** (oracle top-1 K6 = 0.77):

| model | AUROC | top-1 K6 | regret |
|---|---|---|---|
| **A0 MLP** | **0.618 ± 0.044** | **0.318 ± 0.049** | 0.455 |
| A1 random-sparse | 0.541 ± 0.049 | 0.236 ± 0.103 | 0.536 |
| A2 task-HSiKAN | 0.558 ± 0.049 | 0.191 ± 0.077 | 0.582 |
| A3 Steiner-HSiKAN | 0.510 ± 0.025 | 0.209 ± 0.045 | 0.564 |
| A3c degree-matched | 0.530 ± 0.030 | 0.182 ± 0.098 | 0.591 |

## Findings

1. **Learning breaks the retrieval wall (E1).** Every learned critic reaches top-1 K6 0.64–0.73 on unseen scenarios,
   where flat descriptor-nearest *retrieval* got ~0 (R11.7B). A learned `(handoff,θ)→K6` critic recovers
   transportability that amortized retrieval could not. **This is the real R12 advance.**
2. **Structure does NOT beat flat (gate: FAIL).** On E1 task-HSiKAN edges the MLP on top-1 K6 (0.729 vs 0.643) but
   near-overlapping CIs → marginal. On **E2 (unseen family — the real generalization test) the flat MLP wins outright**
   (AUROC 0.618, top-1 K6 0.318); every HSiKAN variant is worse and near chance (AUROC 0.51–0.56). Per the plan's gate
   (a structured model must beat MLP *and* random-sparse on top-1 K6), the structural hypothesis fails at Phase 1.
3. **Steiner ≠ its degree-matched control** (E2: 0.209 vs 0.182; E1: 0.700 vs 0.671, overlapping). No balanced-
   combinatorial-structure win — the pre-registered Steiner hypothesis is not supported here.
4. **Cross-family generalization (E2) is the frontier, hard for everyone.** The best model (MLP) manages only 0.318
   top-1 K6 vs a 0.77 oracle (regret 0.455); AUROC collapses from ~0.85 (E1) to ~0.6 (E2). Structure was expected to
   help *most* here and helped *least*.

## Honest scope (not a first-pass over-claim)

This is a *clean but scoped* negative for structure: **this** node/edge feature mapping, **this** simple hypergraph
message-passing, at **~6k params / 80 epochs / 5 seeds**. It does not prove structure *cannot* help — plausible
confounds: (a) the node features may under-represent contact geometry (e.g. object *orientation* is not explicitly in
the 30-D descriptor — a real gap for the box); (b) the message-passing is a single shared-edge mean-aggregation, not a
genuine KAN or attention; (c) 6k params may be too small to express the interaction; (d) E2's difficulty may be a
*data-coverage* limit (3 train families) not a representation limit. What **is** robust: learning >> retrieval on E1,
and no structured model beats flat on E2 at matched budget.

## Next options (for user steer)

- **Phase 2 — closed-loop on the MLP.** Validate the learned critic *physically*: does its offline top-1 K6 (0.64 E1)
  hold when the picked θ is actually rolled out? The learned critic is a real advance over retrieval regardless of the
  structure null — worth confirming in the loop.
- **Diagnose the structure null** before concluding: add object orientation to the node features / a stronger message
  function / larger budget, re-run E2. Distinguishes "structure doesn't help" from "this HSiKAN is under-powered."
- **Attack E2 as a data-coverage problem:** more train families (O3 ellipse) so cross-family has ≥4 families to
  interpolate — the E2 collapse may be 3-family coverage, not representation.

## Provenance

Env: Python 3.11.15, torch 2.12.0 (MPS), numpy 2.4.6, macOS (Apple Silicon). Deterministic (seeds 0–4). Harness
`hymeko_rl/experiments/r12_hsikan1_ablation.py`, models `hymeko_rl/coin_delivery/transportability_critic.py`. Budget
matched (median 6097, ±20%). Leakage 0.
