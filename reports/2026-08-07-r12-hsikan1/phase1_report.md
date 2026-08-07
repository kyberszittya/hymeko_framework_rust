# R12 / HSiKAN-1 — Phase 1: architecture ablation (offline ranking)

**Date:** 2026-08-08 · **Branch:** `feature/r11-4a-target-conditioned-delivery-teacher`
**Verdict:** `R12_HSIKAN1_STRUCTURE_DOES_NOT_BEAT_FLAT` (Phase-1, offline) — **REINFORCED at ~30k params after the
confound fix**: a stronger/bigger HSiKAN recovers the E2 AUROC collapse (0.51→0.67) but *still* does not beat flat on
the gate metric, and the unstructured controls tie or edge the task/Steiner incidence. **Learning breaks the retrieval
wall** (E1); cross-family (E2) top-1 selection remains the shared, information-bound frontier.

> **Two passes.** *Phase-1a* (below) = the original ~6k-param matched sweep. *Phase-1b* (§ "Confound fix") = the
> stronger-message-function / bigger-budget re-run that rules out the "HSiKAN was under-powered" confound. Read them
> together: 1b does not overturn 1a's structural null, it *strengthens* it by removing the capacity escape hatch.

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

## Confound fix (Phase-1b) — stronger message function + bigger budget

The Phase-1a null carried a legitimate confound: *maybe the HSiKAN was simply under-powered* (single shared-edge
mean-aggregation, ~6k params). Phase-1b removes it. The message function is now **mean‖max member pooling**, **2
propagation rounds** with residual node updates, and a **concat readout** over all 10 nodes (not a lossy mean-pool);
the budget is bumped to **~30k params**, all five models matched within a 1.03 ratio (MLP re-sized to hidden=110/
depth=4 = 29,151). Same data, split, optimizer, seeds. `ablation_sweep_v2.json`.

**E1 — unseen scenario** (oracle 0.86):

| model | AUROC (v1→v2) | top-1 K6 (v1→v2) |
|---|---|---|
| A0 MLP | 0.853 → 0.850 ± 0.023 | 0.643 → 0.586 ± 0.082 |
| A2 task-HSiKAN | 0.849 → **0.878 ± 0.009** | 0.729 → 0.643 ± 0.063 |
| A3 Steiner | 0.852 → 0.872 ± 0.016 | 0.700 → 0.600 ± 0.084 |
| A3c degree-matched | 0.845 → 0.868 ± 0.011 | 0.671 → 0.557 ± 0.082 |
| A1 random-sparse | 0.823 → 0.860 ± 0.011 | 0.686 → 0.614 ± 0.095 |

**E2 — unseen family** (oracle 0.77):

| model | AUROC (v1→v2) | top-1 K6 (v1→v2) |
|---|---|---|
| A0 MLP | 0.618 → 0.669 ± 0.018 | 0.318 → 0.236 ± 0.033 |
| A2 task-HSiKAN | 0.558 → 0.674 ± 0.037 | 0.191 → 0.245 ± 0.036 |
| A3 Steiner | 0.510 → 0.675 ± 0.014 | 0.209 → 0.218 ± 0.052 |
| A3c degree-matched | 0.530 → **0.690 ± 0.050** | 0.182 → **0.282 ± 0.044** |
| A1 random-sparse | 0.541 → 0.641 ± 0.049 | 0.236 → **0.291 ± 0.072** |

**What the fix settled:**

1. **The E2 near-chance collapse WAS under-powering — now corrected.** HSiKAN E2 AUROC rose from 0.51–0.56 (chance-ish)
   to 0.64–0.69, matching the MLP. Phase-1a's "structured models collapse to chance on an unseen family" is withdrawn:
   it was a capacity artifact, not a property of the incidence.
2. **Structure still does not beat flat — more decisively.** On the E2 gate (top-1 K6), the two *highest* scores are the
   **unstructured controls** (random-sparse 0.291, degree-matched 0.282) — above task-contact (0.245) and Steiner
   (0.218). Physical-contact structure provides no signal beyond generic sparse message-passing. On E1, task-HSiKAN
   edges the MLP on AUROC (0.878 vs 0.850, CIs nearly touching) but ties it on the gate (top-1 0.643 vs 0.586, wide
   overlap).
3. **Steiner ≈ degree-matched at both budgets** (E2 AUROC 0.675 vs 0.690; top-1 0.218 vs 0.282 — control *higher*). The
   pre-registered combinatorial-structure hypothesis is unsupported at ~6k and ~30k.
4. **Capacity bought ranking, not selection.** Top-1 K6 did *not* improve with size for anyone (MLP E2 0.318→0.236, E1
   0.643→0.586); only AUROC did. ~6k was already near-optimal for the deployment-relevant top-1 gate — the frontier is
   not model capacity.
5. **The remaining E2 ceiling is shared and information-bound.** Every model tops out at top-1 K6 ≈ 0.22–0.29 vs a 0.77
   oracle. A shared ceiling across flat *and* structured, *and* unmoved by 5× capacity, points to data coverage (3
   train families) / missing information (object orientation absent from the descriptor) — **not** to representation.
   Crucially, orientation is a *shared-input* lever: it would lift flat and structured alike, so it is not expected to
   revive the structure-vs-flat verdict. The ~2 h orientation regen is therefore **not** warranted for the structural
   question (it belongs to a coverage/Phase-2 attack, if pursued).

## Honest scope (not a first-pass over-claim)

This is a *clean but scoped* negative for structure: **this** node/edge feature mapping, tested at **~6k and ~30k
params / 80 epochs / 5 seeds**. Of the four Phase-1a confounds, **1b closes two**: (b) message-passing capacity —
now mean‖max + 2-round residual + concat readout; and (c) budget — now 30k, 5× larger. Neither revived structure over
flat. The two that **remain, both shared-input levers** (help flat and structured equally, so neither can revive the
structural verdict): (a) object *orientation* is absent from the 30-D descriptor — a real information gap, especially
for the box; (d) E2 difficulty may be a *data-coverage* limit (3 train families) not a representation limit. What **is**
robust across both budgets: learning >> retrieval on E1; no structured incidence beats flat *or* its own
degree-matched control on E2; and top-1 selection, not model capacity, is the ceiling.

## Next options (for user steer)

- **Phase 2 — closed-loop on the MLP.** Validate the learned critic *physically*: does its offline top-1 K6 (0.64 E1)
  hold when the picked θ is actually rolled out? The learned critic is a real advance over retrieval regardless of the
  structure null — worth confirming in the loop.
- ~~**Diagnose the structure null**~~ **— DONE (Phase-1b).** Stronger message function + 5× budget ruled out the
  "under-powered HSiKAN" confound; structure still does not beat flat. The null holds at ~30k params.
- **Attack E2 as a data-coverage problem:** more train families (O3 ellipse) so cross-family has ≥4 families to
  interpolate — the E2 top-1 ceiling looks coverage/information-bound (shared across all models, unmoved by capacity),
  not representation-bound. This is now the highest-value structural lever left, *if* the goal is E2.
- **Add object orientation to the descriptor** (~2 h regen): raises the shared E2 information ceiling for *all* models.
  A ceiling lever, not a structure lever — pursue only under a coverage/Phase-2 goal, not to re-litigate structure.

## Provenance

Env: Python 3.11.15, torch 2.12.0 (MPS), numpy 2.4.6, macOS (Apple Silicon), `OMP_NUM_THREADS=1`. Deterministic
(seeds 0–4). Harness `hymeko_rl/experiments/r12_hsikan1_ablation.py`, models
`hymeko_rl/coin_delivery/transportability_critic.py`. Phase-1a budget matched median 6097 (`ablation_sweep.json`);
Phase-1b matched 29,151–30,045, ratio 1.03 (`ablation_sweep_v2.json`, `sweep_v2.log`, ~17 min wall). Leakage 0 both.
