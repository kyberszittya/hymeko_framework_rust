# R12 / HSiKAN-1 — Phase 1: architecture ablation (offline ranking)

**Date:** 2026-08-08 · **Branch:** `feature/r11-4a-target-conditioned-delivery-teacher`
**Verdict:** `STATIC_INCIDENT_TRANSPORTABILITY_RANKER_STRUCTURAL_ADVANTAGE_NOT_SUPPORTED` — the tested FIXED task-contact
and Steiner incidences give no advantage over a flat MLP (or their own matched sparse controls) on this ranker task, at
~6k **and** ~30k params, offline **and** physically. This is **R12.1** of a 6-rung ladder; it is **not** "structure axis
exhausted" — R12.2 (rotor/quaternion) … R12.6 (Steiner as actor–critic routing) are separate, *unmeasured* hypotheses
(see "R12 ladder — scope of this negative" below). Durable positives: **learning breaks the retrieval wall** (E1 learned
top-1 0.64 vs flat retrieval ~0), and the task-contact HSiKAN genuinely *uses* its incidence (T1) — it just buys no
deployment gain here.

> **Passes.** *Phase-1a* = the ~6k-param matched sweep. *Phase-1b* (§ "Confound fix") = the stronger-message / bigger-
> budget re-run that rules out the "under-powered HSiKAN" confound. *Closure checks* (§ "R12.1 closure checks") = T1
> scramble (structure-use), T2 orientation premise-probe, T3 physical closed-loop. Together they scoped-close R12.1
> without over-closing the ladder.

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

## R12.1 closure checks — scoped-close of the static ranker (T1/T2/T3)

Three bounded closure checks (per user directive) before banking the static-incidence ranker line — **no rescue
campaign** — separating "structure doesn't help" from "under-powered / this panel can't pose the question / the offline
metric isn't physical."

### T1 — structure-use (correct vs degree-preserving scrambled incidence) ✅ mandatory

Trained the structured HSiKANs with their intended incidence, then at inference swapped in degree-preserving scrambles
(same per-node degree + edge sizes, random grouping) with the SAME weights — a weight-preserving intervention on the
message-passing topology (shared edge/update fns, per-node encoders). 5 seeds × 8 scrambles. `scramble_test.json`.

| model | panel | correct AUROC | scrambled AUROC | Δ AUROC | Δ top-1 |
|---|---|---|---|---|---|
| task-HSiKAN | E1 | 0.876 | 0.847 | **+0.029 ± 0.014** | **+0.077 ± 0.062** |
| task-HSiKAN | E2 | 0.684 | 0.672 | +0.011 ± 0.022 | **+0.075 ± 0.047** |
| Steiner | E1 | 0.879 | 0.872 | +0.007 ± 0.004 | −0.016 ± 0.047 |
| Steiner | E2 | 0.685 | 0.674 | +0.010 ± 0.016 | +0.006 ± 0.067 |

**Finding:** the task-contact HSiKAN genuinely **USES** its physical incidence — E1 Δ AUROC +0.029 and Δ top-1 +0.077
both CI-exclude-0 vs scramble, and even E2's top-1 benefit (+0.075) excludes 0. The incidence is *load-bearing*, not
inert. **Steiner is largely inert** (Δ ≈ 0.007–0.010, top-1 ≈ 0) — consistent with Steiner ≈ degree-matched. The
more-informative reading the scramble test exists to draw: **the model reads the structure; it just buys no advantage
over flat** — NOT "structure is inert."

### T2 — orientation × architecture (premise probe) ⚠️ premise not met on this panel

Before measuring Δ_HSiKAN − Δ_MLP under an added orientation feature, the premise must hold: object yaw must vary across
the panel AND not be recoverable from the descriptor. Probed the object geom's world-frame yaw (joint-agnostic) across 4
families × 6 scenarios × 2 seeds. `orientation_probe.log`.

| family | yaw spread | reading |
|---|---|---|
| O0 coin | 0.36° | ~constant (rotationally symmetric) |
| O1-L size | 0.59° | ~constant |
| O2-M mass | 1.08° | barely varies |
| O4-S box | 1.87° | varies most — still tiny |

yaw ~ linear(coin-xy): R² = 0.117 → **not** recoverable from the descriptor (genuine new info). BUT certified straddle
grasps **pin the object orientation to a ≤1.9° band even for the box.** So orientation is genuine-new-info *and*
near-constant here. **The interaction is not well-posed on this panel** — with ~2° of variation both architectures gain
≈0, so a measured "Δ≈0" would be an *underpowered null*, not evidence about structure. Honest statement: **the R11.7B
handoff panel does not vary object orientation, so it cannot pose R12.2** (rotor/quaternion). This is precisely why
R12.1's null says nothing about R12.2 — the current symmetric-grasp benchmark literally cannot ask the question. It
needs orientation-*varying* handoffs (grasp-at-yaw / rotating transport), i.e. R12.2 proper, not a re-run here.

### T3 — physical closed-loop top-1 for ALL FOUR (frozen panel) ✅ offline = physical

Each of MLP / random-sparse / task-HSiKAN / Steiner ranks a handoff's pooled-θ candidates, picks its own top-1, and that
θ is ROLLED OUT physically (`_delivery_signals`, fresh deterministic sim) on the identical frozen held-out panel. The
dataset K6 labels are themselves physical rollouts, so a fresh roll must reproduce them — asserted as a reproduction
check. 3 seeds. `closed_loop.json`. **Reproduction 1.000 of 229 fresh rollouts (89 E1 + 140 E2); per-model
offline↔physical pick agreement 1.000** on every model, both panels.

| model | E1 phys top-1 K6 (oracle 0.86) | E2 phys top-1 K6 (oracle 0.77) |
|---|---|---|
| A0 MLP | 0.571 ± 0.081 | 0.242 ± 0.030 |
| A1 random-sparse | 0.643 ± 0.162 | **0.348 ± 0.030** |
| A2 task-HSiKAN | 0.643 ± 0.081 | 0.258 ± 0.030 |
| A3 Steiner | 0.619 ± 0.123 | 0.227 ± 0.089 |
| A3c degree-matched | 0.619 ± 0.047 | 0.303 ± 0.059 |

**Finding:** the offline top-1 K6 ordering survives physical rollout exactly — offline top-1 K6 **is** physical top-1 K6.
The 4-way comparison is a *physical* deployment result, not a proxy: on E1 all incidences sit within overlapping CIs of
the MLP; on the E2 gate the two highest are again the **unstructured** controls (random-sparse 0.348, degree-matched
0.303) over task-HSiKAN 0.258 / Steiner 0.227. No structured incidence beats flat or its own control, physically.

## R12 ladder — scope of this negative

R12.1 is one rung. What it closes: **a fixed task-contact / Steiner incidence gives no ranker advantage over flat on the
R11.7B transportability task.** What it does NOT touch (separate, physically-motivated, *unmeasured* hypotheses the
rotationally-symmetric coin could not even pose): **R12.2** orientation-aware geometric model `(x,v,R,ω)` — T2 shows the
current benchmark can't pose it; **R12.3** Rotor-Spike event-driven propagation; **R12.4** dynamic HyMeKo incidence
(structure that changes with the contact graph / hybrid mode — R12.1's negative is *specifically about fixed*
incidence); **R12.5** k-actor × n-critic tensor `Q_{a,c,p,h,μ,e}`; **R12.6** Steiner as actor↔critic↔mode↔spike routing
(NOT static ranker topology — where Kato-sensei's block-design idea actually lives). A negative on R12.1 must never be
written as closing R12.2–R12.6.

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

R12.1 is scoped-closed (verdict above); the closure checks are done (T1/T2/T3). The remaining moves are **rungs of the
ladder**, not attempts to save R12.1:

- ~~Diagnose the structure null~~ **DONE (Phase-1b + T1).** Under-powering ruled out; the task-HSiKAN provably uses its
  incidence (T1) yet ties flat. ~~Validate the critic physically~~ **DONE (T3)** — offline top-1 K6 *is* physical
  (reproduction 1.000, all 4 models). The learned critic beating retrieval on E1 is confirmed physical.
- **R12.2 — orientation-varying handoffs (the natural next rung).** T2 showed the current benchmark pins object
  orientation to ≤1.9°, so it cannot pose the rotor/quaternion hypothesis. To test it, generate handoffs that *vary*
  object yaw (grasp-at-yaw / rotating transport) and add `(R,ω)` to state; then the orientation×architecture
  interaction becomes well-posed. This is new physical work, not a re-run.
- **Attack E2 as data coverage** (O3 ellipse as a 4th train family) — the E2 top-1 ceiling is shared across all models
  and unmoved by 5× capacity, so it reads as coverage/information-bound. A ladder-agnostic lever that would help flat
  and structured alike.
- **Bank R12.1 and move up the ladder** (R12.3 Rotor-Spike / R12.4 dynamic HyMeKo incidence / R12.5 k-actor×n-critic) —
  each is a distinct hypothesis the static ranker cannot refute.

## Provenance

Env: Python 3.11.15, torch 2.12.0 (MPS), mujoco 3.10.0, numpy 2.4.6, macOS (Apple Silicon), `OMP_NUM_THREADS=1`.
Deterministic (seeds 0–4). Models `hymeko_rl/coin_delivery/transportability_critic.py` (unit tests
`hymeko_rl/tests/test_r12_transportability_critic.py`). Harnesses: ablation `…/r12_hsikan1_ablation.py`, scramble
`…/r12_hsikan1_scramble.py`, closed-loop `…/r12_hsikan1_closed_loop.py`, orientation probe
`…/r12_hsikan1_orientation_probe.py`. Phase-1a budget matched median 6097 (`ablation_sweep.json`); Phase-1b matched
29,151–30,045, ratio 1.03 (`ablation_sweep_v2.json`, ~17 min). Closure: `scramble_test.json`, `orientation_probe.log`,
`closed_loop.json` (T3 3-seed both panels, 39.2 min, reproduction 1.000/229). Leakage 0 all splits.
