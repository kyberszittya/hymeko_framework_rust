# R11.7A U6B — Box-First Teacher-Free Retrieval Pilot (O4-S)

**Date:** 2026-08-06 (bank-gen completed 2026-08-07 00:57) · **Branch:** `feature/r11-4a-target-conditioned-delivery-teacher`
· **Milestone base:** `d52a74f9` (U6A). U6B on the same U6 commit chain.
**Verdict:** `R11_7A_BOX_RETRIEVAL_PILOT_FAIL` on the ≥1-dev-K6 criterion — but a **localized, provisional** negative:
the box is strongly *teacher-deliverable*; the bottleneck is *retrieval generalization*, exactly as for the coin.

---

## Scope

The flagship non-circular family (O4-S, the box — best capture in U6A at 5/6) run end-to-end through the unified
HyMeKo-generated pipeline: per-object teacher-θ bank (N=20 capture population × R=5 delivery restarts, top-1 certified
K6 θ per train scenario) → object-family-keyed standardized top-1 nearest retrieval (no blend/CEM/oracle/teacher) →
exact-zero evaluation on 2 DEV scenarios × 2 seeds, with the 8-class failure taxonomy. Box's own 8/2/2 split from the
certifying band (coin displacement 0.076–0.101, center excluded). O0 frozen retrieval remains the control (not re-run).
Sealed pilot-TEST (`bank_c2_+0.025_+0.015`, `bank_c1_+0.01_+0.02`) held — **not touched** (no dev freeze reached).

## Bank generation — the box is teacher-deliverable (strong)

7/8 train scenarios produced a certified K6 teacher θ (105.6 min, N=20/R=5). Aggregate over the 8 train scenarios:
**84 certified captures, 66 teacher-K6 deliveries.**

| train scenario | certified capture | teacher K6 | best dtz (mm) |
|---|---|---|---|
| `bank_c0_1` | 10/20 | 4 | 11.17 |
| `bank_c1_+0.01_+0.00` | 14/20 | 11 | 12.45 |
| `bank_c1_+0.01_+0.03` | 2/20 | 0 | — (far-high target) |
| `bank_c2_+0.015_+0.000` | 16/20 | 14 | **3.23** |
| `bank_c2_+0.015_-0.015` | 16/20 | 14 | 10.15 |
| `bank_c2_+0.025_+0.000` | 12/20 | 12 | 14.36 |
| `bank_c3_r6_a-30` | 6/20 | 4 | **3.90** |
| `bank_c3_r7_a-15` | 8/20 | 7 | **2.61** |

The box captures and the frozen teacher delivers it to the zone robustly (several ≤4 mm). Only `bank_c1_+0.01_+0.03`
(a far-high delivery target) yielded no K6 — an honest per-scenario delivery limit, not a pipeline fault. **This is the
core positive: the unification produces a working teacher pipeline for a non-circular object.**

## Evaluation — frozen retrieval does not close a held-out dev K6 (first pass)

Dev K6 = **0/4**. 0 model/contract failures. Taxonomy:

| dev scenario | seed | outcome | stage | dtz / support |
|---|---|---|---|---|
| `bank_c2_+0.015_+0.015` | 0 | `delivery_failure_in_support` | **delivery** | 29.96 mm / 5.98 (in-support) |
| `bank_c2_+0.015_+0.015` | 1 | `precontact_handoff_invalid` | **capture** | — |
| `bank_c3_r6_a+15` | 0 | `capture_no_certified_grasp` | **capture** | — |
| `bank_c3_r6_a+15` | 1 | `delivery_failure_in_support` | **delivery** | 56.67 mm / 6.05 (in-support) |

Clean split: **2/4 fail at the CAPTURE stage** (the exact-zero capture didn't certify at that specific dev seed — the
box certifies ~52% overall, so a 2-seed eval catches non-certifying seeds), and **2/4 reach delivery but the frozen
top-1 retrieved θ mis-transports IN-SUPPORT** (30 mm and 57 mm from zone; the θ is in the right ballpark but the wrong
basin). Retrieval was therefore *fairly* tested on only 2 rollouts, and both were near-but-not-K6 delivery misses.

## Interpretation — mirrors the R11.6C coin retrieval ceiling

The box shows the **same pattern as the coin**: the teacher delivers (bank 7/8, 66 K6), but frozen retrieval does not
generalize to held-out scenarios (R11.6C: coin retrieval ceiling held .417, TEST=wall). The bottleneck is
**retrieval generalization + capture-seed consistency**, *not* the object or the generated pipeline. So the unification's
promise holds — a non-circular object flows through generation→capture→teacher exactly like the coin — and it inherits
the coin's *known* open problem (retrieval), rather than a new object-specific one.

**This is a first-pass, provisional negative** (per the repo's "no conclusions from first-pass" rule): the dev panel is
thin (2 scenarios × 2 seeds; only 2 rollouts reached the delivery stage). It does **not** establish "the box cannot be
delivered teacher-free" — the teacher delivers it 7/8. It establishes that *frozen top-1 nearest retrieval, on 2 held-out
dev scenarios, did not close K6 in this bounded configuration.*

## Honest next levers (not run — for user scoping)

- **More eval seeds per dev scenario** (deploy baseline unchanged, top-1 nearest): 2/4 dev rollouts never reached
  delivery due to capture-seed inconsistency; more seeds would give retrieval a fairer test.
- **Retrieval-config diagnostic ablation** (k=3 distance-weighted, the coin's best held-out rule) vs the top-1 deploy —
  cheap (re-eval only, bank frozen); characterizes whether the box's retrieval ceiling can clear a dev K6.
- **Escalate bank N=40/R=11** on the certifying-band scenarios — only marginally relevant (dev failures are retrieval/
  capture-consistency, not a bank sampling gap), but would densify the retrieval table.
- The far-high delivery target (`bank_c1_+0.01_+0.03`) is a documented per-scenario delivery limit.

## Gate

`R11_7A_BOX_RETRIEVAL_PILOT` requires (box slice): bank non-empty ✓ (7), certified capture > 0 ✓ (84), teacher K6 > 0 ✓
(66), 0 model/contract fault in eval ✓, **≥1 dev exact-zero strict-K6 ✗ (0/4)** ⇒ **FAIL** on the flagship criterion.
The generation/capture/teacher half passes decisively; the frozen-retrieval half does not (first pass).

## Provenance

Env: Python 3.11.15, mujoco 3.10.0, numpy 2.4.6, macOS Darwin (Apple Silicon), venv `hymeko_framework_rust/.venv`,
`OMP_NUM_THREADS=1`. Bank-gen 105.6 min (peak RSS ≈ 306 MB); eval peak RSS 310 MB. Deterministic (fixed seeds). Cost
probe measured 38.6 s/seed (`hymeko_rl/experiments/r11_7a_u6b_cost_probe.py`). Artifacts: `bank.json`, `eval.json`.
Sealed pilot-TEST untouched.
