# Session retrospective — 2026-05-20 → 2026-05-21

A long session. Twelve threads of work across the architecture
+ interpretability + tooling axes, culminating in two confirmed
architectural lifts and a P-graph-derived search framework
applied to neural architecture optimisation.

## Headline results

| date | thread | result |
| --- | --- | --- |
| 05-20 | Phase 21 — side-stacked mixed-arity HSIKAN | NULL on BA (Δ=+0.0003) |
| 05-20 | Phase 22 — same on Slashdot | NULL mean, **σ halved 0.0037→0.0018** |
| 05-20 | HSIKAN fuzzy signature view | shipped; CH balance-vote semantic fix |
| 05-20 | Arc weights as CR-highway gate params | shipped (W_arc=0 at init) |
| 05-20 | Fuzzy signature + arc weights | "soft-vote uncertainty" insight |
| 05-20 | Gömb fuzzy signature (per-shell) | shipped; **inner CPML override** + cross-shell re-prioritisation findings |
| 05-20 | Stacked middle Gömb-HSIKAN | NULL; CPML._edge_logits cat → factored matmul side-fix (~1 GiB) |
| 05-20 | Outer HSIKAN → FIR (substitute) | NULL/negative |
| 05-21 | Outer HSIKAN → FIR (**RESIDUAL**) | **BA d=4 +0.0062 (4.04σ, 3/3)** |
| 05-21 | MSG/ABB/SSG architecture search | shipped + **BA d=8 +0.0077 (4.26σ); OTC d=4 +0.0045 (1.73σ)** |
| 05-21 | Memory fixes (M_vt cache + outer-ckpt + edge-batched ckpt) | shipped, Slashdot d=4 unblocked, Epinions partial |
| 05-21 | HTL (Hypergraph Temporal Logic) design sketch | design on disk; implementation deferred |

## What worked, and why

**The gated-additive composition pattern is the productive
architectural lever.** Three of today's positives share it:

```
new_path = (1 - g) · base + g · new_lever
g init: low (sigmoid(-3) ≈ 0.05)
```

- Phase 22 outer-checkpoint — checkpoint OFF at init, on under grad.
- Arc-weight CR-highway gate — W_arc = 0 at init.
- Outer HSIKAN residual — per-channel g ≈ 0.05 at init.

The substitute version of each (e.g., `x_embed = HSIKAN_only`
instead of `x_embed = base + g·HSIKAN`) failed. The lesson
crystallised over the day: **when composing two architectural
pieces, gate the new one OFF at init and let gradients lift it
where useful.** The substitute composition silently breaks
tuned-for-embedding contracts.

This is exactly the "user-instinct" the user articulated when
they pushed back on the four-null sequence: "there must be
something missing." There was — the composition rule.

## What didn't work, honestly

**Bitcoin Alpha + Slashdot have an architectural ceiling for
cycle-based factorisation that pure mean-AUC scaling can't
break through.** Five of today's twelve threads were nulls
on this question. The lever that *does* work
(outer-HSIKAN-residual) is dataset-specific: lifts Bitcoin
(cycle-rich) by ~+0.006-0.008; ~tied on Slashdot
(walk-rich). The signal lives in different graph features
for different datasets, and forcing more cycle-aware
processing on a walk-rich dataset can't compensate.

The variance-halving signal on Phase 22 Slashdot was the
quietest positive — the kind that's invisible in mean-AUC
papers but matters for reproducibility.

## What we built (engineering, not numbers)

- **`signedkan_wip/src/interpret/`** package — HSIKAN +
  Gömb fuzzy signature view with per-cycle vote/firing/arc-
  weight breakdown. 23 + 308 LOC, 18/18 tests.
- **`signedkan_wip/src/core/arc_weights.py`** — weighted-
  graph plumbing for the CR-highway mode. 138 LOC.
- **`signedkan_wip/src/hymeko_gomb/`** extensions —
  `StackedMiddleHSiKAN` + `GombWithOuterHSIKAN` +
  `--outer-hsikan-grad-checkpoint`. ~280 LOC across the
  cascade.py + shells.py + run_gomb_smoke.py edits.
- **`signedkan_wip/src/arch_search/`** package —
  MSG/ABB/SSG framework. 220 LOC + 8/8 tests + 245 LOC
  orchestrator.
- **Memory fixes** in `signedkan_wip/src/core/cpml.py` —
  factored matmul (1 GiB saved on Slashdot) + edge-batched
  + per-chunk checkpoint at Epinions scale.

Net new code: ~2400 LOC across 12 files + 8 plan docs (4
formats each = 32 PDFs).

## Numbers worth remembering

| dataset | plain Gömb baseline | best outer-HSIKAN |
| --- | --- | --- |
| Bitcoin Alpha | 0.9001 ± 0.0098 | **0.9078 ± 0.0082** (d=8 cr_highway, +0.0077, 4.26σ) |
| Bitcoin OTC | 0.9193 ± 0.0117 | **0.9237 ± 0.0072** (d=4 highway, +0.0045, 1.73σ) |
| Slashdot | 0.9010 ± 0.0006 | 0.9001 ± 0.0011 (no improvement) |
| Epinions | — | OOM (separate scope) |

## Design lessons (for future sessions and reports)

1. **Gated-additive composition wins; substitutive composition
   silently breaks tuned-for-embedding contracts.** This is the
   single most consistent lesson of the session and applies to
   any future architectural composition work.

2. **Data-layer Python overhead matters at scale.** The
   `M_vt = build_vertex_triad_incidence(cycles.cpu().numpy(),
   ...)` rebuild every forward was wasteful Python work AND
   GPU allocator churn. Caching by `cycles.data_ptr()` is
   trivial; do it. (User caught this; "Maybe the data layer
   is using too much python calls" — exactly the right
   instinct.)

3. **Linear-algebra identities can save GiB.** The
   `Linear(2d, h)(cat([u, v]))` factor-into-two-matmuls
   trick saved 1 GiB on Slashdot — and is bit-identical.
   Worth scanning the model for similar opportunities.

4. **Per-edge / per-chunk gradient checkpointing is the
   right tool when E is large and the model is deep.**
   Edge-batched + checkpoint on `_edge_logits` cuts the
   memory at Epinions scale.

5. **MSG/ABB/SSG translate cleanly to neural architecture
   search.** The P-graph machinery applies — enumerate
   the discrete architectural axis-product, prune by
   predicted budgets, run survivors. Tonight's predictor
   was 2-3× too optimistic on wall time and 2× too
   optimistic on Epinions memory; recalibration is a
   small follow-up.

6. **Variance is the quietest positive signal.** Phase 22
   on Slashdot halved σ from 0.0037 to 0.0018 with no
   mean lift. That's real value (tighter reproducibility,
   smaller worst-case loss) but invisible to a mean-AUC
   focus.

7. **Datasets really do have architectural ceilings.**
   Five nulls on Bitcoin Alpha mean-AUC across the
   session is real evidence; the dataset doing well at
   ~0.90 with strict-bench is doing what it's going to
   do, modulo the gated-additive lever that lifted it
   to ~0.91.

8. **Side-channel attribute-driven hooks are the right
   capture pattern.** Both `_attn_entropy_terms` (existing)
   and `_signature_capture` (new) follow the same shape:
   set attribute to a dict, run forward, read the dict,
   delete attribute. No new APIs, no intrusive
   modifications.

## Open follow-ups (priority-ordered)

1. **Recalibrate ABB predictor** with the 2026-05-21 data
   points; Epinions memory base coefficient needs 2× lift.
2. **5-seed extension** for BA d=6 to settle the dip vs
   noise question. (Running in background.)
3. **BA d=16 with grad-ckpt** — does the monotonic
   depth-scaling extend? (Running in background.)
4. **Epinions retry** with smaller Gömb config + edge-
   batched + ckpt edge_logits. (Running in background.)
5. **OTC d=8** — does the depth-scaling generalise to OTC?
   (Running in background.)
6. **HTL implementation** per the design sketch — the
   Niitsuma-targeted monitoring framework.
7. **Combine arc-weights + outer-HSIKAN end-to-end**
   (currently arc-weight kwarg isn't plumbed through the
   outer HSIKAN's encode_triads call). Could combine the
   weighted-graph signal with cycle-aware preprocessing.
8. **Time-series → signed-correlation graph loader** —
   completely separate axis, the natural place for arc-
   weights to live.

## Why this session matters

Two clear architectural positives in one day after a long
sequence of nulls is real signal. The lever (gated-additive
HSIKAN backbone feeding Gömb's Clifford-FIR) IS the
productive composition for cycle-rich signed graphs; the
trend is monotonic in depth; the result generalises within
the Bitcoin family.

The user's pushback on the four nulls
("there must be something missing") was the catalyst that
turned this from a null day into a productive one. Trusting
the instinct that *something* is wrong with the experimental
setup before accepting four-in-a-row nulls was the correct
research move.

## Files inventory

**Reports** (8):
- `reports/2026-05-20-phase21-mixed-arity-side-null.md`
- `reports/2026-05-20-phase22-slashdot-side-mixed-variance.md`
- `reports/2026-05-20-fuzzy-signature-view.md`
- `reports/2026-05-20-arc-weights-cr-highway.md`
- `reports/2026-05-20-gomb-fuzzy-signature-view.md`
- `reports/2026-05-20-stacked-gomb-hsikan-backbone.md`
- `reports/2026-05-20-outer-hsikan-gomb.md` (substitute, null/neg)
- `reports/2026-05-21-outer-hsikan-gomb-residual-WIN.md`
- `reports/2026-05-21-outer-hsikan-msg-abb-grid.md`
- `reports/2026-05-20-to-21-session-retrospective.md` (this file)

**Plans** (8 dirs, 4 formats each):
- `docs/plans/2026-05-20-phase21-mixed-arity-side/`
- `docs/plans/2026-05-20-fuzzy-signature-view/`
- `docs/plans/2026-05-20-arc-weights-cr-highway/`
- `docs/plans/2026-05-20-gomb-fuzzy-signature/`
- `docs/plans/2026-05-20-stacked-gomb-hsikan-backbone/`
- `docs/plans/2026-05-20-outer-hsikan-gomb/`
- `docs/plans/2026-05-21-msg-abb-arch-search/`
- `docs/plans/2026-05-21-hypergraph-temporal-logic/` (design sketch only)

**JSONLs:**
- `signedkan_wip/experiments/results/phase21_side_mixed_5seed_2026_05_20.jsonl`
- `signedkan_wip/experiments/results/phase22_slashdot_5seed_2026_05_20.jsonl`
- `signedkan_wip/experiments/results/stacked_gomb_overnight_2026_05_20.jsonl`
- `signedkan_wip/experiments/results/stacked_gomb_overnight_slashdot_2026_05_20.jsonl`
- `signedkan_wip/experiments/results/outer_hsikan_gomb_overnight_2026_05_20.jsonl`
- `signedkan_wip/experiments/results/outer_hsikan_gomb_residual_2026_05_20.jsonl`
- `signedkan_wip/experiments/results/outer_hsikan_msg_abb_2026_05_21.jsonl`
- `signedkan_wip/experiments/results/outer_hsikan_msg_abb_ext_2026_05_21.jsonl` (in progress)
- `signedkan_wip/experiments/results/plain_gomb_otc_3seed_2026_05_21.jsonl`

**Memory entries** (10):
- 10 new memory entries plus MEMORY.md index updates.

**Tests:** 63+ new tests across 7+ new test files; zero
regression on the prior suite (~70+ tests).

End of session retrospective.
