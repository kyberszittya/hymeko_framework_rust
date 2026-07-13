# Scaled refinement — temporal-form selection via a coverage P-graph (our hymeko_pgraph)

**Date:** 2026-07-13 · Aiko · branch `hymeko-neuro-migration` · Mac · **synthetic; no metaworld, no RL.** Extends
the conjunct-pruning reduction to the second refinement axis — *which temporal form* per signal (`F` / `G` /
late-window `G[0,4]`) — over a large candidate pool, via a **coverage** P-graph on our `hymeko_pgraph` crate.
Addresses the two threads "structural refinement beyond conjuncts" (1) and "scale the P-graph" (2). Code:
`hymeko_rl/eval/spec_bench/scale.py`.

> **Honest verdict up front.** Axis 1 (temporal-form selection) is **demonstrated**; axis 2 (scale) is **partial**
> — the coverage model produces a genuinely large SSG space, but the search is SSG-enumerate + minimal-filter +
> F1-rank; **ABB's cost-bounding is not exercised** (F1 is not a native P-graph cost). And the conjunction task I
> first built does *not* isolate temporal value — `F(A AND B)` solves it, so flat pruning already hits 1.0 there.

## The reduction

Pool = `signals × {>=,<=} × {F, G, G[0,4]}` (large; covers temporal form). Coverage P-graph: each *aspect* (signal)
has its temporal-variant units as **alternative producers** of `<aspect>_ok`; `success` consumes every
`<aspect>_ok`. `hymeko_pgraph` SSG enumerates feasible structures; we keep the **minimal** ones (one variant per
aspect), calibrate, and F1-rank. `.hymeko` P-graph, solved via our crate's CLI — no external lib.

## Axis 1 — temporal-form selection: DEMONSTRATED

A single-signal task where temporal form is *decisive*: success = `G[0,4](obj_to_target<=0.1)` (settled over the
final steps); the **touch-then-drift** negative dips to target mid-episode (so `F(obj<=0.1)` is a false accept) but
drifts away by the end. No second signal, so `F(A AND B)` cannot rescue `F`.

| spec | test F1 |
|---|--:|
| `F(obj_to_target <= 0.1)` (accepts drift) | 0.667 |
| `G[0,4](obj_to_target <= 0.1)` | **1.000** |
| **scaled search picked** `G[0,4](obj_to_target <= 0.14)` | **1.000** |

The temporal-variant search over the coverage P-graph **selects the late-window `G`** from the pool and reaches the
ceiling — the structural refinement calibration and conjunct-pruning cannot do.

## Axis 2 — scale: PARTIAL (honest)

The coverage model *is* combinatorial: for 2 aspects × 6 variants, SSG returns **3969** feasible structures (all
subsets of the alternative producers). So the P-graph reduction genuinely scales, and the first naive version was
too slow (calibrating ~4000 supersets). The fix: filter to the **minimal** structures (one variant per aspect →
36), calibrate + F1-rank. **But this uses SSG-enumerate + filter, not ABB's cost-bounding** — because F1 is not a
P-graph cost. Genuinely exercising ABB's axiom-bounding needs a `cost ≈ anti-F1` model (a further step), and the
payoff is at a candidate pool far larger than a robotics monitor set.

## The conjunction task does NOT isolate temporal value (recorded so it is not misread)

I first built a conjunction task (`F(in_place>=0.9) AND G[0,4](obj_to_target<=0.1)`). On it the **flat**
conjunct-pruning reached **1.0** with `F(in_place>=0.854 AND obj_to_target<=0.1)` — "both hold *simultaneously* at
some point" separates the touch-then-drift negative *without* the late-window `G`. So that task is solvable without
temporal-form selection; the scaled machinery ties/loses there (0.99 vs flat 1.0). The single-signal task above is
the valid discriminator.

## Non-claims

- **Not** "the scaled P-graph beats flat pruning" — on the conjunction task flat wins; the scaled machinery's value
  is specifically temporal-form selection, shown only on the single-signal task.
- **ABB cost-bounding is not demonstrated** — SSG-enumerate + minimal-filter + F1-rank; ABB is future work.
- Synthetic; no RL; no real coffee-push.

## Changed / new files

`hymeko_rl/eval/spec_bench/scale.py` (new) · `hymeko_rl/tests/test_scale.py` (6 tests). Uses `target/debug/pgraph`.
6 tests green; ruff + mypy clean. **CORE.YAML untouched; no new deps.**

## Next

- **A `cost ≈ anti-F1` model** so ABB's axiom-bounding is genuinely exercised (the real "scale" claim).
- **Wire the arbitrated spec into CIP-RL-LiNGAM on real coffee-push** (kato15) — where the candidate pool and the
  temporal structure are non-toy and the P-graph earns its keep.
