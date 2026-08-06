# Scaled refinement — temporal-form selection via a coverage P-graph (our hymeko_pgraph)

**Date:** 2026-07-13 · Aiko · branch `hymeko-neuro-migration` · Mac · **synthetic; no metaworld, no RL.** Extends
the conjunct-pruning reduction to the second refinement axis — *which temporal form* per signal (`F` / `G` /
late-window `G[0,4]`) — over a large candidate pool, via a **coverage** P-graph on our `hymeko_pgraph` crate.
Addresses the two threads "structural refinement beyond conjuncts" (1) and "scale the P-graph" (2). Code:
`hymeko_rl/eval/spec_bench/scale.py`.

> **Honest verdict up front.** Axis 1 (temporal-form selection) is **demonstrated**; axis 2 (scale) is now
> **demonstrated too** via a `cost = anti-F1` model so **ABB's branch-and-bound does the pruning** (measurable
> `explored`/`pruned` counts). The one recorded caveat: the conjunction task I first built does *not* isolate
> temporal value — `F(A AND B)` solves it, so flat pruning already hits 1.0 there; the single-signal task is the
> valid discriminator.

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

## Axis 2 — scale via ABB `cost = anti-F1`: DEMONSTRATED

The coverage model is genuinely combinatorial (SSG returns **3969** structures for 2 aspects × 6 variants). To make
**ABB's branch-and-bound** do the work rather than enumerate, each variant unit's `@U <unit> COST` is set to its
**anti-faithfulness** `1 − F1(calibrated variant, verif)`, so the cost-optimal structure is the most-faithful
variant per aspect. ABB (`solve --algorithm abb`) then finds it *and prunes*:

| pool | ABB result | test F1 | explored | pruned |
|---|---|--:|--:|--:|
| 1 aspect (6 variants) | `G[0,4](obj_to_target <= 0.14)` | 1.000 | 67 | 26 |
| 2 aspects (12 variants) | `(G[0,4](in_place>=0.665) AND G[0,4](obj_to_target<=0.157))` | 0.992 | 527 | 230 |

ABB agrees with SSG+F1-rank (correctness) but reaches it by **cost-bounding** — `pruned_by_inclusion` +
`pruned_by_reachability` are non-zero and grow with the pool (26/67 → 230/527), i.e. the bounding bites and its
advantage over enumeration widens with scale. (`refine_scaled_abb` in `scale.py`.)

## The conjunction task does NOT isolate temporal value (recorded so it is not misread)

I first built a conjunction task (`F(in_place>=0.9) AND G[0,4](obj_to_target<=0.1)`). On it the **flat**
conjunct-pruning reached **1.0** with `F(in_place>=0.854 AND obj_to_target<=0.1)` — "both hold *simultaneously* at
some point" separates the touch-then-drift negative *without* the late-window `G`. So that task is solvable without
temporal-form selection; the scaled machinery ties/loses there (0.99 vs flat 1.0). The single-signal task above is
the valid discriminator.

## Non-claims

- **Not** "the scaled P-graph beats flat pruning" — on the conjunction task flat wins; the scaled machinery's value
  is specifically temporal-form selection, shown only on the single-signal task.
- ABB's cost is a *proxy* (`1 − F1` of each variant in isolation) — it selects the best variant *per aspect*
  (separable) and matches SSG+F1-rank here; it is not a global-F1 optimum guarantee for interacting aspects.
- Synthetic; no RL; no real coffee-push (that is the next step).

## Changed / new files

`hymeko_rl/eval/spec_bench/scale.py` (new) · `hymeko_rl/tests/test_scale.py` (6 tests). Uses `target/debug/pgraph`.
6 tests green; ruff + mypy clean. **CORE.YAML untouched; no new deps.**

## Next

- **A `cost ≈ anti-F1` model** so ABB's axiom-bounding is genuinely exercised (the real "scale" claim).
- **Wire the arbitrated spec into CIP-RL-LiNGAM on real coffee-push** (kato15) — where the candidate pool and the
  temporal structure are non-toy and the P-graph earns its keep.
