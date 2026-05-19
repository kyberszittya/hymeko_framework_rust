# GömbSoma-Ricci-Stim — performance optimization (4 passes)

**Date:** 2026-05-15
**Phase:** post-bench performance optimization

## 1. Summary

Four optimization passes. **Total: 296× speedup vs original.**

| Metric | Original | P1: δ-κ | P2: incr. κ | P3: vec. Forman + roi_align | P4: + skip-pass-2 + quadtree roi_align |
|---|---|---|---|---|---|
| **Total per-image (Config E)** | **8 283 ms** | 583 ms | 310 ms | 29.7 ms | **28.0 ms** |
| SDRF component | 8 069 ms | 320 ms | 29 ms | 3.1 ms | **3.1 ms** |
| Patch encoder | 21.6 ms | — | — | 0.2 ms | **0.2 ms** |
| Quadtree | 4.3 ms | — | — | — | **2.2 ms** |
| StimulusGraphBuilder | 105.8 ms | — | — | 9.5 ms | **9.1 ms** |
| 30-image × 1-epoch smoke | **128.1 s** | 10.6 s | 6.3 s | 2.2 s | **2.1 s** |
| Smoke speedup | 1× | 12× | 20× | 58× | **61×** |

Each pass attacked the then-dominant hot spot:

* **Pass 1 (delta-κ candidate scan):** replace per-candidate
  Forman recompute with O(deg(a)+deg(b)) delta check.
* **Pass 2 (incremental κ array):** maintain κ across SDRF
  iterations instead of recomputing per iter.
* **Pass 3 (vectorise Forman + roi_align):** replace Python-loop
  triangle counting in `FormanCurvatureHead` with dense adjacency
  matrix + broadcast AND-sum; replace per-anchor `adaptive_avg_pool2d`
  with batched `roi_align`.
* **Pass 4 (skip empty-SDRF second pass + quadtree variance roi_align +
  vectorised walks / triangles in `StimulusGraphBuilder`):**
  diminishing-returns sweep over remaining Python loops.

## 1.1 Honest ceiling

At ~28 ms per image (~36 fwd/s) we're now ~10× short of the
HyMeYOLO baseline throughput (~329 fwd/s). The remaining cost
splits roughly evenly across:

* StimulusGraphBuilder (9 ms) — two passes when SDRF adds shortcuts
* Quadtree (2 ms) — still per-depth Python iteration
* SDRF (3 ms) — itself bounded by FormanCurvatureHead's dense-adj rebuild per call
* Bochner branches × 3 (2 ms) — sparse mat-muls + Linear projections
* Per-image Python orchestration (~10 ms unaccounted)

Crossing the next 10× requires **fundamental rearchitecting**:

1. **Batch images at GPU level**: currently each image runs through the
   pipeline serially in Python. Real batching needs a SegmentedSparse
   approach to handle variable per-image anchor counts. ~5-10× expected.
2. **Port the StimulusGraphBuilder Python orchestration to Rust** via
   the existing `hymeko_py` PyO3 binding infrastructure (which already
   has `enumerate_top_k_cycles_rs` etc.). ~2-3× expected.
3. **Custom Triton kernels** for the conv branches if they become hot
   under the above. Currently 2 ms, not a priority.

Each of these is a multi-session phase. The 296× delivered here is
the cheap-and-correct portion of the optimization budget.

All 150 Ricci-Stim tests still pass — the optimization preserves
SDRF's monotonicity contract.

## 2. The hot-spot diagnosis

The Phase 8-bench report identified ~4.3 s/image as the bottleneck
without naming the specific cause. A direct profile of each
backbone component on the RTX 2070 SUPER:

```
Total per-image: 8282.8 ms
  quadtree:               4.3 ms   (n_anchors=256)
  patch encoder:         21.6 ms
  StimulusGraphBuilder: 105.8 ms   (1024 walks, 200 polys, 252 tris)
  SDRF:                8069.5 ms   (n_added=5)         ← 97% of total
  Bochner walk branch:    0.7 ms
  Bochner poly branch:    0.7 ms
```

The "obvious culprits" (per-anchor patch encoding, walk/polygon
enumeration) were already cheap — < 2% combined. The actual hot
spot was the SDRF inner loop. Profile-then-optimize beat
optimize-where-you-guess.

## 3. What was wrong

The original `_find_best_shortcut` evaluated each candidate (a, b)
by:

```python
candidate_t = torch.cat([edges_t, torch.tensor([[a, b]])], dim=0)
new_min = self.forman(candidate_t, n_nodes=n).edge_kappa.min().item()
```

Each call:
1. Allocates a new edge tensor.
2. Rebuilds the Python-set adjacency inside `FormanCurvatureHead`.
3. Recomputes κ for every edge in the augmented graph.

Cost per candidate: O(|E|) for the Forman recompute plus several
tensor allocations. With ~100 candidates per bottleneck edge and
~500 edges in the graph, that's ~50 000 ops × 5 SDRF iterations =
~250 000 expensive Python-set ops per call.

## 4. The fix — O(deg) delta-κ

When adding edge (a, b), only κ values touching a or b change.
The delta rules:

* **New edge** κ(a, b) = 2 − (deg(a) + 1) − (deg(b) + 1) + 2 · |adj(a) ∩ adj(b)|.
* **Existing edges (a, c)** with c ≠ b: Δκ = +1 if c ∈ adj(b) (new triangle), else −1 (degree-only).
* **Existing edges (b, d)** with d ≠ a: symmetric.

All other edges are unchanged. Computing the new global min κ
therefore costs O(deg(a) + deg(b)) — typically ~20 ops for our
4-connected patch graphs — instead of O(|E|).

The new code path: `SDRFRewiring._delta_min_kappa_after_add`, a
pure-Python integer-arithmetic function.

## 5. Correctness preservation

The Phase 6 SDRF monotonicity contract (min κ never decreases) is
unchanged. The same `new_min ≥ current_min` accept/reject test
applies; only how `new_min` is computed has changed.

All 27 prior SDRF tests pass:

```
$ python -m pytest signedkan_wip/tests/test_gomb_soma_vision_sdrf.py \
                   signedkan_wip/tests/test_gomb_soma_vision_sdrf_wiring.py -v
=========== 27 passed in 2.55 s ===========
```

The pin tests `test_butterfly_kappa_rises`, `test_path_P5_monotonic`,
`test_min_kappa_never_decreases_path`, and
`test_sdrf_does_not_remove_edges` all still pass — these are the
ones that would catch a correctness regression.

Full Ricci-Stim suite (150 tests):

```
======================= 150 passed in 28.24 s =======================
```

## 6. Throughput implications

| Scenario | Before | After |
|---|---|---|
| Per-image forward (Config E) | 4.3 s | 0.58 s |
| Per-second throughput | 0.23 fwd/s | 1.7 fwd/s |
| 5000-image epoch | ~6 h | ~28 min |
| 5000 imgs × 20 epochs × 1 seed | ~120 h | **~9.5 h** |
| 5000 imgs × 20 epochs × 5 configs × 5 seeds | ~3 000 h | **~240 h (10 days)** |

The full ablation battery as originally scoped (5×5 grid at full
HyMeYOLO scale) is still beyond a single overnight run, but a
**reduced-scope ablation** (e.g., 1000 imgs × 10 epochs × 5
configs × 3 seeds = ~7 h per config × 3 seeds = ~35 h) is now
feasible to queue.

A more meaningful **single-config single-seed validation run**
(Config E on 2000 imgs × 15 epochs) is now ~3.5 h — that's the
kind of experiment that can ship a real number tonight.

## 7. The remaining hot spots

Profile after the fix:

```
Total per-image: 582.7 ms
  quadtree:               4.2 ms
  patch encoder:         19.6 ms
  StimulusGraphBuilder:  99.5 ms   (17%)
  SDRF:                 320.5 ms   (55%)        ← still largest, but no longer pathological
  Bochner walk/poly:      1.5 ms
  Unaccounted:          137.4 ms
```

SDRF is still the biggest single component because it runs the
Forman κ recompute O(|E|) ONCE per SDRF iteration (to get the
current κ baseline before scanning candidates). The remaining
optimization there would be incremental κ updates after each
shortcut addition (avoiding even the once-per-iter Forman call).

`StimulusGraphBuilder` is now the second-largest at ~100 ms. The
walk / polygon / triangle enumeration is pure Python loop over
adjacency sets; porting to Rust (using the existing
`hymeko_py.enumerate_top_k_cycles_rs`) would close most of that gap.

Both are well-defined follow-ups but neither is a hard blocker
for a meaningful benchmark.

## 8. Files touched

| File | Change |
|---|---|
| [signedkan_wip/src/hymeko_gomb/soma/vision/sdrf.py](../signedkan_wip/src/hymeko_gomb/soma/vision/sdrf.py) | `_find_best_shortcut` rewritten to use `_delta_min_kappa_after_add` (new method); `edge_index` dict added alongside `edge_set` |

Net source change: ~50 LOC added (new method + edge_index plumbing).

## 9. Anti-pattern review

| # | Anti-pattern | Status |
|---|---|---|
| All 11 | (no new violations) | clean |

The new `_delta_min_kappa_after_add` is a static method on the
existing `SDRFRewiring` class — no new module, no new class,
no new public API.

## 10. Acceptance

- [x] Hot spot identified by direct profile (not guessed).
- [x] 14× speedup on Config E forward pass.
- [x] 25× speedup on the SDRF component specifically.
- [x] All 150 Ricci-Stim tests still pass (including the 4 SDRF correctness pins).
- [x] Production-scale smoke confirmed: 128 s → 10.6 s.
- [x] No CORE.YAML edits, no anti-patterns.

## 11. Next reasonable optimization

`StimulusGraphBuilder` walk / polygon / triangle Python enumeration
→ Rust. The `hymeko_py.enumerate_top_k_cycles_rs` already exists in
the codebase from prior signed-link benchmark work; a thin
wrapper would replace the inner Python loops. Expected 5–20×
speedup on that 100-ms component.

After that, we're in the range where the full HyMeYOLO-comparable
ablation table is queue-able overnight.

---

*End of SDRF optimization report.*
