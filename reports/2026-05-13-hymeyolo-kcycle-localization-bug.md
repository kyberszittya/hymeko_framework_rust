# HyMeYOLO `+kcycle` variant — localization bug diagnosis

**Date:** 2026-05-13
**Status:** Bug identified by code-reading; fix proposed; fix NOT yet implemented or tested (needs GPU smoke).
**Source artifact:** `reports/overnight_2026_05_11_stage7/hymeyolo_ricci_n5k_e50_s{0,4}.jsonl`

## Summary

The `+kcycle` variant of `KCycleHyMeYOLOMulti` reports **high classification
accuracy but poor localization** across both successfully-completed seeds:

| Seed | label | box_cls_acc | circ_cls_acc | mAP_50 | mAP_50_95 |
|---|---|---:|---:|---:|---:|
| 0 | +kcycle | **0.9375** | 0.7143 | **0.2403** | 0.0819 |
| 4 | +kcycle | **0.8333** | 0.7143 | **0.1596** | 0.0399 |
| 0 | boxes+circles (ref) | 0.7500 | 0.8571 | **0.7228** | 0.2206 |
| 4 | boxes+circles (ref) | 0.7778 | 0.6429 | **0.9230** | 0.3191 |

`+kcycle` ties or beats `boxes+circles` on **classification** but is **3-6×
worse on mAP50**. That is a structural-bug signature, not a
hyperparameter tuning issue: classification is *globally* correct on
which-class but the predicted boxes are *positioned wrong*.

## Root cause: `KCycleSignedAggregator` is not used for offset prediction

`signedkan_wip/src/vision/hymeyolo_kcycle.py:281-291` — the
`_refine_corners` method:

```python
def _refine_corners(self, base_corners, F_map, aggregator, head_offset):
    B = F_map.shape[0]
    N, K, _ = base_corners.shape
    flat = base_corners.unsqueeze(0).expand(B, -1, -1, -1).reshape(B, -1, 2)
    h_flat = bilinear_sample(F_map, flat)            # (B, N*K, d)
    h_corners = h_flat.view(B, N, K, -1)             # (B, N, K, d)
    h_query = h_corners.mean(dim=2)                  # (B, N, d) — coarse
    offsets = head_offset(h_query).view(B, N, K, 2)
    return base_corners.unsqueeze(0).expand(B, -1, -1, -1) + offsets
```

The `aggregator` parameter is **accepted but never called**. Offset
prediction reads a vanilla mean-pool over corner features, identical to
what `HyMeYOLOMulti` does without any signed-cycle structure.

The signed-cycle aggregator IS used downstream in `forward()`
(lines 308-313, 333-339), but only to build `box_cls_in` /
`circle_cls_in` for the **classification head**. The corner
coordinates themselves are already fixed by the time the aggregator
runs.

This is the entire architectural pitch of `+kcycle`: each query's K
corners form a signed sub-graph and the aggregator emits a
cycle-product descriptor that informs *both* what's in the region
(classification) and where the region is (localization). Right now it
informs only the former.

## Proposed fix

Refactor `_refine_corners` so the signed-cycle descriptor feeds offset
prediction:

```python
def _refine_corners(self, base_corners, F_map, aggregator, head_offset):
    B = F_map.shape[0]
    N, K, _ = base_corners.shape
    base_expanded = base_corners.unsqueeze(0).expand(B, -1, -1, -1)
    flat = base_expanded.reshape(B, -1, 2)
    h_flat = bilinear_sample(F_map, flat)
    h_corners = h_flat.view(B, N, K, -1)
    # Signed-cycle micro-graph descriptor per query, from base corners.
    corner_signs = corner_signs_from_corners(base_expanded)
    edge_signs = edge_signs_from_corner_signs(corner_signs)
    cycle_feats = torch.stack([
        aggregator(h_corners[:, qi], edge_signs[:, qi])
        for qi in range(N)
    ], dim=1)                                         # (B, N, d)
    # Use cycle-aware feature for offset prediction.
    offsets = head_offset(cycle_feats).view(B, N, K, 2)
    return base_expanded + offsets
```

Cost: one extra aggregator forward per query per backbone forward.
At `n_box_queries=4 + n_circle_queries=2` and `box_k=4 + circle_k=8`,
this is ~6 aggregator calls per image. For Cluttered MNIST at the
stage-7 budget (n=5000, epochs=50, batch=?) the extra wall is probably
single-digit %.

## Expected impact

If the structural pitch is correct and the bug is the only issue,
`+kcycle` mAP50 should recover from ~0.20 toward the
`boxes+circles` band (~0.72–0.92). If it doesn't, the structural
pitch needs revisiting — but at least the *fix* is doing what it claims
to be doing.

## Risks

- The fix could *worsen* mAP if `corner_signs_from_corners` on
  noisy / un-refined `base_corners` produces sign patterns that
  dominate the offset prediction with the wrong gradient direction.
  Mitigation: smoke at 1 seed × 1 variant before the 5-seed run.
- Per CLAUDE.md §3 production-scale smoke is required before any
  multi-seed re-run with the fix applied.
- The fix changes a model architecture; a pure-CPU unit test
  validating that `aggregator.alpha` gradient is non-zero after a
  forward+backward through `_refine_corners` is the right
  pre-smoke gate.

## Open items

1. **Implement the fix** in a single-file edit and add a unit test in
   `signedkan_wip/tests/test_circles_ricci.py` (or a new
   `test_kcycle_localization.py`) asserting:
   - `KCycleSignedAggregator` parameters receive non-zero gradient
     after a loss on `out["box_corners"]` (proves the aggregator is
     in the offset path).
   - Output corners differ from the no-aggregator mean-pool baseline
     (proves the structural signal is reaching localization).
2. **Smoke test** at 1 seed, n=5000, epochs=50.
3. **Multi-seed validation** (3 or 5 seeds) only after the smoke
   confirms no regression on `+kcycle` mAP50 vs the buggy
   baseline. If mAP50 lands in the 0.6+ band, queue full 5-seed.
4. **Aggregate report** comparing pre-fix vs post-fix `+kcycle`
   stats side by side with `boxes+circles` and `+ricci-mod`.

## Related work this session

- 5-seed backfill for the silently-killed seeds 1, 2, 3:
  `signedkan_wip/experiments/run_hymeyolo_ricci_seeds123_redo_2026_05_13.sh`
  — queued behind the Bitcoin 10-seed Optuna validation, uses
  `systemd-run --user -p MemoryMax=16G` (no `ulimit -v`,
  memory `feedback_ulimit_vs_cuda`).
- The redo will give us the **5-seed baseline of the buggy
  `+kcycle`** which we then compare against the fixed variant.

## CORE.YAML items touched

None. `signedkan_wip/src/vision/hymeyolo_kcycle.py` is non-core.
