# SSG-walk skyline — Kung–Luccio–Preparata sort-and-sweep (D=2)

Date: 2026-06-04
Plan: `docs/plans/2026-06-04-klp-skyline/plan.{tex,pdf,tikz,mmd}`
Implementation: `hymeko_neuro/hyperedge/abb_walks.py`
                (`_ssg_pareto_filter_sweep_2d`, `_ssg_pareto_filter_brute`)
Tests: `hymeko_neuro/tests/test_ssg_pareto_filter.py` — **31/31 PASSED**

## 0. Headline

Replaced the brute O(N²·D) Pareto filter in
`ssg_pareto_filter` with a sort-and-sweep O(N log N) path at the
practical D=2 case. **247× empirical speedup at N=10,000** with
bit-for-bit equivalent output. The public API is unchanged; D ≥ 3
still falls back to brute.

## 1. Algorithm

For D = 2 (the practical HSiKAN case — score vs cost/walk_len):

1. Sort points by `a_0` desc; tie-break `a_1` desc.
2. Walk in `a_0`-groups (one group per distinct `a_0`).
3. For each group, compute `group_max_a1`. The points with
   `a_1 == group_max_a1` are on the skyline iff
   `group_max_a1 > running_max_a1` (max `a_1` seen across strictly
   earlier groups). Other points in the group are dominated by
   the group max.
4. Update `running_max_a1 ← max(running_max_a1, group_max_a1)`.
5. Restore original index order via the inverse permutation.

### Tie-handling (load-bearing edge case)

- **Two identical points (a_0, a_1):** neither dominates the other.
  Both share the group max → both pass the strict `>` check → both
  on skyline. ✓ matches brute.
- **Same a_0, different a_1:** lower-`a_1` dominated by higher within
  the same group.
- **Same a_1, different a_0:** lower-`a_0` group has
  `group_max_a1 == running_max_a1` → strict `>` fails → dropped.

These three cases were the diff between an O(N log N) "looks right"
formulation and the canonical KLP one. The first naive sweep
(strict `>` against a running max without group accumulation) lost
all-identical-points (49 of 50). Fixed by accumulating the group max
once per `a_0` group instead of per row.

## 2. Empirical results

Hardware: laptop CPU, single-thread numpy. Median of 5 iterations
post-warmup.

| N      | brute (ms) | sweep (ms) | speedup |
|---|---|---|---|
|    100 |     1.31   |     0.21   |    6.3× |
|   1000 |    56.96   |     1.97   |   28.9× |
|   5000 |  1250.83   |     9.94   |  125.9× |
|  10000 |  4858.80   |    19.67   | **247.0×** |

The brute path grows quadratically (4.86 s at N=10K). The sweep
grows log-linearly (20 ms at N=10K). For the ABB pool typical
size N ≤ 10K this means the Pareto filter is no longer the
bottleneck of the SSG-walk pipeline.

## 3. Test inventory (31/31 PASSED)

`hymeko_neuro/tests/test_ssg_pareto_filter.py`:

### Equivalence (15 + 6 + 2 = 23)

- `test_sweep_matches_brute_random[10|100|1000 × seed=0..4]`
  (15 parameterised cases) — uniform random `[0, 1)²`.
- `test_sweep_matches_brute_integer_scores[10|100 × seed=0..2]`
  (6 parameterised cases) — `randint(0, 5)` (heavy ties).
- `test_public_dispatch_uses_sweep_at_d2` + `..._falls_back_to_brute_at_d3`.

### Algebraic edge cases (7)

- `test_empty_input_returns_empty_mask`
- `test_no_axes_returns_all_true`
- `test_single_point_is_on_skyline`
- `test_all_identical_points_all_pass` (the load-bearing tie case)
- `test_anticorrelated_all_on_skyline`
- `test_strict_chain_only_dominant_survives`
- `test_validates_axis_shapes`

### Performance (1)

- `test_sweep_is_faster_than_brute_at_n1000` — asserts
  `sweep_wall ≤ 0.3 × brute_wall` at N=1000. Empirical ratio
  `~0.035` → 8.6× the assertion margin against profile noise.

Pytest summary: **31 passed in 3.58 s**.

## 4. Performance contract preservation

The brute path is **kept** as `_ssg_pareto_filter_brute` and remains
the reference specification:

- Every equivalence test compares `mask_brute == mask_sweep`
  bit-for-bit, so any future change to the sweep must produce the
  exact same Pareto set.
- D ≥ 3 routes to brute by default. Future KLP-divide-and-conquer
  for D ≥ 3 can be added behind the same dispatch without
  touching the public API.

## 5. Open work

1. **D ≥ 3 KLP-divide-and-conquer.** O(N log^{D-1} N) for higher
   dimensions. Worth doing if HSiKAN ever runs ≥ 3 score axes
   (currently we use 2: score + walk_len). Plan: extract
   `_ssg_pareto_filter_sweep_3d` via recursion on the median of
   axis 2 (cf. KLP 1975 §3).
2. **Rust port.** The pyo3 wheel for `enumerate_top_k_walks_rs`
   already binds the algorithm side; a parallel `pareto_filter_2d_rs`
   would shave the final 20 ms at N=10K to ~1 ms. Not blocking
   — Python sweep is already well below the rest of the HSiKAN
   inference cost.
3. **D=2 axis-ordering performance.** Currently we sort by `a_0`
   desc first. If `a_0` has higher cardinality of distinct values
   (typical: walk-length axis is integer-discrete, score axis is
   continuous), swapping the axis roles could halve the number of
   group transitions. Marginal; deferred.

## 6. CORE.YAML preservation

No CORE.YAML items touched. `hymeko_neuro` is not core-protected;
the public function signature `ssg_pareto_filter(walks_v,
walks_signs, score_axes)` is unchanged.

## 7. Files touched

| file | change |
|---|---|
| `hymeko_neuro/hyperedge/abb_walks.py` | refactored `ssg_pareto_filter` to dispatch; added `_ssg_pareto_filter_brute` (extracted from original) + `_ssg_pareto_filter_sweep_2d` (new) |
| `hymeko_neuro/tests/test_ssg_pareto_filter.py` | new test module (31 tests) |
| `docs/plans/2026-06-04-klp-skyline/plan.{tex,pdf,tikz,mmd}` | new plan (4-format) |
| `reports/2026-06-04-klp-skyline.md` (this) | new report |

## 8. Document genealogy

- `reports/2026-06-04-msg-abb-ssg-unified-implementation.{md,tex,pdf}`
  (App. A.8: the brute `ssg_pareto_filter` listing it replaces).
- `hymeko_neuro/hyperedge/abb_walks.py` source comment
  `# O(N²·D) brute force; fine for top_k ~ 10⁴ walks. Vectorising to
  # an O(N log N) skyline requires axis-specific sort + sweep that
  # we can write later if profiling demands it.`
  — now fulfilled.
