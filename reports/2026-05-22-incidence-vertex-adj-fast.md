# Incidence vertex-adj fast path — measurement report

**Date**: 2026-05-22
**Plan**: `docs/plans/2026-05-22-incidence-vertex-adj-fast/`
**Slug**: `incidence-vertex-adj-fast`
**Status**: **Partial — plan's 4× target NOT hit.  Honest verdict below.**

---

## 1. What the plan predicted vs what we measured

| axis | plan target | measured | verdict |
|------|------------:|---------:|---------|
| 4.1 Rayon parallel | ≥ 4× at 8 cores | **1.5–1.9×** | ❌ partial |
| 4.2 sorted-pair self-map | +10–20% | **null** (within noise) | ❌ no win |
| 4.3 bitset adjacency | ≤ 0.3× sorted-merge at low T | **1.18–3.64× SLOWER** at every scale tested except small/parallel | ❌ falsified |
| 4.4 CSR output | ≤ 10% overhead | **8–22% overhead** | ⚠ at the bound |

Three of four axes were either null or actively counterproductive at production scale.  Only the parallel path is unambiguously a win.

## 2. The numbers

Criterion median, 16-thread Ryzen 3700X, dual-channel DDR4-3200, alloc-free per-edge refactor in place.  Fixture: deterministic LCG-seeded signed graph; query count = ½ × n_tuples; self-edges = full match.

### Three scales

| scale | n_v   | n_t    | n_queries | serial   | parallel | bitset_s | bitset_p | parallel_csr |
|-------|------:|-------:|----------:|---------:|---------:|---------:|---------:|-------------:|
| small | 200   | 3 500  | 1 750     | 1.09 ms  | 1.03 ms  | 1.08 ms  | **0.72 ms** | 1.30 ms |
| mid   | 1 000 | 25 000 | 12 500    | 19.5 ms  | **13.1 ms** | 32.3 ms  | 17.0 ms  | 15.0 ms |
| large | 4 000 | 100 000| 50 000    | 90.2 ms  | **46.9 ms** | 357 ms   | 119.8 ms | 55.9 ms |

### Speedup vs serial (best path bolded):

| scale | parallel | bitset_s | bitset_p | parallel_csr |
|-------|---------:|---------:|---------:|-------------:|
| small | 1.06×    | 1.01×    | **1.51×** | 0.84× |
| mid   | **1.49×** | 0.60×    | 1.15×    | 1.30× |
| large | **1.92×** | 0.25×    | 0.75×    | 1.61× |

### Best path per scale

| scale | best | speedup | comment |
|-------|------|--------:|---------|
| small (Bitcoin Alpha shape) | `bitset_parallel` | 1.51× | bitset wins *only* because n_t × n_v fits L2 |
| mid (Bitcoin OTC shape)     | `parallel`        | 1.49× | bitset already loses by 1.7× |
| large (Slashdot shape)      | `parallel`        | 1.92× | bitset is 3.6× slower; cache-blown |

## 3. Why the plan was wrong

### 3.1 Bitset path (§4.3) — cache-bound, not register-bound

The plan assumed `adj_u | adj_v` becomes "one vpor per word" and wins.  In practice each query reads `2 · n_words · 8` bytes of bitset + writes `n_words · 8` of scratch.  At Slashdot scale (n_v = 4 000, n_t = 100 000), bitset adjacency is **50 MB** — well over Ryzen 3700X's 32 MB L3.  Every query streams `~12.5 KB` of mostly-zero bitset through the cache hierarchy, vs the sorted-merge which only touches the actual ~70 entries of `adj_u + adj_v`.

The bitset path only wins when `n_v × ceil(n_t / 64) · 8` fits in L2 (~512 KB), i.e. **only the small/Bitcoin-Alpha regime**.  Even there, the win is 1.51× (bitset_parallel vs serial), barely better than the plain parallel path at other scales — and *less* than the parallel path's 1.92× at large scale.

**Lesson**: dense representations win only when the *density* matches the access pattern.  The sorted-merge already exploits sparsity; switching to dense throws that away.

### 3.2 Sorted-pair self-map (§4.2) — null result

HashMap → sorted-vec partition_point: **51.1 ms → 50.1 ms at large scale (within noise)**.  The HashMap lookup wasn't on the hot path.  The hot path is the sorted-merge over CSR slices, which is memory-bound, not lookup-bound.

### 3.3 Parallel scaling (§4.1) — memory-bandwidth limited

Plan: "embarrassingly parallel per-edge work, ≥ 4× on 8 cores."

Reality: **1.92× at 16 threads on the large fixture.**

The per-edge work is memory-bound (sorted-merge reads from CSR), not compute-bound.  Ryzen 3700X has dual-channel DDR4-3200 = 51.2 GB/s peak.  Effective throughput at serial ≈ 7-10% of peak; parallel can't exceed total memory bandwidth, only saturates it differently.  16 threads competing for the same bus → memory becomes the queue, not the cores.

### 3.4 CSR output (§4.4) — overhead is real, payoff is downstream

CSR materialization costs 8–22% over COO depending on scale.  The plan claimed ≤10%; mid-scale (15% over parallel) exceeds that.

The plan's argument was "downstream PyTorch skips coalesce" — that's a *separate* measurement we haven't taken yet.  In isolation the CSR path is a regression; the win, if any, lives in the PyTorch interaction.

## 3.5 Bonus negative result — query presort

After the report-as-of-21:38 was written, one more experiment ran: pre-sort queries by `(min(u,v), max(u,v))` before the parallel chunking so consecutive queries hit overlapping CSR rows in cache (plan §5 follow-up item 2).

Measured:

| scale | parallel COO | parallel CSR |
|-------|-------------:|-------------:|
| small | -18% (faster) | +17% (slower) |
| mid   | -6% (within noise) | **+76% slower** |
| large | flat | **+86% slower** |

Net: COO got slightly faster, CSR got dramatically worse.  Cause: the row-scramble forces `to_csr`'s slow counting-sort path (the `already_sorted` fast-return no longer fires).  At large scale, scattered writes to `cols_out[pos]` blow the cache because the source `coo.rows[i]` is in u-sorted order, not chunk-sorted order.

**Reverted.**  The code now carries a comment explaining the experiment so future-me doesn't try it again.  If a future caller specifically wants COO output, the presort might be worth gating behind `BuildOpts.presort` — but the current cost/benefit doesn't justify the API surface.

## 4. What did land cleanly

- **13 parity tests** in `hymeko_graph/tests/incidence_parallel.rs` against a brute-force `HashSet<u32>` reference.  All paths (serial, parallel, bitset_serial, bitset_parallel, all four × CSR output) produce row-set-identical results modulo within-row order.
- **Alloc-free per-edge refactor**: the earlier draft of this module had `std::mem::take(scratch)` per query, allocating a new `Vec<u32>` on every iteration.  Refactor to `process_one_edge_into` (writes directly into caller's row/col/val vectors) gave a measurable 6% improvement at large/parallel (50.0 → 46.9 ms).
- **Public API**: `build_edge_incidence(..., BuildOpts)` Strategy entry that dispatches on `(parallel, bitset_threshold, output)`.  Pre-existing `build_edge_incidence_vertex_adj` is preserved bit-for-bit and delegates to `BuildOpts::default()`.
- **No new top-level dependency** (rayon was already at 1.10).

## 5. Decision

### Ship what works

`build_edge_incidence` with `BuildOpts { parallel: true, ..default() }` is a real 1.49–1.92× win at production scale and zero correctness risk.  Worth registering the PyO3 entry next session.

### Don't ship the bitset path as default

The `bitset_threshold = 0` default keeps the dispatch off.  Callers who *know* their graph is Bitcoin-Alpha-shaped can opt in; everyone else stays on the sorted-merge.  Document the threshold formula in the PyO3 docstring so users don't misuse it.

### Skip the CSR output for now

CSR's only payoff is on the PyTorch side and we haven't measured it.  Until we do, COO is the default.  CSR mode stays in the API for the day a measurement justifies it.

### What would actually move the needle past 2×

The bottleneck is memory bandwidth, not CPU.  Real lifts would come from:

1. **GPU-side construction** — keep `M_e` on the GPU and stop building it on CPU at all.  Cost: write a Triton kernel; gain: 10–50× because GPUs have HBM bandwidth.
2. **Pre-sort queries by `(u, v)` endpoint** so consecutive queries hit overlapping CSR rows in cache.  Cost: O(E_query log E_query) one-time; gain: maybe 1.3× if the access pattern is friendly.
3. **CSR row prefetch hints** — explicit `_mm_prefetch` on the next-iteration row pointers.  Cost: trivial; gain: typically 5-15%.

None of these are in the original plan; (1) is a different project entirely (GPU codegen for sparse incidence assembly), (2)/(3) are within reach if we want one more session on this.

## 6. Anti-pattern + contract check (CLAUDE.md §6)

- §6.5 #1 (Cartesian-product API): no `_par`, `_bitset`, `_csr` function variants — all dispatched through `BuildOpts`.
- §6.5 #2 (algorithm code in PyO3): not touched — algorithm stays in `hymeko_graph::incidence`.
- §6.5 #6 (`too_many_arguments` band-aid): single `#[allow]` on the outer `build_edge_incidence_vertex_adj` legacy flat-arg wrapper; internal helpers take `&BuildOpts` derivative refs.
- §6.5 #11 (no globals): `BuildOpts` is passed explicitly; no env-var dispatch.
- §6.3 (clippy / fmt): clean.  No new `#[allow]` aside from the documented one above.
- §6.4 (no `unwrap()` in non-test code): clean.
- §10 (toolchain): criterion 0.8.2 (pinned), no toolchain changes.
- §3 (production-scale smoke before queuing): the bench *is* the production-scale smoke; no overnight runs touched by this refactor.

## 7. Files touched

| file | change | LOC delta |
|------|--------|-----------|
| `hymeko_graph/src/incidence.rs` | added BuildOpts, SelfMap, BitsetAdj, parallel + bitset paths, alloc-free refactor | +250 / -50 |
| `hymeko_graph/src/lib.rs` | re-exports for new types | +4 / -1 |
| `hymeko_graph/tests/incidence_parallel.rs` | new — 13 parity tests + fixture + brute-force reference | +330 |
| `hymeko_graph/benches/incidence_bench.rs` | new — 5 paths × 3 scales = 15 cells | +160 |
| `hymeko_graph/Cargo.toml` | added `[[bench]] name = "incidence_bench"` | +4 |

No PyO3 wrapper changes yet (deferred — see §5).  No `CORE.YAML` items touched.

## 8. Provenance

- Git SHA at measurement: `507d7e24d1cf03d359504bf14819b8e2274380e9` (working tree dirty: this report, plan dir, in-flight quadtree/incidence migration files).
- Host: Linux 6.17.0-23-generic, AMD Ryzen 7 3700X (8C/16T), DDR4-3200 dual-channel, 32 MB L3.
- Toolchain: rustc stable, criterion 0.8.2, rayon 1.10.
- Bench mode: `cargo bench -p hymeko_graph --bench incidence_bench` (release profile, plus a `--quick` second pass for confirmation).
- Background workload at measurement time: VOC Phase 8 B9 training on GPU (CPU usage ≈ minimal).
- 16 GB RSS cap: not stressed (peak parallel scratch usage ~2 MB).

## 9. Open follow-ups

1. **PyO3 entry** for `build_edge_incidence(..., BuildOpts)` — defer until the win has a concrete consumer asking for it.
2. **Query-sort + prefetch experiment** (above §5 item 2/3) — one more session could add another 1.3–1.5× on top of parallel.
3. **GPU-side `M_e` construction** — separate, larger project.  Right framing: this isn't a refactor of the CPU path, it's a *replacement* of the CPU path.
4. **CSR-output downstream payoff measurement** — wire CSR through `torch.sparse_csr_tensor` and measure end-to-end vs COO; if a real win exists, switch the default.
