# Cycle-enumeration acceleration plan (2026-05-03)

## Why this plan exists

The inference benchmark (run on 2026-05-03 — see
`experiments/results/inference_bench.json`) made the cost structure of
HSiKAN explicit. Forward-pass latency is a known cost (12-104× SGCN
depending on device + dataset). The bigger and less-discussed cost is
**cycle enumeration setup**: 8 s for Bitcoin Alpha, 36 s for Bitcoin
OTC, 150 s for Slashdot @ 100k, and ≥ 17 min for Slashdot @ 3M (the
SOTA cycle budget). Cycle enumeration is the single largest wall-clock
component of the HSiKAN pipeline and the main blocker for pushing
k=5/k=6 into the SOTA-cycle-budget regime that already worked for k=4.

## Current state

The Rust enumerator at `hymeko_py/src/cycles.rs`:
- Single-threaded DFS rooted at every vertex `0..n`, with smallest-root
  symmetry breaking and the `path[1] < path[k-1]` orientation tiebreak
  for undirected k-cycles
- CSR adjacency, sorted neighbor lists with `binary_search` for edge
  closure tests
- Three sink modes: `Full` (collect all), `Reservoir` (Vitter
  Algorithm-R, unbiased sample of `cap` cycles, but DFS still walks the
  full cycle space), `EarlyStop` (keep first `cap`, then signal DFS to
  bail — biased toward small-vertex cycles, but bounds DFS time)
- Flat `Vec<u32>` of stride k everywhere — no per-cycle heap allocation.
  Slashdot k=4 peak RSS: 253 MB (down from 23 GB pre-rewrite)
- Exposed via PyO3 as `hymeko.enumerate_k_cycles_rs(...)`

What works:
- Memory-bounded for any cycle count (reservoir mode)
- Correct: cycle-count parity with the Python reference DFS

What's slow:
- Single-threaded — leaves 7-15 cores idle on every machine we run on
- Reservoir mode pays the full DFS cost; the saving is only memory, not
  wall-clock
- EarlyStop has the right complexity but introduces measurable bias
  (Slashdot AUC 0.83 → 0.61 in earlier tests)

## The six options on the table (ranked by leverage × confidence)

| # | option | expected speedup | implementation effort | confidence |
|---|---|---|---|---|
| 3 | Rayon parallel DFS | 5-7× on 8-core | ~half day | high |
| 1 | Color-coding sampler | 10-50× per coloring on high-arity | 3-5 days | medium-high |
| 4 | GPU enumeration | 50-200× | 1-2 weeks | medium |
| 5 | Structural pruning (k-core) | 2-3× free | hours | high |
| 2 | Sampler-instead-of-enumerator (path-closure) | 10-100× for triangle/k=4 | days | medium |
| 6 | Spectral counts | 100×+ but only for *counts*, not cycles | days | high |

**This plan implements 3 and 1.** Combining them = "parallel
color-coding sampler" and is the natural endpoint that asymptotically
matches the best published k-cycle enumerators while keeping our flat
`Vec<u32>` memory model.

5 (k-core pruning) is cheap and orthogonal — defer as follow-up; it
multiplies whatever other speedup is in place.

4, 2, 6 are deferred — each is its own multi-day project and should not
block the immediate gains from 3+1.

## Detailed plan: Option 3 — Rayon parallel DFS

### Design

The starting-vertex loop in `enumerate_k_cycles_rs` is embarrassingly
parallel. Each `start ∈ 0..n` produces an independent DFS; smallest-
root symmetry breaking guarantees no cycle is double-counted across
roots. The CSR (`row_ptr`, `col_idx`) is read-only and trivially `Sync`.

Per-thread state:
- `visited: Vec<bool>` of size n  (must be thread-local)
- `path: Vec<u32>` of capacity k  (must be thread-local)
- A thread-local `Sink`

The interesting design problem is **merging per-thread sinks into a
single output of size `cap`** without losing the unbiased-sample
property in `Reservoir` mode.

### Per-mode merge strategy

**Full mode**: trivial. Concat per-thread `Vec<u32>` buffers.

**Reservoir mode** (the non-trivial case):
After all threads finish, each thread `t` has:
- `buf_t`: a uniform sample of `min(seen_t, cap)` cycles from the
  `seen_t` cycles that thread saw
- `seen_t`: the count of cycles thread `t` produced

The combined target is a uniform sample of `cap` cycles from the global
total `seen_total = Σ seen_t`. Two correct merge strategies:

1. **Stratified resampling (chosen)**: each thread is a stratum with
   weight `seen_t / seen_total`. Sample
   `n_t = round(cap · seen_t / seen_total)` items uniformly from `buf_t`
   (which is itself uniform from the full stratum). The result is a
   stratified-uniform sample of size `≈ cap` from the global cycle set.
   Add a per-thread `Δ` correction (max-deficit-first) to land exactly
   at `cap`. **Cleaner, faster, still unbiased per-stratum.**

2. **Re-Vitter on union**: feed `buf_t` items into a final reservoir,
   weighting each item's acceptance probability by `seen_t / |buf_t|`.
   Asymptotically equivalent but harder to implement correctly.

We pick (1).

**EarlyStop mode**: each thread gets a smaller per-thread cap
`cap / n_threads + slack`. Threads early-stop independently. Final
merge concatenates and truncates to `cap`. Still biased (toward
small-vertex roots, *less* biased than serial EarlyStop because each
thread covers a contiguous root range), but throughput is dominated by
the parallelism win.

### Implementation steps

1. **Add `rayon = { workspace = true }` to `hymeko_py/Cargo.toml`**.
2. **Refactor `dfs_from`** to take `&Sink` borrows by mutable reference
   (already does); add `Send` to the closure passed to rayon.
3. **Add `enumerate_k_cycles_parallel_rs`** as a sibling Python entry
   (or extend the existing one with a `n_threads: Option<usize>` arg
   that defaults to "use all cores"). Keep the serial version as a
   correctness baseline so we can A/B test.
4. **Stratified merge for Reservoir mode**.
5. **Per-thread early-stop coordination via `AtomicUsize`** for
   EarlyStop mode (each thread checks shared counter every few cycles).
6. **Validation**: assert per-cycle deduplication (parallel must give
   the same set of cycles modulo sampling) and run against the existing
   `k4_speed_benchmark` fixture.

### Acceptance criteria

- `cargo test` passes (existing correctness tests cover serial path)
- New parallel test: cycle-count from parallel matches serial on
  karate, sbm_n200_k4, sbm_n400_k5 fixtures
- Wall-clock benchmark: parallel ≥ 4× faster than serial on Bitcoin
  Alpha k=4 on an 8-core machine (target: 5-7×)
- Memory: per-thread reservoir × n_threads ≤ 2× single-thread (cap × k
  bytes per thread, n_threads threads)

## Detailed plan: Option 1 — Color-coding sampler

### Algorithmic background

Alon-Yuster-Zwick (1995) randomized algorithm for finding patterns in
graphs:
1. Color each vertex randomly with one of `k` colors
2. A k-cycle is **rainbow** iff all `k` of its vertices have distinct
   colors
3. Probability a fixed k-cycle is rainbow under random coloring:
   `k! / k^k` (≈ 0.094 for k=4, 0.038 for k=5, 0.015 for k=6, 0.006 for
   k=7)
4. To sample each k-cycle with probability `≥ 1 - 2^(-r)`, take
   `K = e^k · r` independent colorings

### Why this gives a speedup

A coloring-aware DFS only extends paths to neighbors of *unused colors*.
At depth `d` of the DFS, the average branching factor drops by a factor
`(k - d) / k` (only `k - d` colors remain). Cumulatively, the search
tree size is reduced by `k! / k^k` — the same factor as the rainbow
probability.

This means: per coloring, the DFS finds `~n_total · k!/k^k` rainbow
cycles in time `~n_total · k!/k^k · per-cycle-cost` instead of
`n_total · per-cycle-cost`. The total work to find `n_total` cycles
across `K = k^k / k!` colorings is **the same** as serial enumeration
asymptotically — *but* the work to find `M < n_total` cycles is
proportional to `M` (not to `n_total`). For a target sample size much
smaller than the cycle space, this is the desired sub-linear speedup.

For Slashdot k=5 (55 M total cycles, target sample 1 M):
- Serial reservoir: full cost ≈ 17 min
- Color-coded sample: ≈ (1/55) of full cost ≈ 20 s

### Design

Add a fourth `Sink` mode? No — this isn't a sink behavior. The DFS
itself needs to know about colors. Add a parallel function:

```rust
#[pyfunction]
pub fn enumerate_k_cycles_color_coded_rs(
    py, edges_u, edges_v, n_nodes, k,
    target_cycles: usize,
    seed: u64,
    n_threads: Option<usize>,
    max_colorings: Option<usize>,
) -> PyResult<Py<PyList>>
```

Internals:
1. CSR adjacency built once (shared across colorings)
2. Outer loop over colorings:
   - Generate vertex colors via `seed_per_coloring` LCG
   - Initialize a per-coloring used-colors bitmask (just a `u32` since
     `k ≤ 32`; `k > 32` is academic)
   - DFS exactly as serial, with the extra check:
     `if used_colors & (1 << color[nxt]) != 0 { continue; }`
   - Emit rainbow k-cycles to a **shared deduplicating sink**
     (canonical form: smallest rotation + lex-smallest direction, used
     as a hash key)
3. Stop when `target_cycles` reached or `max_colorings` exhausted

Deduplication: a `DashMap<u128, ()>` of canonical cycle hashes. For k
≤ 16, each cycle fits in a u128 (k vertices × 8 bits), and the
canonical key can be packed without hashing. For k > 16, use a `[u32;
k]` and `DashSet<Vec<u32>>`.

(For k ≥ 8, dedup overhead may dominate — fall back to "report
duplicates as distinct samples" with explicit caveat. Acceptable for
the αₖ-mixing pipeline because duplicate cycles just contribute a
slightly higher-weight feature for that triad pattern.)

### Combination with Option 3

The outer "loop over colorings" and the inner "loop over starting
vertices" are both parallelizable. Use rayon at both levels:
`par_iter` over colorings × thread-local DFS sweeping all start
vertices. Atomic counter for `target_cycles`; threads bail when
counter reaches target.

### Implementation steps

1. Implement serial color-coded DFS first; correctness-test against
   the existing serial enumerator (rainbow cycles found by both should
   be a subset of serial-enumerated cycles for the same graph)
2. Add deduplication via `DashMap` (need `dashmap` dependency)
3. Wire to PyO3 as new function
4. Parallelize via rayon `par_iter` over colorings
5. Bias check: compare cycle-balance distribution (fraction balanced)
   between color-coded sample and reservoir sample on Bitcoin Alpha
   k=4 — should agree to within 0.01

### Acceptance criteria

- Correctness: every cycle returned by color-coded sampler is a real
  k-cycle in the graph (verified by edge-closure test on each emitted
  tuple)
- Coverage: for a graph small enough to fully enumerate (karate, k=4),
  color-coded sampler with K = `k^k / k!` × 5 colorings recovers ≥ 99%
  of cycles
- Bias: balance fraction of color-coded sample ≈ balance fraction of
  full enumeration ± 0.01 on `sbm_n400_k5_s0`
- Wall-clock: at least 5× faster than parallel-reservoir for sampling
  100k cycles from Slashdot k=5 (target: 10-20×)

## Detailed plan: Option 4 — GPU k-cycle enumeration

### Why DFS-on-GPU is the wrong framing

A literal port of the recursive DFS to a GPU kernel hits SIMT pathologies:
recursion → manual stack in shared memory; per-thread branch divergence
on the depth-first traversal kills warp utilisation; irregular adjacency-
list memory access defeats coalesced reads. Published GPU subgraph
miners (GraphPi, G²Miner, PRESTO, PIVOTER) all replace DFS with **edge-
centric BFS-style adjacency-list intersection**, which maps cleanly to
warp-cooperative work.

### Algorithmic shape

For each undirected edge `(u, v)` in the graph (one work unit per
warp, 32 threads cooperating), find every length-(k-2) path
`u → w_1 → … → w_{k-2} → v` with no repeated vertices. Closing the
path with the edge `(v, u)` yields a k-cycle. The core primitive is
**warp-cooperative neighborhood intersection**: given two sorted
adjacency lists `N(u)` and `N(v)`, all 32 threads in a warp cooperate
to compute their intersection in one pass via merge-style shuffles
(or bitonic-sort + binary-search hybrids, depending on average
list length).

### Concrete two-stage pipeline

**Stage A — color-coded sampler on GPU** (lower complexity, recommended
first build):
1. Generate K random vertex colorings on host, push to device
2. For each coloring kernel launch:
   - Each warp picks one starting edge `(u, v)` with `color(u) ≠
     color(v)`
   - Warp-cooperative DFS-by-BFS-layers: at each depth, intersect the
     current frontier's neighbours with the not-yet-used-colors
     vertex set; emit length-(k-2) extensions
   - Emit closing-edge cycles into a global ring buffer with atomic
     append
3. Host accumulates dedup'd cycle list across launches

**Stage B — exact enumeration on GPU** (higher complexity, deferred):
The same kernel without colour gating, using warp-cooperative
intersection to enumerate every k-cycle. Used as a correctness oracle
against the CPU enumerator on small graphs and as the production
backend on graphs where the cycle count fits in GPU memory.

### Implementation plan

- **Crate**: extend the existing `hymeko_compute` Vulkan crate (already
  green for SpMV + force-directed on the RTX 2070 SUPER) with two new
  shaders. Keep a CUDA backend on the table as a follow-up but lead
  with Vulkan for vendor portability.
- **Adjacency layout**: CSR with sorted neighbours (matches today's
  CPU layout; trivial host→device copy).
- **Cycle storage**: ring buffer in GPU global memory of size
  `target_cycles * k * sizeof(u32)`, append index via atomic counter.
  When buffer fills, signal host to drain, dedup, and reset.
- **Dedup**: keep on host (the GPU isn't great at hash-table ops);
  copy back per-launch and merge into the same `DashMap` infrastructure
  used by the CPU color-coded sampler.
- **PyO3 surface**: new function
  `hymeko.enumerate_k_cycles_color_coded_gpu_rs(...)` mirroring the CPU
  signature, with a `gpu_device: Option<usize>` parameter. Falls back
  to the CPU implementation if no compatible Vulkan device is available.
- **Validation**: every cycle returned by the GPU sampler must be a
  real k-cycle in the graph (verified by edge-closure on host); on
  small fixtures the GPU sampler must recover the same canonical
  cycle set as the CPU enumerator.

### Expected performance

For Slashdot k=4 (current CPU parallel: 27.5 s for 100k cycles):
- GPU color-coded: 20-100× over CPU parallel = **0.3-1.5 s**
- GPU exact enumeration: 50-200× over CPU serial = **0.6-2.3 s**

For Slashdot k=5 (currently impractical at SOTA cycle budgets):
- GPU color-coded: opens the regime — pushes the 17 min CPU serial
  estimate into the **20-60 s range**, making k=5 SOTA training
  feasible

### Risks / unknowns

- Branch-divergence rate inside warp-cooperative intersection on
  high-variance adjacency lists (Slashdot has both very-high-degree
  and very-low-degree vertices; need to bin by degree to keep warps
  balanced)
- Vulkan compute-shader debugging is more painful than CUDA; budget
  extra time for the first kernel
- Memory-bandwidth saturation: at high cycle output rates, the atomic-
  append-to-ring-buffer pattern may bottleneck on memory write
  throughput rather than compute. Keep the kernel parameterized so
  we can tune ring-buffer width vs warp count if this surfaces.

### Order of operations

1. **Stand up the smallest GPU kernel first**: one warp, one starting
   edge, one coloring, k=4 only, no dedup. Just to verify the warp-
   cooperative intersection primitive works on real graph data.
2. Add ring buffer + atomic append + host drain
3. Scale to many warps × many starting edges
4. Add dedup integration with the CPU `DashMap`
5. Extend to k=5
6. Benchmark against CPU color-coded (the new oracle from this plan's
   Option 1 work)

Estimated effort: **1-2 weeks** of focused work for stages 1-4,
**+ 3-5 days** for stage 5 (k=5), assuming the warp-cooperative
intersection primitive comes together cleanly. Higher risk than
Options 1+3 because of GPU-debugging overhead.

## Out of scope (for this plan)

- **Option 2 (path-closure sampler)**: useful for k=3, k=4 only;
  color-coding subsumes it for our purposes. Implementing as a
  cross-validation oracle for color-coding bias is a small follow-up.
- **Option 5 (k-core pruning)**: easy follow-up; can be added as a
  pre-filter to either the parallel or color-coded path in a few hours
  once 3+1 land. Tracked in `FUTURE_DIRECTIONS.md`.
- **Option 6 (spectral counts)**: doesn't give us cycles, only counts.
  Useful as a sanity-check oracle for sampler bias — implement when
  validating 1.

## Validation plan

After both options land:
1. Re-run `run_inference_bench.py` on the same three datasets
2. Add a `setup_parallel_ms` and `setup_color_coded_ms` column
3. Update `inference_bench.json` and the paper draft
4. Re-run Slashdot k=5 SOTA training with the new sampler — first time
   we'd have k=5 in the SOTA-cycle-budget regime (3 M cycles or more).
   This is the actual scientific payoff.

## Memory + dependency budget

- `rayon` already in workspace — zero new deps for Option 3
- `dashmap = "6"` for deduplication in Option 1 — small, well-known,
  cdylib-friendly. Confirm pyo3 + dashmap interop (no Python objects
  shared across threads, only `[u32; k]` keys)

## Why this order (3 first, then 1)

1. Option 3 is a few hours of work and gives a concrete, measurable
   win. It also de-risks the parallel infrastructure that Option 1
   reuses.
2. Option 1 is harder and benefits from the parallel scaffolding.
   Verifying Option 1's correctness is also easier when we can run a
   parallel serial-style enumeration as the oracle.
