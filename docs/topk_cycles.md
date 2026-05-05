# Axiom-aware Top-K Cycle Enumeration

Use this when the full cycle set of a signed graph is too big to
materialise in memory or train on. The machinery cuts the cycle set
by 10-1000x while keeping every vertex covered, and exposes
graph-theoretic axioms (Cartwright-Harary balance, Davis weak
balance, Friedler P-graph A0) as DFS pruners that fire *during*
enumeration, not after.

## When to use what

| situation | use |
|---|---|
| Full cycle set fits in RAM, want all of it | `hymeko.enumerate_k_cycles_rs` (existing) |
| Full set too big, want top-K by score | `enumerate_top_k_cycles_signed_rs` |
| Need every vertex covered, bound \|M_e\| per row | `enumerate_top_k_per_vertex_cycles_signed_rs` (recommended for HSiKAN) |
| Just want vertex-stratified random sample | `enumerate_top_k_per_vertex_cycles_signed_rs` with `score_kind="low_root"` (deterministic vertex-spread) |

For HSiKAN training on Slashdot/Epinions, the per-vertex variant is
the right default — it bounds `|M_e| ≤ n_vertices × m_per_vertex`
and guarantees no row of the cycle incidence matrix is empty.

## Python API

```python
import hymeko, numpy as np

# Inputs: signed edge list (uint32 endpoints + int8 sign).
edges_u = np.array([...], dtype=np.uint32)
edges_v = np.array([...], dtype=np.uint32)
edges_s = np.array([...], dtype=np.int8)   # ±1

# Global top-K: keep the K highest-scoring cycles overall.
cycles, scores = hymeko.enumerate_top_k_cycles_signed_rs(
    edges_u, edges_v, edges_s,
    n_nodes,
    k_len,                       # cycle length (3, 4, 5, ...)
    K,                           # how many cycles to keep globally
    score_kind="balance",        # see "Scorers" below
    pruner_kind="none",          # see "Axiom pruners" below
)

# Vertex-stratified top-m: keep m cycles per vertex, dedup union.
cycles, scores = hymeko.enumerate_top_k_per_vertex_cycles_signed_rs(
    edges_u, edges_v, edges_s,
    n_nodes,
    k_len,
    m_per_vertex,                # m heap slots per vertex
    score_kind="fraction_negative",
    pruner_kind="none",
)
```

Returns:
- `cycles` — `(N, k_len)` `uint32` ndarray of vertex sequences.
- `scores` — `(N,)` `float64` ndarray, sorted descending.

Both functions release the GIL during enumeration and use rayon
parallelism over starting vertices.

## Scorers

`score_kind` is the heuristic used at *emit time* to rank closed
cycles for the heap. Higher score = more preferred.

| value | what | use case |
|---|---|---|
| `"balance"` | sign product of edge signs (+1 balanced, -1 unbalanced) | surface Heider-stable triads |
| `"fraction_negative"` | fraction of negative edges in cycle | surface frustrated/conflict cycles |
| `"sign_product_abs"` | always ±1, useless alone | placeholder |
| `"low_root"` | prefer cycles starting at low-index vertices | deterministic vertex-spread, behaves like a tie-breaker |

For HSiKAN signed link prediction the recommended choice is
`fraction_negative` — frustrated triads carry the
sign-prediction signal that balanced ones largely miss.

## Axiom pruners

`pruner_kind` is the structural pruner consulted at *extend time*
during DFS. This is what makes top-K **cheaper than full**, not just
"full and then sort." A BFS-distance lower-bound is *always* applied
in addition (it dominates on dense graphs at high k).

| value | what | when |
|---|---|---|
| `"none"` | only BFS-distance pruning | default; safe everywhere |
| `"balance"` | Cartwright-Harary, only balanced cycles | when you only want stable triads |
| `"unbalanced"` | only unbalanced cycles | when you only want frustrated triads |
| `"davis"` | Davis weak balance, reject all-negative triads | sociology-inspired filter; mild pruning |

The pruners short-circuit on first reject, so combining a strong
scorer with a strong pruner doesn't double-count work.

## Env-var integration with HSiKAN

`signedkan_wip/src/n_tuples.py` reads four env vars to switch the
existing `_enumerate_cycles_fast` over to the top-K paths:

```bash
HSIKAN_TOPK_MODE=per_vertex            # or "global"; unset = full enum
HSIKAN_TOPK_K=16                       # m_per_vertex (or K when global)
HSIKAN_TOPK_SCORER=fraction_negative   # see Scorers above
HSIKAN_TOPK_PRUNER=none                # see Axiom pruners above
```

Then run training as normal:

```bash
PYTHONPATH=signedkan_wip python3 -m src.run_final_cell \
    --dataset slashdot --model HSiKAN --hidden 16 --n-epochs 20 --seed 0
```

## Recommended configs

Per-dataset configs that have been tested end-to-end (1 seed,
2026-05-05):

| dataset | mode | m | scorer | pruner | wall | AUC | notes |
|---|---|---|---|---|---|---|---|
| bitcoin_alpha | (full) | — | — | — | 30 s | 0.9203 | baseline; m=128 still loses 5pp |
| slashdot | `per_vertex` | 16 | `fraction_negative` | `none` | ~17 min | 0.8368 | -0.024 vs Walk-HSiKAN baseline |
| epinions | `per_vertex` | 16 | `fraction_negative` | `none` | ~7 min | 0.7088 | -0.055 vs historical; previously DNF |

The next sweep to run (if AUC matters more than speed): bigger `m`
(64, 128) and adding `pruner=davis` to see if frustration-targeted
pruning recovers the gap.

## Performance notes

- **BFS-distance pruning is mandatory** at k ≥ 4 on graphs with ≥10K
  nodes. Without it, even rayon-parallel top-K hits multi-minute
  enumeration and OOM risk on Slashdot.
- **Vertex coverage at small m**: at `m=1` you get ~99.7% of
  vertices on the Slashdot top-2000 hub region with only 1.5% of
  the full cycle count. The remaining 0.3% are isolated vertices on
  no triangle.
- **`m=4` is the sweet spot for memory** on full Slashdot at k=4:
  upper bound `4 × 82144 ≈ 328K cycles` vs full `55.5M` = 170×
  reduction.
- **AUC degrades smoothly with m**: bitcoin_alpha sees -0.13 at
  m=16, -0.07 at m=64, -0.05 at m=128. Larger graphs are more
  redundant, so the AUC penalty shrinks with dataset size (Slashdot
  -0.024, Epinions -0.055).

## Building / installing

```bash
cd hymeko_py
maturin build --release
pip install --force-reinstall --no-deps \
    /path/to/target/wheels/hymeko-0.1.0-cp313-*-manylinux_*.whl
```

Verify the new symbols exist:

```python
import hymeko
assert hasattr(hymeko, "enumerate_top_k_cycles_signed_rs")
assert hasattr(hymeko, "enumerate_top_k_per_vertex_cycles_signed_rs")
```

## Where the code lives

| concern | path |
|---|---|
| serial + parallel top-K Rust | `hymeko_graph/src/topk_cycles.rs` |
| BFS-distance pruning | same file, `bfs_distances_capped` |
| per-axiom counters | `hymeko_graph/src/pruner.rs` (`CountingPruner`, `CompositePruner`) |
| PyO3 bridge | `hymeko_py/src/cycles.rs` (search for `enumerate_top_k_*_signed_rs`) |
| HSiKAN integration | `signedkan_wip/src/n_tuples.py`, `_enumerate_cycles_fast` |
| analysis CLI (Rust) | `hymeko_graph/examples/cycle_stats.rs` |
| Python coverage demo | `signedkan_wip/src/topk_cycle_demo.py` |
| PDF report | `reports/topk_cycles_brief.pdf` |

## Running the benches

```bash
# Rust microbenches (DFS, BFS, top-K vs full)
cargo bench -p hymeko_graph --bench graph_bench

# Per-axiom effect comparison
cargo run --release --example axiom_effect -p hymeko_graph

# Full-graph cycle statistics on a real signed-edge file
cargo run --release --example cycle_stats -p hymeko_graph -- \
    signedkan_wip/data/slashdot.txt 600 3
```

## Open work

1. **Color-coding sampler in top-K** — the older `enumerate_k_cycles_rs`
   has Alon-Yuster-Zwick rainbow colouring; not yet ported into the
   top-K path.
2. **Edge-weighted M_e** — bridge already returns per-cycle scores;
   could feed them as edge weights in `M_e` instead of unit weights.
3. **5-seed statistics** for the headline numbers.
4. **m × pruner sweep** to close the AUC gap on Epinions.
