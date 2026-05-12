# Streaming top-m enumerator — note on existing state

The "streaming top-m cycle enumerator" task (Ádám's idea #1) was scoped on the assumption that the current Rust enumerator builds a full cycle list before pruning. **This is not the case.**

## Existing state (after a deeper read of `hymeko_graph/src/topk_cycles.rs`)

`enumerate_top_k_per_vertex_cycles_par` (line 508) already streams during enumeration:

```rust
fn dfs_per_vertex(...) {
    if path.len() == k_len {
        // ... compute score s ...
        for &v in path.iter() {
            let heap = &mut per_vertex[v as usize];
            if heap.len() < m_per_vertex {
                heap.push(HeapEntry { score: s, cycle: path.clone(), signs: signs.clone() });
            } else {
                let beat = heap.peek().map(|min| s > min.score).unwrap_or(true);
                if beat { heap.pop(); heap.push(...); }
            }
        }
    }
    // ... DFS extension ...
}
```

Each cycle is found, scored, and either kept (in bounded per-vertex heaps of size ≤ `m_per_vertex`) or dropped on the spot. The full-enumeration list never materialises.

For Slashdot k=4 m=128 the actual peak heap memory is `O(n_threads × n_vertices × m × ~80 bytes per HeapEntry)` ≈ 800 MB–1.6 GB depending on thread count — already a bounded budget, not a runaway full-enumeration footprint.

## What COULD still be optimized (deferred)

| optimization | savings | effort | complexity |
|---|---|---|---|
| **Pack `HeapEntry`** — replace `Vec<u32> + Vec<i8>` with `[u32; MAX_K]` + sign bitmask. ~76 → ~42 bytes per cycle. | ~45% heap memory | ~half day | Const generics or fixed MAX_K=8; touches every site that constructs / consumes HeapEntry |
| **Write streaming output directly to numpy** — pre-allocate a max-sized `(n_vertices × m, k)` numpy buffer, slice down at the end. Avoids the intermediate `Vec<TopKCycle>` between heap merge and numpy reshape. | ~10–15% peak memory | ~quarter day | Touches the PyO3 wrapper only |
| **Lockless per-vertex heap** — replace per-thread heap-array allocations with a shared lockless concurrent heap. | Removes per-thread duplication; cuts peak memory by ~`(n_threads − 1) × heap_size`. | ~1 day | Needs a lock-free heap (or per-vertex Mutex with low contention) |

None of these are blocking current SOTA pursuit; the existing top-K enumeration does not OOM on the configurations we care about (Bitcoin Alpha m=128, Slashdot k=4 m=128, Epinions m=64).

## Recommendation

Defer until a concrete need surfaces (e.g. k≥6 on Slashdot, m≥256, or running on memory-constrained hardware). The cheapest win when needed is "Pack HeapEntry" — predictable improvement, contained refactor, no architectural change.

The original Ádám conversation's framing was right in spirit (don't store every cycle) but the existing code already does this. The composable next idea — **learn k-enumeration via a separate architecture** — is the un-implemented one, and is independent of the heap-entry packing.
