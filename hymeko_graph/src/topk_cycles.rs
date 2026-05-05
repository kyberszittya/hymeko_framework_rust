//! Top-$k$ cycle enumeration: keep only the $k$ highest-scoring
//! cycles, never materialise the full set.
//!
//! Same DFS skeleton as [`crate::cycle_enum::enumerate_simple_cycles`],
//! but the output is a bounded min-heap of size $k$. Once the heap is
//! full, every closed cycle whose score is $\le$ the current heap
//! minimum is discarded immediately. The full enumeration cost is
//! still paid in the worst case (no admissible upper-bound is
//! assumed on the score function), but the **memory** cost stays
//! $O(k)$ — and the score-comparison short-circuit lets the caller
//! drop millions of cycles without ever cloning the path.
//!
//! ## When to use this over the full enumerator
//!
//! - You only want a ranked top-$k$ (e.g. "the 100 highest-balance
//!   cycles in a Slashdot-class graph"), not the entire cycle set.
//! - The full set would be too large to materialise (Slashdot's k=4
//!   set is 55.5 M cycles; collecting all to disk took 4 minutes —
//!   keeping only the top 1 000 takes the same DFS time but
//!   $\approx 50\,000\times$ less RAM).
//! - You want a quick heuristic-best for a downstream algorithm
//!   (cycle-aware MSG, balance-of-cycles GNN feature) that only
//!   reads the highest-scoring cycles.
//!
//! ## Score functions
//!
//! Several pre-built scorers live in [`scorers`]; pass any
//! `Fn(&[u32], &[i8]) -> f64` to [`enumerate_top_k_cycles`].

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use rayon::prelude::*;

use crate::pruner::{CyclePruner, NoOpPruner, PrunerDecision};
use crate::signed_graph::SignedGraph;

/// Sentinel value for "unreachable" / "not yet visited" in BFS-distance buffers.
const DIST_INF: u8 = u8::MAX;

/// BFS from `start` over the CSR, writing the minimum number of hops
/// from `start` to each vertex into `dist` (capped at `k_len`, since
/// any vertex farther than that can never close a $k$-cycle through
/// `start`).  Reuses provided scratch buffers — caller is responsible
/// for sizing them to `n_nodes`.
#[inline]
fn bfs_distances_capped(
    row_ptr: &[u32],
    col_idx: &[u32],
    start: u32,
    k_len: usize,
    dist: &mut [u8],
    frontier_a: &mut Vec<u32>,
    frontier_b: &mut Vec<u32>,
) {
    for d in dist.iter_mut() {
        *d = DIST_INF;
    }
    let cap = (k_len as u8).saturating_sub(1);
    dist[start as usize] = 0;
    frontier_a.clear();
    frontier_a.push(start);
    let mut depth: u8 = 0;
    while !frontier_a.is_empty() && depth < cap {
        depth += 1;
        frontier_b.clear();
        for &v in frontier_a.iter() {
            let s = row_ptr[v as usize] as usize;
            let e = row_ptr[v as usize + 1] as usize;
            for &nxt in &col_idx[s..e] {
                if dist[nxt as usize] == DIST_INF {
                    dist[nxt as usize] = depth;
                    frontier_b.push(nxt);
                }
            }
        }
        std::mem::swap(frontier_a, frontier_b);
    }
}

/// One entry in the bounded heap. We use the `Reverse`-of-score
/// trick to turn `BinaryHeap` (a max-heap) into a min-heap on score.
#[derive(Clone, Debug)]
struct HeapEntry {
    score: f64,
    cycle: Vec<u32>,
    signs: Vec<i8>,
}

impl Eq for HeapEntry {}
impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        // We never want NaN to crash the heap; treat it as -inf.
        let a = if self.score.is_nan() {
            f64::NEG_INFINITY
        } else {
            self.score
        };
        let b = if other.score.is_nan() {
            f64::NEG_INFINITY
        } else {
            other.score
        };
        a == b
    }
}
impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse on score: BinaryHeap pops the *largest*, so
        // reversing lets us pop the smallest.
        let a = if self.score.is_nan() {
            f64::NEG_INFINITY
        } else {
            self.score
        };
        let b = if other.score.is_nan() {
            f64::NEG_INFINITY
        } else {
            other.score
        };
        b.partial_cmp(&a).unwrap_or(Ordering::Equal)
    }
}

/// Output of [`enumerate_top_k_cycles`]: the $k$ best cycles in
/// descending score order, each as `(score, vertices, edge-signs)`.
pub type TopKCycle = (f64, Vec<u32>, Vec<i8>);

/// DFS-enumerate cycles of length `k_len`, keep the `k_keep`
/// highest-scoring ones according to `score`.
///
/// `score(vertices, edge_signs) -> f64`. Higher = better.
///
/// Returns the surviving cycles sorted by score descending.
pub fn enumerate_top_k_cycles<P, S>(
    graph: &SignedGraph,
    k_len: usize,
    pruner: &P,
    k_keep: usize,
    score: S,
) -> Vec<TopKCycle>
where
    P: CyclePruner,
    S: Fn(&[u32], &[i8]) -> f64,
{
    if k_len < 3 || k_keep == 0 {
        return Vec::new();
    }
    let (row_ptr, col_idx) = graph.build_csr();
    let sign_lookup = graph.build_sign_lookup();
    let n = graph.n_nodes as usize;
    let mut visited = vec![false; n];
    let mut path: Vec<u32> = Vec::with_capacity(k_len);
    let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::with_capacity(k_keep + 1);
    let mut dist: Vec<u8> = vec![DIST_INF; n];
    let mut bfs_a: Vec<u32> = Vec::new();
    let mut bfs_b: Vec<u32> = Vec::new();

    for start in 0..(n as u32) {
        bfs_distances_capped(&row_ptr, &col_idx, start, k_len,
                              &mut dist, &mut bfs_a, &mut bfs_b);
        path.clear();
        path.push(start);
        visited[start as usize] = true;
        dfs(
            start,
            &row_ptr,
            &col_idx,
            &sign_lookup,
            k_len,
            pruner,
            k_keep,
            &score,
            &mut path,
            &mut visited,
            &mut heap,
            &dist,
        );
        visited[start as usize] = false;
    }

    let mut out: Vec<TopKCycle> = heap
        .into_iter()
        .map(|e| (e.score, e.cycle, e.signs))
        .collect();
    // Heap iteration order isn't sorted; sort descending by score.
    out.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
    out
}

#[allow(clippy::too_many_arguments)]
fn dfs<P, S>(
    start: u32,
    row_ptr: &[u32],
    col_idx: &[u32],
    sign_lookup: &std::collections::HashMap<(u32, u32), i8>,
    k_len: usize,
    pruner: &P,
    k_keep: usize,
    score: &S,
    path: &mut Vec<u32>,
    visited: &mut [bool],
    heap: &mut BinaryHeap<HeapEntry>,
    dist: &[u8],
) where
    P: CyclePruner,
    S: Fn(&[u32], &[i8]) -> f64,
{
    if path.len() == k_len {
        // Closing-edge check.
        let last = *path.last().unwrap();
        let key = (last.min(start), last.max(start));
        if !sign_lookup.contains_key(&key) {
            return;
        }
        // Same canonicalisation rule as the full enumerator.
        if path.len() >= 3 && path[1] >= path[k_len - 1] {
            return;
        }
        // Materialise the edge-sign sequence.
        let mut signs: Vec<i8> = Vec::with_capacity(k_len);
        for j in 0..k_len {
            let u = path[j];
            let v = path[(j + 1) % k_len];
            let key = (u.min(v), u.max(v));
            signs.push(*sign_lookup.get(&key).expect("edge present"));
        }
        if pruner.emit_ok(path, &signs) != PrunerDecision::Accept {
            return;
        }
        // Score and push if competitive.
        let s = score(path, &signs);
        if heap.len() < k_keep {
            heap.push(HeapEntry {
                score: s,
                cycle: path.clone(),
                signs,
            });
        } else {
            // Heap min — under our Ord, peek() returns the smallest
            // score (because we inverted the comparator).
            let beat = heap
                .peek()
                .map(|min| s > min.score)
                .unwrap_or(true);
            if beat {
                heap.pop();
                heap.push(HeapEntry {
                    score: s,
                    cycle: path.clone(),
                    signs,
                });
            }
        }
        return;
    }
    let tail = *path.last().unwrap();
    let st = row_ptr[tail as usize] as usize;
    let en = row_ptr[tail as usize + 1] as usize;
    // Remaining edges from `nxt` back to `start` (closing inclusive).
    // After pushing nxt at current path.len()=d, depth becomes d+1.
    // We still owe (k_len - d) edges: (k_len - d - 1) interior +
    // 1 closing.  BFS distance from start to nxt is a lower bound
    // on those edges, so reject if dist[nxt] > k_len - d.
    let remaining_after = (k_len - path.len()) as u8;
    for &nxt in &col_idx[st..en] {
        if nxt < start {
            continue;
        }
        if visited[nxt as usize] {
            continue;
        }
        // BFS-distance pruning: nxt must be ≤ remaining_after hops
        // away from start, otherwise no possible close.
        if !dist.is_empty() {
            let d = dist[nxt as usize];
            if d == DIST_INF || d > remaining_after {
                continue;
            }
        }
        if pruner.extend_ok(path, nxt) == PrunerDecision::Reject {
            continue;
        }
        path.push(nxt);
        visited[nxt as usize] = true;
        dfs(start, row_ptr, col_idx, sign_lookup, k_len,
            pruner, k_keep, score, path, visited, heap, dist);
        path.pop();
        visited[nxt as usize] = false;
    }
}

/// Convenience: top-$k$ with no pruner.
pub fn enumerate_top_k_cycles_noprune<S>(
    graph: &SignedGraph,
    k_len: usize,
    k_keep: usize,
    score: S,
) -> Vec<TopKCycle>
where
    S: Fn(&[u32], &[i8]) -> f64,
{
    enumerate_top_k_cycles(graph, k_len, &NoOpPruner, k_keep, score)
}

// ─── Vertex-stratified top-K ────────────────────────────────────────

/// Vertex-stratified top-$m$ cycle enumeration: for **every** vertex
/// $v$, keep the $m$ highest-scoring cycles that pass through $v$.
///
/// The total cycle set returned is the *union* of those per-vertex
/// top-$m$ sets, with duplicates removed (a cycle that touches $k$
/// vertices appears in $k$ candidate heaps but is emitted once).
///
/// Bound: $|M_e| \le |V| \cdot m$ and **every** vertex is covered as
/// long as it sits on at least one cycle. This is the variant
/// that unblocks Epinions/Slashdot HSiKAN training: it caps the
/// cycle hyperedge incidence matrix per row instead of globally,
/// preserving vertex-uniform information density.
///
/// `score(vertices, edge_signs) -> f64`. Higher = better.
pub fn enumerate_top_k_per_vertex_cycles<P, S>(
    graph: &SignedGraph,
    k_len: usize,
    pruner: &P,
    m_per_vertex: usize,
    score: S,
) -> Vec<TopKCycle>
where
    P: CyclePruner,
    S: Fn(&[u32], &[i8]) -> f64,
{
    if k_len < 3 || m_per_vertex == 0 {
        return Vec::new();
    }
    let (row_ptr, col_idx) = graph.build_csr();
    let sign_lookup = graph.build_sign_lookup();
    let n = graph.n_nodes as usize;
    let mut visited = vec![false; n];
    let mut path: Vec<u32> = Vec::with_capacity(k_len);
    // One min-heap of size m_per_vertex per vertex.
    let mut per_vertex: Vec<BinaryHeap<HeapEntry>> =
        (0..n).map(|_| BinaryHeap::with_capacity(m_per_vertex + 1)).collect();
    let mut dist: Vec<u8> = vec![DIST_INF; n];
    let mut bfs_a: Vec<u32> = Vec::new();
    let mut bfs_b: Vec<u32> = Vec::new();

    for start in 0..(n as u32) {
        bfs_distances_capped(&row_ptr, &col_idx, start, k_len,
                              &mut dist, &mut bfs_a, &mut bfs_b);
        path.clear();
        path.push(start);
        visited[start as usize] = true;
        dfs_per_vertex(
            start,
            &row_ptr,
            &col_idx,
            &sign_lookup,
            k_len,
            pruner,
            m_per_vertex,
            &score,
            &mut path,
            &mut visited,
            &mut per_vertex,
            &dist,
        );
        visited[start as usize] = false;
    }

    // Union the per-vertex heaps and deduplicate by canonical cycle
    // representation (sorted tuple of vertices).
    let mut seen: std::collections::HashSet<Vec<u32>> = std::collections::HashSet::new();
    let mut out: Vec<TopKCycle> = Vec::new();
    for heap in per_vertex {
        for entry in heap {
            let mut canon = entry.cycle.clone();
            canon.sort();
            if seen.insert(canon) {
                out.push((entry.score, entry.cycle, entry.signs));
            }
        }
    }
    out.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
    out
}

#[allow(clippy::too_many_arguments)]
fn dfs_per_vertex<P, S>(
    start: u32,
    row_ptr: &[u32],
    col_idx: &[u32],
    sign_lookup: &std::collections::HashMap<(u32, u32), i8>,
    k_len: usize,
    pruner: &P,
    m_per_vertex: usize,
    score: &S,
    path: &mut Vec<u32>,
    visited: &mut [bool],
    per_vertex: &mut [BinaryHeap<HeapEntry>],
    dist: &[u8],
) where
    P: CyclePruner,
    S: Fn(&[u32], &[i8]) -> f64,
{
    if path.len() == k_len {
        let last = *path.last().unwrap();
        let key = (last.min(start), last.max(start));
        if !sign_lookup.contains_key(&key) {
            return;
        }
        if path.len() >= 3 && path[1] >= path[k_len - 1] {
            return;
        }
        let mut signs: Vec<i8> = Vec::with_capacity(k_len);
        for j in 0..k_len {
            let u = path[j];
            let v = path[(j + 1) % k_len];
            let key = (u.min(v), u.max(v));
            signs.push(*sign_lookup.get(&key).expect("edge present"));
        }
        if pruner.emit_ok(path, &signs) != PrunerDecision::Accept {
            return;
        }
        let s = score(path, &signs);
        // Push into every vertex's heap.
        for &v in path.iter() {
            let heap = &mut per_vertex[v as usize];
            if heap.len() < m_per_vertex {
                heap.push(HeapEntry {
                    score: s,
                    cycle: path.clone(),
                    signs: signs.clone(),
                });
            } else {
                let beat = heap.peek().map(|min| s > min.score).unwrap_or(true);
                if beat {
                    heap.pop();
                    heap.push(HeapEntry {
                        score: s,
                        cycle: path.clone(),
                        signs: signs.clone(),
                    });
                }
            }
        }
        return;
    }
    let tail = *path.last().unwrap();
    let st = row_ptr[tail as usize] as usize;
    let en = row_ptr[tail as usize + 1] as usize;
    let remaining_after = (k_len - path.len()) as u8;
    for &nxt in &col_idx[st..en] {
        if nxt < start {
            continue;
        }
        if visited[nxt as usize] {
            continue;
        }
        if !dist.is_empty() {
            let d = dist[nxt as usize];
            if d == DIST_INF || d > remaining_after {
                continue;
            }
        }
        if pruner.extend_ok(path, nxt) == PrunerDecision::Reject {
            continue;
        }
        path.push(nxt);
        visited[nxt as usize] = true;
        dfs_per_vertex(
            start, row_ptr, col_idx, sign_lookup, k_len, pruner,
            m_per_vertex, score, path, visited, per_vertex, dist,
        );
        path.pop();
        visited[nxt as usize] = false;
    }
}

/// Convenience: vertex-stratified top-$m$ with no pruner.
pub fn enumerate_top_k_per_vertex_cycles_noprune<S>(
    graph: &SignedGraph,
    k_len: usize,
    m_per_vertex: usize,
    score: S,
) -> Vec<TopKCycle>
where
    S: Fn(&[u32], &[i8]) -> f64,
{
    enumerate_top_k_per_vertex_cycles(graph, k_len, &NoOpPruner, m_per_vertex, score)
}

// ─── Rayon-parallel variants ────────────────────────────────────────

/// Parallel vertex-stratified top-$m$.  Each rayon thread takes a
/// disjoint slice of starts and accumulates its own per-vertex heap
/// array; at the end the per-thread heaps are merged into one heap
/// per vertex and the union is dedup'd.
///
/// Memory cost is `O(n_threads × n_vertices × m)` for the per-thread
/// heap arrays — fine on graphs up to a few million vertices with a
/// handful of cores. Lock-free, so contention is zero.
pub fn enumerate_top_k_per_vertex_cycles_par<P, S>(
    graph: &SignedGraph,
    k_len: usize,
    pruner: &P,
    m_per_vertex: usize,
    score: S,
) -> Vec<TopKCycle>
where
    P: CyclePruner + Sync,
    S: Fn(&[u32], &[i8]) -> f64 + Sync,
{
    if k_len < 3 || m_per_vertex == 0 {
        return Vec::new();
    }
    let (row_ptr, col_idx) = graph.build_csr();
    let sign_lookup = graph.build_sign_lookup();
    let n = graph.n_nodes as usize;

    // Each thread collects its own per-vertex heaps via fold/reduce.
    let final_heaps = (0..n as u32)
        .into_par_iter()
        .fold(
            || vec![BinaryHeap::<HeapEntry>::new(); n],
            |mut per_vertex, start| {
                let mut visited = vec![false; n];
                let mut path: Vec<u32> = Vec::with_capacity(k_len);
                let mut dist: Vec<u8> = vec![DIST_INF; n];
                let mut bfs_a: Vec<u32> = Vec::new();
                let mut bfs_b: Vec<u32> = Vec::new();
                bfs_distances_capped(
                    &row_ptr, &col_idx, start, k_len,
                    &mut dist, &mut bfs_a, &mut bfs_b,
                );
                path.push(start);
                visited[start as usize] = true;
                dfs_per_vertex(
                    start,
                    &row_ptr,
                    &col_idx,
                    &sign_lookup,
                    k_len,
                    pruner,
                    m_per_vertex,
                    &score,
                    &mut path,
                    &mut visited,
                    &mut per_vertex,
                    &dist,
                );
                per_vertex
            },
        )
        .reduce(
            || vec![BinaryHeap::<HeapEntry>::new(); n],
            |mut a, b| {
                for (av, bv) in a.iter_mut().zip(b.into_iter()) {
                    for entry in bv {
                        if av.len() < m_per_vertex {
                            av.push(entry);
                        } else {
                            let beat = av
                                .peek()
                                .map(|min| entry.score > min.score)
                                .unwrap_or(true);
                            if beat {
                                av.pop();
                                av.push(entry);
                            }
                        }
                    }
                }
                a
            },
        );

    // Dedup union by canonical vertex tuple.
    let mut seen: std::collections::HashSet<Vec<u32>> = std::collections::HashSet::new();
    let mut out: Vec<TopKCycle> = Vec::new();
    for heap in final_heaps {
        for entry in heap {
            let mut canon = entry.cycle.clone();
            canon.sort();
            if seen.insert(canon) {
                out.push((entry.score, entry.cycle, entry.signs));
            }
        }
    }
    out.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
    out
}

/// Convenience: parallel vertex-stratified top-$m$ with no pruner.
pub fn enumerate_top_k_per_vertex_cycles_par_noprune<S>(
    graph: &SignedGraph,
    k_len: usize,
    m_per_vertex: usize,
    score: S,
) -> Vec<TopKCycle>
where
    S: Fn(&[u32], &[i8]) -> f64 + Sync,
{
    enumerate_top_k_per_vertex_cycles_par(graph, k_len, &NoOpPruner, m_per_vertex, score)
}

/// Parallel global top-$K$.  Each thread keeps a local min-heap of
/// size $K$; at the end the per-thread heaps are merged.
pub fn enumerate_top_k_cycles_par<P, S>(
    graph: &SignedGraph,
    k_len: usize,
    pruner: &P,
    k_keep: usize,
    score: S,
) -> Vec<TopKCycle>
where
    P: CyclePruner + Sync,
    S: Fn(&[u32], &[i8]) -> f64 + Sync,
{
    if k_len < 3 || k_keep == 0 {
        return Vec::new();
    }
    let (row_ptr, col_idx) = graph.build_csr();
    let sign_lookup = graph.build_sign_lookup();
    let n = graph.n_nodes as usize;

    let final_heap = (0..n as u32)
        .into_par_iter()
        .fold(
            || BinaryHeap::<HeapEntry>::with_capacity(k_keep + 1),
            |mut heap, start| {
                let mut visited = vec![false; n];
                let mut path: Vec<u32> = Vec::with_capacity(k_len);
                let mut dist: Vec<u8> = vec![DIST_INF; n];
                let mut bfs_a: Vec<u32> = Vec::new();
                let mut bfs_b: Vec<u32> = Vec::new();
                bfs_distances_capped(
                    &row_ptr, &col_idx, start, k_len,
                    &mut dist, &mut bfs_a, &mut bfs_b,
                );
                path.push(start);
                visited[start as usize] = true;
                dfs(
                    start,
                    &row_ptr,
                    &col_idx,
                    &sign_lookup,
                    k_len,
                    pruner,
                    k_keep,
                    &score,
                    &mut path,
                    &mut visited,
                    &mut heap,
                    &dist,
                );
                heap
            },
        )
        .reduce(
            || BinaryHeap::<HeapEntry>::with_capacity(k_keep + 1),
            |mut a, b| {
                for entry in b {
                    if a.len() < k_keep {
                        a.push(entry);
                    } else {
                        let beat = a
                            .peek()
                            .map(|min| entry.score > min.score)
                            .unwrap_or(true);
                        if beat {
                            a.pop();
                            a.push(entry);
                        }
                    }
                }
                a
            },
        );

    let mut out: Vec<TopKCycle> = final_heap
        .into_iter()
        .map(|e| (e.score, e.cycle, e.signs))
        .collect();
    out.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
    out
}

/// Convenience: parallel global top-$K$ with no pruner.
pub fn enumerate_top_k_cycles_par_noprune<S>(
    graph: &SignedGraph,
    k_len: usize,
    k_keep: usize,
    score: S,
) -> Vec<TopKCycle>
where
    S: Fn(&[u32], &[i8]) -> f64 + Sync,
{
    enumerate_top_k_cycles_par(graph, k_len, &NoOpPruner, k_keep, score)
}

// ─── Pre-built scorers ──────────────────────────────────────────────

/// Stock heuristics that take `(vertices, edge_signs)` and return
/// $f64$. All are pure functions — pass them directly to
/// [`enumerate_top_k_cycles`].
pub mod scorers {
    /// Sign-product magnitude: $\bigl|\prod_i s_i\bigr|$.
    /// Always $1$ on signed graphs; not useful in isolation, but
    /// included for completeness.
    pub fn sign_product_abs(_vs: &[u32], signs: &[i8]) -> f64 {
        signs.iter().map(|&s| s as f64).product::<f64>().abs()
    }

    /// Cartwright–Harary balance: $+1$ if cycle is balanced,
    /// $-1$ if unbalanced. Top-$k$ with this scorer surfaces the
    /// balanced cycles first; combine with $-x$ for unbalanced.
    pub fn balance(_vs: &[u32], signs: &[i8]) -> f64 {
        signs.iter().map(|&s| s as f64).product::<f64>()
    }

    /// Number of negative edges in the cycle, normalised by $k$.
    /// High score = mostly-negative cycle (Heider's "all-enemy"
    /// triad in the limit).
    pub fn fraction_negative(_vs: &[u32], signs: &[i8]) -> f64 {
        if signs.is_empty() {
            return 0.0;
        }
        let n_neg = signs.iter().filter(|&&s| s < 0).count() as f64;
        n_neg / signs.len() as f64
    }

    /// "Lowest-vertex" heuristic: prefer cycles whose canonical
    /// rotation starts at a small index (pulls cycles touching
    /// the densely-connected hubs in many real graphs).
    pub fn low_root(vs: &[u32], _signs: &[i8]) -> f64 {
        vs.first().map(|v| -(*v as f64)).unwrap_or(0.0)
    }

    /// Returns a closure that scores by the negation of the sum of
    /// per-vertex weights — top-$k$ then picks the cycles whose
    /// vertex set has the *lowest* total weight (e.g. lowest cost
    /// when weights are vertex costs).
    pub fn min_vertex_weight(weights: Vec<f64>) -> impl Fn(&[u32], &[i8]) -> f64 {
        move |vs: &[u32], _signs: &[i8]| {
            let s: f64 = vs
                .iter()
                .map(|&v| weights.get(v as usize).copied().unwrap_or(0.0))
                .sum();
            -s
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::balance::{BalanceMode, CartwrightHararyPruner};
    use crate::pruner::NoOpPruner;
    use crate::signed_graph::SignedGraph;

    /// Build a graph with three triangles of known balance:
    ///   Δ_pos:  signs (+, +, +) ⇒ balanced.
    ///   Δ_mix1: signs (+, +, -) ⇒ unbalanced.
    ///   Δ_mix2: signs (+, -, -) ⇒ balanced.
    fn build_three_triangles() -> SignedGraph {
        // Triangles: 0-1-2, 3-4-5, 6-7-8.
        SignedGraph::from_parts(
            9,
            &[0, 1, 2, 3, 4, 5, 6, 7, 8],
            &[1, 2, 0, 4, 5, 3, 7, 8, 6],
            &[1, 1, 1,    1, 1, -1,   1, -1, -1],
        )
    }

    #[test]
    fn top_k_keeps_only_k_cycles() {
        let g = build_three_triangles();
        let out = enumerate_top_k_cycles_noprune(
            &g, 3, 2, scorers::balance,
        );
        assert_eq!(out.len(), 2);
        // Both balanced triangles should win.
        assert!(out.iter().all(|(s, _, _)| (*s - 1.0).abs() < 1e-9));
    }

    #[test]
    fn top_k_full_request_returns_all() {
        // Asking for k_keep = 5 when only 3 cycles exist returns all 3.
        let g = build_three_triangles();
        let out = enumerate_top_k_cycles_noprune(
            &g, 3, 5, scorers::balance,
        );
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn top_k_descending_order() {
        let g = build_three_triangles();
        let out = enumerate_top_k_cycles_noprune(
            &g, 3, 3, scorers::fraction_negative,
        );
        assert_eq!(out.len(), 3);
        // fraction_negative: Δ_mix2 (2/3), Δ_mix1 (1/3), Δ_pos (0).
        assert!(out[0].0 >= out[1].0);
        assert!(out[1].0 >= out[2].0);
    }

    #[test]
    fn top_k_with_pruner_composes() {
        // Same graph, ask Cartwright-Harary OnlyBalanced + top-1 by
        // low_root: should return the lowest-rooted balanced
        // triangle (Δ_pos at vertex 0).
        let g = build_three_triangles();
        let out = enumerate_top_k_cycles(
            &g, 3,
            &CartwrightHararyPruner { mode: BalanceMode::OnlyBalanced },
            1, scorers::low_root,
        );
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].1, vec![0, 1, 2]);
    }

    #[test]
    fn per_vertex_top_k_covers_every_vertex() {
        // Graph with two disjoint triangles 0-1-2 and 3-4-5.
        let g = SignedGraph::from_parts(
            6,
            &[0, 1, 2, 3, 4, 5],
            &[1, 2, 0, 4, 5, 3],
            &[1; 6],
        );
        let out = enumerate_top_k_per_vertex_cycles_noprune(
            &g, 3, 1, scorers::balance,
        );
        // Two triangles, two unique cycles.
        assert_eq!(out.len(), 2);
        // Every vertex appears in at least one returned cycle.
        let mut touched = vec![false; 6];
        for (_, vs, _) in &out {
            for &v in vs {
                touched[v as usize] = true;
            }
        }
        assert!(touched.iter().all(|&b| b),
                "vertex-stratified top-K must cover every vertex");
    }

    #[test]
    fn per_vertex_top_k_dedups_shared_cycles() {
        // A single triangle 0-1-2: every vertex's heap will hold the
        // same cycle, but the union must dedup to one entry.
        let g = SignedGraph::from_parts(
            3,
            &[0, 1, 2],
            &[1, 2, 0],
            &[1; 3],
        );
        let out = enumerate_top_k_per_vertex_cycles_noprune(
            &g, 3, 5, scorers::balance,
        );
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn top_k_min_vertex_weight_picks_cheapest() {
        // 4-vertex graph with two triangles 0-1-2 and 1-2-3.
        // Make vertex weights so 1-2-3 is cheaper: w = [10, 1, 1, 1].
        let g = SignedGraph::from_parts(
            4,
            &[0, 1, 2, 1, 2],
            &[1, 2, 0, 3, 3],
            &[1; 5],
        );
        let weights = vec![10.0, 1.0, 1.0, 1.0];
        let scorer = scorers::min_vertex_weight(weights);
        let out = enumerate_top_k_cycles(
            &g, 3, &NoOpPruner, 1, scorer,
        );
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].1, vec![1, 2, 3]);
    }
}
