use pyo3::prelude::*;
use numpy::{PyArray2, IntoPyArray};
use ndarray::Array2;
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};

fn build_csr(edges: &[(u32, u32)], n_nodes: usize, directed: bool)
    -> (Vec<u32>, Vec<u32>)
{
    // Tightly-packed CSR. Each row is sorted + dedup'd; row_ptr[i+1] -
    // row_ptr[i] is exactly the unique-neighbour count for vertex i. No
    // u32::MAX sentinel; `neighbours` returns a clean slice in O(1).
    //
    // ``directed = false`` (default): store both (u,v) and (v,u).
    // ``directed = true``: store only (u,v) — out-edges only.
    let mut deg = vec![0u32; n_nodes];
    for &(u, v) in edges {
        deg[u as usize] += 1;
        if !directed { deg[v as usize] += 1; }
    }
    let mut raw_ptr = vec![0u32; n_nodes + 1];
    for i in 0..n_nodes {
        raw_ptr[i + 1] = raw_ptr[i] + deg[i];
    }
    let total = raw_ptr[n_nodes] as usize;
    let mut raw_col = vec![0u32; total];
    let mut cursor = raw_ptr.clone();
    for &(u, v) in edges {
        let pu = cursor[u as usize] as usize;
        raw_col[pu] = v;
        cursor[u as usize] += 1;
        if !directed {
            let pv = cursor[v as usize] as usize;
            raw_col[pv] = u;
            cursor[v as usize] += 1;
        }
    }
    let mut col_idx: Vec<u32> = Vec::with_capacity(total);
    let mut row_ptr: Vec<u32> = Vec::with_capacity(n_nodes + 1);
    row_ptr.push(0);
    for i in 0..n_nodes {
        let s = raw_ptr[i] as usize;
        let e = raw_ptr[i + 1] as usize;
        raw_col[s..e].sort_unstable();
        let mut prev: i64 = -1;
        for &v in &raw_col[s..e] {
            if v as i64 != prev {
                col_idx.push(v);
                prev = v as i64;
            }
        }
        row_ptr.push(col_idx.len() as u32);
    }
    (row_ptr, col_idx)
}

#[inline]
fn neighbours<'a>(row_ptr: &'a [u32], col_idx: &'a [u32], v: u32) -> &'a [u32] {
    let s = row_ptr[v as usize] as usize;
    let e = row_ptr[v as usize + 1] as usize;
    &col_idx[s..e]
}

#[inline]
fn has_edge(row_ptr: &[u32], col_idx: &[u32], u: u32, v: u32) -> bool {
    neighbours(row_ptr, col_idx, u).binary_search(&v).is_ok()
}

// ============================================================================
// Bitset visited (1 bit/vertex). Replaces Vec<bool> on the exact-DFS hot
// path — 8× smaller, fits in L1 for graphs up to ~500k nodes.
// ============================================================================

#[inline] fn bs_words(n: usize) -> usize { n.div_ceil(64) }
#[inline] fn bs_get(bits: &[u64], v: u32) -> bool {
    (bits[(v >> 6) as usize] >> (v & 63)) & 1 == 1
}
#[inline] fn bs_set(bits: &mut [u64], v: u32) {
    bits[(v >> 6) as usize] |= 1u64 << (v & 63);
}
#[inline] fn bs_clear(bits: &mut [u64], v: u32) {
    bits[(v >> 6) as usize] &= !(1u64 << (v & 63));
}

// ============================================================================
// BFS distance precompute, smallest-vertex-root domain.
//
// For undirected enumeration the DFS only descends through vertices >= start.
// We do a single-source BFS from `start` restricted to that subgraph and
// store `dist[v]` (capped at u8::MAX = unreachable). The exact DFS then
// drops any candidate `nxt` with dist[nxt] > k - path.len(), because that
// candidate cannot reach `start` in the remaining hops to close a k-cycle.
//
// Only sound for the undirected case: a directed cycle needs an in-edge
// to start, so out-BFS from start is the wrong distance. The directed
// path skips BFS pruning (dist passed in as an empty slice).
//
// Caller-provided `dist` scratch buffer is reset to u8::MAX inside; reuse
// across starts within a worker avoids the per-start allocation that
// otherwise dominates allocator pressure on big graphs.
// ============================================================================

fn bfs_distances_into(
    row_ptr: &[u32],
    col_idx: &[u32],
    start: u32,
    n_nodes: usize,
    max_dist: u8,
    dist: &mut [u8],
    frontier: &mut Vec<u32>,
    next: &mut Vec<u32>,
) {
    debug_assert_eq!(dist.len(), n_nodes);
    dist.fill(u8::MAX);
    dist[start as usize] = 0;
    frontier.clear();
    next.clear();
    frontier.push(start);
    let mut d: u8 = 0;
    while !frontier.is_empty() && d < max_dist {
        for &u in frontier.iter() {
            for &w in neighbours(row_ptr, col_idx, u) {
                if w < start { continue; }
                if dist[w as usize] != u8::MAX { continue; }
                dist[w as usize] = d + 1;
                next.push(w);
            }
        }
        std::mem::swap(frontier, next);
        next.clear();
        d += 1;
    }
}

/// Cycle accumulator. Three modes:
///   Full       — collect every cycle (flat Vec<u32>, stride k)
///   Reservoir  — Vitter Algorithm R; unbiased sample of `cap` cycles,
///                but DFS still enumerates the full cycle space
///   EarlyStop  — keep the first `cap` cycles encountered, then signal
///                the DFS to terminate. Biased toward cycles starting
///                at small-indexed vertices, but DFS time is bounded
///                by O(cap · DFS-cost-per-cycle) instead of O(total-
///                cycles · DFS-cost-per-cycle). Order-of-magnitude
///                speedup for high-arity cycles on dense graphs.
enum Sink {
    Full(Vec<u32>),
    Reservoir {
        buf: Vec<u32>,
        cap: usize,
        seen: usize,
        rng_state: u64,
    },
    /// EarlyStop now consults a shared atomic counter so all parallel
    /// workers stop as soon as the *global* sample reaches cap, not when
    /// each per-segment sink fills its own cap (which would multiply
    /// the wasted DFS work by n_threads).
    EarlyStop {
        buf: Vec<u32>,
        cap: usize,
        global: std::sync::Arc<AtomicUsize>,
    },
}

impl Sink {
    fn new_full() -> Self { Sink::Full(Vec::new()) }
    fn new_reservoir(cap: usize, seed: u64) -> Self {
        Sink::Reservoir {
            buf: Vec::new(),
            cap,
            seen: 0,
            rng_state: seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407),
        }
    }
    fn new_early_stop(cap: usize, global: std::sync::Arc<AtomicUsize>) -> Self {
        Sink::EarlyStop { buf: Vec::new(), cap, global }
    }

    /// LCG step → next u64. Uses Knuth's MMIX constants.
    #[inline]
    fn next_u64(state: &mut u64) -> u64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *state
    }

    /// Returns true if the DFS should keep exploring; false if the
    /// sink is full and further cycles would be discarded (early-stop).
    fn offer(&mut self, path: &[u32]) -> bool {
        match self {
            Sink::Full(buf) => {
                buf.extend_from_slice(path);
                true
            }
            Sink::Reservoir { buf, cap, seen, rng_state } => {
                let k = path.len();
                if *seen < *cap {
                    buf.extend_from_slice(path);
                } else {
                    let r = Self::next_u64(rng_state) >> 33;
                    let j = (r as usize) % (*seen + 1);
                    if j < *cap {
                        let dst = j * k;
                        buf[dst..dst + k].copy_from_slice(path);
                    }
                }
                *seen += 1;
                true
            }
            Sink::EarlyStop { buf, cap, global } => {
                // Atomically claim a slot. If the global counter is
                // already >= cap, drop this cycle and tell the DFS to
                // stop. Relaxed ordering is enough — we only need
                // approximate consensus on "have we hit cap yet".
                let claimed = global.fetch_add(1, Ordering::Relaxed);
                if claimed < *cap {
                    buf.extend_from_slice(path);
                    // Keep going if the global counter hasn't yet
                    // committed to cap. After the increment, claimed+1
                    // is what's been issued; if that equals cap, the
                    // very next caller will be told to bail.
                    claimed + 1 < *cap
                } else {
                    false
                }
            }
        }
    }

    fn into_flat(self) -> Vec<u32> {
        match self {
            Sink::Full(b) => b,
            Sink::Reservoir { buf, .. } => buf,
            Sink::EarlyStop { buf, .. } => buf,
        }
    }
}

fn dfs_recurse(
    row_ptr: &[u32],
    col_idx: &[u32],
    start: u32,
    k: usize,
    directed: bool,
    path: &mut Vec<u32>,
    visited: &mut [u64],
    dist: &[u8],
    sink: &mut Sink,
) -> bool {
    if path.len() == k {
        let last = *path.last().unwrap();
        if has_edge(row_ptr, col_idx, last, start) {
            if directed || path[1] < path[k - 1] {
                return sink.offer(path);
            }
        }
        return true;
    }
    let tail = *path.last().unwrap();
    let max_remaining = (k - path.len()) as u8;
    let prune = !dist.is_empty();
    for &nxt in neighbours(row_ptr, col_idx, tail) {
        if nxt < start { continue; }
        if bs_get(visited, nxt) { continue; }
        if prune && dist[nxt as usize] > max_remaining { continue; }
        path.push(nxt);
        bs_set(visited, nxt);
        let cont = dfs_recurse(row_ptr, col_idx, start, k, directed,
                                 path, visited, dist, sink);
        path.pop();
        bs_clear(visited, nxt);
        if !cont { return false; }
    }
    true
}

fn dfs_from(
    row_ptr: &[u32],
    col_idx: &[u32],
    start: u32,
    k: usize,
    directed: bool,
    visited: &mut [u64],
    path: &mut Vec<u32>,
    dist: &[u8],
    sink: &mut Sink,
) -> bool {
    // Returns false if the sink signalled "full, stop the DFS".
    //
    // UNDIRECTED: each cycle (v_0=start, v_1, ..., v_{k-1}) is enumerated
    // twice (forward + reverse from the same root). We deduplicate on the
    // fly by emitting only the orientation with v_1 < v_{k-1}.
    //
    // DIRECTED: each cycle has only one valid traversal direction (out-
    // edges only), so each cycle is emitted exactly once from its
    // smallest-vertex root. No tiebreak needed.
    path.push(start);
    bs_set(visited, start);
    let cont = dfs_recurse(row_ptr, col_idx, start, k, directed,
                             path, visited, dist, sink);
    path.pop();
    bs_clear(visited, start);
    cont
}

/// Run DFS from a (start, first_hop) pair — used as a parallel work unit
/// to expose intra-root parallelism. The outer level for-loop in
/// `enumerate_parallel` sweeps every (start, first_hop > start) so that
/// the heavy-root vertex-0 DFS gets split into deg(0) independent tasks
/// that rayon can distribute.
fn dfs_from_pair(
    row_ptr: &[u32],
    col_idx: &[u32],
    start: u32,
    first_hop: u32,
    k: usize,
    directed: bool,
    visited: &mut [u64],
    path: &mut Vec<u32>,
    dist: &[u8],
    sink: &mut Sink,
) -> bool {
    path.push(start);
    bs_set(visited, start);
    path.push(first_hop);
    bs_set(visited, first_hop);

    let cont = if k == 2 {
        let last = *path.last().unwrap();
        if has_edge(row_ptr, col_idx, last, start) {
            if directed || path[1] < path[k - 1] {
                sink.offer(path)
            } else { true }
        } else { true }
    } else {
        dfs_recurse(row_ptr, col_idx, start, k, directed,
                      path, visited, dist, sink)
    };

    bs_clear(visited, first_hop);
    path.pop();
    bs_clear(visited, start);
    path.pop();
    cont
}

/// Per-thread merge: stratified for Reservoir, concat for Full,
/// concat-and-truncate for EarlyStop. Approximately preserves the
/// per-mode invariant (uniform sample / full enumeration / first-cap).
fn merge_sinks(a: Sink, b: Sink, k: usize) -> Sink {
    match (a, b) {
        (Sink::Full(mut buf_a), Sink::Full(buf_b)) => {
            buf_a.extend_from_slice(&buf_b);
            Sink::Full(buf_a)
        }
        (Sink::Reservoir { buf: ba, cap, seen: sa, rng_state: rsa },
         Sink::Reservoir { buf: bb, seen: sb, rng_state: rsb, .. }) => {
            // Stratified merge. Each stratum (a, b) is itself a uniform
            // sample of its observed `seen_t` cycles. We allocate the
            // global cap proportionally: target_a = cap * sa / (sa+sb).
            // Items in a Vitter reservoir are uniformly distributed
            // across reservoir slots, so taking the *first* target_a is
            // a uniform sub-sample of the stratum.
            let new_seen = sa + sb;
            let avail_a = ba.len() / k;
            let avail_b = bb.len() / k;
            let target_a = if new_seen == 0 {
                0
            } else {
                ((cap as u128 * sa as u128) / new_seen as u128) as usize
            }.min(avail_a);
            let target_b = cap.saturating_sub(target_a).min(avail_b);
            // Backfill if one side underdelivered (rare, only at boundary).
            let target_a = (cap.saturating_sub(target_b)).min(avail_a).max(target_a);
            let mut new_buf = Vec::with_capacity((target_a + target_b) * k);
            new_buf.extend_from_slice(&ba[..target_a * k]);
            new_buf.extend_from_slice(&bb[..target_b * k]);
            Sink::Reservoir {
                buf: new_buf,
                cap,
                seen: new_seen,
                rng_state: rsa ^ rsb.rotate_left(13),
            }
        }
        (Sink::EarlyStop { mut buf, cap, global },
         Sink::EarlyStop { buf: bb, .. }) => {
            let needed = cap.saturating_sub(buf.len() / k) * k;
            let take = needed.min(bb.len());
            buf.extend_from_slice(&bb[..take]);
            Sink::EarlyStop { buf, cap, global }
        }
        _ => unreachable!("attempted to merge sinks of mismatched modes"),
    }
}

fn make_thread_sink(
    max_cycles: Option<usize>,
    early_stop: Option<&std::sync::Arc<AtomicUsize>>,
    seed: u64,
) -> Sink {
    match (max_cycles, early_stop) {
        (Some(cap), Some(global)) => Sink::new_early_stop(cap, global.clone()),
        (Some(cap), None) => Sink::new_reservoir(cap, seed),
        (None, _) => Sink::new_full(),
    }
}

fn make_identity_sink(
    max_cycles: Option<usize>,
    early_stop: Option<&std::sync::Arc<AtomicUsize>>,
) -> Sink {
    // Identity for the parallel `reduce`. Empty buffer, same mode.
    match (max_cycles, early_stop) {
        (Some(cap), Some(global)) => Sink::new_early_stop(cap, global.clone()),
        (Some(cap), None) => Sink::Reservoir {
            buf: Vec::new(),
            cap,
            seen: 0,
            rng_state: 0,
        },
        (None, _) => Sink::new_full(),
    }
}

/// Run DFS in parallel over starting vertices using rayon, then
/// merge per-thread sinks. Falls back to serial when n_threads == 1.
fn enumerate_parallel(
    row_ptr: &[u32],
    col_idx: &[u32],
    n_nodes: usize,
    k: usize,
    directed: bool,
    max_cycles: Option<usize>,
    seed: u64,
    early_stop: bool,
    n_threads: Option<usize>,
) -> Sink {
    let pool = if let Some(nt) = n_threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(nt.max(1))
            .build()
            .ok()
    } else {
        None
    };

    // Shared atomic counter for early-stop coordination across workers.
    let global_es = if early_stop && max_cycles.is_some() {
        Some(std::sync::Arc::new(AtomicUsize::new(0)))
    } else {
        None
    };

    // Parallelise by `start` so the per-start BFS distances (used for
    // closure-pruning the DFS in the undirected case) are computed once
    // per start and reused across that start's first-hop expansion. For
    // each start we sequentially run all (start, first_hop > start) DFSes;
    // the load-imbalance from heavy vertex 0 is largely offset by the
    // BFS pruning shrinking its DFS tree.
    let starts: Vec<u32> = (0..n_nodes as u32).collect();
    let words = bs_words(n_nodes);

    let do_work = || {
        starts
            .par_iter()
            .copied()
            .fold(
                || {
                    let tid = rayon::current_thread_index().unwrap_or(0) as u64;
                    let s = make_thread_sink(
                        max_cycles, global_es.as_ref(),
                        seed.wrapping_add(tid)
                            .wrapping_mul(0x9e37_79b9_7f4a_7c15),
                    );
                    // Per-fold-segment scratch: visited bitset, BFS dist
                    // buffer and BFS frontier/next queues. Reused across
                    // every start the segment processes — eliminates the
                    // n_nodes × n_nodes bytes of allocator churn that
                    // dominated wall-clock at high n.
                    let dist: Vec<u8> = if directed {
                        Vec::new()
                    } else {
                        vec![u8::MAX; n_nodes]
                    };
                    (vec![0u64; words],
                     Vec::with_capacity(k),
                     dist,
                     Vec::<u32>::new(),
                     Vec::<u32>::new(),
                     s)
                },
                |(mut visited, mut path, mut dist, mut bfs_a, mut bfs_b, mut sink), start| {
                    // Early-stop short-circuit at the segment level.
                    if let Some(g) = global_es.as_ref() {
                        if let Some(cap) = max_cycles {
                            if g.load(Ordering::Relaxed) >= cap {
                                return (visited, path, dist, bfs_a, bfs_b, sink);
                            }
                        }
                    }
                    if !directed {
                        bfs_distances_into(
                            row_ptr, col_idx, start, n_nodes, k as u8,
                            &mut dist, &mut bfs_a, &mut bfs_b,
                        );
                    }
                    for &first_hop in neighbours(row_ptr, col_idx, start) {
                        if first_hop <= start { continue; }
                        // Bound check: first_hop must lie within k-1 hops
                        // of start (it does — it's at distance 1 — but the
                        // check is free and guards future API changes).
                        if !directed && dist[first_hop as usize] as usize > k - 1 {
                            continue;
                        }
                        let cont = dfs_from_pair(
                            row_ptr, col_idx, start, first_hop, k, directed,
                            &mut visited, &mut path, &dist, &mut sink,
                        );
                        if !cont { break; }
                    }
                    (visited, path, dist, bfs_a, bfs_b, sink)
                },
            )
            .map(|(_, _, _, _, _, s)| s)
            .reduce(
                || make_identity_sink(max_cycles, global_es.as_ref()),
                |a, b| merge_sinks(a, b, k),
            )
    };

    if let Some(p) = pool {
        p.install(do_work)
    } else {
        do_work()
    }
}

/// Materialise the flat stride-k buffer as a numpy 2-D ndarray (N, k).
/// This is the single boundary-crossing allocation; no per-cycle Python
/// objects are created. Truncates to `cap` rows for the reservoir/early-
/// stop modes if oversampled by merge slack.
fn flat_to_pyarray2<'py>(
    py: Python<'py>,
    buf: Vec<u32>,
    k: usize,
    cap: Option<usize>,
) -> PyResult<Py<PyArray2<u32>>> {
    let mut n = buf.len() / k;
    if let Some(c) = cap { if n > c { n = c; } }
    let total = n * k;
    let buf = if buf.len() == total { buf } else { buf[..total].to_vec() };
    let arr = Array2::from_shape_vec((n, k), buf)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
            format!("ndarray reshape failed: {e}")))?;
    Ok(arr.into_pyarray(py).unbind())
}

#[pyfunction]
#[pyo3(signature = (edges_u, edges_v, n_nodes, k,
                      max_cycles=None, seed=0, directed=false,
                      early_stop=false, n_threads=None))]
pub fn enumerate_k_cycles_rs(
    py: Python<'_>,
    edges_u: Vec<u32>,
    edges_v: Vec<u32>,
    n_nodes: usize,
    k: usize,
    max_cycles: Option<usize>,
    seed: u64,
    directed: bool,
    early_stop: bool,
    n_threads: Option<usize>,
) -> PyResult<Py<PyArray2<u32>>> {
    if k < 3 {
        // Empty (0, k) ndarray.
        let arr = Array2::<u32>::zeros((0, k.max(1)));
        return Ok(arr.into_pyarray(py).unbind());
    }
    if edges_u.len() != edges_v.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "edges_u and edges_v must have the same length",
        ));
    }
    let edges: Vec<(u32, u32)> = edges_u
        .into_iter()
        .zip(edges_v.into_iter())
        .collect();
    let (row_ptr, col_idx) = build_csr(&edges, n_nodes, directed);

    // Release the GIL for the duration of the cycle search so rayon
    // worker threads can actually run concurrently. They only touch
    // pure-Rust state (row_ptr, col_idx, sinks) — no Python objects.
    let sink = py.detach(|| {
        if matches!(n_threads, Some(1)) {
            // Serial path — kept for correctness verification and as a
            // fallback for tiny graphs where parallel overhead dominates.
            let global_es = if early_stop && max_cycles.is_some() {
                Some(std::sync::Arc::new(AtomicUsize::new(0)))
            } else { None };
            let mut s = match (max_cycles, &global_es) {
                (Some(cap), Some(g)) => Sink::new_early_stop(cap, g.clone()),
                (Some(cap), None) => Sink::new_reservoir(cap, seed),
                (None, _) => Sink::new_full(),
            };
            let mut visited: Vec<u64> = vec![0u64; bs_words(n_nodes)];
            let mut path: Vec<u32> = Vec::with_capacity(k);
            let mut dist: Vec<u8> = if directed {
                Vec::new()
            } else { vec![u8::MAX; n_nodes] };
            let mut bfs_a: Vec<u32> = Vec::new();
            let mut bfs_b: Vec<u32> = Vec::new();
            for start in 0..n_nodes as u32 {
                if !directed {
                    bfs_distances_into(
                        &row_ptr, &col_idx, start, n_nodes, k as u8,
                        &mut dist, &mut bfs_a, &mut bfs_b,
                    );
                }
                let cont = dfs_from(&row_ptr, &col_idx, start, k, directed,
                                      &mut visited, &mut path, &dist, &mut s);
                if !cont { break; }
            }
            s
        } else {
            enumerate_parallel(&row_ptr, &col_idx, n_nodes, k, directed,
                                max_cycles, seed, early_stop, n_threads)
        }
    });

    let buf = sink.into_flat();
    flat_to_pyarray2(py, buf, k, max_cycles)
}

// ============================================================================
// Color-coding sampler (Alon-Yuster-Zwick 1995).
//
// Idea: assign each vertex a random color in [0, k). A k-cycle is
// "rainbow" iff all k vertices have distinct colors. Probability of any
// fixed k-cycle being rainbow is k! / k^k. The DFS only extends paths to
// unused-color neighbors, which prunes the search tree by the same factor
// the rainbow probability is reduced — so per-coloring cost is proportional
// to (rainbow cycles found) / (total cycles), not to total cycles.
//
// To collect a target sample, we run multiple independent colorings until
// the dedup'd sample reaches `target_cycles` or `max_colorings` is hit.
// Cycles found across colorings are deduplicated by their canonical form
// (smallest rotation, lex-smallest direction).
// ============================================================================

/// Canonical form of an undirected k-cycle: rotate so the smallest vertex
/// is first, then choose the lex-smaller of (forward, reverse) traversal.
fn canonical_cycle(path: &[u32]) -> Vec<u32> {
    let k = path.len();
    let min_pos = (0..k).min_by_key(|&i| path[i]).unwrap();
    let forward: Vec<u32> = (0..k).map(|i| path[(min_pos + i) % k]).collect();
    let reverse: Vec<u32> = std::iter::once(forward[0])
        .chain((1..k).map(|i| forward[k - i]))
        .collect();
    if forward < reverse { forward } else { reverse }
}

/// DFS that only extends to vertices with previously-unused colors.
fn dfs_color_coded(
    row_ptr: &[u32],
    col_idx: &[u32],
    start: u32,
    k: usize,
    colors: &[u8],
    visited: &mut [bool],
    path: &mut Vec<u32>,
    used_colors: &mut u32,
    out: &mut Vec<Vec<u32>>,
) {
    path.push(start);
    visited[start as usize] = true;
    *used_colors |= 1u32 << colors[start as usize];

    fn recurse(
        row_ptr: &[u32],
        col_idx: &[u32],
        start: u32,
        k: usize,
        colors: &[u8],
        path: &mut Vec<u32>,
        visited: &mut [bool],
        used_colors: &mut u32,
        out: &mut Vec<Vec<u32>>,
    ) {
        if path.len() == k {
            let last = *path.last().unwrap();
            if has_edge(row_ptr, col_idx, last, start) {
                // Emit canonical form. (Skip in-DFS-direction tiebreak —
                // dedup pass handles it.)
                out.push(canonical_cycle(path));
            }
            return;
        }
        let tail = *path.last().unwrap();
        for &nxt in neighbours(row_ptr, col_idx, tail) {
            if nxt < start { continue; }
            if visited[nxt as usize] { continue; }
            let c = colors[nxt as usize];
            let bit = 1u32 << c;
            if *used_colors & bit != 0 { continue; }
            path.push(nxt);
            visited[nxt as usize] = true;
            *used_colors |= bit;
            recurse(row_ptr, col_idx, start, k, colors,
                     path, visited, used_colors, out);
            path.pop();
            visited[nxt as usize] = false;
            *used_colors &= !bit;
        }
    }

    recurse(row_ptr, col_idx, start, k, colors,
             path, visited, used_colors, out);

    path.pop();
    visited[start as usize] = false;
    *used_colors &= !(1u32 << colors[start as usize]);
}

#[inline]
fn lcg_next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn random_coloring(n_nodes: usize, k: usize, seed: u64) -> Vec<u8> {
    let mut state = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let mut out = vec![0u8; n_nodes];
    let k_u64 = k as u64;
    for v in out.iter_mut() {
        let r = lcg_next(&mut state) >> 33;
        *v = (r % k_u64) as u8;
    }
    out
}

/// Color-coding sampler for undirected k-cycles. Runs colorings in
/// parallel via rayon; deduplicates emitted cycles in a lock-free
/// DashSet (16-shard concurrent hash map). Each coloring also
/// internally parallelises over starting vertices for better load
/// balance on dense graphs where vertex 0's DFS dominates.
#[pyfunction]
#[pyo3(signature = (edges_u, edges_v, n_nodes, k,
                      target_cycles, seed=0,
                      max_colorings=None, n_threads=None))]
pub fn enumerate_k_cycles_color_coded_rs(
    py: Python<'_>,
    edges_u: Vec<u32>,
    edges_v: Vec<u32>,
    n_nodes: usize,
    k: usize,
    target_cycles: usize,
    seed: u64,
    max_colorings: Option<usize>,
    n_threads: Option<usize>,
) -> PyResult<Py<PyArray2<u32>>> {
    if k < 3 || k > 16 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "color-coded enumerator requires 3 <= k <= 16",
        ));
    }
    if edges_u.len() != edges_v.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "edges_u and edges_v must have the same length",
        ));
    }
    let edges: Vec<(u32, u32)> = edges_u
        .into_iter()
        .zip(edges_v.into_iter())
        .collect();
    let (row_ptr, col_idx) = build_csr(&edges, n_nodes, /*directed=*/false);

    // Default budget: enough colorings to cover most of the cycle space
    // with high probability. k! / k^k is the per-cycle rainbow rate;
    // 5 / (k!/k^k) colorings → ≥ 99% coverage of every cycle expected
    // appearance count >= 5.
    let kf: u64 = (1..=k as u64).product();
    let kk: u64 = (k as u64).pow(k as u32);
    let cov_factor: u64 = (kk + kf - 1) / kf.max(1); // ceil(k^k / k!)
    let max_cols = max_colorings.unwrap_or((cov_factor as usize) * 5).max(1);

    let pool = n_threads.and_then(|nt| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(nt.max(1))
            .build()
            .ok()
    });

    use dashmap::DashSet;
    let dedup: DashSet<Vec<u32>> = DashSet::with_capacity(target_cycles * 2);
    let total_kept = std::sync::atomic::AtomicUsize::new(0);

    py.detach(|| {
        let do_work = || {
            (0..max_cols).into_par_iter().for_each(|coloring_idx| {
                if total_kept.load(std::sync::atomic::Ordering::Relaxed)
                    >= target_cycles { return; }
                let coloring_seed = seed
                    .wrapping_add(coloring_idx as u64)
                    .wrapping_mul(0x9e37_79b9_7f4a_7c15);
                let colors = random_coloring(n_nodes, k, coloring_seed);
                let mut visited = vec![false; n_nodes];
                let mut path: Vec<u32> = Vec::with_capacity(k);
                let mut local_out: Vec<Vec<u32>> = Vec::new();
                let mut used: u32 = 0;
                for start in 0..n_nodes as u32 {
                    dfs_color_coded(
                        &row_ptr, &col_idx, start, k, &colors,
                        &mut visited, &mut path, &mut used, &mut local_out,
                    );
                    // Periodic atomic check to bail early.
                    if start.is_multiple_of(512) {
                        if total_kept.load(std::sync::atomic::Ordering::Relaxed)
                            >= target_cycles { break; }
                    }
                }
                for cyc in local_out {
                    if total_kept.load(std::sync::atomic::Ordering::Relaxed)
                        >= target_cycles { break; }
                    if dedup.insert(cyc) {
                        total_kept.fetch_add(
                            1, std::sync::atomic::Ordering::Relaxed);
                    }
                }
            });
        };

        if let Some(p) = pool {
            p.install(do_work);
        } else {
            do_work();
        }
    });

    // Drain dedup into a flat (N, k) buffer, capped at target_cycles.
    let mut flat: Vec<u32> = Vec::with_capacity(target_cycles * k);
    let mut count = 0usize;
    for cyc in dedup.into_iter() {
        if count >= target_cycles { break; }
        flat.extend_from_slice(&cyc);
        count += 1;
    }
    flat_to_pyarray2(py, flat, k, Some(target_cycles))
}

// ============================================================================
// Path-closure sampler.
//
// Idea: pick a random length-(k-1) walk that does not revisit vertices,
// then test edge closure. Each accepted walk is one k-cycle; we
// canonicalise and dedup.
//
// Per-step: from the current tail, sample a uniformly-random
// not-yet-visited neighbour. After k-1 steps (k vertices in the path),
// check whether the (last, start) edge exists. If yes, emit; else reject.
//
// Acceptance probability per attempt:
//   P(close) = #closing-edges / (combinatorial walk weight)
// In practice we don't compute the weight — we just collect samples
// until target_cycles unique cycles found.
//
// Comparison with color-coding:
//   - Path-closure scales naturally to ANY k (no k <= 32 limit)
//   - Path-closure is biased toward cycles whose vertices have low
//     degree (because high-degree vertices are sampled less per step).
//     Not unbiased without rejection re-weighting; treat as a
//     biased-sample baseline. Color-coding is unbiased per cycle.
//   - Path-closure is the fastest single-cycle generator on dense
//     low-arity graphs (k=3, k=4)
//
// The sampler runs colorings in parallel via rayon and uses the same
// DashMap-style shared dedup that color-coding uses.
// ============================================================================

#[inline]
fn lcg_next_in_range(state: &mut u64, n: u32) -> u32 {
    let r = lcg_next(state) >> 33;
    (r as u32) % n
}

/// Try one walk attempt. Returns Some(canonical_cycle) on success, None
/// on rejection (revisit-stuck or no closing edge).
fn try_one_path_closure(
    row_ptr: &[u32],
    col_idx: &[u32],
    n_nodes: usize,
    k: usize,
    rng: &mut u64,
    visited: &mut [bool],
    path: &mut Vec<u32>,
) -> Option<Vec<u32>> {
    debug_assert!(visited.iter().all(|&b| !b));
    debug_assert!(path.is_empty());
    // Pick start vertex uniformly.
    let start = lcg_next_in_range(rng, n_nodes as u32);
    path.push(start);
    visited[start as usize] = true;

    for _ in 1..k {
        let tail = *path.last().unwrap();
        let nbrs = neighbours(row_ptr, col_idx, tail);
        if nbrs.is_empty() {
            // Dead-end: cleanup and reject.
            for &v in path.iter() { visited[v as usize] = false; }
            path.clear();
            return None;
        }
        // Sample a random neighbour, retry up to a few times if the
        // sample is already visited. For most graphs this loop converges
        // in one or two tries; for stuck cases (low-degree dense
        // visited-set situation) bail.
        let mut chosen: Option<u32> = None;
        for _ in 0..8 {
            let idx = lcg_next_in_range(rng, nbrs.len() as u32);
            let cand = nbrs[idx as usize];
            if !visited[cand as usize] {
                chosen = Some(cand);
                break;
            }
        }
        let nxt = match chosen {
            Some(v) => v,
            None => {
                // Couldn't find an unvisited neighbour quickly. Give up.
                for &v in path.iter() { visited[v as usize] = false; }
                path.clear();
                return None;
            }
        };
        path.push(nxt);
        visited[nxt as usize] = true;
    }

    // Closure test.
    let last = *path.last().unwrap();
    let closes = has_edge(row_ptr, col_idx, last, start);
    let cyc = if closes { Some(canonical_cycle(path)) } else { None };

    for &v in path.iter() { visited[v as usize] = false; }
    path.clear();
    cyc
}

/// Path-closure k-cycle sampler. Runs random walks in parallel via
/// rayon until `target_cycles` unique canonical cycles are collected
/// or `max_attempts` total walks have been tried.
#[pyfunction]
#[pyo3(signature = (edges_u, edges_v, n_nodes, k,
                      target_cycles, seed=0,
                      max_attempts=None, n_threads=None))]
pub fn enumerate_k_cycles_path_closure_rs(
    py: Python<'_>,
    edges_u: Vec<u32>,
    edges_v: Vec<u32>,
    n_nodes: usize,
    k: usize,
    target_cycles: usize,
    seed: u64,
    max_attempts: Option<usize>,
    n_threads: Option<usize>,
) -> PyResult<Py<PyArray2<u32>>> {
    if k < 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "path-closure sampler requires k >= 3",
        ));
    }
    if edges_u.len() != edges_v.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "edges_u and edges_v must have the same length",
        ));
    }
    let edges: Vec<(u32, u32)> = edges_u
        .into_iter()
        .zip(edges_v.into_iter())
        .collect();
    let (row_ptr, col_idx) = build_csr(&edges, n_nodes, /*directed=*/false);

    // Default attempt budget: assume ~5% acceptance rate (realistic for
    // sparse graphs) and 50% dedup ratio, so allow ~50× target attempts.
    let max_att = max_attempts.unwrap_or(target_cycles * 50).max(target_cycles);

    let pool = n_threads.and_then(|nt| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(nt.max(1))
            .build()
            .ok()
    });

    use dashmap::DashSet;
    let dedup: DashSet<Vec<u32>> = DashSet::with_capacity(target_cycles * 2);
    let total_kept = std::sync::atomic::AtomicUsize::new(0);

    py.detach(|| {
        let do_work = || {
            // Chunk attempts so we have many parallel work units but
            // each thread amortises the per-walk setup over a batch.
            let chunk = 1024usize;
            let n_chunks = (max_att + chunk - 1) / chunk;
            (0..n_chunks).into_par_iter().for_each(|chunk_i| {
                if total_kept.load(std::sync::atomic::Ordering::Relaxed)
                    >= target_cycles { return; }
                let mut rng = seed
                    .wrapping_add(chunk_i as u64)
                    .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                    .wrapping_add(1);
                let mut visited = vec![false; n_nodes];
                let mut path: Vec<u32> = Vec::with_capacity(k);
                let lo = chunk_i * chunk;
                let hi = ((chunk_i + 1) * chunk).min(max_att);
                for _ in lo..hi {
                    if total_kept.load(std::sync::atomic::Ordering::Relaxed)
                        >= target_cycles { break; }
                    if let Some(cyc) = try_one_path_closure(
                        &row_ptr, &col_idx, n_nodes, k,
                        &mut rng, &mut visited, &mut path,
                    ) {
                        // dedup.insert returns true iff newly inserted.
                        if dedup.insert(cyc) {
                            total_kept.fetch_add(
                                1, std::sync::atomic::Ordering::Relaxed);
                        }
                    }
                }
            });
        };
        if let Some(p) = pool {
            p.install(do_work);
        } else {
            do_work();
        }
    });

    let mut flat: Vec<u32> = Vec::with_capacity(target_cycles * k);
    let mut count = 0usize;
    for cyc in dedup.into_iter() {
        if count >= target_cycles { break; }
        flat.extend_from_slice(&cyc);
        count += 1;
    }
    flat_to_pyarray2(py, flat, k, Some(target_cycles))
}

// ============================================================================
// Open length-`walk_len` walk enumeration. Item #5 — Walk-HSiKAN prototype.
//
// A length-`L` simple walk is a sequence (v_0, v_1, ..., v_L) with each
// consecutive pair an edge in G and no vertex revisits. Output shape:
// (N, L+1). Canonical form: emit only walks with path[0] <= path[L] to
// avoid emitting both (a, ..., b) and (b, ..., a) in undirected graphs.
//
// Reuses the same Sink machinery (Full / Reservoir / EarlyStop) as
// the cycle enumerator. Bitset visited; BFS distance pruning is not
// applicable to walks (they don't close), so dist[] is left empty.
// ============================================================================

fn dfs_walks_recurse(
    row_ptr: &[u32],
    col_idx: &[u32],
    walk_len: usize,
    path: &mut Vec<u32>,
    visited: &mut [u64],
    sink: &mut Sink,
) -> bool {
    if path.len() == walk_len + 1 {
        if path[0] <= path[walk_len] {
            return sink.offer(path);
        }
        return true;
    }
    let tail = *path.last().unwrap();
    for &nxt in neighbours(row_ptr, col_idx, tail) {
        if bs_get(visited, nxt) { continue; }
        path.push(nxt);
        bs_set(visited, nxt);
        let cont = dfs_walks_recurse(row_ptr, col_idx, walk_len,
                                       path, visited, sink);
        path.pop();
        bs_clear(visited, nxt);
        if !cont { return false; }
    }
    true
}

fn dfs_walks_from(
    row_ptr: &[u32],
    col_idx: &[u32],
    start: u32,
    walk_len: usize,
    visited: &mut [u64],
    path: &mut Vec<u32>,
    sink: &mut Sink,
) -> bool {
    path.push(start);
    bs_set(visited, start);
    let cont = dfs_walks_recurse(row_ptr, col_idx, walk_len,
                                   path, visited, sink);
    path.pop();
    bs_clear(visited, start);
    cont
}

/// Enumerate all simple length-`walk_len` walks (open paths, walk_len+1
/// vertices each, no vertex revisits) in an undirected graph.
///
/// Returns a numpy ndarray of shape ``(N, walk_len + 1)``. With
/// ``max_walks`` set, samples uniformly via reservoir.
///
/// Walk-HSiKAN prototype primitive — the closed-cycle enumeration
/// already lives in `enumerate_k_cycles_rs`. This is its open-path
/// sibling: same DFS skeleton, no closure check, canonical-form
/// dedup by smallest endpoint.
#[pyfunction]
#[pyo3(signature = (edges_u, edges_v, n_nodes, walk_len,
                      max_walks=None, seed=0))]
pub fn enumerate_k_walks_rs(
    py: Python<'_>,
    edges_u: Vec<u32>,
    edges_v: Vec<u32>,
    n_nodes: usize,
    walk_len: usize,
    max_walks: Option<usize>,
    seed: u64,
) -> PyResult<Py<PyArray2<u32>>> {
    if walk_len == 0 {
        let arr = Array2::<u32>::zeros((0, 1));
        return Ok(arr.into_pyarray(py).unbind());
    }
    if edges_u.len() != edges_v.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "edges_u and edges_v must have the same length",
        ));
    }
    let edges: Vec<(u32, u32)> = edges_u
        .into_iter()
        .zip(edges_v.into_iter())
        .collect();
    let (row_ptr, col_idx) = build_csr(&edges, n_nodes, false);

    let sink = py.detach(|| {
        let mut s = match max_walks {
            Some(cap) => Sink::new_reservoir(cap, seed),
            None => Sink::new_full(),
        };
        let mut visited: Vec<u64> = vec![0u64; bs_words(n_nodes)];
        let mut path: Vec<u32> = Vec::with_capacity(walk_len + 1);
        for start in 0..n_nodes as u32 {
            let cont = dfs_walks_from(&row_ptr, &col_idx, start,
                                        walk_len, &mut visited,
                                        &mut path, &mut s);
            if !cont { break; }
        }
        s
    });

    let buf = sink.into_flat();
    flat_to_pyarray2(py, buf, walk_len + 1, max_walks)
}

// ============================================================================
// Top-K signed cycle enumeration (axiom-aware approximation).
//
// Bridges hymeko_graph::enumerate_top_k_cycles + the vertex-stratified
// variant into Python. Returns (cycles_array, scores_array) so the caller
// can use scores as edge weights in M_e if desired.
// ============================================================================

use hymeko_graph::{
    EntropyGainScorer, HybridScorer, InverseDegreeScorer,
    balance::{BalanceMode, CartwrightHararyPruner, DavisWeakBalancePruner},
    community::{
        balance_ratio_per_community, label_propagation,
        CommunityAxiomPruner,
    },
    degree_adaptive_m_v,
    enumerate_top_k_cycles_par,
    enumerate_top_k_cycles_par_batched,
    enumerate_top_k_cycles_par_bb,
    enumerate_top_k_cycles_par_bb_batched,
    enumerate_top_k_cycles_par_entropy,
    enumerate_top_k_cycles_par_entropy_batched,
    enumerate_top_k_cycles_par_noprune as g_top_k_par,
    enumerate_top_k_per_vertex_cycles_par,
    enumerate_top_k_per_vertex_cycles_par_adaptive,
    enumerate_top_k_per_vertex_cycles_par_adaptive_batched,
    enumerate_top_k_per_vertex_cycles_par_adaptive_starting,
    enumerate_top_k_per_vertex_cycles_par_adaptive_starting_batched,
    enumerate_top_k_per_vertex_cycles_par_adaptive_starting_bb_batched,
    enumerate_top_k_per_vertex_cycles_par_adaptive_starting_bb_global_batched,
    enumerate_top_k_per_vertex_cycles_par_batched,
    enumerate_top_k_per_vertex_cycles_par_noprune as g_top_k_per_v_par,
    tiered_m_v_by_degree,
    vertex_filter::{
        AndFilter, DegreeFilter, NoFilter, TriangleFilter, VertexFilter,
    },
    topk_cycles::{
        BalanceScorer, FractionNegativeScorer, LowRootScorer,
        SignProductAbsScorer, scorers as g_scorers,
    },
    traversal::Csr, NoOpPruner, SignedGraph as GSignedGraph,
};
use numpy::PyArray1;

/// Build a hymeko_graph::SignedGraph from python-side buffers.
fn build_signed_graph(
    edges_u: &[u32],
    edges_v: &[u32],
    edges_s: &[i8],
    n_nodes: u32,
) -> PyResult<GSignedGraph> {
    if edges_u.len() != edges_v.len() || edges_u.len() != edges_s.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "edges_u, edges_v, edges_s must have equal length",
        ));
    }
    Ok(GSignedGraph::from_parts(n_nodes, edges_u, edges_v, edges_s))
}

/// Convert a [`TopKCyclesBatch`] to the `(cycles_2d, scores_1d)` numpy
/// tuple consumed by Python.  Zero-copy: the underlying `Vec` buffers
/// are moved into numpy ownership via `into_pyarray()` --- no flatten
/// loop, no intermediate `Vec<TopKCycle>`.
fn batch_to_pyarray(
    py: Python<'_>,
    batch: hymeko_graph::TopKCyclesBatch,
    k_len: usize,
) -> PyResult<(Py<PyArray2<u32>>, Py<PyArray1<f64>>)> {
    let n = batch.len();
    let arr = Array2::from_shape_vec((n, k_len), batch.cycles).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("reshape: {e}"))
    })?;
    Ok((
        arr.into_pyarray(py).unbind(),
        batch.scores.into_pyarray(py).unbind(),
    ))
}

/// Look up the right scorer by string tag.
fn pick_scorer(name: &str) -> Option<fn(&[u32], &[i8]) -> f64> {
    match name {
        "balance"           => Some(g_scorers::balance),
        "fraction_negative" => Some(g_scorers::fraction_negative),
        "sign_product_abs"  => Some(g_scorers::sign_product_abs),
        "low_root"          => Some(g_scorers::low_root),
        _ => None,
    }
}

/// Top-K signed cycle enumeration: keep the `k_keep` highest-scoring
/// cycles globally.
///
/// `score_kind` ∈ {"balance", "fraction_negative", "sign_product_abs",
/// "low_root"} — emit-time ranking heuristic.
///
/// `pruner_kind` ∈ {"none", "balance", "unbalanced", "davis"} —
/// extend-time axiom pruner that cuts DFS branches before they
/// materialise. Combined with BFS-distance pruning (always on),
/// `pruner_kind != "none"` is what makes top-K *cheaper than full*,
/// not just "full and then sort."
///
/// Returns `(cycles, scores)`.
#[pyfunction]
#[pyo3(signature = (edges_u, edges_v, edges_s, n_nodes, k_len, k_keep,
                      score_kind="balance", pruner_kind="none"))]
pub fn enumerate_top_k_cycles_signed_rs(
    py: Python<'_>,
    edges_u: Vec<u32>,
    edges_v: Vec<u32>,
    edges_s: Vec<i8>,
    n_nodes: u32,
    k_len: usize,
    k_keep: usize,
    score_kind: &str,
    pruner_kind: &str,
) -> PyResult<(Py<PyArray2<u32>>, Py<PyArray1<f64>>)> {
    let g = build_signed_graph(&edges_u, &edges_v, &edges_s, n_nodes)?;
    let scorer = pick_scorer(score_kind).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "unknown score_kind '{score_kind}'; valid: balance, \
             fraction_negative, sign_product_abs, low_root"
        ))
    })?;

    let batch = py.detach(|| match pruner_kind {
        "balance" => enumerate_top_k_cycles_par_batched(
            &g, k_len,
            &CartwrightHararyPruner { mode: BalanceMode::OnlyBalanced },
            k_keep, scorer,
        ),
        "unbalanced" => enumerate_top_k_cycles_par_batched(
            &g, k_len,
            &CartwrightHararyPruner { mode: BalanceMode::OnlyUnbalanced },
            k_keep, scorer,
        ),
        "davis" => enumerate_top_k_cycles_par_batched(
            &g, k_len,
            &DavisWeakBalancePruner,
            k_keep, scorer,
        ),
        _ => enumerate_top_k_cycles_par_batched(
            &g, k_len, &NoOpPruner, k_keep, scorer,
        ),
    });
    // (g_top_k_par used to be referenced here for the noprune branch;
    // batched path delegates to _par_batched with NoOpPruner.)

    batch_to_pyarray(py, batch, k_len)
}

/// Vertex-stratified top-m signed cycle enumeration: for each vertex
/// `v`, keep the `m_per_vertex` highest-scoring cycles passing
/// through `v`. Total `|cycles|` ≤ `n_nodes * m_per_vertex` and every
/// vertex on at least one cycle is covered.
///
/// `score_kind` ∈ {"balance", "fraction_negative", "sign_product_abs",
/// "low_root"} — emit-time ranking heuristic.
///
/// `pruner_kind` ∈ {"none", "balance", "unbalanced", "davis"} —
/// extend-time axiom pruner. BFS-distance pruning is always on
/// (~10× DFS savings on dense graphs at high k).
///
/// This is the variant that bounds `|M_e|` per row instead of
/// globally — recommended for HSiKAN training on Slashdot/Epinions.
///
/// Returns `(cycles, scores)`.
#[pyfunction]
#[pyo3(signature = (edges_u, edges_v, edges_s, n_nodes, k_len, m_per_vertex,
                      score_kind="fraction_negative", pruner_kind="none"))]
pub fn enumerate_top_k_per_vertex_cycles_signed_rs(
    py: Python<'_>,
    edges_u: Vec<u32>,
    edges_v: Vec<u32>,
    edges_s: Vec<i8>,
    n_nodes: u32,
    k_len: usize,
    m_per_vertex: usize,
    score_kind: &str,
    pruner_kind: &str,
) -> PyResult<(Py<PyArray2<u32>>, Py<PyArray1<f64>>)> {
    let g = build_signed_graph(&edges_u, &edges_v, &edges_s, n_nodes)?;
    let scorer = pick_scorer(score_kind).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "unknown score_kind '{score_kind}'; valid: balance, \
             fraction_negative, sign_product_abs, low_root"
        ))
    })?;

    let batch = py.detach(|| match pruner_kind {
        "balance" => enumerate_top_k_per_vertex_cycles_par_batched(
            &g, k_len,
            &CartwrightHararyPruner { mode: BalanceMode::OnlyBalanced },
            m_per_vertex, scorer,
        ),
        "unbalanced" => enumerate_top_k_per_vertex_cycles_par_batched(
            &g, k_len,
            &CartwrightHararyPruner { mode: BalanceMode::OnlyUnbalanced },
            m_per_vertex, scorer,
        ),
        "davis" => enumerate_top_k_per_vertex_cycles_par_batched(
            &g, k_len,
            &DavisWeakBalancePruner,
            m_per_vertex, scorer,
        ),
        "community" => {
            // Phase A: Louvain-equivalent (label-propagation)
            // community detection → per-community balance ratio →
            // per-community axiom (auto: balance/unbalanced/none).
            let csr = Csr::from_graph(&g);
            let (labels, n_comm) = label_propagation(&csr, 50, 0xC0FFEE);
            let bal = balance_ratio_per_community(
                &g, &csr, &labels, n_comm,
            );
            let comm_pruner = CommunityAxiomPruner::auto(
                labels, &bal, 0.85, 0.75,
            );
            enumerate_top_k_per_vertex_cycles_par_batched(
                &g, k_len, &comm_pruner, m_per_vertex, scorer,
            )
        }
        _ => enumerate_top_k_per_vertex_cycles_par_batched(
            &g, k_len, &NoOpPruner, m_per_vertex, scorer,
        ),
    });

    batch_to_pyarray(py, batch, k_len)
}

/// Build a [`VertexFilter`] from a `filter_kind` string and a
/// minimal parameter set.  Supported kinds match the v1 of the
/// vertex-prefilter plan (`docs/plans/2026-05-11-vertex-prefilter/`):
///   - `"none"`                    → all vertices
///   - `"degree"`                  → degree ≥ `min_degree`
///   - `"triangle"`                → vertex in at least one triangle
///   - `"compose:degree,triangle"` → AND of both
///
/// More expensive filters (clustering, local_sign_entropy,
/// clique_size, balanced_reachable) ship in v2 of the plan.
fn build_vertex_filter(
    filter_kind: &str,
    min_degree: u32,
) -> PyResult<Box<dyn VertexFilter>> {
    match filter_kind {
        "" | "none" | "all" => Ok(Box::new(NoFilter)),
        "degree" => Ok(Box::new(DegreeFilter { min_degree })),
        "triangle" => Ok(Box::new(TriangleFilter)),
        "compose:degree,triangle" | "compose:degree+triangle" => Ok(Box::new(AndFilter(vec![
            Box::new(DegreeFilter { min_degree }),
            Box::new(TriangleFilter),
        ]))),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown filter_kind '{other}'; valid: none | degree | triangle | \
             compose:degree,triangle"
        ))),
    }
}

/// Per-vertex top-$m$ signed cycle enumeration **with vertex
/// pre-filter, SoA output (zero-copy numpy)** (v1 filter + tonight's
/// SoA refactor).
///
/// Same shape as [`enumerate_top_k_per_vertex_cycles_signed_filtered_rs`]
/// (returns `(cycles, scores)` as numpy arrays) but goes through
/// the batched Rust enumerator which produces flat
/// `Vec<u32>` / `Vec<f64>` outputs.  The Vec → numpy conversion is
/// zero-copy via `into_pyarray()`, saving ~2N small heap allocations
/// (one `Vec<u32>` + one `Vec<i8>` per accepted cycle) that the
/// legacy `Vec<TopKCycle>` path produced.
///
/// On Epinions $k{=}4$ at production scale ($N \approx 3.7$M),
/// this saves $\sim 7.4$M small heap allocations and the
/// accompanying allocator pressure.  Empirically expected to save
/// 30-50% wall on the post-enumeration phase.
///
/// All other parameters identical to the legacy
/// `_filtered_rs` binding.
#[pyfunction]
#[pyo3(signature = (
    edges_u, edges_v, edges_s, n_nodes, k_len, m_per_vertex,
    score_kind, pruner_kind, filter_kind="none", filter_min_degree=2u32,
))]
#[allow(clippy::too_many_arguments)]
pub fn enumerate_top_k_per_vertex_cycles_signed_filtered_batched_rs(
    py: Python<'_>,
    edges_u: Vec<u32>,
    edges_v: Vec<u32>,
    edges_s: Vec<i8>,
    n_nodes: u32,
    k_len: usize,
    m_per_vertex: usize,
    score_kind: &str,
    pruner_kind: &str,
    filter_kind: &str,
    filter_min_degree: u32,
) -> PyResult<(Py<PyArray2<u32>>, Py<PyArray1<f64>>)> {
    let g = build_signed_graph(&edges_u, &edges_v, &edges_s, n_nodes)?;
    let scorer = pick_scorer(score_kind).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "unknown score_kind '{score_kind}'; valid: balance, \
             fraction_negative, sign_product_abs, low_root"
        ))
    })?;
    let filter = build_vertex_filter(filter_kind, filter_min_degree)?;
    let m_v: Vec<u32> = vec![m_per_vertex as u32; n_nodes as usize];

    let batch = py.detach(|| {
        let keep_set = filter.keep_set(&g);
        match pruner_kind {
            "balance" => enumerate_top_k_per_vertex_cycles_par_adaptive_starting_batched(
                &g, k_len,
                &CartwrightHararyPruner { mode: BalanceMode::OnlyBalanced },
                &m_v, &keep_set, scorer,
            ),
            "unbalanced" => enumerate_top_k_per_vertex_cycles_par_adaptive_starting_batched(
                &g, k_len,
                &CartwrightHararyPruner { mode: BalanceMode::OnlyUnbalanced },
                &m_v, &keep_set, scorer,
            ),
            "davis" => enumerate_top_k_per_vertex_cycles_par_adaptive_starting_batched(
                &g, k_len,
                &DavisWeakBalancePruner,
                &m_v, &keep_set, scorer,
            ),
            _ => enumerate_top_k_per_vertex_cycles_par_adaptive_starting_batched(
                &g, k_len,
                &NoOpPruner,
                &m_v, &keep_set, scorer,
            ),
        }
    });

    // Zero-copy: Vec → numpy ownership transfer.  No flatten loop.
    let n = batch.len();
    let arr = Array2::from_shape_vec((n, k_len), batch.cycles).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("reshape: {e}"))
    })?;
    Ok((
        arr.into_pyarray(py).unbind(),
        batch.scores.into_pyarray(py).unbind(),
    ))
}

/// Per-vertex top-$m$ signed cycle enumeration **with ABB**
/// (score upper-bound branch-and-bound at start-vertex threshold).
///
/// SoA output, zero-copy numpy.  Trade-off vs the non-BB filtered
/// variant: ABB prunes branches whose UB ≤ start vertex's heap
/// threshold; the cycle MIGHT have been useful for some other
/// cycle-vertex's heap.  Output is a SUBSET of the non-ABB
/// enumerator's output.  Best combined with v2 tiered caps where
/// heaps fill quickly.
///
/// Most effective when:
///   - per-vertex caps are small (heaps fill → threshold is meaningful)
///   - graph is dense (more cycles → more pruning opportunities)
///   - scorer has tight upper bound (less conservative pruning)
#[pyfunction]
#[pyo3(signature = (
    edges_u, edges_v, edges_s, n_nodes, k_len, m_per_vertex,
    score_kind, pruner_kind, filter_kind="none", filter_min_degree=2u32,
))]
#[allow(clippy::too_many_arguments)]
pub fn enumerate_top_k_per_vertex_cycles_signed_filtered_bb_batched_rs(
    py: Python<'_>,
    edges_u: Vec<u32>,
    edges_v: Vec<u32>,
    edges_s: Vec<i8>,
    n_nodes: u32,
    k_len: usize,
    m_per_vertex: usize,
    score_kind: &str,
    pruner_kind: &str,
    filter_kind: &str,
    filter_min_degree: u32,
) -> PyResult<(Py<PyArray2<u32>>, Py<PyArray1<f64>>)> {
    let g = build_signed_graph(&edges_u, &edges_v, &edges_s, n_nodes)?;
    let filter = build_vertex_filter(filter_kind, filter_min_degree)?;
    let m_v: Vec<u32> = vec![m_per_vertex as u32; n_nodes as usize];

    // ABB requires a BoundedScorer (not just a Fn).  Macro-dispatch
    // over the (pruner × scorer) matrix.
    macro_rules! run_bb_pv {
        ($pruner:expr, $scorer:expr) => {{
            let keep = filter.keep_set(&g);
            py.detach(|| {
                enumerate_top_k_per_vertex_cycles_par_adaptive_starting_bb_batched(
                    &g, k_len, &$pruner, &m_v, &keep, &$scorer,
                )
            })
        }};
    }
    macro_rules! dispatch_pv_pruner {
        ($scorer:expr) => {
            match pruner_kind {
                "none" => run_bb_pv!(NoOpPruner, $scorer),
                "balance" => run_bb_pv!(
                    CartwrightHararyPruner { mode: BalanceMode::OnlyBalanced },
                    $scorer
                ),
                "unbalanced" => run_bb_pv!(
                    CartwrightHararyPruner { mode: BalanceMode::OnlyUnbalanced },
                    $scorer
                ),
                "davis" => run_bb_pv!(DavisWeakBalancePruner, $scorer),
                _ => return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "unknown pruner_kind '{pruner_kind}'; valid: \
                     none, balance, unbalanced, davis"
                ))),
            }
        };
    }
    let batch = match score_kind {
        "fraction_negative" => dispatch_pv_pruner!(FractionNegativeScorer),
        "balance" => dispatch_pv_pruner!(BalanceScorer),
        "sign_product_abs" => dispatch_pv_pruner!(SignProductAbsScorer),
        "low_root" => dispatch_pv_pruner!(LowRootScorer),
        _ => return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown score_kind '{score_kind}'; valid: balance, \
             fraction_negative, sign_product_abs, low_root"
        ))),
    };

    batch_to_pyarray(py, batch, k_len)
}

/// Per-vertex top-$m$ signed cycle enumeration with **global-min
/// ABB** (Approach 1) and **adaptive fullness-gated activation**
/// (Approach 5) of the 2026-05-11 ABB-improvement track.
///
/// Differs from `enumerate_top_k_per_vertex_cycles_signed_filtered_bb_batched_rs`:
///   - threshold is the global MIN heap-min across all FULL heaps
///     (shared via `AtomicU64`), rather than the start-vertex's heap min
///   - ABB only fires once `fullness_gate` fraction of vertex heaps
///     are at capacity (default 0.25), avoiding wasted UB-checks
///     during the warm-up phase
///
/// AUC-preserving by construction: global-min ≤ any heap's min, so
/// pruning by UB ≤ global-min only discards cycles that no full heap
/// could accept anyway.  Speedup is necessarily smaller than start-only
/// ABB but does not throw cycles that the lossy v1 variant did.
#[pyfunction]
#[pyo3(signature = (
    edges_u, edges_v, edges_s, n_nodes, k_len, m_per_vertex,
    score_kind, pruner_kind, filter_kind="none", filter_min_degree=2u32,
    fullness_gate=0.25_f64,
))]
#[allow(clippy::too_many_arguments)]
pub fn enumerate_top_k_per_vertex_cycles_signed_filtered_bb_global_batched_rs(
    py: Python<'_>,
    edges_u: Vec<u32>,
    edges_v: Vec<u32>,
    edges_s: Vec<i8>,
    n_nodes: u32,
    k_len: usize,
    m_per_vertex: usize,
    score_kind: &str,
    pruner_kind: &str,
    filter_kind: &str,
    filter_min_degree: u32,
    fullness_gate: f64,
) -> PyResult<(Py<PyArray2<u32>>, Py<PyArray1<f64>>)> {
    let g = build_signed_graph(&edges_u, &edges_v, &edges_s, n_nodes)?;
    let filter = build_vertex_filter(filter_kind, filter_min_degree)?;
    let m_v: Vec<u32> = vec![m_per_vertex as u32; n_nodes as usize];

    macro_rules! run_bb_global_pv {
        ($pruner:expr, $scorer:expr) => {{
            let keep = filter.keep_set(&g);
            py.detach(|| {
                enumerate_top_k_per_vertex_cycles_par_adaptive_starting_bb_global_batched(
                    &g, k_len, &$pruner, &m_v, &keep, &$scorer, fullness_gate,
                )
            })
        }};
    }
    macro_rules! dispatch_pv_pruner_global {
        ($scorer:expr) => {
            match pruner_kind {
                "none" => run_bb_global_pv!(NoOpPruner, $scorer),
                "balance" => run_bb_global_pv!(
                    CartwrightHararyPruner { mode: BalanceMode::OnlyBalanced },
                    $scorer
                ),
                "unbalanced" => run_bb_global_pv!(
                    CartwrightHararyPruner { mode: BalanceMode::OnlyUnbalanced },
                    $scorer
                ),
                "davis" => run_bb_global_pv!(DavisWeakBalancePruner, $scorer),
                _ => return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "unknown pruner_kind '{pruner_kind}'; valid: \
                     none, balance, unbalanced, davis"
                ))),
            }
        };
    }
    let batch = match score_kind {
        "fraction_negative" => dispatch_pv_pruner_global!(FractionNegativeScorer),
        "balance" => dispatch_pv_pruner_global!(BalanceScorer),
        "sign_product_abs" => dispatch_pv_pruner_global!(SignProductAbsScorer),
        "low_root" => dispatch_pv_pruner_global!(LowRootScorer),
        _ => return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown score_kind '{score_kind}'; valid: balance, \
             fraction_negative, sign_product_abs, low_root"
        ))),
    };

    batch_to_pyarray(py, batch, k_len)
}

/// Per-vertex top-$m$ signed cycle enumeration **with vertex
/// pre-filter** (v1 of the vertex-prefilter plan).
///
/// Same surface as [`enumerate_top_k_per_vertex_cycles_signed_rs`]
/// but takes an additional `filter_kind` (and `filter_min_degree` for
/// the degree-based filters).  Vertices outside the filter's
/// keep-set are skipped as DFS roots — they contribute zero cycles
/// to the output.
///
/// `filter_kind ∈ {"none", "degree", "triangle",
///                 "compose:degree,triangle"}` — see
/// [`build_vertex_filter`] for the dispatch table.
///
/// On Epinions, ~30-40% of vertices are leaves or isolated
/// (degree<2) and contribute no cycles — filtering them out saves
/// 30-40% of per-vertex BFS pre-pass and DFS startup cost without
/// changing the output cycle set.
///
/// Returns `(cycles, scores)` with the same shape as the
/// unfiltered binding.
#[pyfunction]
#[pyo3(signature = (
    edges_u, edges_v, edges_s, n_nodes, k_len, m_per_vertex,
    score_kind, pruner_kind, filter_kind="none", filter_min_degree=2u32,
))]
#[allow(clippy::too_many_arguments)]
pub fn enumerate_top_k_per_vertex_cycles_signed_filtered_rs(
    py: Python<'_>,
    edges_u: Vec<u32>,
    edges_v: Vec<u32>,
    edges_s: Vec<i8>,
    n_nodes: u32,
    k_len: usize,
    m_per_vertex: usize,
    score_kind: &str,
    pruner_kind: &str,
    filter_kind: &str,
    filter_min_degree: u32,
) -> PyResult<(Py<PyArray2<u32>>, Py<PyArray1<f64>>)> {
    let g = build_signed_graph(&edges_u, &edges_v, &edges_s, n_nodes)?;
    let scorer = pick_scorer(score_kind).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "unknown score_kind '{score_kind}'; valid: balance, \
             fraction_negative, sign_product_abs, low_root"
        ))
    })?;

    let filter = build_vertex_filter(filter_kind, filter_min_degree)?;
    let m_v: Vec<u32> = vec![m_per_vertex as u32; n_nodes as usize];

    let batch = py.detach(|| {
        let keep_set = filter.keep_set(&g);
        match pruner_kind {
            "balance" => enumerate_top_k_per_vertex_cycles_par_adaptive_starting_batched(
                &g, k_len,
                &CartwrightHararyPruner { mode: BalanceMode::OnlyBalanced },
                &m_v, &keep_set, scorer,
            ),
            "unbalanced" => enumerate_top_k_per_vertex_cycles_par_adaptive_starting_batched(
                &g, k_len,
                &CartwrightHararyPruner { mode: BalanceMode::OnlyUnbalanced },
                &m_v, &keep_set, scorer,
            ),
            "davis" => enumerate_top_k_per_vertex_cycles_par_adaptive_starting_batched(
                &g, k_len,
                &DavisWeakBalancePruner,
                &m_v, &keep_set, scorer,
            ),
            _ => enumerate_top_k_per_vertex_cycles_par_adaptive_starting_batched(
                &g, k_len,
                &NoOpPruner,
                &m_v, &keep_set, scorer,
            ),
        }
    });

    batch_to_pyarray(py, batch, k_len)
}

/// Global top-$K$ signed cycle enumeration **with score-bounded
/// branch-and-bound** (ABB).
///
/// Same surface as [`enumerate_top_k_cycles_signed_rs`] but routes
/// through `enumerate_top_k_cycles_par_bb` with the `BoundedScorer`
/// matching `score_kind`.  On Epinions $k{=}4$, $K{=}10\,000$,
/// `score_kind="fraction_negative"`, `pruner_kind="balance"` this
/// drops wall time from ~100 s to ~4 s (25x speedup; see
/// `reports/2026-05-10-abb-global-topk.md`).
///
/// The output cycle SET may differ from the non-ABB path on score
/// ties (heap's strict `>` replacement rule), but the sorted score
/// vector is identical.  See
/// `tests/abb_global_topk.rs::parity_par_vs_par_bb_*`.
#[pyfunction]
#[pyo3(signature = (edges_u, edges_v, edges_s, n_nodes, k_len, k_keep,
                      score_kind="fraction_negative", pruner_kind="balance"))]
pub fn enumerate_top_k_cycles_signed_bb_rs(
    py: Python<'_>,
    edges_u: Vec<u32>,
    edges_v: Vec<u32>,
    edges_s: Vec<i8>,
    n_nodes: u32,
    k_len: usize,
    k_keep: usize,
    score_kind: &str,
    pruner_kind: &str,
) -> PyResult<(Py<PyArray2<u32>>, Py<PyArray1<f64>>)> {
    let g = build_signed_graph(&edges_u, &edges_v, &edges_s, n_nodes)?;

    // The four BoundedScorer impls live in `hymeko_graph::topk_cycles`;
    // each has its own type, so the dispatch is over types here
    // (not function pointers).  Macro-expand the pruner x scorer
    // matrix so every combination calls the monomorphised inner
    // function with no dynamic dispatch overhead.
    macro_rules! run_bb_batched {
        ($pruner:expr, $scorer:expr) => {
            py.detach(|| enumerate_top_k_cycles_par_bb_batched(
                &g, k_len, &$pruner, k_keep, &$scorer,
            ))
        };
    }
    macro_rules! dispatch_pruner_bb {
        ($scorer:expr) => {
            match pruner_kind {
                "none" => run_bb_batched!(NoOpPruner, $scorer),
                "balance" => run_bb_batched!(
                    CartwrightHararyPruner { mode: BalanceMode::OnlyBalanced },
                    $scorer
                ),
                "unbalanced" => run_bb_batched!(
                    CartwrightHararyPruner { mode: BalanceMode::OnlyUnbalanced },
                    $scorer
                ),
                "davis" => run_bb_batched!(DavisWeakBalancePruner, $scorer),
                _ => return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "unknown pruner_kind '{pruner_kind}'; valid: \
                     none, balance, unbalanced, davis"
                ))),
            }
        };
    }
    let batch = match score_kind {
        "fraction_negative" => dispatch_pruner_bb!(FractionNegativeScorer),
        "balance" => dispatch_pruner_bb!(BalanceScorer),
        "sign_product_abs" => dispatch_pruner_bb!(SignProductAbsScorer),
        "low_root" => dispatch_pruner_bb!(LowRootScorer),
        _ => return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown score_kind '{score_kind}'; valid: balance, \
             fraction_negative, sign_product_abs, low_root"
        ))),
    };

    batch_to_pyarray(py, batch, k_len)
}

/// Global top-$K$ signed cycle enumeration with the
/// **entropy-heuristic** scorer family — selects cycles to
/// maximise the per-vertex incidence-distribution entropy of the
/// kept set.
///
/// `heuristic_kind` ∈ {`"entropy"` (default), `"inverse_degree"`}.
/// `pruner_kind` ∈ {`"none"` (default), `"balance"`, `"unbalanced"`,
/// `"davis"`}.
///
/// On Epinions $k{=}4$ this delivers a vertex-uniform cycle set
/// (per-vertex coverage variance lower than global ABB at the same
/// $K$) at competitive wall time, matching what HSiKAN's $M_e$
/// expects.
///
/// Returns `(cycles, scores)`.
#[pyfunction]
#[pyo3(signature = (edges_u, edges_v, edges_s, n_nodes, k_len, k_keep,
                      heuristic_kind="entropy", pruner_kind="none"))]
pub fn enumerate_top_k_cycles_signed_entropy_rs(
    py: Python<'_>,
    edges_u: Vec<u32>,
    edges_v: Vec<u32>,
    edges_s: Vec<i8>,
    n_nodes: u32,
    k_len: usize,
    k_keep: usize,
    heuristic_kind: &str,
    pruner_kind: &str,
) -> PyResult<(Py<PyArray2<u32>>, Py<PyArray1<f64>>)> {
    let g = build_signed_graph(&edges_u, &edges_v, &edges_s, n_nodes)?;

    macro_rules! run_entropy_batched {
        ($pruner:expr, $heuristic:expr) => {
            py.detach(|| enumerate_top_k_cycles_par_entropy_batched(
                &g, k_len, &$pruner, k_keep, &$heuristic,
            ))
        };
    }
    macro_rules! dispatch_pruner_ent {
        ($heuristic:expr) => {
            match pruner_kind {
                "none" => run_entropy_batched!(NoOpPruner, $heuristic),
                "balance" => run_entropy_batched!(
                    CartwrightHararyPruner { mode: BalanceMode::OnlyBalanced },
                    $heuristic
                ),
                "unbalanced" => run_entropy_batched!(
                    CartwrightHararyPruner { mode: BalanceMode::OnlyUnbalanced },
                    $heuristic
                ),
                "davis" => run_entropy_batched!(DavisWeakBalancePruner, $heuristic),
                _ => return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "unknown pruner_kind '{pruner_kind}'; valid: \
                     none, balance, unbalanced, davis"
                ))),
            }
        };
    }
    let batch = match heuristic_kind {
        "entropy" => dispatch_pruner_ent!(EntropyGainScorer),
        "inverse_degree" => dispatch_pruner_ent!(InverseDegreeScorer),
        _ => return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown heuristic_kind '{heuristic_kind}'; valid: \
             entropy, inverse_degree"
        ))),
    };

    batch_to_pyarray(py, batch, k_len)
}

/// Global top-$K$ signed cycle enumeration with the **hybrid
/// alpha-blended scorer**: `score = alpha * signal + (1-alpha) *
/// diversity`.
///
/// `signal_kind` ∈ {`"fraction_negative"` (default), `"balance"`,
/// `"sign_product_abs"`, `"low_root"`} — the BoundedScorer
/// component.
///
/// `heuristic_kind` ∈ {`"entropy"` (default), `"inverse_degree"`} —
/// the diversity component.
///
/// `alpha` ∈ [0, 1].  alpha = 0 collapses to pure diversity (same
/// as `enumerate_top_k_cycles_signed_entropy_rs`); alpha = 1
/// collapses to pure signal score under the entropy enumerator's
/// greedy-with-rollback semantics.
///
/// `pruner_kind` ∈ {`"none"` (default), `"balance"`, `"unbalanced"`,
/// `"davis"`}.
///
/// Returns `(cycles, scores)`.
#[pyfunction]
#[pyo3(signature = (edges_u, edges_v, edges_s, n_nodes, k_len, k_keep,
                      signal_kind="fraction_negative",
                      heuristic_kind="entropy", alpha=0.5,
                      pruner_kind="none"))]
#[allow(clippy::too_many_arguments)]
pub fn enumerate_top_k_cycles_signed_hybrid_rs(
    py: Python<'_>,
    edges_u: Vec<u32>,
    edges_v: Vec<u32>,
    edges_s: Vec<i8>,
    n_nodes: u32,
    k_len: usize,
    k_keep: usize,
    signal_kind: &str,
    heuristic_kind: &str,
    alpha: f64,
    pruner_kind: &str,
) -> PyResult<(Py<PyArray2<u32>>, Py<PyArray1<f64>>)> {
    if !(0.0..=1.0).contains(&alpha) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "alpha must lie in [0, 1]; got {alpha}"
        )));
    }
    let g = build_signed_graph(&edges_u, &edges_v, &edges_s, n_nodes)?;

    macro_rules! run_hybrid {
        ($pruner:expr, $signal:expr, $div:expr) => {{
            let hybrid = HybridScorer::new($signal, $div, alpha);
            py.detach(|| enumerate_top_k_cycles_par_entropy_batched(
                &g, k_len, &$pruner, k_keep, &hybrid,
            ))
        }};
    }
    macro_rules! dispatch_div {
        ($pruner:expr, $signal:expr) => {
            match heuristic_kind {
                "entropy" => run_hybrid!($pruner, $signal, EntropyGainScorer),
                "inverse_degree" => run_hybrid!($pruner, $signal, InverseDegreeScorer),
                _ => return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "unknown heuristic_kind '{heuristic_kind}'; valid: \
                     entropy, inverse_degree"
                ))),
            }
        };
    }
    macro_rules! dispatch_signal {
        ($pruner:expr) => {
            match signal_kind {
                "fraction_negative" => dispatch_div!($pruner, FractionNegativeScorer),
                "balance" => dispatch_div!($pruner, BalanceScorer),
                "sign_product_abs" => dispatch_div!($pruner, SignProductAbsScorer),
                "low_root" => dispatch_div!($pruner, LowRootScorer),
                _ => return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "unknown signal_kind '{signal_kind}'; valid: balance, \
                     fraction_negative, sign_product_abs, low_root"
                ))),
            }
        };
    }
    let batch = match pruner_kind {
        "none" => dispatch_signal!(NoOpPruner),
        "balance" => dispatch_signal!(CartwrightHararyPruner {
            mode: BalanceMode::OnlyBalanced
        }),
        "unbalanced" => dispatch_signal!(CartwrightHararyPruner {
            mode: BalanceMode::OnlyUnbalanced
        }),
        "davis" => dispatch_signal!(DavisWeakBalancePruner),
        _ => return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown pruner_kind '{pruner_kind}'; valid: none, balance, unbalanced, davis"
        ))),
    };

    batch_to_pyarray(py, batch, k_len)
}

/// Per-vertex top-$m_v$ signed cycle enumeration with a
/// **degree-adaptive** $m_v$ schedule:
/// $m_v[v] = \min(m_\max, \max(m_\min, \lceil c \cdot \deg(v) \rceil))$.
///
/// This is the production-grade variant of
/// `enumerate_top_k_per_vertex_cycles_signed_rs` for graphs with
/// long-tail degree distributions (Epinions, Slashdot): low-degree
/// vertices get small caps they can actually fill, lifting the
/// per-vertex full-heap rate from $\sim 18\%$ to $\ge 50\%$ at
/// $c \ge 2$ on Epinions.
///
/// `c = 0.0` recovers a uniform cap $m_v = m_\min$ for every vertex
/// (use this for behavioral parity with the fixed-$m$ path at
/// $m_\min = m_\max = m_\text{per\_vertex}$).
///
/// `score_kind` ∈ {balance, fraction_negative, sign_product_abs,
/// low_root}. `pruner_kind` ∈ {none, balance, unbalanced, davis,
/// community}.
///
/// Returns `(cycles, scores)`.
///
/// ─── Tiered variant (FPN-style, v2 of vertex-prefilter plan) ───
///
/// See [`enumerate_top_k_per_vertex_cycles_signed_tiered_rs`] below
/// for the concentric step-function alternative.
#[pyfunction]
#[pyo3(signature = (edges_u, edges_v, edges_s, n_nodes, k_len, tiers,
                      score_kind="fraction_negative", pruner_kind="none"))]
#[allow(clippy::too_many_arguments)]
pub fn enumerate_top_k_per_vertex_cycles_signed_tiered_rs(
    py: Python<'_>,
    edges_u: Vec<u32>,
    edges_v: Vec<u32>,
    edges_s: Vec<i8>,
    n_nodes: u32,
    k_len: usize,
    tiers: Vec<(f32, u32)>,
    score_kind: &str,
    pruner_kind: &str,
) -> PyResult<(Py<PyArray2<u32>>, Py<PyArray1<f64>>)> {
    if tiers.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "tiers must be non-empty",
        ));
    }
    for w in tiers.windows(2) {
        if w[0].0 > w[1].0 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "tiers must be sorted ascending by percentile; got {} > {}",
                w[0].0, w[1].0
            )));
        }
    }
    let g = build_signed_graph(&edges_u, &edges_v, &edges_s, n_nodes)?;
    let scorer = pick_scorer(score_kind).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "unknown score_kind '{score_kind}'; valid: balance, \
             fraction_negative, sign_product_abs, low_root"
        ))
    })?;
    let m_v = tiered_m_v_by_degree(&g, &tiers);
    let starting: Vec<u32> = (0..n_nodes).collect();

    let batch = py.detach(|| match pruner_kind {
        "balance" => enumerate_top_k_per_vertex_cycles_par_adaptive_starting_batched(
            &g, k_len,
            &CartwrightHararyPruner { mode: BalanceMode::OnlyBalanced },
            &m_v, &starting, scorer,
        ),
        "unbalanced" => enumerate_top_k_per_vertex_cycles_par_adaptive_starting_batched(
            &g, k_len,
            &CartwrightHararyPruner { mode: BalanceMode::OnlyUnbalanced },
            &m_v, &starting, scorer,
        ),
        "davis" => enumerate_top_k_per_vertex_cycles_par_adaptive_starting_batched(
            &g, k_len, &DavisWeakBalancePruner, &m_v, &starting, scorer,
        ),
        _ => enumerate_top_k_per_vertex_cycles_par_adaptive_starting_batched(
            &g, k_len, &NoOpPruner, &m_v, &starting, scorer,
        ),
    });

    batch_to_pyarray(py, batch, k_len)
}

#[pyfunction]
#[pyo3(signature = (edges_u, edges_v, edges_s, n_nodes, k_len,
                      m_min, m_max, c,
                      score_kind="fraction_negative", pruner_kind="none"))]
#[allow(clippy::too_many_arguments)]
pub fn enumerate_top_k_per_vertex_cycles_signed_adaptive_rs(
    py: Python<'_>,
    edges_u: Vec<u32>,
    edges_v: Vec<u32>,
    edges_s: Vec<i8>,
    n_nodes: u32,
    k_len: usize,
    m_min: u32,
    m_max: u32,
    c: f64,
    score_kind: &str,
    pruner_kind: &str,
) -> PyResult<(Py<PyArray2<u32>>, Py<PyArray1<f64>>)> {
    if m_min > m_max {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "m_min ({m_min}) > m_max ({m_max})"
        )));
    }
    if c < 0.0 || !c.is_finite() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "c must be a non-negative finite number; got {c}"
        )));
    }
    let g = build_signed_graph(&edges_u, &edges_v, &edges_s, n_nodes)?;
    let scorer = pick_scorer(score_kind).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "unknown score_kind '{score_kind}'; valid: balance, \
             fraction_negative, sign_product_abs, low_root"
        ))
    })?;
    let m_v = degree_adaptive_m_v(&g, m_min, m_max, c);

    let batch = py.detach(|| match pruner_kind {
        "balance" => enumerate_top_k_per_vertex_cycles_par_adaptive_batched(
            &g, k_len,
            &CartwrightHararyPruner { mode: BalanceMode::OnlyBalanced },
            &m_v, scorer,
        ),
        "unbalanced" => enumerate_top_k_per_vertex_cycles_par_adaptive_batched(
            &g, k_len,
            &CartwrightHararyPruner { mode: BalanceMode::OnlyUnbalanced },
            &m_v, scorer,
        ),
        "davis" => enumerate_top_k_per_vertex_cycles_par_adaptive_batched(
            &g, k_len, &DavisWeakBalancePruner, &m_v, scorer,
        ),
        "community" => {
            let csr = Csr::from_graph(&g);
            let (labels, n_comm) = label_propagation(&csr, 50, 0xC0FFEE);
            let bal = balance_ratio_per_community(&g, &csr, &labels, n_comm);
            let comm_pruner = CommunityAxiomPruner::auto(labels, &bal, 0.85, 0.75);
            enumerate_top_k_per_vertex_cycles_par_adaptive_batched(
                &g, k_len, &comm_pruner, &m_v, scorer,
            )
        }
        _ => enumerate_top_k_per_vertex_cycles_par_adaptive_batched(
            &g, k_len, &NoOpPruner, &m_v, scorer,
        ),
    });

    batch_to_pyarray(py, batch, k_len)
}
