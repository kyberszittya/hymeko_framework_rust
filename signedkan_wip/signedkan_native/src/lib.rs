//! Native-speed signed n-tuple enumeration.
//!
//! Mirrors the algorithm in `signedkan_wip/src/n_tuples_fast.py`
//! (sparse-matrix-power 4-cycle enumeration with canonical dedup),
//! reimplemented in Rust with rayon parallelism across roots.
//!
//! Exposed as `signedkan_native` via PyO3.
//!
//! Algorithm:
//!   - For each vertex u (parallel root):
//!     - Collect neighbours N(u) > u (canonical: u smallest in cycle)
//!     - For each m ∈ N(u) > u, scan N(m) to count "candidate w"
//!       (vertices reachable in 2 hops from u via m). Track per-w
//!       common-neighbour set.
//!     - For each w with |C(u, w)| ≥ 2: enumerate pairs (m1 < m2)
//!       and emit cycle (u, m1, w, m2).
//!   - σ assignment + Davis weak balance flag computed from cycle
//!     edge signs.
//!
//! Returns four numpy arrays: cycle_v (T, 4), edge_signs (T, 4),
//! sigma (T, 4), balanced (T,).

use ahash::AHashMap;
use ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

/// Build symmetric per-vertex sorted neighbour arrays AND signed
/// adjacency for fast lookups.
fn build_adjacency(
    edges: &[[i64; 2]],
    signs: &[i8],
    n_nodes: usize,
) -> (Vec<Vec<i64>>, Vec<AHashMap<i64, i8>>) {
    let mut nbrs: Vec<Vec<i64>> = vec![Vec::new(); n_nodes];
    let mut sign_adj: Vec<AHashMap<i64, i8>> =
        (0..n_nodes).map(|_| AHashMap::new()).collect();
    for (e, &s) in edges.iter().zip(signs.iter()) {
        let u = e[0] as usize;
        let v = e[1] as usize;
        nbrs[u].push(e[1]);
        nbrs[v].push(e[0]);
        sign_adj[u].insert(e[1], s);
        sign_adj[v].insert(e[0], s);
    }
    for nbr in nbrs.iter_mut() {
        nbr.sort_unstable();
        nbr.dedup();
    }
    (nbrs, sign_adj)
}

/// Per-root enumeration: from vertex u, find all 4-cycles
/// (u, m1, w, m2) with u < m1, m2, w and m1 < m2.
fn enumerate_from_root(
    u: i64,
    nbrs: &[Vec<i64>],
    sign_adj: &[AHashMap<i64, i8>],
) -> Vec<([i64; 4], [i8; 4])> {
    let nbr_u = &nbrs[u as usize];
    if nbr_u.len() < 2 {
        return Vec::new();
    }
    // Neighbours of u strictly above u (so u becomes the smallest in
    // the cycle).
    let nbr_u_above: Vec<i64> = nbr_u.iter().copied().filter(|&v| v > u).collect();
    if nbr_u_above.len() < 2 {
        return Vec::new();
    }

    // Build w → set of common-with-u neighbours, considering only
    // m's > u so the cycle is canonically rooted at u.
    let mut w_to_common: AHashMap<i64, Vec<i64>> = AHashMap::new();
    for &m in &nbr_u_above {
        let nbr_m = &nbrs[m as usize];
        for &w in nbr_m {
            if w > u && w != m {
                w_to_common.entry(w).or_default().push(m);
            }
        }
    }

    let mut out: Vec<([i64; 4], [i8; 4])> = Vec::new();
    let adj_u = &sign_adj[u as usize];
    for (&w, common) in &w_to_common {
        if common.len() < 2 {
            continue;
        }
        // We want m1 < m2 with m1, m2 != w (already filtered).
        // common is in sorted order if we pushed in order of nbr_u_above
        // iteration (which is sorted), but multiple m values can repeat
        // — actually each m only appears once because each m is
        // processed once. So `common` is sorted ascending.
        let adj_w = &sign_adj[w as usize];
        let k = common.len();
        for i in 0..k {
            let m1 = common[i];
            for j in (i + 1)..k {
                let m2 = common[j];
                let s1 = match adj_u.get(&m1) {
                    Some(&v) => v, None => continue,
                };
                let s2 = match sign_adj[m1 as usize].get(&w) {
                    Some(&v) => v, None => continue,
                };
                let s3 = match adj_w.get(&m2) {
                    Some(&v) => v, None => continue,
                };
                let s4 = match sign_adj[m2 as usize].get(&u) {
                    Some(&v) => v, None => continue,
                };
                out.push(([u, m1, w, m2], [s1, s2, s3, s4]));
            }
        }
    }
    out
}

/// PyO3 entry: enumerate all 4-cycles in a signed graph.
///
/// Inputs:
///   edges  : (E, 2) int64 numpy array — undirected edge list
///   signs  : (E,)   int8  numpy array — ±1 per edge
///   n_nodes: usize
///
/// Returns a tuple of four numpy arrays:
///   cycle_v   : (T, 4) int64 — vertex IDs in cycle order, canonical
///   edge_signs: (T, 4) int8  — signs in cycle order
///   sigma     : (T, 4) int8  — per-vertex Davis σ ∈ {±1}
///   balanced  : (T,)   bool  — Davis weak-balance flag
#[pyfunction]
fn enumerate_4_cycles<'py>(
    py: Python<'py>,
    edges: PyReadonlyArray2<'py, i64>,
    signs: PyReadonlyArray1<'py, i8>,
    n_nodes: usize,
) -> PyResult<(
    Bound<'py, PyArray2<i64>>,
    Bound<'py, PyArray2<i8>>,
    Bound<'py, PyArray2<i8>>,
    Bound<'py, PyArray1<bool>>,
)> {
    // Drop the GIL while we do CPU work.
    let edges_view = edges.as_array();
    let signs_view = signs.as_array();
    let edges_vec: Vec<[i64; 2]> = edges_view
        .rows()
        .into_iter()
        .map(|r| [r[0], r[1]])
        .collect();
    let signs_vec: Vec<i8> = signs_view.to_vec();

    let (cycles_v, edge_signs_v) = py.allow_threads(|| {
        let (nbrs, sign_adj) = build_adjacency(&edges_vec, &signs_vec, n_nodes);
        // Parallelise across root vertices.
        let chunks: Vec<Vec<([i64; 4], [i8; 4])>> = (0..n_nodes as i64)
            .into_par_iter()
            .map(|u| enumerate_from_root(u, &nbrs, &sign_adj))
            .collect();
        let total: usize = chunks.iter().map(|c| c.len()).sum();
        let mut all_cycles: Vec<[i64; 4]> = Vec::with_capacity(total);
        let mut all_signs: Vec<[i8; 4]> = Vec::with_capacity(total);
        for chunk in chunks {
            for (c, s) in chunk {
                all_cycles.push(c);
                all_signs.push(s);
            }
        }
        (all_cycles, all_signs)
    });

    let t = cycles_v.len();
    let mut cycle_arr = Array2::<i64>::zeros((t, 4));
    let mut signs_arr = Array2::<i8>::zeros((t, 4));
    let mut sigma_arr = Array2::<i8>::zeros((t, 4));
    let mut bal_arr = Array1::<bool>::from_elem(t, false);

    for (i, (c, s)) in cycles_v.iter().zip(edge_signs_v.iter()).enumerate() {
        for j in 0..4 {
            cycle_arr[(i, j)] = c[j];
            signs_arr[(i, j)] = s[j];
        }
        // Davis: vertex j is incident to cycle edges (j-1) mod 4 and j.
        let mut total_neg = 0u8;
        for j in 0..4 {
            let is_neg_prev = (s[(j + 3) % 4] == -1) as u8;
            let is_neg_curr = (s[j] == -1) as u8;
            let n = is_neg_prev + is_neg_curr;
            sigma_arr[(i, j)] = if n % 2 == 0 { 1 } else { -1 };
            if s[j] == -1 {
                total_neg += 1;
            }
        }
        bal_arr[i] = total_neg % 2 == 0;
    }

    Ok((
        cycle_arr.into_pyarray_bound(py),
        signs_arr.into_pyarray_bound(py),
        sigma_arr.into_pyarray_bound(py),
        bal_arr.into_pyarray_bound(py),
    ))
}

#[pymodule]
fn signedkan_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(enumerate_4_cycles, m)?)?;
    Ok(())
}
