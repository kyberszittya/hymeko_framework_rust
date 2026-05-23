//! PyO3 entries for unsigned cycle / walk enumeration.
//!
//! Canonical entry: [`enumerate_unsigned_rs`] (`backend` string + unified optional
//! parameters). The four legacy `enumerate_k_*_rs` symbols remain thin wrappers
//! for stable Python call sites (CLAUDE.md §6.5 #1 — one dispatch, many names).

use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2};
use pyo3::prelude::*;
use std::sync::atomic::AtomicUsize;

use hymeko_graph::unsigned_cycles::{
    bfs_distances_into, bs_words, build_csr, dfs_from, enumerate_parallel, Sink,
};

use super::io::flat_to_pyarray2;

/// Internal tag for the unsigned enumeration family.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum UnsignedBackend {
    Dfs,
    ColorCoded,
    PathClosure,
    Walks,
}

impl UnsignedBackend {
    fn parse(s: &str) -> PyResult<Self> {
        match s {
            "dfs" | "parallel_dfs" => Ok(Self::Dfs),
            "color_coded" | "color-coded" => Ok(Self::ColorCoded),
            "path_closure" | "path-closure" => Ok(Self::PathClosure),
            "walks" => Ok(Self::Walks),
            other => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unknown unsigned backend '{other}'; valid: dfs, color_coded, path_closure, walks"
            ))),
        }
    }
}

/// Unified optional knobs for all unsigned backends (unused fields ignored per backend).
#[derive(Clone, Debug, Default)]
pub(crate) struct UnsignedOptions {
    pub max_cycles: Option<usize>,
    pub seed: u64,
    pub directed: bool,
    pub early_stop: bool,
    pub n_threads: Option<usize>,
    pub target_cycles: Option<usize>,
    pub max_colorings: Option<usize>,
    pub max_attempts: Option<usize>,
    pub max_walks: Option<usize>,
}

/// Single implementation path for CSR build + backend dispatch.
pub(crate) fn dispatch_unsigned(
    py: Python<'_>,
    backend: UnsignedBackend,
    edges_u: Vec<u32>,
    edges_v: Vec<u32>,
    n_nodes: usize,
    k: usize,
    opt: UnsignedOptions,
) -> PyResult<Py<PyArray2<u32>>> {
    if edges_u.len() != edges_v.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "edges_u and edges_v must have the same length",
        ));
    }

    match backend {
        UnsignedBackend::Dfs => run_dfs(py, edges_u, edges_v, n_nodes, k, opt),
        UnsignedBackend::ColorCoded => run_color_coded(py, edges_u, edges_v, n_nodes, k, opt),
        UnsignedBackend::PathClosure => run_path_closure(py, edges_u, edges_v, n_nodes, k, opt),
        UnsignedBackend::Walks => run_walks(py, edges_u, edges_v, n_nodes, k, opt),
    }
}

fn run_dfs(
    py: Python<'_>,
    edges_u: Vec<u32>,
    edges_v: Vec<u32>,
    n_nodes: usize,
    k: usize,
    opt: UnsignedOptions,
) -> PyResult<Py<PyArray2<u32>>> {
    if k < 3 {
        let arr = Array2::<u32>::zeros((0, k.max(1)));
        return Ok(arr.into_pyarray(py).unbind());
    }
    let edges: Vec<(u32, u32)> = edges_u.into_iter().zip(edges_v).collect();
    let (row_ptr, col_idx) = build_csr(&edges, n_nodes, opt.directed);
    let sink = py.detach(|| {
        if matches!(opt.n_threads, Some(1)) {
            serial_enumerate(DfsSerialInput {
                row_ptr: &row_ptr,
                col_idx: &col_idx,
                n_nodes,
                k,
                directed: opt.directed,
                max_cycles: opt.max_cycles,
                seed: opt.seed,
                early_stop: opt.early_stop,
            })
        } else {
            enumerate_parallel(
                &row_ptr,
                &col_idx,
                n_nodes,
                k,
                opt.directed,
                opt.max_cycles,
                opt.seed,
                opt.early_stop,
                opt.n_threads,
            )
        }
    });
    let buf = sink.into_flat();
    flat_to_pyarray2(py, buf, k, opt.max_cycles)
}

fn run_color_coded(
    py: Python<'_>,
    edges_u: Vec<u32>,
    edges_v: Vec<u32>,
    n_nodes: usize,
    k: usize,
    opt: UnsignedOptions,
) -> PyResult<Py<PyArray2<u32>>> {
    if !(3..=16).contains(&k) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "color-coded enumerator requires 3 <= k <= 16",
        ));
    }
    let Some(target_cycles) = opt.target_cycles else {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "backend 'color_coded' requires target_cycles",
        ));
    };
    let buf = py.detach(|| {
        hymeko_graph::color_coding::enumerate_color_coded(
            &edges_u,
            &edges_v,
            n_nodes,
            k,
            target_cycles,
            opt.seed,
            opt.max_colorings,
            opt.n_threads,
        )
    });
    flat_to_pyarray2(py, buf, k, Some(target_cycles))
}

fn run_path_closure(
    py: Python<'_>,
    edges_u: Vec<u32>,
    edges_v: Vec<u32>,
    n_nodes: usize,
    k: usize,
    opt: UnsignedOptions,
) -> PyResult<Py<PyArray2<u32>>> {
    if k < 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "path-closure sampler requires k >= 3",
        ));
    }
    let Some(target_cycles) = opt.target_cycles else {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "backend 'path_closure' requires target_cycles",
        ));
    };
    let buf = py.detach(|| {
        hymeko_graph::path_closure::enumerate_path_closure(
            &edges_u,
            &edges_v,
            n_nodes,
            k,
            target_cycles,
            opt.seed,
            opt.max_attempts,
            opt.n_threads,
        )
    });
    flat_to_pyarray2(py, buf, k, Some(target_cycles))
}

fn run_walks(
    py: Python<'_>,
    edges_u: Vec<u32>,
    edges_v: Vec<u32>,
    n_nodes: usize,
    walk_len: usize,
    opt: UnsignedOptions,
) -> PyResult<Py<PyArray2<u32>>> {
    if walk_len == 0 {
        let arr = Array2::<u32>::zeros((0, 1));
        return Ok(arr.into_pyarray(py).unbind());
    }
    let buf = py.detach(|| {
        hymeko_graph::walks_unsigned::enumerate_walks(
            &edges_u,
            &edges_v,
            n_nodes,
            walk_len,
            opt.max_walks,
            opt.seed,
        )
    });
    flat_to_pyarray2(py, buf, walk_len + 1, opt.max_walks)
}

/// Canonical unsigned entry: `backend` selects algorithm; only relevant kwargs apply.
#[allow(clippy::too_many_arguments)]
// PyO3 flat kwargs surface; a config struct would not reduce Python arity.
#[pyfunction]
#[pyo3(signature = (
    backend,
    edges_u, edges_v, n_nodes, k,
    max_cycles=None, seed=0u64, directed=false, early_stop=false, n_threads=None,
    target_cycles=None, max_colorings=None, max_attempts=None, max_walks=None,
))]
pub fn enumerate_unsigned_rs(
    py: Python<'_>,
    backend: &str,
    edges_u: Vec<u32>,
    edges_v: Vec<u32>,
    n_nodes: usize,
    k: usize,
    max_cycles: Option<usize>,
    seed: u64,
    directed: bool,
    early_stop: bool,
    n_threads: Option<usize>,
    target_cycles: Option<usize>,
    max_colorings: Option<usize>,
    max_attempts: Option<usize>,
    max_walks: Option<usize>,
) -> PyResult<Py<PyArray2<u32>>> {
    let b = UnsignedBackend::parse(backend)?;
    let opt = UnsignedOptions {
        max_cycles,
        seed,
        directed,
        early_stop,
        n_threads,
        target_cycles,
        max_colorings,
        max_attempts,
        max_walks,
    };
    dispatch_unsigned(py, b, edges_u, edges_v, n_nodes, k, opt)
}

#[allow(clippy::too_many_arguments)]
// Legacy stable name; mirrors `enumerate_unsigned_rs(..., "dfs", ...)`.
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
    dispatch_unsigned(
        py,
        UnsignedBackend::Dfs,
        edges_u,
        edges_v,
        n_nodes,
        k,
        UnsignedOptions {
            max_cycles,
            seed,
            directed,
            early_stop,
            n_threads,
            ..UnsignedOptions::default()
        },
    )
}

/// Serial enumeration path — kept as a 1-thread correctness reference and
/// as a fallback for tiny graphs where parallel overhead dominates.
struct DfsSerialInput<'a> {
    row_ptr: &'a [u32],
    col_idx: &'a [u32],
    n_nodes: usize,
    k: usize,
    directed: bool,
    max_cycles: Option<usize>,
    seed: u64,
    early_stop: bool,
}

fn serial_enumerate(p: DfsSerialInput<'_>) -> Sink {
    let global_es = if p.early_stop && p.max_cycles.is_some() {
        Some(std::sync::Arc::new(AtomicUsize::new(0)))
    } else {
        None
    };
    let mut s = match (p.max_cycles, &global_es) {
        (Some(cap), Some(g)) => Sink::new_early_stop(cap, g.clone()),
        (Some(cap), None) => Sink::new_reservoir(cap, p.seed),
        (None, _) => Sink::new_full(),
    };
    let mut visited: Vec<u64> = vec![0u64; bs_words(p.n_nodes)];
    let mut path: Vec<u32> = Vec::with_capacity(p.k);
    let mut dist: Vec<u8> = if p.directed {
        Vec::new()
    } else {
        vec![u8::MAX; p.n_nodes]
    };
    let mut bfs_a: Vec<u32> = Vec::new();
    let mut bfs_b: Vec<u32> = Vec::new();
    for start in 0..p.n_nodes as u32 {
        if !p.directed {
            bfs_distances_into(
                p.row_ptr,
                p.col_idx,
                start,
                p.n_nodes,
                p.k as u8,
                &mut dist,
                &mut bfs_a,
                &mut bfs_b,
            );
        }
        let cont = dfs_from(
            p.row_ptr,
            p.col_idx,
            start,
            p.k,
            p.directed,
            &mut visited,
            &mut path,
            &dist,
            &mut s,
        );
        if !cont {
            break;
        }
    }
    s
}

#[allow(clippy::too_many_arguments)]
// PyO3 legacy surface; delegates to `dispatch_unsigned`.
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
    dispatch_unsigned(
        py,
        UnsignedBackend::ColorCoded,
        edges_u,
        edges_v,
        n_nodes,
        k,
        UnsignedOptions {
            target_cycles: Some(target_cycles),
            seed,
            max_colorings,
            n_threads,
            ..UnsignedOptions::default()
        },
    )
}

#[allow(clippy::too_many_arguments)]
// PyO3 legacy surface; delegates to `dispatch_unsigned`.
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
    dispatch_unsigned(
        py,
        UnsignedBackend::PathClosure,
        edges_u,
        edges_v,
        n_nodes,
        k,
        UnsignedOptions {
            target_cycles: Some(target_cycles),
            seed,
            max_attempts,
            n_threads,
            ..UnsignedOptions::default()
        },
    )
}

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
    dispatch_unsigned(
        py,
        UnsignedBackend::Walks,
        edges_u,
        edges_v,
        n_nodes,
        walk_len,
        UnsignedOptions {
            max_walks,
            seed,
            ..UnsignedOptions::default()
        },
    )
}
