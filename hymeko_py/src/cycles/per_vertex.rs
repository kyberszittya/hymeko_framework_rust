//! Unified per-vertex cycle enumeration entry point.
//!
//! Replaces 8 legacy `enumerate_top_k_per_vertex_cycles_signed_*`
//! pyfunctions; all toggles enter through kwargs (CLAUDE.md §6.5 #1).
//! Plan: docs/plans/2026-05-11-cycles-strategy-refactor/

use numpy::{PyArray1, PyArray2};
use pyo3::prelude::*;

use super::io::{
    batch_to_pyarray, build_signed_graph, build_vertex_filter, pick_scorer,
    BalanceMode, BalanceScorer, CartwrightHararyPruner, DavisWeakBalancePruner,
    FractionNegativeScorer, GSignedGraph, LowRootScorer, NoOpPruner,
    SignProductAbsScorer, TopKCyclesBatch,
};
use super::io::{degree_adaptive_m_v, tiered_m_v_by_degree};
use super::io::{
    enumerate_top_k_per_vertex_cycles_par_adaptive_starting_batched,
    enumerate_top_k_per_vertex_cycles_par_adaptive_starting_bb_batched,
    enumerate_top_k_per_vertex_cycles_par_adaptive_starting_bb_global_batched,
};

/// `m_v` selection priority: `tiers` (if non-empty) > `adaptive_c > 0`
/// > flat `vec![m_per_vertex; n_nodes]`.
fn compute_m_v(
    g: &GSignedGraph,
    n_nodes: u32,
    m_per_vertex: u32,
    tiers: &[(f32, u32)],
    adaptive_c: f64,
    adaptive_m_min: u32,
    adaptive_m_max: u32,
) -> PyResult<Vec<u32>> {
    if !tiers.is_empty() {
        for w in tiers.windows(2) {
            if w[0].0 > w[1].0 {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "tiers must be sorted ascending by percentile; got {} > {}",
                    w[0].0, w[1].0
                )));
            }
        }
        Ok(tiered_m_v_by_degree(g, tiers))
    } else if adaptive_c > 0.0 {
        let m_min = if adaptive_m_min == 0 {
            std::cmp::max(2, m_per_vertex / 2)
        } else {
            adaptive_m_min
        };
        let m_max = if adaptive_m_max == 0 {
            std::cmp::max(m_per_vertex.saturating_mul(4), 4)
        } else {
            adaptive_m_max
        };
        Ok(degree_adaptive_m_v(g, m_min, m_max, adaptive_c))
    } else {
        Ok(vec![m_per_vertex; n_nodes as usize])
    }
}

#[pyfunction]
#[pyo3(signature = (
    edges_u, edges_v, edges_s, n_nodes,
    k_len, m_per_vertex,
    score_kind="fraction_negative",
    pruner_kind="none",
    filter_kind="none",
    filter_min_degree=2u32,
    abb_mode="none",
    fullness_gate=0.25_f64,
    tiers=Vec::<(f32, u32)>::new(),
    adaptive_c=0.0_f64,
    adaptive_m_min=0u32,
    adaptive_m_max=0u32,
))]
#[allow(clippy::too_many_arguments)]
pub fn enumerate_cycles_rs(
    py: Python<'_>,
    edges_u: Vec<u32>,
    edges_v: Vec<u32>,
    edges_s: Vec<i8>,
    n_nodes: u32,
    k_len: usize,
    m_per_vertex: u32,
    score_kind: &str,
    pruner_kind: &str,
    filter_kind: &str,
    filter_min_degree: u32,
    abb_mode: &str,
    fullness_gate: f64,
    tiers: Vec<(f32, u32)>,
    adaptive_c: f64,
    adaptive_m_min: u32,
    adaptive_m_max: u32,
) -> PyResult<(Py<PyArray2<u32>>, Py<PyArray1<f64>>)> {
    let g = build_signed_graph(&edges_u, &edges_v, &edges_s, n_nodes)?;
    let filter = build_vertex_filter(filter_kind, filter_min_degree)?;
    let m_v = compute_m_v(
        &g, n_nodes, m_per_vertex, &tiers, adaptive_c,
        adaptive_m_min, adaptive_m_max,
    )?;
    match abb_mode {
        "none" | "start_local" | "global_min" => {}
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unknown abb_mode '{other}'; valid: none, start_local, global_min",
            )))
        }
    }
    let batch: TopKCyclesBatch = py.detach(|| -> PyResult<TopKCyclesBatch> {
        let keep = filter.keep_set(&g);
        if abb_mode != "none" {
            dispatch_abb(
                &g, k_len, &m_v, &keep,
                score_kind, pruner_kind, abb_mode, fullness_gate,
            )
        } else {
            dispatch_noabb(&g, k_len, &m_v, &keep, score_kind, pruner_kind)
        }
    })?;
    batch_to_pyarray(py, batch, k_len)
}

#[allow(clippy::too_many_arguments)]
fn dispatch_abb(
    g: &GSignedGraph,
    k_len: usize,
    m_v: &[u32],
    keep: &[u32],
    score_kind: &str,
    pruner_kind: &str,
    abb_mode: &str,
    fullness_gate: f64,
) -> PyResult<TopKCyclesBatch> {
    macro_rules! call_bb {
        ($pruner:expr, $scorer:expr) => {{
            if abb_mode == "start_local" {
                enumerate_top_k_per_vertex_cycles_par_adaptive_starting_bb_batched(
                    g, k_len, &$pruner, m_v, keep, &$scorer,
                )
            } else {
                enumerate_top_k_per_vertex_cycles_par_adaptive_starting_bb_global_batched(
                    g, k_len, &$pruner, m_v, keep, &$scorer, fullness_gate,
                )
            }
        }};
    }
    macro_rules! dispatch_pruner_bb {
        ($scorer:expr) => {
            match pruner_kind {
                "none" => call_bb!(NoOpPruner, $scorer),
                "balance" => call_bb!(
                    CartwrightHararyPruner { mode: BalanceMode::OnlyBalanced },
                    $scorer
                ),
                "unbalanced" => call_bb!(
                    CartwrightHararyPruner { mode: BalanceMode::OnlyUnbalanced },
                    $scorer
                ),
                "davis" => call_bb!(DavisWeakBalancePruner, $scorer),
                other => return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "unknown pruner_kind '{other}'; valid: none, balance, unbalanced, davis"
                ))),
            }
        };
    }
    Ok(match score_kind {
        "fraction_negative" => dispatch_pruner_bb!(FractionNegativeScorer),
        "balance" => dispatch_pruner_bb!(BalanceScorer),
        "sign_product_abs" => dispatch_pruner_bb!(SignProductAbsScorer),
        "low_root" => dispatch_pruner_bb!(LowRootScorer),
        other => return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown score_kind '{other}'; valid: balance, fraction_negative, sign_product_abs, low_root"
        ))),
    })
}

fn dispatch_noabb(
    g: &GSignedGraph,
    k_len: usize,
    m_v: &[u32],
    keep: &[u32],
    score_kind: &str,
    pruner_kind: &str,
) -> PyResult<TopKCyclesBatch> {
    let scorer = pick_scorer(score_kind).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "unknown score_kind '{score_kind}'; valid: balance, fraction_negative, sign_product_abs, low_root"
        ))
    })?;
    Ok(match pruner_kind {
        "none" => enumerate_top_k_per_vertex_cycles_par_adaptive_starting_batched(
            g, k_len, &NoOpPruner, m_v, keep, scorer,
        ),
        "balance" => enumerate_top_k_per_vertex_cycles_par_adaptive_starting_batched(
            g, k_len,
            &CartwrightHararyPruner { mode: BalanceMode::OnlyBalanced },
            m_v, keep, scorer,
        ),
        "unbalanced" => enumerate_top_k_per_vertex_cycles_par_adaptive_starting_batched(
            g, k_len,
            &CartwrightHararyPruner { mode: BalanceMode::OnlyUnbalanced },
            m_v, keep, scorer,
        ),
        "davis" => enumerate_top_k_per_vertex_cycles_par_adaptive_starting_batched(
            g, k_len, &DavisWeakBalancePruner, m_v, keep, scorer,
        ),
        other => return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown pruner_kind '{other}'; valid: none, balance, unbalanced, davis"
        ))),
    })
}
