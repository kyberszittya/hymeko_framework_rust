//! Unified top-K-global Strategy entries (CLAUDE.md §6.5 #1).
//!
//! Two entries (4 legacy variants → 2 cohesive families):
//!   * `enumerate_top_k_cycles_rs`         — regular scorers + optional start-local ABB
//!   * `enumerate_top_k_cycles_entropy_rs` — entropy heuristic + optional hybrid α-blend

use numpy::{PyArray1, PyArray2};
use pyo3::prelude::*;

use super::io::{
    batch_to_pyarray, build_signed_graph, pick_scorer,
    enumerate_top_k_cycles_par_batched, enumerate_top_k_cycles_par_bb_batched,
    enumerate_top_k_cycles_par_entropy_batched,
    BalanceMode, BalanceScorer, CartwrightHararyPruner, DavisWeakBalancePruner,
    EntropyGainScorer, FractionNegativeScorer, HybridScorer, InverseDegreeScorer,
    LowRootScorer, NoOpPruner, SignProductAbsScorer, TopKCyclesBatch,
};

fn topk_run<F>(
    py: Python<'_>,
    edges_u: Vec<u32>,
    edges_v: Vec<u32>,
    edges_s: Vec<i8>,
    n_nodes: u32,
    k_len: usize,
    compute: F,
) -> PyResult<(Py<PyArray2<u32>>, Py<PyArray1<f64>>)>
where
    F: FnOnce(&hymeko_graph::SignedGraph) -> PyResult<TopKCyclesBatch> + Send,
{
    let g = build_signed_graph(&edges_u, &edges_v, &edges_s, n_nodes)?;
    let batch: TopKCyclesBatch = py.detach(|| compute(&g))?;
    batch_to_pyarray(py, batch, k_len)
}

#[allow(clippy::too_many_arguments)]
// PyO3 kwargs surface for top-K global enumeration (scorer + optional ABB).
#[pyfunction]
#[pyo3(signature = (
    edges_u, edges_v, edges_s, n_nodes, k_len, k_keep,
    score_kind="balance", pruner_kind="none", abb_mode="none",
))]
pub fn enumerate_top_k_cycles_rs(
    py: Python<'_>,
    edges_u: Vec<u32>,
    edges_v: Vec<u32>,
    edges_s: Vec<i8>,
    n_nodes: u32,
    k_len: usize,
    k_keep: usize,
    score_kind: &str,
    pruner_kind: &str,
    abb_mode: &str,
) -> PyResult<(Py<PyArray2<u32>>, Py<PyArray1<f64>>)> {
    match abb_mode {
        "none" | "start_local" => {}
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unknown abb_mode '{other}'; valid: none, start_local",
            )));
        }
    }
    let score_kind = score_kind.to_string();
    let pruner_kind = pruner_kind.to_string();
    let abb_mode = abb_mode.to_string();
    topk_run(py, edges_u, edges_v, edges_s, n_nodes, k_len, move |g| {
        if abb_mode == "start_local" {
            dispatch_topk_abb(g, k_len, k_keep, &score_kind, &pruner_kind)
        } else {
            dispatch_topk_noabb(g, k_len, k_keep, &score_kind, &pruner_kind)
        }
    })
}

#[allow(clippy::too_many_arguments)]
// PyO3 kwargs surface for entropy / hybrid top-K global enumeration.
#[pyfunction]
#[pyo3(signature = (
    edges_u, edges_v, edges_s, n_nodes, k_len, k_keep,
    heuristic_kind="entropy", pruner_kind="none",
    hybrid_signal_kind="fraction_negative",
    hybrid_alpha=0.0_f64,
))]
pub fn enumerate_top_k_cycles_entropy_rs(
    py: Python<'_>,
    edges_u: Vec<u32>,
    edges_v: Vec<u32>,
    edges_s: Vec<i8>,
    n_nodes: u32,
    k_len: usize,
    k_keep: usize,
    heuristic_kind: &str,
    pruner_kind: &str,
    hybrid_signal_kind: &str,
    hybrid_alpha: f64,
) -> PyResult<(Py<PyArray2<u32>>, Py<PyArray1<f64>>)> {
    if !(0.0..=1.0).contains(&hybrid_alpha) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "hybrid_alpha must lie in [0, 1]; got {hybrid_alpha}"
        )));
    }
    let heuristic_kind = heuristic_kind.to_string();
    let pruner_kind = pruner_kind.to_string();
    let hybrid_signal_kind = hybrid_signal_kind.to_string();
    topk_run(py, edges_u, edges_v, edges_s, n_nodes, k_len, move |g| {
        if hybrid_alpha > 0.0 {
            dispatch_hybrid(
                g,
                k_len,
                k_keep,
                &heuristic_kind,
                &pruner_kind,
                &hybrid_signal_kind,
                hybrid_alpha,
            )
        } else {
            dispatch_entropy(g, k_len, k_keep, &heuristic_kind, &pruner_kind)
        }
    })
}

// ─── internal dispatchers ───────────────────────────────────────────

fn dispatch_topk_abb(
    g: &hymeko_graph::SignedGraph,
    k_len: usize,
    k_keep: usize,
    score_kind: &str,
    pruner_kind: &str,
) -> PyResult<TopKCyclesBatch> {
    macro_rules! call_bb {
        ($pruner:expr, $scorer:expr) => {{
            enumerate_top_k_cycles_par_bb_batched(g, k_len, &$pruner, k_keep, &$scorer)
        }};
    }
    macro_rules! dispatch_pruner_bb {
        ($scorer:expr) => {
            match pruner_kind {
                "none" => call_bb!(NoOpPruner, $scorer),
                "balance" => call_bb!(
                    CartwrightHararyPruner { mode: BalanceMode::OnlyBalanced }, $scorer
                ),
                "unbalanced" => call_bb!(
                    CartwrightHararyPruner { mode: BalanceMode::OnlyUnbalanced }, $scorer
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

fn dispatch_topk_noabb(
    g: &hymeko_graph::SignedGraph,
    k_len: usize,
    k_keep: usize,
    score_kind: &str,
    pruner_kind: &str,
) -> PyResult<TopKCyclesBatch> {
    let scorer = pick_scorer(score_kind).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "unknown score_kind '{score_kind}'; valid: balance, fraction_negative, sign_product_abs, low_root"
        ))
    })?;
    Ok(match pruner_kind {
        "none" => enumerate_top_k_cycles_par_batched(g, k_len, &NoOpPruner, k_keep, scorer),
        "balance" => enumerate_top_k_cycles_par_batched(
            g, k_len,
            &CartwrightHararyPruner { mode: BalanceMode::OnlyBalanced },
            k_keep, scorer,
        ),
        "unbalanced" => enumerate_top_k_cycles_par_batched(
            g, k_len,
            &CartwrightHararyPruner { mode: BalanceMode::OnlyUnbalanced },
            k_keep, scorer,
        ),
        "davis" => enumerate_top_k_cycles_par_batched(
            g, k_len, &DavisWeakBalancePruner, k_keep, scorer,
        ),
        other => return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown pruner_kind '{other}'; valid: none, balance, unbalanced, davis"
        ))),
    })
}

fn dispatch_entropy(
    g: &hymeko_graph::SignedGraph,
    k_len: usize,
    k_keep: usize,
    heuristic_kind: &str,
    pruner_kind: &str,
) -> PyResult<TopKCyclesBatch> {
    macro_rules! run_with {
        ($pruner:expr, $heuristic:expr) => {
            Ok(enumerate_top_k_cycles_par_entropy_batched(
                g, k_len, &$pruner, k_keep, &$heuristic,
            ))
        };
    }
    macro_rules! dispatch_pruner {
        ($heuristic:expr) => {
            match pruner_kind {
                "none" => run_with!(NoOpPruner, $heuristic),
                "balance" => run_with!(
                    CartwrightHararyPruner { mode: BalanceMode::OnlyBalanced }, $heuristic
                ),
                "unbalanced" => run_with!(
                    CartwrightHararyPruner { mode: BalanceMode::OnlyUnbalanced }, $heuristic
                ),
                "davis" => run_with!(DavisWeakBalancePruner, $heuristic),
                other => return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "unknown pruner_kind '{other}'; valid: none, balance, unbalanced, davis"
                ))),
            }
        };
    }
    match heuristic_kind {
        "entropy" => dispatch_pruner!(EntropyGainScorer),
        "inverse_degree" => dispatch_pruner!(InverseDegreeScorer),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown heuristic_kind '{other}'; valid: entropy, inverse_degree"
        ))),
    }
}

#[allow(clippy::too_many_arguments)]
fn dispatch_hybrid(
    g: &hymeko_graph::SignedGraph,
    k_len: usize,
    k_keep: usize,
    heuristic_kind: &str,
    pruner_kind: &str,
    hybrid_signal_kind: &str,
    hybrid_alpha: f64,
) -> PyResult<TopKCyclesBatch> {
    macro_rules! run_with {
        ($pruner:expr, $heuristic:expr) => {
            Ok(enumerate_top_k_cycles_par_entropy_batched(
                g, k_len, &$pruner, k_keep, &$heuristic,
            ))
        };
    }
    macro_rules! dispatch_pruner {
        ($heuristic:expr) => {
            match pruner_kind {
                "none" => run_with!(NoOpPruner, $heuristic),
                "balance" => run_with!(
                    CartwrightHararyPruner { mode: BalanceMode::OnlyBalanced }, $heuristic
                ),
                "unbalanced" => run_with!(
                    CartwrightHararyPruner { mode: BalanceMode::OnlyUnbalanced }, $heuristic
                ),
                "davis" => run_with!(DavisWeakBalancePruner, $heuristic),
                other => return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "unknown pruner_kind '{other}'; valid: none, balance, unbalanced, davis"
                ))),
            }
        };
    }
    macro_rules! mk_hybrid_and_run {
        ($signal:expr, $div:expr) => {{
            let hybrid = HybridScorer::new($signal, $div, hybrid_alpha);
            dispatch_pruner!(hybrid)
        }};
    }
    macro_rules! by_signal_and_div {
        ($signal:expr) => {
            match heuristic_kind {
                "entropy" => mk_hybrid_and_run!($signal, EntropyGainScorer),
                "inverse_degree" => mk_hybrid_and_run!($signal, InverseDegreeScorer),
                other => Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "unknown heuristic_kind '{other}'; valid: entropy, inverse_degree"
                ))),
            }
        };
    }
    match hybrid_signal_kind {
        "fraction_negative" => by_signal_and_div!(FractionNegativeScorer),
        "balance" => by_signal_and_div!(BalanceScorer),
        "sign_product_abs" => by_signal_and_div!(SignProductAbsScorer),
        "low_root" => by_signal_and_div!(LowRootScorer),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown hybrid_signal_kind '{other}'; valid: balance, fraction_negative, sign_product_abs, low_root"
        ))),
    }
}
