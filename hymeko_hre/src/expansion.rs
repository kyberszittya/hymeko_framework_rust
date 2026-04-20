//! Hypergraph expansion operators — turn a `HyperGraphView` into a sparse
//! `TensorCoo` representation for downstream tensor algebra.
//!
//! Moved from `hymeko_core::tensor::representations::tensor_coo_representation`
//! into `hymeko_hre` on 2026-04-18 as part of the HRE engine-ops consolidation.
//! The functions still operate on `HyperGraphView` (which stays in `hymeko_core`
//! to avoid a cycle with `hymeko_core::tensor::conv` HGNN operators).

use hymeko::common::ids::{EdgeId, NodeId};
use hymeko_hnn::tensor::common::calc_approx_nnz;
use hymeko::tensor::common::{signed_incidence, Real};
use hymeko_hnn::tensor::common_traversal::inc_to_real;
use hymeko::tensor::representations::tensor_coo::TensorCoo;
use hymeko_hnn::tensor::tensor::compute_bipartite_degrees;
use hymeko::tensor::tensor_val::{EdgeWeight, IncVal};
use hymeko_hnn::traversal::hypergraphview::HyperGraphView;

/// Normalized star expansion. Each incidence weight is divided by
/// `sqrt(deg_v * deg_e)` so the spectral radius stays bounded.
pub fn star_expansion_coo_normalized<V, EW, F>(
    hg: &HyperGraphView<V, EW, F>,
    use_abs: bool,
    eps: F,
) -> TensorCoo<F>
where
    V: IncVal<F>,
    EW: EdgeWeight<V, F>,
    F: Real,
{
    let (deg_v, deg_e) = compute_bipartite_degrees(hg, true);

    let num_nodes = hg.num_nodes();
    let num_edges = hg.num_edges();
    let dim = num_nodes + num_edges;
    let edge_base = num_nodes;

    let approx_nnz = (hg.flat_edge_nodes.len() * 2).max(16);
    let mut t = TensorCoo::with_meta(num_edges, dim, dim);
    t.reserve(approx_nnz);

    for e in 0..num_edges {
        let eid = EdgeId::new(e);
        let (s, eend) = hg.edge_span(eid);
        let e_v = edge_base + e;
        let de = deg_e[e].max(eps);

        for p in s..eend {
            let nid = hg.flat_edge_nodes[p];
            let v = nid.0;
            let n_v = v;
            let sign = hg.flat_edge_sign[p];

            let mut b = inc_to_real(hg, p, e) * signed_incidence::<F>(sign);
            if use_abs {
                b = b.abs();
            }

            let dv = deg_v[v].max(eps);
            let w = b / (dv * de).sqrt();
            match sign {
                1 => t.push(e, n_v, e_v, w),
                -1 => t.push(e, e_v, n_v, w),
                _ => {
                    t.push(e, n_v, e_v, w);
                    t.push(e, e_v, n_v, w);
                }
            }
        }
    }

    t
}

/// Clique-expansion tensor (|V| x |V| x |E|) COO.
/// Slice k=e: connect all nodes incident to e (2-section).
/// Direction handling (Bretto-inspired):
/// '+' makes u more "outgoing", '-' makes u more "incoming".
pub fn clique_expansion_coo<V, EW, F>(hg: &HyperGraphView<V, EW, F>) -> TensorCoo<F>
where
    V: IncVal<F>,
    EW: EdgeWeight<V, F>,
    F: Real,
{
    let num_nodes = hg.num_nodes();
    let num_edges = hg.num_edges();

    let mut t = TensorCoo::with_meta(num_edges, num_nodes, num_nodes);
    let approx_nnz = calc_approx_nnz(hg, num_edges);
    t.reserve(approx_nnz);

    for e in 0..num_edges {
        let eid = EdgeId::new(e);
        let (s, eend) = hg.edge_span(eid);

        let mut nodes: Vec<(usize, i8, F)> = Vec::with_capacity(eend - s);
        for p in s..eend {
            let u = hg.flat_edge_nodes[p].0;
            let su = hg.flat_edge_sign[p];
            let wu: F = inc_to_real(hg, p, e);

            nodes.push((u, su, wu));
        }

        for a in 0..nodes.len() {
            let (u, su, wu) = nodes[a];
            for b in (a + 1)..nodes.len() {
                let (v, sv, wv) = nodes[b];
                let w: F = wu * wv;

                match (su, sv) {
                    (1, -1) => {
                        t.push(e, u, v, w);
                    }
                    (-1, 1) => {
                        t.push(e, v, u, w);
                    }
                    _ => {
                        t.push(e, u, v, w);
                        t.push(e, v, u, w);
                    }
                }
            }
        }
    }

    t
}

/// Star-expansion tensor (|V*| x |V*| x |E|) COO.
/// V* := V ∪ E  (edges are placed after nodes)
/// Bretto: '+' means node -> edge, '-' means edge -> node.
pub fn star_expansion_coo<V, EW, F>(hg: &HyperGraphView<V, EW, F>) -> TensorCoo<F>
where
    V: IncVal<F>,
    EW: EdgeWeight<V, F>,
    F: Real,
{
    let num_nodes = hg.num_nodes();
    let num_edges = hg.num_edges();
    let dim = num_nodes + num_edges;
    let edge_base = num_nodes;

    let approx_nnz = (hg.flat_edge_nodes.len() * 2).max(16);

    let mut t = TensorCoo::with_meta(num_edges, dim, dim);
    t.reserve(approx_nnz);

    for e in 0..num_edges {
        let eid = EdgeId::new(e);
        let (s, eend) = hg.edge_span(eid);
        let u_eid = eid.0;
        let e_v = edge_base + u_eid;

        for p in s..eend {
            let nid: NodeId = hg.flat_edge_nodes[p];
            let n_v = nid.0;
            let sign = hg.flat_edge_sign[p];
            let w: F = inc_to_real(hg, p, u_eid);

            match sign {
                1 => {
                    t.push(u_eid, n_v, e_v, w);
                }
                -1 => {
                    t.push(u_eid, e_v, n_v, w);
                }
                _ => {
                    t.push(u_eid, n_v, e_v, w);
                    t.push(u_eid, e_v, n_v, w);
                }
            }
        }
    }

    t
}
