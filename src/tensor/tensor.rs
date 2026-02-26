use crate::common::ids::{EdgeId, NodeId};
use crate::tensor::common::{signed_incidence, Real};
use crate::tensor::tensor_coo::TensorCoo;
use crate::tensor::tensor_val::{EdgeWeight, IncVal};
use crate::traversal::hypergraphview::HyperGraphView;

#[inline(always)]
fn inc_to_real<V, EW, F>(hg: &HyperGraphView<V, EW, F>, p: usize, e: usize) -> F
where
    V: IncVal<F>,
    EW: EdgeWeight<V, F>,
    F: Real
{
    let v = hg.flat_edge_w[p].clone();            // V
    let v2 = hg.edge_weight[e].apply_to(v);       // V
    v2.as_scalar()
}

/// Star-expansion tensor (|V*| x |V*| x |E|) COO.
/// V* := V ∪ E  (edges are placed after nodes)
/// Bretto: '+' means node -> edge, '-' means edge -> node.
pub fn star_expansion_coo<V, EW, F>(hg: &HyperGraphView<V, EW, F>) -> TensorCoo<F>
where
    V: IncVal<F>,
    EW: EdgeWeight<V, F>,
    F: Real
{
    let num_nodes = hg.num_nodes();
    let num_edges = hg.num_edges();
    let dim = num_nodes + num_edges;
    let edge_base = num_nodes;

    // worst-case ~ 2 incidences per (edge,node) if neutral -> both directions
    let approx_nnz = (hg.flat_edge_nodes.len() * 2).max(16);

    let mut t = TensorCoo::with_meta(num_edges, dim, dim);
    t.reserve(approx_nnz);

    for e in 0..(num_edges as usize) {
        let eid = EdgeId(e);
        let (s, eend) = hg.edge_span(eid);
        let u_eid = eid.0 as usize;
        let e_v = edge_base + u_eid; // edge index in V*

        for p in s..eend {
            let nid: NodeId = hg.flat_edge_nodes[p];
            let n_v = nid.0 as usize; // node index in V*
            let sign = hg.flat_edge_sign[p];
            let w: F = inc_to_real(hg, p, u_eid);

            match sign {
                1 => { // '+' node -> edge
                    t.push(u_eid, n_v, e_v, w);
                }
                -1 => { // '-' edge -> node
                    t.push(u_eid, e_v, n_v, w);
                }
                _ => { // neutral: both
                    t.push(u_eid, n_v, e_v, w);
                    t.push(u_eid, e_v, n_v, w);
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
    F: Real
{
    let num_nodes = hg.num_nodes();
    let num_edges = hg.num_edges();

    // rough upper bound: per edge, deg^2 potential pairs (dense), but we stay COO sparse
    let mut t = TensorCoo::with_meta(num_edges, num_nodes, num_nodes);

    for e in 0..(num_edges as usize) {
        let eid = EdgeId(e);
        let (s, eend) = hg.edge_span(eid);

        // gather (node, sign) for this edge
        let mut nodes: Vec<(usize, i8, F)> = Vec::with_capacity(eend - s);
        for p in s..eend {
            let u = hg.flat_edge_nodes[p].0 as usize;
            let su = hg.flat_edge_sign[p];
            let wu: F = inc_to_real(hg, p, e);

            nodes.push((u, su, wu));
        }

        // pairwise fill
        for a in 0..nodes.len() {
            let (u, su, wu) = nodes[a];
            for b in 0..nodes.len() {
                if a == b { continue; }
                let (v, _sv, wv) = nodes[b];
                let w: F = wu * wv;

                match su {
                    1 => {        // '+' : u tends to point outward
                        t.push(e, u, v, w);
                    }
                    -1 => {       // '-' : u tends to be incoming -> flip direction
                        t.push(e, v, u, w);
                    }
                    _ => {        // neutral: treat as undirected pair entry
                        t.push(e, u, v, w);
                    }
                }
            }
        }
    }

    t
}

pub fn dense_view_slice<F: Real>(coo: &TensorCoo<F>, k_sel: usize) -> Vec<Vec<F>> {
    assert!(k_sel < coo.num_slices, "k out of range");

    let mut m = vec![vec![F::one(); coo.dim_j]; coo.dim_i];

    for t in 0..coo.len() {
        if coo.k[t] != k_sel { continue; }
        m[coo.i[t]][coo.j[t]] += coo.v[t]; // coalesce by summation
    }
    m
}

pub fn project_sum_over_slices<F: Real>(coo: &TensorCoo<F>) -> Vec<Vec<F>> {
    let mut m = vec![vec![F::zero(); coo.dim_j]; coo.dim_i];
    for t in 0..coo.len() {
        m[coo.i[t]][coo.j[t]] += coo.v[t];
    }
    m
}

pub fn compute_bipartite_degrees<V, EW, F>(
    hg: &HyperGraphView<V, EW, F>,
    use_abs: bool,
) -> (Vec<F>, Vec<F>)
where
    V: IncVal<F>,
    EW: EdgeWeight<V, F>,
    F: Real
{
    let n = hg.num_nodes();
    let m = hg.num_edges();
    let mut deg_v = vec![F::one(); n];
    let mut deg_e = vec![F::one(); m];

    for e in 0..m {
        let (s, eend) = hg.edge_span(EdgeId(e));
        let mut de = F::zero();
        for p in s..eend {
            let v = hg.flat_edge_nodes[p].0;
            let mut b = hg.flat_edge_w[p].degree_mass();
            if use_abs { b = b.abs(); }
            deg_v[v] += b;
            de += b;
        }
        deg_e[e] = de;
    }

    (deg_v, deg_e)
}

pub fn star_expansion_coo_normalized<V, EW, F>(
    hg: &HyperGraphView<V, EW, F>,
    use_abs: bool,
    eps: F,
) -> TensorCoo<F>
where
    V: IncVal<F>,
    EW: EdgeWeight<V, F>,
    F: Real
{
    let (deg_v, deg_e) = compute_bipartite_degrees(hg, true); // fokszámhoz abs ajánlott

    let num_nodes = hg.num_nodes();
    let num_edges = hg.num_edges();
    let dim = num_nodes + num_edges;
    let edge_base = num_nodes;

    let approx_nnz = (hg.flat_edge_nodes.len() * 2).max(16);
    let mut t = TensorCoo::with_meta(num_edges, dim, dim);
    t.reserve(approx_nnz);

    for e in 0..num_edges {
        let eid = EdgeId(e);
        let (s, eend) = hg.edge_span(eid);
        let e_v = edge_base + e;
        let de = deg_e[e].max(eps);

        for p in s..eend {
            let nid = hg.flat_edge_nodes[p];
            let v = nid.0;
            let n_v = v;
            let sign = hg.flat_edge_sign[p];

            // raw incidence (signed)
            let mut b = inc_to_real(hg, p, e) * signed_incidence::<F>(sign);
            if use_abs { b = b.abs(); }

            // normalize
            let dv = deg_v[v].max(eps);
            let w = b / (dv * de).sqrt();
            match sign {
                1 => t.push(e, n_v, e_v, w),     // node -> edge
                -1 => t.push(e, e_v, n_v, w),    // edge -> node
                _ => {                           // neutral: both
                    t.push(e, n_v, e_v, w);
                    t.push(e, e_v, n_v, w);
                }
            }
        }
    }

    t
}