use crate::common::ids::{EdgeId};
use crate::tensor::common::{Real};
use crate::tensor::representations::tensor_coo::TensorCoo;
use crate::tensor::tensor_val::{EdgeWeight, IncVal};
use crate::traversal::hypergraphview::HyperGraphView;


pub fn dense_view_slice<F: Real>(coo: &TensorCoo<F>, k_sel: usize) -> Vec<Vec<F>> {
    assert!(k_sel < coo.num_slices, "k out of range");

    let mut m = vec![vec![F::zero(); coo.dim_j]; coo.dim_i];

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
    let mut deg_v = vec![F::zero(); n];
    let mut deg_e = vec![F::zero(); m];

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

