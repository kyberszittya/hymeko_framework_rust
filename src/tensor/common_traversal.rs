use crate::tensor::common::Real;
use crate::tensor::tensor_val::{EdgeWeight, IncVal};
use crate::traversal::hypergraphview::HyperGraphView;

#[inline(always)]
pub fn inc_to_real<V, EW, F>(hg: &HyperGraphView<V, EW, F>, p: usize, e: usize) -> F
where
    V: IncVal<F>,
    EW: EdgeWeight<V, F>,
    F: Real,
{
    // apply edge weight to the incidence value, then project to scalar
    hg.edge_weight[e].apply_to(hg.flat_edge_w[p].clone()).as_scalar()
}