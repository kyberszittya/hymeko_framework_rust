use crate::common::ids::EdgeId;
use crate::tensor::common::Real;
use crate::tensor::representations::tensor_csr::TensorCsr;
use crate::tensor::tensor_val::{EdgeWeight, IncVal};
use crate::traversal::hypergraphview::HyperGraphView;

/// Direct IR-to-CSR Star Expansion.
/// V* := V ∪ E. Matrix dimensions: (|V| + |E|) x (|V| + |E|)
pub fn star_expansion_csr<V, EW, F>(hg: &HyperGraphView<V, EW, F>) -> TensorCsr<F>
where
    V: IncVal<F>,
    EW: EdgeWeight<V, F>,
    F: Real,
{
    let num_nodes = hg.num_nodes();
    let num_edges = hg.num_edges();
    let dim = num_nodes + num_edges;
    let edge_base = num_nodes;

    // --- PASS 1: Calculate Degrees & Prefix Sum ---
    // We must count exactly how many non-zero entries (nnz) originate from each row.
    let mut row_counts = vec![0_usize; dim];

    for e in 0..num_edges {
        let eid = EdgeId(e);
        let (s, eend) = hg.edge_span(eid);
        let u_eid = e;
        let e_v = edge_base + u_eid;

        for p in s..eend {
            let n_v = hg.flat_edge_nodes[p].0 as usize;
            let sign = hg.flat_edge_sign[p];

            match sign {
                1 => row_counts[u_eid] += 1,       // node -> edge (stored in row u_eid)
                -1 => row_counts[u_eid] += 1,      // edge -> node (stored in row u_eid)
                _ => row_counts[u_eid] += 2,       // neutral: both directions
            }
        }
    }

    // Prefix sum to build `row_ptr`
    let mut total_nnz = 0;
    let mut row_ptr = vec![0; dim + 1];
    for i in 0..dim {
        row_ptr[i] = total_nnz;
        total_nnz += row_counts[i];
    }
    row_ptr[dim] = total_nnz;

    // --- PASS 2: Direct Injection ---
    let mut csr = TensorCsr::with_capacity(dim, dim, total_nnz);
    csr.row_ptr = row_ptr.clone(); // Safe to clone, we need `row_ptr` as our current insertion offsets

    // We use a mutable copy of row_ptr to track insertion indices for each row
    let mut current_offset = row_ptr;

    for e in 0..num_edges {
        let eid = EdgeId(e);
        let (s, eend) = hg.edge_span(eid);
        let u_eid = e;
        let e_v = edge_base + u_eid;

        for p in s..eend {
            let n_v = hg.flat_edge_nodes[p].0 as usize;
            let sign = hg.flat_edge_sign[p];

            // Note: Replace with your actual inc_to_real function call
            let w = F::zero(); // Placeholder for: inc_to_real(hg, p, u_eid);

            let mut insert = |row: usize, col: usize, val: F| {
                let idx = current_offset[row];
                csr.col_ind[idx] = col;
                csr.val[idx] = val;
                current_offset[row] += 1;
            };

            match sign {
                1 => insert(u_eid, e_v, w),     // '+' node -> edge
                -1 => insert(u_eid, n_v, w),    // '-' edge -> node
                _ => {
                    insert(u_eid, e_v, w);
                    insert(u_eid, n_v, w);
                }
            }
        }
    }

    csr
}