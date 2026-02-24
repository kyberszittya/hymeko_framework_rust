use crate::common::ids::{EdgeId, NodeId};
use crate::traversal::hypergraphview::HyperGraphView;

/// Sparse 3D tensor in COO form (k, i, j, value).
/// Intended as an intermediate export format for JAX/PyTorch.
///
/// - `k` selects the slice (e.g., edge index / hyperedge id).
/// - `i`, `j` are row/col indices inside the slice.
/// - `v` is the weight/value (usually f32).
#[derive(Clone, Debug, Default)]
pub struct TensorCoo {
    /// Number of slices (e.g., |E|)
    pub num_slices: usize,
    /// First dimension size (e.g., |V*| or |V|)
    pub dim_i: usize,
    /// Second dimension size (e.g., |V*| or |V|)
    pub dim_j: usize,
    /// Slice index (0..num_slices)
    pub k: Vec<usize>,
    /// Row index (0..dim_i)
    pub i: Vec<usize>,
    /// Column index (0..dim_j)
    pub j: Vec<usize>,
    /// Value / weight (e.g., f(.) from the dissertation)
    pub v: Vec<f32>,


}

impl TensorCoo {
    pub fn new(num_slices: usize, dim_i: usize, dim_j: usize) -> Self {
        Self { num_slices, dim_i, dim_j, k: vec![], i: vec![], j: vec![], v: vec![] }
    }

    #[inline(always)]
    pub fn with_meta(num_slices: usize, dim_i: usize, dim_j: usize) -> Self {
        Self {
            num_slices,
            dim_i,
            dim_j,
            k: Vec::new(),
            i: Vec::new(),
            j: Vec::new(),
            v: Vec::new(),

        }
    }

    /// Push one non-zero entry (k, i, j, value).
    #[inline(always)]
    pub fn push(&mut self, k: usize, i: usize, j: usize, value: f32) {
        self.k.push(k);
        self.i.push(i);
        self.j.push(j);
        self.v.push(value);
    }

    #[inline(always)]
    pub fn reserve(&mut self, n: usize) {
        self.k.reserve(n);
        self.i.reserve(n);
        self.j.reserve(n);
        self.v.reserve(n);
    }

    #[inline(always)]
    pub fn len(&self) -> usize { self.v.len() }

    #[inline(always)]
    pub fn is_empty(&self) -> bool { self.v.is_empty() }
}

/// Star-expansion tensor (|V*| x |V*| x |E|) COO.
/// V* := V ∪ E  (edges are placed after nodes)
/// Bretto: '+' means node -> edge, '-' means edge -> node.
pub fn star_expansion_coo(hg: &HyperGraphView) -> TensorCoo {
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
            let w: f32 = 1.0;

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
pub fn clique_expansion_coo(hg: &HyperGraphView) -> TensorCoo {
    let num_nodes = hg.num_nodes();
    let num_edges = hg.num_edges();

    // rough upper bound: per edge, deg^2 potential pairs (dense), but we stay COO sparse
    let mut t = TensorCoo::with_meta(num_edges, num_nodes, num_nodes);

    for e in 0..(num_edges as usize) {
        let eid = EdgeId(e);
        let (s, eend) = hg.edge_span(eid);

        // gather (node, sign) for this edge
        let mut nodes: Vec<(usize, i8)> = Vec::with_capacity(eend - s);
        for p in s..eend {
            nodes.push((hg.flat_edge_nodes[p].0, hg.flat_edge_sign[p]));
        }

        // pairwise fill
        for a in 0..nodes.len() {
            for b in 0..nodes.len() {
                if a == b { continue; }
                let (u, su) = nodes[a];
                let (v, _sv) = nodes[b];
                let w: f32 = 1.0;

                match su {
                    1 => {        // '+' : u tends to point outward
                        t.push(eid.0, u, v, w);
                    }
                    -1 => {       // '-' : u tends to be incoming -> flip direction
                        t.push(eid.0, v, u, w);
                    }
                    _ => {        // neutral: treat as undirected pair entry
                        t.push(eid.0, u, v, w);
                    }
                }
            }
        }
    }

    t
}

pub fn dense_view_slice(coo: &TensorCoo, k_sel: usize) -> Vec<Vec<f32>> {
    assert!(k_sel < coo.num_slices, "k out of range");

    let mut m = vec![vec![0.0f32; coo.dim_j]; coo.dim_i];

    for t in 0..coo.len() {
        if coo.k[t] != k_sel { continue; }
        let i = coo.i[t];
        let j = coo.j[t];
        m[i][j] += coo.v[t]; // coalesce by summation
    }
    m
}

pub fn print_dense_block(
    coo: &TensorCoo,
    k_sel: usize,
    row0: usize,
    col0: usize,
    rows: usize,
    cols: usize,
) {
    assert!(k_sel < coo.num_slices, "k out of range");
    assert!(row0 < coo.dim_i && col0 < coo.dim_j, "start out of range");

    let r = rows.min(coo.dim_i - row0);
    let c = cols.min(coo.dim_j - col0);

    let mut block = vec![vec![0.0f32; c]; r];

    for t in 0..coo.len() {
        if coo.k[t] != k_sel { continue; }
        let i = coo.i[t];
        let j = coo.j[t];

        if i >= row0 && i < row0 + r && j >= col0 && j < col0 + c {
            block[i - row0][j - col0] += coo.v[t];
        }
    }

    println!(
        "slice k={k_sel}, block rows [{row0}..{}), cols [{col0}..{})",
        row0 + r,
        col0 + c
    );

    for i in 0..r {
        for j in 0..c {
            let x = block[i][j];
            if (x - x.round()).abs() < 1e-6 {
                print!("{:>3} ", x.round() as i32);
            } else {
                print!("{:>6.2} ", x);
            }
        }
        println!();
    }
}

pub fn project_sum_over_slices(coo: &TensorCoo) -> Vec<Vec<f32>> {
    let mut m = vec![vec![0.0f32; coo.dim_j]; coo.dim_i];
    for t in 0..coo.len() {
        m[coo.i[t]][coo.j[t]] += coo.v[t];
    }
    m
}