use crate::tensor::common::{signed_incidence, Real};
use crate::tensor::common_traversal::inc_to_real;
use crate::tensor::tensor_val::{EdgeWeight, IncVal};
use crate::traversal::hypergraphview::HyperGraphView;

#[derive(Clone, Copy, Debug)]
pub struct CliqueStepCfg {
    pub use_abs: bool,      // default: true
    pub include_self: bool, // default: false (kiveszi az önhurkot)
}

impl Default for CliqueStepCfg {
    fn default() -> Self {
        Self { use_abs: true, include_self: false }
    }
}


/// y = B W B^T x  (implicit, sparse)
/// x_nodes hossza: hg.num_nodes()  (root is benne lehet)
pub fn implicit_clique_step<V, EW, F>(
    hg: &HyperGraphView<V, EW, F>,
    x_nodes: &[F],
    cfg: CliqueStepCfg,
) -> Vec<F>
where
    V: IncVal<F>,
    EW: EdgeWeight<V, F>,
    F: Real
{
    assert_eq!(x_nodes.len(), hg.num_nodes());

    let n = hg.num_nodes();
    let m = hg.num_edges();

    // 1) x_e = B^T x
    let mut x_edges = vec![F::zero(); m];
    for e in 0..m {
        let s = hg.edge_offsets[e];
        let eend = hg.edge_offsets[e + 1];

        let mut acc = F::zero();
        for p in s..eend {
            let v = hg.flat_edge_nodes[p].0; // NodeId -> usize
            let mut b: F = inc_to_real(hg, p, e);
            b *= signed_incidence::<F>(hg.flat_edge_sign[p]);

            if cfg.use_abs { b = b.abs(); }
            acc += b * x_nodes[v];
        }
        x_edges[e] = acc;
    }

    // 2) y = B x_e
    let mut y = vec![F::zero(); n];
    for e in 0..m {
        let s = hg.edge_offsets[e];
        let eend = hg.edge_offsets[e + 1];

        let xe = x_edges[e];

        for p in s..eend {
            let v = hg.flat_edge_nodes[p].0;
            let mut b = inc_to_real(hg, p, e);
            b *= signed_incidence(hg.flat_edge_sign[p]);

            if cfg.use_abs { b = b.abs(); }
            y[v] += b * xe;
        }
    }

    if !cfg.include_self {
        // önhatás eltávolítása (opcionális): y[v] -= x[v] * sum_e b_{v,e}^2
        // Ez a "diag" levétel implicit megfelelője.
        let mut diag = vec![F::zero(); n];
        for e in 0..m {
            let s = hg.edge_offsets[e];
            let eend = hg.edge_offsets[e + 1];

            for p in s..eend {
                let v = hg.flat_edge_nodes[p].0;
                let mut b: F = inc_to_real(hg, p, e);
                b *= signed_incidence::<F>(hg.flat_edge_sign[p]);
                if cfg.use_abs { b = b.abs(); }
                diag[v] += b * b;
            }
        }
        for v in 0..n {
            y[v] -= diag[v] * x_nodes[v];
        }
    }

    y
}



pub fn build_explicit_a<V, EW, F>(
    hg: &HyperGraphView<V, EW, F>, cfg: CliqueStepCfg) -> Vec<Vec<F>>
where
    V: IncVal<F>,
    EW: EdgeWeight<V, F>,
    F: Real
{
    let n = hg.num_nodes();
    let m = hg.num_edges();

    // A = B B^T
    let mut a = vec![vec![F::zero(); n]; n];

    for e in 0..m {
        let s = hg.edge_offsets[e];
        let eend = hg.edge_offsets[e + 1];

        // kis edge-szelet (v, bve)
        let mut nodes: Vec<(usize, F)> = Vec::with_capacity(eend - s);
        for p in s..eend {
            let v = hg.flat_edge_nodes[p].0;
            let mut b: F = inc_to_real(hg, p, e);
            b *= signed_incidence::<F>(hg.flat_edge_sign[p]);
            if cfg.use_abs { b = b.abs(); }
            nodes.push((v, b));
        }

        // outer product: A[u,v] += b(u,e) * b(v,e)
        for &(u, bu) in &nodes {
            for &(v, bv) in &nodes {
                a[u][v] += bu * bv;
            }
        }
    }

    if !cfg.include_self {
        for i in 0..n {
            a[i][i] = F::zero();
        }
    }

    a
}

pub fn print_dense_real<F: Real>(mat: &[Vec<F>], title: &str) {
    let n = mat.len();
    let m = if n > 0 { mat[0].len() } else { 0 };
    println!("{title} ({}x{}):", n, m);
    for i in 0..n {
        for j in 0..m {
            print!("{:7.2} ", mat[i][j].as_f64());
        }
        println!();
    }
}