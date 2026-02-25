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

#[inline(always)]
fn signed_incidence(sign: i8) -> f32 {
    match sign {
        1 => 1.0,
        -1 => -1.0,
        _ => 1.0, // neutral: kezeld +1-nek (abs esetben mindegy)
    }
}

/// y = B W B^T x  (implicit, sparse)
/// x_nodes hossza: hg.num_nodes()  (root is benne lehet)
pub fn implicit_clique_step(
    hg: &HyperGraphView,
    x_nodes: &[f32],
    cfg: CliqueStepCfg,
) -> Vec<f32> {
    assert_eq!(x_nodes.len(), hg.num_nodes());

    let n = hg.num_nodes();
    let m = hg.num_edges();

    // 1) x_e = B^T x
    let mut x_edges = vec![0.0f32; m];
    for eid_usize in 0..m {
        let s = hg.edge_offsets[eid_usize];
        let eend = hg.edge_offsets[eid_usize + 1];

        let ew = hg.edge_weight[eid_usize]; // globális él-súly

        let mut acc = 0.0f32;
        for p in s..eend {
            let v = hg.flat_edge_nodes[p].0; // NodeId -> usize
            let mut b = hg.flat_edge_w[p] * ew;
            let sgn = signed_incidence(hg.flat_edge_sign[p]);
            b *= sgn;

            if cfg.use_abs { b = b.abs(); }
            acc += b * x_nodes[v];
        }
        x_edges[eid_usize] = acc;
    }

    // 2) y = B x_e
    let mut y = vec![0.0f32; n];
    for eid_usize in 0..m {
        let s = hg.edge_offsets[eid_usize];
        let eend = hg.edge_offsets[eid_usize + 1];

        let ew = hg.edge_weight[eid_usize];
        let xe = x_edges[eid_usize];

        for p in s..eend {
            let v = hg.flat_edge_nodes[p].0;
            let mut b = hg.flat_edge_w[p] * ew;
            let sgn = signed_incidence(hg.flat_edge_sign[p]);
            b *= sgn;

            if cfg.use_abs { b = b.abs(); }
            y[v] += b * xe;
        }
    }

    if !cfg.include_self {
        // önhatás eltávolítása (opcionális): y[v] -= x[v] * sum_e b_{v,e}^2
        // Ez a "diag" levétel implicit megfelelője.
        let mut diag = vec![0.0f32; n];
        for eid_usize in 0..m {
            let s = hg.edge_offsets[eid_usize];
            let eend = hg.edge_offsets[eid_usize + 1];

            let ew = hg.edge_weight[eid_usize];
            for p in s..eend {
                let v = hg.flat_edge_nodes[p].0;
                let mut b = hg.flat_edge_w[p] * ew;
                let sgn = signed_incidence(hg.flat_edge_sign[p]);
                b *= sgn;
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

pub fn build_explicit_a(hg: &HyperGraphView, cfg: CliqueStepCfg) -> Vec<Vec<f32>> {
    let n = hg.num_nodes();
    let m = hg.num_edges();

    // A = B B^T
    let mut a = vec![vec![0.0f32; n]; n];

    for e in 0..m {
        let s = hg.edge_offsets[e];
        let eend = hg.edge_offsets[e + 1];
        let ew = hg.edge_weight[e];

        // kis edge-szelet (v, bve)
        let mut nodes: Vec<(usize, f32)> = Vec::with_capacity(eend - s);
        for p in s..eend {
            let v = hg.flat_edge_nodes[p].0;
            let mut b = hg.flat_edge_w[p] * ew;
            let sgn = signed_incidence(hg.flat_edge_sign[p]); // ugyanaz a helper
            b *= sgn;
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
            a[i][i] = 0.0;
        }
    }

    a
}

pub fn print_dense_f32(mat: &[Vec<f32>], title: &str) {
    let n = mat.len();
    let m = if n > 0 { mat[0].len() } else { 0 };
    println!("{title} ({}x{}):", n, m);
    for i in 0..n {
        for j in 0..m {
            print!("{:7.2} ", mat[i][j]);
        }
        println!();
    }
}