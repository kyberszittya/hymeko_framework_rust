//! Hypergraph convolution operations.
//!
//! Three levels, all operating on TensorCsr from star or clique expansion:
//! - L1 GcnConv:        standard GCN on clique-expanded adjacency
//! - L2 HgnnConv:       HGNN on star-expanded incidence
//! - L3 SignedHgnnConv:  novel — separate W₊/W₋ for signed incidence

use crate::tensor::common::Real;
use crate::tensor::representations::tensor_csr::{TensorCsr, build_row_ptr};

// ============================================================
// Trait
// ============================================================

/// A convolution operation over hypergraph tensors.
pub trait HypergraphConv<F: Real> {
    /// Input:  features (|V| × dim_in), row-major flattened
    /// Output: features (|V| × dim_out), row-major flattened
    fn convolve(
        &self,
        adjacency: &TensorCsr<F>,  // from star or clique expansion
        features: &[F],
        num_vertices: usize,
        dim_in: usize,
    ) -> Vec<F>;

    fn dim_out(&self) -> usize;
}

// ============================================================
// Helpers
// ============================================================

/// Dense matmul: C = A * B, A is (m×k), B is (k×n), row-major
fn dense_matmul<F: Real>(a: &[F], b: &[F], m: usize, k: usize, n: usize) -> Vec<F> {
    let mut c = vec![F::zero(); m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = F::zero();
            for p in 0..k {
                s += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = s;
        }
    }
    c
}

/// Row-degree vector from CSR: deg[i] = number of nonzeros in row i
fn row_degrees_csr<F: Real>(csr: &TensorCsr<F>) -> Vec<F> {
    let mut deg = vec![F::zero(); csr.num_rows];
    for i in 0..csr.num_rows {
        let start = csr.row_ptr[i];
        let end = csr.row_ptr[i + 1];
        for k in start..end {
            deg[i] += csr.val[k].abs();
        }
    }
    deg
}

/// Inverse sqrt of degree vector, with zero-protection
fn inv_sqrt_vec<F: Real>(deg: &[F]) -> Vec<F> {
    deg.iter().map(|&d| {
        if d > F::zero() { F::one() / d.sqrt() } else { F::zero() }
    }).collect()
}

/// Scale each row of a flattened matrix by a diagonal vector:
/// out[i, :] = diag[i] * mat[i, :]
fn scale_rows<F: Real>(mat: &[F], diag: &[F], rows: usize, cols: usize) -> Vec<F> {
    let mut out = vec![F::zero(); rows * cols];
    for i in 0..rows {
        let d = diag[i];
        for j in 0..cols {
            out[i * cols + j] = mat[i * cols + j] * d;
        }
    }
    out
}

/// Element-wise ReLU in-place
fn relu<F: Real>(x: &mut [F]) {
    for v in x.iter_mut() {
        if *v < F::zero() { *v = F::zero(); }
    }
}

/// Build the transpose of a CSR matrix → new CSR
fn transpose_csr<F: Real>(csr: &TensorCsr<F>) -> TensorCsr<F> {
    let mut counts = vec![0usize; csr.num_cols];
    for &c in &csr.col_ind {
        counts[c] += 1;
    }
    let (row_ptr, nnz) = build_row_ptr(&counts);
    let mut out = TensorCsr::with_capacity(csr.num_cols, csr.num_rows, nnz);
    out.row_ptr = row_ptr;
    let mut offset = out.row_ptr.clone();

    for i in 0..csr.num_rows {
        let start = csr.row_ptr[i];
        let end = csr.row_ptr[i + 1];
        for k in start..end {
            let c = csr.col_ind[k];
            let idx = offset[c];
            out.col_ind[idx] = i;
            out.val[idx] = csr.val[k];
            offset[c] += 1;
        }
    }
    out
}

/// Split a CSR by sign: positive entries → csr_pos, |negative| entries → csr_neg
fn split_by_sign<F: Real>(csr: &TensorCsr<F>) -> (TensorCsr<F>, TensorCsr<F>) {
    // Count pass
    let mut pos_counts = vec![0usize; csr.num_rows];
    let mut neg_counts = vec![0usize; csr.num_rows];
    for i in 0..csr.num_rows {
        let start = csr.row_ptr[i];
        let end = csr.row_ptr[i + 1];
        for k in start..end {
            if csr.val[k] > F::zero() { pos_counts[i] += 1; }
            else if csr.val[k] < F::zero() { neg_counts[i] += 1; }
        }
    }

    let (pos_rp, pos_nnz) = build_row_ptr(&pos_counts);
    let (neg_rp, neg_nnz) = build_row_ptr(&neg_counts);

    let mut pos = TensorCsr::with_capacity(csr.num_rows, csr.num_cols, pos_nnz);
    let mut neg = TensorCsr::with_capacity(csr.num_rows, csr.num_cols, neg_nnz);
    pos.row_ptr = pos_rp;
    neg.row_ptr = neg_rp;

    let mut pos_off = pos.row_ptr.clone();
    let mut neg_off = neg.row_ptr.clone();

    for i in 0..csr.num_rows {
        let start = csr.row_ptr[i];
        let end = csr.row_ptr[i + 1];
        for k in start..end {
            let v = csr.val[k];
            let c = csr.col_ind[k];
            if v > F::zero() {
                let idx = pos_off[i];
                pos.col_ind[idx] = c;
                pos.val[idx] = v;
                pos_off[i] += 1;
            } else if v < F::zero() {
                let idx = neg_off[i];
                neg.col_ind[idx] = c;
                neg.val[idx] = v.abs();
                neg_off[i] += 1;
            }
        }
    }

    (pos, neg)
}

// ============================================================
// Level 1: GCN on clique-expanded adjacency
// H' = σ(D̃⁻¹/² Ã D̃⁻¹/² H W)
// ============================================================

pub struct GcnConv<F: Real> {
    /// Weight matrix W, (dim_in × dim_out) row-major
    pub weights: Vec<F>,
    pub dim_in: usize,
    pub dim_out: usize,
}

impl<F: Real> HypergraphConv<F> for GcnConv<F> {
    fn dim_out(&self) -> usize { self.dim_out }

    fn convolve(
        &self,
        adj: &TensorCsr<F>,       // clique-expanded |V|×|V|
        features: &[F],
        num_vertices: usize,
        dim_in: usize,
    ) -> Vec<F> {
        // HW
        let hw = dense_matmul(features, &self.weights, num_vertices, dim_in, self.dim_out);

        // D̃ = D + I (self-loops)
        let mut deg = row_degrees_csr(adj);
        for d in deg.iter_mut() { *d += F::one(); }
        let d_inv_sqrt = inv_sqrt_vec(&deg);

        // D̃⁻¹/² HW
        let scaled = scale_rows(&hw, &d_inv_sqrt, num_vertices, self.dim_out);

        // Ã(D̃⁻¹/² HW) = A(D̃⁻¹/² HW) + D̃⁻¹/² HW  (self-loop)
        let a_scaled = adj.spmm(&scaled, self.dim_out);
        let mut out = vec![F::zero(); num_vertices * self.dim_out];
        for i in 0..num_vertices {
            for f in 0..self.dim_out {
                let idx = i * self.dim_out + f;
                out[idx] = (a_scaled[idx] + scaled[idx]) * d_inv_sqrt[i];
            }
        }

        relu(&mut out);
        out
    }
}

// ============================================================
// Level 2: HGNN on star-expanded incidence
// H' = σ(Dv⁻¹/² B De⁻¹ Bᵀ Dv⁻¹/² H W)
// B is the star-expansion CSR (|V*|×|V*|)
// ============================================================

pub struct HgnnConv<F: Real> {
    pub weights: Vec<F>,
    pub dim_in: usize,
    pub dim_out: usize,
}

impl<F: Real> HypergraphConv<F> for HgnnConv<F> {
    fn dim_out(&self) -> usize { self.dim_out }

    fn convolve(
        &self,
        incidence: &TensorCsr<F>,  // star-expanded (|V|+|E|) × (|V|+|E|)
        features: &[F],            // only |V| × dim_in (vertex features)
        num_vertices: usize,       // |V| (not |V|+|E|)
        dim_in: usize,
    ) -> Vec<F> {
        let dim_star = incidence.num_rows; // |V| + |E|
        let bt = transpose_csr(incidence);

        // HW (only vertex rows)
        let hw = dense_matmul(features, &self.weights, num_vertices, dim_in, self.dim_out);

        // Vertex degree from incidence
        let dv = row_degrees_csr(incidence);
        let dv_inv_sqrt = inv_sqrt_vec(&dv[..num_vertices]);

        // Edge degree (rows num_vertices..dim_star in incidence)
        let de_inv: Vec<F> = (num_vertices..dim_star).map(|i| {
            let d = dv[i]; // reuse full degree vec
            if d > F::zero() { F::one() / d } else { F::zero() }
        }).collect();

        let mut out = vec![F::zero(); num_vertices * self.dim_out];

        for f in 0..self.dim_out {
            // Zero-padded vector for star space: [Dv⁻¹/²·(HW)_f ; 0...0]
            let mut x = vec![F::zero(); dim_star];
            for i in 0..num_vertices {
                x[i] = hw[i * self.dim_out + f] * dv_inv_sqrt[i];
            }

            // Bᵀ x → edge aggregation
            let btx = bt.spmv(&x);

            // De⁻¹ · (Bᵀx) — only scale edge rows
            let mut de_btx = vec![F::zero(); dim_star];
            for i in 0..num_vertices {
                de_btx[i] = btx[i]; // vertex rows pass through
            }
            for (idx, i) in (num_vertices..dim_star).enumerate() {
                de_btx[i] = btx[i] * de_inv[idx];
            }

            // B · (De⁻¹ Bᵀ x) → back to full star space
            let y = incidence.spmv(&de_btx);

            // Extract vertex rows, apply Dv⁻¹/²
            for i in 0..num_vertices {
                out[i * self.dim_out + f] = y[i] * dv_inv_sqrt[i];
            }
        }

        relu(&mut out);
        out
    }
}

// ============================================================
// Level 3: Signed-HGNN (novel)
// H' = σ(Dv⁻¹/² B₊W₊B₊ᵀDv⁻¹/² H  +  Dv⁻¹/² B₋W₋B₋ᵀDv⁻¹/² H)
// ============================================================

pub struct SignedHgnnConv<F: Real> {
    pub weights_pos: Vec<F>,
    pub weights_neg: Vec<F>,
    pub dim_in: usize,
    pub dim_out: usize,
}

impl<F: Real> HypergraphConv<F> for SignedHgnnConv<F> {
    fn dim_out(&self) -> usize { self.dim_out }

    fn convolve(
        &self,
        incidence: &TensorCsr<F>,
        features: &[F],
        num_vertices: usize,
        dim_in: usize,
    ) -> Vec<F> {
        let dim_star = incidence.num_rows;

        // Split by sign
        let (b_pos, b_neg) = split_by_sign(incidence);
        let bt_pos = transpose_csr(&b_pos);
        let bt_neg = transpose_csr(&b_neg);

        // HW₊, HW₋
        let hw_pos = dense_matmul(features, &self.weights_pos,
                                  num_vertices, dim_in, self.dim_out);
        let hw_neg = dense_matmul(features, &self.weights_neg,
                                  num_vertices, dim_in, self.dim_out);

        // Combined vertex degree for normalization
        let dv = row_degrees_csr(incidence);
        let dv_inv_sqrt = inv_sqrt_vec(&dv[..num_vertices]);

        // Edge degrees per channel
        let de_pos = row_degrees_csr(&b_pos);
        let de_neg = row_degrees_csr(&b_neg);

        let de_pos_inv: Vec<F> = (num_vertices..dim_star).map(|i| {
            let d = if i < de_pos.len() { de_pos[i] } else { F::zero() };
            if d > F::zero() { F::one() / d } else { F::zero() }
        }).collect();
        let de_neg_inv: Vec<F> = (num_vertices..dim_star).map(|i| {
            let d = if i < de_neg.len() { de_neg[i] } else { F::zero() };
            if d > F::zero() { F::one() / d } else { F::zero() }
        }).collect();

        let mut out = vec![F::zero(); num_vertices * self.dim_out];

        for f in 0..self.dim_out {
            // ── Positive channel ──
            let mut x_pos = vec![F::zero(); dim_star];
            for i in 0..num_vertices {
                x_pos[i] = hw_pos[i * self.dim_out + f] * dv_inv_sqrt[i];
            }
            let btx_pos = bt_pos.spmv(&x_pos);
            let mut de_btx_pos = vec![F::zero(); dim_star];
            for i in 0..num_vertices { de_btx_pos[i] = btx_pos[i]; }
            for (idx, i) in (num_vertices..dim_star).enumerate() {
                de_btx_pos[i] = btx_pos[i] * de_pos_inv[idx];
            }
            let y_pos = b_pos.spmv(&de_btx_pos);

            // ── Negative channel ──
            let mut x_neg = vec![F::zero(); dim_star];
            for i in 0..num_vertices {
                x_neg[i] = hw_neg[i * self.dim_out + f] * dv_inv_sqrt[i];
            }
            let btx_neg = bt_neg.spmv(&x_neg);
            let mut de_btx_neg = vec![F::zero(); dim_star];
            for i in 0..num_vertices { de_btx_neg[i] = btx_neg[i]; }
            for (idx, i) in (num_vertices..dim_star).enumerate() {
                de_btx_neg[i] = btx_neg[i] * de_neg_inv[idx];
            }
            let y_neg = b_neg.spmv(&de_btx_neg);

            // ── Sum channels ──
            for i in 0..num_vertices {
                out[i * self.dim_out + f] =
                    (y_pos[i] + y_neg[i]) * dv_inv_sqrt[i];
            }
        }

        relu(&mut out);
        out
    }
}