use crate::tensor::common::Real;

#[derive(Debug, Clone)]
pub struct TensorCsr<F> {
    pub num_rows: usize,
    pub num_cols: usize,
    pub row_ptr: Vec<usize>, // Length: num_rows + 1
    pub col_ind: Vec<usize>, // Length: nnz (number of non-zeros)
    pub val: Vec<F>,         // Length: nnz
}

impl<F: Real> TensorCsr<F> {
    /// Pre-allocates the exact memory required for the CSR structure.
    pub fn with_capacity(num_rows: usize, num_cols: usize, nnz: usize) -> Self {
        let mut row_ptr = vec![0; num_rows + 1];
        Self {
            num_rows,
            num_cols,
            row_ptr,
            col_ind: vec![0; nnz],
            val: vec![F::zero(); nnz],
        }
    }
}


// ==========================================
// UTILITY: Universal Prefix Sum
// ==========================================
#[inline]
pub fn build_row_ptr(counts: &[usize]) -> (Vec<usize>, usize) {
    let dim = counts.len();
    let mut row_ptr = vec![0; dim + 1];
    let mut total = 0;
    for i in 0..dim {
        row_ptr[i] = total;
        total += counts[i];
    }
    row_ptr[dim] = total;
    (row_ptr, total)
}