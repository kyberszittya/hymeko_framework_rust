use crate::tensor::common::Real;
use crate::tensor::representations::tensor_csr::{TensorCsr, TensorCsrBuilder};

impl<F: Real> TensorCsrBuilder<F> {
    /// O(N log N) time complexity due to sorting the column indices within each row.
    pub fn finalize_coalesced(self) -> TensorCsr<F> {
        let mut new_row_ptr = Vec::with_capacity(self.dim_i + 1);
        let mut new_col_ind = Vec::with_capacity(self.unfinalized_col_ind.len());
        let mut new_val = Vec::with_capacity(self.unfinalized_val.len());

        // We can use a scratch buffer to hold the (col, val) pairs for each row before sorting and coalescing.
        let mut scratch_buffer: Vec<(usize, F)> = Vec::new();

        new_row_ptr.push(0);
        for i in 0..self.dim_i {
            let start = self.unfinalized_row_ptr[i];
            let end = self.unfinalized_row_ptr[i + 1];
            if start == end {
            new_row_ptr.push(new_col_ind.len());
                continue;
            }
            // Clear the scratch buffer and fill it with the (col, val) pairs for the current row.
            scratch_buffer.clear();
            for idx in start..end {
                let col = self.unfinalized_col_ind[idx];
                let val = self.unfinalized_val[idx];
                scratch_buffer.push((col, val));
            }
            // Sort the scratch buffer by column index to bring duplicates together.
            scratch_buffer.sort_by_key(|&(col, _)| col);
            // After sorting, we can coalesce duplicates by iterating through the sorted entries.
            // Coalesce
            let mut current_col = scratch_buffer[0].0;
            let mut current_val = scratch_buffer[0].1;
            for &(col, val) in &scratch_buffer[1..] {
                if col == current_col {
                    current_val += val; // Coalesce by summing values of duplicate columns
                } else {
                    new_col_ind.push(current_col);
                    new_val.push(current_val);
                    current_col = col;
                    current_val = val;
                }
            }
            new_col_ind.push(current_col);
            new_val.push(current_val);
            new_row_ptr.push(new_col_ind.len());
        }
        new_col_ind.shrink_to_fit();
        new_val.shrink_to_fit();

        TensorCsr {
            num_rows: self.dim_i,
            num_cols: self.dim_j,
            row_ptr: new_row_ptr,
            col_ind: new_col_ind,
            val: new_val,
        }

    }
}