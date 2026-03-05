use crate::tensor::common::Real;
use crate::tensor::representations::tensor_csr::{TensorCsr, TensorCsrBuilder};

impl<F: Real> TensorCsrBuilder<F> {
    /// O(N log N) time complexity due to sorting the column indices within each row.
    pub fn finalize_coalesced(self) -> TensorCsr<F> {
        if self.vals.is_empty() {
            return TensorCsr {
                num_rows: self.dim_i,
                num_cols: self.dim_j,
                row_ptr: vec![0; self.dim_i + 1],
                col_ind: Vec::new(),
                val: Vec::new(),
            };
        }
        // 1. Összefűzzük az elemeket (sor, oszlop, érték)
        let mut entries: Vec<(usize, usize, F)> = self.rows.into_iter()
            .zip(self.cols.into_iter())
            .zip(self.vals.into_iter())
            .map(|((r, c), v)| (r, c, v))
            .collect();

        // 2. Globális rendezés: Először sor szerint, azon belül oszlop szerint
        entries.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

        let mut new_row_ptr = vec![0; self.dim_i + 1];
        let mut new_col_ind = Vec::with_capacity(entries.len());
        let mut new_val = Vec::with_capacity(entries.len());

        // 3. Coalescing (duplikátumok összevonása) és CSR építés
        let mut current_row = entries[0].0;
        let mut current_col = entries[0].1;
        let mut current_val = entries[0].2;

        for &(r, c, v) in &entries[1..] {
            if r == current_row && c == current_col {
                current_val += v; // Duplikátumok összevonása
            } else {
                new_col_ind.push(current_col);
                new_val.push(current_val);
                new_row_ptr[current_row + 1] += 1;

                current_row = r;
                current_col = c;
                current_val = v;
            }
        }
        new_col_ind.push(current_col);
        new_val.push(current_val);
        new_row_ptr[current_row + 1] += 1;

        // 4. Prefix sum a row_ptr-en (A valódi CSR sormutatók generálása)
        for i in 0..self.dim_i {
            new_row_ptr[i + 1] += new_row_ptr[i];
        }

        TensorCsr {
            num_rows: self.dim_i,
            num_cols: self.dim_j,
            row_ptr: new_row_ptr,
            col_ind: new_col_ind,
            val: new_val,
        }

    }
}