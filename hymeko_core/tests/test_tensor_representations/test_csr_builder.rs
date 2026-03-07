#[cfg(test)]
mod test_csr_builder {
    use hymeko::tensor::representations::tensor_csr::TensorCsrBuilder;
    use crate::test_helpers::{log_test_footer, log_test_header};
    use log::info;
    use std::time::Instant;

    struct CsrBuilderCase {
        dim_i: usize,
        dim_j: usize,
        rows: &'static [usize],
        cols: &'static [usize],
        vals: &'static [f64],
        expected_row_ptr: &'static [usize],
        expected_col_ind: &'static [usize],
        expected_vals: &'static [f64],
    }

    const COALESCED_DUP_CASE: CsrBuilderCase = CsrBuilderCase {
        dim_i: 3,
        dim_j: 4,
        rows: &[0, 3, 3, 7],
        cols: &[2, 0, 2, 3, 1, 1, 3],
        vals: &[1.0, 2.0, 1.5, 0.5, 1.0, 1.0, 0.5],
        expected_row_ptr: &[0, 2, 2, 4],
        expected_col_ind: &[0, 2, 1, 3],
        expected_vals: &[2.0, 2.5, 2.0, 1.0],
    };

    const SIMPLE_CHAIN_CASE: CsrBuilderCase = CsrBuilderCase {
        dim_i: 2,
        dim_j: 2,
        rows: &[0, 1, 2],
        cols: &[0, 1],
        vals: &[5.0, 7.0],
        expected_row_ptr: &[0, 1, 2],
        expected_col_ind: &[0, 1],
        expected_vals: &[5.0, 7.0],
    };

    const TEST_CASES: &[&CsrBuilderCase] = &[&COALESCED_DUP_CASE, &SIMPLE_CHAIN_CASE];

    #[test]
    fn test_tensor_csr_builder_finalize_coalesced() {
        log_test_header(
            "test_tensor_csr_builder_finalize_coalesced",
            "Coalesces COO data into CSR form across multiple fixtures.",
        );
        let start = Instant::now();
        for (idx, case) in TEST_CASES.iter().enumerate() {
            run_case(case, idx);
        }
        log_test_footer(
            "test_tensor_csr_builder_finalize_coalesced",
            Some(start.elapsed()),
            "All CSR builder cases produced the expected row/col/value buffers.",
        );
    }

    fn run_case(case: &CsrBuilderCase, idx: usize) {
        let builder = TensorCsrBuilder::<f64> {
            dim_i: case.dim_i,
            dim_j: case.dim_j,
            rows: case.rows.to_vec(),
            cols: case.cols.to_vec(),
            vals: case.vals.to_vec(),
        };
        let csr = builder.finalize_coalesced();

        assert_eq!(csr.num_rows, case.dim_i, "Expected {} rows, got {}", case.dim_i, csr.num_rows);
        assert_eq!(csr.num_cols, case.dim_j, "Expected {} columns, got {}", case.dim_j, csr.num_cols);
        assert_eq!(csr.row_ptr, case.expected_row_ptr, "Unexpected row_ptr: {:?}", csr.row_ptr);
        assert_eq!(csr.col_ind, case.expected_col_ind, "Unexpected col_ind: {:?}", csr.col_ind);
        assert_eq!(csr.val, case.expected_vals, "Unexpected val: {:?}", csr.val);
        info!(
            "CSR case {} validated (dim {}x{}, nnz={})",
            idx,
            csr.num_rows,
            csr.num_cols,
            csr.val.len()
        );
    }
}