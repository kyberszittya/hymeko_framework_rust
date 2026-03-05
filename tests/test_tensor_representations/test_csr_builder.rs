#[cfg(test)]
mod test_csr_builder {
    use hymeko::tensor::representations::tensor_csr::TensorCsrBuilder;

    #[test]
    fn test_tensor_csr_builder_finalize_coalesced() {
        let builder = TensorCsrBuilder::<f64> {
            dim_i: 3,
            dim_j: 4,
            rows: vec![0, 3, 3, 7],
            cols: vec![
                // 0th node arcs
                2, 0, 2,
                // 1st node arcs (none)
                // 2nd node arcs (with duplicates)
                3, 1, 1, 3
            ],
            vals: vec![
                // 0th node arc weights
                1.0, 2.0, 1.5,
                // 1st node arc weights (none)
                // 2nd node arc weights (with duplicates)
                0.5, 1.0, 1.0, 0.5
            ],
        };
        let csr = builder.finalize_coalesced();
        assert_eq!(csr.num_rows, 3, "Expected 3 rows, got {}", csr.num_rows);
        assert_eq!(csr.num_cols, 4, "Expected 4 columns, got {}", csr.num_cols);
        assert_eq!(csr.row_ptr, vec![0, 2, 2, 4], "Unexpected row_ptr: {:?}", csr.row_ptr);
        assert_eq!(csr.col_ind, vec![0, 2, 1, 3], "Unexpected col_ind: {:?}", csr.col_ind);
        assert_eq!(csr.val, vec![2.0, 2.5, 2.0, 1.0], "Unexpected val: {:?}", csr.val);


    }
}