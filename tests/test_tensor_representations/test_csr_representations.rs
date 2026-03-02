#[cfg(test)]
mod test_csr_representations {
    use hymeko::tensor::aggregation::{AggCfg, SignAgg, WeightAgg};
    use hymeko::tensor::tensor::project_sum_over_slices;
    use hymeko::tensor::representations::tensor_csr::TensorCsr;
    use hymeko::tensor::representations::tensor_csr_representations::{star_expansion_csr, clique_expansion_csr};
    use hymeko::tensor::tensor_val::{EdgeWScalar, ScalarWeightExtractor};
    use hymeko::traversal::hypergraphview::HyperGraphView;
    use hymeko::tensor::common::Real;
    use hymeko::tensor::representations::tensor_coo_representation::star_expansion_coo;
    use crate::test_helpers::load_and_lower;

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool { (a - b).abs() <= eps }

    /// Helper utility to project a 2D CSR tensor into a dense matrix for easy comparison.
    fn project_csr_to_dense<F: Real>(csr: &TensorCsr<F>) -> Vec<Vec<F>> {
        let mut dense = vec![vec![F::zero(); csr.num_cols]; csr.num_rows];
        for row in 0..csr.num_rows {
            let start = csr.row_ptr[row];
            let end = csr.row_ptr[row + 1];
            for idx in start..end {
                let col = csr.col_ind[idx];
                let val = csr.val[idx];
                dense[row][col] += val; // Coalesce just in case
            }
        }
        dense
    }

    #[test]
    fn csr_star_expansion_matches_coo_minimal_graph() {
        let (_store, compiled) =
            load_and_lower("./data/minimal_examples/testing_edges/minimal_test_tensor_values_2nodes_1_edge.hymeko").unwrap();

        let aggcfg = AggCfg { weight: WeightAgg::Sum, sign: SignAgg::PreferNonNeutral, clamp01: false };
        let ex = ScalarWeightExtractor::default();
        let hg = HyperGraphView::<f32, EdgeWScalar<f32>, f32>::from_ir(&compiled.ir, &aggcfg, &ex);

        // Generate both tensor formats
        let coo = star_expansion_coo(&hg);
        let csr = star_expansion_csr(&hg);

        let dim = hg.num_nodes() + hg.num_edges();
        assert_eq!(csr.num_rows, dim, "CSR row dimension mismatch");
        assert_eq!(csr.num_cols, dim, "CSR col dimension mismatch");
        assert_eq!(csr.row_ptr.len(), dim + 1, "row_ptr array size must be dim + 1");

        // Project both to dense matrices and assert strict equality
        let a_coo = project_sum_over_slices(&coo);
        let a_csr = project_csr_to_dense(&csr);

        let eps = 1e-5_f32;
        for i in 0..dim {
            for j in 0..dim {
                assert!(
                    approx_eq(a_coo[i][j], a_csr[i][j], eps),
                    "Mismatch at ({},{}): COO={}, CSR={}", i, j, a_coo[i][j], a_csr[i][j]
                );
            }
        }
    }

    #[test]
    fn csr_clique_expansion_correctness() {
        let (_store, compiled) =
            load_and_lower("./data/minimal_examples/testing_edges/minimal_test_tensor_values_2nodes_1_edge.hymeko").unwrap();

        let aggcfg = AggCfg { weight: WeightAgg::Sum, sign: SignAgg::PreferNonNeutral, clamp01: false };
        let ex = ScalarWeightExtractor::default();
        let hg = HyperGraphView::<f32, EdgeWScalar<f32>, f32>::from_ir(&compiled.ir, &aggcfg, &ex);

        let csr = clique_expansion_csr(&hg);
        let n = hg.num_nodes();

        assert_eq!(csr.num_rows, n, "Clique CSR should be |V| x |V|");
        assert_eq!(csr.num_cols, n, "Clique CSR should be |V| x |V|");

        let a_csr = project_csr_to_dense(&csr);
        let eps = 1e-5_f32;

        let mut nz = vec![];
        for i in 0..n {
            for j in 0..n {
                // Ensure the diagonal remains cleanly zeroed
                if i == j {
                    assert!(approx_eq(a_csr[i][j], 0.0, eps), "Diagonal must be 0 at ({},{})", i, j);
                    continue;
                }

                // Track any off-diagonal values
                if a_csr[i][j].abs() > eps {
                    nz.push((i, j, a_csr[i][j]));
                }
            }
        }

        // We expect exactly ONE directed connection between the two incident nodes
        assert_eq!(
            nz.len(), 1,
            "Clique projection failed to maintain strict directionality: found {} non-zeros instead of exactly 1",
            nz.len()
        );
    }

    #[test]
    fn csr_star_expansion_scales_to_fano_graph() {
        let (_store, compiled) =
            load_and_lower("./data/typical_graphs/fano_graph.hymeko").unwrap();

        let aggcfg = AggCfg { weight: WeightAgg::Sum, sign: SignAgg::PreferNonNeutral, clamp01: false };
        let ex = ScalarWeightExtractor::default();
        let hg = HyperGraphView::<f32, EdgeWScalar<f32>, f32>::from_ir(&compiled.ir, &aggcfg, &ex);

        let coo = star_expansion_coo(&hg);
        let csr = star_expansion_csr(&hg);

        let a_coo = project_sum_over_slices(&coo);
        let a_csr = project_csr_to_dense(&csr);

        let dim = hg.num_nodes() + hg.num_edges();
        let eps = 1e-6_f32;

        let mut non_zero_count = 0;
        for i in 0..dim {
            for j in 0..dim {
                let v_coo = a_coo[i][j];
                let v_csr = a_csr[i][j];

                assert!(
                    approx_eq(v_coo, v_csr, eps),
                    "Fano mismatch at ({},{}): COO={}, CSR={}", i, j, v_coo, v_csr
                );

                if v_csr.abs() > eps {
                    non_zero_count += 1;
                }
            }
        }

        // Ensure we actually mapped the topology and didn't just compare two empty matrices
        assert!(non_zero_count > 0, "Fano matrix projection resulted in an entirely zero matrix");
        assert_eq!(csr.val.len(), non_zero_count, "CSR nnz count differs from actual projected non-zeros");
    }
}