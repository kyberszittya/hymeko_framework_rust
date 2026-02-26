#[cfg(test)]
mod minimal_tensor_representations {
    use hymeko_framework::tensor::aggregation::{AggCfg, SignAgg, WeightAgg};
    use hymeko_framework::tensor::tensor::{clique_expansion_coo, compute_bipartite_degrees, project_sum_over_slices, star_expansion_coo, star_expansion_coo_normalized};
    use hymeko_framework::tensor::tensor_val::{EdgeWScalar, ScalarWeightExtractor};
    use hymeko_framework::tensor::util::print_dense_block;
    use hymeko_framework::traversal::hypergraphview::HyperGraphView;
    use crate::test_helpers::load_and_lower;

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool { (a - b).abs() <= eps }

    #[test]
    fn tiny_star_2nodes_1edge_directions_and_values() {
        let (_store, compiled) =
            load_and_lower("./data/minimal_examples/testing_edges/minimal_test_tensor_values_2nodes_1_edge.hymeko").unwrap();

        let aggcfg = AggCfg { weight: WeightAgg::Sum, sign: SignAgg::PreferNonNeutral, clamp01: false };
        let ex = ScalarWeightExtractor::default();
        let hg = HyperGraphView::<f32, EdgeWScalar<f32>, f32>::from_ir(&compiled.ir, &aggcfg, &ex);

        let coo = star_expansion_coo(&hg);
        let a = project_sum_over_slices(&coo);

        let n = hg.num_nodes();
        let m = hg.num_edges();
        assert_eq!(m, 1, "expected exactly 1 edge in this tiny graph");

        let e_idx = n + 0;
        let eps = 1e-5_f32;

        // Keressük meg a két várt nemnullát úgy, hogy ne kelljen tudni a node indexeket.
        // Várt: A[e_idx, v_minus] = 0.85 és A[v_plus, e_idx] = 0.9
        let mut found_minus: Option<(usize, f32)> = None; // (v, val)
        let mut found_plus: Option<(usize, f32)> = None;  // (v, val)

        for v in 0..n {
            let em = a[e_idx][v];    // edge -> node
            let pe = a[v][e_idx];    // node -> edge

            if em.abs() > eps { found_minus = Some((v, em)); }
            if pe.abs() > eps { found_plus = Some((v, pe)); }
        }

        let (v_minus, val_minus) = found_minus.expect("expected one edge->node nonzero (minus incidence)");
        let (v_plus,  val_plus)  = found_plus.expect("expected one node->edge nonzero (plus incidence)");

        assert!(approx_eq(val_minus, 0.85, 1e-4), "edge->node value should be 0.85, got {}", val_minus);
        assert!(approx_eq(val_plus,  0.9,  1e-4), "node->edge value should be 0.9, got {}", val_plus);

        // Biztosítsuk, hogy nincs véletlen extra nemnulla a node-edge blokkon kívül
        // (szigorú, de tiny példán jó)
        for i in 0..(n + m) {
            for j in 0..(n + m) {
                let v = a[i][j];
                if v.abs() <= eps { continue; }

                let ok =
                    (i == e_idx && j == v_minus) ||
                        (i == v_plus && j == e_idx);

                assert!(ok, "unexpected nonzero at ({},{}) = {}", i, j, v);
            }
        }
        let dim = hg.num_nodes() + hg.num_edges();
        print_dense_block::<f32>(&coo, 0, 0, 0, dim, dim);
    }

    #[test]
    fn tiny_star_2nodes_1edge_normalized_matches_formula() {
        let (_store, compiled) =
            load_and_lower("./data/minimal_examples/testing_edges/minimal_test_tensor_values_2nodes_1_edge.hymeko").unwrap();

        let aggcfg = AggCfg { weight: WeightAgg::Sum, sign: SignAgg::PreferNonNeutral, clamp01: false };
        let ex = ScalarWeightExtractor::default();
        let hg = HyperGraphView::<f32, EdgeWScalar<f32>, f32>::from_ir(&compiled.ir, &aggcfg, &ex);

        let (deg_v, deg_e) = compute_bipartite_degrees(&hg, true);

        let coo = star_expansion_coo_normalized(&hg, true, 1e-12_f32);
        let a = project_sum_over_slices(&coo);

        let n = hg.num_nodes();
        let e_idx = n + 0;

        // megtaláljuk ugyanúgy a két nemnullát
        let eps = 1e-5_f32;
        let mut v_minus = None;
        let mut v_plus = None;

        for v in 0..n {
            if a[e_idx][v].abs() > eps { v_minus = Some(v); }
            if a[v][e_idx].abs() > eps { v_plus = Some(v); }
        }
        let v_minus = v_minus.unwrap();
        let v_plus = v_plus.unwrap();

        // elvárt normalizált értékek
        let de = deg_e[0].max(1e-12);
        let expected_minus = 0.85 / (deg_v[v_minus] * de).sqrt();
        let expected_plus  = 0.9  / (deg_v[v_plus]  * de).sqrt();

        assert!((a[e_idx][v_minus] - expected_minus).abs() < 1e-4);
        assert!((a[v_plus][e_idx]  - expected_plus ).abs() < 1e-4);
        let dim = hg.num_nodes() + hg.num_edges();
        print_dense_block::<f32>(&coo, 0, 0, 0, dim, dim);

    }

    #[test]
    fn tiny_clique_2nodes_1edge_expected_direction_and_value() {
        let (_store, compiled) =
            load_and_lower("./data/minimal_examples/testing_edges/minimal_test_tensor_values_2nodes_1_edge.hymeko").unwrap();

        let aggcfg = AggCfg { weight: WeightAgg::Sum, sign: SignAgg::PreferNonNeutral, clamp01: false };
        let ex = ScalarWeightExtractor::default();
        let hg = HyperGraphView::<f32, EdgeWScalar<f32>, f32>::from_ir(&compiled.ir, &aggcfg, &ex);

        let clique = clique_expansion_coo(&hg);
        assert_eq!(clique.num_slices, hg.num_edges(), "clique COO slices should match num_edges");

        let n = hg.num_nodes();
        assert_eq!(n, 3, "this tiny graph should have exactly 2 nodes");

        // Debug print (run with -- --nocapture)
        print_dense_block(&clique, 0, 0, 0, n, n);

        let a = project_sum_over_slices(&clique);
        let eps = 1e-5_f32;

        // Find the single nonzero off-diagonal (robust to node index order)
        // Expected: exactly one offdiag ~ 1.53, the other offdiag ~ 0, diag ~0
        assert!((a[0][0]).abs() <= eps, "diag(0,0) should be 0, got {}", a[0][0]);
        assert!((a[1][1]).abs() <= eps, "diag(1,1) should be 0, got {}", a[1][1]);

        let off01 = a[0][1];
        let off10 = a[1][0];

        let expected = 2.0_f32 * 0.85_f32 * 0.9_f32; // 1.53

        // Exactly one direction should carry the weight
        let cond_a = (off01 - expected).abs() <= 1e-4 && off10.abs() <= 1e-4;
        let cond_b = (off10 - expected).abs() <= 1e-4 && off01.abs() <= 1e-4;

        // root row/col is zero
        for i in 0..n {
            assert!(a[0][i].abs() < eps);
            assert!(a[i][0].abs() < eps);
        }

        // count nonzeros off-diagonal (excluding root)
        let mut nz = vec![];
        for i in 1..n {
            for j in 1..n {
                if i == j { continue; }
                if a[i][j].abs() > eps {
                    nz.push((i,j,a[i][j]));
                }
            }
        }
        assert_eq!(nz.len(), 1);
        assert!((nz[0].2 - 1.53).abs() < 1e-4);

    }
}