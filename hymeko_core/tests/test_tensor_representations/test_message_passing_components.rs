mod test_message_passing_components
{
    use hymeko::tensor::message_passing::{
        clique_diag, gather_edges_from_nodes,
        implicit_clique_step, remove_self_effect, scatter_nodes_from_edges, CliqueStepCfg};
    use hymeko::traversal::hypergraphview::HyperGraphView;
    use hymeko::tensor::common_traversal::inc_scalar_signed;
    use hymeko::tensor::tensor_val::{EdgeWScalar, ScalarWeightExtractor};
    use crate::test_helpers::load_and_lower;
    use crate::test_tensor_representations::constants::*;



    fn assert_vec_close(a: &[f32], b: &[f32], eps: f32, msg: &str) {
        assert_eq!(a.len(), b.len(), "len mismatch: {}", msg);
        for i in 0..a.len() {
            let d = (a[i] - b[i]).abs();
            assert!(d <= eps, "{} at i={} a={} b={} diff={}", msg, i, a[i], b[i], d);
        }
    }

    #[test]
    fn test_gather_matches_manual_btx() {
        let (_store, compiled) = load_and_lower(LINEAR_EDGE_VALUES_PATH).unwrap();

        let aggcfg = DEFAULT_AGG_CFG;
        let ex = ScalarWeightExtractor::default();
        let hg = HyperGraphView::<f32, EdgeWScalar<f32>, f32>::from_ir(&compiled.ir, &aggcfg, &ex);

        let n = hg.num_nodes();
        let m = hg.num_edges();
        let use_abs = true;

        // deterministic x
        let mut x = vec![0.0f32; n];
        for i in 0..n { x[i] = 1.0 + (i as f32) * 0.1; }

        let mut x_edges = vec![0.0f32; m];
        gather_edges_from_nodes(&hg, &x, &mut x_edges, use_abs);

        // manual: x_e = Σ_v b_{v,e} x_v
        let mut x_edges_ref = vec![0.0f32; m];
        for e in 0..m {
            let s = hg.edge_offsets[e];
            let eend = hg.edge_offsets[e + 1];
            let mut acc = 0.0f32;
            for p in s..eend {
                let v = hg.flat_edge_nodes[p].0;
                let b = inc_scalar_signed(&hg, p, e, use_abs);
                acc += b * x[v];
            }
            x_edges_ref[e] = acc;
        }

        assert_vec_close(&x_edges, &x_edges_ref, EPS_F32_DEFAULT, "gather_edges_from_nodes mismatch");
    }

    #[test]
    fn test_scatter_matches_manual_bxe() {
        let (_store, compiled) = load_and_lower(LINEAR_EDGE_VALUES_PATH).unwrap();

        let aggcfg = DEFAULT_AGG_CFG;
        let ex = ScalarWeightExtractor::default();
        let hg = HyperGraphView::<f32, EdgeWScalar<f32>, f32>::from_ir(&compiled.ir, &aggcfg, &ex);

        let n = hg.num_nodes();
        let m = hg.num_edges();
        let use_abs = true;

        // deterministic x_edges
        let mut x_edges = vec![0.0f32; m];
        for e in 0..m { x_edges[e] = 0.5 + (e as f32) * 0.2; }

        let mut y = vec![0.0f32; n];
        scatter_nodes_from_edges(&hg, &x_edges, &mut y, use_abs);

        // manual: y_v = Σ_e b_{v,e} x_e
        let mut y_ref = vec![0.0f32; n];
        for e in 0..m {
            let s = hg.edge_offsets[e];
            let eend = hg.edge_offsets[e + 1];
            let xe = x_edges[e];
            for p in s..eend {
                let v = hg.flat_edge_nodes[p].0;
                let b = inc_scalar_signed(&hg, p, e, use_abs);
                y_ref[v] += b * xe;
            }
        }

        assert_vec_close(&y, &y_ref, EPS_F32_DEFAULT, "scatter_nodes_from_edges mismatch");
    }

    #[test]
    fn test_diag_matches_sum_of_squares() {
        let (_store, compiled) = load_and_lower(LINEAR_EDGE_VALUES_PATH).unwrap();

        let aggcfg = DEFAULT_AGG_CFG;
        let ex = ScalarWeightExtractor::default();
        let hg = HyperGraphView::<f32, EdgeWScalar<f32>, f32>::from_ir(&compiled.ir, &aggcfg, &ex);

        let n = hg.num_nodes();
        let m = hg.num_edges();
        let use_abs = true;

        let diag = clique_diag(&hg, use_abs);

        let mut ref_diag = vec![0.0f32; n];
        for e in 0..m {
            let s = hg.edge_offsets[e];
            let eend = hg.edge_offsets[e + 1];
            for p in s..eend {
                let v = hg.flat_edge_nodes[p].0;
                let b = inc_scalar_signed(&hg, p, e, use_abs);
                ref_diag[v] += b * b;
            }
        }

        assert_vec_close(&diag, &ref_diag, EPS_F32_DEFAULT, "clique_diag mismatch");
    }

    #[test]
    fn test_remove_self_effect_matches_definition() {
        // tiny synthetic vector test (no HG needed)
        let x = vec![1.0f32, 2.0, 3.0];
        let diag = vec![10.0f32, 0.5, 2.0];
        let mut y = vec![100.0f32, 100.0, 100.0];

        remove_self_effect(&mut y, &diag, &x);

        let expected = vec![
            100.0 - 10.0 * 1.0,
            100.0 - 0.5 * 2.0,
            100.0 - 2.0 * 3.0,
        ];
        assert_vec_close(&y, &expected, EPS_F32_ULTRA, "remove_self_effect mismatch");
    }

    #[test]
    fn test_implicit_clique_step_equals_pipeline() {
        let (_store, compiled) = load_and_lower(LINEAR_EDGE_VALUES_PATH).unwrap();

        let aggcfg = DEFAULT_AGG_CFG;
        let ex = ScalarWeightExtractor::default();
        let hg = HyperGraphView::<f32, EdgeWScalar<f32>, f32>::from_ir(&compiled.ir, &aggcfg, &ex);

        let n = hg.num_nodes();
        let m = hg.num_edges();
        let mut x = vec![0.0f32; n];
        for i in 0..n { x[i] = 1.0 + (i as f32) * 0.1; }

        let cfg = CliqueStepCfg { use_abs: true, include_self: false };

        // orchestrator
        let diag = clique_diag(&hg, cfg.use_abs);
        let mut y1 = vec![0.0f32; n];
        let mut buffer_edges = vec![0.0f32; m];
        implicit_clique_step(&hg, &x, &mut y1, &mut buffer_edges, Some(&diag), cfg);

        // explicit pipeline
        let mut x_edges = vec![0.0f32; m];
        gather_edges_from_nodes(&hg, &x, &mut x_edges, cfg.use_abs);
        let mut y2 = vec![0.0f32; n];
        scatter_nodes_from_edges(&hg, &x_edges, &mut y2, cfg.use_abs);
        let diag = clique_diag(&hg, cfg.use_abs);
        remove_self_effect(&mut y2, &diag, &x);

        assert_vec_close(&y1, &y2, EPS_F32_DEFAULT, "implicit_clique_step != pipeline");
    }
}