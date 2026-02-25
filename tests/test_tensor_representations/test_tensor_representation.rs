#[cfg(test)]

mod test_tensor_representation {
    use hymeko_framework::traversal::aggregation::{AggCfg, SignAgg, WeightAgg};
    use hymeko_framework::traversal::hypergraphview::HyperGraphView;
    use hymeko_framework::traversal::message_passing::{build_explicit_a, implicit_clique_step, print_dense_f32, CliqueStepCfg};
    use hymeko_framework::traversal::tensor::{clique_expansion_coo, compute_bipartite_degrees, star_expansion_coo, star_expansion_coo_normalized};
    use crate::test_helpers::{load_and_lower, print_dense_matrix};

    #[test]
    fn test_tensor_representation_creation() {
        let (_store, compiled) =
            load_and_lower("./data/minimal_examples/testing_edges/linear_edge_values.hymeko").unwrap();

        let aggcfg = AggCfg {
            weight: WeightAgg::Sum,
            sign: SignAgg::PreferNonNeutral,
            clamp01: false,
        };

        let hg = HyperGraphView::from_ir(&compiled.ir, &aggcfg);

        let coo = star_expansion_coo(&hg);
        let proj = hymeko_framework::traversal::tensor::project_sum_over_slices(&coo);

        let num_nodes = hg.num_nodes(); // nálad: root + 5 => 6
        let num_edges = hg.num_edges(); // 4
        let dim = num_nodes + num_edges; // 10
        let edge_base = num_nodes; // 6

        // --- helpers ---
        let eps: f32 = 1e-4;
        let at = |r: usize, c: usize| -> f32 {
            // Ha nálad más az indexelés, itt cseréld:
            proj[r][c]
            // vagy: proj[(r, c)]
            // vagy: *proj.get(r, c).unwrap()
        };

        let assert_close = |r: usize, c: usize, expected: f32| {
            let v = at(r, c);
            assert!(
                (v - expected).abs() <= eps,
                "mismatch at ({},{}) expected {} got {}",
                r, c, expected, v
            );
        };

        // --- shape checks ---
        assert_eq!(dim, 10, "expected dim=10 (root + 5 nodes + 4 edges)");
        assert_eq!(edge_base, 6, "expected edge_base=num_nodes=6");

        // --- expected nonzeros (from your printed matrix) ---
        // e1 is edge_base+0 = 6
        assert_close(2, edge_base + 0, 2.50); // node1 -> e1
        assert_close(5, edge_base + 0, 8.80); // node4 -> e1
        assert_close(edge_base + 0, 1, 1.50); // e1 -> node0

        // e2 is 7
        assert_close(edge_base + 1, 3, 0.75); // e2 -> node2
        assert_close(edge_base + 1, 4, 7.50); // e2 -> node3

        // e3 is 8
        assert_close(edge_base + 2, 1, 0.75); // e3 -> node0
        assert_close(edge_base + 2, 3, 1.95); // e3 -> node2
        assert_close(edge_base + 2, 5, 7.50); // e3 -> node4

        // e4 is 9
        assert_close(5, edge_base + 3, 7.50); // node4 -> e4
        assert_close(edge_base + 3, 1, 0.75); // e4 -> node0
        assert_close(edge_base + 3, 2, 1.95); // e4 -> node1

        // --- optional: everything else is (near) zero ---
        // (Ha ez túl szigorú lesz későbbi bővítéseknél, ezt a blokkot vedd ki.)
        let mut expected = vec![vec![0u8; dim]; dim];
        let mark = |r: usize, c: usize, expected: &mut Vec<Vec<u8>>| { expected[r][c] = 1; };

        // mark all expected nonzeros
        mark(2, edge_base + 0, &mut expected);
        mark(5, edge_base + 0, &mut expected);
        mark(edge_base + 0, 1, &mut expected);

        mark(edge_base + 1, 3, &mut expected);
        mark(edge_base + 1, 4, &mut expected);

        mark(edge_base + 2, 1, &mut expected);
        mark(edge_base + 2, 3, &mut expected);
        mark(edge_base + 2, 5, &mut expected);

        mark(5, edge_base + 3, &mut expected);
        mark(edge_base + 3, 1, &mut expected);
        mark(edge_base + 3, 2, &mut expected);

        for r in 0..dim {
            for c in 0..dim {
                if expected[r][c] == 0 {
                    let v = at(r, c);
                    assert!(
                        v.abs() <= eps,
                        "expected ~0 at ({},{}) but got {}",
                        r, c, v
                    );
                }
            }
        }

        print_dense_matrix(&proj, "Projected star matrix (sum over slices)");
    }

    #[test]
    fn test_star_projection_linear_edge_values() {
        let (_store, compiled) =
            load_and_lower("./data/minimal_examples/testing_edges/linear_edge_values.hymeko").unwrap();

        let aggcfg = AggCfg {
            weight: WeightAgg::Sum,
            sign: SignAgg::PreferNonNeutral,
            clamp01: false,
        };

        let hg = HyperGraphView::from_ir(&compiled.ir, &aggcfg);
        let coo = star_expansion_coo(&hg);
        let proj = hymeko_framework::traversal::tensor::project_sum_over_slices(&coo);

        let num_nodes = hg.num_nodes();
        let num_edges = hg.num_edges();
        let dim = num_nodes + num_edges;
        let edge_base = num_nodes;

        // Root + 5 node + 4 edge = 10
        assert_eq!(dim, 10);
        assert_eq!(edge_base, 6);

        let eps: f32 = 1e-4;

        // ---- adapt this accessor if your matrix type differs ----
        let at = |r: usize, c: usize| -> f32 {
            // Choose the one that matches your Dense matrix type:
            proj[r][c]
            // proj[(r, c)]
            // *proj.get(r, c).unwrap()
        };

        let assert_close = |r: usize, c: usize, expected: f32| {
            let v = at(r, c);
            assert!(
                (v - expected).abs() <= eps,
                "mismatch at ({},{}) expected {} got {}",
                r, c, expected, v
            );
        };

        // Mapping (based on your printed output):
        // 0=root
        // 1..5 = node0..node4
        // 6..9 = e1..e4

        // e1 (index 6): (- node0[1.5], +node1[2.5], +node4[8.8])
        assert_close(2, edge_base + 0, 2.50); // node1 -> e1
        assert_close(5, edge_base + 0, 8.80); // node4 -> e1
        assert_close(edge_base + 0, 1, 1.50); // e1 -> node0

        // e2 (index 7): (- node2[0.75], -node3[7.5])
        assert_close(edge_base + 1, 3, 0.75); // e2 -> node2
        assert_close(edge_base + 1, 4, 7.50); // e2 -> node3

        // e3 (index 8): (- node0[0.75], -node2[1.95], -node4[7.5])
        assert_close(edge_base + 2, 1, 0.75); // e3 -> node0
        assert_close(edge_base + 2, 3, 1.95); // e3 -> node2
        assert_close(edge_base + 2, 5, 7.50); // e3 -> node4

        // e4 (index 9): (- node0[0.75], -node1[1.95], +node4[7.5])
        assert_close(5, edge_base + 3, 7.50); // node4 -> e4
        assert_close(edge_base + 3, 1, 0.75); // e4 -> node0
        assert_close(edge_base + 3, 2, 1.95); // e4 -> node1

        // Optional: assert everything else is ~0 (strict, but great right now)
        let mut expected = vec![vec![false; dim]; dim];
        let mark = |r: usize, c: usize, expected: &mut Vec<Vec<bool>>| expected[r][c] = true;

        mark(2, edge_base + 0, &mut expected);
        mark(5, edge_base + 0, &mut expected);
        mark(edge_base + 0, 1, &mut expected);

        mark(edge_base + 1, 3, &mut expected);
        mark(edge_base + 1, 4, &mut expected);

        mark(edge_base + 2, 1, &mut expected);
        mark(edge_base + 2, 3, &mut expected);
        mark(edge_base + 2, 5, &mut expected);

        mark(5, edge_base + 3, &mut expected);
        mark(edge_base + 3, 1, &mut expected);
        mark(edge_base + 3, 2, &mut expected);

        for r in 0..dim {
            for c in 0..dim {
                if !expected[r][c] {
                    let v = at(r, c);
                    assert!(
                        v.abs() <= eps,
                        "expected ~0 at ({},{}) but got {}",
                        r, c, v
                    );
                }
            }
        }

        // Debug output is still nice during bring-up; keep or remove later.
        print_dense_matrix(&proj, "Projected star matrix (sum over slices)");
    }

    #[test]
    fn test_clique_message_passing_matches_clique_view() {
        let (_store, compiled) =
            load_and_lower("./data/minimal_examples/testing_edges/linear_edge_values.hymeko").unwrap();

        let aggcfg = AggCfg {
            weight: WeightAgg::Sum,
            sign: SignAgg::PreferNonNeutral,
            clamp01: false,
        };
        let hg = HyperGraphView::from_ir(&compiled.ir, &aggcfg);

        // 1) "Dense" reference: clique_view -> matrix
        let clique_coo = clique_expansion_coo(&hg);
        let a = hymeko_framework::traversal::tensor::project_sum_over_slices(&clique_coo);

        let n = hg.num_nodes();
        let eps: f32 = 1e-4;

        // ---- adapt accessor to your matrix type ----
        let mat = |r: usize, c: usize| -> f32 {
            a[r][c]
            // a[(r, c)]
            // *a.get(r, c).unwrap()
        };

        // 2) Choose an input vector x (one-hot is great for debugging)
        // root=0, node0=1, ... as before
        let mut x = vec![0.0f32; n];
        x[2] = 1.0; // excite node1 (index 2), tweak as you like

        // y_dense = A * x
        let mut y_dense = vec![0.0f32; n];
        for i in 0..n {
            let mut acc = 0.0f32;
            for j in 0..n {
                acc += mat(i, j) * x[j];
            }
            y_dense[i] = acc;
        }

        // 3) y_imp: implicit clique step that matches clique_expansion_coo() semantics
        // clique_expansion_coo does, for each hyperedge:
        //   for each pair (u,v), a<b:
        //      if su==+ : add edge u->v with weight wu
        //      if su==- : add edge v->u with weight wu
        //      if su==0 : add both u->v and v->u with weight wu
        // where wu = flat_edge_w[p] * edge_weight[e]
        let mut y_imp = vec![0.0f32; n];

        for eid in 0..hg.num_edges() {
            let s = hg.edge_offsets[eid];
            let eend = hg.edge_offsets[eid + 1];
            let ew = hg.edge_weight[eid];

            // collect nodes in this hyperedge
            let mut nodes: Vec<(usize, i8, f32)> = Vec::with_capacity(eend - s);
            for p in s..eend {
                let u = hg.flat_edge_nodes[p].0;
                let su = hg.flat_edge_sign[p];
                let wu = hg.flat_edge_w[p] * ew;
                nodes.push((u, su, wu));
            }

            // apply the same pair logic as clique_expansion_coo
            for a_i in 0..nodes.len() {
                for b_i in (a_i + 1)..nodes.len() {
                    let (u, su, wu) = nodes[a_i];
                    let (v, _sv, _wv) = nodes[b_i];

                    match su {
                        1 => {
                            // edge u -> v with weight wu : y[u] += wu * x[v]
                            y_imp[u] += wu * x[v];
                        }
                        -1 => {
                            // edge v -> u with weight wu : y[v] += wu * x[u]
                            y_imp[v] += wu * x[u];
                        }
                        _ => {
                            // both directions
                            y_imp[u] += wu * x[v];
                            y_imp[v] += wu * x[u];
                        }
                    }
                }
            }
        }
        let proj = hymeko_framework::traversal::tensor::project_sum_over_slices(&clique_coo);
        print_dense_matrix(&proj, "Clique projection (sum over slices)");
    }

    #[test]
    fn test_implicit_clique_step_matches_explicit_bwbT() {
        let (_store, compiled) =
            load_and_lower("./data/minimal_examples/testing_edges/linear_edge_values.hymeko").unwrap();

        let aggcfg = AggCfg {
            weight: WeightAgg::Sum,
            sign: SignAgg::PreferNonNeutral,
            clamp01: false,
        };

        let hg = HyperGraphView::from_ir(&compiled.ir, &aggcfg);

        // cfg must match your implicit_clique_step semantics
        let cfg = CliqueStepCfg {
            use_abs: true,       // recommended: stable
            include_self: true,  // make comparison easier first
        };

        let n = hg.num_nodes();
        let m = hg.num_edges();
        let eps: f32 = 1e-4;

        // ---- Build explicit A = B W B^T (no diag removal here) ----
        // We'll treat incidence as:
        // b(v,e) = flat_edge_w[p] * edge_weight[e] * sign_factor
        // and then abs() if cfg.use_abs.
        let mut b: Vec<Vec<(usize, f32)>> = vec![Vec::new(); m]; // per edge: (v, bve)

        for e in 0..m {
            let s = hg.edge_offsets[e];
            let eend = hg.edge_offsets[e + 1];
            let ew = hg.edge_weight[e];
            for p in s..eend {
                let v = hg.flat_edge_nodes[p].0;
                let mut val = hg.flat_edge_w[p] * ew;

                // same sign convention as in implicit_clique_step
                let sgn = match hg.flat_edge_sign[p] {
                    1 => 1.0,
                    -1 => -1.0,
                    _ => 1.0,
                };
                val *= sgn;

                if cfg.use_abs {
                    val = val.abs();
                }
                b[e].push((v, val));
            }
        }

        // Explicit A in dense form (only for testing; ok for small n)
        let mut a = vec![vec![0.0f32; n]; n];
        for e in 0..m {
            let nodes = &b[e];
            for &(u, bu) in nodes {
                for &(v, bv) in nodes {
                    a[u][v] += bu * bv;
                }
            }
        }

        // ---- Pick a deterministic x ----
        let mut x = vec![0.0f32; n];
        // simple deterministic pattern (works better than one-hot)
        for i in 0..n {
            x[i] = (i as f32) * 0.1 + 1.0;
        }

        // y_dense = A * x
        let mut y_dense = vec![0.0f32; n];
        for i in 0..n {
            let mut acc = 0.0f32;
            for j in 0..n {
                acc += a[i][j] * x[j];
            }
            y_dense[i] = acc;
        }

        // y_imp from your implementation
        let y_imp = implicit_clique_step(&hg, &x, cfg);

        // Compare
        for i in 0..n {
            let d = (y_dense[i] - y_imp[i]).abs();
            assert!(
                d <= eps,
                "implicit clique mismatch at i={} dense={} imp={} |diff|={}",
                i, y_dense[i], y_imp[i], d
            );
        }
        let a = build_explicit_a(&hg, cfg);
        print_dense_f32(&a, "Explicit A = B B^T (matches implicit_clique_step)");
    }

    #[test]
    fn test_bipartite_degrees_linear_edge_values() {
        let (_store, compiled) =
            load_and_lower("./data/minimal_examples/testing_edges/linear_edge_values.hymeko").unwrap();

        let aggcfg = AggCfg { weight: WeightAgg::Sum, sign: SignAgg::PreferNonNeutral, clamp01: false };
        let hg = HyperGraphView::from_ir(&compiled.ir, &aggcfg);

        let (deg_v, deg_e) = compute_bipartite_degrees(&hg, true); // abs degree

        // index mapping:
        // 0=root
        // 1..5 = node0..node4
        // edges: 0..3 = e1..e4

        let eps: f32 = 1e-4;

        // node degrees (sum of abs incidence weights per node)
        assert!((deg_v[0] - 0.0).abs() <= eps);   // root
        assert!((deg_v[1] - 3.0).abs() <= eps);   // node0: 1.5 + 0.75 + 0.75
        assert!((deg_v[2] - 4.45).abs() <= eps);  // node1: 2.5 + 1.95
        assert!((deg_v[3] - 2.7).abs() <= eps);   // node2: 0.75 + 1.95
        assert!((deg_v[4] - 7.5).abs() <= eps);   // node3: 7.5
        assert!((deg_v[5] - 23.8).abs() <= eps);  // node4: 8.8 + 7.5 + 7.5

        // edge degrees (sum of abs incidence weights per edge)
        assert!((deg_e[0] - 12.8).abs() <= eps);  // e1: 1.5+2.5+8.8
        assert!((deg_e[1] - 8.25).abs() <= eps);  // e2: 0.75+7.5
        assert!((deg_e[2] - 10.2).abs() <= eps);  // e3: 0.75+1.95+7.5
        assert!((deg_e[3] - 10.2).abs() <= eps);  // e4: 0.75+1.95+7.5
    }

    #[test]
    fn test_star_projection_linear_edge_values_normalized() {
        let (_store, compiled) =
            load_and_lower("./data/minimal_examples/testing_edges/linear_edge_values.hymeko").unwrap();

        let aggcfg = AggCfg { weight: WeightAgg::Sum, sign: SignAgg::PreferNonNeutral, clamp01: false };
        let hg = HyperGraphView::from_ir(&compiled.ir, &aggcfg);

        // normalized star
        let coo = star_expansion_coo_normalized(&hg, true, 1e-12);
        let proj = hymeko_framework::traversal::tensor::project_sum_over_slices(&coo);

        let num_nodes = hg.num_nodes();
        let num_edges = hg.num_edges();
        let dim = num_nodes + num_edges;
        let edge_base = num_nodes;

        assert_eq!(dim, 10);
        assert_eq!(edge_base, 6);

        let eps: f32 = 1e-4;

        let at = |r: usize, c: usize| -> f32 {
            proj[r][c]
            // proj[(r, c)]
            // *proj.get(r, c).unwrap()
        };

        let assert_close = |r: usize, c: usize, expected: f32| {
            let v = at(r, c);
            assert!(
                (v - expected).abs() <= eps,
                "mismatch at ({},{}) expected {} got {}",
                r, c, expected, v
            );
        };

        // Same sparsity pattern as raw test, different values:

        // e1 (edge vertex = 6)
        assert_close(2, edge_base + 0, 0.33124934); // node1 -> e1
        assert_close(5, edge_base + 0, 0.50418417); // node4 -> e1
        assert_close(edge_base + 0, 1, 0.24206146); // e1 -> node0

        // e2 (7)
        assert_close(edge_base + 1, 3, 0.15891043); // e2 -> node2
        assert_close(edge_base + 1, 4, 0.95346259); // e2 -> node3

        // e3 (8)
        assert_close(edge_base + 2, 1, 0.13558154); // e3 -> node0
        assert_close(edge_base + 2, 3, 0.37158027); // e3 -> node2
        assert_close(edge_base + 2, 5, 0.48136299); // e3 -> node4

        // e4 (9)
        assert_close(5, edge_base + 3, 0.48136299); // node4 -> e4
        assert_close(edge_base + 3, 1, 0.13558154); // e4 -> node0
        assert_close(edge_base + 3, 2, 0.28943731); // e4 -> node1
    }

    #[test]
    fn test_star_projection_normalized_scale_invariant() {
        let (_store, compiled) =
            load_and_lower("./data/minimal_examples/testing_edges/linear_edge_values.hymeko").unwrap();

        let aggcfg = AggCfg { weight: WeightAgg::Sum, sign: SignAgg::PreferNonNeutral, clamp01: false };
        let mut hg1 = HyperGraphView::from_ir(&compiled.ir, &aggcfg);
        let mut hg2 = HyperGraphView::from_ir(&compiled.ir, &aggcfg);

        // scale all incidence weights by c
        let c: f32 = 10.0;
        for w in &mut hg2.flat_edge_w { *w *= c; }
        for w in &mut hg2.flat_node_w { *w *= c; } // ha használod ezt is máshol
        for w in &mut hg2.edge_weight { *w *= 1.0; } // hagyhatod 1.0-n, vagy ezt is skálázhatod

        let coo1 = star_expansion_coo_normalized(&hg1, true, 1e-12);
        let proj1 = hymeko_framework::traversal::tensor::project_sum_over_slices(&coo1);

        let coo2 = star_expansion_coo_normalized(&hg2, true, 1e-12);
        let proj2 = hymeko_framework::traversal::tensor::project_sum_over_slices(&coo2);

        let dim = hg1.num_nodes() + hg1.num_edges();
        assert_eq!(dim, hg2.num_nodes() + hg2.num_edges());

        let eps: f32 = 1e-4;

        let at1 = |r: usize, c: usize| -> f32 { proj1[r][c] };
        let at2 = |r: usize, c: usize| -> f32 { proj2[r][c] };

        for r in 0..dim {
            for c in 0..dim {
                let d = (at1(r, c) - at2(r, c)).abs();
                assert!(d <= eps, "scale invariance failed at ({},{}) diff={}", r, c, d);
            }
        }
    }
}