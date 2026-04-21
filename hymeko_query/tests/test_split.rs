//! Integration test for the split-layer rewrite proposer on the real
//! compiled IR of `data/nn/simple_net.hymeko`. Checks that the
//! auto-picked scope is one of the expected candidates and that the
//! proposal is well-formed (both clusters non-empty, all edges
//! accounted for, counts consistent).

#[cfg(test)]
mod test_split_on_simple_net {
    use hymeko::ir::ir::DeclKind;
    use hymeko_query::rewrite::split::{
        propose_split, propose_split_for_highest_h_sign, Cluster,
    };

    use crate::test_helpers::{find_decl, load_and_lower};

    const SIMPLE_NET: &str = "../data/nn/simple_net.hymeko";

    #[test]
    fn proposes_split_of_layer_0() {
        let (_store, compiled) = load_and_lower(SIMPLE_NET).expect("compile simple_net");
        let layer_0 = find_decl(&compiled.ir, &_store.it, "layer_0", DeclKind::Node);

        let proposal = propose_split(&compiled.ir, layer_0)
            .expect("layer_0 has both vertices and edges — should propose");

        // Well-formedness: every edge in layer_0's scope got an assignment.
        assert_eq!(proposal.edge_assignments.len(), 5, "layer_0 has 5 factor edges");

        // Both clusters non-empty.
        assert!(!proposal.cluster_a.is_empty(), "cluster A empty");
        assert!(!proposal.cluster_b.is_empty(), "cluster B empty");
        // Together they cover every vertex that appears in any edge
        // (attribute decls have degree 0 and may end up in either).
        let total = proposal.cluster_a.len() + proposal.cluster_b.len();
        assert!(total >= 13, "clusters should together cover inner vertices, got {total}");

        // Cross-edge count sanity: it's between 0 and edge count inclusive.
        assert!(proposal.n_cross_edges <= 5);
        assert_eq!(
            proposal.n_cross_edges,
            proposal.edge_assignments.iter()
                .filter(|(_, c)| *c == Cluster::Cross).count(),
            "n_cross_edges must match count of Cross assignments",
        );

        // Inertia is finite and non-negative.
        assert!(proposal.inertia.is_finite());
        assert!(proposal.inertia >= 0.0);
    }

    #[test]
    fn auto_picks_scope_with_highest_h_sign() {
        let (_store, compiled) = load_and_lower(SIMPLE_NET).expect("compile simple_net");

        // From the entropy integration test we know simple_net scope
        // has H_sign = ln(3) ≈ 1.0986, the theoretical max for 3-sign
        // alphabet — it should win the auto-pick over layer_0
        // (H_sign ≈ 0.95) and layer_1 (H_sign ≈ 0.80).
        let proposal = propose_split_for_highest_h_sign(&compiled.ir)
            .expect("simple_net has scopes to pick from");

        let simple_net = find_decl(&compiled.ir, &_store.it, "simple_net", DeclKind::Node);
        assert_eq!(
            proposal.target_scope, simple_net,
            "auto-pick should land on `simple_net` (H_sign = ln 3 — the max)"
        );
    }

    #[test]
    fn determinism_on_real_ir() {
        let (_store_a, compiled_a) = load_and_lower(SIMPLE_NET).expect("compile a");
        let (_store_b, compiled_b) = load_and_lower(SIMPLE_NET).expect("compile b");

        let layer_0_a = find_decl(&compiled_a.ir, &_store_a.it, "layer_0", DeclKind::Node);
        let layer_0_b = find_decl(&compiled_b.ir, &_store_b.it, "layer_0", DeclKind::Node);
        // DeclIds should match across recompiles (content-addressable).
        assert_eq!(layer_0_a.raw(), layer_0_b.raw());

        let p_a = propose_split(&compiled_a.ir, layer_0_a).unwrap();
        let p_b = propose_split(&compiled_b.ir, layer_0_b).unwrap();

        assert_eq!(p_a.cluster_a.iter().map(|d| d.raw()).collect::<Vec<_>>(),
                   p_b.cluster_a.iter().map(|d| d.raw()).collect::<Vec<_>>());
        assert_eq!(p_a.cluster_b.iter().map(|d| d.raw()).collect::<Vec<_>>(),
                   p_b.cluster_b.iter().map(|d| d.raw()).collect::<Vec<_>>());
        assert_eq!(p_a.inertia, p_b.inertia);
        assert_eq!(p_a.n_cross_edges, p_b.n_cross_edges);
    }
}
