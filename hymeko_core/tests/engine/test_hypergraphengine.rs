#[cfg(test)]
mod test_hypergraphengine {
    use hymeko::engine::hypergraphengine::HypergraphEngine;
    use log::info;
    use std::time::Instant;
    use crate::test_helpers::{log_test_footer, log_test_header};

    #[test]
    fn fxhashmap_node_registry_is_consistent() {
        log_test_header(
            "fxhashmap_node_registry_is_consistent",
            "Ensures the FxHashMap-based registries deduplicate node names and assign stable IDs.",
        );
        let start = Instant::now();
        let mut engine = HypergraphEngine::new();

        let first_id = engine.get_or_create_node("alpha");
        let duplicate_id = engine.get_or_create_node("alpha");
        let second_id = engine.get_or_create_node("beta");

        info!("alpha -> {first_id}, alpha (duplicate) -> {duplicate_id}, beta -> {second_id}");

        assert_eq!(first_id, duplicate_id, "FxHashMap should deduplicate identical node names");
        assert_ne!(first_id, second_id, "Distinct node names must map to different IDs");
        assert_eq!(engine.node_registry.len(), 2);
        assert_eq!(engine.node_names.len(), 2);
        info!("Node registry entries: {:?}", engine.node_registry);
        info!("Node names vector: {:?}", engine.node_names);
        log_test_footer(
            "fxhashmap_node_registry_is_consistent",
            Some(start.elapsed()),
            "Deduplicated 'alpha' while assigning a unique ID to 'beta'.",
        );
    }

    #[test]
    fn fxhashmap_hashing_load_smoke_test() {
        log_test_header(
            "fxhashmap_hashing_load_smoke_test",
            "Creates 50k unique node/edge pairs to confirm FxHashMap throughput and correctness.",
        );
        const SAMPLE_COUNT: usize = 50_000;
        const MAX_DURATION_SECONDS: f64 = 5.0;
        info!("Settings: sample_count={SAMPLE_COUNT}, max_duration={MAX_DURATION_SECONDS}s");
        let mut engine = HypergraphEngine::new();

        let start = Instant::now();

        for i in 0..SAMPLE_COUNT {
            let node_name = format!("node_{i}");
            let edge_name = format!("edge_{i}");
            let node_id = engine.get_or_create_node(&node_name);
            let edge_id = engine.get_or_create_edge(&edge_name);
            engine.add_arc(0, node_id, edge_id, 1.0).unwrap();
        }

        let elapsed = start.elapsed();
        assert_eq!(engine.node_registry.len(), SAMPLE_COUNT);
        assert_eq!(engine.edge_registry.len(), SAMPLE_COUNT);
        info!("Hashing load test completed in {:.3} seconds", elapsed.as_secs_f64());
        info!("Node registry len: {} | Edge registry len: {}", engine.node_registry.len(), engine.edge_registry.len());
        assert!(elapsed.as_secs_f64() < MAX_DURATION_SECONDS, "hashing load test took too long: {:.3}s", elapsed.as_secs_f64());
        log_test_footer(
            "fxhashmap_hashing_load_smoke_test",
            Some(start.elapsed()),
            &format!("Registered {SAMPLE_COUNT} nodes and edges without collisions."),
        );
    }

}