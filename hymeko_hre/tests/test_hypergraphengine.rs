//! Integration tests for `hymeko_hre::HypergraphEngine`.
//! Moved from `hymeko_core/tests/engine/` during the 2026-04-18 HRE extraction.

use std::time::Instant;

use hymeko_hre::HypergraphEngine;

#[test]
fn fxhashmap_node_registry_is_consistent() {
    let mut engine = HypergraphEngine::new();

    let first_id = engine.get_or_create_node("alpha");
    let duplicate_id = engine.get_or_create_node("alpha");
    let second_id = engine.get_or_create_node("beta");

    assert_eq!(first_id, duplicate_id, "FxHashMap should deduplicate identical node names");
    assert_ne!(first_id, second_id, "Distinct node names must map to different IDs");
    assert_eq!(engine.node_registry.len(), 2);
    assert_eq!(engine.node_names.len(), 2);
}

#[test]
fn fxhashmap_hashing_load_smoke_test() {
    const SAMPLE_COUNT: usize = 50_000;
    const MAX_DURATION_SECONDS: f64 = 5.0;

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
    assert!(
        elapsed.as_secs_f64() < MAX_DURATION_SECONDS,
        "hashing load test took too long: {:.3}s",
        elapsed.as_secs_f64()
    );
}
