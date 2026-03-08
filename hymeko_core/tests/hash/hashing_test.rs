#[cfg(test)]
mod tests {
    use hymeko::common::ids::{DeclId};
    use hymeko::common::pathkey::PathKey;
    use hymeko::ir::hash::{hash_doc, HashId};
    use hymeko::resolution::interner::Interner;
    use hymeko::resolution::resolve::Index;
    use std::collections::BTreeMap;
    use std::time::Instant;

    const SYMBOL_FANO: &str = "fano";
    const SYMBOL_NODE_0: &str = "n0";
    const SYMBOL_NODE_1: &str = "n1";
    const SYMBOL_EDGE_0: &str = "e0";
    const MASSIVE_NODE_COUNT: usize = 10_000;
    const HASH_RUNS: usize = 100;
    const PERF_BUDGET_MS: u128 = 750;
    const PERF_AVG_BUDGET_MS: f64 = 10.0;

    // Helper to mock the environment
    fn setup_mock_env() -> (Interner, Vec<(PathKey, DeclId)>) {
        let mut it = Interner::new();
        // Register some symbols
        let s_fano = it.intern(SYMBOL_FANO);
        let s_n0 = it.intern(SYMBOL_NODE_0);
        let s_n1 = it.intern(SYMBOL_NODE_1);
        let s_e0 = it.intern(SYMBOL_EDGE_0);

        // Create some paths
        let paths = vec![
            (PathKey(vec![s_fano, s_n0]), DeclId(0)),
            (PathKey(vec![s_fano, s_n1]), DeclId(1)),
            (PathKey(vec![s_fano, s_e0]), DeclId(2)),
            (PathKey(vec![s_fano]), DeclId(3)), // Parent path
        ];

        (it, paths)
    }

    #[test]
    fn test_hash_doc_determinism_insertion_order() {
        let (it, paths) = setup_mock_env();

        // Environment A: Inserted in Forward Order
        let mut idx_a = Index { by_path: BTreeMap::new() };
        for (pk, did) in paths.clone().into_iter() {
            idx_a.by_path.insert(pk, did);
        }
        let hash_a = hash_doc(&idx_a, &it);

        // Environment B: Inserted in Reverse Order
        let mut idx_b = Index { by_path: BTreeMap::new() };
        for (pk, did) in paths.clone().into_iter().rev() {
            idx_b.by_path.insert(pk, did);
        }
        let hash_b = hash_doc(&idx_b, &it);

        // Environment C: Inserted in Scrambled Order
        let mut idx_c = Index { by_path: BTreeMap::new() };
        idx_c.by_path.insert(paths[2].0.clone(), paths[2].1);
        idx_c.by_path.insert(paths[0].0.clone(), paths[0].1);
        idx_c.by_path.insert(paths[3].0.clone(), paths[3].1);
        idx_c.by_path.insert(paths[1].0.clone(), paths[1].1);
        let hash_c = hash_doc(&idx_c, &it);

        // THE CRITICAL ASSERTION: All hashes must be perfectly identical
        // despite the chaotic memory insertion order.
        assert_eq!(hash_a, hash_b, "Hash failed reverse order determinism!");
        assert_eq!(hash_a, hash_c, "Hash failed scrambled order determinism!");
    }

    #[test]
    fn test_hash_doc_performance_benchmark() {
        let (mut it, paths) = setup_mock_env();

        // Build a massive index to simulate a heavy hypergraph
        let mut massive_idx = Index { by_path: BTreeMap::new() };
        for i in 0..MASSIVE_NODE_COUNT {
            let s_node = it.intern(&format!("node_{}", i));
            massive_idx.by_path.insert(PathKey(vec![paths[3].0.0[0], s_node]), DeclId(i));
        }

        let start = Instant::now();

        // Run the hash 100 times to test throughput
        let mut last_hash = HashId([0; 32]);
        for _ in 0..HASH_RUNS {
            last_hash = hash_doc(&massive_idx, &it);
        }

        let elapsed = start.elapsed();
        let avg_ms = elapsed.as_secs_f64() * 1_000.0 / HASH_RUNS as f64;

        // Log the telemetry (aligning with your recent CI telemetry cleanup)
        println!(
            "Hashing 10,000 nodes 100 times took: {:?} (avg {:.3} ms/run)",
            elapsed,
            avg_ms
        );

        // Basic sanity check to ensure it actually computed something
        assert_ne!(last_hash.0, [0; 32]);

        // We expect this to be well under a few milliseconds per run
        assert!(elapsed.as_millis() < PERF_BUDGET_MS, "Hashing is suspiciously slow!");
        assert!(avg_ms < PERF_AVG_BUDGET_MS, "Average hash runtime {:.3} ms exceeds budget", avg_ms);
    }
}