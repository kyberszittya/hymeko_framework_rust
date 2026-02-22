#[cfg(test)]
mod ir_blake3_hash_tests {
    use blake3::Hasher;
    use memmap2::Mmap;
    use std::collections::HashMap;
    use std::fs::File;
    use parser::ast::AstStr;
    use parser::common::pathkey::PathKey;
    use parser::intern_pass::{intern_ast, Interned};
    use parser::ir::hash::HashId;
    use parser::ir::hash_pass::compute_merkle_hashes;
    use parser::ir::ir::DeclKind;
    use parser::ir::lower::lower_to_ir;
    use parser::{parse_description, parse_from_mmap};
    use parser::resolve::build_index_sym;

    #[test]
    fn singleton_node_hash_matches_blake3_preimage() {
        let src = r#"
        HashDoc {}
        Root;
        "#;

        let desc: AstStr = parse_description(src).expect("parse failed");
        let Interned { ast, mut interner } = intern_ast(&desc);
        let idx = build_index_sym(&ast, &mut interner).expect("index build failed");
        let mut ir = lower_to_ir(&ast, &idx, &mut interner).expect("lower_to_ir failed");

        compute_merkle_hashes(&mut ir, &interner);

        let sid_root = interner.intern("Root");
        let did_root = *idx
            .by_path
            .get(&PathKey(vec![sid_root]))
            .expect("missing Root decl");

        let actual = ir.decl_hash[did_root.0 as usize]
            .expect("Root hash not computed");

        // Manually reproduce the hashing preimage for this simple node.
        let mut hasher = Hasher::new();
        hasher.update(&[DeclKind::Node as u8]);
        hasher.update(b"Root");
        hasher.update(&(0u64).to_le_bytes()); // zero tags
        hasher.update(&[0]); // no value present
        let expected = HashId(*hasher.finalize().as_bytes());

        assert_eq!(actual.0, expected.0, "Root hash must match raw BLAKE3 digest");
    }

    fn context_hash_from_file(path: &str) -> HashId {
        let file = File::open(path).expect("failed to open hymeko file");
        let mmap = unsafe { Mmap::map(&file).expect("mmap failed") };
        let desc = parse_from_mmap(&mmap).expect("parse_from_mmap failed");
        let Interned { ast, mut interner } = intern_ast(&desc);
        let idx = build_index_sym(&ast, &interner).expect("index build failed");
        let mut ir = lower_to_ir(&ast, &idx, &mut interner).expect("lower_to_ir failed");

        compute_merkle_hashes(&mut ir, &interner);

        let sid_context = interner.intern("context");
        let did_context = *idx
            .by_path
            .get(&PathKey(vec![sid_context]))
            .expect("context missing");

        ir.decl_hash[did_context.0 as usize]
            .expect("context hash missing")
    }

    #[test]
    fn context_hash_differs_between_examples() {
        let edge_fixture = "./data/minimal_examples/testing_edges/minimal_example_with_hierarchy_ref_edges_with_values.hymeko";
        let hierarchy_fixture = "./data/minimal_examples/minimal_example_basic_hierarchy.hymeko";

        let edge_hash = context_hash_from_file(edge_fixture);
        let hierarchy_hash = context_hash_from_file(hierarchy_fixture);
        let edge_hash_again = context_hash_from_file(edge_fixture);

        assert_eq!(edge_hash.0, edge_hash_again.0, "hashing should be deterministic for identical inputs");
        assert_ne!(edge_hash.0, hierarchy_hash.0, "distinct descriptions must yield distinct context hashes");
    }

    fn hashes_for_paths<'a>(path: &str, specs: &[&[&'a str]]) -> HashMap<String, HashId> {
        let file = File::open(path).expect("failed to open hymeko file");
        let mmap = unsafe { Mmap::map(&file).expect("mmap failed") };
        let desc = parse_from_mmap(&mmap).expect("parse_from_mmap failed");
        let Interned { ast, mut interner } = intern_ast(&desc);
        let idx = build_index_sym(&ast, &interner).expect("index build failed");
        let mut ir = lower_to_ir(&ast, &idx, &mut interner).expect("lower_to_ir failed");

        compute_merkle_hashes(&mut ir, &interner);

        let mut out = HashMap::new();
        for spec in specs {
            let mut path_syms = Vec::with_capacity(spec.len());
            for segment in *spec {
                path_syms.push(interner.intern(segment));
            }
            let key = spec.join(".");
            let did = *idx
                .by_path
                .get(&PathKey(path_syms.clone()))
                .unwrap_or_else(|| panic!("missing path {key}"));
            let hash = ir.decl_hash[did.0 as usize]
                .unwrap_or_else(|| panic!("missing hash for {key}"));
            out.insert(key, hash);
        }
        out
    }

    #[test]
    fn edge_fixture_hashes_cover_subelements() {
        let fixture = "./data/minimal_examples/testing_edges/minimal_example_with_hierarchy_ref_edges_with_values.hymeko";
        let specs: [&[&str]; 6] = [
            &["context"],
            &["context", "node_lev_0"],
            &["context", "node_lev_1"],
            &["context", "node_lev_1", "e0"],
            &["context", "node_lev_0", "node0"],
            &["context", "node_lev_0", "node1"],
        ];

        let first = hashes_for_paths(fixture, &specs);
        let second = hashes_for_paths(fixture, &specs);
        assert_eq!(first, second, "hashing must be deterministic for edge fixture");

        let context = first["context"];
        let level0 = first["context.node_lev_0"];
        let level1 = first["context.node_lev_1"];
        let edge_e0 = first["context.node_lev_1.e0"];
        assert_ne!(context.0, level0.0, "parent and child hashes must differ");
        assert_ne!(level1.0, edge_e0.0, "edge should hash differently than containing node");

        let node0 = first["context.node_lev_0.node0"];
        let node1 = first["context.node_lev_0.node1"];
        assert_ne!(node0.0, node1.0, "distinct siblings must yield distinct hashes");
    }

    #[test]
    fn hierarchy_fixture_hashes_cover_subelements() {
        let fixture = "./data/minimal_examples/minimal_example_basic_hierarchy.hymeko";
        let specs: [&[&str]; 6] = [
            &["context"],
            &["context", "node_lev_0"],
            &["context", "node_lev_0", "node0"],
            &["context", "node_lev_0", "node0", "node0"],
            &["context", "node_lev_0", "node1"],
            &["context", "node_lev_1"],
        ];

        let first = hashes_for_paths(fixture, &specs);
        let second = hashes_for_paths(fixture, &specs);
        assert_eq!(first, second, "hashing must be deterministic for hierarchy fixture");

        let context = first["context"];
        let level0 = first["context.node_lev_0"];
        let level1 = first["context.node_lev_1"];
        assert_ne!(context.0, level0.0, "context hash should differ from node_lev_0");
        assert_ne!(context.0, level1.0, "context hash should differ from node_lev_1");

        let inner_node0 = first["context.node_lev_0.node0"];
        let inner_node0_child = first["context.node_lev_0.node0.node0"];
        assert_ne!(inner_node0.0, inner_node0_child.0, "nested child should impact hash");
        assert_ne!(first["context.node_lev_0.node1"].0, inner_node0.0, "sibling hashes must differ");
    }
}
