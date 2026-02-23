#!cfg[(test)]
mod test_ref_values
{
    use memmap2::Mmap;
    use parser::parse_from_mmap;
    use std::fs::File;
    use hymeko_framework::common::pathkey::PathKey;
    use hymeko_framework::ir::ir::{RefAtomR, SignedRefR, ValueR};
    use hymeko_framework::ir::lower::lower_to_ir;
    use hymeko_framework::resolution::intern_pass::{intern_ast, Interned};
    use hymeko_framework::resolution::resolve::build_index_sym;
    use parser::ast::*;


    fn find_node<'a>(items: &'a [HyperItem<'a, &'a str>], name: &str) -> &'a NodeDecl<'a, &'a str> {
        items.iter().find_map(|item| match item {
            HyperItem::Node(n) if n.inner.name == name => Some(n),
            _ => None,
        }).expect("expected node to be present")
    }

    fn find_edge<'a>(items: &'a [HyperItem<'a, &'a str>], name: &str) -> &'a EdgeDecl<'a, &'a str> {
        items.iter().find_map(|item| match item {
            HyperItem::Edge(e) if e.inner.name == name => Some(e),
            _ => None,
        }).expect("expected edge to be present")
    }

    fn edge_arcs<'a>(edge: &'a EdgeDecl<'a, &'a str>) -> Vec<&'a HyperArc<'a, &'a str>> {
        edge.inner.body.iter().filter_map(|item| match item {
            HyperItem::Arc(a) => Some(a),
            _ => None,
        }).collect()
    }

    fn dir_and_atom<'a>(signed: &'a SignedRef<'a, &'a str>) -> (&'static str, &'a RefAtom<'a, &'a str>) {
        match signed {
            SignedRef::Plus(atom) => ("+", atom),
            SignedRef::Minus(atom) => ("-", atom),
            SignedRef::Neutral(atom) => ("~", atom),
        }
    }

    fn assert_weights(atom: &RefAtom<'_, &'_ str>, expected: &[f64]) {
        let actual = atom.anno.weights.as_ref().expect("weights must be present");
        assert_eq!(actual.len(), expected.len(), "weight arity mismatch for {:?}", atom.target.path);
        for (value, exp) in actual.iter().zip(expected.iter()) {
            let num = match value {
                Value::Num(n) => *n,
                other => panic!("expected numeric weight, got {:?}", other),
            };
            assert!((num - exp).abs() < 1e-9, "weight mismatch: got {num}, expected {exp}");
        }
    }

    fn dir_and_atom_ir<'a>(signed: &'a SignedRefR) -> (&'static str, &'a RefAtomR) {
        match signed {
            SignedRefR::Plus(atom) => ("+", atom),
            SignedRefR::Minus(atom) => ("-", atom),
            SignedRefR::Neutral(atom) => ("~", atom),
        }
    }

    fn assert_ir_weights(weights: &Option<Vec<ValueR>>, expected: &[f64]) -> Vec<f64> {
        let actual = weights.as_ref().expect("weights must be present");
        assert_eq!(actual.len(), expected.len(), "weight arity mismatch");
        let mut collected = Vec::with_capacity(actual.len());
        for (value, exp) in actual.iter().zip(expected.iter()) {
            let num = match value {
                ValueR::Num(n) => *n,
                other => panic!("expected numeric weight, got {:?}", other),
            };
            assert!((num - exp).abs() < 1e-9, "weight mismatch: got {num}, expected {exp}");
            collected.push(num);
        }
        collected
    }

    #[test]
    fn edges_with_ref_values() {
        let path = "./data/minimal_examples/testing_edges/minimal_example_with_hierarchy_ref_edges_with_values.hymeko";
        let file = File::open(path).unwrap();
        let mmap = unsafe { Mmap::map(&file).unwrap() };

        let d = parse_from_mmap(&mmap).unwrap();
        assert_eq!(d.name, "Minimal_Example");

        let context = find_node(&d.items, "context");
        let ctx_body = context.inner.body.as_ref().expect("context should contain nested items");
        let node_lev_1 = find_node(ctx_body, "node_lev_1");
        let lev1_body = node_lev_1.inner.body.as_ref().expect("node_lev_1 should contain nested items");

        let edge = find_edge(lev1_body, "e0");
        let arcs = edge_arcs(edge);
        assert_eq!(arcs.len(), 1, "edge e0 should have a single arc");

        let refs = &arcs[0].inner.refs;
        assert_eq!(refs.len(), 4, "arc should reference four nodes");

        let expectations = vec![
            ("-", vec!["node0"], vec![0.85]),
            ("+", vec!["context", "node_lev_0", "node1"], vec![0.9]),
            ("-", vec!["context", "node_lev_0", "node2"], vec![-0.615]),
            ("-", vec!["context", "node_lev_0", "node0"], vec![0.5, 0.6]),
        ];

        for (signed, (dir, path, weights)) in refs.iter().zip(expectations.into_iter()) {
            let (actual_dir, atom) = dir_and_atom(signed);
            assert_eq!(actual_dir, dir, "unexpected reference direction");
            assert_eq!(atom.target.path.as_slice(), path.as_slice(), "reference path mismatch");
            assert_weights(atom, &weights);
        }
    }

    #[test]
    fn edges_with_ref_values_ir() {
        let path = "./data/minimal_examples/testing_edges/minimal_example_with_hierarchy_ref_edges_with_values.hymeko";
        let file = File::open(path).unwrap();
        let mmap = unsafe { Mmap::map(&file).unwrap() };

        let desc = parse_from_mmap(&mmap).unwrap();
        let Interned { ast, mut interner } = intern_ast(&desc);
        let idx = build_index_sym(&ast, &interner).unwrap();
        let ir = lower_to_ir(&ast, &idx, &mut interner).unwrap();

        let sid_context = interner.intern("context");
        let sid_node_lev_0 = interner.intern("node_lev_0");
        let sid_node_lev_1 = interner.intern("node_lev_1");
        let sid_node0 = interner.intern("node0");
        let sid_node1 = interner.intern("node1");
        let sid_node2 = interner.intern("node2");
        let sid_e0 = interner.intern("e0");

        let did_context = *idx.by_path.get(&PathKey(vec![sid_context])).expect("context missing");
        let did_node_lev_0 = *idx.by_path.get(&PathKey(vec![sid_context, sid_node_lev_0])).expect("node_lev_0 missing");
        let did_node_lev_1 = *idx.by_path.get(&PathKey(vec![sid_context, sid_node_lev_1])).expect("node_lev_1 missing");
        let did_node_lev1_node0 = *idx.by_path.get(&PathKey(vec![sid_context, sid_node_lev_1, sid_node0])).expect("node_lev_1.node0 missing");
        let did_node_lev0_node0 = *idx.by_path.get(&PathKey(vec![sid_context, sid_node_lev_0, sid_node0])).expect("node_lev_0.node0 missing");
        let did_node_lev0_node1 = *idx.by_path.get(&PathKey(vec![sid_context, sid_node_lev_0, sid_node1])).expect("node_lev_0.node1 missing");
        let did_node_lev0_node2 = *idx.by_path.get(&PathKey(vec![sid_context, sid_node_lev_0, sid_node2])).expect("node_lev_0.node2 missing");
        let did_e0 = *idx.by_path.get(&PathKey(vec![sid_context, sid_node_lev_1, sid_e0])).expect("edge e0 missing");

        let edge_id = ir.decl_to_edge[did_e0.0 as usize].expect("e0 not lowered as edge");
        let edge_rec = &ir.edges[edge_id.0 as usize];
        assert_eq!(edge_rec.arcs.len(), 1, "expected a single arc in e0");
        let arc_id = edge_rec.arcs[0];
        let arc = &ir.arcs[arc_id.0 as usize];
        assert_eq!(arc.refs.len(), 4, "arc should reference four nodes");

        let expectations = vec![
            ("-", did_node_lev1_node0, vec![0.85]),
            ("+", did_node_lev0_node1, vec![0.9]),
            ("-", did_node_lev0_node2, vec![-0.615]),
            ("-", did_node_lev0_node0, vec![0.5, 0.6]),
        ];
        let mut flattened_weights = Vec::new();

        for (signed, (dir, target, weights)) in arc.refs.iter().zip(expectations.into_iter()) {
            let (actual_dir, atom) = dir_and_atom_ir(signed);
            assert_eq!(actual_dir, dir, "unexpected reference direction");
            assert_eq!(atom.target, target, "reference target mismatch");
            let nums = assert_ir_weights(&atom.weights, &weights);
            flattened_weights.extend(nums);
        }

        let expected_flat = vec![0.85, 0.9, -0.615, 0.5, 0.6];
        assert_eq!(flattened_weights.len(), expected_flat.len(), "flattened weights length mismatch");
        for (actual, expected) in flattened_weights.iter().zip(expected_flat.iter()) {
            assert!((*actual - *expected).abs() < 1e-9, "flattened weight mismatch: got {actual}, expected {expected}");
        }

        // Ensure the context node still maps to a DeclId to avoid unused warnings
        let _ = did_context;
        let _ = did_node_lev_0;
        let _ = did_node_lev_1;
    }
}