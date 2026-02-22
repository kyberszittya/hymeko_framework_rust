#!cfg[(test)]
mod test_ref_values
{
    use memmap2::Mmap;
    use parser::parse_from_mmap;
    use std::fs::File;

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
}