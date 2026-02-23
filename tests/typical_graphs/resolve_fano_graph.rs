#!cfg[(test)]
mod resolve_fano_graph {
    use std::fs::File;
    use memmap2::Mmap;
    use hymeko_framework::{body, find_edge, find_node, find_node_id};
    use hymeko_framework::common::ids::{DeclId, SymId};
    use hymeko_framework::ir::ir::SignedRefR;
    use hymeko_framework::resolution::intern_pass::Interned;
    use hymeko_framework::resolution::{intern_pass, resolve};
    use hymeko_framework::resolution::interner::Interner;
    use parser::ast::*;
    use parser::parse_from_mmap;

    #[test]
    fn parse_fano_graph_resolve() -> Result<(), resolve::ResolveError> {
        let path = "./data/typical_graphs/fano_graph.hymeko";
        let file = File::open(path).unwrap();
        let mmap = unsafe { Mmap::map(&file).unwrap() };

        // The AST is valid as long as 'mmap' is in scope.
        let desc = parse_from_mmap(&mmap).unwrap();
        let Interned { ast: ast_sym, mut interner } = intern_pass::intern_ast(&desc);
        let idx = resolve::build_index_sym(&ast_sym, &interner)?;
        resolve::validate_all_refs_sym(&ast_sym, &idx, &mut interner)?;


        // Top-level: "Fano_graph" név + üres header
        assert_eq!(desc.name, "Fano_graph");
        assert!(desc.header.is_empty(), "Expected empty header");

        // Top-levelben legyen a fano block (NodeDecl body-val)
        let fano = find_node(&desc.items, "fano");
        let fano_body = body(fano);

        // 7 node: n0..n6
        for i in 0..7 {
            let n = format!("n{}", i);
            let _ = find_node(fano_body, &n);
        }

        // 7 edge: e0..e6
        // Itt csak azt ellenőrizzük, hogy mindegyik EdgeDecl megvan,
        // és hogy a body-jában 1 arc van, és az 3 referenciát tartalmaz.
        for i in 0..7 {
            let ename = format!("e{}", i);

            let edge = fano_body
                .iter()
                .find_map(|it| match it {
                    HyperItem::Edge(e) if e.inner.name == ename => Some(e),
                    _ => None,
                })
                .unwrap_or_else(|| panic!("Expected Edge({}) in fano body", ename));

            // edge.inner.body : Vec<HyperItem>
            let arc_items: Vec<&parser::ast::HyperArc<&str>> = edge
                .inner
                .body
                .iter()
                .filter_map(|x| match x {
                    HyperItem::Arc(a) => Some(a),
                    _ => None,
                })
                .collect();

            assert_eq!(
                arc_items.len(),
                1,
                "Each edge should contain exactly 1 HyperArc statement; edge={}",
                ename
            );

            let arc = arc_items[0];

            // A te jelenlegi grammarodban:
            // HyperArc { inner: ArcInner { refs } }
            // ahol refs tipikusan Vec<SignedRef> vagy Vec<DirectedRef>.
            // A Fano input "~ n0" -> SignedRef::Neutral várható.
            assert_eq!(
                arc.inner.refs.len(),
                3,
                "Expected 3 endpoints in arc inside edge {}",
                edge.inner.name
            );
        }

        // extra sanity: a fano body-ban ne legyen top-level arc
        // (minden arc edge body-ban van)
        assert!(
            !fano_body.iter().any(|it| matches!(it, HyperItem::Arc(_))),
            "Did not expect HyperArc directly under `fano graph`"
        );
        // Check that all references are resolved (iterate over all edges)


        Ok(())
    }

    #[test]
    fn fano_graph_shape() -> Result<(), Box<dyn std::error::Error>> {
        // parse -> AST<String>
        let path = "./data/typical_graphs/fano_graph.hymeko";
        let file = File::open(path).unwrap();
        let mmap = unsafe { Mmap::map(&file).unwrap() };

        // The AST is valid as long as 'mmap' is in scope.
        let d_str = parse_from_mmap(&mmap).unwrap();

        // intern -> AST<SymId> + Interner
        let interned = intern_pass::intern_ast(&d_str);
        let ast = &interned.ast;
        let it = &interned.interner;

        let fano_id = it.get_id("fano").expect("missing SymId for 'fano'");
        let n_ids: Vec<SymId> = (0..=6).map(|i| it.get_id(&format!("n{i}")).unwrap()).collect();
        let e_ids: Vec<SymId> = (0..=6).map(|i| it.get_id(&format!("e{i}")).unwrap()).collect();

        // top-level items: itt kell lennie a fano node-nak
        let fano = find_node_id(&ast.items, fano_id);

        let fano_body = fano.inner.body.as_deref().expect("fano should have a body");
        assert_eq!(fano_body.len(), 14, "fano body should contain 7 nodes + 7 edges");

        // Check node count
        assert_eq!(fano_body.iter().filter(|it| matches!(it, HyperItem::Node(_))).count(), 7, "fano body should contain exactly 7 nodes");
        for &nid in &n_ids {
            let _ = find_node_id(fano_body, nid);
        }

        // Check edge count and structure
        assert_eq!(fano_body.iter().filter(|it| matches!(it, HyperItem::Edge(_))).count(), 7, "fano body should contain exactly 7 edges");
        // 7 edges megvannak, mindegyikben 1 arc, és az arcban 3 neutral ref
        for &eid in &e_ids {
            let e = find_edge(fano_body, eid);
            assert_eq!(e.inner.body.len(), 1, "edge body should contain exactly 1 item (the arc)");

            let arc = match &e.inner.body[0] {
                HyperItem::Arc(a) => a,
                other => panic!("edge body should be Arc, got: {:?}", other),
            };

            assert_eq!(arc.inner.refs.len(), 3, "each edge arc should contain 3 refs");

            for sref in &arc.inner.refs {
                let atom = match sref {
                    SignedRef::Neutral(a) => a,
                    other => panic!("expected Neutral (~) refs, got: {:?}", other),
                };

                assert_eq!(atom.target.path.len(), 1, "Fano refs should be simple names like ~n0");
                let target = atom.target.path[0];
                assert!(n_ids.contains(&target), "arc target should be one of n0..n6");
            }
        }

        Ok(())
    }

    fn invert_index(
        idx: &resolve::Index,
        it: &Interner,
    ) -> std::collections::HashMap<DeclId, String> {
        let mut inv = std::collections::HashMap::new();
        for (k, &did) in &idx.by_path {
            let s = k.0.iter().map(|&sid| it.resolve(sid).to_string()).collect::<Vec<_>>().join(".");
            inv.insert(did, s);
        }
        inv
    }

    // kinyeri a "fano.n0" típusú resolved célneveket az arc refs-ből
    fn resolved_targets_as_strings(
        refs: &[SignedRefR],
        inv: &std::collections::HashMap<DeclId, String>,
    ) -> Vec<String> {
        refs.iter()
            .map(|r| {
                let did = match r {
                    SignedRefR::Plus(a) => a.target,
                    SignedRefR::Minus(a) => a.target,
                    SignedRefR::Neutral(a) => a.target,
                };
                inv.get(&did)
                    .cloned()
                    .unwrap_or_else(|| format!("<unknown {did:?}>"))
            })
            .collect()
    }

    #[test]
    fn fano_edges_resolve_to_expected_nodes() -> Result<(), Box<dyn std::error::Error>> {
        // 1) Parse -> AST<String>
        let path = "./data/typical_graphs/fano_graph.hymeko";
        let file = File::open(path).unwrap();
        let mmap = unsafe { Mmap::map(&file).unwrap() };

        // The AST is valid as long as 'mmap' is in scope.
        let d_str = parse_from_mmap(&mmap).unwrap();

        // 2) Intern -> AST<SymId> + Interner
        let Interned { ast, mut interner } = intern_pass::intern_ast(&d_str);
        

        // 3) Index build (PathKey(Vec<SymId>) -> DeclId), duplikátum tiltással
        let idx = resolve::build_index_sym(&ast, &interner).unwrap();

        // 4) Invert index a teszt-összehasonlításhoz
        let inv = invert_index(&idx, &interner);

        // 5) Keresd meg a `fano` node-ot és a body-t
        let nid = interner.get_id("fano").unwrap();
        let fano = find_node_id(&ast.items, nid);
        let fano_body = fano.inner.body.as_deref().expect("fano should have body");

        // 6) Várt Fano-incidenciák (edge -> 3 node)
        //    (a te inputod alapján)
        let expected: [(&str, [&str; 3]); 7] = [
            ("e0", ["n0", "n1", "n3"]),
            ("e1", ["n0", "n2", "n6"]),
            ("e2", ["n0", "n4", "n5"]),
            ("e3", ["n1", "n2", "n4"]),
            ("e4", ["n2", "n3", "n5"]),
            ("e5", ["n3", "n4", "n6"]),
            ("e6", ["n1", "n5", "n6"]),
        ];

        // SymId-k a scope-hoz
        let fano_sid: SymId = interner.get_id("fano").expect("missing SymId for 'fano'");

        for (ename, nodes) in expected {
            let nid = interner.get_id(ename).unwrap();
            let e = find_edge(fano_body, nid);
            assert_eq!(e.inner.body.len(), 1, "{ename}: edge body should contain exactly 1 item");

            let arc = match &e.inner.body[0] {
                HyperItem::Arc(a) => a,
                other => panic!("{ename}: expected Arc in edge body, got {other:?}"),
            };

            // sanity: mindhárom ref neutral (~)
            for r in &arc.inner.refs {
                match r {
                    SignedRef::Neutral(_) => {}
                    other => panic!("{ename}: expected Neutral (~) refs, got {other:?}"),
                }
            }

            // Scope: [fano, eX]
            let e_sid = interner.get_id(ename).expect("missing SymId for edge name");
            let scope = vec![fano_sid, e_sid];

            // Resolve refs -> DeclId
            let resolved = resolve::resolve_arc_refs(&idx, &scope, arc, &mut interner).unwrap();

            // DeclId -> "fano.nK" string
            let mut got = resolved_targets_as_strings(&resolved, &inv);
            got.sort();

            let mut exp = nodes.iter().map(|n| format!("fano.{n}")).collect::<Vec<_>>();
            exp.sort();
            // Print info everytime for easier debugging
            println!("Edge {ename}: expected targets = {exp:?}, got = {got:?}");

            assert_eq!(got, exp, "{ename}: resolved targets mismatch");
        }

        Ok(())
    }

}