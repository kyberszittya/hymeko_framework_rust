#![cfg(test)]
mod test_import_graphs
{
    use std::path::Path;
    use hymeko::common::pathkey::PathKey;
    use hymeko::ir::hash_pass::compute_merkle_hashes;
    use hymeko::ir::ir::{DeclKind, SignedRefR};
    use hymeko::ir::lower::lower_to_ir;
    use hymeko::module_store::module_store::ModuleStore;
    use hymeko::resolution::intern_pass::{intern_ast, Interned};
    use hymeko::resolution::resolve::build_index_sym;
    use hymeko::module_store::source_provider::StdFsProvider;
    use crate::minimal_tests::TestParser;

    #[test]
    fn check_import_graph_library() {
        let path = "data/minimal_examples/import_examples/minimal_example_library.hymeko";
        let source_code = parser::read_source_file(&path).expect("failed to read source file");

        // 2. Parse it, tying the AST lifetimes to the String
        let desc = parser::parse_description(&source_code).unwrap();
        assert_eq!(desc.name, "basic_library");
        let Interned { ast, mut interner } = intern_ast(&desc);
        let idx = build_index_sym(&ast, &mut interner).expect("index build failed");
        let mut ir = lower_to_ir(&ast, &idx, &mut interner).expect("lower failed");
        compute_merkle_hashes(&mut ir, &interner);

        let sid_elements = interner.intern("elements");
        let sid_operand = interner.intern("operand");
        let sid_operand2 = interner.intern("operand2");
        let sid_operator = interner.intern("operator");

        let did_elements = *idx.by_path.get(&PathKey(vec![sid_elements])).expect("elements missing");
        assert_eq!(ir.decl_nodes[did_elements.0].kind, DeclKind::Node);

        let did_operand = *idx.by_path.get(&PathKey(vec![sid_elements, sid_operand])).expect("operand missing");
        let did_operand2 = *idx.by_path.get(&PathKey(vec![sid_elements, sid_operand2])).expect("operand2 missing");
        assert_eq!(ir.decl_nodes[did_operand.0].kind, DeclKind::Node);
        assert_eq!(ir.decl_nodes[did_operand2.0].kind, DeclKind::Node);

        let did_operator = *idx.by_path.get(&PathKey(vec![sid_elements, sid_operator])).expect("operator edge missing");
        assert_eq!(ir.decl_nodes[did_operator.0].kind, DeclKind::Edge);

        let edge_id = ir.decl_to_edge[did_operator.0].expect("operator not lowered as edge");
        let edge_rec = &ir.edges[edge_id.0];
        assert_eq!(edge_rec.arcs.len(), 1, "operator edge should contain one arc");

        let arc_id = edge_rec.arcs[0];
        let arc = &ir.arcs[arc_id.0];
        assert_eq!(arc.refs.len(), 2, "operator arc should connect two refs");

        match (&arc.refs[0], &arc.refs[1]) {
            (SignedRefR::Plus(lhs), SignedRefR::Minus(rhs)) => {
                assert_eq!(lhs.target, did_operand, "+ operand should target operand node");
                assert_eq!(rhs.target, did_operand, "- operand should target operand node");
                assert!(lhs.weights.as_ref().is_some(), "lhs weight missing");
                assert!(rhs.weights.as_ref().is_some(), "rhs weight missing");
            }
            other => panic!("unexpected arc refs ordering: {other:?}"),
        }
    }

    #[test]
    fn check_import_graph_library_with_import() {



        // ugyanaz a root file, mint eddig
        let root_path = Path::new("./data/minimal_examples/import_examples/minimal_example_import.hymeko");

        // Parser adapter a LALRPOP-hoz (igazítsd a modulneveket, ha kell)


        let mut ms = ModuleStore::new(StdFsProvider::new(), TestParser);

        let compiled = ms.compile(&root_path).expect("compile failed");

        let ns = ms.it.intern("basic_library");
        let elements = ms.it.intern("elements");
        let operand = ms.it.intern("operand");
        assert!(compiled.idx.by_path.contains_key(&PathKey(vec![ns, elements, operand])));

        assert!(
            compiled.imports.iter().any(|(ns, _)| *ns == ms.it.intern("basic_library")),
            "expected imported namespace basic_library"
        );

        let did = *compiled.idx.by_path
            .get(&PathKey(vec![ns, elements, operand]))
            .expect("missing basic_library.elements.operand in global index");

        let kind = compiled.ir.decl_nodes[did.0].kind;
        assert!(matches!(kind, DeclKind::Node|DeclKind::Edge|DeclKind::HyperArc));

        let mut referenced = false;
        for arc in &compiled.ir.arcs {
            for r in &arc.refs {
                let tgt = match r {
                    SignedRefR::Plus(a) | SignedRefR::Minus(a) | SignedRefR::Neutral(a) => a.target,
                };
                if tgt == did {
                    referenced = true;
                    break;
                }
            }
            if referenced { break; }
        }
        assert!(referenced, "expected at least one arc ref to target basic_library.elements.operand");
        assert!(
            compiled.ir.decl_hash.get(did.0).and_then(|x| *x).is_some(),
            "expected decl hash for operand to be computed"
        );

    }
}