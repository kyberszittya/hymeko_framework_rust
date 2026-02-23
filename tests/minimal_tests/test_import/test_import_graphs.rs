#![cfg(test)]
mod test_import_graphs
{
    use memmap2::Mmap;
    use parser::parse_from_mmap;
    use std::fs::File;
    use std::path::Path;
    use hymeko_framework::common::pathkey::PathKey;
    use hymeko_framework::ir::hash_pass::compute_merkle_hashes;
    use hymeko_framework::ir::ir::{DeclKind, SignedRefR};
    use hymeko_framework::ir::lower::lower_to_ir;
    use hymeko_framework::module_store::module_store::{HymekoParser, ModuleStore};
    use hymeko_framework::resolution::intern_pass::{intern_ast, Interned};
    use hymeko_framework::resolution::resolve::build_index_sym;
    use hymeko_framework::module_store::source_provider::StdFsProvider;
    use parser::ast::{AstStr};
    use parser::hymeko::DescriptionParser;
    use parser::lexer::simd::Lexer;

    #[test]
    fn check_import_graph_library() {
        let path = "data/minimal_examples/import_examples/minimal_example_library.hymeko";
        let file = File::open(path).expect("failed to open hymeko file");
        let mmap = unsafe { Mmap::map(&file).expect("mmap failed") };
        let desc = parse_from_mmap(&mmap).expect("parse_from_mmap failed");
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
        assert_eq!(ir.decl_kind[did_elements.0 as usize], DeclKind::Node);

        let did_operand = *idx.by_path.get(&PathKey(vec![sid_elements, sid_operand])).expect("operand missing");
        let did_operand2 = *idx.by_path.get(&PathKey(vec![sid_elements, sid_operand2])).expect("operand2 missing");
        assert_eq!(ir.decl_kind[did_operand.0 as usize], DeclKind::Node);
        assert_eq!(ir.decl_kind[did_operand2.0 as usize], DeclKind::Node);

        let did_operator = *idx.by_path.get(&PathKey(vec![sid_elements, sid_operator])).expect("operator edge missing");
        assert_eq!(ir.decl_kind[did_operator.0 as usize], DeclKind::Edge);

        let edge_id = ir.decl_to_edge[did_operator.0 as usize].expect("operator not lowered as edge");
        let edge_rec = &ir.edges[edge_id.0 as usize];
        assert_eq!(edge_rec.arcs.len(), 1, "operator edge should contain one arc");

        let arc_id = edge_rec.arcs[0];
        let arc = &ir.arcs[arc_id.0 as usize];
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
        let root_path = Path::new("data/minimal_examples/import_examples/minimal_example_import.hymeko");

        // Parser adapter a LALRPOP-hoz (igazítsd a modulneveket, ha kell)
        struct TestParser;
        impl HymekoParser for TestParser {
            fn parse<'a>(&self, src: &'a str) -> Result<AstStr<'a>, String> {
                let p = DescriptionParser::new();
                p.parse(Lexer::new(src))
                    .map_err(|e| format!("{e:?}"))
            }
        }

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

        let kind = compiled.ir.decl_kind[did.0 as usize];
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
            compiled.ir.decl_hash.get(did.0 as usize).and_then(|x| *x).is_some(),
            "expected decl hash for operand to be computed"
        );

    }
}