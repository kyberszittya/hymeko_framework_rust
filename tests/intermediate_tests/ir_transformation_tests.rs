
#[cfg(test)]
mod basic_transformation_tests {
    use hymeko::common::ids::{DeclId, SymId};
    use hymeko::common::pathkey::PathKey;
    use hymeko::resolution::intern_pass::{intern_ast, Interned};
    use hymeko::ir::common::{ref_sign, ref_target};
    use hymeko::ir::ir::{DeclKind};
    use hymeko::ir::lower::lower_to_ir;
    use parser::parse_description;
    use hymeko::resolution::resolve::build_index_sym;



    #[test]
    fn node_child_chain_order_is_body_order() {
        // Minimal, self-contained input that creates node→node nesting
        let src = r#"
        test{}
        D {
            Root {
                A;
                B;
                C;
            }
        }
        "#;

        // parse -> intern -> index -> lower
        let ast_str = parse_description(src).expect("parse failed");
        let Interned { ast, mut interner } = intern_ast(&ast_str);
        let idx = build_index_sym(&ast, &interner).expect("index build failed");
        let ir = lower_to_ir(&ast, &idx, &mut interner).expect("lower_to_ir failed");
        // List all elements stored in interner for debugging
        println!("Interner contents:");
        for (sid, s) in interner.iter() {
            println!("  SymId({}): '{}'", sid.0, s);
        }
        // List all paths in the index for debugging
        println!("Index paths:");
        for (path_key, did) in &idx.by_path {
            let path_str = path_key.0.iter().map(|&sid| interner.resolve(sid)).collect::<Vec<_>>().join(".");
            println!("  PathKey({}): '{}'", did.0, path_str);
        }

        // Resolve DeclIds by path: Root, Root.A, Root.B, Root.C
        let sid_root = interner.intern("Root");
        let sid_a = interner.intern("A");
        let sid_b = interner.intern("B");
        let sid_c = interner.intern("C");
        // Print resolved sids for debugging
        println!("Resolved SymIds:");
        println!("  Root: SymId({})", sid_root.0);
        println!("  A: SymId({})", sid_a.0);
        println!("  B: SymId({})", sid_b.0);
        println!("  C: SymId({})", sid_c.0);

        // Assert Interner contents
        assert_eq!(interner.resolve(SymId(0)), "test", "SymId(0) should be 'test'");
        assert_eq!(interner.resolve(SymId(1)), "D", "SymId(1) should be 'D'");
        assert_eq!(interner.resolve(SymId(2)), "Root", "SymId(2) should be 'Root'");
        assert_eq!(interner.resolve(SymId(3)), "A", "SymId(3) should be 'A'");
        assert_eq!(interner.resolve(SymId(4)), "B", "SymId(4) should be 'B'");
        assert_eq!(interner.resolve(SymId(5)), "C", "SymId(5) should be 'C'");

        // Assert resolved SymIds
        assert_eq!(sid_root.0, 2, "Root SymId should be 2");
        assert_eq!(sid_a.0, 3, "A SymId should be 3");
        assert_eq!(sid_b.0, 4, "B SymId should be 4");
        assert_eq!(sid_c.0, 5, "C SymId should be 5");

        // Assert Index paths contain expected DeclIds
        let sid_d = interner.intern("D");
        assert_eq!(sid_d.0, 1, "D SymId should be 1");

        let did_d = *idx.by_path.get(&PathKey(vec![sid_d])).expect("missing D");
        let did_root = *idx.by_path.get(&PathKey(vec![sid_d, sid_root])).expect("missing D.Root");
        let did_a = *idx.by_path.get(&PathKey(vec![sid_d, sid_root, sid_a])).expect("missing D.Root.A");
        let did_b = *idx.by_path.get(&PathKey(vec![sid_d, sid_root, sid_b])).expect("missing D.Root.B");
        let did_c = *idx.by_path.get(&PathKey(vec![sid_d, sid_root, sid_c])).expect("missing D.Root.C");

        // Assert Index paths match expected DeclIds
        assert_eq!(did_d.0, 0, "D DeclId should be 0");
        assert_eq!(did_root.0, 1, "D.Root DeclId should be 1");
        assert_eq!(did_a.0, 2, "D.Root.A DeclId should be 2");
        assert_eq!(did_b.0, 3, "D.Root.B DeclId should be 3");
        assert_eq!(did_c.0, 4, "D.Root.C DeclId should be 4");
        assert_eq!(idx.by_path.len(), 5, "Index should contain exactly 5 paths");

        // Root must be a node in IR
        let root_nid = ir.decl_to_node[did_root.0 as usize].expect("Root not lowered as node");
        let root_rec = &ir.nodes[root_nid.0 as usize];

        // first_child points to A, and A->B->C via next_sibling
        assert_eq!(ir.first_child(did_root), did_a, "Root.first_child should be Root.A");

        let a_nid = ir.decl_to_node[did_a.0 as usize].expect("A not lowered as node");
        let b_nid = ir.decl_to_node[did_b.0 as usize].expect("B not lowered as node");
        let c_nid = ir.decl_to_node[did_c.0 as usize].expect("C not lowered as node");

        assert_eq!(
            ir.next_sibling(did_a),
            did_b,
            "A.next_sibling should be B"
        );
        assert_eq!(
            ir.next_sibling(did_b),
            did_c,
            "B.next_sibling should be C"
        );
        assert_eq!(
            ir.next_sibling(did_c),
            DeclId::NONE,
            "C.next_sibling should be None"
        );
    }

    #[test]
    fn node_child_chain_order_is_body_order_same_names() {
        // Minimal, self-contained input that creates node→node nesting
        let src = r#"
        test{}
        D {
            Root {
                A {
                  A;
                  B;
                  C;
                }
                B;
                C;
            }
        }
        "#;

        // parse -> intern -> index -> lower
        let ast_str = parse_description(src).expect("parse failed");
        let Interned { ast, mut interner } = intern_ast(&ast_str);
        let idx = build_index_sym(&ast, &interner).expect("index build failed");
        let ir = lower_to_ir(&ast, &idx, &mut interner).expect("lower_to_ir failed");
        // List all elements stored in interner for debugging
        println!("Interner contents:");
        for (sid, s) in interner.iter() {
            println!("  SymId({}): '{}'", sid.0, s);
        }
        // List all paths in the index for debugging
        println!("Index paths:");
        for (path_key, did) in &idx.by_path {
            let path_str = path_key.0.iter().map(|&sid| interner.resolve(sid)).collect::<Vec<_>>().join(".");
            println!("  PathKey({}): '{}'", did.0, path_str);
        }

        // Resolve DeclIds by path: Root, Root.A, Root.B, Root.C
        let sid_root = interner.intern("Root");
        let sid_a = interner.intern("A");
        let sid_b = interner.intern("B");
        let sid_c = interner.intern("C");
        // Print resolved sids for debugging
        println!("Resolved SymIds:");
        println!("  Root: SymId({})", sid_root.0);
        println!("  A: SymId({})", sid_a.0);
        println!("  B: SymId({})", sid_b.0);
        println!("  C: SymId({})", sid_c.0);

        // Assert Interner contents
        assert_eq!(interner.resolve(SymId(0)), "test", "SymId(0) should be 'test'");
        assert_eq!(interner.resolve(SymId(1)), "D", "SymId(1) should be 'D'");
        assert_eq!(interner.resolve(SymId(2)), "Root", "SymId(2) should be 'Root'");
        assert_eq!(interner.resolve(SymId(3)), "A", "SymId(3) should be 'A'");
        assert_eq!(interner.resolve(SymId(4)), "B", "SymId(4) should be 'B'");
        assert_eq!(interner.resolve(SymId(5)), "C", "SymId(5) should be 'C'");

        // Assert resolved SymIds
        assert_eq!(sid_root.0, 2, "Root SymId should be 2");
        assert_eq!(sid_a.0, 3, "A SymId should be 3");
        assert_eq!(sid_b.0, 4, "B SymId should be 4");
        assert_eq!(sid_c.0, 5, "C SymId should be 5");

        // Assert Index paths contain expected DeclIds
        let sid_d = interner.intern("D");
        assert_eq!(sid_d.0, 1, "D SymId should be 1");

        let did_d = *idx.by_path.get(&PathKey(vec![sid_d])).expect("missing D");
        let did_root = *idx.by_path.get(&PathKey(vec![sid_d, sid_root])).expect("missing D.Root");
        let did_root_a = *idx.by_path.get(&PathKey(vec![sid_d, sid_root, sid_a])).expect("missing D.Root.A");
        let did_root_a_a = *idx.by_path.get(&PathKey(vec![sid_d, sid_root, sid_a, sid_a])).expect("missing D.Root.A.A");
        let did_root_a_b = *idx.by_path.get(&PathKey(vec![sid_d, sid_root, sid_a, sid_b])).expect("missing D.Root.A.B");
        let did_root_a_c = *idx.by_path.get(&PathKey(vec![sid_d, sid_root, sid_a, sid_c])).expect("missing D.Root.A.C");
        let did_root_b = *idx.by_path.get(&PathKey(vec![sid_d, sid_root, sid_b])).expect("missing D.Root.B");
        let did_root_c = *idx.by_path.get(&PathKey(vec![sid_d, sid_root, sid_c])).expect("missing D.Root.C");

        // Assert Index paths match expected DeclIds
        assert_eq!(did_d.0, 0, "D DeclId should be 0");
        assert_eq!(did_root.0, 1, "D.Root DeclId should be 1");
        assert_eq!(did_root_a.0, 2, "D.Root.A DeclId should be 2");
        assert_eq!(did_root_a_a.0, 3, "D.Root.A.A DeclId should be 3");
        assert_eq!(did_root_a_b.0, 4, "D.Root.A.B DeclId should be 4");
        assert_eq!(did_root_a_c.0, 5, "D.Root.A.C DeclId should be 5");
        assert_eq!(did_root_b.0, 6, "D.Root.B DeclId should be 6");
        assert_eq!(did_root_c.0, 7, "D.Root.C DeclId should be 7");
        assert_eq!(idx.by_path.len(), 8, "Index should contain exactly 8 paths");

        // Root must be a node in IR
        let root_nid = ir.decl_to_node[did_root.0 as usize].expect("Root not lowered as node");
        let root_rec = &ir.nodes[root_nid.0 as usize];

        // first_child points to Root.A, and Root.A->Root.B->Root.C via next_sibling
        assert_eq!(ir.first_child(did_root), did_root_a, "Root.first_child should be Root.A");

        let root_a_nid = ir.decl_to_node[did_root_a.0 as usize].expect("Root.A not lowered as node");
        let root_b_nid = ir.decl_to_node[did_root_b.0 as usize].expect("Root.B not lowered as node");
        let root_c_nid = ir.decl_to_node[did_root_c.0 as usize].expect("Root.C not lowered as node");

        assert_eq!(
            ir.next_sibling(did_root_a),
            did_root_b,
            "Root.A.next_sibling should be Root.B"
        );
        assert_eq!(
            ir.next_sibling(did_root_b),
            did_root_c,
            "Root.B.next_sibling should be Root.C"
        );
        assert_eq!(
            ir.next_sibling(did_root_c),
            DeclId::NONE,
            "Root.C.next_sibling should be None"
        );

        // Root.A must also have children: Root.A.A->Root.A.B->Root.A.C
        let root_a_rec = &ir.nodes[root_a_nid.0 as usize];
        assert_eq!(ir.first_child(did_root_a), did_root_a_a, "Root.A.first_child should be Root.A.A");

        let root_a_a_nid = ir.decl_to_node[did_root_a_a.0 as usize].expect("Root.A.A not lowered as node");
        let root_a_b_nid = ir.decl_to_node[did_root_a_b.0 as usize].expect("Root.A.B not lowered as node");
        let root_a_c_nid = ir.decl_to_node[did_root_a_c.0 as usize].expect("Root.A.C not lowered as node");
        assert_ne!(root_a_a_nid, root_a_b_nid);
        assert_ne!(root_a_b_nid, root_a_c_nid);

        assert_eq!(
            ir.next_sibling(did_root_a_a),
            did_root_a_b,
            "Root.A.A.next_sibling should be Root.A.B"
        );
        assert_eq!(
            ir.next_sibling(did_root_a_b),
            did_root_a_c,
            "Root.A.B.next_sibling should be Root.A.C"
        );
        assert_eq!(
            ir.next_sibling(did_root_a_c),
            DeclId::NONE,
            "Root.A.C.next_sibling should be None"
        );
        let kids_root: Vec<_> = ir.children(did_root).collect();
        assert_eq!(kids_root, vec![did_root_a, did_root_b, did_root_c]);

        let kids_a: Vec<_> = ir.children(did_root_a).collect();
        assert_eq!(kids_a, vec![did_root_a_a, did_root_a_b, did_root_a_c]);
    }

    #[test]
    fn edge_children_include_arcs_as_decls() {
        let src = r#"
        edge_example{}
        D {
            Root0 {}
            Root {
                @E { (+Root, -Root0); }
            }
        }
        "#;

        let ast_str = parse_description(src).expect("parse failed");
        let Interned { ast, mut interner } = intern_ast(&ast_str);
        let idx = build_index_sym(&ast, &interner).expect("index build failed");
        let ir = lower_to_ir(&ast, &idx, &mut interner).expect("lower_to_ir failed");


        let sid_d = interner.intern("D");
        let sid_root = interner.intern("Root");
        let sid_root0 = interner.intern("Root0");
        let sid_e    = interner.intern("E");
        let did_root  = *idx.by_path.get(&PathKey(vec![sid_d, sid_root]))
            .expect("missing D.Root");
        let did_root0 = *idx.by_path.get(&PathKey(vec![sid_d, sid_root0]))
            .expect("missing D.Root0");
        let did_edge = *idx.by_path.get(&PathKey(vec![sid_d, sid_root, sid_e])).unwrap();

        let kids: Vec<DeclId> = ir.decl_children(did_edge).collect();

        // 1) Edge children (decl-level) contains exactly one Arc decl
        let kids: Vec<DeclId> = ir.decl_children(did_edge).collect();
        assert_eq!(kids.len(), 1, "edge should have exactly one decl-child (the arc)");

        let arc_decl = kids[0];
        assert_eq!(ir.decl_nodes[arc_decl.0 as usize].kind, DeclKind::HyperArc, "child should be Arc");

        // 2) Downcast: Arc decl -> ArcId -> ArcRec
        let arc_id = ir.decl_to_arc[arc_decl.0 as usize]
            .expect("arc decl should map to ArcId via decl_to_arc");
        let arc = &ir.arcs[arc_id.0 as usize];

        // Arc decl konzisztencia (ArcRec-ben nincs `decl`, ezért mappinget tesztelünk)
        assert_eq!(ir.decl_nodes[arc_decl.0 as usize].kind, DeclKind::HyperArc, "child decl should be Arc");
        assert_eq!(ir.decl_nodes[arc_decl.0 as usize].parent, did_edge, "Arc decl parent should be the edge");
        let arc_id = ir.decl_to_arc[arc_decl.0 as usize]
            .expect("arc decl should map to ArcId via decl_to_arc");
        let arc = &ir.arcs[arc_id.0 as usize];

        assert_eq!(arc.in_edge, did_edge, "ArcRec.in_edge should be the edge decl");

        // Refs: (+Root, -Root0) sorrendben
        assert_eq!(arc.refs.len(), 2, "arc should have 2 refs");

        assert_eq!(ref_sign(&arc.refs[0]),  1, "first ref should be +");
        assert_eq!(ref_target(&arc.refs[0]), did_root, "first ref should target Root");

        assert_eq!(ref_sign(&arc.refs[1]), -1, "second ref should be -");
        assert_eq!(ref_target(&arc.refs[1]), did_root0, "second ref should target Root0");
    }
}