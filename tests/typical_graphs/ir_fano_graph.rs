#!cfg[(test)]
mod ir_fano_graph {
    use std::fs::File;
    use memmap2::Mmap;
    use hymeko_framework::common::ids::{DeclId, SymId};
    use hymeko_framework::common::pathkey::PathKey;
    use hymeko_framework::ir::common::ref_target;
    use hymeko_framework::ir::lower::lower_to_ir;
    use hymeko_framework::resolution::intern_pass::Interned;
    use hymeko_framework::resolution::interner::Interner;
    use hymeko_framework::resolution::{intern_pass, resolve};
    use parser::parse_from_mmap;

    fn invert_index(
        idx: &resolve::Index,
        it: &Interner,
    ) -> std::collections::HashMap<DeclId, String> {
        let mut inv = std::collections::HashMap::new();

        for (k, did) in idx.by_path.iter() {
            // k: &PathKey
            let s = k.0
                .iter()
                .map(|&sid| it.resolve(sid).to_string())
                .collect::<Vec<_>>()
                .join(".");
            inv.insert(*did, s);
        }

        inv
    }

    // --- segéd: edge DeclId lekérése path alapján ---
    fn did_of_path(
        idx: &resolve::Index,
        path: &[SymId],
    ) -> DeclId {
        *idx.by_path.get(&PathKey(path.to_vec()))
            .unwrap_or_else(|| panic!("Missing DeclId for path: {:?}", path))
    }


    #[test]
    fn fano_graph_lowers_to_ir_with_correct_arc_targets() -> Result<(), Box<dyn std::error::Error>> {
        // 1) parse -> AST<String>
        let path = "./data/typical_graphs/fano_graph.hymeko";
        let file = File::open(path).unwrap();
        let mmap = unsafe { Mmap::map(&file)? };

        // The AST is valid as long as 'mmap' is in scope.
        let d_str = parse_from_mmap(&mmap).unwrap();

        // 2) intern -> AST<SymId> + interner
        let Interned { ast, mut interner } = intern_pass::intern_ast(&d_str);        

        // 3) resolve index (PathKey -> DeclId)
        let idx = resolve::build_index_sym(&ast, &interner).unwrap();

        // 4) lower -> IR
        let ir = lower_to_ir(&ast, &idx, &mut interner).unwrap();

        // invert index: DeclId -> "fano.n0"
        let inv = invert_index(&idx, &interner);

        // várt incidenciák
        let expected: [(&str, [&str; 3]); 7] = [
            ("e0", ["n0", "n1", "n3"]),
            ("e1", ["n0", "n2", "n6"]),
            ("e2", ["n0", "n4", "n5"]),
            ("e3", ["n1", "n2", "n4"]),
            ("e4", ["n2", "n3", "n5"]),
            ("e5", ["n3", "n4", "n6"]),
            ("e6", ["n1", "n5", "n6"]),
        ];

        let fano_sid = interner.get_id("fano").expect("missing SymId for 'fano'");

        for (ename, nodes) in expected {
            let e_sid = interner.get_id(ename).unwrap_or_else(|| panic!("missing SymId for '{ename}'"));

            // edge DeclId = path [fano, eX]
            let edge_did = did_of_path(&idx, &[fano_sid, e_sid]);

            // IR: DeclId -> EdgeId -> EdgeRec
            let edge_id = ir.decl_to_edge[edge_did.0 as usize]
                .unwrap_or_else(|| panic!("Edge DeclId {:?} not mapped to EdgeId in IR", edge_did));

            let edge = &ir.edges[edge_id.0 as usize];

            // Print edge info  fordebugging
            println!("Checking edge '{ename}' (DeclId: {:?}, EdgeId: {:?})", edge_did, edge_id);
            // Print arc info
            for (i, arc_ref) in edge.arcs.iter().enumerate() {
                let arc_did = ir.decl_to_arc.iter().position(|&aid| aid == Some(*arc_ref))
                    .map(|i| DeclId(i as u32))
                    .unwrap_or_else(|| panic!("ArcId {:?} not mapped to DeclId in IR", arc_ref));
                println!("  Arc {i}: DeclId: {:?}, Parent Edge DeclId: {:?}", arc_did, edge_did);
                // Print arc refs
                let arc = &ir.arcs[arc_ref.0 as usize];
                for (j, r) in arc.refs.iter().enumerate() {
                    let target_did = ref_target(r);
                    let target_name = inv.get(&target_did)                        .cloned()
                        .unwrap_or_else(|| format!("<unknown {target_did:?}>"));
                    println!("    Ref {j}: target DeclId: {:?}, name: {}", target_did, target_name);
                }

            }


            assert_eq!(edge.arcs.len(), 1, "{ename}: expected exactly 1 arc");

            let arc = &ir.arcs[edge.arcs[0].0 as usize];

            // targetok (DeclId) -> "fano.nK"
            let mut got = arc.refs
                .iter()
                .map(|r| {
                    let did = ref_target(r);
                    inv.get(&did)
                        .cloned()
                        .unwrap_or_else(|| format!("<unknown {did:?}>"))
                })
                .collect::<Vec<_>>();
            got.sort();

            let mut exp = nodes.iter().map(|n| format!("fano.{n}")).collect::<Vec<_>>();
            exp.sort();

            assert_eq!(got, exp, "{ename}: IR arc targets mismatch");
        }

        Ok(())
    }
}