use parser::{intern_pass, read_parse_file, resolve};
use parser::common::ids::{DeclId, SymId};
use parser::common::pathkey::PathKey;
use parser::interner::Interner;
use parser::ir::ir::SignedRefR;
use parser::ir::lower::lower_to_ir;


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

// --- segéd: ir SignedRefR -> DeclId ---
fn ref_declid(r: SignedRefR) -> DeclId {
    match r {
        SignedRefR::Plus(d) => d,
        SignedRefR::Minus(d) => d,
        SignedRefR::Neutral(d) => d,
    }
}

#[test]
fn fano_graph_lowers_to_ir_with_correct_arc_targets() -> Result<(), Box<dyn std::error::Error>> {
    // 1) parse -> AST<String>
    let path = "./data/typical_graphs/fano_graph.hymeko";
    let d_str = read_parse_file(&path).unwrap();

    // 2) intern -> AST<SymId> + interner
    let interned = intern_pass::intern_ast(&d_str);
    let ast = &interned.ast;
    let it = &interned.interner;

    // 3) resolve index (PathKey -> DeclId)
    let idx = resolve::build_index_sym(ast, it).unwrap();

    // 4) lower -> IR
    let ir = lower_to_ir(ast, &idx, it).unwrap();

    // invert index: DeclId -> "fano.n0"
    let inv = invert_index(&idx, it);

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

    let fano_sid = it.get_id("fano").expect("missing SymId for 'fano'");

    for (ename, nodes) in expected {
        let e_sid = it.get_id(ename).unwrap_or_else(|| panic!("missing SymId for '{ename}'"));

        // edge DeclId = path [fano, eX]
        let edge_did = did_of_path(&idx, &[fano_sid, e_sid]);

        // IR: DeclId -> EdgeId -> EdgeRec
        let edge_id = ir.decl_to_edge[edge_did.0 as usize]
            .unwrap_or_else(|| panic!("Edge DeclId {:?} not mapped to EdgeId in IR", edge_did));

        let edge = &ir.edges[edge_id.0 as usize];

        // A te inputodban minden edge body-ban 1 arc van
        assert_eq!(edge.arcs.len(), 1, "{ename}: expected exactly 1 arc");

        let arc = &ir.arcs[edge.arcs[0].0 as usize];

        // targetok (DeclId) -> "fano.nK"
        let mut got = arc.refs
            .iter()
            .map(|r| inv[&ref_declid(*r)].clone())
            .collect::<Vec<_>>();
        got.sort();

        let mut exp = nodes.iter().map(|n| format!("fano.{n}")).collect::<Vec<_>>();
        exp.sort();

        assert_eq!(got, exp, "{ename}: IR arc targets mismatch");
    }

    Ok(())
}