use crate::ast::{AstSym, HyperItem};
use crate::interner::Interner;
use crate::common::ids::{ArcId, DeclId, EdgeId, NodeId, SymId};
use crate::common::pathkey::PathKey;
use crate::ir::ir::{ArcRec, DeclKind, EdgeRec, Ir, NodeRec, SignedRefR};
use crate::ir::meta::Meta;
use crate::ir::time::now_ns;
use crate::resolve::{self, Index, ResolveError};


fn decl_id_of(idx: &Index, path: &[SymId]) -> DeclId {
    *idx.by_path.get(&PathKey(path.to_vec()))
        .unwrap_or_else(|| panic!("Missing DeclId for path {:?}", path))
}

fn ensure_decl_capacity(ir: &mut Ir, did: DeclId) {
    let i = did.0 as usize;
    let need = i + 1;
    if ir.decl_kind.len() < need {
        ir.decl_kind.resize(need, DeclKind::Node);
        ir.decl_name.resize(need, SymId(0));
        ir.decl_parent.resize(need, None);
        ir.decl_to_node.resize(need, None);
        ir.decl_to_edge.resize(need, None);
        ir.decl_hash.resize(need, None); // <-- EZ HIÁNYZOTT tipikusan
    }
}

pub fn lower_to_ir(ast: &AstSym, idx: &Index, it: &Interner) -> Result<Ir, ResolveError> {
    let mut ir = Ir::new(Meta { created_at_unix_ns: now_ns() });

    // header nodes (ha van)
    for n in &ast.header {
        lower_node(&mut ir, idx, it, &[], n)?;
    }

    // top-level items
    lower_items(&mut ir, idx, it, &[], &ast.items)?;

    Ok(ir)
}

fn lower_items(
    ir: &mut Ir,
    idx: &Index,
    it: &Interner,
    scope: &[SymId],
    items: &[HyperItem<SymId>],
) -> Result<(), ResolveError> {
    for item in items {
        match item {
            HyperItem::Node(n) => lower_node(ir, idx, it, scope, n)?,
            HyperItem::Edge(e) => lower_edge(ir, idx, it, scope, e)?,
            HyperItem::Arc(a)  => {
                // Arc önmagában csak edge-bodyban legyen; ha mégis top-level, akkor itt is lekezelheted,
                // de én inkább hibára futtatnám.
                let _ = a; // ha nem kell most
            }
        }
    }
    Ok(())
}

fn lower_node(
    ir: &mut Ir,
    idx: &Index,
    _it: &Interner,
    scope: &[SymId],
    n: &crate::ast::NodeDecl<SymId>,
) -> Result<(), ResolveError> {
    // Fully qualified path = scope + [name_sym]
    let mut path = scope.to_vec();
    path.push(n.inner.name);

    let did = decl_id_of(idx, &path);
    ensure_decl_capacity(ir, did);

    ir.decl_kind[did.0 as usize] = DeclKind::Node;
    ir.decl_name[did.0 as usize] = n.inner.name;
    ir.decl_parent[did.0 as usize] = scope.last().copied().map(|_| {
        // parent DeclId = scope path itself
        let parent_did = decl_id_of(idx, scope);
        parent_did
    });

    // NodeId kiosztás (tömör)
    let nid = NodeId(ir.nodes.len() as u32);
    ir.nodes.push(NodeRec::new(did));
    ir.decl_to_node[did.0 as usize] = Some(nid);

    // children
    if let Some(body) = &n.inner.body {
        lower_items(ir, idx, _it, &path, body)?;
    }

    Ok(())
}

fn lower_edge(
    ir: &mut Ir,
    idx: &Index,
    it: &Interner,
    scope: &[SymId],
    e: &crate::ast::EdgeDecl<SymId>,
) -> Result<(), ResolveError> {
    let mut path = scope.to_vec();
    path.push(e.inner.name);

    let did = decl_id_of(idx, &path);
    ensure_decl_capacity(ir, did);

    ir.decl_kind[did.0 as usize] = DeclKind::Edge;
    ir.decl_name[did.0 as usize] = e.inner.name;
    ir.decl_parent[did.0 as usize] = scope.last().copied().map(|_| decl_id_of(idx, scope));

    // EdgeId kiosztás
    let eid = EdgeId(ir.edges.len() as u32);
    ir.edges.push(EdgeRec::new(did));
    ir.decl_to_edge[did.0 as usize] = Some(eid);

    // edge body: Arc-ok kellenek az IR-be
    let edge_decl = did;

    // scope az edge belsejében = path (scope + edge_name)
    for item in &e.inner.body {
        match item {
            HyperItem::Arc(a) => {
                let resolved = resolve::resolve_arc_refs(idx, &path, a, it)?; // Vec<resolve::SignedRefR>
                let refs = resolved.into_iter().map(|r| match r {
                    SignedRefR::Plus(d) => SignedRefR::Plus(d),
                    SignedRefR::Minus(d) => SignedRefR::Minus(d),
                    SignedRefR::Neutral(d) => SignedRefR::Neutral(d),
                }).collect::<Vec<_>>();

                let aid = ArcId(ir.arcs.len() as u32);
                ir.arcs.push(ArcRec { in_edge: edge_decl, refs });

                // hozzákötjük az edge-hez
                ir.edges[eid.0 as usize].arcs.push(aid);
            }
            HyperItem::Node(n) => {
                // ha edge-bodyban engedsz node-ot: akkor ezek is scope alatt legyenek
                lower_node(ir, idx, it, &path, n)?;
            }
            HyperItem::Edge(ed) => {
                lower_edge(ir, idx, it, &path, ed)?;
            }
        }
    }

    Ok(())
}