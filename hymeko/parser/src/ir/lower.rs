use crate::ast::{AstSym, HyperArc, HyperItem};
use crate::interner::Interner;
use crate::common::ids::{ArcId, DeclId, EdgeId, NodeId, SymId};
use crate::common::pathkey::PathKey;
use crate::ir::ir::{AnnoR, ArcRec, DeclKind, EdgeRec, Ir, NodeRec};
use crate::ir::meta::Meta;
use crate::ir::time::now_ns;
use crate::resolve::{self, Index, ResolveError};


fn decl_id_of(idx: &Index, path: &[SymId]) -> DeclId {
    *idx.by_path.get(&PathKey(path.to_vec()))
        .unwrap_or_else(|| panic!("Missing DeclId for path {:?}", path))
}

fn link_decl_child(ir: &mut Ir, parent: DeclId, child: DeclId) {
    // ha parentnek még nincs gyereke
    let slot = &mut ir.decl_first_child[parent.0 as usize];
    if slot.is_none() {
        *slot = child;
        return;
    }

    // különben a sibling lánc végére fűzzük
    let mut cur = *slot;
    loop {
        let next = ir.decl_next_sibling[cur.0 as usize];
        if next.is_none() {
            ir.decl_next_sibling[cur.0 as usize] = child;
            break;
        }
        cur = next;
    }
}

fn ensure_decl_capacity(ir: &mut Ir, did: DeclId) {
    let i = did.0 as usize;
    let need = i + 1;
    if ir.decl_kind.len() < need {
        ir.decl_kind.resize(need, DeclKind::Node);
        ir.decl_name.resize(need, SymId(0));
        ir.decl_parent.resize(need, DeclId::NONE);

        ir.decl_first_child.resize(need, DeclId::NONE);
        ir.decl_next_sibling.resize(need, DeclId::NONE);

        ir.decl_to_node.resize(need, None);
        ir.decl_to_edge.resize(need, None);
        ir.decl_to_arc.resize(need, None);

        ir.decl_hash.resize(need, None);
        ir.decl_to_arc.resize(need, None);
        ir.decl_anno.resize(need, AnnoR::default());
    }
}

pub fn lower_to_ir(ast: &AstSym, idx: &Index, it: &mut Interner) -> Result<Ir, ResolveError> {
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
    it: &mut Interner,
    scope: &[SymId],
    items: &[HyperItem<SymId>],
) -> Result<(), ResolveError> {
    for item in items {
        match item {
            HyperItem::Node(n) => lower_node(ir, idx, it, scope, n)?,
            HyperItem::Edge(e) => lower_edge(ir, idx, it, scope, e)?,
            HyperItem::Arc(a)  => {let _ = a;   }
        }
    }
    Ok(())
}

fn lower_node(
    ir: &mut Ir,
    idx: &Index,
    _it: &mut Interner,
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

    let parent_did = if scope.is_empty() { DeclId::NONE } else { decl_id_of(idx, scope) };
    ir.decl_parent[did.0 as usize] = parent_did;

    let anno = resolve::resolve_anno(idx, &path, &n.anno, _it)?;
    ir.decl_anno[did.0 as usize] = anno;

    let nid = NodeId(ir.nodes.len() as u32);
    ir.nodes.push(NodeRec::new(did));
    ir.decl_to_node[did.0 as usize] = Some(nid);

    if parent_did.is_some() {
        link_decl_child(ir, parent_did, did);
    }

    // children
    if let Some(body) = &n.inner.body {
        lower_items(ir, idx, _it, &path, body)?;
    }

    Ok(())
}

fn lower_arc(
    ir: &mut Ir,
    idx: &Index,
    it: &mut Interner,
    edge_decl: DeclId,
    path: &[SymId],
    eid: EdgeId,
    a: &HyperArc<SymId>,

) -> Result<(), ResolveError> {
    // 1) allocate a fresh anonymous DeclId for the arc
    let arc_decl = DeclId(ir.decl_kind.len() as u32);
    ensure_decl_capacity(ir, arc_decl);

    // 2) fill decl tables
    ir.decl_kind[arc_decl.0 as usize] = DeclKind::Arc;
    ir.decl_name[arc_decl.0 as usize] = SymId(0); // névtelen arc
    ir.decl_parent[arc_decl.0 as usize] = edge_decl;

    // 3) resolve arc payload once
    let anno = resolve::resolve_anno(idx, &path, &a.anno, it)?;
    let refs = resolve::resolve_arc_refs(idx, &path, a, it)?;

    // 4) store decl anno (decl-level canonical place)
    ir.decl_anno[arc_decl.0 as usize] = anno.clone();

    // 5) create ArcRec + mapping
    let aid = ArcId(ir.arcs.len() as u32);
    ir.arcs.push(ArcRec { anno, in_edge: edge_decl, refs });
    ir.decl_to_arc[arc_decl.0 as usize] = Some(aid);

    // 6) link into edge's decl-children chain (for traversal)
    link_decl_child(ir, edge_decl, arc_decl);

    // 7) keep per-edge arc list once
    ir.edges[eid.0 as usize].arcs.push(aid);

    Ok(())
}

fn lower_edge(
    ir: &mut Ir,
    idx: &Index,
    it: &mut Interner,
    scope: &[SymId],
    e: &crate::ast::EdgeDecl<SymId>,
) -> Result<(), ResolveError> {
    let mut path = scope.to_vec();
    path.push(e.inner.name);

    let did = decl_id_of(idx, &path);
    ensure_decl_capacity(ir, did);

    ir.decl_kind[did.0 as usize] = DeclKind::Edge;
    ir.decl_name[did.0 as usize] = e.inner.name;

    let parent_did = if scope.is_empty() { DeclId::NONE } else { decl_id_of(idx, scope) };
    ir.decl_parent[did.0 as usize] = parent_did;

    let anno = resolve::resolve_anno(idx, &path, &e.anno, it)?;
    ir.decl_anno[did.0 as usize] = anno;

    // EdgeId kiosztás
    let eid = EdgeId(ir.edges.len() as u32);
    ir.edges.push(EdgeRec::new(did));
    ir.decl_to_edge[did.0 as usize] = Some(eid);
    if parent_did.is_some() {
        link_decl_child(ir, parent_did, did);
    }

    let edge_decl = did;

    for item in &e.inner.body {
        match item {
            HyperItem::Arc(a) => {
                lower_arc(ir, idx, it, edge_decl, &path, eid, a)?;
            }
            HyperItem::Node(n) => {
                lower_node(ir, idx, it, &path, n)?;
            }
            HyperItem::Edge(ed) => {
                lower_edge(ir, idx, it, &path, ed)?;
            }
        }
    }

    Ok(())
}