use crate::common::SymId;
use crate::ir::hash::HashId;
use crate::ir::ids::{ArcId, DeclId, EdgeId, NodeId};
use crate::ir::meta::Meta;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeclKind { Node, Edge }

#[derive(Debug, Default)]
pub struct Ir {
    pub meta: Option<Meta>,
    pub doc_hash: Option<HashId>,
    pub decl_hash: Vec<Option<HashId>>,
    pub arc_hash: Vec<Option<HashId>>,
    // deklarációk (globális nézet)
    pub decl_kind: Vec<DeclKind>,     // DeclId -> kind
    pub decl_name: Vec<SymId>,        // DeclId -> name symbol (nem String!)
    pub decl_parent: Vec<Option<DeclId>>, // hierarchia/scope (opcionális, de hasznos)

    // “konkrét” táblák
    pub nodes: Vec<NodeRec>,          // NodeId indexel
    pub edges: Vec<EdgeRec>,          // EdgeId indexel
    pub arcs: Vec<ArcRec>,            // ArcId indexel

    // DeclId -> NodeId/EdgeId leképzés (gyors jump)
    pub decl_to_node: Vec<Option<NodeId>>,
    pub decl_to_edge: Vec<Option<EdgeId>>,
}



#[derive(Debug)]
pub struct NodeRec {
    pub decl: DeclId,
    pub first_child: Option<DeclId>,
    pub next_sibling: Option<DeclId>,
}
impl NodeRec {
    pub fn new(decl: DeclId) -> Self {
        Self { decl, first_child: None, next_sibling: None }
    }
}

#[derive(Debug)]
pub struct EdgeRec {
    pub decl: DeclId,
    pub arcs: Vec<ArcId>,
}
impl EdgeRec {
    pub fn new(decl: DeclId) -> Self {
        Self { decl, arcs: Vec::new() }
    }
}

#[derive(Debug, Clone)]
pub struct ArcRec {
    pub in_edge: DeclId,              // melyik edge scope-jában volt (általában EdgeDecl)
    pub refs: Vec<SignedRefR>,        // DeclId-kra mutat (determinista)
    // később: attribútumok/value külön, lásd lent
}

#[derive(Debug, Clone, Copy)]
pub enum SignedRefR {
    Plus(DeclId),
    Minus(DeclId),
    Neutral(DeclId),
}