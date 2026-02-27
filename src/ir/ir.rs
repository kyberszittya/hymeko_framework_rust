use crate::ir::hash::HashId;
use crate::common::ids::{HyperArcId, DeclId, EdgeId, NodeId, SymId};
use crate::ir::meta::Meta;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeclKind { Node, Edge, HyperArc }

#[derive(Debug, Clone)]
pub struct DeclNode {
    pub kind: DeclKind,
    pub name: SymId,
    pub parent: DeclId,
    pub first_child: DeclId,
    pub next_sibling: DeclId,
    pub anno: AnnoR,
}

#[derive(Debug, Default)]
pub struct Ir {
    pub meta: Option<Meta>,
    pub doc_hash: Option<HashId>,
    pub decl_hash: Vec<Option<HashId>>,
    pub arc_hash: Vec<Option<HashId>>,
    // deklarációk (globális nézet)
    pub decl_kind: Vec<DeclKind>,     // DeclId -> kind
    pub decl_name: Vec<SymId>,        // DeclId -> name symbol (nem String!)
    pub decl_parent: Vec<DeclId>, // hierarchia/scope (opcionális, de hasznos)
    pub decl_first_child: Vec<DeclId>,
    pub decl_next_sibling: Vec<DeclId>,
    pub decl_anno: Vec<AnnoR>,

    // “konkrét” táblák
    pub nodes: Vec<NodeRec>,          // NodeId indexel
    pub edges: Vec<EdgeRec>,          // EdgeId indexel
    pub arcs: Vec<ArcRec>,            // ArcId indexel

    // DeclId -> NodeId/EdgeId leképzés (gyors jump)
    pub decl_to_node: Vec<Option<NodeId>>,
    pub decl_to_edge: Vec<Option<EdgeId>>,
    pub decl_to_arc: Vec<Option<HyperArcId>>,
}


impl Ir {
    pub fn new(meta: Meta) -> Self {
        Self {
            meta: Some(meta),
            ..Default::default()
        }
    }

    pub fn decl_children(&self, parent: DeclId) -> DeclChildren<'_> {
        DeclChildren { ir: self, next: self.decl_first_child[parent.0 as usize] }
    }

    pub fn decl_kind(&self, did: DeclId) -> DeclKind {
        self.decl_kind[did.0 as usize]
    }

    #[inline]
    pub fn first_child(&self, item: DeclId) -> DeclId {
        self.decl_first_child[item.0 as usize]
    }

    #[inline]
    pub fn next_sibling(&self, item: DeclId) -> DeclId {
        self.decl_next_sibling[item.0 as usize]
    }

    #[inline]
    pub fn parent(&self, item: DeclId) -> DeclId {
        self.decl_parent[item.0 as usize]
    }

    #[inline]
    pub fn kind(&self, item: DeclId) -> DeclKind {
        self.decl_kind[item.0 as usize]
    }

    /// Children iterator for *any* item (node/edge/arc).
    pub fn children(&self, parent: DeclId) -> DeclChildren<'_> {
        DeclChildren { ir: self, next: self.first_child(parent) }
    }

    pub fn as_node(&self, d: DeclId) -> Option<NodeId> { self.decl_to_node[d.0 as usize] }
    pub fn as_edge(&self, d: DeclId) -> Option<EdgeId> { self.decl_to_edge[d.0 as usize] }
    pub fn as_arc(&self, d: DeclId)  -> Option<HyperArcId>  { self.decl_to_arc[d.0 as usize] }
}

pub struct DeclChildren<'a> {
    ir: &'a Ir,
    next: DeclId,
}

impl<'a> Iterator for DeclChildren<'a> {
    type Item = DeclId;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next.is_none() { return None; }
        let cur = self.next;
        self.next = self.ir.next_sibling(cur);
        Some(cur)
    }
}

#[derive(Debug)]
pub struct NodeRec {
    pub decl: DeclId,
    pub bases: Vec<SignedRefR>,
}
impl NodeRec {
    pub fn new(decl: DeclId, bases: Vec<SignedRefR>) -> Self { Self { decl, bases } }
}

#[derive(Debug)]
pub struct EdgeRec {
    pub decl: DeclId,
    pub bases: Vec<SignedRefR>,
    pub arcs: Vec<HyperArcId>,
}
impl EdgeRec {
    pub fn new(decl: DeclId, bases: Vec<SignedRefR>) -> Self {
        Self { decl, bases, arcs: Vec::new() }
    }
}

#[derive(Debug, Clone)]
pub struct ArcRec {
    pub anno: AnnoR,
    pub in_edge: DeclId,              // melyik edge scope-jában volt (általában EdgeDecl)
    pub refs: Vec<SignedRefR>,        // DeclId-kra mutat (determinista)
    // később: attribútumok/value külön, lásd lent
}

#[derive(Debug, Clone, PartialEq)]
pub struct RefAtomR {
    pub target: DeclId,
    pub anno: AnnoR,
    pub weights: Option<Vec<ValueR>>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SignedRefR {
    Plus(RefAtomR),
    Minus(RefAtomR),
    Neutral(RefAtomR),
}

#[derive(Debug, Clone, PartialEq)]
pub enum ValueR {
    Str(SymId),
    Num(f64),
    List(Vec<ValueR>),
    Ref(DeclId), // AST Ref(path) -> IR Ref(DeclId)
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct AnnoR {
    pub tags: Vec<SymId>,
    pub value: Option<ValueR>,
}