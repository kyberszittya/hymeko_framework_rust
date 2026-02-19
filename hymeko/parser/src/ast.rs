// src/ast.rs

use crate::common::ids::SymId;

#[derive(Debug, Clone, PartialEq)]
pub struct Description<Id> {
    pub name: Id,
    pub header: Vec<NodeDecl<Id>>,
    pub items: Vec<HyperItem<Id>>,
}
#[derive(Debug, Clone, PartialEq)]
pub enum HyperItem<Id> {
    Node(NodeDecl<Id>),
    Edge(EdgeDecl<Id>),
    Arc(HyperArc<Id>),
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Anno<Id> {
    pub tags: Vec<String>,        // <...>
    pub value: Option<Value<Id>>,     // opcionális érték
}

#[derive(Debug, Clone, PartialEq)]
pub struct HyperAnnotatedElement<T, Id> {
    pub anno: Anno<Id>,
    pub inner: T,
}

pub type NodeDecl<Id>  = HyperAnnotatedElement<NodeInner<Id>, Id>;
pub type EdgeDecl<Id>  = HyperAnnotatedElement<EdgeInner<Id>, Id>;
pub type HyperArc<Id>  = HyperAnnotatedElement<ArcInner<Id>, Id>;

#[derive(Debug, Clone, PartialEq)]
pub struct NodeInner<Id> {
    pub name: Id,
    pub body: Option<Vec<HyperItem<Id>>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EdgeInner<Id> {
    pub name: Id,
    pub body: Vec<HyperItem<Id>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ArcInner<Id> {
    pub refs: Vec<SignedRef<Id>>,
}

//----
// References
//----

#[derive(Debug, Clone, PartialEq)]
pub enum ArcDir {
    In,   // +
    Out,  // -
    Bi,   // ~
}

#[derive(Debug, Clone, PartialEq)]
pub struct DirectedRef<Id> {
    pub dir: ArcDir,
    pub target: RefAtom<Id>,
    pub weights: Option<Vec<Value<Id>>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Ref<Id> {
    pub path: Vec<Id>
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct RefAnno<Id> {
    pub weights: Option<Vec<Value<Id>>>,
    pub tags: Vec<String>,
    pub value: Option<Value<Id>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RefAtom<Id> {
    pub target: Ref<Id>,
    pub anno: RefAnno<Id>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SignedRef<Id> {
    Plus(RefAtom<Id>),
    Minus(RefAtom<Id>),
    Neutral(RefAtom<Id>)
}

#[derive(Debug, Clone, PartialEq)]
pub enum Value<Id> {
    Str(String),
    Num(f64),
    List(Vec<Value<Id>>),
    Ref(Ref<Id>),
}

pub type AstStr = Description<String>;
pub type AstSym = Description<SymId>;