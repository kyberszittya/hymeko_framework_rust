// src/ast.rs

#[derive(Debug, Clone, PartialEq)]
pub struct Description {
    pub name: String,
    pub header: Vec<NodeDecl>,
    pub items: Vec<HyperItem>,
}
#[derive(Debug, Clone, PartialEq)]
pub enum HyperItem {
    Node(NodeDecl),
    Edge(EdgeDecl),
    Arc(HyperArc),
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Anno {
    pub tags: Vec<String>,        // <...>
    pub value: Option<Value>,     // opcionális érték
}

#[derive(Debug, Clone, PartialEq)]
pub struct HyperAnnotatedElement<T> {
    pub anno: Anno,
    pub inner: T,
}

pub type NodeDecl  = HyperAnnotatedElement<NodeInner>;
pub type EdgeDecl  = HyperAnnotatedElement<EdgeInner>;
pub type HyperArc  = HyperAnnotatedElement<ArcInner>;

#[derive(Debug, Clone, PartialEq)]
pub struct NodeInner {
    pub name: String,
    pub body: Option<Vec<HyperItem>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EdgeInner {
    pub name: String,
    pub body: Vec<HyperItem>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ArcInner {
    pub refs: Vec<SignedRef>,
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
pub struct DirectedRef {
    pub dir: ArcDir,
    pub target: RefAtom,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Ref {
    pub path: Vec<String>
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct RefAnno {
    pub tags: Vec<String>,
    pub value: Option<Value>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RefAtom {
    pub target: Ref,
    pub anno: RefAnno,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SignedRef {
    Plus(RefAtom),
    Minus(RefAtom),
    Neutral(RefAtom)
}

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Str(String),
    Num(f64),
    List(Vec<Value>),
    Ref(Ref),
}