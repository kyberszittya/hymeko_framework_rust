use std::borrow::Cow;

#[derive(Debug, Clone, PartialEq)]
pub struct Description<'a, Id> {
    pub name: Id,
    pub header: Vec<NodeDecl<'a, Id>>,
    pub imports: Vec<ImportStmt<'a, Id>>,
    pub items: Vec<HyperItem<'a, Id>>,
}

pub enum HeaderStmt<'a, S> {
    Node(NodeDecl<'a, S>),
    Import(ImportStmt<'a, S>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum HyperItem<'a, Id> {
    Node(NodeDecl<'a, Id>),
    Edge(EdgeDecl<'a, Id>),
    Arc(HyperArc<'a, Id>),
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Anno<'a, Id> {
    pub tags: Vec<Cow<'a, str>>,        // Zero-copy or minimal allocation
    pub value: Option<Value<'a, Id>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HyperAnnotatedElement<'a, T, Id> {
    pub anno: Anno<'a, Id>,
    pub inner: T,
}

pub type NodeDecl<'a, Id>  = HyperAnnotatedElement<'a, NodeInner<'a, Id>, Id>;
pub type EdgeDecl<'a, Id>  = HyperAnnotatedElement<'a, EdgeInner<'a, Id>, Id>;
pub type HyperArc<'a, Id>  = HyperAnnotatedElement<'a, ArcInner<'a, Id>, Id>;
pub type HeaderBlock<'a> = (Vec<ImportStmt<'a, &'a str>>, Vec<NodeDecl<'a, &'a str>>);

#[derive(Debug, Clone, PartialEq)]
pub struct NodeInner<'a, Id> {
    pub name: Id,
    pub body: Option<Vec<HyperItem<'a, Id>>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EdgeInner<'a, Id> {
    pub name: Id,
    pub body: Vec<HyperItem<'a, Id>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ArcInner<'a, Id> {
    pub refs: Vec<SignedRef<'a, Id>>,
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
pub struct DirectedRef<'a, Id> {
    pub dir: ArcDir,
    pub target: RefAtom<'a, Id>,
    pub weights: Option<Vec<Value<'a, Id>>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Ref<Id> {
    pub path: Vec<Id>
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct RefAnno<'a, Id> {
    pub weights: Option<Vec<Value<'a, Id>>>,
    pub tags: Vec<Cow<'a, str>>,
    pub value: Option<Value<'a, Id>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RefAtom<'a, Id> {
    pub target: Ref<Id>,
    pub anno: RefAnno<'a, Id>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SignedRef<'a, Id> {
    Plus(RefAtom<'a, Id>),
    Minus(RefAtom<'a, Id>),
    Neutral(RefAtom<'a, Id>)
}

#[derive(Debug, Clone, PartialEq)]
pub enum Value<'a, Id> {
    Str(Cow<'a, str>), // Zero-copy for fast-path strings
    Num(f64),
    List(Vec<Value<'a, Id>>),
    Ref(Ref<Id>),
}

// The core type alias now strictly binds to the file buffer's lifetime
pub type AstStr<'a> = Description<'a, &'a str>;


#[derive(Debug, Clone, PartialEq)]
pub struct ImportStmt<'a, Id> {
    pub path: Cow<'a, str>,
    pub alias: Option<Id>,
}
