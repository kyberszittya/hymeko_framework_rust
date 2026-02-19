use crate::ast::*;
use crate::common::ids::SymId;
use crate::interner::Interner;

pub struct Interned {
    pub ast: AstSym,
    pub interner: Interner,
}

pub fn intern_ast(src: &AstStr) -> Interned {
    let mut it = Interner::new();
    let ast = lower_desc(src, &mut it);
    Interned { ast, interner: it }
}

fn sid(it: &mut Interner, s: &str) -> SymId { it.intern(s) }

fn lower_desc(src: &Description<String>, it: &mut Interner) -> Description<SymId> {
    Description {
        name: sid(it, &src.name),
        header: src.header.iter().map(|n| lower_node(n, it)).collect(),
        items: src.items.iter().map(|x| lower_item(x, it)).collect(),
    }
}

fn lower_item(src: &HyperItem<String>, it: &mut Interner) -> HyperItem<SymId> {
    match src {
        HyperItem::Node(n) => HyperItem::Node(lower_node(n, it)),
        HyperItem::Edge(e) => HyperItem::Edge(lower_edge(e, it)),
        HyperItem::Arc(a)  => HyperItem::Arc(lower_arc(a, it)),
    }
}

fn lower_anno(src: &Anno<String>, it: &mut Interner) -> Anno<SymId> {
    Anno {
        tags: src.tags.clone(), // tags marad String (most)
        value: src.value.as_ref().map(|v| lower_value(v, it)),
    }
}

fn lower_node(src: &NodeDecl<String>, it: &mut Interner) -> NodeDecl<SymId> {
    HyperAnnotatedElement {
        anno: lower_anno(&src.anno, it),
        inner: NodeInner {
            name: sid(it, &src.inner.name),
            body: src.inner.body.as_ref().map(|xs| xs.iter().map(|x| lower_item(x, it)).collect()),
        },
    }
}

fn lower_edge(src: &EdgeDecl<String>, it: &mut Interner) -> EdgeDecl<SymId> {
    HyperAnnotatedElement {
        anno: lower_anno(&src.anno, it),
        inner: EdgeInner {
            name: sid(it, &src.inner.name),
            body: src.inner.body.iter().map(|x| lower_item(x, it)).collect(),
        },
    }
}

fn lower_arc(src: &HyperArc<String>, it: &mut Interner) -> HyperArc<SymId> {
    HyperAnnotatedElement {
        anno: lower_anno(&src.anno, it),
        inner: ArcInner {
            refs: src.inner.refs.iter().map(|r| lower_signed_ref(r, it)).collect(),
        },
    }
}

fn lower_signed_ref(src: &SignedRef<String>, it: &mut Interner) -> SignedRef<SymId> {
    match src {
        SignedRef::Plus(a)    => SignedRef::Plus(lower_ref_atom(a, it)),
        SignedRef::Minus(a)   => SignedRef::Minus(lower_ref_atom(a, it)),
        SignedRef::Neutral(a) => SignedRef::Neutral(lower_ref_atom(a, it)),
    }
}

fn lower_ref_atom(src: &RefAtom<String>, it: &mut Interner) -> RefAtom<SymId> {
    RefAtom {
        target: Ref { path: src.target.path.iter().map(|p| sid(it, p)).collect() },
        anno: RefAnno {
            weights: src.anno.weights.as_ref().map(|ws| ws.iter().map(|v| lower_value(v, it)).collect()),
            tags: src.anno.tags.clone(), // marad String
            value: src.anno.value.as_ref().map(|v| lower_value(v, it)),
        },
    }
}

fn lower_value(src: &Value<String>, it: &mut Interner) -> Value<SymId> {
    match src {
        Value::Str(s) => Value::Str(s.clone()),
        Value::Num(n) => Value::Num(*n),
        Value::List(xs) => Value::List(xs.iter().map(|v| lower_value(v, it)).collect()),
        Value::Ref(r) => Value::Ref(Ref { path: r.path.iter().map(|p| sid(it, p)).collect() }),
    }
}