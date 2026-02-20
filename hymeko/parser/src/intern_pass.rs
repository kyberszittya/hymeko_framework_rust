use crate::ast::*;
use crate::common::ids::SymId;
use crate::interner::Interner;

pub struct Interned<'a> {
    pub ast: AstSym<'a>,
    pub interner: Interner,
}

pub fn intern_ast<'a>(src: &AstStr<'a>) -> Interned<'a> {
    let mut it = Interner::new();
    let ast = lower_desc(src, &mut it);
    Interned { ast, interner: it }
}

fn sid(it: &mut Interner, s: &str) -> SymId { it.intern(s) }

fn lower_desc<'a>(src: &Description<'a, &'a str>, it: &mut Interner) -> Description<'a, SymId> {
    Description {
        name: sid(it, src.name),
        header: src.header.iter().map(|n| lower_node(n, it)).collect(),
        items: src.items.iter().map(|x| lower_item(x, it)).collect(),
    }
}

fn lower_item<'a>(src: &HyperItem<'a, &'a str>, it: &mut Interner) -> HyperItem<'a, SymId> {
    match src {
        HyperItem::Node(n) => HyperItem::Node(lower_node(n, it)),
        HyperItem::Edge(e) => HyperItem::Edge(lower_edge(e, it)),
        HyperItem::Arc(a)  => HyperItem::Arc(lower_arc(a, it)),
    }
}

fn lower_anno<'a>(src: &Anno<'a, &'a str>, it: &mut Interner) -> Anno<'a, SymId> {
    Anno {
        tags: src.tags.clone(), // Cheap O(1) pointer clone for Cow
        value: src.value.as_ref().map(|v| lower_value(v, it)),
    }
}

fn lower_node<'a>(src: &NodeDecl<'a, &'a str>, it: &mut Interner) -> NodeDecl<'a, SymId> {
    HyperAnnotatedElement {
        anno: lower_anno(&src.anno, it),
        inner: NodeInner {
            name: sid(it, src.inner.name),
            body: src.inner.body.as_ref().map(|xs| xs.iter().map(|x| lower_item(x, it)).collect()),
        },
    }
}

fn lower_edge<'a>(src: &EdgeDecl<'a, &'a str>, it: &mut Interner) -> EdgeDecl<'a, SymId> {
    HyperAnnotatedElement {
        anno: lower_anno(&src.anno, it),
        inner: EdgeInner {
            name: sid(it, src.inner.name),
            body: src.inner.body.iter().map(|x| lower_item(x, it)).collect(),
        },
    }
}

fn lower_arc<'a>(src: &HyperArc<'a, &'a str>, it: &mut Interner) -> HyperArc<'a, SymId> {
    HyperAnnotatedElement {
        anno: lower_anno(&src.anno, it),
        inner: ArcInner {
            refs: src.inner.refs.iter().map(|r| lower_signed_ref(r, it)).collect(),
        },
    }
}

fn lower_signed_ref<'a>(src: &SignedRef<'a, &'a str>, it: &mut Interner) -> SignedRef<'a, SymId> {
    match src {
        SignedRef::Plus(a)    => SignedRef::Plus(lower_ref_atom(a, it)),
        SignedRef::Minus(a)   => SignedRef::Minus(lower_ref_atom(a, it)),
        SignedRef::Neutral(a) => SignedRef::Neutral(lower_ref_atom(a, it)),
    }
}

fn lower_ref_atom<'a>(src: &RefAtom<'a, &'a str>, it: &mut Interner) -> RefAtom<'a, SymId> {
    RefAtom {
        target: Ref { path: src.target.path.iter().map(|&p| sid(it, p)).collect() },
        anno: RefAnno {
            weights: src.anno.weights.as_ref().map(|ws| ws.iter().map(|v| lower_value(v, it)).collect()),
            tags: src.anno.tags.clone(),
            value: src.anno.value.as_ref().map(|v| lower_value(v, it)),
        },
    }
}

fn lower_value<'a>(src: &Value<'a, &'a str>, it: &mut Interner) -> Value<'a, SymId> {
    match src {
        Value::Str(s) => Value::Str(s.clone()),
        Value::Num(n) => Value::Num(*n),
        Value::List(xs) => Value::List(xs.iter().map(|v| lower_value(v, it)).collect()),
        Value::Ref(r) => Value::Ref(Ref { path: r.path.iter().map(|&p| sid(it, p)).collect() }),
    }
}