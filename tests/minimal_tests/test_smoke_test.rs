use parser::ast::*;

fn ra(name: &str) -> RefAtom<'static, String> {
    RefAtom {
        target: Ref { path: vec![name.to_string()] },
        anno: Anno { tags: vec![], value: None },
    }
}

#[test]
fn ast_variants_are_constructible() {
    let n: NodeDecl<'static, String> = HyperAnnotatedElement {
        anno: Anno { tags: Vec::new(), value: None },
        inner: NodeInner { name: "n".to_string(), bases: vec![], body: None },
    };
    let _hi = HyperItem::Node(n);

    // Edge
    let e: EdgeDecl<'static, String> = HyperAnnotatedElement {
        anno: Anno { tags: Vec::new(), value: None },
        inner: EdgeInner { name: "e".to_string(), bases: vec![], body: Vec::new() },
    };
    let _hi = HyperItem::Edge(e);

    // Arc + SignedRef
    let a: HyperArc<'static, String> = HyperAnnotatedElement {
        anno: Anno { tags: Vec::new(), value: Some(Value::Num(0.2)) },
        inner: ArcInner {
            refs: vec![
                SignedRef::Plus(ra("A")),
                SignedRef::Minus(ra("B")),
            ],
        },
    };
    let _hi = HyperItem::Arc(a);
}