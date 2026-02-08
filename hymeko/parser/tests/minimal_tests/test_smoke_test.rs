use parser::ast::*;

#[test]
fn ast_variants_are_constructible() {
    // HyperItem::Node
    let n: NodeDecl = HyperAnnotatedElement {
        anno: Anno { tags: Vec::new(), value: None },
        inner: NodeInner { name: "n".to_string(), body: None },
    };
    let _hi = HyperItem::Node(n);

    // HyperItem::Edge
    let e: EdgeDecl = HyperAnnotatedElement {
        anno: Anno { tags: Vec::new(), value: None },
        inner: EdgeInner { name: "e".to_string(), body: Vec::new() },
    };
    let _hi = HyperItem::Edge(e);

    /*
    // HyperItem::Arc + SignedRef
    let a: HyperArc = HyperAnnotatedElement {
        anno: Anno { tags: Vec::new(), value: Some(Value::Num(0.2)) },
        inner: ArcInner {
            refs: vec![
                SignedRef::Plus("A".to_string()),
                SignedRef::Minus("B".to_string()),
            ],
        },
    };
    let _hi = HyperItem::Arc(a);
    */

    // Value variants
    let _v1 = Value::Str("x".to_string());
    let _v2 = Value::Num(1.0);
    let _v3 = Value::List(vec![Value::Num(1.0), Value::Num(2.0)]);
}