use parser::ast::*;

#[test]
fn ast_variants_are_constructible() {
    // HyperItem::Node
    let n: NodeDecl<String> = HyperAnnotatedElement {
        anno: Anno { tags: Vec::new(), value: None },
        inner: NodeInner { name: "n".to_string(), body: None },
    };
    let _hi = HyperItem::Node(n);

    // HyperItem::Edge
    let e: EdgeDecl<String> = HyperAnnotatedElement {
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
    
}