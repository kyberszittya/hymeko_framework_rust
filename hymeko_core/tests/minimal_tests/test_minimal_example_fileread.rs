use parser::ast::*;
use super::constants::*;


#[test]
fn test_minimal_example() {
    let source_code = parser::read_source_file(MINIMAL_EXAMPLE_PATH).expect("failed to read source file");

    // 2. Parse it, tying the AST lifetimes to the String
    let d = parser::parse_description(&source_code).unwrap();

    assert_eq!(d.name, DESC_MINIMAL_EXAMPLE_NAME);
    /*
    assert_eq!(d.meta.inner.name, "Minimal_Example");
    assert!(d.meta.inner.body.is_some());

     */

    assert_eq!(d.items.len(), 1);
    match &d.items[0] {
        HyperItem::Node(n) => assert_eq!(n.inner.name, CONTEXT_NODE_NAME),
        _ => panic!("Expected Node(context)"),
    }
}

#[test]
fn test_minimal_example_base_elements() {
    let source_code = parser::read_source_file(MINIMAL_EXAMPLE_BASE_ELEMENTS_PATH).expect("failed to read source file");

    // 2. Parse it, tying the AST lifetimes to the String
    let d = parser::parse_description(&source_code).unwrap();

    assert_eq!(d.name, DESC_MY_DESC);

    assert_eq!(d.items.len(), BASE_ITEM_COUNT);

    // Hypernode A
    match &d.items[0] {
        HyperItem::Node(n) => assert_eq!(n.inner.name, NODE_A_NAME),
        _ => panic!("Expected Node(A)"),
    }

    // Hypernode B
    match &d.items[1] {
        HyperItem::Node(n) => assert_eq!(n.inner.name, NODE_B_NAME),
        _ => panic!("Expected Node(B)"),
    }

    // Edge E1: arcs már a body-ban vannak
    match &d.items[2] {
        HyperItem::Edge(e) => {
            assert_eq!(e.inner.name, EDGE_E1_NAME);

            // szedjük ki az arcokat a body-ból
            let arcs: Vec<&HyperArc<&str>> = e.inner.body.iter().filter_map(|it| {
                if let HyperItem::Arc(a) = it { Some(a) } else { None }
            }).collect();

            assert_eq!(arcs.len(), SINGLE_ARC_COUNT);
            assert_eq!(arcs[0].inner.refs.len(), ARC_REF_PAIR_COUNT);
        }
        _ => panic!("Expected Edge(E1)"),
    }
}