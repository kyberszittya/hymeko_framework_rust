use hymeko::{as_node, body, find_node};

#[test]
fn parses_minimal_example_context_fields() {
    let path = "./data/minimal_examples/minimal_example_basic_hierarchy.hymeko";
    let source_code = parser::read_source_file(&path).expect("failed to read source file");

    // 2. Parse it, tying the AST lifetimes to the String
    let d = parser::parse_description(&source_code).unwrap();
    

    // Header
    assert_eq!(d.name, "Minimal_Example");

    // Top-level: only context
    assert_eq!(d.items.len(), 1);
    let context = as_node(&d.items[0]).unwrap();
    assert_eq!(context.inner.name, "context");

    // context body: node_lev_0, node_lev_1
    let ctx = body(context);
    assert_eq!(ctx.len(), 2);

    let lev0 = find_node(ctx, "node_lev_0").unwrap();
    let lev1 = find_node(ctx, "node_lev_1").unwrap();

    // node_lev_0 body: node0 (block), node1; node2; node3;
    let lev0_body = body(lev0);
    assert_eq!(lev0_body.len(), 4);
    assert_eq!(as_node(&lev0_body[0]).unwrap().inner.name, "node0");
    assert_eq!(as_node(&lev0_body[1]).unwrap().inner.name, "node1");
    assert_eq!(as_node(&lev0_body[2]).unwrap().inner.name, "node2");
    assert_eq!(as_node(&lev0_body[3]).unwrap().inner.name, "node3");

    // node0 is a block and contains: node0;
    let node0_block = as_node(&lev0_body[0]).unwrap();
    let node0_body = body(node0_block);
    assert_eq!(node0_body.len(), 1);
    let inner_node0 = as_node(&node0_body[0]).unwrap();
    assert_eq!(inner_node0.inner.name, "node0");
    assert!(inner_node0.inner.body.is_none(), "inner node0 should be a statement node");

    // node_lev_1 body: node0;
    let lev1_body = body(lev1);
    assert_eq!(lev1_body.len(), 1);
    assert_eq!(as_node(&lev1_body[0]).unwrap().inner.name, "node0");
}