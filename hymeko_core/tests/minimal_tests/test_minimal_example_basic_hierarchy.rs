use hymeko::{as_node, body, find_node};
use super::constants::*;

#[test]
fn parses_minimal_example_context_fields() {
    let source_code = parser::read_source_file(MINIMAL_BASIC_HIERARCHY_PATH).expect("failed to read source file");

    // 2. Parse it, tying the AST lifetimes to the String
    let d = parser::parse_description(&source_code).unwrap();
    assert_eq!(d.name, DESC_MINIMAL_EXAMPLE_NAME);

    // Top-level: only context
    assert_eq!(d.items.len(), 1);
    let context = as_node(&d.items[0]).unwrap();
    assert_eq!(context.inner.name, CONTEXT_NODE_NAME);

    // context body: node_lev_0, node_lev_1
    let ctx = body(context);
    assert_eq!(ctx.len(), BASIC_CONTEXT_CHILD_COUNT);

    let lev0 = find_node(ctx, "node_lev_0").unwrap();
    let lev1 = find_node(ctx, "node_lev_1").unwrap();

    // node_lev_0 body: node0 (block), node1; node2; node3;
    let lev0_body = body(lev0);
    assert_eq!(lev0_body.len(), BASIC_LEVEL0_BODY_NAMES.len());
    for (item, expected) in lev0_body.iter().zip(BASIC_LEVEL0_BODY_NAMES.iter()) {
        assert_eq!(as_node(item).unwrap().inner.name, *expected);
    }

    // node0 is a block and contains: node0;
    let node0_block = as_node(&lev0_body[0]).unwrap();
    let node0_body = body(node0_block);
    assert_eq!(node0_body.len(), BASIC_NODE0_CHILD_COUNT);
    let inner_node0 = as_node(&node0_body[0]).unwrap();
    assert_eq!(inner_node0.inner.name, BASIC_LEVEL0_BODY_NAMES[0]);
    assert!(inner_node0.inner.body.is_none(), "inner node0 should be a statement node");

    // node_lev_1 body: node0;
    let lev1_body = body(lev1);
    assert_eq!(lev1_body.len(), BASIC_LEVEL1_BODY_NAMES.len());
    for (item, expected) in lev1_body.iter().zip(BASIC_LEVEL1_BODY_NAMES.iter()) {
        assert_eq!(as_node(item).unwrap().inner.name, *expected);
    }
}