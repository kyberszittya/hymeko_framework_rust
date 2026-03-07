use parser::ast::*;
use super::constants::{MINIMAL_WITH_FIELDS_EXPECTATIONS, MINIMAL_WITH_FIELDS_PATH, CONTEXT_NODE_NAME, DESC_MINIMAL_EXAMPLE_NAME};
use super::helpers::assert_expected_fields;

#[test]
fn parses_minimal_example_context_fields() {
    let source_code = parser::read_source_file(MINIMAL_WITH_FIELDS_PATH).expect("failed to read source file");

    // 2. Parse it, tying the AST lifetimes to the String
    let d = parser::parse_description(&source_code).unwrap();
    // description name
    assert_eq!(d.name, DESC_MINIMAL_EXAMPLE_NAME);
    // top-level: context node
    assert_eq!(d.items.len(), 1);
    let context = match &d.items[0] {
        HyperItem::Node(n) => n,
        _ => panic!("Expected top-level Node(context)"),
    };
    assert_eq!(context.inner.name, CONTEXT_NODE_NAME);

    let ctx_body = context.inner.body.as_ref().expect("context must have a body");
    assert_expected_fields(ctx_body.as_slice(), MINIMAL_WITH_FIELDS_EXPECTATIONS);
}
