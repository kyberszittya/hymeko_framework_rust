use hymeko::body;
use parser::ast::*;
use super::constants::*;
use super::helpers::assert_expected_fields;

#[test]
fn parses_legacy_context_with_refs() {
    assert_field_ref_fixture(MINIMAL_FIELDS_REF_PATH);
}

#[test]
fn parses_legacy_context_with_refs_alternative() {
    assert_field_ref_fixture(MINIMAL_FIELDS_REF_ALT_PATH);
}

fn assert_field_ref_fixture(path: &str) {
    let source_code = parser::read_source_file(path).expect("failed to read source file");
    let d = parser::parse_description(&source_code).unwrap();

    assert_eq!(d.name, DESC_MINIMAL_EXAMPLE_NAME);
    assert_eq!(d.items.len(), 1);
    let context = match &d.items[0] {
        HyperItem::Node(n) => n,
        _ => panic!("Expected Node(context)"),
    };
    assert_eq!(context.inner.name, CONTEXT_NODE_NAME);

    let ctx = body(context);
    assert_expected_fields(ctx, FIELD_REF_EXPECTATIONS);
}