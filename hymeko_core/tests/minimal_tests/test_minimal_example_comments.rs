use parser::{ast::HyperItem, parse_description};
use super::constants::*;
use super::helpers::assert_expected_fields;

#[test]
fn parses_minimal_example_context_fields_with_comments() {
    assert_comment_fixture(MINIMAL_COMMENTS_WITH_FIELDS_PATH);
}

#[test]
fn parses_minimal_example_context_fields_with_line_comments() {
    assert_comment_fixture(MINIMAL_COMMENTS_WITH_LINE_PATH);
}

#[test]
fn parses_minimal_example_context_fields_with_header_comment() {
    assert_comment_fixture(MINIMAL_COMMENTS_WITH_HEADER_PATH);
}

#[test]
fn parses_minimal_example_context_fields_with_bad_comments() {
    let path = MINIMAL_COMMENTS_WITH_BAD_PATH;
    // This should throw an error
    let source_code = parser::read_source_file(&path).expect("failed to read source file");

    // 2. Parse it, tying the AST lifetimes to the String
    let res = parser::parse_description(&source_code);
    // Expect error
    assert!(res.is_err());

}

#[test]
fn parses_minimal_example_context_fields_with_bad_comments_inline() {
    let res = parse_description(BAD_INLINE_COMMENT_SRC);
    // Expect error
    assert!(res.is_err());


}

fn assert_comment_fixture(path: &str) {
    let source_code = parser::read_source_file(path).expect("failed to read source file");
    let d = parser::parse_description(&source_code).unwrap();

    assert_eq!(d.name, DESC_MINIMAL_EXAMPLE_NAME);
    assert_eq!(d.items.len(), 1);
    let context = match &d.items[0] {
        HyperItem::Node(n) => n,
        _ => panic!("Expected top-level Node(context)"),
    };
    assert_eq!(context.inner.name, CONTEXT_NODE_NAME);

    let ctx_body = context.inner.body.as_ref().expect("context must have a body");
    assert_expected_fields(ctx_body.as_slice(), MINIMAL_WITH_FIELDS_COMMENTS_EXPECTATIONS);
}
