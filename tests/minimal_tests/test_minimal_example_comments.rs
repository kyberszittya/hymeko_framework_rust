use hymeko_framework::{find_node};
use parser::ast::HyperItem;
use parser::{parse_description};
use crate::test_asserts::test_helpers::{assert_list_nums, assert_no_value, assert_num_value, assert_str_value, assert_tags};

#[test]
fn parses_minimal_example_context_fields_with_comments() {
    let path = "./data/minimal_examples/comments/minimal_example_with_fields_with_comments.hymeko";
    let source_code = parser::read_source_file(&path).expect("failed to read source file");

    // 2. Parse it, tying the AST lifetimes to the String
    let d = parser::parse_description(&source_code).unwrap();

    // description name
    assert_eq!(d.name, "Minimal_Example");

    // meta node
    
    // top-level: context node
    assert_eq!(d.items.len(), 1);
    let context = match &d.items[0] {
        HyperItem::Node(n) => n,
        _ => panic!("Expected top-level Node(context)"),
    };
    assert_eq!(context.inner.name, "context");

    let ctx_body = context.inner.body.as_ref().expect("context must have a body");

    // val0 <int> 56;
    let val0 = find_node(ctx_body, "val0").unwrap();
    assert_tags(val0, &["int"]);
    assert_num_value(val0, 56.0);

    // val1 <string> "vakond";
    let val1 = find_node(ctx_body, "val1").unwrap();
    assert_tags(val1, &["string"]);
    assert_str_value(val1, "vakond");

    // val2 <real> 56.891;
    let val2 = find_node(ctx_body, "val2").unwrap();
    assert_tags(val2, &["real"]);
    assert_num_value(val2, 56.891);

    // val_undef <real>;
    let val_undef = find_node(ctx_body, "val_undef").unwrap();
    assert_tags(val_undef, &["real"]);
    assert_no_value(val_undef);

    // val3 3444.4623;
    let val3 = find_node(ctx_body, "val3").unwrap();
    assert_tags(val3, &[]);
    assert_num_value(val3, 3444.4623);

    // pi 3.14156;
    let pi = find_node(ctx_body, "pi").unwrap();
    assert_tags(pi, &[]);
    assert_num_value(pi, 3.14156);

    // val_float;
    let val_float = find_node(ctx_body, "val_float").unwrap();
    assert_tags(val_float, &[]);
    assert_no_value(val_float);

    // vector [..];
    let vector = find_node(ctx_body, "vector").unwrap();
    assert_tags(vector, &[]);
    assert_list_nums(vector, &[15.6, 17.8, 16.3, 12.3, 67.8, 45.0, 2.0]);
}

#[test]
fn parses_minimal_example_context_fields_with_line_comments() {
    let path = "./data/minimal_examples/comments/minimal_example_with_fields_with_line_comments.hymeko";
    let source_code = parser::read_source_file(&path).expect("failed to read source file");

    // 2. Parse it, tying the AST lifetimes to the String
    let d = parser::parse_description(&source_code).unwrap();

    // description name
    assert_eq!(d.name, "Minimal_Example");

    // meta node
    /*
    assert_eq!(d.meta.inner.name, "Minimal_Example");
    let meta_body = d.meta.inner.body.as_ref().expect("meta must have a body");
    let author = find_node(meta_body, "author");
    assert_str_value(author, "Csaba");

     */

    // top-level: context node
    assert_eq!(d.items.len(), 1);
    let context = match &d.items[0] {
        HyperItem::Node(n) => n,
        _ => panic!("Expected top-level Node(context)"),
    };
    assert_eq!(context.inner.name, "context");

    let ctx_body = context.inner.body.as_ref().expect("context must have a body");

    // val0 <int> 56;
    let val0 = find_node(ctx_body, "val0").unwrap();
    assert_tags(val0, &["int"]);
    assert_num_value(val0, 56.0);

    // val1 <string> "vakond";
    let val1 = find_node(ctx_body, "val1").unwrap();
    assert_tags(val1, &["string"]);
    assert_str_value(val1, "vakond");

    // val2 <real> 56.891;
    let val2 = find_node(ctx_body, "val2").unwrap();
    assert_tags(val2, &["real"]);
    assert_num_value(val2, 56.891);

    // val_undef <real>;
    let val_undef = find_node(ctx_body, "val_undef").unwrap();
    assert_tags(val_undef, &["real"]);
    assert_no_value(val_undef);

    // val3 3444.4623;
    let val3 = find_node(ctx_body, "val3").unwrap();
    assert_tags(val3, &[]);
    assert_num_value(val3, 3444.4623);

    // pi 3.14156;
    let pi = find_node(ctx_body, "pi").unwrap();
    assert_tags(pi, &[]);
    assert_num_value(pi, 3.14156);

    // val_float;
    let val_float = find_node(ctx_body, "val_float").unwrap();
    assert_tags(val_float, &[]);
    assert_no_value(val_float);

    // vector [..];
    let vector = find_node(ctx_body, "vector").unwrap();
    assert_tags(vector, &[]);
    assert_list_nums(vector, &[15.6, 17.8, 16.3, 12.3, 67.8, 45.0, 2.0]);
}

#[test]
fn parses_minimal_example_context_fields_with_header_comment() {
    let path = "./data/minimal_examples/comments/minimal_example_with_fields_with_block_header_comment.hymeko";
    let source_code = parser::read_source_file(&path).expect("failed to read source file");

    // 2. Parse it, tying the AST lifetimes to the String
    let d = parser::parse_description(&source_code).unwrap();

    // description name
    assert_eq!(d.name, "Minimal_Example");

    
    // top-level: context node
    assert_eq!(d.items.len(), 1);
    let context = match &d.items[0] {
        HyperItem::Node(n) => n,
        _ => panic!("Expected top-level Node(context)"),
    };
    assert_eq!(context.inner.name, "context");

    let ctx_body = context.inner.body.as_ref().expect("context must have a body");

    // val0 <int> 56;
    let val0 = find_node(ctx_body, "val0").unwrap();
    assert_tags(val0, &["int"]);
    assert_num_value(val0, 56.0);

    // val1 <string> "vakond";
    let val1 = find_node(ctx_body, "val1").unwrap();
    assert_tags(val1, &["string"]);
    assert_str_value(val1, "vakond");

    // val2 <real> 56.891;
    let val2 = find_node(ctx_body, "val2").unwrap();
    assert_tags(val2, &["real"]);
    assert_num_value(val2, 56.891);

    // val_undef <real>;
    let val_undef = find_node(ctx_body, "val_undef").unwrap();
    assert_tags(val_undef, &["real"]);
    assert_no_value(val_undef);

    // val3 3444.4623;
    let val3 = find_node(ctx_body, "val3").unwrap();
    assert_tags(val3, &[]);
    assert_num_value(val3, 3444.4623);

    // pi 3.14156;
    let pi = find_node(ctx_body, "pi").unwrap();
    assert_tags(pi, &[]);
    assert_num_value(pi, 3.14156);

    // val_float;
    let val_float = find_node(ctx_body, "val_float").unwrap();
    assert_tags(val_float, &[]);
    assert_no_value(val_float);

    // vector [..];
    let vector = find_node(ctx_body, "vector").unwrap();
    assert_tags(vector, &[]);
    assert_list_nums(vector, &[15.6, 17.8, 16.3, 12.3, 67.8, 45.0, 2.0]);
}

#[test]
fn parses_minimal_example_context_fields_with_bad_comments() {
    let path = "./data/minimal_examples/comments/minimal_example_with_fields_with_bad_comments.hymeko";
    // This should throw an error
    let source_code = parser::read_source_file(&path).expect("failed to read source file");

    // 2. Parse it, tying the AST lifetimes to the String
    let res = parser::parse_description(&source_code);
    // Expect error
    assert!(res.is_err());

}

#[test]
fn parses_minimal_example_context_fields_with_bad_comments_inline() {
    let input = r#"
        Minimal_Example { author "Csaba"; }
        context {
        /* Multi-line comment without closing tag
        val0 <int> 56;
        }
    "#;

    let res = parse_description(input);
    // Expect error
    assert!(res.is_err());


}