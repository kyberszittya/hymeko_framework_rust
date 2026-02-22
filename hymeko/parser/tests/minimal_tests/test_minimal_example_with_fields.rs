use std::fs::File;
use memmap2::Mmap;
use parser::ast::*;
use parser::{assert_list_nums, assert_no_value, assert_num_value, assert_str_value, 
             assert_tags, find_node, parse_from_mmap};



#[test]
fn parses_minimal_example_context_fields() {
    let path = "./data/minimal_examples/minimal_example_with_fields.hymeko";
    let file = File::open(path).unwrap();
    let mmap = unsafe { Mmap::map(&file).unwrap() };

    // The AST is valid as long as 'mmap' is in scope.
    let d = parse_from_mmap(&mmap).unwrap();
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
    let val0 = find_node(ctx_body, "val0");
    assert_tags(val0, &["int"]);
    assert_num_value(val0, 56.0);

    // val1 <string> "vakond";
    let val1 = find_node(ctx_body, "val1");
    assert_tags(val1, &["string"]);
    assert_str_value(val1, "vakond");

    // val2 <real> 56.891;
    let val2 = find_node(ctx_body, "val2");
    assert_tags(val2, &["real"]);
    assert_num_value(val2, 56.891);

    // val_undef <real>;
    let val_undef = find_node(ctx_body, "val_undef");
    assert_tags(val_undef, &["real"]);
    assert_no_value(val_undef);

    // val3 3444.4623;
    let val3 = find_node(ctx_body, "val3");
    assert_tags(val3, &[]);
    assert_num_value(val3, 3444.4623);

    // pi 3.14156;
    let pi = find_node(ctx_body, "pi");
    assert_tags(pi, &[]);
    assert_num_value(pi, 3.14156);

    // val_float;
    let val_float = find_node(ctx_body, "val_float");
    assert_tags(val_float, &[]);
    assert_no_value(val_float);

    // vector [..];
    let vector = find_node(ctx_body, "vector");
    assert_tags(vector, &[]);
    assert_list_nums(vector, &[15.6, -17.8, 16.3, 12.3, 67.8, 45.0, 2.0]);
}


