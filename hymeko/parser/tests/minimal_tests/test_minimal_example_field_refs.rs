use parser::ast::*;
use parser::read_parse_file;
use crate::lib::{body, find_node};

#[test]
fn parses_legacy_context_with_refs() {
    let path = "./data/minimal_examples/minimal_example_fields_with_reference.hymeko";
    let d = read_parse_file(path).unwrap();

    assert_eq!(d.name, "Minimal_Example");

    // meta kötelező a mostani formában
    /*
    assert_eq!(d.meta.inner.name, "Minimal_Example");
    
     */

    // context
    assert_eq!(d.items.len(), 1);
    let context = match &d.items[0] {
        HyperItem::Node(n) => n,
        _ => panic!("Expected Node(context)"),
    };
    assert_eq!(context.inner.name, "context");

    let ctx = body(context);

    // val0
    let val0 = find_node(ctx, "val0");
    assert_eq!(val0.anno.tags, vec!["int".to_string()]);
    match val0.anno.value {
        Some(Value::Num(x)) => assert!((x - 56.0).abs() < 1e-9),
        _ => panic!("Expected val0 numeric value"),
    }

    // val1
    let val1 = find_node(ctx, "val1");
    assert_eq!(val1.anno.tags, vec!["string".to_string()]);
    match &val1.anno.value {
        Some(Value::Str(s)) => assert_eq!(s, "vakond"),
        _ => panic!("Expected val1 string value"),
    }

    // val_node -> node.node0
    let val_node = find_node(ctx, "val_node");
    match &val_node.anno.value {
        Some(Value::Ref(r)) => assert_eq!(r.path, vec!["node".to_string(), "node0".to_string()]),
        other => panic!("Expected val_node ref value, got {:?}", other),
    }
}

#[test]
fn parses_legacy_context_with_refs_alternative() {
    let path = "./data/minimal_examples/minimal_example_fields_with_reference2.hymeko";
    let d = read_parse_file(path).unwrap();

    assert_eq!(d.name, "Minimal_Example");

    // meta kötelező a mostani formában
    /*
    assert_eq!(d.meta.inner.name, "Minimal_Example");

     */

    // context
    assert_eq!(d.items.len(), 1);
    let context = match &d.items[0] {
        HyperItem::Node(n) => n,
        _ => panic!("Expected Node(context)"),
    };
    assert_eq!(context.inner.name, "context");

    let ctx = body(context);

    // val0
    let val0 = find_node(ctx, "val0");
    assert_eq!(val0.anno.tags, vec!["int".to_string()]);
    match val0.anno.value {
        Some(Value::Num(x)) => assert!((x - 56.0).abs() < 1e-9),
        _ => panic!("Expected val0 numeric value"),
    }

    // val1
    let val1 = find_node(ctx, "val1");
    assert_eq!(val1.anno.tags, vec!["string".to_string()]);
    match &val1.anno.value {
        Some(Value::Str(s)) => assert_eq!(s, "vakond"),
        _ => panic!("Expected val1 string value"),
    }

    // val_node -> node.node0
    let val_node = find_node(ctx, "val_node");
    match &val_node.anno.value {
        Some(Value::Ref(r)) => assert_eq!(r.path, vec!["node".to_string(), "node0".to_string()]),
        other => panic!("Expected val_node ref value, got {:?}", other),
    }
}