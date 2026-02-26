use parser::ast::{EdgeDecl, HyperItem, NodeDecl, Value};
use crate::common::ids::SymId;

pub mod common;
pub mod ir;
pub mod traversal;
pub mod writers;
pub mod sym_ast;
pub mod resolution;
pub mod util;
pub mod module_store;
pub mod tensor;

pub fn find_node<'ast, 'slice>(
    items: &'slice [HyperItem<'ast, &'ast str>],
    name: &str
) -> Option<&'slice NodeDecl<'ast, &'ast str>> {
    items.iter().find_map(|it| match it {
            HyperItem::Node(n) if n.inner.name == name => Some(n),
            _ => None,
        })
        
}

pub fn assert_tags<'a>(n: &NodeDecl<'a, &'a str>, expected: &[&str]) {
    let got = &n.anno.tags;
    assert_eq!(got.len(), expected.len(), "Tag count mismatch for {}", n.inner.name);
    for (g, e) in got.iter().zip(expected.iter()) {
        // Directly compare the string slices
        assert_eq!(*g, *e, "Tag mismatch for {}", n.inner.name);
    }
}

pub fn assert_num_value<'a>(n: &NodeDecl<'a, &'a str>, expected: f64) {
    match &n.anno.value {
        Some(Value::Num(x)) => assert!(
            (*x - expected).abs() < 1e-9,
            "Numeric value mismatch for {}: got {}, expected {}",
            n.inner.name,
            x,
            expected
        ),
        other => panic!("Expected numeric value for {}, got {:?}", n.inner.name, other),
    }
}

pub fn assert_str_value<'a>(n: &NodeDecl<'a, &'a str>, expected: &str) {
    match &n.anno.value {
        Some(Value::Str(s)) => assert_eq!(s.as_ref(), expected, "String value mismatch for {}", n.inner.name),
        other => panic!("Expected string value for {}, got {:?}", n.inner.name, other),
    }
}

pub fn assert_no_value<'a>(n: &NodeDecl<'a, &'a str>) {
    assert!(
        n.anno.value.is_none(),
        "Expected no value for {}, got {:?}",
        n.inner.name,
        n.anno.value
    );
}

pub fn assert_list_nums<'a>(n: &NodeDecl<'a, &'a str>, expected: &[f64]) {
    match &n.anno.value {
        Some(Value::List(xs)) => {
            assert_eq!(xs.len(), expected.len(), "List length mismatch for {}", n.inner.name);
            for (i, (x, e)) in xs.iter().zip(expected.iter()).enumerate() {
                match x {
                    Value::Num(v) => assert!(
                        (*v - *e).abs() < 1e-9,
                        "List numeric mismatch for {} at idx {}: got {}, expected {}",
                        n.inner.name,
                        i,
                        v,
                        e
                    ),
                    other => panic!(
                        "Expected numeric list element for {} at idx {}, got {:?}",
                        n.inner.name, i, other
                    ),
                }
            }
        }
        other => panic!("Expected list value for {}, got {:?}", n.inner.name, other),
    }
}

pub fn as_node<'ast, 'slice, Id>(it: &'slice HyperItem<'ast, Id>) -> Option<&'slice NodeDecl<'ast, Id>> {
    match it {
        HyperItem::Node(n) => Some(n),
        _ => None,
    }
}

pub fn body<'ast, 'slice>(n: &'slice NodeDecl<'ast, &'ast str>) -> &'slice [HyperItem<'ast, &'ast str>] {
    n.inner
        .body
        .as_deref()
        .unwrap_or_else(|| panic!("Expected node {} to have a body", n.inner.name))
}

pub fn find_edge<'ast, 'slice>(
    items: &'slice [HyperItem<'ast, SymId>], 
    name: SymId
) -> Option<&'slice EdgeDecl<'ast, SymId>> {
    items.iter().find_map(|it| match it {
        HyperItem::Edge(e) if e.inner.name == name => Some(e),
        _ => None
    })
}

pub fn find_node_id<'ast, 'slice>(
    items: &'slice [HyperItem<'ast, SymId>], 
    name: SymId
) -> Option<&'slice NodeDecl<'ast, SymId>> {
    items.iter().find_map(|it| match it {
        HyperItem::Node(n) if n.inner.name == name => Some(n),
        _ => None
    })
}