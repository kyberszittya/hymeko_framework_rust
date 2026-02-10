use parser::ast::{HyperItem, NodeDecl, Value};


pub fn find_node<'a>(items: &'a [HyperItem], name: &str) -> &'a NodeDecl {
    items
        .iter()
        .find_map(|it| match it {
            HyperItem::Node(n) if n.inner.name == name => Some(n),
            _ => None,
        })
        .unwrap_or_else(|| panic!("Expected Node({}) in body", name))
}

pub fn assert_tags(n: &NodeDecl, expected: &[&str]) {
    let got = &n.anno.tags;
    assert_eq!(got.len(), expected.len(), "Tag count mismatch for {}", n.inner.name);
    for (g, e) in got.iter().zip(expected.iter()) {
        assert_eq!(g, *e, "Tag mismatch for {}", n.inner.name);
    }
}

pub fn assert_num_value(n: &NodeDecl, expected: f64) {
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

pub fn assert_str_value(n: &NodeDecl, expected: &str) {
    match &n.anno.value {
        Some(Value::Str(s)) => assert_eq!(s, expected, "String value mismatch for {}", n.inner.name),
        other => panic!("Expected string value for {}, got {:?}", n.inner.name, other),
    }
}

pub fn assert_no_value(n: &NodeDecl) {
    assert!(
        n.anno.value.is_none(),
        "Expected no value for {}, got {:?}",
        n.inner.name,
        n.anno.value
    );
}

pub fn assert_list_nums(n: &NodeDecl, expected: &[f64]) {
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

pub fn as_node<'a>(it: &'a HyperItem) -> &'a NodeDecl {
    match it {
        HyperItem::Node(n) => n,
        other => panic!("Expected Node, got {:?}", other),
    }
}

pub fn body<'a>(n: &'a NodeDecl) -> &'a [HyperItem] {
    n.inner
        .body
        .as_deref()
        .unwrap_or_else(|| panic!("Expected node {} to have a body", n.inner.name))
}