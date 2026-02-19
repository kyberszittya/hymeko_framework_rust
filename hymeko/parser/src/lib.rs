pub mod common;
pub mod ast;
pub mod lexer;
pub mod interner;
pub mod intern_pass;
pub mod resolve;
pub mod ir;

use crate::hymeko::DescriptionParser;
use lalrpop_util::lalrpop_mod;
use crate::ast::{AstStr, EdgeDecl, HyperItem, NodeDecl, Value};
use crate::common::SymId;
use crate::lexer::{LexError, Token};

lalrpop_mod!(pub hymeko);



pub fn parse_description(input: &str) -> Result<AstStr, lalrpop_util::ParseError<usize, Token, LexError>> {
    let lexer = crate::lexer::simd::Lexer::new(input);
    let ast = DescriptionParser::new().parse(lexer)?;
    Ok(ast)
}


// Read file and parse
pub fn read_parse_file(path: &str) -> Result<AstStr, lalrpop_util::ParseError<usize, Token, LexError>>
{
    let input = std::fs::read_to_string(path).unwrap();
    parse_description(&input)
}


pub fn find_node<'a>(items: &'a [HyperItem<String>], name: &str) -> &'a NodeDecl<String> {
    items
        .iter()
        .find_map(|it| match it {
            HyperItem::Node(n) if n.inner.name == name => Some(n),
            _ => None,
        })
        .unwrap_or_else(|| panic!("Expected Node({}) in body", name))
}

pub fn assert_tags(n: &NodeDecl<String>, expected: &[&str]) {
    let got = &n.anno.tags;
    assert_eq!(got.len(), expected.len(), "Tag count mismatch for {}", n.inner.name);
    for (g, e) in got.iter().zip(expected.iter()) {
        assert_eq!(g, *e, "Tag mismatch for {}", n.inner.name);
    }
}

pub fn assert_num_value(n: &NodeDecl<String>, expected: f64) {
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

pub fn assert_str_value(n: &NodeDecl<String>, expected: &str) {
    match &n.anno.value {
        Some(Value::Str(s)) => assert_eq!(s, expected, "String value mismatch for {}", n.inner.name),
        other => panic!("Expected string value for {}, got {:?}", n.inner.name, other),
    }
}

pub fn assert_no_value(n: &NodeDecl<String>) {
    assert!(
        n.anno.value.is_none(),
        "Expected no value for {}, got {:?}",
        n.inner.name,
        n.anno.value
    );
}

pub fn assert_list_nums(n: &NodeDecl<String>, expected: &[f64]) {
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

pub fn as_node<Id>(it: &HyperItem<Id>) -> Option<&NodeDecl<Id>> {
    match it {
        HyperItem::Node(n) => Some(n),
        _ => None,
    }
}

pub fn body<'a>(n: &'a NodeDecl<String>) -> &'a [HyperItem<String>] {
    n.inner
        .body
        .as_deref()
        .unwrap_or_else(|| panic!("Expected node {} to have a body", n.inner.name))
}


pub fn find_edge<'a>(items: &'a [HyperItem<SymId>], name: SymId) -> &'a EdgeDecl<SymId> {
    items.iter().find_map(|it| match it {
        HyperItem::Edge(e) if e.inner.name == name => Some(e),
        _ => None
    }).unwrap()
}

pub fn find_node_id<'a>(items: &'a [HyperItem<SymId>], name: SymId) -> &'a NodeDecl<SymId> {
    items.iter().find_map(|it| match it {
        HyperItem::Node(n) if n.inner.name == name => Some(n),
        _ => None
    }).unwrap()
}



