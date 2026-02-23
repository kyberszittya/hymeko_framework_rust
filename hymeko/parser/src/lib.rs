
pub mod ast;
pub mod lexer;
pub mod interner;
pub mod intern_pass;
pub mod resolve;
pub mod ir;
pub mod common;
pub mod traversal;
pub mod writers;
pub mod module_store;
pub mod module_view;
pub mod source_provider;

use std::fs::File;
use crate::hymeko::DescriptionParser;
use lalrpop_util::lalrpop_mod;
use memmap2::Mmap;
use crate::ast::{AstStr, EdgeDecl, HyperItem, NodeDecl, Value};
use crate::common::ids::SymId;
use crate::lexer::{LexError, Token};

lalrpop_mod!(pub hymeko);



pub fn parse_description<'a>(input: &'a str) -> Result<AstStr<'a>, lalrpop_util::ParseError<usize, Token<'a>, LexError>> {
    let lexer = crate::lexer::simd::Lexer::new(input);

    // DescriptionParser now correctly returns AstStr<'a> (Description<'a, &'a str>)
    DescriptionParser::new().parse(lexer)
}

pub struct ParsedFile<'a> {
    pub mmap: Mmap,
    pub ast: AstStr<'a>,
}

pub fn read_parse_file(path: &str) -> Result<Mmap, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    // SAFETY: We assume the file is not being modified concurrently.
    let mmap = unsafe { Mmap::map(&file)? };
    Ok(mmap)
}

pub fn parse_from_mmap<'a>(mmap: &'a Mmap) -> Result<AstStr<'a>, lalrpop_util::ParseError<usize, Token<'a>, LexError>> {
    let input = std::str::from_utf8(mmap).map_err(|_| {
        lalrpop_util::ParseError::User {
            error: LexError { at: 0, msg: "File is not valid UTF-8".into() }
        }
    })?;
    parse_description(input)
}


pub fn find_node<'ast, 'slice>(items: &'slice [HyperItem<'ast, &'ast str>], name: &str) -> &'slice NodeDecl<'ast, &'ast str> {
    items
        .iter()
        .find_map(|it| match it {
            HyperItem::Node(n) if n.inner.name == name => Some(n),
            _ => None,
        })
        .unwrap_or_else(|| panic!("Expected Node({}) in body", name))
}

pub fn assert_tags<'a>(n: &NodeDecl<'a, &'a str>, expected: &[&str]) {
    let got = &n.anno.tags;
    assert_eq!(got.len(), expected.len(), "Tag count mismatch for {}", n.inner.name);
    for (g, e) in got.iter().zip(expected.iter()) {
        // Use .as_ref() to extract the &str from the Cow for comparison
        assert_eq!(g.as_ref(), *e, "Tag mismatch for {}", n.inner.name);
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

pub fn find_edge<'ast, 'slice>(items: &'slice [HyperItem<'ast, SymId>], name: SymId) -> &'slice EdgeDecl<'ast, SymId> {
    items.iter().find_map(|it| match it {
        HyperItem::Edge(e) if e.inner.name == name => Some(e),
        _ => None
    }).unwrap()
}

pub fn find_node_id<'ast, 'slice>(items: &'slice [HyperItem<'ast, SymId>], name: SymId) -> &'slice NodeDecl<'ast, SymId> {
    items.iter().find_map(|it| match it {
        HyperItem::Node(n) if n.inner.name == name => Some(n),
        _ => None
    }).unwrap()
}
