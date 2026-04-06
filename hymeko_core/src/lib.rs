use parser::ast::{EdgeDecl, HyperItem, NodeDecl};
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
pub mod engine;

pub fn find_node<'ast, 'slice>(
    items: &'slice [HyperItem<'ast, &'ast str>],
    name: &str
) -> Option<&'slice NodeDecl<'ast, &'ast str>> {
    items.iter().find_map(|it| match it {
            HyperItem::Node(n) if n.inner.name == name => Some(n),
            _ => None,
        })
        
}



pub fn as_node<'ast, 'slice, Id>(it: &'slice HyperItem<'ast, Id>) -> Option<&'slice NodeDecl<'ast, Id>> {
    match it {
        HyperItem::Node(n) => Some(n),
        _ => None,
    }
}

pub fn body<'ast, 'slice>(n: &'slice NodeDecl<'ast, &'ast str>) -> Option<&'slice [HyperItem<'ast, &'ast str>]> {
    n.inner
        .body
        .as_deref()
        
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

