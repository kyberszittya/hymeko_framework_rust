use pyo3::prelude::*;
use parser::ast::{EdgeDecl, HyperItem, NodeDecl};
use crate::common::ids::SymId;

use crate::interface_python::api::{
    PyGraphTopology,
    PyHypergraphIR,
    PyHypergraphBuilder,
    PyHypergraphEngine,
};

pub mod common;
pub mod ir;
pub mod traversal;
pub mod writers;
pub mod sym_ast;
pub mod resolution;
pub mod util;
pub mod module_store;
pub mod tensor;
mod interface_python;
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

// ==========================================
// LEGACY API SIGNATURE (for pyo3 < 0.21)
// If you update PyO3 later, this will change to (m: &Bound<'_, PyModule>)
// ==========================================
#[pymodule]
fn hymeko(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Explicitly register EVERY class to make them visible in Python
    m.add_class::<PyGraphTopology>()?;
    m.add_class::<PyHypergraphIR>()?;
    m.add_class::<PyHypergraphBuilder>()?;
    m.add_class::<PyHypergraphEngine>()?;

    Ok(())
}