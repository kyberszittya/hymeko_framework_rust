// PyO3 / NumPy return types are inherently verbose; keep `clippy -D warnings` practical.
#![allow(
    clippy::collapsible_if,
    clippy::new_without_default,
    clippy::redundant_closure,
    clippy::type_complexity,
    clippy::useless_conversion,
)]

pub mod interface_python;
pub mod cycles;
pub mod hymeko_parse;

use pyo3::prelude::*;
use crate::interface_python::api::{
    PyHypergraphIR,
    PyHypergraphEngine,
    PyTensorCoo3D,
    PySparseMatrix2D
};

// ==========================================
// LEGACY API SIGNATURE (for pyo3 < 0.21)
// If you update PyO3 later, this will change to (m: &Bound<'_, PyModule>)
// ==========================================
#[pymodule]
fn hymeko(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Explicitly register EVERY class to make them visible in Python
    m.add_class::<PyHypergraphIR>()?;
    m.add_class::<PyHypergraphEngine>()?;
    m.add_class::<PyTensorCoo3D>()?;
    m.add_class::<PySparseMatrix2D>()?;

    m.add_function(wrap_pyfunction!(crate::cycles::enumerate_unsigned_rs, m)?)?;
    m.add_function(wrap_pyfunction!(crate::cycles::enumerate_k_cycles_rs, m)?)?;
    m.add_function(wrap_pyfunction!(crate::cycles::enumerate_k_cycles_color_coded_rs, m)?)?;
    m.add_function(wrap_pyfunction!(crate::cycles::enumerate_k_cycles_path_closure_rs, m)?)?;
    m.add_function(wrap_pyfunction!(crate::cycles::enumerate_k_walks_rs, m)?)?;
    // Unified entries (Strategy refactor 2026-05-11; CLAUDE.md §6.5 #1).
    // - enumerate_cycles_rs:           per-vertex (replaces 8 legacy)
    // - enumerate_top_k_cycles_rs:     top-K global, regular scorers (replaces 2 legacy)
    // - enumerate_top_k_cycles_entropy_rs: top-K global, entropy/hybrid (replaces 2 legacy)
    m.add_function(wrap_pyfunction!(crate::cycles::enumerate_cycles_rs, m)?)?;
    m.add_function(wrap_pyfunction!(crate::cycles::enumerate_top_k_cycles_rs, m)?)?;
    m.add_function(wrap_pyfunction!(crate::cycles::enumerate_top_k_cycles_entropy_rs, m)?)?;
    m.add_function(wrap_pyfunction!(crate::hymeko_parse::parse_hymeko_rs, m)?)?;

    Ok(())
}