pub mod interface_python;
pub mod cycles;
pub mod hymeko_parse;

pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

use pyo3::prelude::*;
use crate::interface_python::api::{
    PyHypergraphIR,
    PyHypergraphEngine,
    PyTensorCoo3D,
    PySparseMatrix2D
};

#[cfg(test)]
mod tests {
    
}

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

    m.add_function(wrap_pyfunction!(crate::cycles::enumerate_k_cycles_rs, m)?)?;
    m.add_function(wrap_pyfunction!(crate::cycles::enumerate_k_cycles_color_coded_rs, m)?)?;
    m.add_function(wrap_pyfunction!(crate::cycles::enumerate_k_cycles_path_closure_rs, m)?)?;
    m.add_function(wrap_pyfunction!(crate::cycles::enumerate_k_walks_rs, m)?)?;
    m.add_function(wrap_pyfunction!(crate::cycles::enumerate_top_k_cycles_signed_rs, m)?)?;
    m.add_function(wrap_pyfunction!(crate::cycles::enumerate_top_k_per_vertex_cycles_signed_rs, m)?)?;
    m.add_function(wrap_pyfunction!(crate::hymeko_parse::parse_hymeko_rs, m)?)?;

    Ok(())
}