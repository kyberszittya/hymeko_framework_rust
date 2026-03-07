pub mod interface_python;

pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

use pyo3::prelude::*;
use crate::interface_python::api::{
    PyGraphTopology,
    PyHypergraphIR,
    PyHypergraphBuilder,
    PyHypergraphEngine,
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
    m.add_class::<PyGraphTopology>()?;
    m.add_class::<PyHypergraphIR>()?;
    m.add_class::<PyHypergraphBuilder>()?;
    m.add_class::<PyHypergraphEngine>()?;

    Ok(())
}