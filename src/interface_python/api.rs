use pyo3::prelude::*;
use numpy::{PyArray1, IntoPyArray};
use crate::ir::ir::Ir;
use crate::tensor::representations::tensor_csr::TensorCsr;

#[pyclass]
pub struct PyGraphTopology {
    #[pyo3(get)]
    pub num_rows: usize,
    #[pyo3(get)]
    pub num_cols: usize,
    #[pyo3(get)]
    pub nnz: usize,

    // We hold the Rust vectors here until Python requests them
    row_ptr: Option<Vec<usize>>,
    col_ind: Option<Vec<usize>>,
    val: Option<Vec<f32>>,
}

#[pymethods]
impl PyGraphTopology {
    /// Consumes the row_ptr vector and hands ownership to Python/NumPy.
    /// Can only be called once to prevent memory aliasing.
    pub fn take_row_ptr<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<usize>>> {
        let vec = self.row_ptr.take().expect("row_ptr already consumed by Python");
        Ok(vec.into_pyarray(py))
    }

    pub fn take_col_ind<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<usize>>> {
        let vec = self.col_ind.take().expect("col_ind already consumed by Python");
        Ok(vec.into_pyarray(py))
    }

    pub fn take_val<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let vec = self.val.take().expect("val already consumed by Python");
        Ok(vec.into_pyarray(py))
    }
}

// A simple constructor to wrap your existing TensorCsr
impl From<TensorCsr<f32>> for PyGraphTopology {
    fn from(csr: TensorCsr<f32>) -> Self {
        Self {
            num_rows: csr.num_rows,
            num_cols: csr.num_cols,
            nnz: csr.val.len(),
            row_ptr: Some(csr.row_ptr),
            col_ind: Some(csr.col_ind),
            val: Some(csr.val),
        }
    }
}

#[pyclass]
pub struct PyHypergraphIR {
    // Keep the compiled IR alive
    ir: Ir
}

#[pymethods]
impl PyHypergraphIR {
    /// Maps a CSR matrix row/col index back to its string identifier
    pub fn get_node_name(&self, index: usize) -> PyResult<String> {
        // Implementation mapping index -> DeclId -> Interner String
        Ok("Placeholder".to_string())
    }

    pub fn get_node_annotations(&self, index: usize) -> PyResult<Vec<String>> {
        // Fetch original AnnoR metadata attached to this node
        Ok(vec![])
    }
}

#[pyclass]
pub struct PyHypergraphBuilder {
    // We hold the mutable symbolic IR here
    ir: Ir,
}

#[pymethods]
impl PyHypergraphBuilder {
    #[new]
    pub fn new() -> Self {
        // Initialize an empty or default IR
        Self { ir: Ir::default() } // Replace with your actual IR constructor
    }

    /// Path 2: Mutate the Topology (Low Frequency)
    pub fn add_node(&mut self, name: &str) -> PyResult<usize> {
        // Map this to your actual IR mutation logic
        // let node_id = self.ir.create_node(name);
        // Ok(node_id.0)
        Ok(0)
    }

    /// Path 1: Compile the current state into the locked CSR matrix
    pub fn compile_epoch(&self) -> PyResult<PyGraphTopology> {
        // 1. Lower the IR into the View
        // 2. Run the star_expansion_csr
        // 3. Wrap and return memory to Python

        /* let aggcfg = ...; // Your default aggregation config
        let ex = ...;     // Your weight extractor
        let hg = HyperGraphView::from_ir(&self.ir, &aggcfg, &ex);
        let csr = star_expansion_csr(&hg);
        Ok(PyGraphTopology::from(csr))
        */

        // Placeholder returning empty struct to satisfy compiler:
        Ok(PyGraphTopology {
            num_rows: 0, num_cols: 0, nnz: 0,
            row_ptr: None, col_ind: None, val: None
        })
    }
}