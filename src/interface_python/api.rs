use pyo3::prelude::*;
use numpy::{PyArray1, IntoPyArray};
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