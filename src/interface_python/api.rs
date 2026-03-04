use pyo3::prelude::*;
use numpy::{PyArray1, IntoPyArray};
use pyo3::exceptions::{PyIndexError, PyValueError};
use crate::engine::hypergraphengine::HypergraphEngine;
use crate::ir::ir::{Ir, SignedRefR};
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

    #[staticmethod]
    pub fn from_dsl(source_code: &str) -> PyResult<Self> {
        // Itt hívod meg a tényleges LALRPOP parsert a hymeko/parser ládából.
        // Példa a valós bekötésre:
        // let parsed_ir = parser::parse(source_code)
        //     .map_err(|e| PySyntaxError::new_err(format!("LALRPOP hiba: {:?}", e)))?;

        println!("[RUST] Parsed {} bytes into strictly typed IR.", source_code.len());

        // Helyőrző a fordításig, amíg be nem kötöd a valós LALRPOP hívást:
        Ok(Self { ir: crate::ir::ir::Ir::default() })
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

#[pyclass(unsendable)]
pub struct PyHypergraphEngine {
    inner: HypergraphEngine, // Kompozíció: a wrapper birtokolja a magot
}

#[pymethods]
impl PyHypergraphEngine {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: HypergraphEngine::new(),
        }
    }

    pub fn add_node(&mut self) -> PyResult<usize> {
        Ok(self.inner.add_node())
    }

    pub fn add_edge(&mut self) -> PyResult<usize> {
        Ok(self.inner.add_edge())
    }

    pub fn add_arc(&mut self, node_id: usize, edge_id: usize, weight: f64) -> PyResult<()> {
        // A Rust Error-t tiszta Python Exceptionné alakítjuk
        self.inner.add_arc(node_id, edge_id, weight)
            .map_err(|e| PyIndexError::new_err(e))
    }

    pub fn compile_epoch<'py>(&mut self, py: Python<'py>) -> PyResult<(Bound<'py, PyArray1<usize>>, Bound<'py, PyArray1<usize>>, Bound<'py, PyArray1<f64>>)> {
        // 1. Meghívjuk a tiszta Rust logikát
        let final_csr = self.inner.compile_epoch();

        // 2. Kezeljük a PyO3 / NumPy interfészt (Zéró-másolat)
        let row_ptr_py = numpy::ndarray::Array1::from_vec(final_csr.row_ptr).into_pyarray(py);
        let col_ind_py = numpy::ndarray::Array1::from_vec(final_csr.col_ind).into_pyarray(py);
        let val_py = numpy::ndarray::Array1::from_vec(final_csr.val).into_pyarray(py);

        Ok((row_ptr_py, col_ind_py, val_py))
    }

    pub fn apply_ir(&mut self, py_ir: &PyHypergraphIR) -> PyResult<()> {
        println!("[RUST] Ingesting IR topology into the State Machine...");
        let ir = &py_ir.ir;

        // 1. Csomópontok (Nodes) leképezése: DeclId -> CSR Mátrix Sor (Row)
        let mut decl_to_csr_node = std::collections::HashMap::new();
        for node_rec in &ir.nodes {
            // Ide jön majd az Interner lekérdezés: interner.resolve(ir.decl_nodes[node_rec.decl.0].name)
            let node_name = format!("node_{}", node_rec.decl.0);
            let csr_id = self.inner.get_or_create_node(&node_name);
            decl_to_csr_node.insert(node_rec.decl.0, csr_id);
        }

        // 2. Élek (Edges) leképezése: DeclId -> CSR Mátrix Oszlop (Col)
        let mut decl_to_csr_edge = std::collections::HashMap::new();
        for edge_rec in &ir.edges {
            let edge_name = format!("edge_{}", edge_rec.decl.0);
            let csr_id = self.inner.get_or_create_edge(&edge_name);
            decl_to_csr_edge.insert(edge_rec.decl.0, csr_id);
        }

        // 3. Ívek (Arcs) feloldása és betöltése a CSR Builderbe
        for arc_rec in &ir.arcs {
            // Az arc_rec.in_edge mutatja meg, melyik élhez (oszlophoz) tartozik
            let edge_decl_id = arc_rec.in_edge.0;

            if let Some(&csr_edge_id) = decl_to_csr_edge.get(&edge_decl_id) {
                // Végigiterálunk a hivatkozott csomópontokon (SignedRefR)
                for signed_ref in &arc_rec.refs {
                    // Az irányultság határozza meg az alap súlyozást (1.0 vagy -1.0)
                    let (target_decl, base_weight) = match signed_ref {
                        SignedRefR::Plus(atom) => (atom.target.0, 1.0),
                        SignedRefR::Minus(atom) => (atom.target.0, -1.0),
                        SignedRefR::Neutral(atom) => (atom.target.0, 1.0),
                    };

                    if let Some(&csr_node_id) = decl_to_csr_node.get(&target_decl) {
                        // Injektáljuk a Rust magba, ami automatikusan kezeli az O(N log N) coalescingot
                        self.inner.add_arc(csr_node_id, csr_edge_id, base_weight)
                            .map_err(|e| PyValueError::new_err(e))?;
                    }
                }
            }
        }

        Ok(())
    }
}



