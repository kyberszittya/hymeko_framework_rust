use std::path::Path;
use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::exceptions::{PyIndexError, PySyntaxError, PyValueError};

// The exact Arrow imports required for zero-copy FFI.
use arrow::array::{Array, Int64Array, Float32Array};
use arrow::pyarrow::IntoPyArrow;
use hymeko::engine::hypergraphengine::HypergraphEngine;
use hymeko::module_store::module_store::{CompiledProgram, ModuleKey, ModuleStore};
use hymeko::module_store::source_provider::MemProvider;
use hymeko::resolution::string_table::StringTable;
use hymeko::util::real_parser::RealParser;
use hymeko::writers::cbor_writer::CborPayload;


#[pyclass]
pub struct PyHypergraphIR {
    // Keep the compiled IR alive
    pub compiled: Arc<CompiledProgram>,
    pub strings: StringTable,
}

#[pymethods]
impl PyHypergraphIR {

    // -- Counts --

    #[getter]
    pub fn node_count(&self) -> usize {
        self.compiled.ir.nodes.len()
    }

    #[getter]
    pub fn edge_count(&self) -> usize {
        self.compiled.ir.edges.len()
    }

    #[getter]
    pub fn arc_count(&self) -> usize {
        self.compiled.ir.arcs.len()
    }

    // -- Name Resolution --

    /// Maps a CSR matrix row/col index back to its string identifier
    pub fn get_node_name(&self, index: usize) -> PyResult<String> {
        let ir = &self.compiled.ir;
        let rec = ir.nodes.get(index)
            .ok_or_else(|| PyIndexError::new_err("Node index out of bounds"))?;
        Ok(self.strings.resolve(ir.decl_nodes[rec.decl.0].name).to_string())
    }

    pub fn get_edge_name(&self, index: usize) -> PyResult<String> {
        let ir = &self.compiled.ir;
        let rec = ir.edges.get(index)
            .ok_or_else(|| PyIndexError::new_err("Edge index out of bounds"))?;
        Ok(self.strings.resolve(ir.decl_nodes[rec.decl.0].name).to_string())
    }

    #[getter]
    pub fn node_names(&self) -> Vec<String> {
        let ir = &self.compiled.ir;
        ir.nodes.iter()
            .map(|rec| self.strings.resolve(ir.decl_nodes[rec.decl.0].name).to_string())
            .collect()
    }

    #[getter]
    pub fn edge_names(&self) -> Vec<String> {
        let ir = &self.compiled.ir;
        ir.edges.iter()
            .map(|rec| self.strings.resolve(ir.decl_nodes[rec.decl.0].name).to_string())
            .collect()
    }

    // -- Annotations --

    pub fn get_node_annotations(&self, index: usize) -> PyResult<Vec<String>> {
        let ir = &self.compiled.ir;
        let rec = ir.nodes.get(index)
            .ok_or_else(|| PyIndexError::new_err("Node index out of bounds"))?;
        let anno = &ir.decl_nodes[rec.decl.0].anno;
        Ok(anno.tags.iter().map(|&s| self.strings.resolve(s).to_string()).collect())
    }

    pub fn get_edge_annotations(&self, index: usize) -> PyResult<Vec<String>> {
        let ir = &self.compiled.ir;
        let rec = ir.edges.get(index)
            .ok_or_else(|| PyIndexError::new_err("Edge index out of bounds"))?;
        let anno = &ir.decl_nodes[rec.decl.0].anno;
        Ok(anno.tags.iter().map(|&s| self.strings.resolve(s).to_string()).collect())
    }

    pub fn edge_arity(&self, index: usize) -> PyResult<usize> {
        let ir = &self.compiled.ir;
        let rec = ir.edges.get(index)
            .ok_or_else(|| PyIndexError::new_err("Edge index out of bounds"))?;
        Ok(rec.bases.len())
    }

    pub fn to_cbor(&self) -> PyResult<Vec<u8>> {
        let mut buffer = Vec::new();
        ciborium::into_writer(&self.compiled.ir, &mut buffer)
            .map_err(|e| PyValueError::new_err(format!("CBOR Serialization Error: {}", e)))?;
        Ok(buffer)
    }

    // -- Serialization --

    #[staticmethod]
    pub fn from_cbor(data: &[u8]) -> PyResult<Self> {
        let payload: CborPayload = ciborium::from_reader(data)
            .map_err(|e| PyValueError::new_err(format!("CBOR Deserialization Error: {}", e)))?;
        let strings = StringTable::from_vec(payload.interned_strings);
        let compiled = CompiledProgram {
            root: ModuleKey(payload.root_path),
            idx: payload.index,
            ir: payload.ir,
            imports: payload.imports,
            canon_hash: payload.canon_hash,
        };


        // Note: If your PyHypergraphIR wrapper also requires the Interner
        // to resolve strings back to Python, you must store `reconstructed_interner`
        // inside PyHypergraphIR alongside `compiled`.

        Ok(PyHypergraphIR { compiled: Arc::new(compiled), strings }) // Adjust according to your exact PyHypergraphIR struct
    }

    fn __repr__(&self) -> String {
        format!("<PyHypergraphIR(nodes={}, edges={}, arcs={})",
            self.node_count(), self.edge_count(), self.arc_count())
    }

}

// ================================================
// PyTensorCoo3D: A simple COO format for 3D sparse tensors (for clique expansions)
// ================================================
/// 3D sparse tensor in COO format.
///
/// Returned by `compile_star_expansion` and `compile_clique_tensor_expansion`.
///
/// ```python
/// coo = engine.compile_star_expansion(ir)
/// print(coo.shape, coo.nnz)
/// indices, values = coo.export_to_pytorch()
/// t = torch.sparse_coo_tensor(indices, values, coo.shape)
/// ```

#[pyclass]
pub struct PyTensorCoo3D {
    #[pyo3(get)]
    pub dim_k: i64,
    #[pyo3(get)]
    pub dim_i: i64,
    #[pyo3(get)]
    pub dim_j: i64,

    k_ind: Int64Array,
    i_ind: Int64Array,
    j_ind: Int64Array,
    val: Float32Array,
}

impl PyTensorCoo3D {
    fn from_raw(
        k: Vec<usize>, i: Vec<usize>, j: Vec<usize>, v: Vec<f32>,
        shape: (usize, usize, usize)
    ) -> Self {
        Self {
            dim_k: shape.0 as i64,
            dim_i: shape.1 as i64,
            dim_j: shape.2 as i64,
            k_ind: k.into_iter().map(|x| x as i64).collect(),
            i_ind: i.into_iter().map(|x| x as i64).collect(),
            j_ind: j.into_iter().map(|x| x as i64).collect(),
            val: v.into_iter().map(|x| x as f32).collect(),
        }
    }
}


#[pymethods]
impl PyTensorCoo3D {
    #[getter]
    pub fn shape(&self) -> (i64, i64, i64) {
        (self.dim_k, self.dim_i, self.dim_j)
    }




    #[getter]
    pub fn nnz(&self) -> usize {
        self.val.len()
    }



    /// True zero-copy export to Python using the Arrow C Data Interface.
    pub fn export_to_pytorch<'py>(
        &self,
        py: Python<'py>
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>, Bound<'py, PyAny>, Bound<'py, PyAny>)> {
        // No more .into() - we return the safe Bound directly
        let k_py = self.k_ind.to_data().into_pyarrow(py)?;
        let i_py = self.i_ind.to_data().into_pyarrow(py)?;
        let j_py = self.j_ind.to_data().into_pyarrow(py)?;
        let val_py = self.val.to_data().into_pyarrow(py)?;

        Ok((k_py, i_py, j_py, val_py))
    }

    fn __repr__(&self) -> String {
        format!("<PyTensorCoo3D shape=({}, {}, {}) nnz={}>",
            self.dim_k, self.dim_i, self.dim_j, self.nnz())
    }
}

// ================================================
// PySparseMatrix2D - struct for 2D sparse matrices
// ================================================
#[pyclass]
pub struct PySparseMatrix2D {
    #[pyo3(get)]
    dim_i: i64,
    #[pyo3(get)]
    dim_j: i64,
    i_ind: Int64Array,
    j_ind: Int64Array,
    val: Float32Array,
}

impl PySparseMatrix2D {
    pub fn from_raw(i: Vec<usize>, j: Vec<usize>, v: Vec<f32>, shape: (usize, usize)) -> Self {
        Self {
            dim_i: shape.0 as i64,
            dim_j: shape.1 as i64,
            i_ind: Int64Array::from(i.into_iter().map(|x| x as i64).collect::<Vec<i64>>()),
            j_ind: Int64Array::from(j.into_iter().map(|x| x as i64).collect::<Vec<i64>>()),
            val: Float32Array::from(v.into_iter().map(|x| x).collect::<Vec<f32>>()),
        }
    }
}

#[pymethods]
impl PySparseMatrix2D {
    #[getter]
    pub fn shape(&self) -> (i64, i64) {
        (self.dim_i, self.dim_j)
    }

    #[getter]
    pub fn nnz(&self) -> usize {
        self.val.len()
    }

    /// True zero-copy export to Python using the Arrow C Data Interface.
    pub fn export_to_pytorch<'py>(
        &self,
        py: Python<'py>
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>, Bound<'py, PyAny>)> {
        // No more .into()
        let i_py = self.i_ind.to_data().into_pyarrow(py)?;
        let j_py = self.j_ind.to_data().into_pyarrow(py)?;
        let val_py = self.val.to_data().into_pyarrow(py)?;

        Ok((i_py, j_py, val_py))
    }

    fn __repr__(&self) -> String {
        format!("<PySparseMatrix2D shape=({}, {}) nnz={}>", self.dim_i, self.dim_j, self.nnz())
    }
}

// ================================================
// Shared helper: sync IR declarations to engine's global registries
// ================================================



// ================================================
// PyHypergraphEngine: Main interface for Python users to interact with the HypergraphEngine and compile IRs
// ================================================
/// The main engine for compiling `.hymeko` sources and extracting tensor
/// representations from the resulting hypergraph IR.
///
/// ```python
/// engine = hymeko.PyHypergraphEngine()
/// ir = engine.load_file("fano_graph.hymeko")
///
/// star = engine.compile_star_expansion(ir)
/// print(star)  # PyTensorCoo3D(shape=(7,14,14), nnz=42)
/// ```

#[pyclass(unsendable)]
pub struct PyHypergraphEngine {
    inner: HypergraphEngine,
    store: ModuleStore<MemProvider, RealParser>,
}

#[pymethods]
impl PyHypergraphEngine {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: HypergraphEngine::new(),
            store: ModuleStore::new(MemProvider::default(), RealParser),
        }
    }

    pub fn load_file(&mut self, file_path: &str) -> PyResult<PyHypergraphIR> {
        let content = std::fs::read_to_string(file_path)
            .map_err(|e| PyValueError::new_err(format!("File read error: {}", e)))?;
        self.parse_dsl_internal(file_path, &content)
    }

    pub fn parse_dsl(&mut self, source_code: &str) -> PyResult<PyHypergraphIR> {
        let path = format!("memory_module_{}", self.store.it.iter().count());
        self.parse_dsl_internal(&path, source_code)
    }

    fn parse_dsl_internal(&mut self, path: &str, content: &str) -> PyResult<PyHypergraphIR> {
        // Inject content directly into memory
        self.store.provider_mut().insert_file(path, content);

        let compiled = self.store.compile(Path::new(path))
            .map_err(|e| PySyntaxError::new_err(format!("Compile error: {:?}", e)))?;
        let strings = StringTable::from_interner(&self.store.it);

        Ok(PyHypergraphIR { compiled, strings })
    }

    pub fn add_node(&mut self) -> PyResult<usize> {
        Ok(self.inner.add_node())
    }

    pub fn add_edge(&mut self) -> PyResult<usize> {
        Ok(self.inner.add_edge())
    }

    pub fn add_arc(&mut self, k: usize, node_id: usize, edge_id: usize, weight: f64) -> PyResult<()> {
        self.inner.add_arc(k, node_id, edge_id, weight)
            .map_err(|e| PyIndexError::new_err(e))
    }



    pub fn compile_star_expansion(&mut self, py_ir: &PyHypergraphIR) -> PyResult<PyTensorCoo3D> {
        let ir = &py_ir.compiled.ir;
        let core_coo = self.inner.compile_star_expansion_core::<f32>(ir);

        // Step 5: Thin Arrow wrapper
        Ok(PyTensorCoo3D::from_raw(
            core_coo.k, core_coo.i, core_coo.j, core_coo.v,
            (core_coo.num_slices, core_coo.dim_i, core_coo.dim_j)
        ))
    }

    pub fn compile_clique_expansion(&mut self, py_ir: &PyHypergraphIR) -> PyResult<PySparseMatrix2D> {
        let ir = &py_ir.compiled.ir;

        let core_coo = self.inner.compile_clique_expansion_core::<f32>(ir);

        Ok(PySparseMatrix2D::from_raw(
            core_coo.i, core_coo.j, core_coo.v,
            (core_coo.dim_i, core_coo.dim_j)
        ))
    }

    /// Natively extracts the 3D Edge-Colored Clique Expansion Tensor.
    /// Yields an E x V x V tensor where each slice `k` is the fully connected clique for edge `k`.
    pub fn compile_clique_tensor_expansion(&mut self, py_ir: &PyHypergraphIR) -> PyResult<PyTensorCoo3D> {
        let ir = &py_ir.compiled.ir;
        let core_coo = self.inner.compile_clique_expansion_core::<f32>(ir);

        Ok(PyTensorCoo3D::from_raw(
            core_coo.k, core_coo.i, core_coo.j, core_coo.v,
            (core_coo.num_slices, core_coo.dim_i, core_coo.dim_j)
        ))
    }
}

