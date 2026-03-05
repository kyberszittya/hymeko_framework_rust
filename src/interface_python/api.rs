use std::path::Path;
use std::sync::Arc;
use pyo3::prelude::*;
use numpy::{PyArray1};
use pyo3::exceptions::{PyIndexError, PySyntaxError, PyValueError};
use crate::engine::hypergraphengine::HypergraphEngine;
use crate::ir::ir::{Ir, SignedRefR};
use crate::module_store::module_store::{CompiledProgram, ModuleStore};
use crate::module_store::source_provider::{MemProvider};
use crate::util::real_parser::RealParser;

#[pyclass]
pub struct PyGraphTopology {
    #[pyo3(get)]
    pub k: Py<PyArray1<usize>>,
    #[pyo3(get)]
    pub i: Py<PyArray1<usize>>,
    #[pyo3(get)]
    pub j: Py<PyArray1<usize>>,
    #[pyo3(get)]
    pub val: Py<PyArray1<f64>>,
}




#[pyclass]
pub struct PyHypergraphIR {
    // Keep the compiled IR alive
    pub compiled: Arc<CompiledProgram>
}

#[pymethods]
impl PyHypergraphIR {


    /// Maps a CSR matrix row/col index back to its string identifier
    pub fn get_node_name(&self, index: usize) -> PyResult<String> {
        if let Some(node_rec) = self.compiled.ir.nodes.get(index) {
            Ok(format!("decl_{}", node_rec.decl.0))
        } else {
            Err(PyIndexError::new_err("Node index out of bounds"))
        }
    }

    pub fn get_node_annotations(&self, _index: usize) -> PyResult<Vec<String>> {
        Ok(vec![])
    }

}

#[pyclass]
pub struct PyHypergraphBuilder {
    // We hold the mutable symbolic IR here
    ir: Ir,
}

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



    pub fn get_node_count(&self) -> usize {
        self.inner.current_nodes
    }

    pub fn get_edge_count(&self) -> usize {
        self.inner.current_edges
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

        Ok(PyHypergraphIR { compiled })
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

    pub fn compile_clique_expansion(&mut self, py_ir: &PyHypergraphIR) -> PyResult<(Vec<usize>, Vec<usize>, Vec<f64>, (usize, usize))> {
        let ir = &py_ir.compiled.ir;
        let it = &self.store.it;

        let mut decl_to_csr_node = std::collections::HashMap::new();
        for node_rec in &ir.nodes {
            let name = it.resolve(ir.decl_nodes[node_rec.decl.0 as usize].name);
            decl_to_csr_node.insert(node_rec.decl.0, self.inner.get_or_create_node(name));
        }

        let v_count = self.inner.current_nodes;
        let mut i_vec = Vec::new();
        let mut j_vec = Vec::new();
        let mut v_vec = Vec::new();

        for arc in &ir.arcs {
            // Collect all nodes participating in this specific hyperedge
            let mut edge_nodes = Vec::new();
            for reference in &arc.refs {
                let target_decl = crate::ir::common::ref_target(reference);
                if let Some(&node_id) = decl_to_csr_node.get(&target_decl.0) {
                    edge_nodes.push(node_id);
                }
            }

            // Project the clique (fully connected subgraph) for these nodes
            for &u in &edge_nodes {
                for &v in &edge_nodes {
                    if u != v { // Subtract the diagonal block (no self-loops)
                        i_vec.push(u);
                        j_vec.push(v);
                        v_vec.push(1.0);
                    }
                }
            }
        }

        Ok((i_vec, j_vec, v_vec, (v_count, v_count)))
    }

    pub fn compile_star_expansion(&mut self, py_ir: &PyHypergraphIR) -> PyResult<(Vec<usize>, Vec<usize>, Vec<usize>, Vec<f64>, (usize, usize, usize))> {
        let ir = &py_ir.compiled.ir;
        let it = &self.store.it;

        // 1. Sync local IR declarations to the global Engine Registry
        let mut decl_to_csr_node = std::collections::HashMap::new();
        let mut decl_to_csr_edge = std::collections::HashMap::new();

        for node_rec in &ir.nodes {
            let name = it.resolve(ir.decl_nodes[node_rec.decl.0 as usize].name);
            decl_to_csr_node.insert(node_rec.decl.0, self.inner.get_or_create_node(name));
        }

        for edge_rec in &ir.edges {
            let name = it.resolve(ir.decl_nodes[edge_rec.decl.0 as usize].name);
            decl_to_csr_edge.insert(edge_rec.decl.0, self.inner.get_or_create_edge(name));
        }

        let v_count = self.inner.current_nodes;
        let e_count = self.inner.current_edges;
        let dim_star = v_count + e_count;

        let mut k_vec = Vec::new();
        let mut i_vec = Vec::new();
        let mut j_vec = Vec::new();
        let mut v_vec = Vec::new();

        // 2. Extract topological arcs directly using the globally mapped IDs
        for arc in &ir.arcs {
            let edge_idx = *decl_to_csr_edge.get(&arc.in_edge.0)
                .ok_or_else(|| PyValueError::new_err("Mathematical Error: Missing parent edge in registry"))?;

            let e_mapped = edge_idx + v_count; // Strict offset into the Edge quadrant [V, V+E-1]
            let k = edge_idx;

            for reference in &arc.refs {
                let target_decl = crate::ir::common::ref_target(reference);

                // Check the explicit target mapping. If it is the fano root (neither node nor edge), it is ignored.
                let target_mapped = if let Some(&node_id) = decl_to_csr_node.get(&target_decl.0) {
                    node_id // Node quadrant [0, V-1]
                } else if let Some(&edge_id) = decl_to_csr_edge.get(&target_decl.0) {
                    edge_id + v_count // Edge quadrant [V, V+E-1]
                } else {
                    continue;
                };

                let weight = 1.0;

                // Natively resolve spatial symmetry from the AST operators (+, -, ~)
                match reference {
                    SignedRefR::Plus(_) => {
                        // Forward directed arc: Source -> Target
                        k_vec.push(k); i_vec.push(target_mapped); j_vec.push(e_mapped); v_vec.push(weight);
                    },
                    SignedRefR::Minus(_) => {
                        // Reverse directed arc: Target -> Source
                        k_vec.push(k); i_vec.push(e_mapped); j_vec.push(target_mapped); v_vec.push(weight);
                    },
                    SignedRefR::Neutral(_) => {
                        // Perfect Symmetry for neutral operators (~)
                        k_vec.push(k); i_vec.push(target_mapped); j_vec.push(e_mapped); v_vec.push(weight);
                        k_vec.push(k); i_vec.push(e_mapped); j_vec.push(target_mapped); v_vec.push(weight);
                    }
                }
            }
        }

        // The Z-axis slices natively correspond to the total global edge count
        Ok((k_vec, i_vec, j_vec, v_vec, (e_count, dim_star, dim_star)))
    }

    /// Natively extracts the 3D Edge-Colored Clique Expansion Tensor.
    /// Yields an E x V x V tensor where each slice `k` is the fully connected clique for edge `k`.
    pub fn compile_clique_tensor_expansion(&mut self, py_ir: &PyHypergraphIR) -> PyResult<(Vec<usize>, Vec<usize>, Vec<usize>, Vec<f64>, (usize, usize, usize))> {
        let ir = &py_ir.compiled.ir;
        let it = &self.store.it;

        // 1. Map declarations to global Engine IDs
        let mut decl_to_csr_node = std::collections::HashMap::new();
        let mut decl_to_csr_edge = std::collections::HashMap::new();

        for node_rec in &ir.nodes {
            let name = it.resolve(ir.decl_nodes[node_rec.decl.0 as usize].name);
            decl_to_csr_node.insert(node_rec.decl.0, self.inner.get_or_create_node(name));
        }

        for edge_rec in &ir.edges {
            let name = it.resolve(ir.decl_nodes[edge_rec.decl.0 as usize].name);
            decl_to_csr_edge.insert(edge_rec.decl.0, self.inner.get_or_create_edge(name));
        }

        let v_count = self.inner.current_nodes;
        let e_count = self.inner.current_edges;

        let mut k_vec = Vec::new();
        let mut i_vec = Vec::new();
        let mut j_vec = Vec::new();
        let mut v_vec = Vec::new();

        // 2. Build the Edge-Colored Tensor Space
        for arc in &ir.arcs {
            let edge_idx = *decl_to_csr_edge.get(&arc.in_edge.0)
                .ok_or_else(|| PyValueError::new_err("Mathematical Error: Missing parent edge in registry"))?;

            // Collect all pure nodes participating in this specific hyperedge
            let mut edge_nodes = Vec::new();
            for reference in &arc.refs {
                let target_decl = crate::ir::common::ref_target(reference);
                if let Some(&node_id) = decl_to_csr_node.get(&target_decl.0) {
                    edge_nodes.push(node_id);
                }
            }

            let k = edge_idx; // The slice identity is the Edge ID

            // Project the clique exclusively into slice `k`
            for &u in &edge_nodes {
                for &v in &edge_nodes {
                    if u != v { // Exclude self-loops on the diagonal
                        k_vec.push(k);
                        i_vec.push(u);
                        j_vec.push(v);
                        v_vec.push(1.0);
                    }
                }
            }
        }

        // Return dimensions strictly bound to (E, V, V)
        Ok((k_vec, i_vec, j_vec, v_vec, (e_count, v_count, v_count)))
    }
}