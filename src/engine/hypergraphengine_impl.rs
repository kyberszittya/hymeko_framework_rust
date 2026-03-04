use crate::engine::hypergraphengine::HypergraphEngine;
use crate::tensor::representations::tensor_csr::{TensorCsr, TensorCsrBuilder};

impl HypergraphEngine {
    pub fn new() -> Self {
        Self {
            builder: TensorCsrBuilder {
                dim_i: 0,
                dim_j: 0,
                unfinalized_row_ptr: vec![0],
                unfinalized_col_ind: Vec::new(),
                unfinalized_val: Vec::new(),
            },
            current_nodes: 0,
            current_edges: 0,
        }
    }

    pub fn add_node(&mut self) -> usize {
        let id = self.current_nodes;
        self.current_nodes += 1;
        self.builder.dim_i = self.current_nodes;
        self.builder.unfinalized_row_ptr.push(
            self.builder.unfinalized_col_ind.len(),
        );
        id
    }

    pub fn add_edge(&mut self) -> usize {
        let id = self.current_edges;
        self.current_edges += 1;
        self.builder.dim_j = self.current_edges;
        id
    }

    pub fn add_arc(&mut self, node_id: usize, edge_id: usize, weight: f64) -> Result<(), String> {
        if node_id >= self.builder.dim_i || edge_id >= self.builder.dim_j {
            return Err("Node ID or Edge ID out of bounds".to_string());
        }
        self.builder.unfinalized_col_ind.push(edge_id);
        self.builder.unfinalized_val.push(weight);
        let current_nnz = self.builder.unfinalized_col_ind.len();

        while self.builder.unfinalized_row_ptr.len() <= node_id + 1 {
            self.builder.unfinalized_row_ptr.push(current_nnz);
        }

        for i in (node_id + 1)..self.builder.unfinalized_row_ptr.len() {
            self.builder.unfinalized_row_ptr[i] = current_nnz;
        }

        Ok(())
    }

    pub fn compile_epoch(&mut self) -> TensorCsr<f64> {
        let current_builder = std::mem::replace(
            &mut self.builder, TensorCsrBuilder {
                dim_i: self.current_nodes,
                dim_j: self.current_edges,
                // We need to ensure the row_ptr is correctly sized for the current number of nodes
                unfinalized_row_ptr: vec![0; self.current_nodes + 1],
                unfinalized_col_ind: Vec::new(),
                unfinalized_val: Vec::new(),
        });
        current_builder.finalize_coalesced()
    }
}