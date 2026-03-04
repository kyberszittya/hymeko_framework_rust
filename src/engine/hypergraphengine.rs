use std::collections::HashMap;
use crate::ir::ir::Ir;
use crate::tensor::representations::tensor_csr::TensorCsrBuilder;

pub struct HypergraphEngine {
    pub builder: TensorCsrBuilder<f64>,
    pub node_registry: HashMap<String, usize>,
    pub edge_registry: HashMap<String, usize>,
    pub node_names: Vec<String>,
    pub edge_names: Vec<String>,
    pub current_nodes: usize,
    pub current_edges: usize,
    pub ir_repository: HashMap<String, Ir>,
}

