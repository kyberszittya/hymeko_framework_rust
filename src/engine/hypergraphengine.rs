use crate::tensor::representations::tensor_csr::TensorCsrBuilder;

pub struct HypergraphEngine {
    pub builder: TensorCsrBuilder<f64>,
    pub current_nodes: usize,
    pub current_edges: usize,
}

