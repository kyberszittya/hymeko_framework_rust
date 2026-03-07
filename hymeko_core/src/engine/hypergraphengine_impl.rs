use std::collections::HashMap;
use crate::engine::hypergraphengine::HypergraphEngine;
use crate::ir::common::ref_target;
use crate::ir::ir::{DeclKind, Ir, SignedRefR};
use crate::tensor::representations::tensor_coo::TensorCoo;

impl HypergraphEngine {
    pub fn new() -> Self {
        Self {
            builder: TensorCoo::with_meta(0, 0, 0),
            node_registry: HashMap::new(),
            edge_registry: HashMap::new(),
            node_names: Vec::new(),
            edge_names: Vec::new(),
            current_nodes: 0,
            current_edges: 0,
            ir_repository: HashMap::new(),
        }
    }

    pub fn register_ir(&mut self, name: &str, ir: Ir) {
        self.ir_repository.insert(name.to_string(), ir);
    }

    /// Lekér egy tárolt IR-t név szerint (O(1) lookup).
    pub fn get_ir(&self, name: &str) -> Option<&Ir> {
        self.ir_repository.get(name)
    }

    /// Idempotent node creation: Returns existing ID if name is known, otherwise creates a new one.
    pub fn get_or_create_node(&mut self, name: &str) -> usize {
        if let Some(&id) = self.node_registry.get(name) {
            return id;
        }
        let id = self.current_nodes;
        self.current_nodes += 1;
        self.node_registry.insert(name.to_string(), id);
        self.node_names.push(name.to_string());
        id
    }



    /// Idempotent edge creation.
    pub fn get_or_create_edge(&mut self, name: &str) -> usize {
        if let Some(&id) = self.edge_registry.get(name) {
            return id;
        }
        let id = self.current_edges;
        self.current_edges += 1;
        self.edge_registry.insert(name.to_string(), id);
        self.edge_names.push(name.to_string());
        id
    }

    /// Add an arc. We can now use strings directly, keeping the API semantic!
    pub fn add_arc_by_name(&mut self, k: usize, node_name: &str, edge_name: &str, weight: f64) -> Result<(), String> {
        let node_id = self.get_or_create_node(node_name);
        let edge_id = self.get_or_create_edge(edge_name);
        self.builder.push(k, node_id, edge_id, weight);
        Ok(())
    }

    pub fn add_node(&mut self) -> usize {
        let id = self.current_nodes;
        self.current_nodes += 1;
        id
    }

    pub fn add_edge(&mut self) -> usize {
        let id = self.current_edges;
        self.current_edges += 1;
        id
    }

    pub fn add_arc(&mut self, k: usize, node_id: usize, edge_id: usize, weight: f64)
        -> Result<(), String> {
        self.builder.push(k, node_id, edge_id, weight);
        Ok(())
    }

    pub fn compile_from_ir(&self, ir_name: &str) -> Result<TensorCoo<f64>, String> {
        let ir = self.ir_repository.get(ir_name)
            .ok_or_else(|| format!("Architectural Error: IR '{}' not found in registry.", ir_name))?;

        let v_count = ir.nodes.len();
        let e_count = ir.edges.len();
        let dim_star = v_count + e_count;

        // The slice dimension (Z-axis) corresponds to the individual hyperedges.
        let num_slices = e_count;

        let mut tensor = TensorCoo::with_meta(num_slices, dim_star, dim_star);

        // Iterate ONLY over the explicit topological arcs.
        // This naturally filters out structural AST hierarchy (like the fano root container).
        for arc in &ir.arcs {
            let edge_decl = arc.in_edge;

            // Resolve the parent edge's spatial ID
            let edge_id = ir.as_edge(edge_decl)
                .ok_or_else(|| "Mathematical Error: Arc's parent is not registered as an Edge.".to_string())?.0;

            let k = edge_id;
            let e_mapped = edge_id + v_count; // Strict offset into the Edge quadrant

            for reference in &arc.refs {
                let target_decl = ref_target(reference);

                // Natively map the target to its true spatial coordinate based on its IR type
                let target_mapped = match ir.decl_kind(target_decl) {
                    DeclKind::Node => ir.as_node(target_decl).unwrap().0, // Node quadrant [0, V-1]
                    DeclKind::Edge => ir.as_edge(target_decl).unwrap().0 + v_count, // Edge quadrant [V, V+E-1]
                    DeclKind::HyperArc => continue, // Arcs do not point to arcs in this topology
                };

                let weight = 1.0; // Weights can be extracted from arc.anno later if specified

                // Natively resolve spatial symmetry from the AST operators (+, -, ~)
                match reference {
                    SignedRefR::Plus(_) => {
                        // Forward directed arc: Source -> Target (Top-Right Quadrant)
                        tensor.push(k, target_mapped, e_mapped, weight);
                    },
                    SignedRefR::Minus(_) => {
                        // Reverse directed arc: Target -> Source (Bottom-Left Quadrant)
                        tensor.push(k, e_mapped, target_mapped, weight);
                    },
                    SignedRefR::Neutral(_) => {
                        // Perfect Symmetry for neutral operators (~)
                        // Places connections in both the Top-Right and Bottom-Left quadrants
                        tensor.push(k, target_mapped, e_mapped, weight);
                        tensor.push(k, e_mapped, target_mapped, weight);
                    }
                }
            }
        }

        Ok(tensor)
    }


}