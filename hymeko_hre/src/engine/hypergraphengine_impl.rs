use rustc_hash::FxHashMap;
use std::slice;
use crate::engine::hypergraphengine::HypergraphEngine;
use hymeko::ir::common::ref_target;
use hymeko::ir::ir::{DeclKind, Ir, SignedRefR};
use hymeko::resolution::string_table::StringTable;
use hymeko::tensor::aggregation::{AggCfg, SignAgg, WeightAgg};
use hymeko::tensor::common::Real;
use hymeko::tensor::representations::tensor_coo::TensorCoo;
use hymeko::tensor::representations::tensor_coo_representation;
use hymeko::tensor::tensor_val::{EdgeWScalar, ScalarWeightExtractor};
use hymeko::traversal::hypergraphview::HyperGraphView;
#[cfg(feature = "ipc")]
use hymeko::tensor::shared_state::{ExpansionHeader, ExpansionKind};

impl HypergraphEngine {
    pub fn new() -> Self {
        Self {
            builder: TensorCoo::with_meta(0, 0, 0),
            node_registry: FxHashMap::default(),
            edge_registry: FxHashMap::default(),
            node_names: Vec::new(),
            edge_names: Vec::new(),
            current_nodes: 0,
            current_edges: 0,
            ir_repository: FxHashMap::default(),
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

    /// Internal engine logic for Star Expansion. Returns pure Rust TensorCoo<f32>.
    pub fn compile_star_expansion_core<F: Real>(&self, ir: &Ir) -> TensorCoo<F> {
        let cfg = AggCfg {
            sign: SignAgg::PreferNonNeutral,
            weight: WeightAgg::Sum,
            clamp01: false,
        };
        let ex = ScalarWeightExtractor;

        // Build the optimized mathematical view
        let view: HyperGraphView<F, EdgeWScalar<F>, F> =
            HyperGraphView::from_ir(ir, &cfg, &ex);

        // Execute core expansion math
        tensor_coo_representation::star_expansion_coo(&view)
    }

    fn sync_ir_to_engine(
        ir: &Ir,
        strings: &StringTable,
        engine: &mut HypergraphEngine
    ) -> (FxHashMap<usize, usize>, FxHashMap<usize, usize>) {
        let mut decl_to_csr_node = FxHashMap::default();
        let mut decl_to_csr_edge = FxHashMap::default();

        for node_rec in &ir.nodes {
            let name = strings.resolve(ir.decl_nodes[node_rec.decl.0].name);
            decl_to_csr_node.insert(node_rec.decl.0, engine.get_or_create_node(name));
        }

        for edge_rec in &ir.edges {
            let name = strings.resolve(ir.decl_nodes[edge_rec.decl.0].name);
            decl_to_csr_edge.insert(edge_rec.decl.0, engine.get_or_create_edge(name));
        }

        (decl_to_csr_node, decl_to_csr_edge)
    }

    /// Internal engine logic for Clique Expansion.
    pub fn compile_clique_expansion_core<F: Real>(&self, ir: &Ir) -> TensorCoo<F> {
        let cfg = AggCfg {
            sign: SignAgg::PreferNonNeutral,
            weight: WeightAgg::Sum,
            clamp01: false,
        };
        let ex = ScalarWeightExtractor;

        let view: HyperGraphView<F, EdgeWScalar<F>, F> =
            HyperGraphView::from_ir(ir, &cfg, &ex);

        tensor_coo_representation::clique_expansion_coo(&view)
    }

    #[cfg(feature = "ipc")]
    /// Streams the star expansion directly into raw buffers that live inside an `iceoryx2` sample.
    ///
    /// # Safety
    /// The caller must guarantee that the provided pointers reference at least `capacity`
    /// writable elements and that the backing memory outlives the write.
    pub unsafe fn write_tensor_into_raw(
        header: &ExpansionHeader,
        coo: &TensorCoo<f32>,
        header_ptr: *mut ExpansionHeader,
        k_ptr: *mut i64,
        i_ptr: *mut i64,
        j_ptr: *mut i64,
        values_ptr: *mut f32,
        capacity: usize,
    ) -> Result<usize, &'static str> {
        if header_ptr.is_null() || k_ptr.is_null() || i_ptr.is_null() || j_ptr.is_null() || values_ptr.is_null() {
            return Err("raw star expansion pointers must not be null");
        }

        if coo.len() > capacity {
            return Err("provided buffer capacity is too small for star expansion output");
        }

        unsafe { *header_ptr = *header; }

        let k_slice = unsafe { slice::from_raw_parts_mut(k_ptr, capacity) };
        let i_slice = unsafe { slice::from_raw_parts_mut(i_ptr, capacity) };
        let j_slice = unsafe { slice::from_raw_parts_mut(j_ptr, capacity) };
        let values_slice = unsafe { slice::from_raw_parts_mut(values_ptr, capacity) };

        for idx in 0..coo.len() {
            let entry = coo.entry(idx);
            k_slice[idx] = entry.k as i64;
            i_slice[idx] = entry.i as i64;
            j_slice[idx] = entry.j as i64;
            values_slice[idx] = entry.v;
        }

        Ok(coo.len())
    }

    #[cfg(feature = "ipc")]
    /// Streams the star expansion directly into raw buffers that live inside an `iceoryx2` sample.
    ///
    /// # Safety
    /// The caller must guarantee that the provided pointers reference at least `capacity`
    /// writable elements and that the backing memory outlives the write.
    pub unsafe fn write_star_tensor_into_raw(
        &self,
        header: &ExpansionHeader,
        coo: &TensorCoo<f32>,
        header_ptr: *mut ExpansionHeader,
        k_ptr: *mut i64,
        i_ptr: *mut i64,
        j_ptr: *mut i64,
        values_ptr: *mut f32,
        capacity: usize,
    ) -> Result<usize, &'static str> {
        unsafe { Self::write_tensor_into_raw(header, coo, header_ptr, k_ptr, i_ptr, j_ptr, values_ptr, capacity) }
    }

    #[cfg(feature = "ipc")]
    pub unsafe fn write_star_expansion_into_raw(
         &self,
         ir: &Ir,
         header_ptr: *mut ExpansionHeader,
         k_ptr: *mut i64,
         i_ptr: *mut i64,
         j_ptr: *mut i64,
         values_ptr: *mut f32,
         capacity: usize,
    ) -> Result<usize, &'static str> {
        let coo = self.compile_star_expansion_core::<f32>(ir);
        let header = ExpansionHeader::new(ExpansionKind::Star3D, coo.len(), coo.num_slices, coo.dim_i, coo.dim_j);
        unsafe { Self::write_tensor_into_raw(&header, &coo, header_ptr, k_ptr, i_ptr, j_ptr, values_ptr, capacity) }
    }
  }
