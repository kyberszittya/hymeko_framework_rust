use std::mem::size_of;
use std::path::Path;
use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::exceptions::{PyIndexError, PySyntaxError, PyValueError};
use pyo3::types::PyModule;
use pyo3::PyRef;
use iceoryx2::prelude::*;

// The exact Arrow imports required for zero-copy FFI.
use arrow::array::{Array, Int64Array, Float32Array};
use arrow::pyarrow::IntoPyArrow;
use hymeko_hre::HypergraphEngine;
use hymeko::tensor::shared_state::ExpansionHeader;
use hymeko::module_store::module_store::{CompiledProgram, ModuleKey, ModuleStore};
use hymeko::module_store::source_provider::MemProvider;
use hymeko::resolution::string_table::StringTable;
use hymeko::util::real_parser::RealParser;
use hymeko::writers::cbor_writer::CborPayload;
use hymeko::common::ids::DeclId;
use hymeko::ir::ir::{DeclKind, Ir};
use hymeko_formats::urdf::generate_urdf;
use hymeko_formats::sdf::generate_sdf;


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

        // Count distinct node targets across all arcs of this edge
        let mut count = 0usize;
        for &aid in &rec.arcs {
            let arc = &ir.arcs[aid.0];
            for r in &arc.refs {
                let target = match r {
                    hymeko::ir::ir::SignedRefR::Plus(a)
                    | hymeko::ir::ir::SignedRefR::Minus(a)
                    | hymeko::ir::ir::SignedRefR::Neutral(a) => a.target,
                };
                // Only count if the target resolves to a node (not another edge)
                if ir.decl_to_node.get(target.0).and_then(|x| *x).is_some() {
                    count += 1;
                }
            }
        }
        Ok(count)
    }

    pub fn edge_base_count(&self, index: usize) -> PyResult<usize> {
        let ir = &self.compiled.ir;
        let rec = ir.edges.get(index)
            .ok_or_else(|| PyIndexError::new_err("Edge index out of bounds"))?;
        Ok(rec.bases.len())
    }

    pub fn to_cbor(&self) -> PyResult<Vec<u8>> {
        // Construct the full payload that the daemon expects
        let payload = CborPayload {
            root_path: self.compiled.root.0.clone(), // Extract the inner ID from ModuleKey
            index: self.compiled.idx.clone(),
            ir: self.compiled.ir.clone(),
            imports: self.compiled.imports.clone(),
            canon_hash: self.compiled.canon_hash,
            // Assuming your StringTable has a method like to_vec() or into_inner()
            // to retrieve the original Vec<String>. Adjust the method name if needed.
            interned_strings: self.strings.to_vec(),
        };

        let mut buffer = Vec::new();
        ciborium::into_writer(&payload, &mut buffer)
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

    // -- Codegen --

    /// Emit URDF XML from the compiled IR.
    pub fn to_urdf(&self, robot_name: &str) -> String {
        generate_urdf(&self.compiled.ir, &self.strings, robot_name)
    }

    /// Emit SDF 1.7 XML from the compiled IR.
    pub fn to_sdf(&self, model_name: &str) -> String {
        generate_sdf(&self.compiled.ir, &self.strings, model_name)
    }

    /// Emit a DOT (Graphviz) serialisation of the signed-incidence
    /// hypergraph — vertices as ellipses, hyperedges as rounded boxes,
    /// signed arcs colour-coded (blue +1, red −1, grey ~0). Mirrors the
    /// browser demo's `to_dot` so Python scripts can produce the same
    /// artefact. Does NOT go through the workspace `transforms/` dir,
    /// so the output is deterministic and self-contained.
    pub fn to_dot(&self, graph_name: &str) -> String {
        emit_dot_graph(&self.compiled.ir, &self.strings, graph_name)
    }

    /// JSON snapshot of the compiled IR — graph-viewer-ready shape.
    /// Top-level keys: `node_count`, `edge_count`, `arc_count`,
    /// `nodes` (list of decl-Node entries), `edges` (list of decl-Edge
    /// entries with their signed arc refs). Same schema as the browser
    /// demo's `snapshot_json()`.
    pub fn snapshot_json(&self) -> PyResult<String> {
        emit_snapshot_json(&self.compiled.ir, &self.strings)
            .map_err(|e| PyValueError::new_err(e))
    }

    // -- Predicate queries --

    /// Run a predicate-string query over the IR and return the list of
    /// matching decl names. Supported atoms (same surface as
    /// `queries/standard.qlist`):
    ///
    ///   KIND(<name>)                — decl whose first inherited base is <name>
    ///   INHERITS(<name>)            — decl transitively inheriting <name>
    ///   SCOPEDIN(<name>)            — decl has an ancestor inheriting <name>
    ///   HASARCREF(<sign>, <inner>)  — edge with an arc-ref of <sign> (+1/-1)
    ///                                 pointing at a decl matching <inner>
    ///   <a> AND <b>                 — conjunction
    ///   ANY                         — always true
    pub fn query(&self, predicate: &str) -> Vec<String> {
        let ir = &self.compiled.ir;
        let mut out = Vec::new();
        for i in 0..ir.decl_nodes.len() {
            let did = DeclId::new(i);
            if pred_match_expr(predicate, did, ir, &self.strings) {
                out.push(self.strings.resolve(ir.decl_nodes[i].name).to_string());
            }
        }
        out
    }

    /// Convenience: return the match count only.
    pub fn query_count(&self, predicate: &str) -> usize {
        let ir = &self.compiled.ir;
        (0..ir.decl_nodes.len())
            .filter(|i| pred_match_expr(predicate, DeclId::new(*i), ir, &self.strings))
            .count()
    }

}

// ================================================
// Predicate string evaluator (mirrors queries/standard.qlist grammar).
// Generalized over StringTable; kept local to hymeko_py for now.
// ================================================

fn pred_match_expr(expr: &str, did: DeclId, ir: &Ir, st: &StringTable) -> bool {
    expr.split(" AND ").all(|p| pred_match_atom(p.trim(), did, ir, st))
}

fn pred_match_atom(atom: &str, did: DeclId, ir: &Ir, st: &StringTable) -> bool {
    if atom == "ANY" { return true; }
    if let Some(rest) = atom.strip_prefix("KIND(") {
        let name = rest.trim_end_matches(')');
        return pred_decl_kind_name(did, ir, st) == name;
    }
    if let Some(rest) = atom.strip_prefix("INHERITS(") {
        let name = rest.trim_end_matches(')');
        return pred_decl_inherits(did, name, ir, st);
    }
    if let Some(rest) = atom.strip_prefix("SCOPEDIN(") {
        let name = rest.trim_end_matches(')');
        return pred_decl_scoped_in(did, name, ir, st);
    }
    if let Some(rest) = atom.strip_prefix("HASARCREF(") {
        let rest = rest.trim_end_matches(')');
        let (sign_s, inner) = rest.split_once(',').unwrap_or((rest, ""));
        let sign: i8 = sign_s.trim().trim_start_matches('+').parse().unwrap_or(0);
        return pred_has_arc_ref(did, sign, inner.trim(), ir, st);
    }
    false
}

fn pred_decl_kind_name<'a>(did: DeclId, ir: &'a Ir, st: &'a StringTable) -> &'a str {
    let decl = &ir.decl_nodes[did.0];
    match decl.kind {
        DeclKind::Node => {
            if let Some(nid) = ir.as_node(did) {
                if let Some(b) = ir.nodes[nid.0].bases.first() {
                    return st.resolve(ir.decl_nodes[b.target().0].name);
                }
            }
            ""
        }
        DeclKind::Edge => {
            if let Some(eid) = ir.as_edge(did) {
                if let Some(b) = ir.edges[eid.0].bases.first() {
                    return st.resolve(ir.decl_nodes[b.target().0].name);
                }
            }
            ""
        }
        DeclKind::HyperArc => "",
    }
}

fn pred_decl_inherits(did: DeclId, target_name: &str, ir: &Ir, st: &StringTable) -> bool {
    let mut visited = std::collections::HashSet::new();
    let mut stack = vec![did];
    while let Some(d) = stack.pop() {
        if !visited.insert(d) { continue; }
        let decl = &ir.decl_nodes[d.0];
        if st.resolve(decl.name) == target_name { return true; }
        match decl.kind {
            DeclKind::Node => {
                if let Some(nid) = ir.as_node(d) {
                    for b in &ir.nodes[nid.0].bases { stack.push(b.target()); }
                }
            }
            DeclKind::Edge => {
                if let Some(eid) = ir.as_edge(d) {
                    for b in &ir.edges[eid.0].bases { stack.push(b.target()); }
                }
            }
            _ => {}
        }
    }
    false
}

fn pred_decl_scoped_in(did: DeclId, name: &str, ir: &Ir, st: &StringTable) -> bool {
    let mut cur = ir.decl_nodes[did.0].parent;
    while cur.is_some() {
        if pred_decl_inherits(cur, name, ir, st) { return true; }
        if st.resolve(ir.decl_nodes[cur.0].name) == name { return true; }
        cur = ir.decl_nodes[cur.0].parent;
    }
    false
}

fn pred_has_arc_ref(did: DeclId, sign: i8, inner: &str, ir: &Ir, st: &StringTable) -> bool {
    let Some(eid) = ir.as_edge(did) else { return false };
    for &aid in &ir.edges[eid.0].arcs {
        for r in &ir.arcs[aid.0].refs {
            if r.sign() != sign { continue; }
            if pred_match_expr(inner, r.target(), ir, st) { return true; }
        }
    }
    false
}

// ================================================
// DOT emitter — mirrors hymeko_wasm::compile::CompiledDoc::to_dot
// ================================================

fn dot_escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

fn emit_dot_graph(ir: &Ir, st: &StringTable, graph_name: &str) -> String {
    let mut out = String::with_capacity(4096);
    out.push_str(&format!("digraph \"{}\" {{\n", dot_escape(graph_name)));
    out.push_str("  rankdir=LR;\n");
    out.push_str("  node [fontname=\"Helvetica\"];\n\n");

    for rec in &ir.nodes {
        let name = st.resolve(ir.decl_nodes[rec.decl.0].name);
        out.push_str(&format!(
            "  \"n{}\" [label=\"{}\", shape=ellipse, style=filled, fillcolor=\"#EEF1F5\"];\n",
            rec.decl.0, dot_escape(name)
        ));
    }
    for rec in &ir.edges {
        let name = st.resolve(ir.decl_nodes[rec.decl.0].name);
        out.push_str(&format!(
            "  \"e{}\" [label=\"{}\", shape=box, style=\"rounded,filled\", fillcolor=\"#D7E4F5\"];\n",
            rec.decl.0, dot_escape(name)
        ));
    }
    out.push('\n');

    for rec in &ir.edges {
        let eid_num = rec.decl.0;
        for &aid in &rec.arcs {
            for r in &ir.arcs[aid.0].refs {
                let tgt = r.target();
                if tgt.is_none() { continue; }
                let target_is_edge = ir
                    .decl_nodes
                    .get(tgt.0)
                    .map(|d| matches!(d.kind, DeclKind::Edge))
                    .unwrap_or(false);
                let tgt_id = if target_is_edge {
                    format!("e{}", tgt.0)
                } else {
                    format!("n{}", tgt.0)
                };
                let (color, arrowhead, penwidth) = match r.sign() {
                     1 => ("#1b6ca8", "normal", 1.4),
                    -1 => ("#b02a2a", "inv",    1.4),
                     _ => ("#888888", "odot",   1.0),
                };
                out.push_str(&format!(
                    "  \"e{}\" -> \"{}\" [color=\"{}\", arrowhead=\"{}\", penwidth={:.1}];\n",
                    eid_num, tgt_id, color, arrowhead, penwidth
                ));
            }
        }
    }
    out.push_str("}\n");
    out
}

// ================================================
// Snapshot JSON — same schema as hymeko_wasm::compile::SnapshotDto
// ================================================

#[derive(serde::Serialize)]
struct ArcDto<'a> {
    sign: i8,
    target_id: usize,
    target_name: &'a str,
}

#[derive(serde::Serialize)]
struct NodeDto<'a> {
    id: usize,
    name: &'a str,
    kind: &'static str,
    bases: Vec<&'a str>,
    tags: Vec<&'a str>,
    arcs: Vec<ArcDto<'a>>,
}

#[derive(serde::Serialize)]
struct SnapshotDto<'a> {
    node_count: usize,
    edge_count: usize,
    arc_count: usize,
    nodes: Vec<NodeDto<'a>>,
    edges: Vec<NodeDto<'a>>,
}

fn emit_snapshot_json(ir: &Ir, st: &StringTable) -> Result<String, String> {
    let mk = |did: DeclId, with_arcs: bool| -> NodeDto<'_> {
        let decl = &ir.decl_nodes[did.0];
        let name = st.resolve(decl.name);
        let kind_str = match decl.kind {
            DeclKind::Node => "Node",
            DeclKind::Edge => "Edge",
            DeclKind::HyperArc => "HyperArc",
        };
        let bases: Vec<&str> = match decl.kind {
            DeclKind::Node => ir.as_node(did).map(|nid|
                ir.nodes[nid.0].bases.iter()
                    .map(|b| st.resolve(ir.decl_nodes[b.target().0].name)).collect()
            ).unwrap_or_default(),
            DeclKind::Edge => ir.as_edge(did).map(|eid|
                ir.edges[eid.0].bases.iter()
                    .map(|b| st.resolve(ir.decl_nodes[b.target().0].name)).collect()
            ).unwrap_or_default(),
            _ => Vec::new(),
        };
        let tags: Vec<&str> = decl.anno.tags.iter().map(|&s| st.resolve(s)).collect();
        let mut arcs: Vec<ArcDto> = Vec::new();
        if with_arcs {
            if let Some(eid) = ir.as_edge(did) {
                for &aid in &ir.edges[eid.0].arcs {
                    for r in &ir.arcs[aid.0].refs {
                        let tgt = r.target();
                        if !tgt.is_none() {
                            arcs.push(ArcDto {
                                sign: r.sign(),
                                target_id: tgt.0,
                                target_name: st.resolve(ir.decl_nodes[tgt.0].name),
                            });
                        }
                    }
                }
            }
        }
        NodeDto { id: did.0, name, kind: kind_str, bases, tags, arcs }
    };

    let mut nodes = Vec::with_capacity(ir.nodes.len());
    let mut edges = Vec::with_capacity(ir.edges.len());
    for rec in &ir.nodes { nodes.push(mk(rec.decl, false)); }
    for rec in &ir.edges { edges.push(mk(rec.decl, true)); }

    let snap = SnapshotDto {
        node_count: ir.nodes.len(),
        edge_count: ir.edges.len(),
        arc_count:  ir.arcs.len(),
        nodes,
        edges,
    };
    serde_json::to_string(&snap).map_err(|e| format!("json encode: {e}"))
}

/*
#[pyclass]
pub struct HymekoRuntime {
    // The node keeps the IPC connection alive
    node: Arc<Node<ipc::Service>>,
    // The subscriber listens for the specific memory segment
    subscriber: Subscriber<ipc::Service, HypergraphWeights, ()>,
}
*/

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
            val: Float32Array::from(v),
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
            val: Float32Array::from(v),
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
// Shared Memory Expansion Struct (Direct Memory Bridge)
// ================================================

#[pyclass]
pub struct PySharedExpansion {
    base_ptr: usize,
    header: ExpansionHeader,
}

impl PySharedExpansion {
    fn nnz_value(&self) -> usize {
        self.header.nnz as usize
    }

    fn layout_offsets(&self) -> (usize, usize, usize, usize) {
        let header_size = size_of::<ExpansionHeader>();
        let i64_bytes = size_of::<i64>();
        let nnz = self.nnz_value();

        let k_addr = self.base_ptr + header_size;
        let i_addr = k_addr + nnz * i64_bytes;
        let j_addr = i_addr + nnz * i64_bytes;
        let values_addr = j_addr + nnz * i64_bytes;

        (k_addr, i_addr, j_addr, values_addr)
    }
}

#[pymethods]
impl PySharedExpansion {
    #[new]
    pub fn new(base_ptr: usize) -> PyResult<Self> {
        if base_ptr == 0 {
            return Err(PyValueError::new_err("Shared memory pointer cannot be null"));
        }

        let header_ptr = base_ptr as *const ExpansionHeader;
        let header = unsafe { header_ptr.read() };

        Ok(Self { base_ptr, header })
    }

    /// Returns the raw memory address of the payload
    #[getter]
    pub fn payload_address(&self) -> usize {
        self.base_ptr
    }

    #[getter]
    pub fn dims(&self) -> PyResult<(i64, i64, i64)> {
        Ok((
            i64::try_from(self.header.dim_k).map_err(|_| PyValueError::new_err("dim_k exceeds i64"))?,
            i64::try_from(self.header.dim_i).map_err(|_| PyValueError::new_err("dim_i exceeds i64"))?,
            i64::try_from(self.header.dim_j).map_err(|_| PyValueError::new_err("dim_j exceeds i64"))?,
        ))
    }

    #[getter]
    pub fn nnz(&self) -> usize {
        self.nnz_value()
    }

    /// Returns the underlying pyarrow buffers for (k, i, j, values).
    pub fn buffers<'py>(slf: PyRef<'py, Self>, py: Python<'py>)
        -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>, Bound<'py, PyAny>, Bound<'py, PyAny>)>
    {
        let pyarrow = PyModule::import(py, "pyarrow")
            .map_err(|_| PyValueError::new_err("pyarrow module is required for shared expansions"))?;
        let foreign_buffer = pyarrow.getattr("foreign_buffer")
            .map_err(|_| PyValueError::new_err("pyarrow.foreign_buffer is missing"))?;

        let nnz = slf.nnz_value();
        let i64_bytes = nnz * size_of::<i64>();
        let f32_bytes = nnz * size_of::<f32>();
        let (k_addr, i_addr, j_addr, v_addr) = slf.layout_offsets();
        let owner = slf.into_pyobject(py)?.unbind();

         let make_buffer = |addr: usize, len: usize| -> PyResult<Bound<'py, PyAny>> {
             foreign_buffer.call1((addr, len, owner.clone_ref(py)))
         };

        let k_buf = make_buffer(k_addr, i64_bytes)?;
        let i_buf = make_buffer(i_addr, i64_bytes)?;
        let j_buf = make_buffer(j_addr, i64_bytes)?;
        let v_buf = make_buffer(v_addr, f32_bytes)?;

        Ok((k_buf, i_buf, j_buf, v_buf))
    }
}


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
        let soa = core_coo.into_soa();

        // Step 5: Thin Arrow wrapper
        Ok(PyTensorCoo3D::from_raw(
            soa.k, soa.i, soa.j, soa.v,
            (soa.num_slices, soa.dim_i, soa.dim_j)
        ))
    }

    pub fn compile_clique_expansion(&mut self, py_ir: &PyHypergraphIR) -> PyResult<PySparseMatrix2D> {
        let ir = &py_ir.compiled.ir;

        let core_coo = self.inner.compile_clique_expansion_core::<f32>(ir);
        let soa = core_coo.into_soa();

        Ok(PySparseMatrix2D::from_raw(
            soa.i, soa.j, soa.v,
            (soa.dim_i, soa.dim_j)
        ))
    }

    /// Natively extracts the 3D Edge-Colored Clique Expansion Tensor.
    /// Yields an E x V x V tensor where each slice `k` is the fully connected clique for edge `k`.
    pub fn compile_clique_tensor_expansion(&mut self, py_ir: &PyHypergraphIR) -> PyResult<PyTensorCoo3D> {
        let ir = &py_ir.compiled.ir;
        let core_coo = self.inner.compile_clique_expansion_core::<f32>(ir);
        let soa = core_coo.into_soa();

        Ok(PyTensorCoo3D::from_raw(
            soa.k, soa.i, soa.j, soa.v,
            (soa.num_slices, soa.dim_i, soa.dim_j)
        ))
    }
}

