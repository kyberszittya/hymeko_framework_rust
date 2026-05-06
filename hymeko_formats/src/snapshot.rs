//! Snapshot + DOT emitter — single source of truth for the
//! signed-incidence-hypergraph IR's debug visualisation.
//!
//! Replaces byte-identical (modulo lifetime annotations) copies that
//! previously lived in `hymeko_py::interface_python::api` and
//! `hymeko_wasm::compile`.  Both call-sites now delegate to this
//! module.
//!
//! Owned `String` shapes (matching the wasm-bindgen surface that the
//! browser demo consumes) — the small per-call allocation cost is
//! negligible compared to serde_json::to_string at the same call site.

use hymeko::common::ids::DeclId;
use hymeko::ir::ir::{DeclKind, Ir};
use hymeko::resolution::string_table::StringTable;

use serde::{Deserialize, Serialize};


// ─── Snapshot DTOs (JSON-shaped) ────────────────────────────────────


#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ArcDto {
    pub sign: i8,
    pub target_id: usize,
    pub target_name: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct NodeDto {
    pub id: usize,
    pub name: String,
    /// "Node" | "Edge" | "HyperArc"
    pub kind: String,
    /// Base types this decl inherits from (resolved, first-level).
    pub bases: Vec<String>,
    /// Annotation tags attached at declaration (e.g. `<temperature>`).
    pub tags: Vec<String>,
    /// Signed arc references — populated only for Edge decls.
    pub arcs: Vec<ArcDto>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SnapshotDto {
    pub node_count: usize,
    pub edge_count: usize,
    pub arc_count: usize,
    /// All `Node` decls (plain vertices).
    pub nodes: Vec<NodeDto>,
    /// All `Edge` decls (hyperedges).
    pub edges: Vec<NodeDto>,
}


// ─── Snapshot construction ──────────────────────────────────────────


/// Build a `SnapshotDto` from an IR + its string table.
///
/// Used by both the Python wheel (`PyHypergraphIR::snapshot_json`) and
/// the wasm demo (`CompiledDoc::snapshot_json`) — exposed via
/// `snapshot_json()` below.
pub fn snapshot(ir: &Ir, st: &StringTable) -> SnapshotDto {
    let mk = |did: DeclId, with_arcs: bool| -> NodeDto {
        let decl = &ir.decl_nodes[did.0];
        let name = st.resolve(decl.name).to_string();
        let kind_str = match decl.kind {
            DeclKind::Node => "Node",
            DeclKind::Edge => "Edge",
            DeclKind::HyperArc => "HyperArc",
        };

        let bases: Vec<String> = match decl.kind {
            DeclKind::Node => ir
                .as_node(did)
                .map(|nid| ir.nodes[nid.0].bases.iter()
                    .map(|b| st.resolve(ir.decl_nodes[b.target().0].name).to_string())
                    .collect())
                .unwrap_or_default(),
            DeclKind::Edge => ir
                .as_edge(did)
                .map(|eid| ir.edges[eid.0].bases.iter()
                    .map(|b| st.resolve(ir.decl_nodes[b.target().0].name).to_string())
                    .collect())
                .unwrap_or_default(),
            _ => Vec::new(),
        };

        let tags: Vec<String> = decl.anno.tags.iter()
            .map(|&s| st.resolve(s).to_string())
            .collect();

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
                                target_name: st.resolve(ir.decl_nodes[tgt.0].name).to_string(),
                            });
                        }
                    }
                }
            }
        }
        NodeDto { id: did.0, name, kind: kind_str.to_string(), bases, tags, arcs }
    };

    let mut nodes = Vec::with_capacity(ir.nodes.len());
    let mut edges = Vec::with_capacity(ir.edges.len());
    for rec in &ir.nodes { nodes.push(mk(rec.decl, false)); }
    for rec in &ir.edges { edges.push(mk(rec.decl, true)); }

    SnapshotDto {
        node_count: ir.nodes.len(),
        edge_count: ir.edges.len(),
        arc_count:  ir.arcs.len(),
        nodes,
        edges,
    }
}


/// JSON-encode a snapshot.  Returns `Err` only if serde itself
/// somehow fails — the schema is fixed so this is effectively
/// infallible for well-formed IRs.
pub fn snapshot_json(ir: &Ir, st: &StringTable) -> Result<String, String> {
    serde_json::to_string(&snapshot(ir, st))
        .map_err(|e| format!("json encode: {e}"))
}


// ─── DOT (Graphviz) emitter ─────────────────────────────────────────


/// Escape `\` and `"` for DOT label syntax.
pub fn dot_escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}


/// Emit a DOT (Graphviz) string for the signed-incidence hypergraph.
///
/// Vertices render as ellipses, hyperedges as rounded boxes; signed
/// arcs colour-coded (blue +1, red -1, grey ~0) with arrow heads
/// matching the sign (normal, inv, odot).
pub fn emit_dot_graph(ir: &Ir, st: &StringTable, graph_name: &str) -> String {
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
