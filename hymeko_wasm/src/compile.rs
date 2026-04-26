//! Parse / compile / query / emit surface for the browser.
//!
//! This module is native-testable (no wasm-bindgen here); the
//! `wasm::CompiledIR` façade in `crate::wasm` forwards to the methods
//! here. Everything is built against the `MemProvider` source backend
//! so no filesystem access is needed — the caller passes a raw
//! `.hymeko` source string and gets back a compiled IR wrapper.

use std::sync::Arc;

use hymeko::common::ids::DeclId;
use hymeko::ir::ir::{DeclKind, Ir};
use hymeko::module_store::module_store::{CompiledProgram, HymekoParser, ModuleStore};
use hymeko::module_store::source_provider::MemProvider;
use hymeko::resolution::string_table::StringTable;
use hymeko_formats::sdf::generate_sdf;
use hymeko_formats::urdf::generate_urdf;
use parser::ast::AstStr;
use serde::{Deserialize, Serialize};

/// Parser glue — mirrors `hymeko_py::interface_python::api`'s
/// `RealParser` but staying inside the wasm crate so we can keep
/// `hymeko_core` free of its optional `util` module if we ever prune it.
pub struct LalrpopParser;

impl HymekoParser for LalrpopParser {
    fn parse<'a>(&self, src: &'a str) -> Result<AstStr<'a>, String> {
        parser::parse_description(src).map_err(|e| format!("{e:?}"))
    }
}

/// Compiled-IR wrapper exposed to both native tests and (via a thin
/// `wasm_bindgen` façade) the browser.
pub struct CompiledDoc {
    pub compiled: Arc<CompiledProgram>,
    pub strings:  StringTable,
}

/// Parse a single-file `.hymeko` source string and return a compiled IR.
///
/// The virtual filename used internally is `inline.hymeko`; it's
/// exposed here so the error messages are predictable.
pub fn compile_source(source: &str) -> Result<CompiledDoc, String> {
    let mut store = ModuleStore::new(MemProvider::default(), LalrpopParser);
    store.provider_mut().insert_file("inline.hymeko", source);
    let compiled = store
        .compile(std::path::Path::new("inline.hymeko"))
        .map_err(|e| format!("compile error: {e:?}"))?;
    let strings = StringTable::from_interner(&store.it);
    Ok(CompiledDoc { compiled, strings })
}

// --------------------------------------------------------------------- //
// Snapshot JSON — shape tuned for a force-directed graph viewer.
// --------------------------------------------------------------------- //

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

impl CompiledDoc {
    pub fn node_count(&self) -> usize { self.compiled.ir.nodes.len() }
    pub fn edge_count(&self) -> usize { self.compiled.ir.edges.len() }
    pub fn arc_count(&self) -> usize  { self.compiled.ir.arcs.len() }

    pub fn snapshot(&self) -> SnapshotDto {
        let ir = &self.compiled.ir;
        let st = &self.strings;

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

            let tags: Vec<String> = decl
                .anno
                .tags
                .iter()
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

    pub fn snapshot_json(&self) -> Result<String, String> {
        serde_json::to_string(&self.snapshot())
            .map_err(|e| format!("json encode: {e}"))
    }

    pub fn to_urdf(&self, robot_name: &str) -> String {
        generate_urdf(&self.compiled.ir, &self.strings, robot_name)
    }

    pub fn to_sdf(&self, model_name: &str) -> String {
        generate_sdf(&self.compiled.ir, &self.strings, model_name)
    }

    pub fn to_dot(&self, graph_name: &str) -> String {
        // Hand-rolled DOT emitter — no template path in wasm (the
        // TransformRegistry template dispatcher wants filesystem access
        // to workspace `transforms/`). Produces the same kind of
        // signed-incidence graph as the template output: vertices as
        // ellipses, hyperedges as boxes, signed arcs as coloured arrows.
        let ir = &self.compiled.ir;
        let st = &self.strings;
        let mut out = String::with_capacity(4096);
        out.push_str(&format!("digraph \"{}\" {{\n", xml_escape(graph_name)));
        out.push_str("  rankdir=LR;\n");
        out.push_str("  node [fontname=\"Helvetica\"];\n\n");

        for rec in &ir.nodes {
            let name = st.resolve(ir.decl_nodes[rec.decl.0].name);
            out.push_str(&format!(
                "  \"n{}\" [label=\"{}\", shape=ellipse, style=filled, fillcolor=\"#EEF1F5\"];\n",
                rec.decl.0, xml_escape(name)
            ));
        }
        for rec in &ir.edges {
            let name = st.resolve(ir.decl_nodes[rec.decl.0].name);
            out.push_str(&format!(
                "  \"e{}\" [label=\"{}\", shape=box, style=\"rounded,filled\", fillcolor=\"#D7E4F5\"];\n",
                rec.decl.0, xml_escape(name)
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
                         1 => ("#1b6ca8", "normal", 1.4),   // +
                        -1 => ("#b02a2a", "inv",    1.4),   // -
                         _ => ("#888888", "odot",   1.0),   // ~
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

    // ------------------------ predicate queries ------------------------

    /// Evaluate the `queries/standard.qlist` predicate language and
    /// return the names of all matching decls.
    ///
    /// Supported atoms:
    ///   KIND(<name>)                 — first inherited base equals <name>
    ///   INHERITS(<name>)             — transitively inherits <name>
    ///   SCOPEDIN(<name>)             — ancestor inherits <name>
    ///   HASARCREF(<sign>, <inner>)   — edge with sign-matching arc-ref
    ///                                  pointing at a decl matching <inner>
    ///   <a> AND <b>                  — conjunction
    ///   ANY                          — always true
    pub fn query(&self, predicate: &str) -> Vec<String> {
        let ir = &self.compiled.ir;
        let st = &self.strings;
        let mut out = Vec::new();
        for i in 0..ir.decl_nodes.len() {
            let did = DeclId::new(i);
            if pred_match_expr(predicate, did, ir, st) {
                out.push(st.resolve(ir.decl_nodes[i].name).to_string());
            }
        }
        out
    }

    pub fn query_count(&self, predicate: &str) -> usize {
        let ir = &self.compiled.ir;
        (0..ir.decl_nodes.len())
            .filter(|i| pred_match_expr(predicate, DeclId::new(*i), ir, &self.strings))
            .count()
    }
}

// ------------------------------------------------------------------- //
// Predicate-string evaluator — same grammar subset as hymeko_py.       //
// ------------------------------------------------------------------- //

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

fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
     .replace('<', "&lt;")
     .replace('>', "&gt;")
     .replace('"', "&quot;")
}
