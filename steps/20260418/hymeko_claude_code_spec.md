# HyMeKo WASM Editor + Visualization — Claude Code Integration Spec

## Context
HyMeKo is a Rust workspace (~15k+ lines) implementing a compiled, statically-typed
description language for signed-incidence directed hypergraphs (G-SPHF).
This document is the technical specification for building the WASM visual editor,
MCP server, and wire layer on top of the existing codebase.

---

## 1. Workspace Structure Target

```
hymeko_workspace/
├── hymeko_core/          ← EXISTS — stabilize public API first
├── hymeko_ir/            ← EXTRACT from core — standalone IR crate
├── hymeko_parser/        ← EXISTS (lalrpop) — expose clean T2M API
├── hymeko_emitter/       ← NEW — M2T: .hymeko, SysML v2, Rust stubs, Lean 4
├── hymeko_wasm/          ← NEW — wasm-bindgen exposure of IR
├── hymeko_server/        ← NEW — Axum: serves WASM app + REST file I/O
├── hymeko_mcp/           ← NEW — MCP server via rmcp crate
├── hymeko_wire/          ← NEW — CBOR + zstd + Reed-Solomon serialization
└── hymeko_p2p/           ← FUTURE — iroh P2P gossip layer
```

---

## 2. Step 0 — IR Extraction (prerequisite for everything)

Before touching WASM, extract a clean IR crate.

### hymeko_ir/Cargo.toml
```toml
[package]
name = "hymeko_ir"
version = "0.1.0"
edition = "2021"

[dependencies]
serde = { version = "1", features = ["derive"] }
slotmap = "1"
```

### hymeko_ir/src/lib.rs
```rust
use slotmap::{SlotMap, new_key_type};
use serde::{Serialize, Deserialize};

new_key_type! {
    pub struct VertexId;
    pub struct EdgeId;
    pub struct PatchId;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vertex {
    pub name: String,
    pub level: i8,           // G-SPHF level: -2 through 8
    pub attributes: Vec<Attribute>,
    pub position: Option<Position>, // canvas layout hint
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperEdge {
    pub incident: Vec<VertexId>,  // N vertices — not just 2
    pub sign: f64,                // signed: positive or negative
    pub weight: f64,
    pub patch_id: Option<PatchId>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Patch {
    pub id: PatchId,
    pub level: i8,
    pub vertices: Vec<VertexId>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Attribute {
    pub key: String,
    pub value: AttributeValue,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AttributeValue {
    Int(i64),
    Float(f64),
    Str(String),
    Bool(bool),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    pub x: f64,
    pub y: f64,
}

/// The core IR — single source of truth
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct HyMeKoIR {
    pub vertices: SlotMap<VertexId, Vertex>,
    pub hyperedges: SlotMap<EdgeId, HyperEdge>,
    pub patches: SlotMap<PatchId, Patch>,
}

/// Atomic mutation unit — also the P2P gossip unit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IRDelta {
    AddVertex { data: Vertex },
    RemoveVertex { id: VertexId },
    AddHyperEdge { data: HyperEdge },
    RemoveEdge { id: EdgeId },
    MoveVertex { id: VertexId, position: Position },
    UpdateWeight { id: EdgeId, weight: f64 },
    UpdateSign { id: EdgeId, sign: f64 },
    AttachAttribute { id: VertexId, attr: Attribute },
    DetachAttribute { id: VertexId, key: String },
    AddPatch { data: Patch },
}

impl HyMeKoIR {
    pub fn new() -> Self { Self::default() }

    pub fn apply(&mut self, delta: IRDelta) -> Result<(), IRError> {
        match delta {
            IRDelta::AddVertex { data } => { self.vertices.insert(data); Ok(()) }
            IRDelta::RemoveVertex { id } => {
                self.vertices.remove(id).ok_or(IRError::NotFound)?; Ok(())
            }
            IRDelta::AddHyperEdge { data } => { self.hyperedges.insert(data); Ok(()) }
            IRDelta::MoveVertex { id, position } => {
                self.vertices.get_mut(id)
                    .ok_or(IRError::NotFound)?
                    .position = Some(position);
                Ok(())
            }
            IRDelta::UpdateWeight { id, weight } => {
                self.hyperedges.get_mut(id)
                    .ok_or(IRError::NotFound)?
                    .weight = weight;
                Ok(())
            }
            _ => Ok(()) // implement remaining arms
        }
    }

    pub fn apply_batch(&mut self, deltas: Vec<IRDelta>) -> Result<(), IRError> {
        for delta in deltas { self.apply(delta)?; }
        Ok(())
    }
}

#[derive(Debug, thiserror::Error)]
pub enum IRError {
    #[error("entity not found")]
    NotFound,
    #[error("invalid delta: {0}")]
    Invalid(String),
}
```

---

## 3. hymeko_emitter — M2T and Synthesis

### hymeko_emitter/src/lib.rs
```rust
use hymeko_ir::{HyMeKoIR, VertexId, EdgeId};

pub fn emit_hymeko(ir: &HyMeKoIR) -> String {
    // serialize IR to .hymeko textual notation
    // deterministic output — same IR always produces same text
    let mut out = String::new();
    for (_, vertex) in &ir.vertices {
        out.push_str(&format!("vertex {} level {} {{\n", vertex.name, vertex.level));
        for attr in &vertex.attributes {
            out.push_str(&format!("    {} = {:?};\n", attr.key, attr.value));
        }
        out.push_str("}\n");
    }
    // hyperedges
    for (_, edge) in &ir.hyperedges {
        out.push_str(&format!("hyperedge sign({}) weight({}) {{\n",
            edge.sign, edge.weight));
        out.push_str("}\n");
    }
    out
}

pub fn emit_sysml(ir: &HyMeKoIR) -> String {
    // G-SPHF hyperedges encoded as annotated SysML v2 connections
    // Uses metadata profile for hyperedge arity (SysML v2 has no native hyperedge)
    let mut out = String::new();
    out.push_str("// Generated by HyMeKo — G-SPHF → SysML v2\n\n");
    out.push_str("metadata def HyperedgeAnnotation {\n");
    out.push_str("    attribute arity : Integer;\n");
    out.push_str("    attribute sign : Real;\n");
    out.push_str("    attribute patch_id : String;\n");
    out.push_str("}\n\n");

    for (_, vertex) in &ir.vertices {
        out.push_str(&format!("part def {} {{\n", vertex.name));
        out.push_str(&format!("    // G-SPHF level: {}\n", vertex.level));
        out.push_str("}\n\n");
    }

    for (_, edge) in &ir.hyperedges {
        out.push_str(&format!(
            "connection def HyperEdge_{} {{\n    @HyperedgeAnnotation {{ arity = {}; sign = {}; }}\n}}\n\n",
            edge.incident.len(),
            edge.incident.len(),
            edge.sign
        ));
    }
    out
}

pub fn emit_rust_stubs(ir: &HyMeKoIR) -> String {
    // Generate Rust trait skeletons from IR structure
    let mut out = String::new();
    out.push_str("// Generated by HyMeKo — IR → Rust stubs\n\n");
    for (_, vertex) in &ir.vertices {
        let trait_name = to_pascal_case(&vertex.name);
        out.push_str(&format!("pub trait {} {{\n", trait_name));
        out.push_str("    fn process(&self) -> Result<(), Box<dyn std::error::Error>>;\n");
        out.push_str("}\n\n");
    }
    out
}

pub fn emit_lean4(ir: &HyMeKoIR) -> String {
    // Generate Lean 4 proof obligation templates
    let mut out = String::new();
    out.push_str("-- Generated by HyMeKo — IR → Lean 4 obligations\n\n");
    out.push_str("import Mathlib\n\n");
    for (_, vertex) in &ir.vertices {
        out.push_str(&format!("-- Obligation: {} level invariant\n", vertex.name));
        out.push_str(&format!("theorem {}_level_invariant : True := by\n  trivial\n\n",
            vertex.name.to_lowercase()));
    }
    out
}

fn to_pascal_case(s: &str) -> String {
    s.split('_')
        .map(|w| {
            let mut c = w.chars();
            match c.next() {
                None => String::new(),
                Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
            }
        })
        .collect()
}
```

---

## 4. hymeko_wasm — wasm-bindgen Layer

### hymeko_wasm/Cargo.toml
```toml
[package]
name = "hymeko_wasm"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib", "rlib"]

[dependencies]
hymeko_ir = { path = "../hymeko_ir" }
hymeko_emitter = { path = "../hymeko_emitter" }
wasm-bindgen = "0.2"
serde = { version = "1", features = ["derive"] }
serde-wasm-bindgen = "0.6"
console_error_panic_hook = "0.1"

[profile.release]
opt-level = "z"    # optimize for size
lto = true
```

### hymeko_wasm/src/lib.rs
```rust
use wasm_bindgen::prelude::*;
use hymeko_ir::{HyMeKoIR, IRDelta, Vertex, HyperEdge, Position, Attribute, AttributeValue};
use hymeko_emitter::{emit_hymeko, emit_sysml, emit_rust_stubs, emit_lean4};

// Better panic messages in browser console
#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
}

#[wasm_bindgen]
pub struct EditorSession {
    ir: HyMeKoIR,
}

#[wasm_bindgen]
impl EditorSession {
    #[wasm_bindgen(constructor)]
    pub fn new() -> EditorSession {
        EditorSession { ir: HyMeKoIR::new() }
    }

    /// Add a vertex — returns JS-safe string ID
    pub fn add_vertex(&mut self, name: &str, level: i8) -> JsValue {
        let vertex = Vertex {
            name: name.to_string(),
            level,
            attributes: vec![],
            position: None,
        };
        let id = self.ir.vertices.insert(vertex);
        serde_wasm_bindgen::to_value(&id).unwrap()
    }

    /// Add a hyperedge connecting N vertices
    pub fn add_hyperedge(&mut self, incident_ids: JsValue, sign: f64, weight: f64) -> JsValue {
        let ids: Vec<hymeko_ir::VertexId> = serde_wasm_bindgen::from_value(incident_ids).unwrap();
        let edge = HyperEdge {
            incident: ids,
            sign,
            weight,
            patch_id: None,
        };
        let id = self.ir.hyperedges.insert(edge);
        serde_wasm_bindgen::to_value(&id).unwrap()
    }

    /// Move vertex — called by canvas drag
    pub fn move_vertex(&mut self, id: JsValue, x: f64, y: f64) -> Result<(), JsValue> {
        let id: hymeko_ir::VertexId = serde_wasm_bindgen::from_value(id)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        self.ir.apply(IRDelta::MoveVertex {
            id,
            position: Position { x, y },
        }).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Apply arbitrary delta from JS
    pub fn apply_delta(&mut self, delta: JsValue) -> Result<(), JsValue> {
        let delta: IRDelta = serde_wasm_bindgen::from_value(delta)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        self.ir.apply(delta)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Snapshot — full IR state as JS object for canvas rendering
    pub fn snapshot(&self) -> JsValue {
        serde_wasm_bindgen::to_value(&self.ir).unwrap()
    }

    /// T2M — parse .hymeko text into IR
    pub fn parse_text(&mut self, src: &str) -> Result<(), JsValue> {
        // TODO: call hymeko_parser
        // placeholder until parser is wired
        Ok(())
    }

    // M2T emitters
    pub fn emit_hymeko(&self) -> String { emit_hymeko(&self.ir) }
    pub fn emit_sysml(&self) -> String { emit_sysml(&self.ir) }
    pub fn emit_rust_stubs(&self) -> String { emit_rust_stubs(&self.ir) }
    pub fn emit_lean4(&self) -> String { emit_lean4(&self.ir) }

    /// Export IR as CBOR bytes
    pub fn export_cbor(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        ciborium::into_writer(&self.ir, &mut buf).unwrap();
        buf
    }

    /// Import IR from CBOR bytes
    pub fn import_cbor(&mut self, data: &[u8]) -> Result<(), JsValue> {
        self.ir = ciborium::from_reader(data)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(())
    }
}
```

### Build command
```bash
wasm-pack build hymeko_wasm --target web --out-dir ../hymeko_server/static/pkg
```

---

## 5. hymeko_server — Axum Static Server + REST

### hymeko_server/Cargo.toml
```toml
[package]
name = "hymeko_server"
version = "0.1.0"
edition = "2021"

[[bin]]
name = "hymeko_server"
path = "src/main.rs"

[dependencies]
axum = "0.7"
tokio = { version = "1", features = ["full"] }
tower-http = { version = "0.5", features = ["fs", "cors"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
tokio-fs = "0.1"
```

### hymeko_server/src/main.rs
```rust
use axum::{
    routing::{get, post},
    Router,
    Json,
    extract::Path,
};
use tower_http::{services::ServeDir, cors::CorsLayer};
use std::net::SocketAddr;

#[tokio::main]
async fn main() {
    let app = Router::new()
        // REST: file I/O (bridges WASM ↔ filesystem)
        .route("/api/files/:filename", get(read_file))
        .route("/api/files/:filename", post(write_file))
        .route("/api/workspace", get(list_workspace))
        // serve WASM app + static assets
        .nest_service("/", ServeDir::new("static"))
        .layer(CorsLayer::permissive());

    let addr = SocketAddr::from(([127, 0, 0, 1], 3000));
    println!("HyMeKo editor running at http://localhost:{}", addr.port());
    axum::Server::bind(&addr)
        .serve(app.into_make_service())
        .await
        .unwrap();
}

async fn read_file(Path(filename): Path<String>) -> Result<String, String> {
    tokio::fs::read_to_string(&filename)
        .await
        .map_err(|e| e.to_string())
}

async fn write_file(
    Path(filename): Path<String>,
    body: String,
) -> Result<(), String> {
    tokio::fs::write(&filename, body)
        .await
        .map_err(|e| e.to_string())
}

async fn list_workspace() -> Json<Vec<String>> {
    // list .hymeko and .sysml files in working directory
    let mut files = vec![];
    if let Ok(mut entries) = tokio::fs::read_dir(".").await {
        while let Ok(Some(entry)) = entries.next_entry().await {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.ends_with(".hymeko") || name.ends_with(".sysml") {
                files.push(name);
            }
        }
    }
    Json(files)
}
```

---

## 6. Frontend — React + SVG Hypergraph Canvas

### hymeko_server/static/index.html
```html
<!DOCTYPE html>
<html>
<head>
    <title>HyMeKo Editor</title>
    <meta charset="utf-8">
</head>
<body>
    <div id="root"></div>
    <script type="module" src="/app.js"></script>
</body>
</html>
```

### hymeko_server/static/app.js (React via CDN, no bundler needed for MVP)
```javascript
import init, { EditorSession } from '/pkg/hymeko_wasm.js';

// Hyperedge rendering — convex hull approach
// This is the key: hyperedges as filled polygons, not lines
function renderHyperedge(ctx, vertices, edge) {
    const points = edge.incident.map(id => vertices[id].position);
    if (points.length < 2) return;
    
    if (points.length === 2) {
        // binary edge — simple line
        ctx.beginPath();
        ctx.moveTo(points[0].x, points[0].y);
        ctx.lineTo(points[1].x, points[1].y);
        ctx.strokeStyle = edge.sign >= 0 ? '#4A90D9' : '#E74C3C';
        ctx.lineWidth = Math.abs(edge.weight) * 2;
        ctx.stroke();
        return;
    }

    // N-ary hyperedge — convex hull as filled polygon
    const hull = convexHull(points);
    ctx.beginPath();
    ctx.moveTo(hull[0].x, hull[0].y);
    hull.slice(1).forEach(p => ctx.lineTo(p.x, p.y));
    ctx.closePath();
    
    // fill: low opacity
    ctx.fillStyle = edge.sign >= 0
        ? 'rgba(74, 144, 217, 0.15)'   // positive = blue
        : 'rgba(231, 76, 60, 0.15)';   // negative = red
    ctx.fill();
    
    // border: sign color, weight = thickness
    ctx.strokeStyle = edge.sign >= 0 ? '#4A90D9' : '#E74C3C';
    ctx.lineWidth = Math.max(1, Math.abs(edge.weight));
    ctx.stroke();
}

// Simple convex hull (Graham scan)
function convexHull(points) {
    if (points.length <= 3) return points;
    const sorted = [...points].sort((a, b) => a.x - b.x || a.y - b.y);
    const cross = (O, A, B) =>
        (A.x - O.x) * (B.y - O.y) - (A.y - O.y) * (B.x - O.x);
    const lower = [];
    for (const p of sorted) {
        while (lower.length >= 2 && cross(lower[lower.length-2], lower[lower.length-1], p) <= 0)
            lower.pop();
        lower.push(p);
    }
    return lower;
}

async function main() {
    await init();
    const session = new EditorSession();
    
    // demo: build a small hypergraph
    const v1 = session.add_vertex("SystemA", 0);
    const v2 = session.add_vertex("SystemB", 0);
    const v3 = session.add_vertex("SystemC", 1);
    
    session.add_hyperedge([v1, v2, v3], 1.0, 1.5);
    session.add_hyperedge([v1, v2], -1.0, 0.8);
    
    // get snapshot for rendering
    const snapshot = session.snapshot();
    console.log("IR snapshot:", snapshot);
    console.log("HyMeKo text:\n", session.emit_hymeko());
    console.log("SysML v2:\n", session.emit_sysml());
    
    // TODO: wire up React canvas component
    document.getElementById('root').innerHTML =
        '<pre>' + session.emit_hymeko() + '</pre>' +
        '<hr/>' +
        '<pre>' + session.emit_sysml() + '</pre>';
}

main();
```

---

## 7. hymeko_mcp — MCP Server

### hymeko_mcp/Cargo.toml
```toml
[package]
name = "hymeko_mcp"
version = "0.1.0"
edition = "2021"

[[bin]]
name = "hymeko_mcp"
path = "src/main.rs"

[dependencies]
hymeko_ir = { path = "../hymeko_ir" }
hymeko_emitter = { path = "../hymeko_emitter" }
rmcp = { version = "0.1", features = ["server", "transport-io"] }
tokio = { version = "1", features = ["full"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
```

### hymeko_mcp/src/main.rs
```rust
use rmcp::{ServerHandler, tool, model::*, service::RequestContext};
use hymeko_ir::{HyMeKoIR, IRDelta, Vertex, HyperEdge};
use hymeko_emitter::{emit_hymeko, emit_sysml};
use std::sync::{Arc, Mutex};

#[derive(Clone)]
struct HyMeKoMCPServer {
    ir: Arc<Mutex<HyMeKoIR>>,
}

#[rmcp::tool(tool_box)]
impl HyMeKoMCPServer {
    #[tool(description = "Add a vertex to the HyMeKo IR")]
    async fn add_vertex(
        &self,
        #[tool(description = "Vertex name")] name: String,
        #[tool(description = "G-SPHF level (-2 to 8)")] level: i8,
    ) -> String {
        let mut ir = self.ir.lock().unwrap();
        let id = ir.vertices.insert(Vertex {
            name: name.clone(),
            level,
            attributes: vec![],
            position: None,
        });
        format!("Added vertex '{}' at level {} with id {:?}", name, level, id)
    }

    #[tool(description = "Add a hyperedge connecting multiple vertices")]
    async fn add_hyperedge(
        &self,
        #[tool(description = "Comma-separated vertex names to connect")] vertices: String,
        #[tool(description = "Sign: positive (1.0) or negative (-1.0)")] sign: f64,
        #[tool(description = "Edge weight")] weight: f64,
    ) -> String {
        let ir = self.ir.lock().unwrap();
        // find vertex ids by name
        let ids: Vec<_> = ir.vertices.iter()
            .filter(|(_, v)| vertices.split(',').any(|n| n.trim() == v.name))
            .map(|(id, _)| id)
            .collect();
        drop(ir);
        let mut ir = self.ir.lock().unwrap();
        ir.hyperedges.insert(HyperEdge {
            incident: ids.clone(),
            sign,
            weight,
            patch_id: None,
        });
        format!("Added hyperedge connecting {} vertices, sign={}, weight={}", ids.len(), sign, weight)
    }

    #[tool(description = "Get current IR as HyMeKo text")]
    async fn emit_hymeko(&self) -> String {
        emit_hymeko(&self.ir.lock().unwrap())
    }

    #[tool(description = "Get current IR as SysML v2 text")]
    async fn emit_sysml(&self) -> String {
        emit_sysml(&self.ir.lock().unwrap())
    }

    #[tool(description = "Get IR snapshot as JSON")]
    async fn snapshot(&self) -> String {
        let ir = self.ir.lock().unwrap();
        serde_json::to_string_pretty(&*ir).unwrap_or_default()
    }

    #[tool(description = "Clear the IR and start fresh")]
    async fn reset(&self) -> String {
        *self.ir.lock().unwrap() = HyMeKoIR::new();
        "IR reset to empty state".to_string()
    }
}

#[tokio::main]
async fn main() {
    let server = HyMeKoMCPServer {
        ir: Arc::new(Mutex::new(HyMeKoIR::new())),
    };
    
    // stdio transport — works with Claude Desktop and Claude Code
    let transport = rmcp::transport::io::stdio();
    rmcp::serve_server(server, transport).await.unwrap();
}
```

---

## 8. hymeko_wire — CBOR Serialization Layer

### hymeko_wire/Cargo.toml
```toml
[package]
name = "hymeko_wire"
version = "0.1.0"
edition = "2021"

[dependencies]
hymeko_ir = { path = "../hymeko_ir" }
ciborium = "0.2"          # CBOR RFC 8949
bytes = "1"               # zero-copy buffer
serde = { version = "1", features = ["derive"] }
zstd = "0.13"             # compression RFC 8878
xxhash-rust = { version = "0.8", features = ["xxh3"] }
```

### hymeko_wire/src/lib.rs
```rust
use bytes::Bytes;
use hymeko_ir::{HyMeKoIR, IRDelta};
use xxhash_rust::xxh3::xxh3_32;

/// Packet header — precedes every wire packet
#[repr(C)]
pub struct PacketHeader {
    pub magic: u32,        // 0x484D4B4F ("HMKO")
    pub version: u16,      // schema version
    pub ir_level: u8,      // G-SPHF level 0-8
    pub flags: u8,         // compression, FEC flags
    pub patch_id: u64,     // patch locality hint for sharding
    pub delta_seq: u64,    // CRDT sequence number
    pub checksum: u32,     // xxhash3 of payload
}

pub const MAGIC: u32 = 0x484D4B4F;

/// Zero-copy view over raw CBOR bytes
pub struct IRDeltaView<'a> {
    raw: &'a [u8],
    // offsets into raw buffer — computed once on construction
    payload_offset: usize,
}

impl<'a> IRDeltaView<'a> {
    pub fn from_bytes(raw: &'a [u8]) -> Result<Self, WireError> {
        if raw.len() < std::mem::size_of::<PacketHeader>() {
            return Err(WireError::TooShort);
        }
        let magic = u32::from_le_bytes(raw[0..4].try_into().unwrap());
        if magic != MAGIC {
            return Err(WireError::BadMagic);
        }
        Ok(Self { raw, payload_offset: std::mem::size_of::<PacketHeader>() })
    }

    /// Read delta without allocation — directly from buffer
    pub fn read_delta(&self) -> Result<IRDelta, WireError> {
        ciborium::from_reader(&self.raw[self.payload_offset..])
            .map_err(|_| WireError::ParseError)
    }
}

/// Encode an IRDelta to wire format
pub fn encode_delta(delta: &IRDelta, patch_id: u64, seq: u64) -> Result<Bytes, WireError> {
    // 1. CBOR encode
    let mut payload = Vec::new();
    ciborium::into_writer(delta, &mut payload).map_err(|_| WireError::EncodeError)?;

    // 2. zstd compress
    let compressed = zstd::encode_all(payload.as_slice(), 3)
        .map_err(|_| WireError::CompressError)?;

    // 3. build header
    let checksum = xxh3_32(&compressed);
    let header = PacketHeader {
        magic: MAGIC,
        version: 1,
        ir_level: 0,
        flags: 0x01, // zstd compressed
        patch_id,
        delta_seq: seq,
        checksum,
    };

    // 4. assemble packet
    let mut packet = Vec::with_capacity(
        std::mem::size_of::<PacketHeader>() + compressed.len()
    );
    // SAFETY: PacketHeader is repr(C), plain old data
    let header_bytes = unsafe {
        std::slice::from_raw_parts(
            &header as *const PacketHeader as *const u8,
            std::mem::size_of::<PacketHeader>(),
        )
    };
    packet.extend_from_slice(header_bytes);
    packet.extend_from_slice(&compressed);

    Ok(Bytes::from(packet))
}

/// Decode wire packet to IRDelta
pub fn decode_delta(packet: Bytes) -> Result<IRDelta, WireError> {
    let header_size = std::mem::size_of::<PacketHeader>();
    if packet.len() < header_size { return Err(WireError::TooShort); }

    let magic = u32::from_le_bytes(packet[0..4].try_into().unwrap());
    if magic != MAGIC { return Err(WireError::BadMagic); }

    let flags = packet[7];
    let compressed = &packet[header_size..];

    let payload = if flags & 0x01 != 0 {
        zstd::decode_all(compressed).map_err(|_| WireError::DecompressError)?
    } else {
        compressed.to_vec()
    };

    ciborium::from_reader(payload.as_slice()).map_err(|_| WireError::ParseError)
}

#[derive(Debug, thiserror::Error)]
pub enum WireError {
    #[error("packet too short")]
    TooShort,
    #[error("bad magic number")]
    BadMagic,
    #[error("encode error")]
    EncodeError,
    #[error("parse error")]
    ParseError,
    #[error("compress error")]
    CompressError,
    #[error("decompress error")]
    DecompressError,
}
```

---

## 9. Workspace Cargo.toml

```toml
[workspace]
members = [
    "hymeko_core",
    "hymeko_ir",
    "hymeko_emitter",
    "hymeko_parser",
    "hymeko_wasm",
    "hymeko_server",
    "hymeko_mcp",
    "hymeko_wire",
]
resolver = "2"

[workspace.dependencies]
serde = { version = "1", features = ["derive"] }
tokio = { version = "1", features = ["full"] }
thiserror = "1"
```

---

## 10. Build + Run Commands

```bash
# 1. Build WASM (output → hymeko_server/static/pkg/)
wasm-pack build hymeko_wasm --target web --out-dir ../hymeko_server/static/pkg

# 2. Run local server (open http://localhost:3000)
cargo run --bin hymeko_server

# 3. Run MCP server (stdio transport, add to Claude Code MCP config)
cargo run --bin hymeko_mcp

# 4. Run all tests
cargo test --workspace

# 5. Check WASM size
wasm-opt -Oz hymeko_server/static/pkg/hymeko_wasm_bg.wasm -o optimized.wasm
ls -lh optimized.wasm
```

---

## 11. Claude Code MCP Config

Add to `.claude/mcp.json` in your workspace root:

```json
{
  "mcpServers": {
    "hymeko": {
      "command": "cargo",
      "args": ["run", "--bin", "hymeko_mcp"],
      "cwd": "/path/to/hymeko_workspace"
    }
  }
}
```

Claude Code can then call `add_vertex`, `add_hyperedge`, `emit_sysml` etc. directly.

---

## 12. Implementation Order

```
Step 1: hymeko_ir crate — extract, define IRDelta, test apply()
Step 2: hymeko_emitter — emit_hymeko() + emit_sysml() working
Step 3: hymeko_wasm — wasm-pack build succeeds, snapshot() works in browser
Step 4: hymeko_server — serves static WASM, REST file I/O works
Step 5: hymeko_mcp — Claude Code can add_vertex and emit_sysml
Step 6: hymeko_wire — encode/decode round-trip test passes
Step 7: React canvas — SVG hyperedge rendering (convex hull)
Step 8: T2M sync — parse_text wired to existing parser
```

Steps 1-5 = your draw.io pain is gone.
Step 5 alone = publishable system paper.

---

## 13. Key Design Invariants (Do Not Violate)

1. **IR is the single source of truth** — canvas and text are views only
2. **emit_hymeko() is deterministic** — same IR always same text
3. **IRDelta is the only mutation path** — never mutate IR fields directly
4. **WASM has no file I/O** — all file ops go through hymeko_server REST
5. **MCP server is stateful per session** — Arc<Mutex<HyMeKoIR>> is correct
6. **CBOR encoding is deterministic mode** — required for CRDT checksumming
