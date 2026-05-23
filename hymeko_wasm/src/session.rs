//! Native editor session — portable across wasm32 and host targets.
//!
//! `EditorSession` holds a `HyMeKoEditorIR` and exposes the mutation
//! surface the WASM canvas needs: add vertex / add hyperedge / apply
//! delta / snapshot / CBOR import+export. All methods use plain Rust
//! types so the module is natively testable without a wasm runtime.
//!
//! `emit_*` front-ends bridge the editor IR into a fresh arena IR via
//! `hymeko_emitter::bridge::to_compiler_ir` and hand off to the M2T
//! emitters. The session owns its own `Interner` for these conversions;
//! each call starts with a fresh arena so repeated emission is
//! deterministic.

use hymeko_emitter::bridge::to_compiler_ir;
use hymeko_emitter::editor_ir::{
    Attribute, EdgeKey, HyMeKoEditorIR, HyperEdge, IRDelta, IRError, Position, Sign, Vertex,
    VertexKey,
};
use hymeko_emitter::{emit_hymeko, emit_lean4, emit_rust_stubs, emit_sysml};
use hymeko::resolution::interner::Interner;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SessionError {
    #[error("apply failed: {0}")]
    Apply(#[from] IRError),
    #[error("json serialisation failed: {0}")]
    Json(String),
    #[error("cbor serialisation failed: {0}")]
    Cbor(String),
}

/// Stable summary of the session state — handy for the canvas to render
/// without pulling the full IR across the wasm boundary.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SessionSummary {
    pub vertex_count: usize,
    pub edge_count: usize,
    pub patch_count: usize,
}

pub struct EditorSession {
    ir: HyMeKoEditorIR,
}

impl Default for EditorSession {
    fn default() -> Self {
        Self::new()
    }
}

impl EditorSession {
    pub fn new() -> Self {
        Self {
            ir: HyMeKoEditorIR::new(),
        }
    }

    /// Read-only borrow of the underlying editor IR (used by tests and the
    /// bridge once it lands in Step 2c).
    pub fn ir(&self) -> &HyMeKoEditorIR {
        &self.ir
    }

    pub fn summary(&self) -> SessionSummary {
        SessionSummary {
            vertex_count: self.ir.vertices.len(),
            edge_count: self.ir.hyperedges.len(),
            patch_count: self.ir.patches.len(),
        }
    }

    pub fn apply(&mut self, delta: IRDelta) -> Result<(), SessionError> {
        self.ir.apply(delta)?;
        Ok(())
    }

    /// Convenience — direct vertex creation returning the allocated key.
    pub fn add_vertex(&mut self, name: impl Into<String>, level: i8) -> VertexKey {
        self.ir.vertices.insert(Vertex {
            name: name.into(),
            level,
            attributes: Vec::new(),
            position: None,
        })
    }

    /// Convenience — direct hyperedge creation.
    pub fn add_hyperedge(
        &mut self,
        name: impl Into<String>,
        incident: Vec<(VertexKey, Sign)>,
        weight: f64,
    ) -> EdgeKey {
        self.ir.hyperedges.insert(HyperEdge {
            name: name.into(),
            incident,
            weight,
            patch_id: None,
        })
    }

    pub fn move_vertex(&mut self, key: VertexKey, x: f64, y: f64) -> Result<(), SessionError> {
        self.apply(IRDelta::MoveVertex {
            key,
            position: Position { x, y },
        })
    }

    pub fn attach_attribute(
        &mut self,
        key: VertexKey,
        attr: Attribute,
    ) -> Result<(), SessionError> {
        self.apply(IRDelta::AttachAttribute { key, attr })
    }

    /// Full snapshot as JSON. Preserves slotmap keys stably.
    pub fn snapshot_json(&self) -> Result<String, SessionError> {
        serde_json::to_string(&self.ir).map_err(|e| SessionError::Json(e.to_string()))
    }

    /// Full snapshot as CBOR bytes. Canonicalised so the output is
    /// content-hashable for P2P gossip (see `hymeko_wire`).
    pub fn export_cbor(&self) -> Result<Vec<u8>, SessionError> {
        let mut buf = Vec::new();
        ciborium::into_writer(&self.ir, &mut buf).map_err(|e| SessionError::Cbor(e.to_string()))?;
        Ok(buf)
    }

    /// Replace the session state with a CBOR-encoded snapshot.
    pub fn import_cbor(&mut self, bytes: &[u8]) -> Result<(), SessionError> {
        self.ir = ciborium::from_reader(bytes).map_err(|e| SessionError::Cbor(e.to_string()))?;
        Ok(())
    }

    pub fn reset(&mut self) {
        self.ir = HyMeKoEditorIR::new();
    }

    // ---- M2T emission via the bridge ------------------------------------

    /// Emit the current editor IR as a deterministic `.hymeko` source
    /// wrapped under `description_name`. Uses a fresh interner + arena
    /// per call so the output is reproducible.
    pub fn emit_hymeko(&self, description_name: &str) -> String {
        let mut interner = Interner::new();
        let arena = to_compiler_ir(&self.ir, &mut interner);
        emit_hymeko(&arena, &interner, description_name)
    }

    /// Emit the current editor IR as a SysML v2 model wrapped under
    /// `package_name`.
    pub fn emit_sysml(&self, package_name: &str) -> String {
        let mut interner = Interner::new();
        let arena = to_compiler_ir(&self.ir, &mut interner);
        emit_sysml(&arena, &interner, package_name)
    }

    /// Emit Rust trait skeletons — one `pub trait <PascalName>` per
    /// vertex in the editor IR.
    pub fn emit_rust_stubs(&self) -> String {
        let mut interner = Interner::new();
        let arena = to_compiler_ir(&self.ir, &mut interner);
        emit_rust_stubs(&arena, &interner)
    }

    /// Emit Lean 4 proof-obligation skeletons, one trivial theorem per
    /// vertex.
    pub fn emit_lean4(&self) -> String {
        let mut interner = Interner::new();
        let arena = to_compiler_ir(&self.ir, &mut interner);
        emit_lean4(&arena, &interner)
    }
}
