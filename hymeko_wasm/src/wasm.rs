//! `wasm-bindgen` façade over [`crate::session::EditorSession`].
//!
//! Gated behind `cfg(target_arch = "wasm32")` so native builds don't pull
//! the wasm-bindgen machinery. Build with:
//!
//! ```text
//! wasm-pack build hymeko_wasm --target web
//! ```
//!
//! The exposed surface is intentionally thin — it returns JSON strings
//! (not `JsValue` structs) so the JS side can decide how to deserialise
//! and the Rust side stays ergonomic to unit-test.

use wasm_bindgen::prelude::*;

use crate::session::EditorSession as NativeSession;

#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
}

#[wasm_bindgen]
pub struct EditorSession {
    inner: NativeSession,
}

#[wasm_bindgen]
impl EditorSession {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: NativeSession::new(),
        }
    }

    /// Returns a JSON-encoded snapshot of the full editor IR.
    #[wasm_bindgen]
    pub fn snapshot_json(&self) -> Result<String, JsValue> {
        self.inner
            .snapshot_json()
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Returns the full editor IR as CBOR bytes.
    #[wasm_bindgen]
    pub fn export_cbor(&self) -> Result<Vec<u8>, JsValue> {
        self.inner
            .export_cbor()
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Replaces the session state from a CBOR byte slice.
    #[wasm_bindgen]
    pub fn import_cbor(&mut self, bytes: &[u8]) -> Result<(), JsValue> {
        self.inner
            .import_cbor(bytes)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Add a vertex with the given name + level. Returns the slotmap key
    /// encoded as a `u64` so JS can pass it back on later calls.
    #[wasm_bindgen]
    pub fn add_vertex(&mut self, name: String, level: i8) -> u64 {
        let key = self.inner.add_vertex(name, level);
        slotmap::Key::data(&key).as_ffi()
    }

    #[wasm_bindgen]
    pub fn vertex_count(&self) -> usize {
        self.inner.summary().vertex_count
    }

    #[wasm_bindgen]
    pub fn edge_count(&self) -> usize {
        self.inner.summary().edge_count
    }

    #[wasm_bindgen]
    pub fn reset(&mut self) {
        self.inner.reset();
    }
}
