//! `hymeko_wasm` — wasm-bindgen exposure of
//! [`hymeko_emitter::editor_ir::HyMeKoEditorIR`] for the browser canvas.
//!
//! The crate ships a portable [`session::EditorSession`] that wraps the
//! editor IR with an ergonomic native API (JSON / CBOR round-trip, delta
//! apply). A thin [`wasm`] module re-exports the same surface to
//! JavaScript via `wasm-bindgen` — that module is gated behind
//! `cfg(target_arch = "wasm32")` so native `cargo test` and `cargo
//! check` still succeed without touching the wasm toolchain.
//!
//! Build for the browser with:
//!
//! ```text
//! wasm-pack build hymeko_wasm --target web
//! ```

pub mod session;

#[cfg(target_arch = "wasm32")]
pub mod wasm;

pub use session::{EditorSession, SessionError};
