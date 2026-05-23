//! `hymeko_mcp` — MCP-compatible stdio JSON-RPC 2.0 server exposing
//! HyMeKo editor operations as Claude Code tools.
//!
//! Rather than pulling in `rmcp`, this crate implements the MCP subset
//! we need directly: `initialize`, `tools/list`, `tools/call`. That keeps
//! the dependency graph small and lets us unit-test every request shape
//! with `handle_request(json)` calls rather than spinning up a real
//! stdio process.
//!
//! See `docs/plans/06_wasm_editor/outline.md` for the protocol context
//! and the spec at `steps/20260418/hymeko_claude_code_spec.md § 7` for
//! the original per-tool signatures.

pub mod protocol;
pub mod tools;
pub mod server;

pub use protocol::{JsonRpcError, JsonRpcRequest, JsonRpcResponse};
pub use server::McpServer;
