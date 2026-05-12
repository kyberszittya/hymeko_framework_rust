//! `McpServer` — synchronous, in-memory dispatcher for MCP JSON-RPC
//! requests. The `main` binary wires `stdin`/`stdout` around this struct;
//! tests can call [`McpServer::handle_request`] directly.

use std::collections::BTreeMap;
use std::sync::Mutex;

use hymeko_emitter::editor_ir::{HyperEdge, Sign, Vertex, VertexKey};
use hymeko_wasm::EditorSession;
use serde_json::{Value, json};

use crate::protocol::{
    INTERNAL_ERROR, INVALID_PARAMS, JsonRpcRequest, JsonRpcResponse, METHOD_NOT_FOUND,
};
use crate::tools::{ToolDescriptor, catalogue};

pub struct McpServer {
    session: Mutex<EditorSession>,
    /// Reverse map from user-supplied names to slotmap keys (so
    /// `add_hyperedge` can reference vertices by their declared name).
    name_index: Mutex<BTreeMap<String, VertexKey>>,
}

impl Default for McpServer {
    fn default() -> Self {
        Self {
            session: Mutex::new(EditorSession::new()),
            name_index: Mutex::new(BTreeMap::new()),
        }
    }
}

impl McpServer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Handle a single JSON-RPC request and return the JSON-encoded
    /// response body. Never panics on malformed input.
    pub fn handle_request(&self, body: &str) -> String {
        let req: JsonRpcRequest = match serde_json::from_str(body) {
            Ok(r) => r,
            Err(e) => {
                return serde_json::to_string(&JsonRpcResponse::error(
                    None,
                    -32700,
                    format!("parse error: {e}"),
                ))
                .unwrap();
            }
        };
        let response = self.dispatch(req);
        serde_json::to_string(&response).unwrap()
    }

    fn dispatch(&self, req: JsonRpcRequest) -> JsonRpcResponse {
        match req.method.as_str() {
            "initialize" => JsonRpcResponse::success(
                req.id,
                json!({
                    "protocolVersion": "2024-11-05",
                    "serverInfo": { "name": "hymeko_mcp", "version": env!("CARGO_PKG_VERSION") },
                    "capabilities": { "tools": {} }
                }),
            ),
            "tools/list" => {
                let tools: Vec<_> = catalogue()
                    .into_iter()
                    .map(|t: ToolDescriptor| {
                        json!({
                            "name": t.name,
                            "description": t.description,
                            "inputSchema": t.input_schema,
                        })
                    })
                    .collect();
                JsonRpcResponse::success(req.id, json!({ "tools": tools }))
            }
            "tools/call" => self.call_tool(req.id, req.params),
            other => JsonRpcResponse::error(
                req.id,
                METHOD_NOT_FOUND,
                format!("unknown method `{other}`"),
            ),
        }
    }

    fn call_tool(&self, id: Option<Value>, params: Option<Value>) -> JsonRpcResponse {
        let params = match params {
            Some(p) => p,
            None => return JsonRpcResponse::error(id, INVALID_PARAMS, "missing params"),
        };
        let tool_name = match params.get("name").and_then(|v| v.as_str()) {
            Some(n) => n.to_string(),
            None => return JsonRpcResponse::error(id, INVALID_PARAMS, "missing `name`"),
        };
        let args = params.get("arguments").cloned().unwrap_or(json!({}));

        let result: Result<Value, String> = match tool_name.as_str() {
            "add_vertex" => self.tool_add_vertex(&args),
            "add_hyperedge" => self.tool_add_hyperedge(&args),
            "snapshot" => self.tool_snapshot(),
            "summary" => self.tool_summary(),
            "reset" => self.tool_reset(),
            "export_cbor" => self.tool_export_cbor(),
            other => Err(format!("unknown tool `{other}`")),
        };

        match result {
            Ok(v) => JsonRpcResponse::success(
                id,
                json!({
                    "content": [{ "type": "text", "text": v.to_string() }],
                    "isError": false,
                }),
            ),
            Err(e) => JsonRpcResponse::error(id, INTERNAL_ERROR, e),
        }
    }

    // ---- individual tools ------------------------------------------------

    fn tool_add_vertex(&self, args: &Value) -> Result<Value, String> {
        let name = args
            .get("name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "missing `name`".to_string())?
            .to_string();
        let level = args.get("level").and_then(|v| v.as_i64()).unwrap_or(0) as i8;

        let mut sess = self.session.lock().unwrap();
        let key = sess.add_vertex(name.clone(), level);
        self.name_index.lock().unwrap().insert(name.clone(), key);

        let key_ffi: u64 = slotmap::Key::data(&key).as_ffi();
        Ok(json!({ "name": name, "key": key_ffi }))
    }

    fn tool_add_hyperedge(&self, args: &Value) -> Result<Value, String> {
        let name = args
            .get("name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "missing `name`".to_string())?
            .to_string();
        let vertices: Vec<String> = args
            .get("vertices")
            .and_then(|v| v.as_array())
            .ok_or_else(|| "missing `vertices`".to_string())?
            .iter()
            .filter_map(|x| x.as_str().map(String::from))
            .collect();
        let signs: Vec<Sign> = args
            .get("signs")
            .and_then(|v| v.as_array())
            .ok_or_else(|| "missing `signs`".to_string())?
            .iter()
            .map(|x| match x.as_str() {
                Some("+") => Ok(Sign::Plus),
                Some("-") => Ok(Sign::Minus),
                Some("~") => Ok(Sign::Neutral),
                _ => Err("sign must be \"+\", \"-\", or \"~\"".to_string()),
            })
            .collect::<Result<_, _>>()?;
        if signs.len() != vertices.len() {
            return Err("signs and vertices arrays must have the same length".into());
        }
        let weight = args
            .get("weight")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0);

        let name_index = self.name_index.lock().unwrap();
        let incident: Vec<(VertexKey, Sign)> = vertices
            .iter()
            .zip(signs.iter().copied())
            .map(|(vn, s)| {
                name_index
                    .get(vn)
                    .copied()
                    .map(|k| (k, s))
                    .ok_or_else(|| format!("unknown vertex `{vn}`"))
            })
            .collect::<Result<_, _>>()?;
        drop(name_index);

        let mut sess = self.session.lock().unwrap();
        let key = sess.add_hyperedge(name.clone(), incident, weight);
        let key_ffi: u64 = slotmap::Key::data(&key).as_ffi();
        Ok(json!({ "name": name, "key": key_ffi }))
    }

    fn tool_snapshot(&self) -> Result<Value, String> {
        let sess = self.session.lock().unwrap();
        let json = sess
            .snapshot_json()
            .map_err(|e| format!("snapshot failed: {e}"))?;
        serde_json::from_str(&json).map_err(|e| format!("snapshot not JSON: {e}"))
    }

    fn tool_summary(&self) -> Result<Value, String> {
        let sum = self.session.lock().unwrap().summary();
        Ok(json!({
            "vertex_count": sum.vertex_count,
            "edge_count": sum.edge_count,
            "patch_count": sum.patch_count,
        }))
    }

    fn tool_reset(&self) -> Result<Value, String> {
        self.session.lock().unwrap().reset();
        self.name_index.lock().unwrap().clear();
        Ok(json!({ "ok": true }))
    }

    fn tool_export_cbor(&self) -> Result<Value, String> {
        let bytes = self
            .session
            .lock()
            .unwrap()
            .export_cbor()
            .map_err(|e| format!("cbor export failed: {e}"))?;
        // Base64-encode for JSON friendliness.
        let b64 = base64_encode(&bytes);
        Ok(json!({ "cbor_base64": b64, "size": bytes.len() }))
    }
}

// Tiny hand-rolled base64 so we don't add a dep just for this tool.
fn base64_encode(input: &[u8]) -> String {
    const TABLE: &[u8; 64] =
        b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(input.len().div_ceil(3) * 4);
    for chunk in input.chunks(3) {
        let b0 = chunk[0];
        let b1 = if chunk.len() > 1 { chunk[1] } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] } else { 0 };
        out.push(TABLE[(b0 >> 2) as usize] as char);
        out.push(TABLE[(((b0 & 0x03) << 4) | (b1 >> 4)) as usize] as char);
        if chunk.len() > 1 {
            out.push(TABLE[(((b1 & 0x0F) << 2) | (b2 >> 6)) as usize] as char);
        } else {
            out.push('=');
        }
        if chunk.len() > 2 {
            out.push(TABLE[(b2 & 0x3F) as usize] as char);
        } else {
            out.push('=');
        }
    }
    out
}

// Silence the unused import warning — Vertex / HyperEdge are used via
// EditorSession methods and re-exported for documentation/clarity.
#[allow(dead_code)]
const _: Option<&Vertex> = None;
#[allow(dead_code)]
const _: Option<&HyperEdge> = None;
