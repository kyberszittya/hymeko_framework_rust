//! Tool catalogue — each tool has a JSON-schema-lite descriptor for
//! `tools/list` plus a `dispatch` arm in `server.rs`.

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDescriptor {
    pub name: &'static str,
    pub description: &'static str,
    pub input_schema: Value,
}

/// The six tools the HyMeKo MCP server exposes to Claude Code.
pub fn catalogue() -> Vec<ToolDescriptor> {
    vec![
        ToolDescriptor {
            name: "add_vertex",
            description: "Add a vertex to the editor IR and return its key.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "name": { "type": "string" },
                    "level": { "type": "integer", "minimum": -2, "maximum": 8 }
                },
                "required": ["name"]
            }),
        },
        ToolDescriptor {
            name: "add_hyperedge",
            description: "Add a hyperedge connecting vertices by name. \
                          Signs is an array matching vertices, each one of \"+\", \"-\", \"~\".",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "name": { "type": "string" },
                    "vertices": { "type": "array", "items": { "type": "string" } },
                    "signs": { "type": "array", "items": { "enum": ["+", "-", "~"] } },
                    "weight": { "type": "number", "default": 1.0 }
                },
                "required": ["name", "vertices", "signs"]
            }),
        },
        ToolDescriptor {
            name: "snapshot",
            description: "Return the current editor IR as JSON.",
            input_schema: json!({ "type": "object" }),
        },
        ToolDescriptor {
            name: "summary",
            description: "Return vertex / edge / patch counts.",
            input_schema: json!({ "type": "object" }),
        },
        ToolDescriptor {
            name: "reset",
            description: "Clear the editor IR and start fresh.",
            input_schema: json!({ "type": "object" }),
        },
        ToolDescriptor {
            name: "export_cbor",
            description: "Return the editor IR as base64-encoded CBOR bytes.",
            input_schema: json!({ "type": "object" }),
        },
    ]
}
