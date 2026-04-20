//! Unit tests for the MCP dispatcher.

use hymeko_mcp::McpServer;
use serde_json::{Value, json};

fn parse(resp: &str) -> Value {
    serde_json::from_str(resp).expect("dispatcher returned invalid JSON")
}

#[test]
fn initialize_responds_with_protocol_version() {
    let s = McpServer::new();
    let resp = s.handle_request(
        &json!({ "jsonrpc": "2.0", "id": 1, "method": "initialize" }).to_string(),
    );
    let v = parse(&resp);
    assert_eq!(v["jsonrpc"], "2.0");
    assert_eq!(v["id"], 1);
    assert!(v["result"]["protocolVersion"].is_string());
    assert_eq!(v["result"]["serverInfo"]["name"], "hymeko_mcp");
}

#[test]
fn tools_list_returns_six_tools() {
    let s = McpServer::new();
    let resp = s.handle_request(
        &json!({ "jsonrpc": "2.0", "id": 2, "method": "tools/list" }).to_string(),
    );
    let v = parse(&resp);
    let tools = v["result"]["tools"].as_array().unwrap();
    assert_eq!(tools.len(), 6);
    let names: Vec<&str> = tools.iter().map(|t| t["name"].as_str().unwrap()).collect();
    for expected in ["add_vertex", "add_hyperedge", "snapshot", "summary", "reset", "export_cbor"] {
        assert!(names.contains(&expected), "missing tool {expected}");
    }
}

#[test]
fn add_vertex_and_summary_tools_flow() {
    let s = McpServer::new();

    // Add two vertices.
    let _ = s.handle_request(
        &json!({
            "jsonrpc": "2.0", "id": 10, "method": "tools/call",
            "params": { "name": "add_vertex", "arguments": { "name": "a" } }
        })
        .to_string(),
    );
    let _ = s.handle_request(
        &json!({
            "jsonrpc": "2.0", "id": 11, "method": "tools/call",
            "params": { "name": "add_vertex", "arguments": { "name": "b" } }
        })
        .to_string(),
    );

    // Query summary.
    let resp = s.handle_request(
        &json!({
            "jsonrpc": "2.0", "id": 12, "method": "tools/call",
            "params": { "name": "summary", "arguments": {} }
        })
        .to_string(),
    );
    let v = parse(&resp);
    let text = v["result"]["content"][0]["text"].as_str().unwrap();
    let summary: Value = serde_json::from_str(text).unwrap();
    assert_eq!(summary["vertex_count"], 2);
    assert_eq!(summary["edge_count"], 0);
}

#[test]
fn add_hyperedge_references_vertices_by_name() {
    let s = McpServer::new();
    for name in ["a", "b", "c"] {
        s.handle_request(
            &json!({
                "jsonrpc": "2.0", "id": 1, "method": "tools/call",
                "params": { "name": "add_vertex", "arguments": { "name": name } }
            })
            .to_string(),
        );
    }
    let resp = s.handle_request(
        &json!({
            "jsonrpc": "2.0", "id": 20, "method": "tools/call",
            "params": {
                "name": "add_hyperedge",
                "arguments": {
                    "name": "ab_c",
                    "vertices": ["a", "b", "c"],
                    "signs": ["+", "-", "-"],
                    "weight": 1.0
                }
            }
        })
        .to_string(),
    );
    let v = parse(&resp);
    assert!(v["error"].is_null(), "unexpected error: {v}");
    let summary_resp = s.handle_request(
        &json!({
            "jsonrpc": "2.0", "id": 21, "method": "tools/call",
            "params": { "name": "summary", "arguments": {} }
        })
        .to_string(),
    );
    let summary_v = parse(&summary_resp);
    let text = summary_v["result"]["content"][0]["text"].as_str().unwrap();
    let summary: Value = serde_json::from_str(text).unwrap();
    assert_eq!(summary["edge_count"], 1);
}

#[test]
fn unknown_method_returns_method_not_found() {
    let s = McpServer::new();
    let resp = s.handle_request(
        &json!({ "jsonrpc": "2.0", "id": 99, "method": "does_not_exist" }).to_string(),
    );
    let v = parse(&resp);
    assert_eq!(v["error"]["code"], -32601);
}

#[test]
fn tools_call_with_unknown_tool_errors() {
    let s = McpServer::new();
    let resp = s.handle_request(
        &json!({
            "jsonrpc": "2.0", "id": 100, "method": "tools/call",
            "params": { "name": "nope", "arguments": {} }
        })
        .to_string(),
    );
    let v = parse(&resp);
    assert_eq!(v["error"]["code"], -32603);
    assert!(v["error"]["message"].as_str().unwrap().contains("unknown tool"));
}

#[test]
fn add_hyperedge_with_sign_mismatch_errors() {
    let s = McpServer::new();
    s.handle_request(
        &json!({
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": { "name": "add_vertex", "arguments": { "name": "a" } }
        })
        .to_string(),
    );
    let resp = s.handle_request(
        &json!({
            "jsonrpc": "2.0", "id": 2, "method": "tools/call",
            "params": {
                "name": "add_hyperedge",
                "arguments": {
                    "name": "e",
                    "vertices": ["a"],
                    "signs": ["+", "-"]  // mismatched length
                }
            }
        })
        .to_string(),
    );
    let v = parse(&resp);
    assert!(v["error"]["message"].as_str().unwrap().contains("same length"));
}

#[test]
fn malformed_json_returns_parse_error() {
    let s = McpServer::new();
    let resp = s.handle_request("{ not json");
    let v = parse(&resp);
    assert_eq!(v["error"]["code"], -32700);
}

#[test]
fn reset_clears_state() {
    let s = McpServer::new();
    s.handle_request(
        &json!({
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": { "name": "add_vertex", "arguments": { "name": "x" } }
        })
        .to_string(),
    );
    s.handle_request(
        &json!({
            "jsonrpc": "2.0", "id": 2, "method": "tools/call",
            "params": { "name": "reset", "arguments": {} }
        })
        .to_string(),
    );
    let resp = s.handle_request(
        &json!({
            "jsonrpc": "2.0", "id": 3, "method": "tools/call",
            "params": { "name": "summary", "arguments": {} }
        })
        .to_string(),
    );
    let v = parse(&resp);
    let text = v["result"]["content"][0]["text"].as_str().unwrap();
    let summary: Value = serde_json::from_str(text).unwrap();
    assert_eq!(summary["vertex_count"], 0);
}
