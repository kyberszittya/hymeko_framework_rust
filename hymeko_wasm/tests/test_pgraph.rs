//! Native tests for the P-graph browser-app core ([`hymeko_wasm::pgraph`]).
//! The `cfg(wasm32)` `#[wasm_bindgen]` shims in `wasm.rs` are one-line wrappers
//! over these functions, so testing here covers the logic the browser runs.

use std::path::PathBuf;

use hymeko_wasm::pgraph::{dot, solve_json, transform_text};

fn read(rel: &str) -> String {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join(rel);
    std::fs::read_to_string(p).unwrap()
}

fn example() -> (String, String) {
    (
        read("hymeko_pgraph/data/prgraph_ex_3_1.hymeko"),
        read("hymeko_pgraph/data/meta_pgraph.hymeko"),
    )
}

#[test]
fn solve_json_reports_msg_and_abb() {
    let (instance, meta) = example();
    let json = solve_json(&instance, &meta).unwrap();
    // MSG keeps {u1,u4,u5}; ABB present.
    assert!(json.contains("u1") && json.contains("u4") && json.contains("u5"));
    assert!(json.contains("\"abb\""));
    assert!(!json.contains("\"u2\"")); // pruned (F unproduced)
}

#[test]
fn transform_text_shows_incidence() {
    let (instance, meta) = example();
    let out = transform_text(&instance, &meta).unwrap();
    assert!(out.contains("M-nodes (7)"));
    assert!(out.contains("O-nodes (5)"));
}

#[test]
fn dot_is_valid_digraph() {
    let (instance, meta) = example();
    let d = dot(&instance, &meta).unwrap();
    assert!(d.starts_with("digraph"));
    assert!(d.contains("->"));
}

#[test]
fn literal_tag_instance_solves_with_empty_meta() {
    // hda.hymeko is literal-tag (no pgraph archetypes) ⇒ fallback path; meta
    // is unused, pass empty.
    let hda = read("data/pgraph/hda.hymeko");
    let json = solve_json(&hda, "").unwrap();
    assert!(json.contains("\"ok\":true") || json.contains("\"ok\": true"));
}
