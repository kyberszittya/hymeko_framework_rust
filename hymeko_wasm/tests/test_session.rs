//! Native-only tests for `EditorSession`. The wasm-bindgen façade in
//! `crate::wasm` is gated behind `cfg(target_arch = "wasm32")` so these
//! tests exercise the portable inner surface.

use hymeko_emitter::editor_ir::{Attribute, AttributeValue, IRDelta, Sign};
use hymeko_wasm::EditorSession;

#[test]
fn new_session_is_empty() {
    let s = EditorSession::new();
    let sum = s.summary();
    assert_eq!(sum.vertex_count, 0);
    assert_eq!(sum.edge_count, 0);
    assert_eq!(sum.patch_count, 0);
}

#[test]
fn add_vertex_increments_counts() {
    let mut s = EditorSession::new();
    let _a = s.add_vertex("base", 0);
    let _b = s.add_vertex("link", 0);
    assert_eq!(s.summary().vertex_count, 2);
}

#[test]
fn add_hyperedge_connects_vertices() {
    let mut s = EditorSession::new();
    let a = s.add_vertex("a", 0);
    let b = s.add_vertex("b", 0);
    let _e = s.add_hyperedge("ab", vec![(a, Sign::Plus), (b, Sign::Minus)], 1.0);
    assert_eq!(s.summary().edge_count, 1);
    assert_eq!(s.summary().vertex_count, 2);
}

#[test]
fn move_vertex_updates_position() {
    let mut s = EditorSession::new();
    let k = s.add_vertex("x", 0);
    s.move_vertex(k, 1.5, 2.5).unwrap();
    let v = s.ir().vertices.get(k).unwrap();
    assert_eq!(v.position.map(|p| (p.x, p.y)), Some((1.5, 2.5)));
}

#[test]
fn attach_attribute_via_session() {
    let mut s = EditorSession::new();
    let k = s.add_vertex("link", 0);
    s.attach_attribute(
        k,
        Attribute {
            key: "mass".into(),
            value: AttributeValue::Float(5.0),
        },
    )
    .unwrap();
    let v = s.ir().vertices.get(k).unwrap();
    assert_eq!(v.attributes.len(), 1);
}

#[test]
fn cbor_roundtrip_preserves_session() {
    let mut s1 = EditorSession::new();
    let a = s1.add_vertex("a", 0);
    let b = s1.add_vertex("b", 0);
    s1.add_hyperedge("ab", vec![(a, Sign::Plus), (b, Sign::Minus)], 2.0);

    let bytes = s1.export_cbor().unwrap();
    assert!(!bytes.is_empty());

    let mut s2 = EditorSession::new();
    s2.import_cbor(&bytes).unwrap();

    assert_eq!(s2.summary(), s1.summary());
}

#[test]
fn snapshot_json_is_parseable_and_non_empty() {
    let mut s = EditorSession::new();
    s.add_vertex("a", 0);
    let snapshot = s.snapshot_json().unwrap();
    assert!(!snapshot.is_empty());
    // Must be valid JSON.
    let v: serde_json::Value = serde_json::from_str(&snapshot).unwrap();
    assert!(v.is_object());
}

#[test]
fn reset_clears_all_state() {
    let mut s = EditorSession::new();
    s.add_vertex("x", 0);
    s.reset();
    assert_eq!(s.summary().vertex_count, 0);
}

#[test]
fn apply_batch_via_session() {
    let mut s = EditorSession::new();
    s.apply(IRDelta::Batch {
        deltas: vec![
            IRDelta::AddVertex {
                data: hymeko_emitter::editor_ir::Vertex {
                    name: "a".into(),
                    level: 0,
                    attributes: vec![],
                    position: None,
                },
            },
            IRDelta::AddVertex {
                data: hymeko_emitter::editor_ir::Vertex {
                    name: "b".into(),
                    level: 0,
                    attributes: vec![],
                    position: None,
                },
            },
        ],
    })
    .unwrap();
    assert_eq!(s.summary().vertex_count, 2);
}

// ---- Emitter front-ends via the bridge (Step 2c) ------------------------

#[test]
fn emit_hymeko_roundtrips_from_editor_session() {
    let mut s = EditorSession::new();
    let a = s.add_vertex("base_link", 0);
    let b = s.add_vertex("spinner", 0);
    s.add_hyperedge(
        "spin_joint",
        vec![(a, Sign::Plus), (b, Sign::Minus)],
        1.0,
    );
    let out = s.emit_hymeko("mini_arm_from_canvas");
    assert!(
        out.contains("mini_arm_from_canvas {}"),
        "description header missing:\n{out}"
    );
    assert!(out.contains("base_link"));
    assert!(out.contains("spinner"));
    assert!(out.contains("@spin_joint"));
    // Parse the output back — confirms the bridge-produced arena IR
    // emits parseable source.
    let _ast = parser::parse_description(&out).expect("emitted .hymeko should re-parse");
}

#[test]
fn emit_sysml_from_editor_session_wraps_in_package() {
    let mut s = EditorSession::new();
    s.add_vertex("a", 0);
    s.add_vertex("b", 0);
    let out = s.emit_sysml("Demo");
    assert!(out.contains("package Demo {"));
    assert!(out.contains("part def a"));
    assert!(out.contains("part def b"));
}

#[test]
fn emit_rust_stubs_from_editor_session_produces_traits() {
    let mut s = EditorSession::new();
    s.add_vertex("base_link", 0);
    let out = s.emit_rust_stubs();
    assert!(out.contains("pub trait BaseLink {"));
    assert!(out.contains("fn process(&self)"));
}

#[test]
fn emit_lean4_from_editor_session_produces_theorem() {
    let mut s = EditorSession::new();
    s.add_vertex("base_link", 0);
    let out = s.emit_lean4();
    assert!(out.contains("import Mathlib"));
    assert!(out.contains("theorem base_link_level_invariant"));
}

#[test]
fn emit_methods_are_deterministic_across_calls() {
    let mut s = EditorSession::new();
    s.add_vertex("x", 0);
    assert_eq!(s.emit_hymeko("m"), s.emit_hymeko("m"));
    assert_eq!(s.emit_sysml("M"), s.emit_sysml("M"));
    assert_eq!(s.emit_rust_stubs(), s.emit_rust_stubs());
    assert_eq!(s.emit_lean4(), s.emit_lean4());
}
