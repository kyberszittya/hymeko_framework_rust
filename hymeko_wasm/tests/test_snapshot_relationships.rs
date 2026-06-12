//! Native test: the parse → snapshot path (the same `compile_source` the
//! browser editor calls) emits the uniform `relationships` list with both
//! `scope` (containment) and `isa` (template) kinds.

use hymeko_wasm::compile_source;

#[test]
fn snapshot_emits_scope_isa_and_ref_relationships() {
    let src = "Lib {}\n\
               ns {\n\
                   meta {}\n\
                   thing: + <isa> meta {}\n\
                   target_decl {}\n\
                   pointer -> target_decl;\n\
               }\n";
    let doc = compile_source(src).expect("source should compile");
    let json = doc.snapshot_json().expect("snapshot_json");
    let v: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");

    let rels = v["relationships"].as_array().expect("relationships is an array");
    assert!(!rels.is_empty(), "expected non-empty relationships:\n{json}");

    let kinds: std::collections::HashSet<&str> =
        rels.iter().filter_map(|r| r["kind"].as_str()).collect();
    assert!(kinds.contains("scope"), "expected scope (containment) relationships");
    assert!(kinds.contains("isa"), "expected isa relationships (thing <isa> meta)");
    assert!(kinds.contains("ref"), "expected ref relationships (pointer -> target_decl)");

    // Every relationship references valid usize endpoints.
    for r in rels {
        assert!(r["from"].is_u64() && r["to"].is_u64(), "endpoints must be ids: {r}");
    }
}
