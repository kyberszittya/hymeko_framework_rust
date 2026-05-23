//! Native repro for the demo's "Compile failed: unreachable" panic.
//!
//! `compile_source` runs the full module-store / resolve / lower pipeline
//! against a `MemProvider` with a single inline file. Same code path the
//! browser hits, so any panic that fires in a browser tab fires here too.

use std::fs;

use hymeko_wasm::compile::compile_source;

// Mirrors the demo's FALLBACK_EXAMPLE in `docs/demo/demo.js`. Must
// stay in sync — if either form drifts, the demo's "Load canonical
// example" button breaks silently when the fetch falls back inline.
const TINY_INLINE_EXAMPLE: &str = r#"tiny_arm_description {}

tiny_arm {
    link {}
    rev_joint {}
    AXIS_Z {}

    base_link:    + <isa> link {}
    spinner_link: + <isa> link {}

    @j1: + <isa> rev_joint {
        (+ base_link, - spinner_link, - AXIS_Z);
    }
}
"#;

#[test]
fn compile_fallback_inline_example() {
    let r = compile_source(TINY_INLINE_EXAMPLE);
    let doc = r.expect("inline tiny_arm should compile");
    assert!(doc.node_count() > 0);
    assert!(doc.edge_count() > 0);
}

#[test]
fn compile_canonical_paper_example() {
    let path = "../examples/paper/hymeko_robot.hymeko";
    let src = fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("read {}: {}", path, e));
    let r = compile_source(&src);
    let doc = r.expect("canonical example should compile");
    assert!(doc.node_count() > 0);
    assert!(doc.edge_count() > 0);
}

#[test]
fn compile_empty_source_returns_err_not_panic() {
    // Empty / whitespace source should cleanly Err — never panic — so the
    // demo's "Compile failed: <msg>" stays informative.
    let r = compile_source("");
    assert!(r.is_err(), "empty source should error, not succeed");
}
