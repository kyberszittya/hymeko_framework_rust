//! Native repro for the demo's "Compile failed: unreachable" panic.
//!
//! `compile_source` runs the full module-store / resolve / lower pipeline
//! against a `MemProvider` with a single inline file. Same code path the
//! browser hits, so any panic that fires in a browser tab fires here too.

use std::fs;

use hymeko_wasm::compile::{compile_source, compile_sources};

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
    let src = fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {}", path, e));
    let r = compile_source(&src);
    let doc = r.expect("canonical example should compile");
    assert!(doc.node_count() > 0);
    assert!(doc.edge_count() > 0);
}

#[test]
fn to_sysml_emits_package_and_part_defs() {
    // The SysML lens (SMC #5 Phase 2): the template is embedded via include_str!
    // so it runs FS-free in WASM. A model with links + a joint must produce a
    // SysML package with part defs, not the error fallback.
    let doc = compile_source(TINY_INLINE_EXAMPLE).expect("compiles");
    let sysml = doc.to_sysml("tiny_arm");
    assert!(!sysml.starts_with("// SysML generation error"), "{sysml}");
    assert!(
        sysml.contains("package tiny_arm"),
        "missing package: {sysml}"
    );
    assert!(sysml.contains("part def Link"), "missing part def: {sysml}");
    // Deterministic: same input → same output.
    assert_eq!(sysml, doc.to_sysml("tiny_arm"));
}

#[test]
fn compile_empty_source_returns_err_not_panic() {
    // Empty / whitespace source should cleanly Err — never panic — so the
    // demo's "Compile failed: <msg>" stays informative.
    let r = compile_source("");
    assert!(r.is_err(), "empty source should error, not succeed");
}

/// Multi-file "space" compile — the editor profile/imports feature. A root file
/// that `@"…"`-includes a meta vocabulary resolves against the other files
/// registered in the in-memory provider, so meta elements can live outside the
/// current context.
mod multi_file {
    use super::*;

    #[test]
    fn root_resolves_an_included_meta_vocabulary() {
        // Real shipping pair: triad_hri.hymeko opens with `@"meta_hri.hymeko";`
        // and `using hri_meta as hri;`, then declares agents typed by hri.* .
        let meta =
            fs::read_to_string("../data/coalitions/meta_hri.hymeko").expect("read meta_hri.hymeko");
        let root = fs::read_to_string("../data/coalitions/triad_hri.hymeko")
            .expect("read triad_hri.hymeko");
        let doc = compile_sources(
            "triad_hri.hymeko",
            &[("triad_hri.hymeko", &root), ("meta_hri.hymeko", &meta)],
        )
        .expect("multi-file compile with a resolved include should succeed");
        // Compiling Ok already means every `hri.*` ref resolved against the
        // imported vocabulary; the agents are typed nodes.
        assert!(doc.node_count() > 0, "agents (nodes) should resolve");

        // Prove the include is genuinely needed: drop the meta file and the same
        // root must fail to resolve its `hri.*` references.
        let without = compile_sources("triad_hri.hymeko", &[("triad_hri.hymeko", &root)]);
        assert!(
            without.is_err(),
            "without the meta file the include must not resolve"
        );
    }

    #[test]
    fn editor_profile_roots_compile_with_their_metas() {
        // The editor ships these profile roots (data/profiles/*) + their meta
        // vocabularies as a compile space. Validate each (root, meta) pair here
        // so a broken template fails in CI, not in the browser.
        // (root file, meta file) — both under data/profiles/; the root includes
        // the meta by its bare filename.
        let cases = [
            ("hri_cell.hymeko", "meta_hri.hymeko"),
            ("sysml_cell.hymeko", "meta_sysml_trace.hymeko"),
            ("robot_arm.hymeko", "meta_kinematics.hymeko"),
        ];
        for (root_file, meta_name) in cases {
            let root = fs::read_to_string(format!("../data/profiles/{root_file}"))
                .unwrap_or_else(|e| panic!("read {root_file}: {e}"));
            let meta = fs::read_to_string(format!("../data/profiles/{meta_name}"))
                .unwrap_or_else(|e| panic!("read {meta_name}: {e}"));
            let doc = compile_sources(root_file, &[(root_file, &root), (meta_name, &meta)])
                .unwrap_or_else(|e| panic!("{root_file}: {e}"));
            assert!(doc.node_count() > 0, "{root_file}: should have nodes");
        }
    }

    #[test]
    fn missing_include_errors_not_panics() {
        // Root references a file the space does not provide → clean Err.
        let root = "r_description {\n    @\"absent.hymeko\";\n}\nr {}\n";
        let res = compile_sources("input.hymeko", &[("input.hymeko", root)]);
        assert!(res.is_err(), "an unresolved include must error, not panic");
    }

    #[test]
    fn single_file_space_matches_compile_source() {
        // compile_source is just the 1-file case of compile_sources.
        let src = "d {}\nm { a {} b {} @e{ (~a, ~b); } }\n";
        let a = compile_source(src).expect("compiles");
        let b = compile_sources("inline.hymeko", &[("inline.hymeko", src)]).expect("compiles");
        assert_eq!(a.node_count(), b.node_count());
        assert_eq!(a.edge_count(), b.edge_count());
    }
}

/// Classic-hypergraph examples surfaced by the editor gallery
/// (`docs/editor/views/examples.js`) embed these fixtures verbatim — the JS
/// `examples.test.mjs` pins embed ≡ fixture, and this module pins that the
/// fixtures compile through the editor's exact pipeline (`compile_source`) and
/// produce the intended hypergraph (node / edge counts + per-hyperedge arity).
/// Together: the embedded sources render the intended classic hypergraphs.
mod typical_graph_examples {
    use super::*;
    use std::time::Instant;

    struct Expect {
        file: &'static str,
        nodes: usize,
        edges: usize,
        /// Sorted hyperedge arities (member count per edge).
        arities: &'static [usize],
    }

    // Node counts include the enclosing block decl (`fano`, `k4_3uniform`, …),
    // which is itself a Node decl — exactly as `robot`/`kit` count in the
    // kinematic example. So nodes = 1 container + the point vertices.
    const CASES: &[Expect] = &[
        // Fano plane S(2,3,7): 7 points + `fano`; 7 lines, every line a triple.
        Expect {
            file: "fano_graph.hymeko",
            nodes: 8,
            edges: 7,
            arities: &[3, 3, 3, 3, 3, 3, 3],
        },
        // Complete 3-uniform on 4 vertices: 4 points + `k4_3uniform`; all C(4,3)=4 triples.
        Expect {
            file: "k4_3uniform.hymeko",
            nodes: 5,
            edges: 4,
            arities: &[3, 3, 3, 3],
        },
        // Sunflower, 3 petals over a 2-element core: 8 points + `sunflower`; each edge arity 4.
        Expect {
            file: "sunflower_delta_system.hymeko",
            nodes: 9,
            edges: 3,
            arities: &[4, 4, 4],
        },
        // Generic mixed-arity: 7 points + `generic`; three triples + one pair.
        Expect {
            file: "generic_hypergraph.hymeko",
            nodes: 8,
            edges: 4,
            arities: &[2, 3, 3, 3],
        },
    ];

    // Tests run with CWD = the crate dir (hymeko_wasm/), so `..` is the repo
    // root — same convention as `compile_canonical_paper_example` above.
    fn read_fixture(file: &str) -> String {
        let path = format!("../data/typical_graphs/{file}");
        fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"))
    }

    #[test]
    fn typical_graphs_compile_to_expected_hypergraphs() {
        for c in CASES {
            let src = read_fixture(c.file);
            let doc =
                compile_source(&src).unwrap_or_else(|e| panic!("{}: compile failed: {e}", c.file));
            let snap = doc.snapshot();
            assert_eq!(snap.nodes.len(), c.nodes, "{}: node count", c.file);
            assert_eq!(snap.edges.len(), c.edges, "{}: edge count", c.file);
            let mut arities: Vec<usize> = snap.edges.iter().map(|e| e.arcs.len()).collect();
            arities.sort_unstable();
            assert_eq!(arities, c.arities, "{}: hyperedge arities", c.file);
        }
    }

    // Representative outputs of the editor's "Generate" tab
    // (`docs/editor/views/generators.js`), captured verbatim. The JS
    // `generators.test.mjs` proves the *combinatorics*; this proves the emitted
    // source — including the generated identifiers (`sts9`/`STS_9`, `k5_3u`,
    // `Sunflower_k3`) — compiles through the editor's exact WASM pipeline.
    const GEN_STS9: &str = r#"STS_9{}
sts9
{
    n0 {}
    n1 {}
    n2 {}
    n3 {}
    n4 {}
    n5 {}
    n6 {}
    n7 {}
    n8 {}

    @e0{ (~n0, ~n3, ~n6); }
    @e1{ (~n1, ~n4, ~n7); }
    @e2{ (~n2, ~n5, ~n8); }
    @e3{ (~n0, ~n1, ~n5); }
    @e4{ (~n3, ~n4, ~n8); }
    @e5{ (~n6, ~n7, ~n2); }
    @e6{ (~n0, ~n2, ~n4); }
    @e7{ (~n3, ~n5, ~n7); }
    @e8{ (~n6, ~n8, ~n1); }
    @e9{ (~n1, ~n2, ~n3); }
    @e10{ (~n4, ~n5, ~n6); }
    @e11{ (~n7, ~n8, ~n0); }
}
"#;

    const GEN_SUNFLOWER: &str = r#"Sunflower_k3{}
sunflower
{
    c0 {}
    c1 {}
    p0v0 {}
    p0v1 {}
    p1v0 {}
    p1v1 {}
    p2v0 {}
    p2v1 {}

    @petal0{ (~c0, ~c1, ~p0v0, ~p0v1); }
    @petal1{ (~c0, ~c1, ~p1v0, ~p1v1); }
    @petal2{ (~c0, ~c1, ~p2v0, ~p2v1); }
}
"#;

    const GEN_COMPLETE_5_3: &str = r#"Complete_5_3{}
k5_3u
{
    v0 {}
    v1 {}
    v2 {}
    v3 {}
    v4 {}

    @e0{ (~v0, ~v1, ~v2); }
    @e1{ (~v0, ~v1, ~v3); }
    @e2{ (~v0, ~v1, ~v4); }
    @e3{ (~v0, ~v2, ~v3); }
    @e4{ (~v0, ~v2, ~v4); }
    @e5{ (~v0, ~v3, ~v4); }
    @e6{ (~v1, ~v2, ~v3); }
    @e7{ (~v1, ~v2, ~v4); }
    @e8{ (~v1, ~v3, ~v4); }
    @e9{ (~v2, ~v3, ~v4); }
}
"#;

    #[test]
    fn generated_sources_compile_to_expected_hypergraphs() {
        // (source, nodes incl. block decl, edges, arity-of-every-edge)
        let cases = [
            (GEN_STS9, 10usize, 12usize, 3usize),
            (GEN_SUNFLOWER, 9, 3, 4),
            (GEN_COMPLETE_5_3, 6, 10, 3),
        ];
        for (src, nodes, edges, arity) in cases {
            let doc = compile_source(src).expect("generated source should compile");
            let snap = doc.snapshot();
            assert_eq!(snap.nodes.len(), nodes, "node count");
            assert_eq!(snap.edges.len(), edges, "edge count");
            assert!(
                snap.edges.iter().all(|e| e.arcs.len() == arity),
                "every hyperedge should have arity {arity}"
            );
        }
    }

    #[test]
    fn typical_graphs_compile_within_budget() {
        // Gross-regression guard, not a reported criterion benchmark: these are
        // tiny fixed fixtures with no algorithmic hot path (see report §3
        // deviation). Median of 5 after one warm-up; ceiling is generous to
        // avoid host-speed flakiness.
        const CEILING_US: u128 = 250_000; // 250 ms
        for c in CASES {
            let src = read_fixture(c.file);
            let _ = compile_source(&src).expect("warm-up compile");
            let mut times = [0u128; 5];
            for t in times.iter_mut() {
                let start = Instant::now();
                let _ = compile_source(&src).expect("compile");
                *t = start.elapsed().as_micros();
            }
            times.sort_unstable();
            let median = times[2];
            eprintln!("{}: compile median {median} µs", c.file);
            assert!(
                median < CEILING_US,
                "{}: compile median {median} µs exceeds {CEILING_US} µs",
                c.file
            );
        }
    }
}
