// Editor example gallery — data-driven catalog of built-in `.hymeko` sources.
//
// Each entry is { id, label, source, view }:
//   • source : the canonical `.hymeko` text, embedded as a string. The editor
//              is served from docs/editor/ (README "Build + serve"), so the
//              repo's data/ tree is NOT fetch-reachable — sources must be
//              embedded, exactly as the original EXAMPLE/EXAMPLE_TRACE were.
//   • view   : the view tab to switch to after loading (null = stay put). The
//              classic hypergraphs jump to "hyper3d" so the 3D star/clique/
//              prism rendering is shown immediately.
//
// The four classic-hypergraph entries embed data/typical_graphs/*.hymeko
// VERBATIM. examples.test.mjs asserts embed ≡ fixture (line-ending-normalised)
// so the two cannot silently drift. Pure data + one lookup helper: no DOM, no
// three.js, no globals, so this runs unchanged under `node --test`.

// ── Kinematic arm (default) ───────────────────────────────────────────
// Self-contained: the kinematics kinds are the inline `kit` namespace, so the
// browser needs no `@"…"` include (the WASM compiles a single in-memory source).
const KINEMATIC = `hymeko_editor_example {}

kit {
    elements {
        meta_element {}
        link: + <isa> meta_element {}
    }
    joints {
        meta_joint {}
        rev_joint: + <isa> meta_joint {}
        conti_joint: + <isa> meta_joint {}
        prismatic_joint: + <isa> meta_joint {}
        fixed_joint: + <isa> meta_joint {}
    }
    geometry { box {} cylinder {} sphere {} }
}

robot: kit.elements, kit.joints, kit.geometry {
    base_link: kit.elements.link {
        mass 5.0;
        link_geometry: kit.geometry.cylinder { dimension [0.05, 0.2]; }
        visual -> link_geometry;
        origin [0.0, 0.0, 0.1];
    }
    spinner: kit.elements.link {
        mass 1.0;
        link_geometry: kit.geometry.box { dimension [0.1, 0.1, 0.1]; }
        visual -> link_geometry;
        origin [0.0, 0.0, 0.25];
    }

    @spin_joint: + <isa> kit.joints.conti_joint {
        (+ base_link [[0.0, 0.0, 0.2], [0.0, 0.0, 0.0]], - spinner);
    }
}
`;

// ── SysML requirements-trace witness (SMC #5) ─────────────────────────
// Self-contained: the `sysml_trace` vocabulary is inline. Requirements and
// blocks are vertices; satisfies / derives / allocated_to are signed hyperedges.
const REQ_TRACE = `hymeko_editor_trace_example {}

sysml_trace {
    elements {
        meta_element {}
        requirement: + <isa> meta_element {}
        block:       + <isa> meta_element {}
    }
    @satisfies {}
    @derives {}
    @allocated_to {}
}

pick_place_cell: sysml_trace.elements {
    R1_safe_stop:      sysml_trace.elements.requirement { text "Cell halts within 200 ms on E-stop."; }
    R2_pos_accuracy:   sysml_trace.elements.requirement { text "Positioning error <= 0.5 mm."; }
    R3_cycle_time:     sysml_trace.elements.requirement { text "Cycle <= 4 s."; }
    R2a_repeatability: sysml_trace.elements.requirement { text "Repeatability <= 0.1 mm."; }

    SafetyController: sysml_trace.elements.block {}
    MotionPlanner:    sysml_trace.elements.block {}
    Arm:              sysml_trace.elements.block {}
    Gripper:          sysml_trace.elements.block {}

    @sat_safe:    sysml_trace.satisfies { (+ SafetyController, - R1_safe_stop); }
    @sat_pos_mp:  sysml_trace.satisfies { (+ MotionPlanner, - R2_pos_accuracy); }
    @sat_pos_arm: sysml_trace.satisfies { (+ Arm, - R2_pos_accuracy); }
    @sat_cycle:   sysml_trace.satisfies { (+ MotionPlanner, - R3_cycle_time); }
    @sat_grip:    sysml_trace.satisfies { (+ Gripper, - R3_cycle_time); }
    @der_rep:     sysml_trace.derives      { (+ R2a_repeatability, - R2_pos_accuracy); }
    @alloc_rep:   sysml_trace.allocated_to { (+ R2a_repeatability, - Arm); }
}
`;

// ── Classic combinatorial hypergraphs ─────────────────────────────────
// Pure undirected hypergraphs (neutral `~` arcs). Embedded VERBATIM from
// data/typical_graphs/*.hymeko — keep these byte-faithful; the consistency
// test guards drift. Each is a Steiner-/design-theory object that renders as a
// set of arity-≥3 prisms in the Hypergraph 3D view.

// Fano plane = Steiner triple system S(2,3,7): 7 points, 7 lines, every pair of
// points on exactly one line; every point on exactly 3 lines.
const FANO = `Fano_graph{}
fano
{
    n0 {}
    n1 {}
    n2 {}
    n3 {}
    n4 {}
    n5 {}
    n6 {}

    @e0{ (~n0, ~n1, ~n3 ); }
    @e1{ (~n0, ~n2, ~n6 ); }
    @e2{ (~n0, ~n4, ~n5 ); }
    @e3{ (~n1, ~n2, ~n4 ); }
    @e4{ (~n2, ~n3, ~n5 ); }
    @e5{ (~n3, ~n4, ~n6 ); }
    @e6{ (~n1, ~n5, ~n6 ); }
}`;

// Sunflower / Δ-system: 3 petals sharing a common core {core_a, core_b}; the
// pairwise intersection of any two edges is exactly the core.
const SUNFLOWER = `Sunflower_delta{}
sunflower
{
    core_a {}
    core_b {}
    p1a {}
    p1b {}
    p2a {}
    p2b {}
    p3a {}
    p3b {}

    @petal1{ (~core_a, ~core_b, ~p1a, ~p1b); }
    @petal2{ (~core_a, ~core_b, ~p2a, ~p2b); }
    @petal3{ (~core_a, ~core_b, ~p3a, ~p3b); }
}
`;

// Complete 3-uniform hypergraph on 4 vertices: all C(4,3)=4 triples — the four
// triangular faces of a tetrahedron.
const K4_3UNIFORM = `K4_3uniform{}
k4_3uniform
{
    v1 {}
    v2 {}
    v3 {}
    v4 {}

    @f123{ (~v1, ~v2, ~v3); }
    @f124{ (~v1, ~v2, ~v4); }
    @f134{ (~v1, ~v3, ~v4); }
    @f234{ (~v2, ~v3, ~v4); }
}
`;

// Generic mixed-arity hypergraph: a small irregular example (two arity-3 and one
// arity-2 edge sharing vertices) for contrast with the regular designs above.
const GENERIC = `Generic_hypergraph{}
generic
{
    a {}
    b {}
    c {}
    d {}
    e {}
    f {}
    g {}

    @e0{ (~a, ~b, ~c); }
    @e1{ (~c, ~d, ~f); }
    @e2{ (~d, ~e); }
    @e3{ (~c, ~g, ~f); }
}
`;

/** Built-in example catalog, in gallery order. */
export const EXAMPLES = [
  { id: "kinematic", label: "Kinematic arm",                 source: KINEMATIC,   view: null },
  { id: "trace",     label: "Requirements trace (SysML)",    source: REQ_TRACE,   view: null },
  { id: "fano",      label: "Fano plane — S(2,3,7)",         source: FANO,        view: "hyper3d" },
  { id: "sunflower", label: "Sunflower (Δ-system)",          source: SUNFLOWER,   view: "hyper3d" },
  { id: "k4_3",      label: "K₄ complete 3-uniform",         source: K4_3UNIFORM, view: "hyper3d" },
  { id: "generic",   label: "Generic hypergraph",            source: GENERIC,     view: "hyper3d" },
];

/**
 * Look up an example by id.
 * Preconditions: `id` is a string.
 * Postconditions: returns the matching catalog entry, or null if none.
 */
export function exampleById(id) {
  return EXAMPLES.find((e) => e.id === id) ?? null;
}

/**
 * Map a catalog id → its canonical fixture filename under data/typical_graphs/.
 * Used ONLY by the consistency test (the browser never reads these files).
 * Entries absent here are inline-only examples with no on-disk fixture.
 */
export const FIXTURE_OF = {
  fano: "fano_graph.hymeko",
  sunflower: "sunflower_delta_system.hymeko",
  k4_3: "k4_3uniform.hymeko",
  generic: "generic_hypergraph.hymeko",
};
