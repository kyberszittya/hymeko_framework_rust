// Editor vocabulary profiles. A *profile* is a meta-vocabulary the editor can
// author against; the robot `kit` is just one of several. Each profile bundles:
//   • files   — meta `.hymeko` sources loaded into the multi-file compile
//               "space" (so meta elements live OUTSIDE the current context and
//               are `@"..."`-imported by the root);
//   • root    — a starter root document that imports those files;
//   • palette — the "add" kinds for that vocabulary (drives the palette UI
//               generically — no per-profile UI code).
//
// The HRI / SysML meta files + roots are embedded VERBATIM from data/profiles/*
// (profiles.test.mjs pins embed ≡ file; the Rust `editor_profile_roots_compile_with_their_metas`
// test pins that each (root, meta) pair compiles). The kinematics root reuses
// the inline example (single-file, vocabulary inlined) — no duplication.
import { exampleById } from "./examples.js?v=20";

/** Virtual filename of the editable root document in the compile space. */
export const ROOT_NAME = "input.hymeko";

// ── Embedded meta vocabularies (== data/profiles/*.hymeko) ────────────
const META_HRI = `// HRI coalition vocabulary (editor profile). Compact subset of
// data/coalitions/meta_hri.hymeko: agents are nodes, relations are signed
// links between them.
hri_meta_description {}

hri_meta {
    human {}
    robot {}
    interpersonal {}
    hri_relation {}
}
`;

const META_SYSML = `// SysML requirements-trace vocabulary (editor profile). Requirements and
// blocks are nodes; satisfies / derives / allocated_to are signed hyperedges.
meta_sysml_trace {}

sysml_trace {
    elements {
        meta_element {}
        requirement: + <isa> meta_element {}
        block: + <isa> meta_element {}
    }
    @satisfies {}
    @derives {}
    @allocated_to {}
}
`;

// ── Embedded profile roots (== data/profiles/*.hymeko) ────────────────
const HRI_ROOT = `// Editor profile root — HRI coalition. The vocabulary lives outside this file
// (imported from meta_hri.hymeko), so the meta elements are organised in the
// shared "space", not inline. Mirrors data/coalitions/triad_hri.hymeko.
hri_cell_description {
    @"meta_hri.hymeko";
    using hri_meta as hri;
}

hri_cell: hri {
    alice: hri.human {}
    bob:   hri.human {}
    r1:    hri.robot {}

    // Relations are signed hyperedges over the agents (a coalition triangle),
    // not property nodes. All-positive arcs → a balanced σ-cycle.
    @r_ab: + <isa> hri.interpersonal { (+ alice, + bob); }
    @r_ar: + <isa> hri.hri_relation  { (+ alice, + r1); }
    @r_br: + <isa> hri.hri_relation  { (+ bob,   + r1); }
}
`;

const SYSML_ROOT = `// Editor profile root — SysML requirements trace. The vocabulary is imported
// from meta_sysml_trace.hymeko (kept in the shared "space"), not inlined.
sysml_cell_description {
    @"meta_sysml_trace.hymeko";
    using sysml_trace as st;
}

sysml_cell: st.elements {
    R1_safe_stop:    st.elements.requirement { text "Cell halts within 200 ms on E-stop."; }
    R2_pos_accuracy: st.elements.requirement { text "Positioning error <= 0.5 mm."; }

    SafetyController: st.elements.block {}
    MotionPlanner:    st.elements.block {}

    @sat_safe: st.satisfies { (+ SafetyController, - R1_safe_stop); }
    @sat_pos:  st.satisfies { (+ MotionPlanner, - R2_pos_accuracy); }
}
`;

// Kinematics vocabulary + a small robot-arm hero cell (multi-file: the arm
// imports its vocabulary, like the real data/robotics fixtures).
const META_KINEMATICS = `// Kinematics vocabulary (editor profile). Compact subset of
// data/robotics/meta_kinematics.hymeko covering what the robot_arm hero cell
// needs: link elements, joint edge-types, geometry primitives, and axes.
meta_kinematics {}

kinematics {
    elements {
        meta_element {}
        link: + <isa> meta_element {}
        @joint {}
    }

    geometry {
        box {}
        cylinder {}
        sphere {}
    }

    axes {
        axis_definition {}
        AXIS_X: + <isa> axis_definition { ax [1.0, 0.0, 0.0]; }
        AXIS_Y: + <isa> axis_definition { ax [0.0, 1.0, 0.0]; }
        AXIS_Z: + <isa> axis_definition { ax [0.0, 0.0, 1.0]; }
    }

    @fixed_joint:      + <isa> elements.joint {}
    @rev_joint:        + <isa> elements.joint {}
    @conti_joint:      + <isa> elements.joint {}
    @prismatic_joint:  + <isa> elements.joint {}
}
`;

const ROBOT_ARM_ROOT = `// Editor profile root — a small robot arm (the hero-demo kinematic cell). The
// kinematics vocabulary is imported from meta_kinematics.hymeko (kept in the
// shared "space"), not inlined. Two links + one continuous joint whose arc
// carries an origin transform; emits to URDF / SDF / MJCF.
robot_arm_description {
    @"meta_kinematics.hymeko";
}

robot_arm: meta_kinematics.kinematics.elements,
           meta_kinematics.kinematics.geometry,
           meta_kinematics.kinematics.axes
{
    base_link: meta_kinematics.kinematics.elements.link {
        mass 5.0;
        link_geometry: meta_kinematics.kinematics.geometry.box {
            dimension [0.3, 0.3, 0.1];
        }
        visual    -> link_geometry;
        collision -> link_geometry;
        origin [0.0, 0.0, 0.05];
    }

    spinner: meta_kinematics.kinematics.elements.link {
        mass 1.0;
        link_geometry: meta_kinematics.kinematics.geometry.cylinder {
            dimension [0.1, 0.2];
        }
        visual    -> link_geometry;
        collision -> link_geometry;
        origin [0.0, 0.0, 0.1];
    }

    @spin_joint: meta_kinematics.kinematics.conti_joint {
        (+ base_link [[0.0, 0.0, 0.1], [0.0, 0.0, 0.0]],
         - spinner,
         - meta_kinematics.kinematics.axes.AXIS_Z);
    }
}
`;

// Embedded files that mirror an on-disk fixture (consistency-tested). The inline
// kinematics root is reused from examples.js, so it has no entry here.
export const EMBEDDED_FILES = {
  "meta_hri.hymeko": META_HRI,
  "meta_sysml_trace.hymeko": META_SYSML,
  "hri_cell.hymeko": HRI_ROOT,
  "sysml_cell.hymeko": SYSML_ROOT,
  "meta_kinematics.hymeko": META_KINEMATICS,
  "robot_arm.hymeko": ROBOT_ARM_ROOT,
};

// ── Profile registry ──────────────────────────────────────────────────
// palette entry: { label, isEdge, base, isa? (edges), hasMass? (nodes) }.
//   node → `name: base { … }`;  edge → `@name[: + <isa>] base { (+P, -C); }`.
export const PROFILES = [
  {
    id: "kinematics",
    label: "Kinematics (robot)",
    blurb: "URDF/SDF-style links and joints (the kit vocabulary).",
    files: {}, // single-file: the kit vocabulary is inlined in the root
    root: exampleById("kinematic")?.source ?? "",
    palette: [
      { label: "+ Link", isEdge: false, base: "kit.elements.link", hasMass: true },
      { label: "+ Revolute joint", isEdge: true, isa: true, base: "kit.joints.rev_joint" },
      { label: "+ Continuous joint", isEdge: true, isa: true, base: "kit.joints.conti_joint" },
      { label: "+ Prismatic joint", isEdge: true, isa: true, base: "kit.joints.prismatic_joint" },
      { label: "+ Fixed joint", isEdge: true, isa: true, base: "kit.joints.fixed_joint" },
    ],
  },
  {
    id: "hri",
    label: "HRI coalition",
    blurb: "Humans + robots as agents; signed relations between them.",
    files: { "meta_hri.hymeko": META_HRI },
    root: HRI_ROOT,
    palette: [
      { label: "+ Human", isEdge: false, base: "hri.human" },
      { label: "+ Robot", isEdge: false, base: "hri.robot" },
      { label: "+ Relation", isEdge: true, isa: true, base: "hri.hri_relation" },
    ],
  },
  {
    id: "robot_arm",
    label: "Robot arm (imported kinematics)",
    blurb: "A small arm whose link/joint vocabulary is imported from a separate meta file.",
    files: { "meta_kinematics.hymeko": META_KINEMATICS },
    root: ROBOT_ARM_ROOT,
    palette: [
      { label: "+ Link", isEdge: false, base: "meta_kinematics.kinematics.elements.link", hasMass: true },
      { label: "+ Continuous joint", isEdge: true, isa: false, base: "meta_kinematics.kinematics.conti_joint" },
      { label: "+ Revolute joint", isEdge: true, isa: false, base: "meta_kinematics.kinematics.rev_joint" },
      { label: "+ Prismatic joint", isEdge: true, isa: false, base: "meta_kinematics.kinematics.prismatic_joint" },
      { label: "+ Fixed joint", isEdge: true, isa: false, base: "meta_kinematics.kinematics.fixed_joint" },
    ],
  },
  {
    id: "sysml_trace",
    label: "SysML trace",
    blurb: "Requirements + blocks; satisfies / derives / allocated_to edges.",
    files: { "meta_sysml_trace.hymeko": META_SYSML },
    root: SYSML_ROOT,
    palette: [
      { label: "+ Requirement", isEdge: false, base: "st.elements.requirement" },
      { label: "+ Block", isEdge: false, base: "st.elements.block" },
      { label: "+ Satisfies", isEdge: true, isa: false, base: "st.satisfies" },
      { label: "+ Derives", isEdge: true, isa: false, base: "st.derives" },
      { label: "+ Allocated to", isEdge: true, isa: false, base: "st.allocated_to" },
    ],
  },
];

/** Look up a profile by id, or null. */
export function profileById(id) {
  return PROFILES.find((p) => p.id === id) ?? null;
}
