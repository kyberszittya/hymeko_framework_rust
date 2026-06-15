// Unit tests for the pure view adapters. Run:
//   node --test docs/editor/views/adapters.test.mjs
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  snapshotToHyperedges,
  snapshotToKinematicGraph,
  snapshotRelationships,
  scopeDepths,
  bidirectionalEdgeIds,
  cycleArity,
  parseUrdf,
} from "./adapters.js";

// A minimal snapshot: 3 vertex nodes + 1 hyperedge over all three, signs +,+,-.
const SNAP = {
  nodes: [
    { id: 10, name: "n0", kind: "Node", bases: [], tags: [], arcs: [] },
    { id: 11, name: "n1", kind: "Node", bases: [], tags: [], arcs: [] },
    { id: 12, name: "n2", kind: "Node", bases: [], tags: [], arcs: [] },
  ],
  edges: [
    {
      id: 20, name: "e0", kind: "Edge", bases: [], tags: [],
      arcs: [
        { sign: 1, target_id: 10, target_name: "n0" },
        { sign: 1, target_id: 11, target_name: "n1" },
        { sign: -1, target_id: 12, target_name: "n2" },
      ],
    },
  ],
};

test("snapshotToHyperedges: members/sign/arity from arcs", () => {
  const h = snapshotToHyperedges(SNAP);
  assert.equal(h.n_vertices, 3);
  assert.deepEqual(h.vertex_labels, ["n0", "n1", "n2"]);
  assert.equal(h.hyperedges.length, 1);
  assert.deepEqual(h.hyperedges[0].members, [0, 1, 2]);
  assert.equal(h.hyperedges[0].arity, 3);
  assert.equal(h.hyperedges[0].sign, -1); // one negative arc → product −1
});

test("snapshotToHyperedges: arcs to non-vertex targets are dropped, not invented", () => {
  const snap = {
    nodes: [{ id: 1, name: "a", arcs: [] }],
    edges: [{ id: 2, name: "e", arcs: [
      { sign: 1, target_id: 1 }, { sign: 1, target_id: 999 },
    ] }],
  };
  const h = snapshotToHyperedges(snap);
  assert.deepEqual(h.hyperedges[0].members, [0]); // 999 unknown → skipped
});

test("snapshotToKinematicGraph: (+parent,−child) → directed link pair", () => {
  const snap = {
    nodes: [{ id: 1, name: "base_link", arcs: [] }, { id: 2, name: "spinner", arcs: [] }],
    edges: [{ id: 3, name: "spin_joint", arcs: [
      { sign: 1, target_id: 1 }, { sign: -1, target_id: 2 },
    ] }],
  };
  const g = snapshotToKinematicGraph(snap);
  assert.deepEqual(g.links, ["base_link", "spinner"]);
  assert.equal(g.joints.length, 1);
  assert.deepEqual(g.joints[0], { name: "spin_joint", sign: -1, parent: "base_link", child: "spinner" });
  assert.deepEqual(g.edges, [[0, 1]]);
});

test("snapshotToKinematicGraph: nodes not referenced by any joint are excluded", () => {
  const snap = {
    nodes: [
      { id: 1, name: "base_link", arcs: [] },
      { id: 2, name: "spinner", arcs: [] },
      { id: 9, name: "kit_namespace_decl", arcs: [] }, // scaffolding, no joint
    ],
    edges: [{ id: 3, name: "j", arcs: [
      { sign: 1, target_id: 1 }, { sign: -1, target_id: 2 },
    ] }],
  };
  const g = snapshotToKinematicGraph(snap);
  assert.deepEqual(g.links, ["base_link", "spinner"]); // kit_namespace_decl dropped
  assert.equal(g.nLinks, 2);
  assert.deepEqual(g.edges, [[0, 1]]);
});

test("snapshotRelationships: groups by kind, maps ids→indices, drops danglers", () => {
  const snap = {
    nodes: [{ id: 5, name: "kit" }, { id: 6, name: "elements" }, { id: 7, name: "link" }],
    relationships: [
      { kind: "scope", from: 6, to: 5 },   // elements ⊂ kit  → [1,0]
      { kind: "scope", from: 7, to: 6 },   // link ⊂ elements → [2,1]
      { kind: "isa", from: 7, to: 99 },    // base 99 not a vertex → dropped
      { kind: "scope", from: 5, to: 5 },   // self-loop → dropped
    ],
  };
  const r = snapshotRelationships(snap);
  assert.deepEqual(r.scope, [[1, 0], [2, 1]]);
  assert.equal(r.isa, undefined); // the only isa had a dangling endpoint
});

test("snapshotRelationships: empty when snapshot has no relationships", () => {
  assert.deepEqual(snapshotRelationships({ nodes: [{ id: 1, name: "a" }] }), {});
});

test("cycleArity: a triangle has exactly one 3-cycle", () => {
  const g = { nLinks: 3, edges: [[0, 1], [1, 2], [0, 2]] };
  assert.deepEqual(cycleArity(g), { 3: 1, 4: 0, 5: 0, 6: 0 });
});

test("cycleArity: a 4-cycle is counted once as k=4, not as two triangles", () => {
  const g = { nLinks: 4, edges: [[0, 1], [1, 2], [2, 3], [3, 0]] };
  assert.deepEqual(cycleArity(g), { 3: 0, 4: 1, 5: 0, 6: 0 });
});

test("cycleArity: K4 has 4 triangles and 3 quadrilaterals", () => {
  const g = { nLinks: 4, edges: [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]] };
  assert.deepEqual(cycleArity(g), { 3: 4, 4: 3, 5: 0, 6: 0 });
});

// Real HyMeKo-emitted URDF (cylinder + box links, one continuous joint).
const URDF_GEO = `<?xml version="1.0" encoding="UTF-8"?>
<robot name="robot">
  <link name="base_link">
    <inertial><mass value="5"/></inertial>
    <visual>
      <origin xyz="0 0 0.1"/>
      <geometry><cylinder radius="0.05" length="0.2"/></geometry>
    </visual>
  </link>
  <link name="spinner">
    <inertial><mass value="1"/></inertial>
    <visual>
      <origin xyz="0 0 0.25"/>
      <geometry><box size="0.1 0.1 0.1"/></geometry>
    </visual>
  </link>
  <joint name="spin_joint" type="continuous">
    <parent link="base_link"/>
    <child link="spinner"/>
    <origin xyz="0 0 0.2" rpy="0.0000 0.0000 0.0000"/>
    <limit effort="100" velocity="1.0"/>
  </joint>
</robot>`;

test("parseUrdf: extracts cylinder/box geometry, origins, and the joint", () => {
  const { links, joints } = parseUrdf(URDF_GEO);
  assert.equal(links.length, 2);
  assert.deepEqual(links[0].geometry, { shape: "cylinder", radius: 0.05, length: 0.2 });
  assert.deepEqual(links[0].origin, [0, 0, 0.1]);
  assert.deepEqual(links[1].geometry, { shape: "box", size: [0.1, 0.1, 0.1] });
  assert.equal(joints.length, 1);
  assert.equal(joints[0].parent, "base_link");
  assert.equal(joints[0].child, "spinner");
  assert.deepEqual(joints[0].origin_xyz, [0, 0, 0.2]);
});

// URDF with no <visual> and a joint without <origin>/<axis> — the graceful case.
const URDF_BARE = `<robot name="r">
  <link name="a"><inertial><mass value="5"/></inertial></link>
  <link name="b"><inertial><mass value="1"/></inertial></link>
  <joint name="j" type="continuous"><parent link="a"/><child link="b"/></joint>
</robot>`;

test("parseUrdf: links without geometry → null; joint without origin/axis → defaults", () => {
  const { links, joints } = parseUrdf(URDF_BARE);
  assert.equal(links[0].geometry, null);
  assert.deepEqual(joints[0].origin_xyz, [0, 0, 0]);
  assert.equal(joints[0].axis, null);
});

test("scopeDepths: chain a⊃b⊃c gives depths 0/1/2; multi-root each 0", () => {
  const snap = {
    nodes: [{ id: 0 }, { id: 1 }, { id: 2 }, { id: 7 }],
    edges: [{ id: 5 }],
    relationships: [
      { kind: "scope", from: 1, to: 0 }, // b in a
      { kind: "scope", from: 2, to: 1 }, // c in b
      { kind: "scope", from: 5, to: 0 }, // edge 5 in a
      { kind: "isa", from: 2, to: 9 },   // non-scope ignored
      { kind: "scope", from: 7, to: 7 }, // self-scope ignored
    ],
  };
  const d = scopeDepths(snap);
  assert.equal(d.get(0), 0); // root container
  assert.equal(d.get(1), 1);
  assert.equal(d.get(2), 2);
  assert.equal(d.get(5), 1); // edge nested one level
  assert.equal(d.get(7), 0); // self-scope → still a root
});

test("scopeDepths: empty / no-relationship snapshots", () => {
  assert.equal(scopeDepths({}).size, 0);
  const d = scopeDepths({ nodes: [{ id: 0 }, { id: 1 }], edges: [] });
  assert.equal(d.get(0), 0);
  assert.equal(d.get(1), 0);
});

test("bidirectionalEdgeIds: only reciprocal pairs, no self-loops or one-way", () => {
  const edges = [
    { id: "x", source: "a", target: "b" },  // reciprocal with y
    { id: "y", source: "b", target: "a" },
    { id: "z", source: "a", target: "c" },  // one-way
    { id: "p", source: "a", target: "b" },  // parallel same-dir as x (still bidir: b->a exists)
    { id: "s", source: "d", target: "d" },  // self-loop
  ];
  const ids = bidirectionalEdgeIds(edges);
  assert.ok(ids.has("x") && ids.has("y"));
  assert.ok(ids.has("p")); // a->b with a reverse b->a present is bidirectional
  assert.ok(!ids.has("z"));
  assert.ok(!ids.has("s"));
  assert.equal(bidirectionalEdgeIds([]).size, 0);
  assert.equal(bidirectionalEdgeIds(undefined).size, 0);
});
