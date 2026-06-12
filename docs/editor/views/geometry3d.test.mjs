// Units for the pure 3D-geometry + colour helpers.
//   node --test docs/editor/views/geometry3d.test.mjs
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  attributeValue, categoricalColorMap, newellNormal, prismPositions, treePositions,
  coneTreePositions, PALETTE,
} from "./geometry3d.js";

test("attributeValue: first base / first tag / fallback", () => {
  assert.equal(attributeValue({ bases: ["link", "meta_element"] }, "base"), "link");
  assert.equal(attributeValue({ tags: ["actuated"] }, "tag"), "actuated");
  assert.equal(attributeValue({ bases: [] }, "base"), "(none)");
  assert.equal(attributeValue({}, "uniform"), "(all)");
});

test("categoricalColorMap: stable, one colour per distinct value", () => {
  const m = categoricalColorMap(["link", "joint", "link", "frame"]);
  assert.equal(m.size, 3);
  assert.equal(m.get("frame"), PALETTE[0]); // sorted: frame, joint, link
  assert.equal(m.get("joint"), PALETTE[1]);
  assert.equal(m.get("link"), PALETTE[2]);
  // Stable across calls regardless of input order.
  const m2 = categoricalColorMap(["frame", "link", "joint"]);
  assert.equal(m2.get("link"), m.get("link"));
});

test("newellNormal: unit square in XY plane → ±Z, unit length", () => {
  const n = newellNormal([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]);
  assert.ok(Math.abs(Math.abs(n[2]) - 1) < 1e-9, `z≈±1, got ${n[2]}`);
  assert.ok(Math.abs(Math.hypot(...n) - 1) < 1e-9, "unit length");
});

test("prismPositions: < 3 points → null (connector handled elsewhere)", () => {
  assert.equal(prismPositions([[0, 0, 0], [1, 0, 0]]), null);
});

test("prismPositions: square prism → normal ≈ Z and correct triangle count", () => {
  const r = prismPositions([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], 0.2);
  assert.ok(r);
  assert.ok(Math.abs(Math.abs(r.normal[2]) - 1) < 1e-9);
  // n=4: sides 2n=8 tris, caps 2(n-2)=4 tris → 12 tris → 36 verts → 108 floats.
  assert.equal(r.positions.length, 108);
});

test("prismPositions: triangle prism (n=3) → 8 triangles (6 sides + 2 caps)", () => {
  const r = prismPositions([[0, 0, 0], [1, 0, 0], [0, 1, 0]], 0.1);
  // n=3: sides 2n=6, caps 2(n-2)=2 → 8 tris → 24 verts → 72 floats.
  assert.equal(r.positions.length, 72);
});

test("treePositions: barycentric composition tree (root over its children)", () => {
  // 0 ⊃ {1,2};  1 ⊃ {3,4}. scope edges are [child, parent].
  const { x, depth, width } = treePositions(5, [[1, 0], [2, 0], [3, 1], [4, 1]]);
  assert.deepEqual([...depth], [0, 1, 1, 2, 2]);
  assert.equal(x[3], 0); assert.equal(x[4], 1);   // leaves sequential
  assert.equal(x[1], 0.5);                         // parent at mean of 3,4
  assert.equal(x[2], 2);
  assert.equal(x[0], (0.5 + 2) / 2);               // root at mean of 1,2
  assert.equal(width, 3);                          // 3 leaves
});

test("treePositions: a node with no parent is a root at depth 0", () => {
  const { depth } = treePositions(3, [[1, 0]]); // 2 is isolated
  assert.equal(depth[0], 0); assert.equal(depth[1], 1); assert.equal(depth[2], 0);
});

test("coneTreePositions: root sits at the 3D barycentre of its subtree", () => {
  // 0 ⊃ {1,2}; 1 ⊃ {3,4}. By symmetry the root collapses to the axis.
  const { X, Y, Z, depth } = coneTreePositions(5, [[1, 0], [2, 0], [3, 1], [4, 1]]);
  assert.deepEqual([...depth], [0, 1, 1, 2, 2]);
  assert.ok(Math.abs(X[0]) < 1e-9 && Math.abs(Z[0]) < 1e-9, "root on the axis (barycentre)");
  assert.ok(Y[0] === 0, "root at the top"); // -0 === 0
  assert.ok(Y[3] < Y[1] && Y[1] < Y[0], "deeper nodes are lower");
  // An internal node is the mean of its children's X/Z.
  assert.ok(Math.abs(X[1] - (X[3] + X[4]) / 2) < 1e-9);
  assert.ok(Math.abs(Z[1] - (Z[3] + Z[4]) / 2) < 1e-9);
});
