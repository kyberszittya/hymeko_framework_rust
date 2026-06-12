// Pure 3D-geometry + colour helpers for the hypergraph view. No three.js, no
// DOM — node-testable (see geometry3d.test.mjs). Points are plain [x,y,z].

// Categorical palette (distinct, legible on the dark 3D background).
export const PALETTE = [
  0x7dd3fc, 0xfbbf24, 0xf472b6, 0x34d399, 0xa78bfa,
  0xfb7185, 0x38bdf8, 0xfcd34d, 0x4ade80, 0xc084fc,
  0xf87171, 0x22d3ee,
];
const FALLBACK = 0x9fd6ff;

/**
 * The value used to colour/group a vertex, for a chosen attribute.
 * attr "base" → first inherited base; "tag" → first tag; else a single group.
 * Postcondition: always a string ("(none)" when the attribute is absent).
 */
export function attributeValue(vertex, attr) {
  if (attr === "base") return (vertex?.bases && vertex.bases[0]) || "(none)";
  if (attr === "tag") return (vertex?.tags && vertex.tags[0]) || "(none)";
  return "(all)";
}

/**
 * Stable value → colour map. Values are sorted so colours don't shuffle between
 * renders; more values than the palette wrap around.
 * Postcondition: a Map(value → 0xRRGGBB) with one entry per distinct value.
 */
export function categoricalColorMap(values) {
  const uniq = [...new Set(values)].sort();
  const map = new Map();
  uniq.forEach((v, i) => map.set(v, uniq.length === 1 && v === "(all)" ? FALLBACK : PALETTE[i % PALETTE.length]));
  return map;
}

// ── small vector helpers (plain arrays) ────────────────────────────
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const scale = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a) => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };

/**
 * Newell's-method polygon normal for (possibly non-planar) ordered points.
 * Postcondition: a unit vector; robust to slight non-planarity.
 */
export function newellNormal(points) {
  let n = [0, 0, 0];
  for (let i = 0; i < points.length; i++) {
    const c = points[i], d = points[(i + 1) % points.length];
    n = add(n, [(c[1] - d[1]) * (c[2] + d[2]), (c[2] - d[2]) * (c[0] + d[0]), (c[0] - d[0]) * (c[1] + d[1])]);
  }
  return norm(n);
}

/**
 * Composition-tree layout from scope (containment) edges. Children stack under
 * their parent; each parent sits at the barycentre (mean slot) of its children
 * — "barycentric closeness". Pure (no THREE), so it's unit-tested.
 * Preconditions: scopeSegs = [[childIdx, parentIdx], …] with indices in [0,n).
 * Postconditions: { x:Float64Array(n), depth:Int32Array(n), width:number };
 *   x is the horizontal slot (leaves sequential left→right, internal nodes the
 *   mean of their children), depth is the level below a root, width is the leaf
 *   count (used to centre the tree). Robust to cycles / multi-parent (first
 *   parent wins; any unreached node is placed at depth 0).
 */
function buildScopeTree(n, scopeSegs) {
  const parent = new Int32Array(n).fill(-1);
  const children = Array.from({ length: n }, () => []);
  for (const [c, p] of scopeSegs || []) {
    if (c >= 0 && c < n && p >= 0 && p < n && c !== p && parent[c] === -1) {
      parent[c] = p; children[p].push(c);
    }
  }
  return { parent, children };
}

export function treePositions(n, scopeSegs) {
  const { parent, children } = buildScopeTree(n, scopeSegs);
  const x = new Float64Array(n);
  const depth = new Int32Array(n);
  const seen = new Uint8Array(n);
  let cursor = 0;
  const visit = (u, d) => {
    if (seen[u]) return;
    seen[u] = 1; depth[u] = d;
    if (children[u].length === 0) { x[u] = cursor++; return; }
    let sum = 0, cnt = 0;
    for (const c of children[u]) { visit(c, d + 1); sum += x[c]; cnt++; }
    x[u] = cnt ? sum / cnt : cursor++;
  };
  for (let i = 0; i < n; i++) if (parent[i] === -1) visit(i, 0);
  for (let i = 0; i < n; i++) if (!seen[i]) { seen[i] = 1; x[i] = cursor++; depth[i] = 0; }
  return { x, depth, width: cursor };
}

/**
 * 3D barycentric cone-tree layout from scope (containment) edges. Leaves sit on
 * a ring whose radius grows with depth (deeper → wider); each internal node is
 * placed at the **3D centroid (barycentre) of its children** in the X–Z plane,
 * one level higher (Y = −depth·levelHeight). Pure (no THREE), so it's tested.
 * Preconditions: scopeSegs = [[childIdx, parentIdx], …], indices in [0,n).
 * Postconditions: { X, Y, Z: Float64Array(n), depth: Int32Array(n) }.
 */
export function coneTreePositions(n, scopeSegs, ringRadius = 90, levelHeight = 120) {
  const { parent, children } = buildScopeTree(n, scopeSegs);
  const X = new Float64Array(n), Y = new Float64Array(n), Z = new Float64Array(n);
  const depth = new Int32Array(n);
  const seen = new Uint8Array(n);
  const leaves = [];
  const visit = (u, d) => {
    if (seen[u]) return;
    seen[u] = 1; depth[u] = d;
    if (children[u].length === 0) { leaves.push(u); return; }
    for (const c of children[u]) visit(c, d + 1);
  };
  for (let i = 0; i < n; i++) if (parent[i] === -1) visit(i, 0);
  for (let i = 0; i < n; i++) if (!seen[i]) { seen[i] = 1; depth[i] = 0; leaves.push(i); }

  const L = leaves.length || 1;
  leaves.forEach((u, idx) => {
    const th = (2 * Math.PI * idx) / L, r = depth[u] * ringRadius;
    X[u] = r * Math.cos(th); Z[u] = r * Math.sin(th);
  });
  const setPos = (u) => {
    if (children[u].length === 0) return;
    let sx = 0, sz = 0, cnt = 0;
    for (const c of children[u]) { setPos(c); sx += X[c]; sz += Z[c]; cnt += 1; }
    if (cnt) { X[u] = sx / cnt; Z[u] = sz / cnt; }
  };
  for (let i = 0; i < n; i++) if (parent[i] === -1) setPos(i);
  for (let i = 0; i < n; i++) Y[i] = -depth[i] * levelHeight;
  return { X, Y, Z, depth };
}

/**
 * Triangle-soup positions for a prism whose cross-section is the polygon of
 * `points`, extruded ±thickness/2 along the polygon normal.
 * Preconditions: points.length ≥ 3.
 * Postconditions: { positions:number[] (flat xyz, length = 3·3·(2n + 2(n−2))),
 *   normal:[x,y,z] }. Returns null for < 3 points (caller draws a connector).
 */
export function prismPositions(points, thickness = 0.18) {
  const n = points.length;
  if (n < 3) return null;
  const N = newellNormal(points);
  const centroid = scale(points.reduce(add, [0, 0, 0]), 1 / n);
  // In-plane basis to order the ring by angle.
  let U = cross(N, Math.abs(N[2]) < 0.9 ? [0, 0, 1] : [0, 1, 0]);
  U = norm(U); const V = cross(N, U);
  const ordered = points
    .map((p) => ({ p, a: Math.atan2(dot(sub(p, centroid), V), dot(sub(p, centroid), U)) }))
    .sort((x, y) => x.a - y.a)
    .map((o) => o.p);
  const half = scale(N, thickness / 2);
  const top = ordered.map((p) => add(p, half));
  const bot = ordered.map((p) => sub(p, half));
  const out = [];
  const tri = (a, b, c) => out.push(a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]);
  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n;
    tri(top[i], top[j], bot[j]); tri(top[i], bot[j], bot[i]); // side quad
  }
  for (let i = 1; i < n - 1; i++) { tri(top[0], top[i], top[i + 1]); tri(bot[0], bot[i + 1], bot[i]); } // caps
  return { positions: out, normal: N };
}
