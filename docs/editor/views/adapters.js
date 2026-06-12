// Pure adapters: turn the WASM `snapshot_json()` / `to_urdf()` output into the
// shapes the 3D views consume. No DOM, no three.js, no globals — so these run
// under `node --test` (see adapters.test.mjs) and in the browser unchanged.
//
// snapshot shape (hymeko_formats/src/snapshot.rs):
//   { node_count, edge_count, arc_count,
//     nodes: [{ id, name, kind, bases[], tags[], arcs[] }],
//     edges: [{ id, name, kind, bases[], tags[], arcs: [{ sign, target_id, target_name }] }] }

/**
 * Build the star/clique-viewer input from a snapshot.
 * Preconditions: `snapshot.nodes` / `snapshot.edges` are arrays; each edge arc
 *   carries `target_id` referencing a node id and a `sign` in {-1,0,+1}.
 * Postconditions: returns { n_vertices, vertex_labels[], vertices[], hyperedges[] }
 *   where each hyperedge is { label, members:[vertexIndex], sign:±1, arity } and
 *   each vertex is { label, bases[], tags[] } (metadata for attribute colouring).
 *   Arcs that target a non-vertex (e.g. another edge) are dropped, never
 *   invented. `sign` is the product of member arc signs (0 leaves it unchanged).
 */
export function snapshotToHyperedges(snapshot) {
  const nodes = snapshot?.nodes ?? [];
  const edges = snapshot?.edges ?? [];
  const idToIdx = new Map();
  const vertex_labels = [];
  const vertices = [];
  for (const n of nodes) {
    idToIdx.set(n.id, vertex_labels.length);
    vertex_labels.push(n.name);
    vertices.push({ label: n.name, bases: n.bases ?? [], tags: n.tags ?? [] });
  }
  const hyperedges = [];
  for (const e of edges) {
    const members = [];
    let sign = 1;
    for (const a of e.arcs ?? []) {
      if (!idToIdx.has(a.target_id)) continue;
      members.push(idToIdx.get(a.target_id));
      if (a.sign < 0) sign = -sign;
    }
    if (members.length === 0) continue;
    hyperedges.push({ label: e.name, members, sign, arity: members.length });
  }
  return { n_vertices: vertex_labels.length, vertex_labels, vertices, hyperedges };
}

/**
 * Derive a kinematic link/joint graph from a snapshot. Joints = edge decls;
 * links = only the node decls actually referenced by a joint (so the regime
 * panels show the robot, not the namespace scaffolding). The arc convention
 * (+ parent, − child) maps an edge to a directed link pair.
 * Preconditions: as above.
 * Postconditions: { links:[name], joints:[{name,sign,parent,child}],
 *   edges:[[parentIdx,childIdx]], nLinks }. Edges with < 2 vertex arcs are
 *   skipped. If sign labels are absent, the first/last arcs stand in for
 *   parent/child so a graph still forms.
 */
export function snapshotToKinematicGraph(snapshot) {
  const nodes = snapshot?.nodes ?? [];
  const edges = snapshot?.edges ?? [];
  const nodeById = new Map(nodes.map((n) => [n.id, n]));
  // Restrict links to node decls referenced by at least one edge (joint).
  const referenced = new Set();
  for (const e of edges) for (const a of e.arcs ?? []) if (nodeById.has(a.target_id)) referenced.add(a.target_id);
  const idToIdx = new Map();
  const links = [];
  for (const n of nodes) {
    if (!referenced.has(n.id)) continue;
    idToIdx.set(n.id, links.length);
    links.push(n.name);
  }
  const joints = [];
  const graphEdges = [];
  for (const e of edges) {
    const arcs = (e.arcs ?? []).filter((a) => idToIdx.has(a.target_id));
    if (arcs.length < 2) continue;
    const parentArc = arcs.find((a) => a.sign > 0) ?? arcs[0];
    const childArc = arcs.find((a) => a.sign < 0) ?? arcs[arcs.length - 1];
    let sign = 1;
    for (const a of arcs) if (a.sign < 0) sign = -sign;
    const pu = idToIdx.get(parentArc.target_id);
    const cu = idToIdx.get(childArc.target_id);
    joints.push({ name: e.name, sign, parent: links[pu], child: links[cu] });
    graphEdges.push([pu, cu]);
  }
  return { links, joints, edges: graphEdges, nLinks: links.length };
}

/**
 * Count simple undirected cycles by length k = 3..maxK. Mirrors the stdlib DFS
 * in demo_web/export_kinematic_data.py so the browser αₖ fingerprint matches the
 * Python export.
 * Preconditions: graph = { nLinks, edges:[[u,v]] } with 0 ≤ u,v < nLinks.
 * Postconditions: { 3:n, 4:n, ..., maxK:n }; each cycle counted exactly once
 *   (canonical: smallest start vertex, fixed traversal direction).
 */
export function cycleArity(graph, maxK = 6) {
  const n = graph?.nLinks ?? 0;
  const adj = Array.from({ length: n }, () => new Set());
  for (const [u, v] of graph?.edges ?? []) {
    if (u === v || u < 0 || v < 0 || u >= n || v >= n) continue;
    adj[u].add(v);
    adj[v].add(u);
  }
  const counts = {};
  for (let k = 3; k <= maxK; k++) counts[k] = 0;
  const onPath = new Array(n).fill(false);
  const path = [];

  const dfs = (start, u, depth) => {
    onPath[u] = true;
    path.push(u);
    for (const w of adj[u]) {
      if (w === start && depth >= 3) {
        // Close the cycle. Count once by fixing direction: the first step out
        // of `start` must be smaller than the last step back into it.
        if (path[1] < path[path.length - 1]) counts[depth]++;
      } else if (w > start && !onPath[w] && depth < maxK) {
        dfs(start, w, depth + 1);
      }
    }
    onPath[u] = false;
    path.pop();
  };

  for (let s = 0; s < n; s++) dfs(s, s, 1);
  return counts;
}

/**
 * Group the snapshot's named relationships by kind, mapping DeclId endpoints to
 * vertex indices (same index space as snapshotToHyperedges).
 * Preconditions: `snapshot.relationships` (if present) is [{kind,from,to}] with
 *   from/to referencing node ids.
 * Postconditions: { kind: [[fromIdx, toIdx], ...] }; relationships whose
 *   endpoints aren't both drawn vertices, or are self-loops, are dropped (never
 *   fabricated). Returns {} when the snapshot carries no relationships.
 */
export function snapshotRelationships(snapshot) {
  const nodes = snapshot?.nodes ?? [];
  const idToIdx = new Map();
  nodes.forEach((n, i) => idToIdx.set(n.id, i));
  const byKind = {};
  for (const r of snapshot?.relationships ?? []) {
    const a = idToIdx.get(r.from), b = idToIdx.get(r.to);
    if (a == null || b == null || a === b) continue;
    (byKind[r.kind] ||= []).push([a, b]);
  }
  return byKind;
}

const _xyz = (s) => s.trim().split(/\s+/).map(Number);

/**
 * Parse a HyMeKo-emitted URDF string into links + joints for a 3D render.
 * The engine's URDF is machine-generated and regular, so a regex pass is safe
 * and keeps this dependency-free (works under node --test, no DOMParser).
 * Preconditions: `urdf` is the string from `ir.to_urdf(name)`.
 * Postconditions: { links:[{name,geometry|null,origin[3]}],
 *   joints:[{name,type,parent,child,origin_xyz[3],origin_rpy[3],axis[3]|null}] }.
 *   Links without <visual> get geometry=null (caller draws a placeholder);
 *   missing origin/axis default to zero/null — never throws on a sparse URDF.
 */
export function parseUrdf(urdf) {
  const links = [];
  const linkRe = /<link\s+name="([^"]+)">([\s\S]*?)<\/link>/g;
  let m;
  while ((m = linkRe.exec(urdf)) !== null) {
    const [, name, body] = m;
    const link = { name, geometry: null, origin: [0, 0, 0] };
    const vis = /<visual>([\s\S]*?)<\/visual>/.exec(body);
    if (vis) {
      const o = /<origin\s+xyz="([^"]+)"(?:\s+rpy="([^"]+)")?/.exec(vis[1]);
      if (o) link.origin = _xyz(o[1]);
      const cyl = /<cylinder\s+radius="([^"]+)"\s+length="([^"]+)"/.exec(vis[1]);
      const box = /<box\s+size="([^"]+)"/.exec(vis[1]);
      const sph = /<sphere\s+radius="([^"]+)"/.exec(vis[1]);
      if (cyl) link.geometry = { shape: "cylinder", radius: +cyl[1], length: +cyl[2] };
      else if (box) link.geometry = { shape: "box", size: _xyz(box[1]) };
      else if (sph) link.geometry = { shape: "sphere", radius: +sph[1] };
    }
    links.push(link);
  }
  const joints = [];
  const jointRe = /<joint\s+name="([^"]+)"\s+type="([^"]+)">([\s\S]*?)<\/joint>/g;
  while ((m = jointRe.exec(urdf)) !== null) {
    const [, name, type, body] = m;
    const o = /<origin\s+xyz="([^"]+)"(?:\s+rpy="([^"]+)")?/.exec(body);
    const ax = /<axis\s+xyz="([^"]+)"/.exec(body);
    joints.push({
      name,
      type,
      parent: (/<parent\s+link="([^"]+)"/.exec(body) ?? [])[1] ?? null,
      child: (/<child\s+link="([^"]+)"/.exec(body) ?? [])[1] ?? null,
      origin_xyz: o ? _xyz(o[1]) : [0, 0, 0],
      origin_rpy: o && o[2] ? _xyz(o[2]) : [0, 0, 0],
      axis: ax ? _xyz(ax[1]) : null,
    });
  }
  return { links, joints };
}
