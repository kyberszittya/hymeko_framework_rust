// Hypergraph 3D view — force-directed star / clique / prism renderer (three.js
// r128, global THREE from the CDN script in index.html). Reimplements the
// look of demo_web/star_expansion_viewer.html as a clean View module. Adds:
//   • three modes: star (hub per edge), clique (all-pairs), prism (translucent
//     polytope over each hyperedge's members);
//   • node labels (sprites, toggle);
//   • attribute colouring (by first base / first tag) with a legend.
// Fed by snapshotToHyperedges(snapshot); pure math in geometry3d.js.
//
// View contract: { name, mount(container), render(snapshot, ir), unmount() }.
import { snapshotToHyperedges, snapshotRelationships } from "./adapters.js?v=13";
import { attributeValue, categoricalColorMap, prismPositions, treePositions, coneTreePositions } from "./geometry3d.js?v=13";

const HUB_POS = 0x38bdf8, HUB_NEG = 0xf472b6; // hyperedge sign colours
const LINE_COLOR = 0x4a6ba0;
const MODES = ["star", "clique", "prism"];
const COLOR_BYS = ["base", "tag", "uniform"];
const LAYOUTS = ["force", "tree", "cone"]; // force-directed | flat composition tree | 3D barycentric cone
const TREE_SRCS = ["scope", "isa", "ref"]; // which relationship drives the tree / nesting
const NEST_COLORS = [0x60a5fa, 0x34d399, 0xfbbf24, 0xf472b6, 0xc084fc, 0xf87171]; // nested-box hue by depth
// Named, annotated relationship layers from snapshot.relationships. Membership
// (signed arcs) is the primary hyperedge structure; these are the secondary
// structural relations, each a toggleable colour-coded layer.
const REL_KINDS = [
  { key: "isa", color: 0xf59e0b, label: "isa" },     // template / inheritance
  { key: "ref", color: 0xc084fc, label: "ref" },     // field reference (a -> b)
  { key: "scope", color: 0x34d399, label: "scope" }, // containment / nesting
];

export function createHypergraphView() {
  let host, canvas, stats, legend, relLegend, modeBtn, colorBtn, labelBtn;
  let THREE, scene, camera, renderer, raf = 0;
  let mode = "star", colorBy = "base", showLabels = false, light = true, layout = "force";
  let treeSrc = "scope", showNest = false;
  let relOn = { isa: true, ref: true, scope: false };
  let nestBoxes = []; // { desc:[vertexIdx], mesh } translucent containment volumes
  let hyper = { n_vertices: 0, vertices: [], hyperedges: [] };
  let relSegs = {};            // kind -> [[fromIdx,toIdx], ...]
  let H = { n: 0, edges: [], signs: [] };
  let P = [], Vel = [], meshes = [], hubMeshes = [], prismMeshes = [], sprites = [];
  let relSegMeshes = {};       // kind -> THREE.LineSegments
  let lineSeg = null, vGeo = null, hGeo = null, colorMap = new Map();
  const BG_DARK = 0x12101f, BG_LIGHT = 0xf7f7f8;
  let theta = 0.7, phi = 1.05, radius = 340, autoRot = true;
  let drag = false, lx = 0, ly = 0, onUp = null, onMove = null, moved = false;
  let raycaster = null, pointer = null, tooltip = null, selBox = null, selectedMesh = null;

  function mount(container) {
    host = container; host.innerHTML = "";
    const tb = el("div", "view3d-toolbar");
    modeBtn = btn("Mode: Star", () => { mode = MODES[(MODES.indexOf(mode) + 1) % MODES.length]; modeBtn.textContent = "Mode: " + cap(mode); rebuild(); });
    colorBtn = btn("Color: base", () => { colorBy = COLOR_BYS[(COLOR_BYS.indexOf(colorBy) + 1) % COLOR_BYS.length]; colorBtn.textContent = "Color: " + colorBy; rebuild(); });
    labelBtn = btn("Labels: off", () => { showLabels = !showLabels; labelBtn.textContent = "Labels: " + (showLabels ? "on" : "off"); labelBtn.classList.toggle("off", !showLabels); rebuild(); });
    const relBtns = REL_KINDS.map((rk) => {
      const b = btn(`${rk.label}: ${relOn[rk.key] ? "on" : "off"}`, () => {
        relOn[rk.key] = !relOn[rk.key];
        b.textContent = `${rk.label}: ${relOn[rk.key] ? "on" : "off"}`;
        b.classList.toggle("off", !relOn[rk.key]);
        rebuild();
      });
      if (!relOn[rk.key]) b.classList.add("off");
      return b;
    });
    const layoutBtn = btn("Layout: force", () => {
      layout = LAYOUTS[(LAYOUTS.indexOf(layout) + 1) % LAYOUTS.length];
      layoutBtn.textContent = "Layout: " + layout;
      if (layout === "tree") { autoRot = false; theta = Math.PI / 2; phi = Math.PI / 2; } // flat tree, front-on
      else if (layout === "cone") { autoRot = true; theta = 0.7; phi = 1.05; }            // 3D cone, orbit
      rebuild();
    });
    const treeBtn = btn("Tree: scope", () => {
      treeSrc = TREE_SRCS[(TREE_SRCS.indexOf(treeSrc) + 1) % TREE_SRCS.length];
      treeBtn.textContent = "Tree: " + treeSrc; // drives tree/cone layout + nesting
      rebuild();
    });
    const nestBtn = btn("Nest: off", () => {
      showNest = !showNest;
      nestBtn.textContent = "Nest: " + (showNest ? "on" : "off");
      nestBtn.classList.toggle("off", !showNest);
      rebuild();
    });
    nestBtn.classList.add("off");
    const bgBtn = btn("BG", () => { light = !light; if (scene) scene.background = new THREE.Color(light ? BG_LIGHT : BG_DARK); });
    const rotBtn = btn("⟳ spin", () => { autoRot = !autoRot; rotBtn.classList.toggle("off", !autoRot); });
    tb.append(modeBtn, colorBtn, labelBtn, ...relBtns, layoutBtn, treeBtn, nestBtn, bgBtn, rotBtn);
    stats = el("div", "view3d-stats");
    legend = el("div", "view3d-legend");
    tooltip = el("div", "view3d-tip");
    selBox = el("div", "view3d-sel");
    relLegend = el("div", "view3d-rel");
    canvas = el("canvas", "view3d-canvas");
    host.append(tb, canvas, stats, legend, relLegend, tooltip, selBox);

    THREE = globalThis.THREE;
    if (!THREE) { stats.textContent = "three.js failed to load (offline?)"; return; }
    scene = new THREE.Scene(); scene.background = new THREE.Color(light ? BG_LIGHT : BG_DARK);
    camera = new THREE.PerspectiveCamera(55, 1, 1, 4000);
    renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    scene.add(new THREE.AmbientLight(0xffffff, 0.8));
    const dir = new THREE.DirectionalLight(0xffffff, 0.7); dir.position.set(1, 1, 1); scene.add(dir);
    raycaster = new THREE.Raycaster(); pointer = new THREE.Vector2();
    bindPointer();
    bindPicking();
    loop();
  }

  function render(snapshot) {
    if (!THREE) return;
    hyper = snapshotToHyperedges(snapshot || {});
    relSegs = snapshotRelationships(snapshot || {});
    H = { n: hyper.n_vertices, edges: hyper.hyperedges.map((e) => e.members), signs: hyper.hyperedges.map((e) => e.sign) };
    const total = H.n + H.edges.length;
    P = []; Vel = [];
    for (let i = 0; i < total; i++) {
      const u = Math.random() * 2 - 1, a = Math.random() * Math.PI * 2, r = 120 * Math.cbrt(Math.random());
      const s = Math.sqrt(1 - u * u);
      P.push(new THREE.Vector3(r * s * Math.cos(a), r * s * Math.sin(a), r * u));
      Vel.push(new THREE.Vector3());
    }
    rebuild();
  }

  function currentSegments() {
    const segs = [];
    if (mode === "clique") {
      H.edges.forEach((e) => { for (let i = 0; i < e.length; i++) for (let j = i + 1; j < e.length; j++) segs.push([e[i], e[j]]); });
    } else { // star (and arity-2 connectors reused by prism mode)
      H.edges.forEach((e, ei) => { const hub = H.n + ei; e.forEach((v) => segs.push([hub, v])); });
    }
    return segs;
  }

  function clearObjects() {
    for (const m of meshes) { scene.remove(m); m.material.dispose(); }
    for (const m of hubMeshes) { scene.remove(m); m.material.dispose(); }
    for (const m of prismMeshes) { scene.remove(m); m.geometry.dispose(); m.material.dispose(); }
    for (const s of sprites) { scene.remove(s); if (s.material.map) s.material.map.dispose(); s.material.dispose(); }
    if (vGeo) { vGeo.dispose(); vGeo = null; }
    if (hGeo) { hGeo.dispose(); hGeo = null; }
    if (lineSeg) { scene.remove(lineSeg); lineSeg.geometry.dispose(); lineSeg.material.dispose(); lineSeg = null; }
    for (const k in relSegMeshes) { const s = relSegMeshes[k]; scene.remove(s); s.geometry.dispose(); s.material.dispose(); }
    for (const nb of nestBoxes) { scene.remove(nb.mesh); nb.mesh.geometry.dispose(); nb.mesh.material.dispose(); }
    relSegMeshes = {}; nestBoxes = [];
    meshes = []; hubMeshes = []; prismMeshes = []; sprites = [];
  }

  function rebuild() {
    if (!THREE || !scene) return;
    clearObjects();
    if (layout === "tree") applyTreeLayout();            // flat composition tree
    else if (layout === "cone") applyConeLayout();       // 3D barycentric cone tree
    else if (mode === "prism") for (let i = 0; i < 160; i++) step(); // settle prisms
    colorMap = categoricalColorMap(hyper.vertices.map((v) => attributeValue(v, colorBy)));

    selectedMesh = null; if (selBox) selBox.style.display = "none";
    vGeo = new THREE.SphereGeometry(5, 16, 16);
    for (let i = 0; i < H.n; i++) {
      const col = colorMap.get(attributeValue(hyper.vertices[i], colorBy)) ?? 0x9fd6ff;
      const m = new THREE.Mesh(vGeo, new THREE.MeshLambertMaterial({ color: col }));
      m.userData = { kind: "vertex", i }; scene.add(m); meshes.push(m);
    }
    hGeo = new THREE.SphereGeometry(3.4, 12, 12);
    const showHubs = mode === "star";
    H.edges.forEach((_e, ei) => {
      const m = new THREE.Mesh(hGeo, new THREE.MeshLambertMaterial({ color: H.signs[ei] < 0 ? HUB_NEG : HUB_POS }));
      m.userData = { kind: "edge", ei }; m.visible = showHubs; scene.add(m); hubMeshes.push(m);
    });

    if (mode === "prism") buildPrisms(); else rebuildLines();
    buildRelLayers();
    if (showLabels) buildLabels();
    sync();
    if (showNest) { buildNestBoxes(); updateNestBoxes(); }
    updateLegendAndStats();
  }

  function rebuildLines() {
    const segs = currentSegments();
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(segs.length * 6), 3));
    lineSeg = new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ color: LINE_COLOR, transparent: true, opacity: 0.5 }));
    lineSeg._segs = segs; scene.add(lineSeg);
  }

  function buildPrisms() {
    // Arity ≥3 → translucent prism over the (settled) member positions; arity 2
    // → a thin connector line. Static (the layout is frozen in prism mode).
    const lines = [];
    H.edges.forEach((members, ei) => {
      if (members.length < 3) { if (members.length === 2) lines.push([members[0], members[1]]); return; }
      // Expand the cross-section out from its centroid so the prism visibly
      // encloses the member spheres, and extrude thicker — "bigger" prisms.
      const c = members.reduce((a, v) => [a[0] + P[v].x, a[1] + P[v].y, a[2] + P[v].z], [0, 0, 0]).map((x) => x / members.length);
      const pts = members.map((v) => [
        c[0] + (P[v].x - c[0]) * 1.28, c[1] + (P[v].y - c[1]) * 1.28, c[2] + (P[v].z - c[2]) * 1.28,
      ]);
      const pr = prismPositions(pts, 34);
      if (!pr) return;
      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(pr.positions), 3));
      geo.computeVertexNormals();
      const col = H.signs[ei] < 0 ? HUB_NEG : HUB_POS;
      const mesh = new THREE.Mesh(geo, new THREE.MeshLambertMaterial({ color: col, transparent: true, opacity: 0.24, side: THREE.DoubleSide }));
      mesh.userData = { kind: "edge", ei }; scene.add(mesh); prismMeshes.push(mesh);
    });
    if (lines.length) {
      const geo = new THREE.BufferGeometry();
      const arr = new Float32Array(lines.length * 6);
      lines.forEach(([a, b], k) => { arr.set([P[a].x, P[a].y, P[a].z, P[b].x, P[b].y, P[b].z], k * 6); });
      geo.setAttribute("position", new THREE.BufferAttribute(arr, 3));
      lineSeg = new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ color: LINE_COLOR, transparent: true, opacity: 0.6 }));
      lineSeg._segs = lines; scene.add(lineSeg);
    }
  }

  function buildLabels() {
    for (let i = 0; i < H.n; i++) sprites.push(makeLabel(hyper.vertices[i].label, i));
    for (const s of sprites) scene.add(s);
  }

  // Named relationship layers (isa, scope, …) from snapshot.relationships. Each
  // enabled kind with edges becomes one dashed colour-coded LineSegments layer;
  // it also feeds the force layout (step) so related nodes cluster instead of
  // drifting apart in the floating space.
  // The tree-source relationship is always drawn in the composition layouts.
  const showRel = (key) => relOn[key] || (key === treeSrc && layout !== "force");

  // Composition-tree layout: place vertices by the scope (containment) tree —
  // children stacked under their parent, parent at the barycentre of its
  // children. Static; hubs sit at their members' centroid.
  function applyTreeLayout() {
    const { x, depth, width } = treePositions(H.n, relSegs[treeSrc] || []);
    const SX = 72, SY = 110, x0 = width / 2;
    let maxD = 0;
    for (let i = 0; i < H.n; i++) if (depth[i] > maxD) maxD = depth[i];
    const yMid = (maxD * SY) / 2; // centre the tree vertically on the origin
    for (let i = 0; i < H.n; i++) P[i].set((x[i] - x0) * SX, -depth[i] * SY + yMid, 0);
    H.edges.forEach((members, ei) => {
      let cx = 0, cy = 0, cz = 0;
      for (const v of members) { cx += P[v].x; cy += P[v].y; cz += P[v].z; }
      const k = members.length || 1;
      P[H.n + ei].set(cx / k, cy / k, cz / k);
    });
    radius = Math.max(220, Math.max(width * SX, (maxD + 1) * SY) * 0.72); // fit
  }

  // 3D barycentric cone tree: parents float at the centroid of their children,
  // leaves spread on depth rings (deeper = wider). Static; orbit to view.
  function applyConeLayout() {
    const { X, Y, Z, depth } = coneTreePositions(H.n, relSegs[treeSrc] || []);
    let maxD = 0;
    for (let i = 0; i < H.n; i++) if (depth[i] > maxD) maxD = depth[i];
    const yMid = (maxD * 120) / 2; // centre vertically (matches levelHeight)
    for (let i = 0; i < H.n; i++) P[i].set(X[i], Y[i] + yMid, Z[i]);
    H.edges.forEach((members, ei) => {
      let cx = 0, cy = 0, cz = 0;
      for (const v of members) { cx += P[v].x; cy += P[v].y; cz += P[v].z; }
      const k = members.length || 1;
      P[H.n + ei].set(cx / k, cy / k, cz / k);
    });
    radius = Math.max(260, Math.max(maxD * 90 * 2, maxD * 120) * 0.85); // fit ring + height
  }

  // ── Transparent nested-container notation ─────────────────────────
  // Each internal node of the tree-source hierarchy gets a translucent box that
  // encloses its whole subtree — so containment reads as nested volumes. Boxes
  // are unit cubes transformed per frame (cheap) from the live node positions.
  function subtreeTree() {
    const parent = new Int32Array(H.n).fill(-1);
    const children = Array.from({ length: H.n }, () => []);
    for (const [c, p] of relSegs[treeSrc] || []) {
      if (c < H.n && p < H.n && c !== p && parent[c] === -1) { parent[c] = p; children[p].push(c); }
    }
    const desc = new Array(H.n); const depth = new Int32Array(H.n); const seen = new Uint8Array(H.n);
    const collect = (u, d) => {
      if (seen[u]) return desc[u] || [u];
      seen[u] = 1; depth[u] = d;
      let list = [u];
      for (const c of children[u]) list = list.concat(collect(c, d + 1));
      desc[u] = list; return list;
    };
    for (let i = 0; i < H.n; i++) if (parent[i] === -1) collect(i, 0);
    for (let i = 0; i < H.n; i++) if (!seen[i]) { seen[i] = 1; desc[i] = [i]; }
    return { children, desc, depth };
  }

  function buildNestBoxes() {
    const { children, desc, depth } = subtreeTree();
    for (let u = 0; u < H.n; u++) {
      if (children[u].length === 0) continue; // only nodes that contain others
      const col = NEST_COLORS[depth[u] % NEST_COLORS.length];
      const mat = new THREE.MeshLambertMaterial({
        color: col, transparent: true, opacity: 0.11, side: THREE.DoubleSide, depthWrite: false,
      });
      const mesh = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), mat);
      mesh.renderOrder = -1; scene.add(mesh);
      nestBoxes.push({ desc: desc[u], mesh });
    }
  }

  function updateNestBoxes() {
    const pad = 16;
    for (const nb of nestBoxes) {
      let nx = Infinity, ny = Infinity, nz = Infinity, xx = -Infinity, xy = -Infinity, xz = -Infinity;
      for (const v of nb.desc) {
        const p = P[v];
        if (p.x < nx) nx = p.x; if (p.x > xx) xx = p.x;
        if (p.y < ny) ny = p.y; if (p.y > xy) xy = p.y;
        if (p.z < nz) nz = p.z; if (p.z > xz) xz = p.z;
      }
      nb.mesh.position.set((nx + xx) / 2, (ny + xy) / 2, (nz + xz) / 2);
      nb.mesh.scale.set((xx - nx) + pad * 2, (xy - ny) + pad * 2, (xz - nz) + pad * 2);
    }
  }

  function buildRelLayers() {
    for (const rk of REL_KINDS) {
      if (!showRel(rk.key)) continue;
      const segs = relSegs[rk.key];
      if (!segs || !segs.length) continue;
      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(segs.length * 6), 3));
      const mat = new THREE.LineDashedMaterial({ color: rk.color, dashSize: 7, gapSize: 5, transparent: true, opacity: 0.85 });
      const seg = new THREE.LineSegments(geo, mat);
      seg._segs = segs; scene.add(seg); relSegMeshes[rk.key] = seg;
    }
  }

  function makeLabel(text, idx) {
    const cv = document.createElement("canvas"); const ctx = cv.getContext("2d");
    const fs = 44; ctx.font = `${fs}px Segoe UI`;
    cv.width = Math.ceil(ctx.measureText(text).width) + 18; cv.height = fs + 14;
    ctx.font = `${fs}px Segoe UI`;
    ctx.fillStyle = "rgba(12,14,26,0.66)"; roundRect(ctx, 0, 0, cv.width, cv.height, 10); ctx.fill();
    ctx.fillStyle = "#e8eefc"; ctx.textBaseline = "middle"; ctx.fillText(text, 9, cv.height / 2 + 1);
    const tex = new THREE.CanvasTexture(cv);
    const spr = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false }));
    spr.scale.set(cv.width * 0.16, cv.height * 0.16, 1);
    spr._idx = idx; return spr;
  }

  function step() {
    const total = P.length; if (!total) return;
    for (let i = 0; i < total; i++) for (let j = i + 1; j < total; j++) {
      const d = P[i].clone().sub(P[j]); const dist = d.length() || 0.01;
      d.multiplyScalar(1400 / (dist * dist) / dist); Vel[i].add(d); Vel[j].sub(d);
    }
    for (const [a, b] of currentSegments()) {
      const d = P[b].clone().sub(P[a]); const dist = d.length() || 0.01;
      d.multiplyScalar((dist - 60) * 0.02 / dist); Vel[a].add(d); Vel[b].sub(d);
    }
    // Enabled relationship layers also pull related nodes together (so the
    // structure is visible, not scattered in the floating space).
    for (const rk of REL_KINDS) {
      if (!relOn[rk.key]) continue;
      for (const [a, b] of relSegs[rk.key] || []) {
        const d = P[b].clone().sub(P[a]); const dist = d.length() || 0.01;
        d.multiplyScalar((dist - 64) * 0.02 / dist); Vel[a].add(d); Vel[b].sub(d);
      }
    }
    for (let i = 0; i < total; i++) { Vel[i].add(P[i].clone().multiplyScalar(-0.002)); Vel[i].multiplyScalar(0.82); P[i].add(Vel[i]); }
  }

  function sync() {
    for (let i = 0; i < meshes.length; i++) meshes[i].position.copy(P[i]);
    for (let i = 0; i < hubMeshes.length; i++) hubMeshes[i].position.copy(P[H.n + i]);
    for (const s of sprites) s.position.copy(P[s._idx]).add(new THREE.Vector3(0, 9, 0));
    if (lineSeg && lineSeg._segs && mode !== "prism") {
      const attr = lineSeg.geometry.getAttribute("position");
      lineSeg._segs.forEach(([a, b], k) => { attr.setXYZ(k * 2, P[a].x, P[a].y, P[a].z); attr.setXYZ(k * 2 + 1, P[b].x, P[b].y, P[b].z); });
      attr.needsUpdate = true;
    }
    for (const k in relSegMeshes) {
      const s = relSegMeshes[k]; const attr = s.geometry.getAttribute("position");
      s._segs.forEach(([a, b], idx) => { attr.setXYZ(idx * 2, P[a].x, P[a].y, P[a].z); attr.setXYZ(idx * 2 + 1, P[b].x, P[b].y, P[b].z); });
      attr.needsUpdate = true;
      s.computeLineDistances(); // dashed material needs distances after move
    }
    if (showNest && nestBoxes.length) updateNestBoxes();
  }

  function updateLegendAndStats() {
    const nSeg = mode === "prism" ? prismMeshes.length : currentSegments().length;
    const unit = mode === "prism" ? "prisms" : "edges";
    stats.textContent = `${H.n} vertices · ${H.edges.length} hyperedges · ${mode} (${nSeg} ${unit})`;
    legend.innerHTML = "";
    if (colorBy !== "uniform") for (const [val, col] of colorMap) legend.append(legendRow(col, val));

    // Relationship-type legend — the named/annotated relationships currently
    // drawn, grouped. (Membership sign from arcs; isa from bases.)
    relLegend.innerHTML = "";
    const rows = [];
    if (hyper.hyperedges.some((e) => e.sign >= 0)) rows.push([0x38bdf8, "— membership (+)"]);
    if (hyper.hyperedges.some((e) => e.sign < 0)) rows.push([0xf472b6, "— membership (−)"]);
    for (const rk of REL_KINDS) if (showRel(rk.key) && (relSegs[rk.key] || []).length) rows.push([rk.color, "⇠ " + rk.label]);
    if (rows.length) { relLegend.append(legendHead("relationships")); for (const [c, t] of rows) relLegend.append(legendRow(c, t)); }
  }

  function legendRow(col, text) {
    const row = el("div", "legend-row");
    const sw = el("span", "legend-sw"); sw.style.background = "#" + col.toString(16).padStart(6, "0");
    const tx = el("span", "legend-tx"); tx.textContent = text;
    row.append(sw, tx); return row;
  }
  function legendHead(text) { const h = el("div", "legend-head"); h.textContent = text; return h; }

  function loop() {
    raf = requestAnimationFrame(loop);
    resize();
    if (P.length && mode !== "prism" && layout === "force") { step(); sync(); }
    if (autoRot) theta += 0.0016;
    camera.position.set(radius * Math.sin(phi) * Math.cos(theta), radius * Math.cos(phi), radius * Math.sin(phi) * Math.sin(theta));
    camera.lookAt(0, 0, 0);
    renderer.render(scene, camera);
  }

  function resize() {
    const r = host.getBoundingClientRect(); const dpr = window.devicePixelRatio || 1;
    const w = r.width || 1, h = r.height || 1;
    if (canvas.width !== Math.floor(w * dpr) || canvas.height !== Math.floor(h * dpr)) {
      renderer.setPixelRatio(dpr); renderer.setSize(w, h, false);
      camera.aspect = w / h; camera.updateProjectionMatrix();
    }
  }

  function bindPointer() {
    canvas.addEventListener("pointerdown", (e) => { drag = true; moved = false; lx = e.clientX; ly = e.clientY; });
    canvas.addEventListener("wheel", (e) => { e.preventDefault(); radius = Math.max(120, Math.min(900, radius + e.deltaY * 0.4)); }, { passive: false });
    onUp = () => { drag = false; };
    onMove = (e) => { if (!drag) return; moved = true; theta += (e.clientX - lx) * 0.01; autoRot = false; phi = Math.max(0.15, Math.min(3.0, phi - (e.clientY - ly) * 0.008)); lx = e.clientX; ly = e.clientY; };
    window.addEventListener("pointerup", onUp); window.addEventListener("pointermove", onMove);
  }

  // ── Hover tooltip + click selection (raycasting) ──────────────────
  function bindPicking() {
    canvas.addEventListener("pointermove", (e) => {
      if (drag) { tooltip.style.display = "none"; return; }
      const hit = pick(e.clientX, e.clientY);
      if (!hit) { tooltip.style.display = "none"; return; }
      const info = infoFor(hit.object);
      tooltip.innerHTML = `<b>${esc(info.title)}</b>` + info.lines.map((l) => `<div>${esc(l)}</div>`).join("");
      const r = canvas.getBoundingClientRect();
      tooltip.style.left = (e.clientX - r.left + 14) + "px";
      tooltip.style.top = (e.clientY - r.top + 12) + "px";
      tooltip.style.display = "block";
    });
    canvas.addEventListener("pointerleave", () => { tooltip.style.display = "none"; });
    canvas.addEventListener("click", (e) => {
      if (moved) return; // it was an orbit drag, not a click
      const hit = pick(e.clientX, e.clientY);
      selectMesh(hit ? hit.object : null);
    });
  }

  function pickables() {
    return meshes.concat(hubMeshes.filter((m) => m.visible), prismMeshes);
  }

  function pick(clientX, clientY) {
    if (!raycaster) return null;
    const r = canvas.getBoundingClientRect();
    pointer.set(((clientX - r.left) / r.width) * 2 - 1, -((clientY - r.top) / r.height) * 2 + 1);
    raycaster.setFromCamera(pointer, camera);
    return raycaster.intersectObjects(pickables(), false)[0] || null;
  }

  function infoFor(obj) {
    const u = obj.userData;
    if (u.kind === "vertex") {
      const v = hyper.vertices[u.i];
      return { title: v.label, lines: [
        v.bases.length ? "isa: " + v.bases.join(", ") : null,
        v.tags.length ? "tags: " + v.tags.join(", ") : null,
      ].filter(Boolean) };
    }
    const e = hyper.hyperedges[u.ei];
    return { title: e.label, lines: [
      `hyperedge · sign ${e.sign > 0 ? "+" : "−"} · arity ${e.arity}`,
      "members: " + e.members.map((m) => hyper.vertices[m].label).join(", "),
    ] };
  }

  function selectMesh(obj) {
    if (selectedMesh && selectedMesh.material) selectedMesh.material.emissive?.setHex(0x000000);
    selectedMesh = obj;
    if (!obj) { selBox.style.display = "none"; return; }
    obj.material.emissive?.setHex(0x555555);
    const info = infoFor(obj);
    selBox.innerHTML = `<b>${esc(info.title)}</b>` + info.lines.map((l) => `<div>${esc(l)}</div>`).join("");
    selBox.style.display = "block";
  }

  function unmount() {
    if (raf) cancelAnimationFrame(raf); raf = 0;
    if (onUp) window.removeEventListener("pointerup", onUp);
    if (onMove) window.removeEventListener("pointermove", onMove);
    onUp = onMove = null;
    if (scene) clearObjects();
    if (renderer) { renderer.forceContextLoss?.(); renderer.dispose(); }
    if (host) host.innerHTML = "";
    scene = null; renderer = null; P = []; Vel = [];
  }

  return { name: "Hypergraph 3D", mount, render, unmount };
}

function el(tag, cls) { const e = document.createElement(tag); if (cls) e.className = cls; return e; }
function btn(label, onclick) { const b = el("button", "view3d-btn"); b.textContent = label; b.onclick = onclick; return b; }
function esc(s) { return String(s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c])); }
function cap(s) { return s.charAt(0).toUpperCase() + s.slice(1); }
function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath(); ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
}
