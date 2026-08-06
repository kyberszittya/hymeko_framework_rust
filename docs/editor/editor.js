// HyMeKo Editor — visual hypergraph design.
//
// Architecture: source-text-as-truth.
//   1. The textarea holds the canonical .hymeko source.
//   2. After every text change, we recompile via WASM and read the snapshot.
//   3. Cytoscape renders the snapshot.
//   4. Edit operations (palette buttons / properties panel) are
//      string transformations on the source text + recompile + redraw.
//
// This keeps the editor in sync with the rest of the toolchain: anything
// the .hymeko text expresses is what gets emitted by URDF/SDF/DOT.

import init, { parse_and_compile, parse_and_compile_files } from "./pkg/hymeko_wasm.js";
import { createHypergraphView } from "./views/hypergraph3d.js?v=26";
import { createKinematicView } from "./views/kinematic.js?v=20";
import { createSysmlView } from "./views/sysml.js?v=19";
import { highlightHymeko } from "./views/highlight.js?v=19";
import { EXAMPLES, exampleById, examplesByGroup } from "./views/examples.js?v=21";
import { createGeneratorView } from "./views/generator_view.js?v=21";
import { PROFILES, profileById, ROOT_NAME } from "./views/profiles.js?v=26";
import { PROJECTS, projectById, projectFile, spaceForFile } from "./views/projects.js?v=1";
import { parseArcTuple, rewriteArcTuple } from "./views/arcs.js?v=23";
import { scopeDepths, bidirectionalEdgeIds } from "./views/adapters.js?v=28";
await init();

// Older bundles lack the multi-file binding; feature-detect so the editor still
// works (single-file) if pkg/ wasn't rebuilt.
const hasMultiFile = typeof parse_and_compile_files === "function";

// ── State ────────────────────────────────────────────────────────────
let lastIR = null;        // CompiledIR handle
let lastSnapshot = null;  // parsed snapshot JSON
let cy = null;            // Cytoscape instance
let selected = null;      // {type: 'node'|'edge', name: ...} or null
let showIsa = true;       // graph-view <isa> (meta) edge visibility
let graphLayout = "cose"; // 2D layout: "cose" (force) | "concentric" (roots centred)
// Compile "space": auxiliary `.hymeko` files the root may @"…"-import (the
// active profile's meta vocabulary). Empty for single-file profiles/examples.
let space = {};
let activeProfile = null;  // current vocabulary profile (drives the palette)

// ── DOM refs ─────────────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);
const sourceEl   = $("source");
const sourceHL   = $("sourceHL");
const errorBox   = $("errorBox");
const nodeCountEl = $("nodeCount");
const edgeCountEl = $("edgeCount");
const arcCountEl  = $("arcCount");

// Repaint the syntax-highlight layer from the textarea, and keep it scroll-synced.
function updateHighlight() {
  if (sourceHL) sourceHL.innerHTML = highlightHymeko(sourceEl.value);
}
if (sourceEl && sourceHL) {
  sourceEl.addEventListener("input", updateHighlight);
  sourceEl.addEventListener("scroll", () => {
    sourceHL.scrollTop = sourceEl.scrollTop;
    sourceHL.scrollLeft = sourceEl.scrollLeft;
  });
}

// ── Cytoscape init ───────────────────────────────────────────────────
cy = cytoscape({
  container: $("cy"),
  style: [
    { selector: "node",
      style: {
        "background-color": "#EEF1F5",
        "border-color": "#6b7280",
        "border-width": 1,
        "label": "data(label)",
        "color": "#1f2937",
        "font-size": 12,
        "text-valign": "center",
        "text-halign": "center",
        "width": "label",
        "height": 30,
        "padding": "8px",
        "shape": "ellipse",
      } },
    { selector: "node.edge-decl",
      style: {
        "background-color": "#D7E4F5",
        "border-color": "#3b82f6",
        "shape": "round-rectangle",
      } },
    { selector: "node:selected",
      style: {
        "border-color": "#dc2626",
        "border-width": 3,
      } },
    { selector: "edge",
      style: {
        "width": 1.6,
        "curve-style": "bezier",
        "target-arrow-shape": "triangle",
        "label": "data(sign)",
        "font-size": 10,
        "color": "#6b7280",
      } },
    { selector: "edge.sign-pos",
      style: { "line-color": "#1b6ca8", "target-arrow-color": "#1b6ca8" } },
    { selector: "edge.sign-neg",
      style: { "line-color": "#b02a2a", "target-arrow-color": "#b02a2a", "target-arrow-shape": "tee" } },
    { selector: "edge.sign-zero",
      style: { "line-color": "#888", "target-arrow-color": "#888" } },
    { selector: "edge:selected",
      style: { "line-color": "#dc2626", "target-arrow-color": "#dc2626", "width": 3 } },
    // Template / inheritance (<isa>) relationships: dashed, hollow arrow.
    { selector: "edge.isa",
      style: {
        "line-color": "#9ca3af",
        "line-style": "dashed",
        "width": 1.2,
        "curve-style": "bezier",
        "target-arrow-shape": "triangle-tee",
        "target-arrow-color": "#9ca3af",
        "target-arrow-fill": "hollow",
        "label": "",
      } },
    // Bidirectional relation: a reciprocal pair (X→Y and Y→X). Arrowheads on
    // BOTH ends, and bezier bows the two members apart so the pair reads as a
    // double connector rather than one overlapped line.
    { selector: "edge.bidir",
      style: {
        "curve-style": "bezier",
        "control-point-step-size": 24,
        "source-arrow-shape": "triangle",
        "source-arrow-color": "#7c3aed",
        "target-arrow-color": "#7c3aed",
        "line-color": "#7c3aed",
        "width": 2,
      } },
  ],
  layout: { name: "cose", animate: false, padding: 30 },
  wheelSensitivity: 0.2,
});

cy.on("tap", (e) => {
  if (e.target === cy) {
    selected = null;
    renderSelectionPanel();
    return;
  }
  if (e.target.isNode()) {
    selected = { type: "node", name: e.target.data("label"), id: e.target.id() };
  } else if (e.target.isEdge()) {
    // Cytoscape edges represent arc-refs; the underlying HyMeKo edge is
    // the parent decl-Edge node with kind="Edge".
    selected = { type: "arc", source: e.target.data("source"), target: e.target.data("target") };
  }
  renderSelectionPanel();
});

// ── Compile + render ─────────────────────────────────────────────────

function clearError() { errorBox.style.display = "none"; errorBox.textContent = ""; }
function showError(msg) {
  errorBox.style.display = "block";
  errorBox.textContent = msg;
}

function recompile() {
  clearError();
  updateHighlight();
  try {
    // Multi-file when the active profile contributes meta files; the live
    // textarea is always the root. Single-file otherwise (examples, generated
    // hypergraphs, single-file profiles).
    if (hasMultiFile && Object.keys(space).length) {
      const files = { [ROOT_NAME]: sourceEl.value, ...space };
      lastIR = parse_and_compile_files(ROOT_NAME, JSON.stringify(files));
    } else {
      lastIR = parse_and_compile(sourceEl.value);
    }
    lastSnapshot = JSON.parse(lastIR.snapshot_json());
    nodeCountEl.textContent = lastIR.node_count;
    edgeCountEl.textContent = lastIR.edge_count;
    arcCountEl.textContent  = lastIR.arc_count;
    renderActiveView();
  } catch (e) {
    lastIR = null;
    lastSnapshot = null;
    showError("Compile failed:\n" + (e.message || e));
    nodeCountEl.textContent = edgeCountEl.textContent = arcCountEl.textContent = "–";
  }
}

function renderGraph() {
  const elements = [];
  if (!lastSnapshot) { cy.elements().remove(); return; }

  // Decl-Nodes (vertices) and decl-Edges (hyperedges) both become
  // Cytoscape "nodes" — the latter styled differently. Arc-refs become
  // Cytoscape edges.
  for (const n of lastSnapshot.nodes) {
    elements.push({
      data: { id: `n${n.id}`, label: n.name, kind: "node",
              bases: n.bases, tags: n.tags },
    });
  }
  for (const e of lastSnapshot.edges) {
    elements.push({
      data: { id: `e${e.id}`, label: e.name, kind: "edge",
              bases: e.bases, tags: e.tags },
      classes: "edge-decl",
    });
  }
  for (const e of lastSnapshot.edges) {
    for (const arc of (e.arcs || [])) {
      const cls = arc.sign === 1 ? "sign-pos"
                : arc.sign === -1 ? "sign-neg"
                : "sign-zero";
      const targetType = lastSnapshot.nodes.find(n => n.id === arc.target_id) ? "n" : "e";
      elements.push({
        data: {
          id: `a${e.id}_${arc.target_id}_${arc.sign}`,
          source: `e${e.id}`,
          target: `${targetType}${arc.target_id}`,
          sign: arc.sign === 1 ? "+" : arc.sign === -1 ? "−" : "~",
        },
        classes: cls,
      });
    }
  }
  // Template / inheritance (<isa>) edges: a dashed line from each decl to the
  // first-level base type it inherits. Bases are resolved names; map them to a
  // drawn decl by name (a base not drawn as a node is simply skipped).
  const nameToCyId = new Map();
  for (const n of lastSnapshot.nodes) nameToCyId.set(n.name, `n${n.id}`);
  for (const e of lastSnapshot.edges) if (!nameToCyId.has(e.name)) nameToCyId.set(e.name, `e${e.id}`);
  const addIsaEdges = (decl, srcId) => {
    for (const base of (decl.bases || [])) {
      const baseId = nameToCyId.get(base);
      if (!baseId || baseId === srcId) continue;
      elements.push({
        data: { id: `isa_${srcId}_${baseId}`, source: srcId, target: baseId },
        classes: "isa",
      });
    }
  };
  for (const n of lastSnapshot.nodes) addIsaEdges(n, `n${n.id}`);
  for (const e of lastSnapshot.edges) addIsaEdges(e, `e${e.id}`);

  // Bidirectional relations: a reciprocal pair (X→Y and Y→X — mutual arc refs or
  // mutual <isa>) is tagged `bidir` so it renders as a double-headed, bowed-apart
  // connector instead of two overlapping one-way arrows.
  const relEdges = elements.filter((el) => el.data.source && el.data.target);
  const bidir = bidirectionalEdgeIds(relEdges.map((el) => el.data));
  for (const el of relEdges) if (bidir.has(el.data.id)) el.classes = `${el.classes || ""} bidir`.trim();

  cy.elements().remove();
  cy.add(elements);
  runGraphLayout();
  applyIsaVisibility();
}

function applyIsaVisibility() {
  if (cy) cy.elements(".isa").style("display", showIsa ? "element" : "none");
}

// Run the selected 2D layout. "concentric" places top-level container decls at
// the centre and nested children on outer rings (composite stacking by scope
// depth); "cose" is the force-directed default.
function runGraphLayout() {
  if (!cy) return;
  if (graphLayout === "concentric") {
    setConcentricLevels();
    cy.layout({
      name: "concentric",
      concentric: (node) => node.data("clevel") ?? 0, // higher → centre
      levelWidth: () => 1,
      minNodeSpacing: 26,
      animate: false,
      padding: 30,
    }).run();
  } else {
    cy.layout({ name: "cose", animate: false, padding: 30 }).run();
  }
}

// Tag each cy node with `clevel = maxDepth - scopeDepth`, so depth-0 roots get
// the largest value and land in the centre of the concentric layout.
function setConcentricLevels() {
  if (!lastSnapshot) return;
  const depth = scopeDepths(lastSnapshot); // declId → depth
  const cyDepth = (id) => {
    const m = /^[ne](\d+)$/.exec(id); // cy ids are `n<declId>` / `e<declId>`
    return m ? (depth.get(Number(m[1])) ?? 0) : 0;
  };
  let maxD = 0;
  cy.nodes().forEach((n) => { const d = cyDepth(n.id()); if (d > maxD) maxD = d; });
  cy.nodes().forEach((n) => n.data("clevel", maxD - cyDepth(n.id())));
}

// ── View tabs: graph (Cytoscape) | hypergraph 3D | kinematic ─────────
// Each view satisfies { name, mount(container), render(snapshot, ir),
// unmount() }. The graph view wraps the persistent Cytoscape instance; the 3D
// views are mounted lazily and unmounted on leave so only one render loop runs.
const graphView = {
  name: "Graph",
  mount() { if (cy) cy.resize(); },
  render() { renderGraph(); if (cy) cy.resize(); },
  unmount() {},
};
// Callback the Generate view uses to push a freshly-built hypergraph into the
// editor: set source text → recompile → jump to the 3D view. `recompile` and
// `showView` are hoisted function declarations, so referencing them here (above
// their definitions) is fine.
function loadGenerated(source) {
  space = {};            // generated hypergraphs are standalone (no imports)
  sourceEl.value = source;
  recompile();
  showView("hyper3d");
}
const VIEWS = {
  graph:     { view: graphView,             pane: "view-graph" },
  hyper3d:   { view: createHypergraphView(), pane: "view-hyper3d" },
  kinematic: { view: createKinematicView(),  pane: "view-kinematic" },
  sysml:     { view: createSysmlView(),       pane: "view-sysml" },
  generate:  { view: createGeneratorView(loadGenerated), pane: "view-generate" },
};
let activeView = "graph";
const mounted = { graph: true }; // Cytoscape created at load
graphView.mount();

function renderActiveView() {
  const entry = VIEWS[activeView];
  if (!mounted[activeView]) { entry.view.mount($(entry.pane)); mounted[activeView] = true; }
  if (lastSnapshot) entry.view.render(lastSnapshot, lastIR);
}

function showView(name) {
  if (!VIEWS[name] || name === activeView) return;
  // Stop / free the 3D views on leave (graph stays alive — cy is reused).
  if (activeView !== "graph") { VIEWS[activeView].view.unmount(); mounted[activeView] = false; }
  Object.values(VIEWS).forEach(({ pane }) => $(pane).classList.remove("active"));
  document.querySelectorAll(".view-tab").forEach(
    (b) => b.classList.toggle("active", b.dataset.view === name));
  $(VIEWS[name].pane).classList.add("active");
  activeView = name;
  renderActiveView();
}

document.querySelectorAll(".view-tab").forEach((b) => {
  b.onclick = () => showView(b.dataset.view);
});

const isaToggle = $("toggleIsa");
if (isaToggle) isaToggle.onchange = () => { showIsa = isaToggle.checked; applyIsaVisibility(); };

const layoutSel = $("graphLayout");
if (layoutSel) layoutSel.onchange = () => { graphLayout = layoutSel.value; runGraphLayout(); };

// ── Floating source panel: collapse + drag (by the header) ──────────
const sourcePanel = $("sourcePanel");
const sourceHead = $("sourceHead");
const sourceToggle = $("sourceToggle");
if (sourceToggle && sourcePanel) {
  sourceToggle.onclick = (e) => {
    e.stopPropagation();
    const collapsed = sourcePanel.classList.toggle("collapsed");
    sourceToggle.textContent = collapsed ? "▸" : "▾";
  };
}
if (sourceHead && sourcePanel) {
  let dragging = false, sx = 0, sy = 0, ox = 0, oy = 0;
  sourceHead.addEventListener("pointerdown", (e) => {
    if (e.target === sourceToggle) return;
    const pr = sourcePanel.parentElement.getBoundingClientRect();
    const r = sourcePanel.getBoundingClientRect();
    ox = r.left - pr.left; oy = r.top - pr.top; sx = e.clientX; sy = e.clientY;
    sourcePanel.style.left = ox + "px"; sourcePanel.style.top = oy + "px"; sourcePanel.style.right = "auto";
    dragging = true; sourceHead.setPointerCapture(e.pointerId);
  });
  sourceHead.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    const pr = sourcePanel.parentElement.getBoundingClientRect();
    const nx = Math.max(0, Math.min(ox + (e.clientX - sx), pr.width - 60));
    const ny = Math.max(0, Math.min(oy + (e.clientY - sy), pr.height - 30));
    sourcePanel.style.left = nx + "px"; sourcePanel.style.top = ny + "px";
  });
  const stop = (e) => { dragging = false; try { sourceHead.releasePointerCapture(e.pointerId); } catch { /* ignore */ } };
  sourceHead.addEventListener("pointerup", stop);
  sourceHead.addEventListener("pointercancel", stop);
}

// ── Source-text mutations ────────────────────────────────────────────
//
// The .hymeko grammar lets us be lazy here: we insert into the body of
// whichever description block holds the most context (the LAST `}` in
// the file). For an MVP this works on every example we ship; round-
// tripping arbitrary user code would need a proper parser-based
// rewriter.

function insertIntoMainContext(block) {
  const src = sourceEl.value;
  // Find the LAST `}` that closes a top-level body and insert before it.
  const lastBrace = src.lastIndexOf("}");
  if (lastBrace < 0) {
    showError("Cannot insert: no closing `}` in source.");
    return;
  }
  const before = src.slice(0, lastBrace);
  const after  = src.slice(lastBrace);
  // Trim trailing whitespace/newlines from `before`, then insert.
  sourceEl.value = before.trimEnd() + "\n" + block + after;
  recompile();
}

// Profile-driven "add" operations. A palette kind is
//   node: { isEdge:false, base, hasMass? }  →  `name: base { [mass …;] }`
//   edge: { isEdge:true, base, isa }         →  `@name[: + <isa>] base { (+P,-C); }`
// (the `isa` form matches the kit joints; the bare form matches the SysML
// trace edges). `base` carries the profile's namespace alias.
function addNodeDecl(kind, name, mass) {
  let block = `    ${name}: ${kind.base} {`;
  if (kind.hasMass && mass !== undefined && mass !== "") {
    block += `\n        mass ${mass};\n    }\n`;
  } else {
    block += `}\n`;
  }
  insertIntoMainContext(block);
}

function addEdgeDecl(kind, name, parent, child) {
  const head = kind.isa ? `@${name}: + <isa> ${kind.base}` : `@${name}: ${kind.base}`;
  insertIntoMainContext(`    ${head} {\n        (+ ${parent}, - ${child});\n    }\n`);
}

// Escape a string for embedding in a regex.
function reEscape(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

// Find the position of the `{` opening this decl's body, or -1 if none.
function findDeclOpenBrace(src, declName, isEdge) {
  const prefix = isEdge ? `@${declName}` : declName;
  const re = new RegExp(`(^|\\n)\\s*${reEscape(prefix)}\\s*:`, "");
  const m = src.match(re);
  if (!m) return { start: -1, brace: -1 };
  const declStart = m.index + (m[0].startsWith("\n") ? 1 : 0);
  const brace = src.indexOf("{", declStart);
  return { start: declStart, brace };
}

// Given a `{` position in src, find the position of its matching `}`,
// counting nested braces. Returns -1 on unbalanced.
function findMatchingClose(src, openIdx) {
  let depth = 1;
  for (let i = openIdx + 1; i < src.length; i++) {
    const c = src[i];
    if (c === "{") depth++;
    else if (c === "}") {
      depth--;
      if (depth === 0) return i;
    }
  }
  return -1;
}

function deleteDecl(name, isEdge) {
  const src = sourceEl.value;
  const { start, brace } = findDeclOpenBrace(src, name, isEdge);
  if (start < 0) {
    showError(`Could not locate declaration ${name}.`);
    return;
  }
  let end;
  if (brace < 0) {
    // No body — declaration ends at `;`.
    end = src.indexOf(";", start);
    if (end < 0) { showError("Malformed declaration (no `;`)"); return; }
    end += 1;
  } else {
    const close = findMatchingClose(src, brace);
    if (close < 0) { showError("Unbalanced `{` in declaration."); return; }
    end = close + 1;
  }
  // Eat trailing newline.
  if (src[end] === "\n") end += 1;
  sourceEl.value = src.slice(0, start) + src.slice(end);
  recompile();
}

function setMass(linkName, mass) {
  const src = sourceEl.value;
  const { start, brace } = findDeclOpenBrace(src, linkName, false);
  if (start < 0 || brace < 0) {
    showError(`Could not locate ${linkName}'s body.`);
    return;
  }
  const close = findMatchingClose(src, brace);
  if (close < 0) { showError("Unbalanced `{` in body."); return; }

  let body = src.slice(brace + 1, close);
  const massRe = /^(\s*)mass\s+[^;]+;\s*\n?/m;

  if (massRe.test(body)) {
    body = mass !== ""
      ? body.replace(massRe, (_m, ws) => `${ws}mass ${mass};\n`)
      : body.replace(massRe, "");
  } else if (mass !== "") {
    // Insert at top of body, indented one level deeper than the `{`.
    body = `\n        mass ${mass};` + body;
  }

  sourceEl.value = src.slice(0, brace + 1) + body + src.slice(close);
  recompile();
}

// ── Arc-ref editing (edge bodies) ─────────────────────────────────────
// Read the arc-refs of an edge decl by parsing its body from the source
// (source-as-truth). Returns { brace, close, body, tuple } or null.
function readEdgeArcs(edgeName) {
  const src = sourceEl.value;
  const { brace } = findDeclOpenBrace(src, edgeName, true);
  if (brace < 0) return null;
  const close = findMatchingClose(src, brace);
  if (close < 0) return null;
  const body = src.slice(brace + 1, close);
  return { brace, close, body, tuple: parseArcTuple(body) };
}

// Rewrite an edge's arc tuple from edited refs (replacing the existing tuple, or
// inserting one if the edge had none). `refs` is [{sign,target,value}].
function applyArcs(edgeName, refs) {
  const src = sourceEl.value;
  const loc = readEdgeArcs(edgeName);
  if (!loc) { showError(`Could not locate ${edgeName}'s body.`); return; }
  const { brace, close, body } = loc;
  const newBody = rewriteArcTuple(body, refs);
  if (newBody === body) return; // nothing to do
  sourceEl.value = src.slice(0, brace + 1) + newBody + src.slice(close);
  recompile();
}

// ── Modal helpers ────────────────────────────────────────────────────

function showModal(title, fields, onOk) {
  const m = $("modal");
  $("modalTitle").textContent = title;
  const body = $("modalBody");
  body.innerHTML = "";
  const inputs = {};
  for (const f of fields) {
    const row = document.createElement("div");
    row.className = "row";
    const label = document.createElement("label");
    label.textContent = f.label + ":";
    let input;
    if (f.type === "select") {
      input = document.createElement("select");
      for (const opt of f.options) {
        const o = document.createElement("option");
        o.value = opt.value || opt;
        o.textContent = opt.label || opt;
        input.appendChild(o);
      }
    } else {
      input = document.createElement("input");
      input.type = f.type || "text";
      input.value = f.value ?? "";
      input.placeholder = f.placeholder || "";
    }
    inputs[f.key] = input;
    row.appendChild(label);
    row.appendChild(input);
    body.appendChild(row);
  }
  m.style.display = "flex";

  const cleanup = () => {
    m.style.display = "none";
    $("modalOk").onclick = null;
    $("modalCancel").onclick = null;
  };
  $("modalOk").onclick = () => {
    const values = {};
    for (const k in inputs) values[k] = inputs[k].value.trim();
    cleanup();
    onOk(values);
  };
  $("modalCancel").onclick = cleanup;
}

// ── Selection / properties panel ─────────────────────────────────────

function renderSelectionPanel() {
  const panel = $("selectionPanel");
  if (!selected) {
    panel.innerHTML = `<p class="hint">Click a node or edge on the canvas.</p>`;
    return;
  }
  if (selected.type === "node") {
    const decl = lastSnapshot.nodes.find(n => `n${n.id}` === selected.id) ||
                 lastSnapshot.edges.find(n => `e${n.id}` === selected.id);
    if (!decl) {
      panel.innerHTML = `<p class="hint">Selection lost on redraw.</p>`;
      return;
    }
    const isEdge = lastSnapshot.edges.some(n => `e${n.id}` === selected.id);
    panel.innerHTML = `
      <div class="row"><label>Name</label><input id="selName" value="${decl.name}" disabled /></div>
      <div class="row"><label>Kind</label><input value="${decl.kind}${decl.bases.length ? ' ('+decl.bases.join(',')+')' : ''}" disabled /></div>
      ${ !isEdge ? `<div class="row"><label>Mass</label><input id="selMass" placeholder="(none)" /></div>
         <button id="btnSetMass" class="palette-btn" style="margin-top:6px;">Set mass</button>` : ""}
      ${ isEdge ? `<div id="arcEditor" class="arc-editor"></div>` : ""}
      <button class="danger" id="btnDelete">Delete</button>
    `;
    if (!isEdge) {
      $("btnSetMass").onclick = () => {
        const v = $("selMass").value.trim();
        setMass(decl.name, v);
      };
    } else {
      renderArcEditor($("arcEditor"), decl.name);
    }
    $("btnDelete").onclick = () => {
      if (confirm(`Delete ${isEdge ? '@' : ''}${decl.name}?`)) {
        deleteDecl(decl.name, isEdge);
        selected = null;
        renderSelectionPanel();
      }
    };
  } else if (selected.type === "arc") {
    // Clicking an arc line edits the arcs of the edge it belongs to.
    const edge = lastSnapshot?.edges.find((e) => `e${e.id}` === selected.source);
    if (edge) { selected = { type: "node", id: `e${edge.id}` }; renderSelectionPanel(); return; }
    panel.innerHTML = `<p class="hint">Select the edge (rounded box) to edit its arcs.</p>`;
  }
}

// Inline arc-ref editor for a selected edge: one row per ref (sign · target ·
// value), plus add / remove / apply. "value" is the arc payload (e.g. a joint
// origin transform `[[x,y,z],[r,p,y]]`) — previously not editable at all.
function renderArcEditor(host, edgeName) {
  if (!host) return;
  const loc = readEdgeArcs(edgeName);
  let refs = loc?.tuple ? loc.tuple.refs.map((r) => ({ ...r })) : [];
  const nodeNames = lastSnapshot ? lastSnapshot.nodes.map((n) => n.name) : [];

  const mk = (tag, cls) => { const e = document.createElement(tag); if (cls) e.className = cls; return e; };
  const mkBtn = (label, onclick, cls) => { const b = mk("button", cls); b.textContent = label; b.onclick = onclick; return b; };

  const collect = () => {
    refs = [...host.querySelectorAll(".arc-row")].map((row) => ({
      sign: row.querySelector(".arc-sign").value,
      target: row.querySelector(".arc-target").value,
      value: row.querySelector(".arc-value").value.trim(),
    }));
  };

  const arcRow = (r, i) => {
    const row = mk("div", "arc-row");
    const sign = mk("select", "arc-sign");
    for (const s of ["+", "-", "~"]) { const o = mk("option"); o.value = s; o.textContent = s; sign.appendChild(o); }
    sign.value = r.sign;
    const tgt = mk("select", "arc-target");
    const opts = nodeNames.includes(r.target) || !r.target ? nodeNames : [r.target, ...nodeNames];
    for (const n of opts) { const o = mk("option"); o.value = n; o.textContent = n; tgt.appendChild(o); }
    if (r.target) tgt.value = r.target;
    const val = mk("input", "arc-value");
    val.placeholder = "value (e.g. [[0,0,0.2],[0,0,0]])";
    val.value = r.value ?? "";
    const rm = mkBtn("×", () => { collect(); refs.splice(i, 1); draw(); }, "arc-rm");
    row.append(sign, tgt, val, rm);
    return row;
  };

  function draw() {
    host.innerHTML = "";
    const head = mk("div", "arc-head"); head.textContent = "Arc-refs"; host.append(head);
    if (!refs.length) { const p = mk("p", "hint"); p.textContent = "No arc-refs yet."; host.append(p); }
    refs.forEach((r, i) => host.append(arcRow(r, i)));
    const add = mkBtn("+ arc-ref", () => {
      collect(); refs.push({ sign: "+", target: nodeNames[0] ?? "", value: "" }); draw();
    }, "arc-add");
    const apply = mkBtn("Apply arcs", () => { collect(); applyArcs(edgeName, refs); }, "palette-btn");
    apply.style.marginTop = "6px";
    host.append(add, apply);
  }

  draw();
}

// ── Toolbar wiring ───────────────────────────────────────────────────

$("btnRedraw").onclick = recompile;

$("btnExport").onclick = () => {
  if (!lastIR) return;
  const fmt = $("exportFormat").value;
  let content, ext;
  switch (fmt) {
    case "hymeko": content = sourceEl.value; ext = "hymeko"; break;
    case "urdf":   content = lastIR.to_urdf("robot"); ext = "urdf"; break;
    case "sdf":    content = lastIR.to_sdf("robot"); ext = "sdf"; break;
    case "dot":    content = lastIR.to_dot("robot"); ext = "dot"; break;
    case "sysml":  content = lastIR.to_sysml("robot"); ext = "sysml"; break;
    default: return;
  }
  const blob = new Blob([content], { type: "text/plain" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = `hymeko_export.${ext}`;
  a.click();
  URL.revokeObjectURL(url);
};

$("btnLoad").onclick = () => $("fileInput").click();
$("fileInput").onchange = (e) => {
  const f = e.target.files?.[0];
  if (!f) return;
  const r = new FileReader();
  r.onload = () => { sourceEl.value = r.result; recompile(); };
  r.readAsText(f);
};

// Palette — rebuilt from the active profile's kinds (data-driven; no per-kind
// markup). A node kind prompts name (+ mass); an edge kind prompts name + the
// two signed endpoints, picked from the current nodes.
function rebuildPalette() {
  const box = $("paletteAdd");
  if (!box) return;
  box.innerHTML = "";
  for (const kind of activeProfile?.palette ?? []) {
    const b = document.createElement("button");
    b.className = "palette-btn";
    b.textContent = kind.label;
    b.onclick = () => addKind(kind);
    box.appendChild(b);
  }
}

function addKind(kind) {
  const title = "Add " + kind.label.replace(/^\+\s*/, "");
  if (!kind.isEdge) {
    const fields = [{ key: "name", label: "Name", placeholder: "e.g. part1" }];
    if (kind.hasMass) fields.push({ key: "mass", label: "Mass", placeholder: "(optional)" });
    showModal(title, fields, (v) => { if (v.name) addNodeDecl(kind, v.name, v.mass); });
    return;
  }
  const nodeOpts = lastSnapshot ? lastSnapshot.nodes.map((n) => n.name) : [];
  if (nodeOpts.length < 2) { showError("Add at least two nodes before adding an edge."); return; }
  showModal(title, [
    { key: "name", label: "Name", placeholder: "e.g. e1" },
    { key: "parent", label: "+ (from)", type: "select", options: nodeOpts.map((n) => ({ value: n, label: n })) },
    { key: "child", label: "− (to)", type: "select", options: nodeOpts.map((n) => ({ value: n, label: n })) },
  ], (v) => { if (v.name && v.parent && v.child) addEdgeDecl(kind, v.name, v.parent, v.child); });
}

// Query
$("btnQuery").onclick = () => {
  if (!lastIR) return;
  try {
    const matches = lastIR.query($("predicate").value);
    $("queryResult").textContent = matches.length
      ? `${matches.length} match: ` + matches.join(", ")
      : "(no matches)";
  } catch (e) {
    $("queryResult").textContent = "Query failed: " + (e.message || e);
  }
};

// Live recompile when source changes.
let recompileTimer = null;
sourceEl.oninput = () => {
  clearTimeout(recompileTimer);
  recompileTimer = setTimeout(recompile, 400);
};

// ── Example gallery ──────────────────────────────────────────────────
// The example sources live in views/examples.js (the data-driven catalog).
// They are embedded there because the editor is served from docs/editor/,
// so the repo's data/ tree is not fetch-reachable. The dropdown is built
// from the catalog; picking an entry loads its source, recompiles, and
// (for the classic hypergraphs) jumps to the 3D view.
const exampleSelect = $("exampleSelect");
if (exampleSelect) {
  for (const { group, entries } of examplesByGroup()) {
    const og = document.createElement("optgroup");
    og.label = group;
    for (const ex of entries) {
      const o = document.createElement("option");
      o.value = ex.id;
      o.textContent = ex.label;
      og.appendChild(o);
    }
    exampleSelect.appendChild(og);
  }
  exampleSelect.onchange = () => {
    const ex = exampleById(exampleSelect.value);
    exampleSelect.value = ""; // reset to the placeholder so re-picking re-fires
    if (!ex) return;
    space = {};               // examples are standalone single-file sources
    sourceEl.value = ex.source;
    recompile();
    if (ex.view) showView(ex.view);
  };
}

// ── Project tree (multi-file MDP) ─────────────────────────────────────
// A project (views/projects.js) is several editable instance files sharing a meta vocabulary,
// rendered in the left palette as a collapsible tree: the project header toggles its file list;
// clicking a file loads it as the compile root with its `@"…"`-imported metas in the `space`.
const projectTreeEl = $("projectTree");
let activeProjectFileBtn = null;
function loadProjectFile(project, fileName, fileBtn) {
  const f = projectFile(project, fileName);
  if (!f) return;
  space = spaceForFile(project, fileName);   // the metas THIS file imports
  sourceEl.value = f.source;
  recompile();
  if (f.view) showView(f.view);
  if (activeProjectFileBtn) activeProjectFileBtn.classList.remove("active");
  if (fileBtn) {
    fileBtn.classList.add("active");
    activeProjectFileBtn = fileBtn;
  }
}
function buildProjectTree() {
  if (!projectTreeEl) return;
  projectTreeEl.replaceChildren();
  for (const p of PROJECTS) {
    const node = document.createElement("div");
    node.className = "proj-node";
    node.dataset.project = p.id;
    const header = document.createElement("button");
    header.className = "proj-header";
    header.setAttribute("aria-expanded", "true");
    header.textContent = p.label;
    header.onclick = () => {
      const open = header.getAttribute("aria-expanded") === "true";
      header.setAttribute("aria-expanded", String(!open));
      node.classList.toggle("collapsed", open);
    };
    const list = document.createElement("ul");
    list.className = "proj-files";
    for (const f of p.files) {
      const li = document.createElement("li");
      const btn = document.createElement("button");
      btn.className = "proj-file";
      btn.textContent = f.label;
      btn.title = f.name;
      btn.onclick = () => loadProjectFile(p, f.name, btn);
      li.appendChild(btn);
      list.appendChild(li);
    }
    node.append(header, list);
    projectTreeEl.appendChild(node);
  }
}
// Open a project's first file (used by the ?project= deep-link), highlighting it in the tree.
function openProject(id) {
  const p = projectById(id);
  if (!p || !p.files.length) return;
  const firstBtn = projectTreeEl?.querySelector(`[data-project="${id}"] .proj-file`);
  loadProjectFile(p, p.files[0].name, firstBtn);
}
buildProjectTree();

// Left palette tabs: "Project" (outline tree) / "Tools" (element palette, stats, query).
const paletteTabs = [...document.querySelectorAll(".palette-tab")];
function showPaletteTab(name) {
  for (const t of paletteTabs) t.classList.toggle("active", t.dataset.tab === name);
  for (const pane of document.querySelectorAll(".palette-tabpane"))
    pane.classList.toggle("active", pane.id === `tab-${name}`);
}
for (const tab of paletteTabs) tab.onclick = () => showPaletteTab(tab.dataset.tab);

// ── Profile picker ───────────────────────────────────────────────────
// A profile is a vocabulary: it loads its meta file(s) into the compile space
// (so meta elements live outside the current context, imported by the root)
// and rebinds the palette to that vocabulary's kinds.
function setProfile(id) {
  const p = profileById(id);
  if (!p) return;
  activeProfile = p;
  space = { ...p.files };
  sourceEl.value = p.root;
  rebuildPalette();
  recompile();
}

const profileSelect = $("profileSelect");
if (profileSelect) {
  for (const p of PROFILES) {
    const o = document.createElement("option");
    o.value = p.id;
    o.textContent = p.label;
    profileSelect.appendChild(o);
  }
  profileSelect.onchange = () => setProfile(profileSelect.value);
}

// First paint: the kinematics (robot) profile — same default model as before,
// now selectable rather than hard-wired.
if (profileSelect) profileSelect.value = "kinematics";
setProfile("kinematics");

// Deep-link: ?profile=<id> selects a vocabulary profile, ?view=<name> a view.
const _params = new URLSearchParams(location.search);
const _initProfile = _params.get("profile");
if (_initProfile && profileById(_initProfile)) {
  if (profileSelect) profileSelect.value = _initProfile;
  setProfile(_initProfile);
}
// ?project=<id> loads a multi-file project (its first file), overriding the default profile.
const _initProject = _params.get("project");
if (_initProject && projectById(_initProject)) openProject(_initProject);
// ?tab=<outline|tools> selects which left-palette tab is shown on load.
const _initTab = _params.get("tab");
if (_initTab) showPaletteTab(_initTab);
const _initView = _params.get("view");
if (_initView && VIEWS[_initView]) showView(_initView);
// ?layout=concentric|cose selects the 2D graph layout on load.
const _initLayout = _params.get("layout");
if (_initLayout === "concentric" || _initLayout === "cose") {
  graphLayout = _initLayout;
  if (layoutSel) layoutSel.value = _initLayout;
  runGraphLayout();
}
// ?select=<decl name> opens that node/edge in the properties panel (an edge
// opens its arc-ref editor) — a shareable deep-link to a selection.
const _initSelect = _params.get("select");
if (_initSelect && lastSnapshot) {
  const e = lastSnapshot.edges.find((x) => x.name === _initSelect);
  const n = lastSnapshot.nodes.find((x) => x.name === _initSelect);
  if (e) selected = { type: "node", id: `e${e.id}` };
  else if (n) selected = { type: "node", id: `n${n.id}` };
  if (selected) renderSelectionPanel();
}
