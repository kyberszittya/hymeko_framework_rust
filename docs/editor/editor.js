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

import init, { parse_and_compile } from "./pkg/hymeko_wasm.js";
import { createHypergraphView } from "./views/hypergraph3d.js?v=13";
import { createKinematicView } from "./views/kinematic.js?v=13";
import { highlightHymeko } from "./views/highlight.js?v=13";
await init();

// ── State ────────────────────────────────────────────────────────────
let lastIR = null;        // CompiledIR handle
let lastSnapshot = null;  // parsed snapshot JSON
let cy = null;            // Cytoscape instance
let selected = null;      // {type: 'node'|'edge', name: ...} or null
let showIsa = true;       // graph-view <isa> (meta) edge visibility

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
    lastIR = parse_and_compile(sourceEl.value);
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

  cy.elements().remove();
  cy.add(elements);
  cy.layout({ name: "cose", animate: false, padding: 30 }).run();
  applyIsaVisibility();
}

function applyIsaVisibility() {
  if (cy) cy.elements(".isa").style("display", showIsa ? "element" : "none");
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
const VIEWS = {
  graph:     { view: graphView,             pane: "view-graph" },
  hyper3d:   { view: createHypergraphView(), pane: "view-hyper3d" },
  kinematic: { view: createKinematicView(),  pane: "view-kinematic" },
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

function addLink({ name, mass }) {
  let block = `    ${name}: kit.elements.link {\n`;
  if (mass !== undefined && mass !== "") {
    block += `        mass ${mass};\n`;
  }
  block += `    }\n`;
  insertIntoMainContext(block);
}

function addJoint({ name, kind, parent, child }) {
  // Joint kinds: rev_joint, conti_joint, prismatic_joint, fixed_joint —
  // resolved through the inline `kit.joints` namespace (see EXAMPLE).
  // Convention: edge body is a single arc with `+` parent, `-` child.
  const block = `    @${name}: + <isa> kit.joints.${kind} {\n` +
                `        (+ ${parent}, - ${child});\n` +
                `    }\n`;
  insertIntoMainContext(block);
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
      <button class="danger" id="btnDelete">Delete</button>
    `;
    if (!isEdge) {
      $("btnSetMass").onclick = () => {
        const v = $("selMass").value.trim();
        setMass(decl.name, v);
      };
    }
    $("btnDelete").onclick = () => {
      if (confirm(`Delete ${isEdge ? '@' : ''}${decl.name}?`)) {
        deleteDecl(decl.name, isEdge);
        selected = null;
        renderSelectionPanel();
      }
    };
  } else if (selected.type === "arc") {
    panel.innerHTML = `<p class="hint">Arc-refs are part of an edge's body — select the edge (rounded box) to edit it.</p>`;
  }
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

// Palette buttons
document.querySelectorAll(".palette-btn[data-add]").forEach(btn => {
  btn.onclick = () => {
    const kind = btn.dataset.add;
    if (kind === "link") {
      showModal("Add link", [
        { key: "name", label: "Name",  placeholder: "e.g. shoulder" },
        { key: "mass", label: "Mass",  placeholder: "(optional)" },
      ], (v) => {
        if (!v.name) return;
        addLink({ name: v.name, mass: v.mass });
      });
    } else {
      // Joint kinds need parent+child link names.
      const linkOpts = lastSnapshot
        ? lastSnapshot.nodes.map(n => n.name)
        : [];
      if (linkOpts.length < 1) {
        showError("Add at least one link before adding joints.");
        return;
      }
      showModal(`Add ${kind}`, [
        { key: "name",   label: "Name",   placeholder: "e.g. shoulder_pan" },
        { key: "parent", label: "Parent", type: "select",
          options: linkOpts.map(n => ({ value: n, label: n })) },
        { key: "child",  label: "Child",  type: "select",
          options: linkOpts.map(n => ({ value: n, label: n })) },
      ], (v) => {
        if (!v.name || !v.parent || !v.child) return;
        addJoint({ name: v.name, kind, parent: v.parent, child: v.child });
      });
    }
  };
});

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

// ── Example loader ───────────────────────────────────────────────────

// Self-contained: the kinematics kinds are defined inline as the `kit`
// namespace, so the editor needs no `@"…"` include (the WASM compiles a
// single in-memory source — it cannot resolve file includes in the
// browser). To use the full kinematics stdlib instead, the WASM must carry
// meta_kinematics.hymeko in its MemProvider (a Rust-side change + rebuild).
const EXAMPLE = `hymeko_editor_example {}

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

$("btnExample").onclick = () => {
  sourceEl.value = EXAMPLE;
  recompile();
};

// Load on first paint.
sourceEl.value = EXAMPLE;
recompile();
