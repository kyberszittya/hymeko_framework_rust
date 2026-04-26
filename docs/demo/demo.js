// Minimal JS harness for the hymeko_wasm browser demo.
//
// Assumes `wasm-pack build --target web` has been run into ./pkg/, so
// ./pkg/hymeko_wasm.js is importable as an ES module.
//
// All DOM wiring is a thin façade over `parse_and_compile` + methods on
// the returned CompiledIR handle. No framework — vanilla ES modules.

import init, {
  parse_and_compile,
} from "./pkg/hymeko_wasm.js";

// Loaded once on page load; re-used for every compile.
await init();

// -------- state --------
let lastIR = null;

// -------- DOM refs --------
const $ = (id) => document.getElementById(id);
const source     = $("source");
const errorBox   = $("errorBox");
const nodeCount  = $("nodeCount");
const edgeCount  = $("edgeCount");
const arcCount   = $("arcCount");
const predicate  = $("predicate");
const queryCount = $("queryCount");
const queryMatches = $("queryMatches");
const output     = $("output");

// -------- helpers --------
function showError(msg) {
  errorBox.style.display = "block";
  errorBox.textContent = msg;
}
function clearError() {
  errorBox.style.display = "none";
  errorBox.textContent = "";
}
function updateCounts(ir) {
  nodeCount.textContent = ir.node_count;
  edgeCount.textContent = ir.edge_count;
  arcCount.textContent  = ir.arc_count;
}
function requireIR() {
  if (!lastIR) {
    showError("Compile a .hymeko source first.");
    return false;
  }
  return true;
}

// -------- event wiring --------
$("btnCompile").addEventListener("click", () => {
  clearError();
  try {
    lastIR = parse_and_compile(source.value);
    updateCounts(lastIR);
    output.textContent = "Compiled. Use the buttons above to emit / query.";
  } catch (e) {
    lastIR = null;
    showError("Compile failed: " + (e.message || e));
    nodeCount.textContent = edgeCount.textContent = arcCount.textContent = "–";
  }
});

$("btnQuery").addEventListener("click", () => {
  if (!requireIR()) return;
  clearError();
  try {
    const pred = predicate.value;
    const matches = lastIR.query(pred);
    queryCount.textContent = `${matches.length} match` + (matches.length === 1 ? "" : "es");
    queryMatches.style.display = "block";
    queryMatches.textContent = matches.length
      ? matches.join(", ")
      : "(no matches)";
  } catch (e) {
    showError("Query failed: " + (e.message || e));
  }
});

$("btnUrdf").addEventListener("click", () => {
  if (!requireIR()) return;
  output.textContent = lastIR.to_urdf("robot");
});
$("btnSdf").addEventListener("click", () => {
  if (!requireIR()) return;
  output.textContent = lastIR.to_sdf("robot");
});
$("btnDot").addEventListener("click", () => {
  if (!requireIR()) return;
  output.textContent = lastIR.to_dot("robot");
});
$("btnSnapshot").addEventListener("click", () => {
  if (!requireIR()) return;
  try {
    const json = lastIR.snapshot_json();
    // Pretty-print for readability in the demo textarea.
    output.textContent = JSON.stringify(JSON.parse(json), null, 2);
  } catch (e) {
    showError("Snapshot failed: " + (e.message || e));
  }
});

// File picker
$("fileInput").addEventListener("change", (ev) => {
  const f = ev.target.files?.[0];
  if (!f) return;
  const r = new FileReader();
  r.onload = () => { source.value = r.result; };
  r.readAsText(f);
});

// Canonical example loader — pulled at runtime from the repo's
// `examples/paper/hymeko_robot.hymeko` if the demo is served from the
// repo root; falls back to a small inline sample otherwise.
const FALLBACK_EXAMPLE = `// Tiny inline example — two links joined by one revolute joint.

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
`;

$("btnLoadExample").addEventListener("click", async () => {
  clearError();
  try {
    const r = await fetch("../../examples/paper/hymeko_robot.hymeko");
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    source.value = await r.text();
  } catch (_e) {
    source.value = FALLBACK_EXAMPLE;
  }
});
