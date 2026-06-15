// "Generate" tab — a data-driven parameter form over the pure generators in
// generators.js. Satisfies the View contract { name, mount, render, unmount };
// `render` is a no-op because this view PRODUCES source rather than consuming a
// snapshot. On "Generate" it calls back into the editor via `loadSource(src)`,
// which sets the source text, recompiles, and switches to the 3D view.
//
// The form is built generically from each generator's `params` descriptor — no
// per-generator UI code (adding a generator in generators.js is enough).
import { GENERATORS, generatorById, generateSource } from "./generators.js?v=21";

const GO_LABEL = "Generate & view in 3D";

export function createGeneratorView(loadSource) {
  let host, genSel, blurbEl, paramBox, summaryEl, errEl, goBtn, progressEl;
  let current = GENERATORS[0];
  let params = {};
  let busy = false;

  function resetParams() {
    params = {};
    for (const p of current.params) params[p.key] = p.default;
  }

  function makeInput(p) {
    let input;
    if (p.type === "select") {
      input = el("select", "gen-input");
      for (const o of p.options) {
        const opt = document.createElement("option");
        opt.value = String(o);
        opt.textContent = String(o);
        input.appendChild(opt);
      }
    } else {
      input = el("input", "gen-input");
      input.type = "number";
      if (p.min != null) input.min = String(p.min);
      if (p.max != null) input.max = String(p.max);
    }
    input.value = String(params[p.key]);
    const onChange = () => { params[p.key] = Number(input.value); update(); };
    input.onchange = onChange;
    input.oninput = onChange;
    return input;
  }

  function rebuildParams() {
    paramBox.innerHTML = "";
    for (const p of current.params) paramBox.append(labelRow(p.label, makeInput(p)));
  }

  // Live validation + summary; the Generate button stays usable but surfaces the
  // error on click too (single source of truth = current.validate).
  function update() {
    blurbEl.textContent = current.blurb || "";
    const err = current.validate(params);
    if (err) {
      errEl.textContent = err; errEl.style.display = "block"; summaryEl.textContent = "";
    } else {
      errEl.style.display = "none"; summaryEl.textContent = "→ " + current.summary(params);
    }
  }

  function setBusy(on) {
    busy = on;
    if (progressEl) progressEl.style.display = on ? "block" : "none";
    if (goBtn) { goBtn.disabled = on; goBtn.textContent = on ? "Generating…" : GO_LABEL; }
  }

  // Building is synchronous, but the larger STS searches (e.g. n = 19, 25) and
  // big complete hypergraphs are perceptible. Flip to the busy state, then defer
  // the actual build one tick so "Generating…" + the bar paint first.
  function onGenerate() {
    if (busy) return;
    errEl.style.display = "none";
    setBusy(true);
    setTimeout(() => {
      try {
        loadSource(generateSource(current.id, params));
      } catch (e) {
        errEl.textContent = e.message || String(e); errEl.style.display = "block";
      } finally {
        setBusy(false);
      }
    }, 0);
  }

  function mount(container) {
    host = container; host.innerHTML = "";
    const wrap = el("div", "gen-form");
    const title = el("h3", "gen-title"); title.textContent = "Generate a hypergraph";

    genSel = el("select", "gen-input");
    for (const g of GENERATORS) {
      const o = document.createElement("option");
      o.value = g.id; o.textContent = g.label;
      genSel.appendChild(o);
    }
    genSel.onchange = () => {
      current = generatorById(genSel.value) ?? GENERATORS[0];
      resetParams(); rebuildParams(); update();
    };

    blurbEl = el("p", "gen-blurb");
    paramBox = el("div", "gen-params");
    summaryEl = el("div", "gen-summary");
    errEl = el("div", "gen-err"); errEl.style.display = "none";
    goBtn = btn(GO_LABEL, onGenerate); goBtn.classList.add("gen-go");
    progressEl = el("div", "gen-progress"); progressEl.style.display = "none";
    progressEl.append(el("div", "gen-bar"));

    wrap.append(title, labelRow("Type", genSel), blurbEl, paramBox, summaryEl, errEl, goBtn, progressEl);
    host.append(wrap);

    current = GENERATORS[0]; genSel.value = current.id;
    busy = false;
    resetParams(); rebuildParams(); update();
  }

  function render() { /* no snapshot input — this view emits source */ }

  function unmount() { busy = false; if (host) host.innerHTML = ""; }

  return { name: "Generate", mount, render, unmount };
}

function el(tag, cls) { const e = document.createElement(tag); if (cls) e.className = cls; return e; }
function btn(label, onclick) { const b = el("button", "gen-btn"); b.textContent = label; b.onclick = onclick; return b; }
function labelRow(text, control) {
  const row = el("div", "gen-row");
  const lab = el("label", "gen-label"); lab.textContent = text;
  row.append(lab, control);
  return row;
}
