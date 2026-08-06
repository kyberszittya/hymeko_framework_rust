// Live SysML 2 lens — shows the editor's `to_sysml()` output, syntax-highlighted,
// updating on every recompile. The SysML codegen runs in-browser via WASM
// (CompiledIR.to_sysml, SMC #5 Phase 2); this view is the read-only display.
//
// View contract: { name, mount(container), render(snapshot, ir), unmount() }.

// Keywords longest-first so multi-word forms ("part def") win the alternation.
const SYSML_KEYWORDS = [
  "package", "part def", "connection def", "part", "connection",
  "attribute", "import", "def",
];
const KW_RE = new RegExp(`\\b(${SYSML_KEYWORDS.join("|")})\\b`, "g");

const escapeHtml = (s) =>
  s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

/**
 * Highlight SysML 2 source to HTML. Pure (no DOM) so it is unit-testable.
 * Full-line comments are wrapped whole (avoids nesting keyword spans inside a
 * comment); other lines get keyword spans. Preconditions: `text` is a string.
 * Postcondition: returned HTML is escaped except for the inserted `<span>`s.
 */
export function highlightSysml(text) {
  return text
    .split("\n")
    .map((line) => {
      const esc = escapeHtml(line);
      if (/^\s*\/\//.test(line)) {
        return `<span class="sy-comment">${esc}</span>`;
      }
      return esc.replace(KW_RE, '<span class="sy-kw">$1</span>');
    })
    .join("\n");
}

export function createSysmlView() {
  let pre = null;
  let code = null;

  function mount(container) {
    container.innerHTML = "";
    pre = document.createElement("pre");
    pre.className = "sysml-lens";
    code = document.createElement("code");
    pre.appendChild(code);
    container.appendChild(pre);
  }

  function render(_snapshot, ir) {
    if (!code) return;
    if (!ir) {
      code.innerHTML =
        '<span class="sy-comment">// compile a model to see its SysML 2</span>';
      return;
    }
    let text;
    try {
      text = ir.to_sysml("robot");
    } catch (e) {
      text = `// SysML generation error: ${e}`;
    }
    code.innerHTML = highlightSysml(text);
  }

  function unmount() {
    if (pre && pre.parentNode) pre.parentNode.removeChild(pre);
    pre = null;
    code = null;
  }

  return { name: "sysml", mount, render, unmount };
}
