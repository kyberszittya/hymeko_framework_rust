// Minimal, dependency-free syntax highlighter for the .hymeko DSL. Pure
// (string -> HTML), so it's unit-tested. The editor renders this HTML in a
// <pre> layer behind a transparent <textarea> (the textarea stays the source of
// truth — no editing logic changes).

const ESC = { "&": "&amp;", "<": "&lt;", ">": "&gt;" };
const esc = (s) => s.replace(/[&<>]/g, (c) => ESC[c]);

// Sticky token rules, tried in order at each position.
const RULES = [
  ["comment", /\/\/[^\n]*/y],                 // // line comment
  ["string", /"(?:[^"\\]|\\.)*"/y],           // "string"
  ["tag", /<[^>\n]*>/y],                       // <isa>, <…> annotations
  ["number", /\d+(?:\.\d+)?/y],               // 5, 0.05
  ["kw", /\b(?:using|const|as)\b/y],          // keywords
  ["op", /->|[.:@{}[\]();,+\-]/y],            // -> : @ { } [ ] ( ) ; , + . -
  ["ident", /[A-Za-z_]\w*/y],                  // identifiers
  ["ws", /\s+/y],
];

/**
 * Tokenize `.hymeko` source into highlighted HTML.
 * Preconditions: `src` is a string.
 * Postconditions: HTML-escaped output where recognised tokens are wrapped in
 *   `<span class="hl-KIND">…</span>` (KIND ∈ comment/string/tag/number/kw/op/
 *   ident); whitespace and any unmatched char are escaped but unwrapped. The
 *   visible text content equals the input (so it aligns 1:1 under the textarea).
 */
export function highlightHymeko(src) {
  let out = "";
  let i = 0;
  const n = src.length;
  while (i < n) {
    let matched = false;
    for (const [cls, re] of RULES) {
      re.lastIndex = i;
      const m = re.exec(src);
      if (m && m.index === i && m[0].length > 0) {
        const tok = m[0];
        out += cls === "ws" ? esc(tok) : `<span class="hl-${cls}">${esc(tok)}</span>`;
        i += tok.length;
        matched = true;
        break;
      }
    }
    if (!matched) { out += esc(src[i]); i += 1; }
  }
  return out;
}
