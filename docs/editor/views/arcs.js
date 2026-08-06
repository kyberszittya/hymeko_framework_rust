// Pure parsing/formatting of an edge's arc tuple, e.g.
//   (+ base_link [[0.0, 0.0, 0.2], [0.0, 0.0, 0.0]], - spinner, ~ axis_z)
// An arc-ref is  <sign> <target> [<value>]  where sign ∈ {+,-,~}, target is an
// identifier (possibly dotted), and value is an optional payload expression (the
// joint origin/axis transform, a list, etc.). No DOM — runs under `node --test`
// (arcs.test.mjs). The editor uses these to read + rewrite arc values as a
// string transformation on the source (source-as-truth).

/**
 * Split `s` on top-level `sep`, ignoring separators nested inside () [] {}.
 * Preconditions: `s` is a string. Postconditions: returns [] for blank input,
 * otherwise the pieces (not trimmed) in order.
 */
export function splitTopLevel(s, sep = ",") {
  if (s.trim() === "") return [];
  const out = [];
  let depth = 0, cur = "";
  for (const ch of s) {
    if (ch === "(" || ch === "[" || ch === "{") depth++;
    else if (ch === ")" || ch === "]" || ch === "}") depth--;
    if (ch === sep && depth === 0) { out.push(cur); cur = ""; }
    else cur += ch;
  }
  out.push(cur);
  return out;
}

/**
 * Parse a single arc-ref. Returns { sign, target, value } or null if it does not
 * match `<sign> <target> [value]`. `value` is "" when absent.
 */
export function parseArcRef(text) {
  const m = text.trim().match(/^([+\-~])\s*([A-Za-z_]\w*(?:\.\w+)*)\s*([\s\S]*)$/);
  if (!m) return null;
  return { sign: m[1], target: m[2], value: m[3].trim() };
}

/**
 * Locate the first `(...)` arc tuple in an edge body and parse its refs.
 * Postconditions: returns { start, end, inner, refs } with `start`/`end` the
 * indices of `(` and `)` in `body`, or null if there is no balanced tuple.
 */
export function parseArcTuple(body) {
  const start = body.indexOf("(");
  if (start < 0) return null;
  let depth = 0, end = -1;
  for (let i = start; i < body.length; i++) {
    const c = body[i];
    if (c === "(") depth++;
    else if (c === ")") { depth--; if (depth === 0) { end = i; break; } }
  }
  if (end < 0) return null;
  const inner = body.slice(start + 1, end);
  const refs = splitTopLevel(inner, ",").map(parseArcRef).filter(Boolean);
  return { start, end, inner, refs };
}

/** Format one arc-ref back to source text. */
export function formatArcRef(r) {
  const v = (r.value ?? "").trim();
  return `${r.sign} ${r.target}${v ? " " + v : ""}`;
}

/** Format a list of arc-refs as a `(…)` tuple. */
export function formatArcTuple(refs) {
  return `(${refs.map(formatArcRef).join(", ")})`;
}

/**
 * Return a new edge body with its arc tuple rewritten from `refs`.
 * - If the body already has a `(...)` tuple, it is replaced in place
 *   (surrounding text, including the trailing `;`, is preserved).
 * - Otherwise a fresh `(refs);` statement is appended (for typed/empty edges).
 * - Empty `refs` with no existing tuple leaves the body unchanged.
 */
export function rewriteArcTuple(body, refs) {
  const tuple = parseArcTuple(body);
  const tupleText = formatArcTuple(refs);
  if (tuple) return body.slice(0, tuple.start) + tupleText + body.slice(tuple.end + 1);
  if (refs.length) return `${body.replace(/\s*$/, "")}\n        ${tupleText};\n    `;
  return body;
}
