// Units for the SysML 2 lens highlighter.
//   node --test docs/editor/views/sysml.test.mjs
import { test } from "node:test";
import assert from "node:assert/strict";
import { highlightSysml } from "./sysml.js";

// Strip tags to recover the visible text — must equal the (unescaped) input.
const visible = (html) =>
  html.replace(/<[^>]*>/g, "")
    .replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&amp;/g, "&");

test("keywords are classified, longest-first", () => {
  const h = highlightSysml("part def Link {");
  assert.match(h, /<span class="sy-kw">part def<\/span>/);
  // "part def" must win over a bare "part" — no standalone part span here.
  assert.doesNotMatch(h, /<span class="sy-kw">part<\/span> def/);
});

test("part / connection / attribute keywords", () => {
  assert.match(highlightSysml("part base : Link;"), /<span class="sy-kw">part<\/span>/);
  assert.match(highlightSysml("connection j1;"), /<span class="sy-kw">connection<\/span>/);
  assert.match(highlightSysml("attribute mass : Real;"), /<span class="sy-kw">attribute<\/span>/);
});

test("full-line comments wrap whole, no nested keyword spans", () => {
  const h = highlightSysml("// a part def comment");
  assert.match(h, /^<span class="sy-comment">.*<\/span>$/);
  assert.doesNotMatch(h, /sy-kw/); // keywords inside a comment are not re-wrapped
});

test("html is escaped and round-trips to the input text", () => {
  const src = "part p : A<B> & C;\n// note: x < y";
  const h = highlightSysml(src);
  assert.ok(h.includes("&lt;B&gt;"));
  assert.equal(visible(h), src);
});
