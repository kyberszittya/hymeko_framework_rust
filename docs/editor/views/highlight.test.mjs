// Units for the .hymeko syntax highlighter.
//   node --test docs/editor/views/highlight.test.mjs
import { test } from "node:test";
import assert from "node:assert/strict";
import { highlightHymeko } from "./highlight.js";

// Strip tags to recover the visible text — must equal the input exactly.
const visible = (html) => html.replace(/<[^>]*>/g, "")
  .replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&amp;/g, "&");

test("keywords, idents, ops are classified", () => {
  const h = highlightHymeko("using kinematics.elements as el;");
  assert.match(h, /<span class="hl-kw">using<\/span>/);
  assert.match(h, /<span class="hl-kw">as<\/span>/);
  assert.match(h, /<span class="hl-ident">kinematics<\/span>/);
  assert.match(h, /<span class="hl-op">\.<\/span>/);
  assert.match(h, /<span class="hl-op">;<\/span>/);
});

test("numbers, strings, tags, comments, and the -> operator", () => {
  assert.match(highlightHymeko("mass 5.0;"), /<span class="hl-number">5\.0<\/span>/);
  assert.match(highlightHymeko('type "camera";'), /<span class="hl-string">&quot;camera&quot;<\/span>|<span class="hl-string">"camera"<\/span>/);
  assert.match(highlightHymeko("x: + <isa> y {}"), /<span class="hl-tag">&lt;isa&gt;<\/span>/);
  assert.match(highlightHymeko("// a note"), /<span class="hl-comment">\/\/ a note<\/span>/);
  assert.match(highlightHymeko("visual -> geo;"), /<span class="hl-op">-&gt;<\/span>/);
});

test("HTML is escaped (no injection) and text round-trips", () => {
  const src = 'a <isa> b // & < >\n"q" 3.14';
  const h = highlightHymeko(src);
  assert.ok(!h.includes("<isa>"), "raw < must be escaped");
  assert.equal(visible(h), src, "visible text must equal the input");
});

test("a keyword-like identifier is not a keyword", () => {
  const h = highlightHymeko("using_thing");
  assert.match(h, /<span class="hl-ident">using_thing<\/span>/);
  assert.doesNotMatch(h, /hl-kw/);
});
