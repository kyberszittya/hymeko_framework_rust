// node --test — pure arc tuple parsing/formatting.

import { test } from "node:test";
import assert from "node:assert/strict";
import {
  splitTopLevel,
  parseArcRef,
  parseArcTuple,
  formatArcRef,
  formatArcTuple,
  rewriteArcTuple,
} from "./arcs.js";

test("splitTopLevel respects bracket nesting", () => {
  assert.deepEqual(splitTopLevel("a, b, c").map((s) => s.trim()), ["a", "b", "c"]);
  assert.deepEqual(
    splitTopLevel("+ x [[0,0,1],[0,0,0]], - y").map((s) => s.trim()),
    ["+ x [[0,0,1],[0,0,0]]", "- y"],
  );
  assert.deepEqual(splitTopLevel(""), []);
  assert.deepEqual(splitTopLevel("solo").map((s) => s.trim()), ["solo"]);
});

test("parseArcRef extracts sign, target, value", () => {
  assert.deepEqual(parseArcRef("+ base_link"), { sign: "+", target: "base_link", value: "" });
  assert.deepEqual(parseArcRef("~ n0"), { sign: "~", target: "n0", value: "" });
  assert.deepEqual(
    parseArcRef("+ base_link [[0.0, 0.0, 0.2], [0.0, 0.0, 0.0]]"),
    { sign: "+", target: "base_link", value: "[[0.0, 0.0, 0.2], [0.0, 0.0, 0.0]]" },
  );
  assert.deepEqual(parseArcRef("- a.b.c"), { sign: "-", target: "a.b.c", value: "" });
  assert.equal(parseArcRef("no sign here"), null);
});

test("parseArcTuple finds the tuple and its refs", () => {
  const body = ` (+ base_link [[0.0, 0.0, 0.2], [0.0, 0.0, 0.0]], - spinner); `;
  const t = parseArcTuple(body);
  assert.ok(t);
  assert.equal(t.refs.length, 2);
  assert.deepEqual(t.refs[0], {
    sign: "+", target: "base_link", value: "[[0.0, 0.0, 0.2], [0.0, 0.0, 0.0]]",
  });
  assert.deepEqual(t.refs[1], { sign: "-", target: "spinner", value: "" });
  // start/end bracket the parens
  assert.equal(body[t.start], "(");
  assert.equal(body[t.end], ")");
});

test("parseArcTuple returns null when there is no tuple", () => {
  assert.equal(parseArcTuple(" mass 5.0; "), null);
  assert.equal(parseArcTuple(""), null);
});

test("format round-trips a parsed tuple (value-preserving)", () => {
  const body = "(+ a [[1,2,3],[0,0,0]], ~ b, - c)";
  const t = parseArcTuple(body);
  assert.equal(formatArcTuple(t.refs), "(+ a [[1,2,3],[0,0,0]], ~ b, - c)");
});

test("formatArcRef sets a new value and omits empty ones", () => {
  assert.equal(formatArcRef({ sign: "+", target: "x", value: "[1,0,0]" }), "+ x [1,0,0]");
  assert.equal(formatArcRef({ sign: "-", target: "y", value: "" }), "- y");
  assert.equal(formatArcRef({ sign: "~", target: "z" }), "~ z");
});

test("rewriteArcTuple replaces the tuple, preserving surrounding text", () => {
  const body = "\n        (+ base_link, - spinner);\n    ";
  const out = rewriteArcTuple(body, [
    { sign: "+", target: "base_link", value: "[[0,0,0.2],[0,0,0]]" },
    { sign: "-", target: "spinner", value: "" },
  ]);
  assert.equal(out, "\n        (+ base_link [[0,0,0.2],[0,0,0]], - spinner);\n    ");
  // re-parsing yields the new value (round-trip through the editor's path)
  assert.equal(parseArcTuple(out).refs[0].value, "[[0,0,0.2],[0,0,0]]");
});

test("rewriteArcTuple inserts a tuple into a typed/empty edge body", () => {
  const out = rewriteArcTuple("", [{ sign: "+", target: "a", value: "" }, { sign: "-", target: "b", value: "" }]);
  assert.match(out, /\(\+ a, - b\);/);
  assert.ok(parseArcTuple(out));
});

test("rewriteArcTuple leaves a tuple-less body unchanged when refs is empty", () => {
  assert.equal(rewriteArcTuple(" mass 5.0; ", []), " mass 5.0; ");
});
