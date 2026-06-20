// node --test  — unit tests for the editor example catalog (pure, no DOM/WASM).
//
// Two jobs:
//   1. Catalog integrity (unique non-empty ids, labels, sources; valid views).
//   2. Embed ≡ fixture: every fixture-backed entry's embedded source matches
//      data/typical_graphs/<file>.hymeko byte-for-byte (line-endings normalised,
//      trailing whitespace trimmed). This is the guard that keeps the browser
//      copies from drifting from the canonical, Rust-compiled fixtures.

import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

import { EXAMPLES, exampleById, examplesByGroup, FIXTURE_OF } from "./examples.js";

const HERE = dirname(fileURLToPath(import.meta.url));
// docs/editor/views → repo root is three levels up.
const REPO_ROOT = join(HERE, "..", "..", "..");
const FIXTURE_DIR = join(REPO_ROOT, "data", "typical_graphs");

const norm = (s) => s.replace(/\r\n/g, "\n").trimEnd();

const VALID_VIEWS = new Set([null, "graph", "hyper3d", "kinematic", "sysml"]);

test("catalog ids are unique and non-empty", () => {
  const ids = EXAMPLES.map((e) => e.id);
  assert.equal(new Set(ids).size, ids.length, "duplicate id in catalog");
  for (const id of ids) assert.ok(id && typeof id === "string", `bad id: ${id}`);
});

test("every entry has a label, source, and a valid view", () => {
  for (const e of EXAMPLES) {
    assert.ok(e.label && typeof e.label === "string", `entry ${e.id} missing label`);
    assert.ok(e.source && e.source.trim().length > 0, `entry ${e.id} empty source`);
    assert.ok(VALID_VIEWS.has(e.view), `entry ${e.id} invalid view: ${e.view}`);
  }
});

test("exampleById returns the entry or null", () => {
  assert.equal(exampleById("fano").id, "fano");
  assert.equal(exampleById("kinematic").label, "Kinematic arm");
  assert.equal(exampleById("does-not-exist"), null);
});

test("the four classic hypergraphs are present and jump to the 3D view", () => {
  for (const id of ["fano", "sunflower", "k4_3", "generic"]) {
    const e = exampleById(id);
    assert.ok(e, `missing hypergraph example: ${id}`);
    assert.equal(e.view, "hyper3d", `${id} should switch to hyper3d`);
  }
});

test("the Galambos RL state is present and renders in the 3D view", () => {
  const e = exampleById("galambos");
  assert.ok(e, "missing galambos RL-state example");
  assert.equal(e.view, "hyper3d");
  assert.equal(e.group, "Galambos (RL grasp)");
  // self-contained: no `@\"…\"` imports (the browser cannot fetch them).
  assert.ok(!/@\"/.test(e.source), "galambos source must be self-contained (no @\"…\" import)");
});

test("examplesByGroup buckets every entry into exactly one project group", () => {
  const groups = examplesByGroup();
  const names = groups.map((g) => g.group);
  assert.ok(names.includes("Galambos (RL grasp)"), "Galambos project group present");
  assert.equal(new Set(names).size, names.length, "duplicate group");
  const flat = groups.flatMap((g) => g.entries);
  assert.equal(flat.length, EXAMPLES.length, "every entry in exactly one group");
  const comb = groups.find((g) => g.group === "Combinatorial hypergraphs");
  assert.deepEqual(comb.entries.map((e) => e.id), ["fano", "sunflower", "k4_3", "generic"]);
});

test("embedded hypergraph sources match data/typical_graphs fixtures", () => {
  for (const [id, file] of Object.entries(FIXTURE_OF)) {
    const entry = exampleById(id);
    assert.ok(entry, `FIXTURE_OF references unknown id: ${id}`);
    const onDisk = readFileSync(join(FIXTURE_DIR, file), "utf8");
    assert.equal(
      norm(entry.source),
      norm(onDisk),
      `embedded source for "${id}" diverged from ${file}`,
    );
  }
});
