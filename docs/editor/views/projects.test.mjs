// node --test — unit tests for the editor multi-file project catalog (pure, no DOM/WASM).
//
// Guards the invariant the editor relies on: every project file's `@"…"` imports are present in
// the project's meta map, so the compile `space` always resolves. The sources are GENERATED
// (scripts/gen_editor_projects.py); re-run the generator if a project .hymeko changes.
import { test } from "node:test";
import assert from "node:assert/strict";

import { PROJECTS, projectById, projectFile, spaceForFile } from "./projects.js";

const VALID_VIEWS = new Set([null, "graph", "hyper3d", "kinematic", "sysml"]);

test("the Galambos MDP project bundles the four instance files", () => {
  const p = projectById("galambos");
  assert.ok(p, "missing galambos project");
  assert.deepEqual(
    p.files.map((f) => f.name),
    ["galambos_planar.hymeko", "galambos_env.hymeko",
     "galambos_task.hymeko", "galambos_strategy.hymeko"],
  );
});

test("every project file's imports resolve in the project meta map", () => {
  for (const p of PROJECTS) {
    assert.ok(p.id && typeof p.id === "string");
    for (const f of p.files) {
      assert.ok(f.source && f.source.trim().length > 0, `${f.name} empty source`);
      assert.ok(VALID_VIEWS.has(f.view), `${f.name} invalid view ${f.view}`);
      for (const imp of f.imports) {
        assert.ok(imp in p.meta, `${p.id}/${f.name} imports "${imp}" missing from meta`);
      }
    }
  }
});

test("spaceForFile returns exactly the file's imported metas", () => {
  const p = projectById("galambos");
  const f = projectFile(p, "galambos_task.hymeko");
  const sp = spaceForFile(p, "galambos_task.hymeko");
  assert.deepEqual(Object.keys(sp).sort(), [...f.imports].sort());
  for (const k of f.imports) assert.equal(sp[k], p.meta[k]);
  assert.deepEqual(spaceForFile(p, "does-not-exist.hymeko"), {});
});

test("a file's source actually declares the imports it lists", () => {
  // the generator parses @"…" from the source, so the listed imports must appear in the text.
  for (const p of PROJECTS) {
    for (const f of p.files) {
      for (const imp of f.imports) {
        assert.ok(f.source.includes(`@"${imp}"`), `${f.name} lists ${imp} but source lacks it`);
      }
    }
  }
});
