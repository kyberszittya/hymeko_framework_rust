// node --test — profile registry integrity + embed ≡ data/profiles fixtures.

import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

import { PROFILES, profileById, EMBEDDED_FILES, ROOT_NAME } from "./profiles.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const PROFILE_DIR = join(HERE, "..", "..", "..", "data", "profiles");
const norm = (s) => s.replace(/\r\n/g, "\n").trimEnd();

test("profile ids are unique and each has root + palette", () => {
  const ids = PROFILES.map((p) => p.id);
  assert.equal(new Set(ids).size, ids.length, "duplicate profile id");
  for (const p of PROFILES) {
    assert.ok(p.label, `${p.id}: missing label`);
    assert.ok(p.root && p.root.trim().length, `${p.id}: empty root`);
    assert.ok(Array.isArray(p.palette) && p.palette.length, `${p.id}: empty palette`);
    assert.equal(typeof p.files, "object", `${p.id}: files must be an object`);
  }
});

test("ROOT_NAME is a plain .hymeko name", () => {
  assert.match(ROOT_NAME, /^[\w.-]+\.hymeko$/);
});

test("each profile root imports every meta file it ships", () => {
  for (const p of PROFILES) {
    for (const name of Object.keys(p.files)) {
      assert.ok(
        p.root.includes(`@"${name}"`),
        `${p.id}: root does not import ${name}`,
      );
    }
  }
});

test("palette entries are well-formed", () => {
  for (const p of PROFILES) {
    for (const k of p.palette) {
      assert.ok(k.label && k.base, `${p.id}: palette entry missing label/base`);
      assert.equal(typeof k.isEdge, "boolean", `${p.id}: ${k.label} isEdge must be boolean`);
      if (k.isEdge) assert.equal(typeof k.isa, "boolean", `${p.id}: ${k.label} edge needs isa flag`);
    }
  }
});

test("embedded profile files match data/profiles fixtures", () => {
  for (const [name, embedded] of Object.entries(EMBEDDED_FILES)) {
    const onDisk = readFileSync(join(PROFILE_DIR, name), "utf8");
    assert.equal(norm(embedded), norm(onDisk), `embedded ${name} diverged from data/profiles/${name}`);
  }
});

test("kinematics is single-file; hri/sysml import a meta vocabulary", () => {
  assert.deepEqual(profileById("kinematics").files, {});
  assert.ok(Object.keys(profileById("hri").files).includes("meta_hri.hymeko"));
  assert.ok(Object.keys(profileById("sysml_trace").files).includes("meta_sysml_trace.hymeko"));
});
