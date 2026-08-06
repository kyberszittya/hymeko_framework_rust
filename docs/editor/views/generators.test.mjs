// node --test — property + perf tests for the pure hypergraph generators.

import { test } from "node:test";
import assert from "node:assert/strict";

import {
  GENERATORS,
  generatorById,
  generateSource,
  specToHymeko,
  stsTriples,
  combinations,
  binom,
} from "./generators.js";

const ID_RE = /^[A-Za-z_]\w*$/;
const STS_NS = [7, 9, 13, 15, 19, 21, 25];

// ── combinatorial helpers ─────────────────────────────────────────────

test("binom matches small known values", () => {
  assert.equal(binom(5, 3), 10);
  assert.equal(binom(6, 3), 20);
  assert.equal(binom(4, 2), 6);
  assert.equal(binom(7, 0), 1);
  assert.equal(binom(3, 5), 0);
});

test("combinations are distinct, sized, and counted right", () => {
  const c = combinations(5, 3);
  assert.equal(c.length, binom(5, 3));
  const keys = new Set(c.map((x) => x.join(",")));
  assert.equal(keys.size, c.length, "duplicate combination");
  for (const x of c) {
    assert.equal(x.length, 3);
    assert.ok(x.every((i, k) => k === 0 || x[k - 1] < i), "not strictly ascending");
  }
});

// ── Steiner triple systems ────────────────────────────────────────────

function assertValidSTS(n, triples) {
  assert.equal(triples.length, (n * (n - 1)) / 6, `S(2,3,${n}) triple count`);
  const seen = new Set();
  for (const tri of triples) {
    assert.equal(tri.length, 3, "block must be a triple");
    assert.equal(new Set(tri).size, 3, "block has a repeated point");
    for (const m of tri) assert.ok(m >= 0 && m < n, `point ${m} out of range`);
    const s = [...tri].sort((a, b) => a - b);
    for (const [x, y] of [[s[0], s[1]], [s[0], s[2]], [s[1], s[2]]]) {
      const k = `${x},${y}`;
      assert.ok(!seen.has(k), `pair ${k} covered more than once`);
      seen.add(k);
    }
  }
  assert.equal(seen.size, (n * (n - 1)) / 2, "not every pair is covered");
}

test("stsTriples yields a valid Steiner triple system for every offered n", () => {
  for (const n of STS_NS) assertValidSTS(n, stsTriples(n));
});

test("stsTriples rejects n not ≡ 1 or 3 (mod 6) or too small", () => {
  for (const bad of [6, 8, 10, 11, 12, 14, 5, 4]) {
    assert.throws(() => stsTriples(bad), /Steiner triple system/, `should reject n=${bad}`);
  }
});

// ── sunflower ─────────────────────────────────────────────────────────

test("sunflower: k edges, arity c+p, pairwise intersection = the core", () => {
  for (const [petals, core, petal] of [[3, 2, 2], [5, 1, 3], [4, 0, 2]]) {
    const spec = generatorById("sunflower").build({ petals, core, petal });
    assert.equal(spec.edges.length, petals);
    for (const e of spec.edges) assert.equal(e.members.length, core + petal);
    const coreSet = new Set(spec.edges[0].members.slice(0, core));
    for (let i = 0; i < spec.edges.length; i++) {
      for (let j = i + 1; j < spec.edges.length; j++) {
        const inter = spec.edges[i].members.filter((m) => spec.edges[j].members.includes(m));
        assert.equal(inter.length, core, "intersection size ≠ core");
        assert.deepEqual(new Set(inter), coreSet, "intersection ≠ the core set");
      }
    }
  }
});

// ── complete r-uniform ────────────────────────────────────────────────

test("complete r-uniform: C(n,r) distinct edges, each arity r", () => {
  for (const [n, r] of [[4, 3], [5, 3], [6, 2], [6, 4]]) {
    const spec = generatorById("complete").build({ n, r });
    assert.equal(spec.edges.length, binom(n, r));
    const keys = new Set();
    for (const e of spec.edges) {
      assert.equal(e.members.length, r);
      keys.add([...e.members].sort((a, b) => a - b).join(","));
    }
    assert.equal(keys.size, spec.edges.length, "duplicate hyperedge");
  }
});

// ── registry + serialiser ─────────────────────────────────────────────

test("every generator has params, validate, build, summary", () => {
  for (const g of GENERATORS) {
    assert.ok(g.id && g.label, `bad generator ${g.id}`);
    assert.ok(Array.isArray(g.params) && g.params.length > 0);
    assert.equal(typeof g.validate, "function");
    assert.equal(typeof g.build, "function");
    assert.equal(typeof g.summary, "function");
  }
});

test("generateSource validates and emits the compile-proven line shape", () => {
  const vtxLine = /^ {4}[A-Za-z_]\w* \{\}$/;
  const edgeLine = /^ {4}@[A-Za-z_]\w*\{ \(~[A-Za-z_]\w*(, ~[A-Za-z_]\w*)*\); \}$/;
  const cases = [
    ["sts", { n: 9 }],
    ["sunflower", { petals: 4, core: 2, petal: 2 }],
    ["complete", { n: 5, r: 3 }],
  ];
  for (const [id, params] of cases) {
    const g = generatorById(id);
    const spec = g.build(params);
    // identifiers
    assert.ok(ID_RE.test(spec.name) && ID_RE.test(spec.block), `${id}: bad header/block`);
    assert.notEqual(spec.name, spec.block, `${id}: header and block collide`);
    assert.equal(new Set(spec.vertices).size, spec.vertices.length, `${id}: dup vertex`);
    for (const v of spec.vertices) assert.ok(ID_RE.test(v), `${id}: bad vertex ${v}`);
    for (const e of spec.edges) assert.ok(ID_RE.test(e.name), `${id}: bad edge ${e.name}`);
    // serialised lines
    const src = specToHymeko(spec);
    const lines = src.split("\n");
    assert.equal(lines[0], `${spec.name}{}`);
    assert.equal(lines[1], spec.block);
    assert.equal(lines[2], "{");
    for (const v of spec.vertices) assert.ok(lines.some((l) => l === `    ${v} {}`));
    for (const ln of lines) {
      if (ln.startsWith("    @")) assert.ok(edgeLine.test(ln), `bad edge line: ${ln}`);
      else if (ln.startsWith("    ") && ln.trim()) assert.ok(vtxLine.test(ln), `bad vtx line: ${ln}`);
    }
    // generateSource == specToHymeko(build) for valid params
    assert.equal(generateSource(id, params), src);
  }
});

test("generateSource throws on invalid params and unknown id", () => {
  assert.throws(() => generateSource("complete", { n: 3, r: 5 }), /2 ≤ r ≤ n/);
  assert.throws(() => generateSource("sunflower", { petals: 99, core: 2, petal: 2 }), /petals/);
  assert.throws(() => generateSource("nope", { n: 9 }), /unknown generator/);
});

// ── performance (gates the offered STS n list) ────────────────────────

test("each offered configuration builds well under budget", () => {
  const BUDGET_MS = 100;
  const median5 = (fn) => {
    const t = [];
    for (let i = 0; i < 5; i++) {
      const s = process.hrtime.bigint();
      fn();
      t.push(Number(process.hrtime.bigint() - s) / 1e6);
    }
    t.sort((a, b) => a - b);
    return t[2];
  };
  const configs = [
    ...STS_NS.map((n) => ["sts", { n }]),
    ["sunflower", { petals: 12, core: 2, petal: 4 }],
    ["complete", { n: 10, r: 3 }],
  ];
  for (const [id, params] of configs) {
    generateSource(id, params); // warm-up + smoke
    const ms = median5(() => generateSource(id, params));
    assert.ok(ms < BUDGET_MS, `${id} ${JSON.stringify(params)}: ${ms.toFixed(2)} ms ≥ ${BUDGET_MS} ms`);
  }
});
