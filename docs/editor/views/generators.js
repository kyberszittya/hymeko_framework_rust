// Parametric hypergraph generators — PURE (no DOM, no globals, no three.js), so
// this runs unchanged under `node --test` (see generators.test.mjs). The DOM
// form that drives it lives in generator_view.js.
//
// Each generator builds a HypergraphSpec
//   { name, block, vertices: string[], edges: [{ name, members: number[] }] }
// (members index into `vertices`); `specToHymeko` serialises it to the same
// `.hymeko` shape as data/typical_graphs/*.hymeko — bare `v {}` vertices and
// neutral-arc hyperedges `@e{ (~a, ~b, ~c); }` — a form already proven to
// compile by the `typical_graph_examples` Rust test (Fano = S(2,3,7)).

// ── Combinatorial helpers ─────────────────────────────────────────────

/** C(n, r) as an exact integer. */
export function binom(n, r) {
  if (r < 0 || r > n) return 0;
  r = Math.min(r, n - r);
  let c = 1;
  for (let i = 0; i < r; i++) c = (c * (n - i)) / (i + 1);
  return Math.round(c);
}

/** All r-subsets of {0,…,n-1} as ascending index arrays, in lexicographic order. */
export function combinations(n, r) {
  if (r === 0) return [[]];
  if (r > n) return [];
  const out = [];
  const idx = Array.from({ length: r }, (_, i) => i);
  for (;;) {
    out.push(idx.slice());
    let i = r - 1;
    while (i >= 0 && idx[i] === n - r + i) i--;
    if (i < 0) break;
    idx[i]++;
    for (let j = i + 1; j < r; j++) idx[j] = idx[j - 1] + 1;
  }
  return out;
}

// ── Steiner triple systems S(2,3,n) ───────────────────────────────────
// Exists iff n ≡ 1 or 3 (mod 6). Closed-form Bose construction for n ≡ 3
// (no search); deterministic backtracking for n ≡ 1. Both are validated by the
// "every pair exactly once" property test.

// Bose construction (n = 3v, v odd): points are Z_v × {0,1,2}; the idempotent
// commutative quasigroup a∘b = ((v+1)/2)(a+b) mod v ties the layers together.
function stsBose(n) {
  const v = n / 3;
  const half = (v + 1) / 2; // multiplicative inverse of 2 (mod v), v odd
  const op = (a, b) => (half * ((a + b) % v)) % v;
  const pt = (layer, i) => layer * v + i;
  const triples = [];
  for (let i = 0; i < v; i++) triples.push([pt(0, i), pt(1, i), pt(2, i)]);
  for (let i = 0; i < v; i++) {
    for (let j = i + 1; j < v; j++) {
      const m = op(i, j);
      for (let L = 0; L < 3; L++) triples.push([pt(L, i), pt(L, j), pt((L + 1) % 3, m)]);
    }
  }
  return triples;
}

// Backtracking with most-constrained-pair selection + forward checking
// (deterministic; STS always exists for valid n, so this always succeeds). MRV
// — always extend the uncovered pair with the fewest legal completions, bailing
// immediately on a pair with zero — keeps the search effectively backtrack-free
// even at n = 25. The step cap turns any pathological case into a clean error
// rather than a hung tab.
function stsBacktrack(n) {
  const used = new Array(n * n).fill(false); // used[min*n+max] for a covered pair
  const key = (a, b) => (a < b ? a * n + b : b * n + a);
  const triples = [];
  let steps = 0;
  const CAP = 2_000_000;

  const completions = (a, b) => {
    const list = [];
    for (let c = 0; c < n; c++) {
      if (c === a || c === b) continue;
      if (!used[key(a, c)] && !used[key(b, c)]) list.push(c);
    }
    return list;
  };

  // Return the uncovered pair with the fewest legal third points (MRV), or null
  // when every pair is covered.
  const pickPair = () => {
    let best = null, bestList = null, bestCount = Infinity;
    for (let a = 0; a < n; a++) {
      for (let b = a + 1; b < n; b++) {
        if (used[a * n + b]) continue;
        const list = completions(a, b);
        if (list.length < bestCount) {
          best = [a, b]; bestList = list; bestCount = list.length;
          if (bestCount <= 1) return { pair: best, list: bestList }; // forced/dead
        }
      }
    }
    return best ? { pair: best, list: bestList } : null;
  };

  const rec = () => {
    if (++steps > CAP) throw new Error(`STS(${n}) search exceeded step cap`);
    const sel = pickPair();
    if (!sel) return true;            // all pairs covered → complete
    const { pair: [a, b], list } = sel;
    if (list.length === 0) return false; // dead end
    for (const c of list) {
      const kab = key(a, b), kac = key(a, c), kbc = key(b, c);
      used[kab] = used[kac] = used[kbc] = true;
      triples.push([a, b, c]);
      if (rec()) return true;
      triples.pop();
      used[kab] = used[kac] = used[kbc] = false;
    }
    return false;
  };

  if (!rec()) throw new Error(`no STS found for n=${n}`);
  return triples;
}

/**
 * Triples of a Steiner triple system on n points.
 * Preconditions: n ≡ 1 or 3 (mod 6), n ≥ 7.
 * Postconditions: returns n(n-1)/6 triples; every unordered pair of points
 *   appears in exactly one triple. Throws on invalid n.
 */
export function stsTriples(n) {
  if (!Number.isInteger(n) || n < 7 || !(n % 6 === 1 || n % 6 === 3)) {
    throw new Error(
      `Steiner triple system S(2,3,n) needs n ≡ 1 or 3 (mod 6), n ≥ 7; got ${n}`,
    );
  }
  return n % 6 === 3 ? stsBose(n) : stsBacktrack(n);
}

// ── Spec builders (one per generator family) ──────────────────────────

function stsSpec({ n }) {
  const triples = stsTriples(n);
  const vertices = Array.from({ length: n }, (_, i) => `n${i}`);
  const edges = triples.map((t, k) => ({ name: `e${k}`, members: t }));
  return { name: `STS_${n}`, block: `sts${n}`, vertices, edges };
}

function sunflowerSpec({ petals, core, petal }) {
  const vertices = [];
  const coreIdx = [];
  for (let i = 0; i < core; i++) {
    coreIdx.push(vertices.length);
    vertices.push(`c${i}`);
  }
  const edges = [];
  for (let pi = 0; pi < petals; pi++) {
    const members = coreIdx.slice();
    for (let j = 0; j < petal; j++) {
      members.push(vertices.length);
      vertices.push(`p${pi}v${j}`);
    }
    edges.push({ name: `petal${pi}`, members });
  }
  return { name: `Sunflower_k${petals}`, block: "sunflower", vertices, edges };
}

function completeUniformSpec({ n, r }) {
  const vertices = Array.from({ length: n }, (_, i) => `v${i}`);
  const edges = combinations(n, r).map((c, k) => ({ name: `e${k}`, members: c }));
  return { name: `Complete_${n}_${r}`, block: `k${n}_${r}u`, vertices, edges };
}

// ── Serialiser ────────────────────────────────────────────────────────

/**
 * Serialise a HypergraphSpec to `.hymeko` source.
 * Preconditions: vertices/edges are valid HyMeKo identifiers; each edge member
 *   indexes into `vertices`.
 * Postconditions: returns source in the compile-proven typical_graphs shape.
 */
export function specToHymeko({ name, block, vertices, edges }) {
  const lines = [`${name}{}`, block, "{"];
  for (const v of vertices) lines.push(`    ${v} {}`);
  lines.push("");
  for (const e of edges) {
    const refs = e.members.map((m) => `~${vertices[m]}`).join(", ");
    lines.push(`    @${e.name}{ (${refs}); }`);
  }
  lines.push("}");
  return lines.join("\n") + "\n";
}

// ── Generator registry (Strategy) ─────────────────────────────────────
// `params` drives the form generically (no per-generator UI code). All params
// here are numeric — the view coerces inputs with Number().

export const GENERATORS = [
  {
    id: "sts",
    label: "Steiner triple system S(2,3,n)",
    blurb: "Every pair of points lies on exactly one triple. Fano = S(2,3,7).",
    params: [
      { key: "n", label: "n (points)", type: "select", options: [7, 9, 13, 15, 19, 21, 25], default: 9 },
    ],
    validate: ({ n }) =>
      Number.isInteger(n) && n >= 7 && (n % 6 === 1 || n % 6 === 3)
        ? null
        : "n must be ≡ 1 or 3 (mod 6), n ≥ 7",
    build: stsSpec,
    summary: ({ n }) => `${n} points, ${(n * (n - 1)) / 6} triples`,
  },
  {
    id: "sunflower",
    label: "Sunflower (Δ-system)",
    blurb: "k petals sharing a common core; pairwise edge intersection = the core.",
    params: [
      { key: "petals", label: "petals (k)", type: "int", default: 4, min: 1, max: 12 },
      { key: "core", label: "core size", type: "int", default: 2, min: 0, max: 5 },
      { key: "petal", label: "petal size", type: "int", default: 2, min: 1, max: 6 },
    ],
    validate: ({ petals, core, petal }) => {
      if (!(petals >= 1 && petals <= 12)) return "petals must be 1..12";
      if (!(core >= 0 && core <= 5)) return "core size must be 0..5";
      if (!(petal >= 1 && petal <= 6)) return "petal size must be 1..6";
      const verts = core + petals * petal;
      if (verts > 60) return `too many vertices (${verts} > 60)`;
      return null;
    },
    build: sunflowerSpec,
    summary: ({ petals, core, petal }) =>
      `${core + petals * petal} vertices, ${petals} edges (arity ${core + petal})`,
  },
  {
    id: "complete",
    label: "Complete r-uniform K_n^(r)",
    blurb: "All C(n,r) subsets of size r over n vertices.",
    params: [
      { key: "n", label: "n (vertices)", type: "int", default: 5, min: 2, max: 10 },
      { key: "r", label: "r (uniformity)", type: "int", default: 3, min: 2, max: 6 },
    ],
    validate: ({ n, r }) => {
      if (!(n >= 2 && n <= 10)) return "n must be 2..10";
      if (!(r >= 2 && r <= n)) return "need 2 ≤ r ≤ n";
      const e = binom(n, r);
      if (e > 200) return `too many edges (C(${n},${r}) = ${e} > 200)`;
      return null;
    },
    build: completeUniformSpec,
    summary: ({ n, r }) => `${n} vertices, ${binom(n, r)} edges (arity ${r})`,
  },
];

/** Look up a generator by id, or null. */
export function generatorById(id) {
  return GENERATORS.find((g) => g.id === id) ?? null;
}

/**
 * Validate `params`, build the spec, and serialise to `.hymeko`.
 * Throws with the validation message on bad params, or on an unknown id.
 */
export function generateSource(id, params) {
  const g = generatorById(id);
  if (!g) throw new Error(`unknown generator: ${id}`);
  const err = g.validate(params);
  if (err) throw new Error(err);
  return specToHymeko(g.build(params));
}
