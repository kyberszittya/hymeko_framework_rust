# Tutorial — Solving a P-graph with HyMeKo

This is the **5-minute hands-on path** from a HyMeKo source file to a solved
process-network synthesis (PNS) problem: the cost-optimal set of operating units
that turns raw materials into the required products.

A P-graph (Friedler et al., 1992) is a bipartite graph of **materials** and
**operating units**. HyMeKo models it on its signed-incidence hypergraph: a
material is a *node*, an operating unit is a *hyperedge* whose `-` references are
consumed materials and `+` references are produced ones. The `pgraph` CLI reads
that, builds the P-graph, and runs the three classic algorithms — **MSG**
(maximal structure), **SSG** (solution structures), **ABB** (accelerated
branch-and-bound for the optimum).

You'll author one file, then `read` / `transform` / `solve` / `generate` it.

---

## Step 0 — Build the CLI

```bash
cargo build --release -p hymeko_pgraph --bin pgraph
# binary at target/release/pgraph  (examples below use `cargo run` for brevity)
```

---

## Step 1 — The meta-model (the vocabulary)

P-graph instances are authored in the general HyMeKo *meta-model* style: a small
shared file declares the vocabulary once, and instances reference it. It ships at
[`hymeko_pgraph/data/meta_pgraph.hymeko`](../../hymeko_pgraph/data/meta_pgraph.hymeko):

```hymeko
Pgraph_meta {
    author "Hajdu Csaba";
    source "Friedler et al.";
}

pgraph {
    meta_material {}
    raw:          + <isa> meta_material {}
    product:      + <isa> meta_material {}
    intermediate: + <isa> meta_material {}

    @process {}
}
```

You normally don't edit this — it just defines the four archetypes the engine
understands: the material roles `raw` / `product` / `intermediate`, and the
`@process` operating-unit type.

---

## Step 2 — Author a problem

Here is the worked example
[`hymeko_pgraph/data/prgraph_ex_3_1.hymeko`](../../hymeko_pgraph/data/prgraph_ex_3_1.hymeko):

```hymeko
PNS_Example_3_1 {
    @"meta_pgraph.hymeko";              // pull in the vocabulary
    using pgraph.raw as raw;
    using pgraph.product as product;
    using pgraph.intermediate as inter;
    using pgraph.process as process;    // optional: units are detected structurally
}

pns_3_1 {
    // Materials: classify each by <isa> against an archetype.
    A: + <isa> raw {}
    B: + <isa> raw {}
    G: + <isa> product {}
    C: + <isa> inter {}
    D: + <isa> inter {}
    E: + <isa> inter {}
    F: + <isa> inter {}

    // Operating units: an @-edge whose arcs are (-consumed, +produced).
    @u1 { (-B, +D); }
    @u2 { (-F, +D, +E); }
    @u3 { (-E, +G); }
    @u4 { (-D, +C, +G); }
    @u5 { (-A, -C, +G); }
}
```

The rules:

- **A material** is any node whose `<isa>` ancestry reaches `raw`, `product`, or
  `intermediate`. The role (R / P) follows from which archetype it reaches.
- **An operating unit** is an `@`-edge. By the *hybrid* rule, any `@`-edge whose
  arcs all reference materials is a unit — so you don't have to tag each one
  `<isa> process` (though you may). Inside the body, `-X` means "consumes X" and
  `+X` means "produces X".
- **Cost** is the edge's numeric value, e.g. `@u1 250 { (-B, +D); }`. Omitted ⇒
  `1.0`.

---

## Step 3 — `read`: what's in the file

```bash
cargo run -q -p hymeko_pgraph --bin pgraph -- read hymeko_pgraph/data/prgraph_ex_3_1.hymeko
```

```text
P-graph: prgraph_ex_3_1
  materials (7):
    A              [raw]
    B              [raw]
    C              [intermediate]
    D              [intermediate]
    E              [intermediate]
    F              [intermediate]
    G              [product]
  operating units (5):
    u1         cost     1.00   in: B                  out: D
    u2         cost     1.00   in: F                  out: D, E
    u3         cost     1.00   in: E                  out: G
    u4         cost     1.00   in: D                  out: C, G
    u5         cost     1.00   in: A, C               out: G
```

---

## Step 4 — `transform`: the bipartite P-graph

```bash
cargo run -q -p hymeko_pgraph --bin pgraph -- transform hymeko_pgraph/data/prgraph_ex_3_1.hymeko
```

```text
P-graph: prgraph_ex_3_1
  M-nodes (7): A B C D E F G
  O-nodes (5): u1 u2 u3 u4 u5
  signed incidence (13 edges):
    B              ──consumed──▶ u1
    u1             ──produced──▶ D
    F              ──consumed──▶ u2
    ...
```

This is the engine's actual view: the M/O partition and the directed signed
incidence (`m → u` consumed, `u → m` produced).

---

## Step 5 — `solve`: MSG, SSG, ABB

```bash
cargo run -q -p hymeko_pgraph --bin pgraph -- solve hymeko_pgraph/data/prgraph_ex_3_1.hymeko
```

```text
P-graph: prgraph_ex_3_1  (strict no-excess)
  MSG  maximal structure: { u1, u4, u5 }   [2 of 5 units pruned]
  SSG  feasible solution structures: 1
  ABB  optimum: { u1, u4, u5 }   cost 3.00   [explored 7]
```

How to read it:

- **MSG** pruned `u2` and `u3`. Why? `F` is declared `intermediate` but **no unit
  produces it**, so `u2` (which consumes `F`) can never run; `u3` depends on `u2`'s
  output `E`, so it goes too. MSG keeps only the units that can participate in
  some feasible solution.
- **SSG** found 1 combinatorially feasible structure inside MSG.
- **ABB** is the cost-optimal one: `{u1, u4, u5}` at total cost 3.0. Under *strict
  no-excess*, `u4`'s by-product `C` must be consumed by something — `u5` does — so
  `{u1, u4}` alone is infeasible.

### Useful flags

| Flag | Effect |
| --- | --- |
| `--relaxed` | Relaxed no-excess (P-graph Studio default): by-products may be vented. Affects MSG + ABB. |
| `--weights "1,0.5"` | Multi-objective ABB: weighted sum over the source's `cost <dim> N;` dimensions. |
| `--json` | Machine-readable analysis (the same JSON the `hymeko_pgraph_dump` tool emits). |

```bash
cargo run -q -p hymeko_pgraph --bin pgraph -- solve <file> --relaxed --json
```

---

## Step 6 — `generate`: a graph artifact

Graphviz DOT (prints to stdout, or `--out file.dot`):

```bash
cargo run -q -p hymeko_pgraph --bin pgraph -- generate hymeko_pgraph/data/prgraph_ex_3_1.hymeko > pns.dot
# raws green, products gold, units blue boxes
```

Directly to **PNG or SVG** (requires [Graphviz](https://graphviz.org) `dot` on
your `PATH`; `--out` is required):

```bash
cargo run -q -p hymeko_pgraph --bin pgraph -- generate <file> --format png --out pns.png
cargo run -q -p hymeko_pgraph --bin pgraph -- generate <file> --format svg --out pns.svg
```

If `dot` isn't installed, the command fails with an install hint rather than a
crash — install Graphviz (`apt install graphviz` / `brew install graphviz` /
`choco install graphviz`) or fall back to `--format dot` and render it yourself.

A P-graph Studio file (bakes in the ABB optimum):

```bash
cargo run -q -p hymeko_pgraph --bin pgraph -- generate <file> --format pgip --out problem.pgip
```

---

## Other input shapes

The CLI auto-detects the input, so the same four commands work on:

- **Literal-tag `.hymeko`** — the older idiom that tags nodes/edges directly
  instead of using `<isa>`, e.g. `A <material> <raw> {}` and
  `@u1 <unit> 250 { (-B, +D); }` (see
  [`data/pgraph/hda.hymeko`](../../data/pgraph/hda.hymeko)). No meta include needed.
- **`.pgip`** — a P-graph Studio SQLite file: `pgraph solve problem.pgip`.

---

## In the browser (WASM)

The same engine runs client-side via `hymeko_wasm`. Build the bundle:

```bash
rustup target add wasm32-unknown-unknown
wasm-pack build hymeko_wasm --target web
```

Then from JavaScript, pass the instance source and the meta source as strings:

```js
import init, { pgraph_solve, pgraph_dot } from "./pkg/hymeko_wasm.js";
await init();

const analysis = JSON.parse(pgraph_solve(instanceSrc, metaSrc)); // {msg_units, abb, ...}
const dot = pgraph_dot(instanceSrc, metaSrc);                     // Graphviz string
```

(For a literal-tag instance, pass `""` for `metaSrc`.)

---

## Cheat sheet

```bash
pgraph read      <file>                          # materials + units
pgraph transform <file>                          # bipartite incidence
pgraph solve     <file> [--relaxed] [--weights "…"] [--json]
pgraph generate  <file> [--format dot|png|svg|pgip] [--out PATH]   # png/svg need Graphviz
```

Authoring an instance: include `@"meta_pgraph.hymeko"`, `using pgraph.<role> as …`,
type materials with `<isa>`, write units as `@u { (-in, +out); }` with an optional
numeric cost on the edge.
