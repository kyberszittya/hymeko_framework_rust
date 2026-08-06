# HyMeKo — Technology milestone (2026-06-16)

**Purpose.** A point-in-time maturity snapshot of the framework, framed around the
question a reviewer (Kato) will ask: *what is proven, what is a prototype, what is
still a research program?* This is a status assessment, not a task report; the
living index is [ROADMAP.md](../ROADMAP.md), open work is [BACKLOG.md](../BACKLOG.md).

## One-line state

The **infrastructure half** (canonical hypergraph IR → many faithful targets,
validation-gated) is solid and demonstrable. The **learning half** (structural
priors) is proven on signed graphs under a strict protocol, and **falsified for
vision accuracy** — honestly. The central claim to defend is the *unified
representation-and-learning substrate*, not any single benchmark win.

## Maturity by component

Legend: **Proven** = tested + measured + reproducible; **Prototype** = runs
end-to-end, not yet hardened/benchmarked at scale; **Program** = designed/partial,
research-open.

| Component | Maturity | Evidence / note |
|---|---|---|
| DSL + canonical IR + parser (`hymeko_core`, `parser`) | **Proven** | LALR(1) parser, IR lowering, module store; core crate, CORE.YAML-locked. |
| Signed-incidence tensor; star vs clique expansion | **Proven** | star (sparse, O(\|E\|·d̄)) vs clique (dense, O(\|E\|·d̄²)); anchor: 1,498 vs 10,991 NNZ, 79.7 ms vs 496 ms. |
| Hypergraph generators (Steiner/sunflower/complete) | **Proven** | `hymeko::generators`, single-sourced; 10 core + 18 hive tests; `S(2,3,25)` ≪ 100 ms. |
| HIVE canonical store (transactions, queries, gates) | **Proven** | content-hashed state, atomic deltas, typed + association queries; 22 tests. |
| Transforms: `.hymeko` → SysML / DOT / `torch.nn` | **Prototype** | structural parity holds; runnable numeric round-trip not yet (pulls torch). |
| Web editor (views, generators, profiles, arcs) | **Proven** (as a tool) | WASM compile → graph/3D/SysML/kinematic/generate views; multi-file imports, profiles. |
| P-graph methodology (A1–A5, MSG/SSG/ABB, reachability) | **Proven** | validated against Pimentel textbook examples; `hymeko_pgraph`. |
| Signed-link learning: SignedKAN / HSiKAN / Gömb | **Proven** (signed graphs) | honest strict protocol, 5-seed paired; the 0.996 is transductive, the strict baseline is the claim. |
| Leakage / honesty protocol (σ-masked strict) | **Prototype→Proven** | harness + ~6× vectorization done; 5-seed grid running → Nature Table 1. |
| Soma vision (RicciStim: quadtree/Forman/Hodge/stim) | **Falsified for accuracy** | < MLP on MNIST/Cluttered-MNIST. Aggregator upgrade helps (+11–27 %) but does not overturn it. |
| Cortical brain-predictivity (Cichy-92) | **Program** | infra + synthetic smoke done; needs real fMRI — the un-falsified axis. |
| HSMM → FPGA (Nagare → HSMM → Zynq) | **Program** | RTL + ISA spec exist (CORE-locked); compiler plan frozen. |
| Robotics (UR5e ROS2, contextual hypergraph layer) | **Prototype** | sim stack + pick-and-place; SISY control paper. |

## Evidence highlights (honest numbers)

- **Star vs clique efficiency** — the cleanest engineering claim; 1,498 vs 10,991
  NNZ, 79.7 ms vs 496 ms. *Caveat (own framing): star uses fewer NNZ but more
  dimensions; the sparse structure is preferred.*
- **Aggregator upgrade A/B** (mixer+highway+pyramid vs bare sum): MNIST cls
  0.272→0.303 (+11 %, 5× lower variance); Cluttered-MNIST detection 0.106→0.135
  (+27 %, still climbing at 20 ep). Consistent improvement; **not** a vision win.
- **Generators** single-sourced into core; 28 tests; sub-100 ms at n=25.

## Proven vs prototype vs program (Kato's question)

- **Proven now:** the IR + tensor + generators + HIVE + P-graph + the signed-link
  strict-protocol results + the star/clique efficiency argument.
- **Prototype:** the multi-target transforms (structural parity, not yet numeric
  round-trip), the leakage-audit grid (running), the robotics sim.
- **Program (research-open):** vision-as-structure (falsified for accuracy; brain
  axis untested), HSMM→FPGA compiler, P-graph-as-architecture-search.

## Risks / open questions

- **Vision falsification stands.** Do not over-claim it in the talk; frame as a
  stress test. The strong aggregator helps but the approach is below a linear
  baseline.
- **GPU did not rescue training wall** — the per-image graph build is CPU-bound;
  the topology cache (T1) is the dependency for any honest full-scale vision run.
- **Single-seed scale results** for the detection A/B — multi-seed needed before
  any published number.
- **One IR, many claims** — the reviewer risk is breadth, not depth; the framing
  slides (Kato reframe) exist to control it.

## Next milestones

1. Topology cache → full 5000-img/40-ep detection run (multi-seed) — settle the
   vision-headroom question honestly.
2. Nature leakage 5-seed σ-strict grid → Table 1.
3. Signed-link baseline table (tier-2/3 audit, prefilter→3-dataset).
4. Deliver the PhD seminar (deck reframed) and the editor-driven overview demo.

## Provenance

Not a single experiment — a synthesis of the 2026-06-15/16 reports
(`soma-vision-backbone-upgrade`, `generators-into-core`, `base-soma-vs-linear`,
`seminar-deck-additions`) and project memory. Git working tree dirty (active
session). No measurement was run for this report; all numbers cite their source
report.
