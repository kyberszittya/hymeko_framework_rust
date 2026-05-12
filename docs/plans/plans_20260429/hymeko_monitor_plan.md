# HyMeKo Runtime Monitoring & Formal Verification — Integration Plan

## 1. Motivation

Industry feedback consistently identifies two distinct needs:

- **Runtime verification** — online monitors deployed alongside executing systems
- **Testing** — offline trace checking, oracle generation, and falsification during development

These require overlapping but non-identical infrastructure. The plan addresses both under a unified formalism layer, keeping the *desc-as-query* invariant throughout: a monitor's scope is always a query result, never a separately maintained scope language.

---

## 2. Formalism Landscape

### Tier 1 — Core (implement first)

| Formalism | Use case | Notes |
|---|---|---|
| STL | Continuous/real-valued signals, robustness | Extend existing `verification/STL` |
| LTL | Discrete event traces, FSM properties | Büchi automaton backend |
| Past-LTL | Online monitoring, no future look-ahead | O(n) incremental; critical for runtime |
| MTL | Real-time with metric intervals | Zone automaton backend |

### Tier 2 — Extended

| Formalism | Use case |
|---|---|
| CTL / CTL* | Branching-time, state-space exploration |
| HyperLTL | Hyperproperties: information flow, noninterference |
| PSL (subset) | HDL-style assertions; natural for FPGA/SVA emission |
| Regex over traces | Lightweight pattern matching, test oracles |

### Tier 3 — Research (original contributions)

| Formalism | Description |
|---|---|
| **HTL** (Hypergraph Temporal Logic) | Path quantifiers range over hypergraph walks, not linear traces. Semantics grounded in G-SPHF signed incidence. Novel — no prior art. |
| STL with hypergraph robustness | Robustness ρ as a signed incidence function; direct G-SPHF connection. |

---

## 3. Architecture

```
┌──────────────────────────────────────────────────────┐
│               HyMeKo Property Layer                  │
│   .hko / .hkm files with monitor{} and formula{})   │
└─────────────────────┬────────────────────────────────┘
                      │  HIR → MIR → LIR (existing pipeline)
                      ▼
┌──────────────────────────────────────────────────────┐
│              hymeko_monitor  crate                   │
│                                                      │
│  ┌──────────────┐  ┌─────────────┐  ┌────────────┐  │
│  │ Formula AST  │  │  Synthesis  │  │  Runtime   │  │
│  │  + Parser    │→ │  Engine     │→ │  Engine    │  │
│  └──────────────┘  └─────────────┘  └────────────┘  │
│         ↑                ↑                ↑          │
│   LTL/MTL/STL     Büchi / Zone /    Online/Offline   │
│   PSL / HTL       Observer auto.    + Robustness     │
└──────────────────────────────────────────────────────┘
                      │
         ┌────────────┼──────────────┐
         ▼            ▼              ▼
      Trace DB     Test Gen       HSMM integration
      (offline)   (falsification) (hardware monitors)
```

---

## 4. `hymeko_monitor` Crate — Module Breakdown

```
hymeko_monitor/
├── ast/
│   ├── ltl.rs          # LTL formula AST
│   ├── mtl.rs          # MTL with time intervals
│   ├── stl.rs          # extend existing STL
│   ├── ctl.rs          # CTL / CTL*
│   └── htl.rs          # Hypergraph Temporal Logic (novel)
├── synthesis/
│   ├── buchi.rs        # LTL → Büchi automaton
│   ├── zone.rs         # MTL → timed automaton / zone graph
│   ├── observer.rs     # Past-LTL → O(n) online observer
│   └── stl_robust.rs   # STL robustness (min/max semantics, gradient)
├── runtime/
│   ├── online.rs       # Incremental monitor (streaming)
│   ├── offline.rs      # Full trace checking
│   └── verdict.rs      # Verdict: {True, False, Unknown, Robustness(f64)}
├── testing/
│   ├── trace_gen.rs    # Trace generation from spec (bounded unrolling)
│   ├── falsifier.rs    # STL robustness-guided falsification (CMA-ES)
│   └── coverage.rs     # Temporal logic coverage criteria
└── integration/
    ├── hsmm_bridge.rs  # HSMM state → monitor event stream
    └── ros2_bridge.rs  # ROS2 topic → signal (future)
```

---

## 5. HyMeKo Language Extensions

### 5.1 Grammar Additions (LALRPOP — indicative)

Three new top-level constructs; existing grammar unchanged:

```lalrpop
Item = {
    HypergraphDecl,
    MonitorDecl,      // standalone monitor
    MonitorLibDecl,   // parameterized .hkm library entry
    UsingDecl,
    ...
}

// Inline monitor inside hypergraph — query IS the scope
InlineMonitor: MonitorItem = {
    "monitor" <name:Ident> "{"
        "query"    ":" <query:QueryExpr> ";"
        "formula"  ":" <formula:FormulaExpr> ";"
        "verdict"  ":" <verdict:VerdictMode> ";"
        ("robustness" ":" <rob:Bool> ";")?
    "}"
}

FormulaExpr: FormulaExpr = {
    "ltl" "(" <s:StringLit> ")"            => FormulaExpr::Ltl(s),
    "mtl" "(" <s:StringLit> <i:Interval> ")" => FormulaExpr::Mtl(s, i),
    "stl" "(" <s:StringLit> ")"            => FormulaExpr::Stl(s),
    "htl" "(" <s:StringLit> ")"            => FormulaExpr::Htl(s),
    "psl" "(" <s:StringLit> ")"            => FormulaExpr::Psl(s),
}

VerdictMode: VerdictMode = {
    "online"  => VerdictMode::Online,
    "offline" => VerdictMode::Offline,
    "both"    => VerdictMode::Both,
}

// Parameterized library monitor
MonitorLibDecl: MonitorLibDecl = {
    "monitor" <name:Ident>
    ("<" <params:Comma<ParamDecl>> ">")?
    "{" <formula:FormulaExpr> ";"
        ("applies_to" ":" <filter:QueryExpr> ";")? "}"
}
```

### 5.2 What Changes vs What Doesn't

| Element | Changed? |
|---|---|
| `QueryExpr` grammar | **No** — reused verbatim as monitor scope |
| `HypergraphDecl` body | Add `InlineMonitor` as valid body item |
| Top-level `Item` | Add `MonitorDecl`, `MonitorLibDecl` variants |
| `UsingDecl` / alias | **No** — can alias monitor libs as-is |
| String literal handling | **No** — formula strings are opaque at parse time |
| `ParamDecl` | **No** — reuse existing parameterization |

### 5.3 AST Additions (`ast.rs`)

```rust
pub enum Item {
    Hypergraph(HypergraphDecl),
    Monitor(MonitorDecl),
    MonitorLib(MonitorLibDecl),
    Using(UsingDecl),
}

pub struct MonitorItem {
    pub name: Ident,
    pub query: QueryExpr,      // existing type, unchanged
    pub formula: FormulaExpr,
    pub verdict: VerdictMode,
    pub robustness: bool,
}

pub enum FormulaExpr {
    Ltl(String),
    Mtl(String, Interval),
    Stl(String),
    Htl(String),
    Psl(String),
}

pub enum VerdictMode { Online, Offline, Both }

pub struct MonitorLibDecl {
    pub name: Ident,
    pub params: Vec<ParamDecl>,
    pub formula: FormulaExpr,
    pub filter: Option<QueryExpr>,
}

pub struct Interval {
    pub lo: f64,
    pub hi: Option<f64>,   // None = unbounded
}
```

---

## 6. Desc-as-Query Invariant

The invariant: **a monitor's scope is always a `QueryExpr` result — no separate scope language exists**.

```
QueryExpr  →  BindingSet { vertex bindings, hyperedge bindings }
                    │
                    ▼
FormulaExpr evaluated over BindingSet as signal domain
```

Synthesis pipeline:

```
MonitorItem
    ├─ query   →  hymeko_query engine  →  BindingSet  (already implemented)
    └─ formula →  hymeko_monitor::synthesis  →  Observer(BindingSet)
```

Monitor library `applies_to` clauses are query filters — same mechanism, no special cases.
`using...as` alias imports work on monitor libraries without modification.

---

## 7. Syntax Examples

### Inline monitor in `.hko`

```hko
hypergraph ControlLoop {
    node Sensor, Actuator, Controller;
    hyperedge feedback: [Sensor] -> [Controller];

    monitor safety_response {
        query:   edge where attr("type") == "request_response";
        formula: ltl("□(req → ◇ack)");
        verdict: online;
    }

    monitor signal_bounds {
        query:   node where attr("role") == "sensor";
        formula: stl("□[0,1](|signal| < 5.0)");
        verdict: both;
        robustness: true;
    }
}
```

### Parameterized library monitor in `.hkm`

```hko
monitor BoundedResponse<T: Duration> {
    formula: mtl("□(req → ◇ack)", [0, T]);
    applies_to: edge where attr("type") == "request_response";
}
```

### Import via existing `using...as`

```hko
using "safety.hkm" as Safety;

monitor my_monitor = Safety::BoundedResponse<10ms>;
```

---

## 8. Compiler Pipeline Integration

```
.hko source
    │
    ▼ HIR  — formula parsed, scope resolved to hypergraph elements via QueryExpr
    │
    ▼ MIR  — formula normalized (NNF, rewriting); scope → signal extraction plan
    │
    ├── LTL  →  Büchi automaton  →  product with hypergraph FSM  →  LIR observer
    ├── MTL  →  zone automaton   →  LIR timed observer
    └── STL  →  evaluation tree  →  LIR vectorized evaluator (SIMD-friendly)
    │
    ▼ LIR  →  Rust monitor code  /  SystemVerilog assertions (HSMM FPGA target)
```

SVA/PSL emission via the LIR → SystemVerilog path is a direct deliverable for the
Zynq UltraScale+ target already in use for the HSMM kernel.

---

## 9. Testing Subsystem

This is the primary industry-facing differentiator.

### 9.1 Trace-based oracle generation

- From STL/LTL spec, generate conformance test traces (satisfying and violating)
- Bounded model checking unrolling: spec → SAT/SMT query → concrete trace
- Backend: `z3` or `cadical` (pure Rust) via FFI

### 9.2 STL falsification

- Treat STL robustness ρ as cost function to minimize
- Optimizer: CMA-ES or Bayesian optimization over input signal space
- Wire ρ gradient through existing `hymeko_core/autograd` tape
- Output: minimal robustness counterexample + witness trace

### 9.3 Coverage

- LTL sub-formula coverage: which obligations were exercised?
- Hyperedge coverage: which scoped monitor locations were activated?
- MC/DC-style conditions extracted from formula structure

---

## 10. Integration with Existing Modules

| Existing module | Monitor integration |
|---|---|
| `behavior/FSM` | LTL monitors as product automata with hypergraph FSMs |
| `verification/STL` | Extend with robustness gradient, falsifier hook |
| `autograd/` | ∂ρ/∂input for STL falsification |
| `planning/` | LTL goal specs ↔ plan synthesis duality |
| `gsphf/` | Robustness as signed incidence function → HTL semantics |
| HSMM | Monitor as HSMM program; NEWE/ATT primitives emit events |

---

## 11. Phased Roadmap

| Phase | Scope | Target venue |
|---|---|---|
| **P1** (now – June 2026) | STL robustness gradient; Past-LTL online observer; `monitor{}` block in grammar | MDPI Actuators / SISY |
| **P2** (Nagoya, summer 2026) | LTL → Büchi synthesis; offline trace checker; `.hkm` monitor library files | Japanese conferences |
| **P3** (autumn 2026) | MTL zone automaton; STL falsifier (CMA-ES + autograd); SVA/PSL LIR emission | IEEE SMC 2027 |
| **P4** (2027) | HTL formal semantics; monitor synthesis for hypergraph walks; G-SPHF robustness semantics | Journal target |

---

## 12. Original Contributions

Two publishable novelties not present in existing tools (Breach, S-TaLiRo, AMT, RTAMon):

**Structural scope quantification**
Monitors range over hyperedges and vertex sets selected by query, not over scalar
signal names. Scope is first-class and compositional.

**Hypergraph Temporal Logic (HTL)**
Path quantifiers range over hypergraph walks. Semantics grounded in G-SPHF signed
incidence; robustness is a signed incidence function rather than a scalar distance.
No prior art in this formulation.

These two points together constitute the load-bearing claim for a journal submission
targeting the P4 slot.
