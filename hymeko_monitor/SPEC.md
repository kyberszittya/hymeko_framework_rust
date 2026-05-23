# `hymeko_monitor` — Specification

## Purpose

A Rust crate implementing runtime monitoring of temporal-logic properties
over signed-incidence directed hypergraphs (the HyMeKo IR). This is the
implementation substrate for the paper
`paper/paper_outline.tex` targeting RV 2026 / 2027.

## Scope — What This Crate IS

- Bounded-memory **online monitor** for bounded-horizon Signal Temporal
  Logic over hypergraph traces.
- **Structural robustness**: robustness degree extended to hypergraph
  predicates (counting, arity, attribute-threshold, existential).
- Incremental predicate evaluation: attribute updates trigger re-evaluation
  only of predicates whose match set could have changed.
- Integration points with `hymeko_core` (the IR), consumed through traits
  so the monitor is not coupled to a specific hypergraph representation.

## Scope — What This Crate IS NOT (in v0.1)

- **Not** a CTL model checker. CTL requires a computation tree which
  requires structural rewrites as transitions; v0.1 restricts to
  attribute-update transitions only. CTL arrives when structural rewrites
  are added (v0.2).
- **Not** an unbounded-horizon LTL monitor. The bounded-memory guarantee
  requires bounded STL intervals. Unbounded LTL requires 3-valued
  semantics (impartiality) and is deferred.
- **Not** a distributed monitor. v0.1 is single-process.
- **Not** a shield synthesiser or policy trainer. The robustness output
  is a building block for RL reward shaping (separate follow-up paper);
  training code lives in a sibling crate, not here.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│ hymeko_monitor                                               │
│                                                              │
│  formula/                                                    │
│    stl.rs     — STL AST, bounded-horizon fragment            │
│    ltl.rs     — LTL AST, reduced to STL with [0, ∞)          │
│    ctl.rs     — CTL AST (parse + print only in v0.1)         │
│                                                              │
│  predicate.rs — HypergraphPredicate trait + builders         │
│                  (COUNT, EXISTS, FORALL, ARITY, ATTR, ...)   │
│                                                              │
│  robustness.rs — robustness combinators (min, max, sup, inf) │
│                                                              │
│  window.rs     — ring-buffer for sliding-window evaluation   │
│                                                              │
│  incremental.rs — dependency tracking, delta propagation     │
│                                                              │
│  monitor/                                                    │
│    mod.rs     — Monitor trait + Verdict type                 │
│    stl.rs     — STL online monitor (main)                    │
│    ltl.rs     — LTL online monitor (thin wrapper over STL)   │
│                                                              │
│  trace.rs     — Sample, Timestamp                            │
│                                                              │
│  lib.rs       — public API                                   │
└──────────────────────────────────────────────────────────────┘
```

## Core Traits

```rust
/// A snapshot of a HyMeKo structure at a point in time. Implemented
/// externally by `hymeko_core` (or a test fixture). The monitor never
/// owns the hypergraph; it borrows read-only views.
pub trait HypergraphState {
    type VertexId: Copy + Eq + std::hash::Hash;
    type EdgeId:   Copy + Eq + std::hash::Hash;
    type TypeId:   Copy + Eq;
    type Attr;

    fn vertices(&self)  -> Box<dyn Iterator<Item = Self::VertexId> + '_>;
    fn edges(&self)     -> Box<dyn Iterator<Item = Self::EdgeId> + '_>;
    fn incidences(&self, e: Self::EdgeId)
                        -> Box<dyn Iterator<Item = (Self::VertexId, Sign)> + '_>;
    fn vertex_type(&self, v: Self::VertexId) -> Self::TypeId;
    fn edge_type(&self,   e: Self::EdgeId)   -> Self::TypeId;
    fn attr(&self, v: Self::VertexId, key: &str) -> Option<&Self::Attr>;
    fn inherits(&self, t: Self::TypeId, base: Self::TypeId) -> bool;
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Sign { Plus, Minus, Neutral }

/// A predicate over a hypergraph state. `eval` is the boolean reading;
/// `robustness` is the quantitative reading (±∞ for structural-only
/// predicates without natural numeric content).
pub trait HypergraphPredicate<H: HypergraphState> {
    fn eval(&self, h: &H) -> bool;
    fn robustness(&self, h: &H) -> f64 {
        if self.eval(h) { f64::INFINITY } else { f64::NEG_INFINITY }
    }
    /// For incremental evaluation: the set of vertex/edge ids whose
    /// change would invalidate this predicate's current verdict.
    fn dependencies(&self, h: &H) -> Dependencies<H>;
}

pub struct Dependencies<H: HypergraphState> {
    pub vertices: Vec<H::VertexId>,
    pub edges:    Vec<H::EdgeId>,
    pub global:   bool,  // true = always re-evaluate (e.g. ACYCLIC)
}

/// Online monitor: consumes samples one at a time, maintains bounded
/// state, produces verdicts at the window-trailing edge.
pub trait Monitor<H: HypergraphState> {
    type Output;
    fn observe(&mut self, sample: Sample<H>);
    fn verdict(&self) -> Option<(Timestamp, Self::Output)>;
}

pub struct Sample<H: HypergraphState> {
    pub state: H,
    pub t:     Timestamp,
}

pub type Timestamp = f64;
```

## Formula Construction — Target DSL

Builder API so formulas read like formulas, not like AST constructors:

```rust
use hymeko_monitor::formula::stl::*;
use hymeko_monitor::predicate::*;

// Property: always, if mode is collaborative, then for every joint,
// the joint's position stays within limits for the next 100 ms.
let phi = always(
    implies(
        has_tag("collaborative_mode"),
        always_bounded(0.0, 0.1,
            forall(
                kind("joint"),
                attr_in_range(
                    joint_key("position"),
                    JOINT_MIN, JOINT_MAX,
                )
            )
        )
    )
);
```

The combinators `always`, `eventually`, `always_bounded`, `until_bounded`,
`implies`, `and`, `or`, `not` produce STL AST nodes; `forall`, `exists`,
`count`, `arity`, `attr_in_range`, `has_tag`, `kind`, `inherits` produce
predicates.

## STL AST (sketched)

```rust
pub enum Stl<P> {
    True,
    Pred(P),
    Not(Box<Stl<P>>),
    And(Box<Stl<P>>, Box<Stl<P>>),
    Or(Box<Stl<P>>, Box<Stl<P>>),
    /// F_{[a, b]} phi
    Eventually(f64, f64, Box<Stl<P>>),
    /// G_{[a, b]} phi
    Always(f64, f64, Box<Stl<P>>),
    /// phi U_{[a, b]} psi
    Until(f64, f64, Box<Stl<P>>, Box<Stl<P>>),
}

impl<P> Stl<P> {
    /// Maximum time horizon required for online evaluation at time t.
    pub fn horizon(&self) -> f64 { /* recursive max */ }

    /// Depth of the bounded-window required per subformula.
    pub fn subformula_horizons(&self) -> Vec<(NodeId, f64)> { todo!() }
}
```

## Robustness Semantics

```rust
// where t is the evaluation time and pi is the trace
rho(Pred(P), pi, t)          = P.robustness(pi.state_at(t))
rho(Not(phi))                = -rho(phi)
rho(And(phi, psi))           = min(rho(phi), rho(psi))
rho(Or(phi, psi))            = max(rho(phi), rho(psi))
rho(Eventually([a,b], phi))  = sup_{t' in [t+a, t+b]} rho(phi, pi, t')
rho(Always([a,b], phi))      = inf_{t' in [t+a, t+b]} rho(phi, pi, t')
rho(Until([a,b], phi, psi))  = sup_{t' in [t+a, t+b]}
                                 min( rho(psi, pi, t'),
                                      inf_{t'' in [t, t']} rho(phi, pi, t'') )
```

Computed via sliding-window DP (Donze-Maler 2010 style). Required
property to preserve: `rho > 0 ⟺ phi holds`, `rho < 0 ⟺ phi fails`,
`rho = 0` on the boundary.

## Case Study Scenarios (must pass)

### 1. Kinematic Limit

Property: `G(collaborative_mode → G(∀joint. within_limits(joint)))`

Trace: `anthropomorphic_arm` fixture, 30s simulated motion, 100 Hz.
Expected: robustness crosses zero on violation within 1 sample.

### 2. Cross-Layer Health-to-Speed (Szilágyi scenario)

Property: `G((health < θ_h) → G_{[0, 0.1]}(speed ≤ v_maint_limit))`

Trace: maintenance context updates health attribute; production context
updates speed; monitor detects delayed-response violation.

See `tests/stl_kinematic.rs` for the skeleton.

## Dependencies

- `hymeko_core` — for `HypergraphState` implementation on the IR
- `thiserror` — error types
- No heavy ML/GPU deps. This crate is small (~2000 LOC target) and
  pure-Rust.

## Deliverables for v0.1

1. Compilable crate with public API from `lib.rs`.
2. STL AST + builders passing doctests.
3. Sliding-window monitor producing correct robustness on the two case
   studies.
4. Incremental-evaluation path (predicate dependency tracking).
5. Two integration tests (`tests/stl_kinematic.rs`, `tests/stl_cross_layer.rs`).
6. Bench harness: per-observation latency across window-size sweep.
7. README with the formula DSL + examples.

## Non-Goals for v0.1 (Claude Code, DO NOT implement these)

- CTL model checking (needs structural rewrite transitions)
- Unbounded LTL with 3-valued verdicts
- Distributed monitoring
- Shield synthesis / RL training loops
- GPU acceleration

## Pitfalls to Watch

- **Floating-point robustness** near zero: use a small epsilon for the
  boundary case, document it in the `Monitor` trait.
- **Window sizing**: compute per-subformula horizons, not one global
  horizon. Over-allocating windows is memory death for long-horizon
  formulas.
- **Incremental delta correctness**: it is tempting to skip predicate
  re-evaluation when a vertex attribute updates. Only skip if the
  predicate's `dependencies()` explicitly excludes that vertex. When in
  doubt, re-evaluate. Correctness before performance for v0.1.
- **Don't invent predicate types**: reuse the HyMeKo predicate algebra
  via the `HypergraphPredicate` trait. New predicate forms go through
  Csaba, not through auto-generation.
