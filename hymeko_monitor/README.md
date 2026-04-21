# `hymeko_monitor` — Runtime Monitoring of Signed-Incidence Hypergraphs

Two deliverables in one package, intended for two audiences:

| Path                        | Audience     | Purpose                                                |
|-----------------------------|--------------|--------------------------------------------------------|
| `paper/paper_outline.tex`   | Csaba        | LNCS-formatted paper skeleton with dense per-section   |
|                             |              | technical content; target RV 2026 or RV 2027.          |
| `SPEC.md`                   | Claude Code  | Implementation brief: scope, architecture, traits,     |
|                             |              | pitfalls, non-goals.                                   |
| `src/`, `tests/`, `Cargo.toml` | Claude Code | Rust scaffold ready to build on — types, module layout,|
|                             |              | combinator DSL, sliding-window skeleton, test target.  |

## For the paper

Open `paper/paper_outline.tex`. Every section has bullet-level notes
describing what goes in it. Section 3 (Semantics) has the full definition
list sketched in comments — this is the technical core and the hardest
writing; start there once the semantics is settled. Section 5 (Case
Studies) has two scenarios drafted; wire them up to the crate output.

## For Claude Code

Read `SPEC.md` first. Then look at:

- `src/predicate.rs` — the contract surface with `hymeko_core`.
- `src/formula/stl.rs` — AST and combinators are **implemented**;
  robustness of temporal operators is **stubbed** because it requires
  the sliding-window monitor.
- `src/monitor/stl.rs` — the main work. `observe()` and
  `allocate_windows()` are `todo!()`.
- `src/window.rs` — implemented and tested; use it.
- `src/robustness.rs` — implemented and tested; use it.
- `src/incremental.rs` — v0.1 ships pessimistic (always re-evaluate);
  leave as-is for v0.1.
- `tests/stl_kinematic.rs` — the target integration test. Uncomment and
  wire up once `observe()` is done.

### v0.1 definition of done

1. `cargo build --release` clean.
2. `cargo test --release` passes all unit tests and the kinematic
   integration test.
3. One-paragraph benchmark entry in the paper's §5.1.

### Do **not** implement in v0.1

- CTL model checking
- Unbounded LTL with 3-valued verdicts
- Distributed monitors
- Shield synthesis / RL training loops
- GPU acceleration

See `SPEC.md` for the full list and the pitfalls section.

## Directory layout

```
hymeko_monitor/
├── Cargo.toml
├── README.md
├── SPEC.md
├── paper/
│   └── paper_outline.tex
├── src/
│   ├── lib.rs
│   ├── predicate.rs
│   ├── trace.rs
│   ├── window.rs            (implemented + tested)
│   ├── robustness.rs        (implemented + tested)
│   ├── incremental.rs       (v0.1 pessimistic)
│   ├── formula/
│   │   ├── mod.rs
│   │   ├── stl.rs           (AST + combinators implemented)
│   │   └── ltl.rs           (wraps stl.rs)
│   └── monitor/
│       ├── mod.rs           (Monitor trait, Verdict)
│       └── stl.rs           (SKELETON — main work here)
└── tests/
    └── stl_kinematic.rs     (target integration test)
```
