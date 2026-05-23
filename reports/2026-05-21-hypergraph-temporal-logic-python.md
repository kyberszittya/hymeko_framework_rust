# Hypergraph Temporal Logic (HTL) — pure-Python implementation — 2026-05-21

## Summary

Shipped a self-contained pure-Python HTL monitoring framework at
`signedkan_wip/src/htl/`.  Robust-STL semantics, recursive-descent parser
for a small temporal-logic language, online monitor with a bounded
history, predicate registry for hypergraph-native signals, and a
`--monitor` CLI flag on `run_gomb_smoke`.

The shape this lands in is the *API* and the *formula language* — the
Rust crate (with PyO3 bindings, the corpus-callosum-of-monitoring story
for the Niitsuma audience) is a deferred follow-up.  This implementation
makes the API concrete, the unit tests passing, and the integration with
the training loop demonstrable on Bitcoin Alpha.

## What HTL gives the family

Robust STL adapted to hypergraph-native signals:

```
G(val_auc > 0.85)                   # always above the bar
F(val_auc > 0.90)                   # eventually crosses the bar
G(val_auc > 0.85) AND F(val_auc > 0.90)   # both: held floor + a peak
F(val_auc > 0.55) AND G(loss < 1.0)       # acceptance pattern
G(NOT val_auc < 0.7)                # equivalently, always not below
```

Boolean satisfaction is `rho > 0`; the numeric value is the *slack* — how
much room the formula has before it fires.  This is the standard
robust-STL semantics from Donzé–Maler / Fainekos, here implemented in
~150 LOC of evaluator code and ~200 LOC of recursive-descent parser.

The hypergraph-native part comes through the **predicate registry**.
Default scalar signals (`val_auc`, `loss`, `best_auc`) are pulled from
`event.scalar_signals` by name, but a user can register

```python
@register("balanced_fraction")
def _bal(event):
    cycles = event.hypergraph.cycles
    signs  = event.hypergraph.signs
    return (signs.prod(dim=1) == +1).float().mean().item()

@register("alpha[c5]")
def _alpha5(event):
    return event.hypergraph.alpha_slots["c5"]
```

…and the same robust-STL machinery composes those with the temporal
operators.  That is the lever Niitsuma will care about: *the same logic
fires on signed-cycle invariants as on scalar metrics*.

## Files touched

| File | Status | LOC |
| --- | --- | --- |
| `signedkan_wip/src/htl/__init__.py` | new | 52 (re-exports) |
| `signedkan_wip/src/htl/ast.py` | new | 76 (AST dataclasses, CmpOp enum) |
| `signedkan_wip/src/htl/event.py` | new | 22 (HypergraphEvent) |
| `signedkan_wip/src/htl/predicates.py` | new | 65 (registry + signal_value) |
| `signedkan_wip/src/htl/parser.py` | new | 218 (tokeniser + recursive-descent) |
| `signedkan_wip/src/htl/eval.py` | new | 174 (robustness + HtlMonitor) |
| `signedkan_wip/tests/test_htl.py` | new | 199 (18 unit tests) |
| `signedkan_wip/experiments/runs/run_gomb_smoke.py` | extended | +47 (--monitor CLI + per-epoch hook) |
| `docs/plans/2026-05-21-htl-python-impl/{plan.tex,plan.pdf,plan.tikz,plan_figure.pdf,plan.mmd}` | new | 4-format plan |
| `reports/2026-05-21-hypergraph-temporal-logic-python.md` | new | this file |

## CORE.YAML items touched

None.

## Architecture

```
formula string
   │
   ▼
parser.parse  (recursive-descent over a 5-token grammar)
   │
   ▼
HtlNode AST  (ScalarPred | Not | And | Or | Globally | Eventually)
   │
   ▼
HtlMonitor.observe(event)
   │  - append HypergraphEvent (t, scalar_signals, hypergraph)
   │  - evict on horizon (deque maxlen)
   ▼
eval.robustness(node, history, t_now)
   │
   ▼
scalar ρ      (satisfied = ρ > 0)
```

Robust-STL semantics (plan §3):

```
ρ(name >  v)              = signal(name, t) - v
ρ(name <  v)              = v - signal(name, t)
ρ(name == v)              = -|signal(name, t) - v|
ρ(Not(ψ))                 = -ρ(ψ)
ρ(And(ψ1, ψ2))            = min(ρ(ψ1), ρ(ψ2))
ρ(Or(ψ1, ψ2))             = max(ρ(ψ1), ρ(ψ2))
ρ(G_{[a,b]}(ψ), t_now)    = inf over t' ∈ [t_now − b, t_now − a]
ρ(F_{[a,b]}(ψ), t_now)    = sup over t' ∈ [t_now − b, t_now − a]
```

Interval semantics is **past-pointing** (the standard online-monitoring
convention): `G[0, inf]` = "across all observed events".  `F[0, T]` =
"in the last T units there exists a satisfying event".  Plan-style
forward-pointing intervals can be supported later by adding a flag —
deliberately not done in v0 because the training-loop use case wants
past observation, not look-ahead prediction.

## Grammar

Whitespace-insensitive recursive-descent parser:

```
formula   := or_expr
or_expr   := and_expr ("OR"  and_expr)*
and_expr  := unary    ("AND" unary)*
unary     := "NOT" unary
           | "G" interval? "(" formula ")"
           | "F" interval? "(" formula ")"
           | atom
atom      := "(" formula ")" | predicate
predicate := IDENT ("[" IDENT "]")? CMP NUMBER
interval  := "[" NUMBER "," NUMBER "]"
CMP       := "<" | "<=" | ">" | ">=" | "=="
```

Identifiers with bracketed subscripts (e.g. `alpha[c5]`) are first-class
to support the hypergraph-native predicates that index by cycle arity.

## Tests

| Suite | Result |
| --- | --- |
| `pytest signedkan_wip/tests/test_htl.py` | **18 / 18 pass** in 2.3 s |

Coverage:

- **Parser (7 tests)** — scalar pred, G-around-pred, nested G(F(x>0)),
  bracketed subscripts (`alpha[c5]`), NOT/AND/OR precedence,
  `G[a,b]` interval syntax, malformed-input rejection.
- **Evaluator (5 tests)** — scalar margin (>, <, ==), Not negation,
  And/Or as min/max, Globally as inf-over-history, Eventually as
  sup-over-history.
- **Monitor (3 tests)** — multi-epoch streaming, horizon eviction,
  rejection of out-of-order events.
- **Predicate registry (2 tests)** — custom decorated callable,
  unknown-signal KeyError.
- **Satisfaction (1 test)** — boolean from sign of ρ.

## Demo: training-loop monitor on Bitcoin Alpha

Tiny config (d_embed=16, 6 epochs, CPU), formula
`F(val_auc > 0.55) AND G(loss < 1.0)`:

```
[htl] monitor active: formula='F(val_auc > 0.55) AND G(loss < 1.0)' horizon=64
  ep 00  loss=0.6553  val_auc=0.4874  best=0.4874
  [htl] ep 00  rho=-0.0626  satisfied=N
  ep 01  loss=0.6128  val_auc=0.5527
  [htl] ep 01  rho=+0.0026  satisfied=Y   ← F flips on at 0.5527
  ep 02  loss=0.5851  val_auc=0.5695
  [htl] ep 02  rho=+0.0195  satisfied=Y
  ep 03  loss=0.5697  val_auc=0.5761
  [htl] ep 03  rho=+0.0261  satisfied=Y
  ep 04  loss=0.5576  val_auc=0.5783
  [htl] ep 04  rho=+0.0283  satisfied=Y
  ep 05  loss=0.5309  val_auc=0.5786
  [htl] ep 05  rho=+0.0286  satisfied=Y
final: rho=0.0286, satisfied=True
```

The trace captures three things at once: when the formula starts firing
(F flips at the first event satisfying `val_auc > 0.55`), how much slack
the conjunction has at each epoch (the min over the two branches), and
whether the held-globally constraint ever breaks (it doesn't here — loss
stays well under 1.0).

Counter-example with `G(val_auc > 0.85)` (always-above the SOTA-floor
bar):

```
[htl] ep 00  rho=-0.3626  satisfied=N
[htl] ep 05  rho=-0.3626  satisfied=N
final: rho=-0.3626, satisfied=False
```

`G` correctly takes the inf, so a single past violation locks the
formula at the worst-seen margin.  This is robust STL behaving as
expected.

## Acceptance check

- [x] Plan in 4 formats on disk (`docs/plans/2026-05-21-htl-python-impl/{plan.tex,plan.pdf,plan.tikz,plan_figure.pdf,plan.mmd}`).
- [x] CORE.YAML items touched = 0.
- [x] 18 / 18 unit tests pass.
- [x] `run_gomb_smoke --monitor "<formula>"` works end-to-end; per-epoch
      `rho` and `satisfied` printed alongside `val_auc`; final
      `htl_final_robustness` and full `htl_trace` in the JSON summary.
- [x] §6.5 anti-pattern audit clean (one class per node type via
      `@dataclass`, single visitor function `robustness_at`, no
      Cartesian product of operator × signal-source combinations).
- [x] Report on disk.

## §6.5 anti-pattern audit

- **No Cartesian-product API surface.** Operators (G/F/Not/And/Or) are
  AST node types; signals are resolved via the registry; comparison ops
  are an enum.  Adding a new operator means a new dataclass + one arm in
  `robustness`; adding a new signal means one `@register("name")`.
  Neither requires touching the other.
- **No string-typed config that should be an enum.** `CmpOp` is an
  enum with a `parse(token)` classmethod that runs once at parse time;
  internal evaluator code never sees the string form.
- **Algorithm code stays out of the binding boundary.** The Rust port
  is deferred; the Python module is the algorithm.  When the Rust crate
  ships, the PyO3 binding will be a thin wrapper around the same trait
  surface (`HtlNode` → `Box<dyn Robustness>`, `predicates` →
  `HashMap<String, Box<dyn Fn(&Event) -> f64>>`).

## Niitsuma-audience pitch

Hypergraph neural networks are usually deployed open-loop: train, eval,
ship.  HTL gives us a **closed-loop monitoring substrate** that
composes the same signed-cycle invariants the network is trained on
(per-cycle balance, α_k ratios, shell-dominance maps) with temporal
operators.  A trained Gömb is a *signed-cycle policy*; HTL lets you
write down a *signed-cycle specification* and check whether the policy
satisfies it at every observed training step.

The pieces are now in place:
- AST and parser → formulas as values.
- Predicate registry → hypergraph-native signals first-class.
- Robust STL → graded satisfaction, not just boolean.
- Online monitor → bounded-history, training-loop-ready.

What's deferred:
- Rust port with PyO3 (~3 days work; the Python tests double as the
  parity oracle).
- Until-operator `ψ U_{[t1, t2]} φ` (~50 LOC; not needed for v0).
- Predicate library for hypergraph state (balanced_fraction,
  alpha[ck], shell_dominance) — straightforward additions once the
  Gömb hypergraph-state surface is stable.

## Open follow-ups

1. **Rust port + PyO3.**  Same trait shape; the 18 Python tests become
   the parity oracle.  Estimated 3 days.
2. **Until operator** `U_{[t1, t2]}`.  Robust-STL semantics is the
   sup-then-inf double quantifier; ~50 LOC.
3. **Hypergraph predicate library** at
   `signedkan_wip/src/htl/predicates/hypergraph.py` (balanced_fraction,
   alpha[ck], cycle_count[k], shell_dominance).
4. **Wire the monitor into the rest of the runners**
   (`run_outer_hsikan_gomb`, `run_outer_hsikan_msg_abb_grid`,
   `run_hsikan_sota_gate`).  One-line change each.
5. **CSV/TensorBoard export** of `htl_trace` for visual inspection of
   formula firing over training.

## Experiment provenance

- **Git SHA:** uncommitted (branch `refactor/extract-hymeko-hre`).
- **Tests:** `pytest signedkan_wip/tests/test_htl.py -q` →
  `18 passed in 2.31s`.
- **Demo:** `python -m signedkan_wip.experiments.runs.run_gomb_smoke
  --dataset bitcoin_alpha --seed 0 --n-epochs 6 --device cpu
  --d-embed 16 --M-outer 4 --d-outer 8 --d-middle 12 --d-core 16
  --n-tiers 3 --k 3 --topk 16 --monitor "F(val_auc > 0.55) AND G(loss < 1.0)"`.
- **Hardware:** dev box, CPU, single seed.
- **No GPU**, no large datasets — this is a framework smoke run, not a
  training experiment.
