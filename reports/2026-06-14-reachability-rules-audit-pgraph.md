# Report — reachability rules: a transductive audit arm that unifies with the P-graph axioms

**Date:** 2026-06-14
**Slug:** `reachability-rules-audit-pgraph`
**Plan + argument:** `docs/plans/2026-06-14-reachability-rules-audit-pgraph/`
(`argument.md`, `plan.{tex,pdf,tikz,mmd}`, `plan-figure.pdf`)
**Author:** Csaba Hajdu
**Branch:** `feature/ac-hsikan`

## Summary

Formalizes the leakage-audit's strict / topology-only / full protocols as
**reachability rules** — predicates on which edges and labels seed a reachability
closure — and shows they are the same object the P-graph engine already evaluates
(A4/S4, E-StrictNoExcess, SSG feasibility, the ABB `close_producible` bound). One
`ReachabilityRule` enum is implemented and **tested on both sides**: the
signed-link audit (Python) and the `hymeko_pgraph` PNS engine (Rust). The
conceptual argument (`argument.md`) is written to seed a standalone article. This
delivers the user's "plan + argument + test cases"; the deeper wiring (ABB
threading; per-model topology neutral-sign handling) is scoped as follow-up.

## What was built

- **`argument.md`** — the unification: reachability is already axiomatic here
  (A4, E-StrictNoExcess, `close_producible`); the audit protocols are reachability
  rules forming a monotone lattice `strict ⊆ topo ⊆ full`; leakage = the held-out
  label reachable at the readout; the same rule is a synthesis regime alongside
  S1–S5 (the `axiom_extensions.rs` precedent). Includes the two soundness
  obligations (reduction; admissibility) and the empirical refinement (§5a).
- **`plan.{tex,pdf,tikz,mmd}`** — phased implementation plan (Phase 1 audit arm,
  Phase 2 pgraph regime), all four formats compile.
- **`hymeko_neuro/baselines/reachability.py`** — `ReachabilityRule` enum +
  `reachable_edges` / `reachable_nodes`; the audit-side rule semantics.
- **`hymeko_neuro/experiments/runs/run_baseline_audit.py`** — `reachability`
  param + `--reachability {strict,topo,full}`; default `strict` keeps every call
  bit-identical (reduction).
- **`hymeko_pgraph/src/reachability.rs`** — `ReachabilityRule` +
  `close_producible_under_rule`; the synthesis-side rule, reusing
  `msg::close_producible`. Registered in `lib.rs`.

## Test cases (the requested deliverable)

| layer | file | tests | what they pin |
|---|---|---|---|
| audit semantics (Python) | `tests/test_reachability.py` | 8 | reduction (strict=train), edge/node monotone lattice, **leakage invariant** (test signs reachable only under `full`), topology sign-masking, parse/failure/determinism |
| pgraph soundness (Rust) | `reachability.rs::tests` | 4 | **reduction** (`Strict` == canonical `close_producible`), candidate-unlocks-product, **monotone** closure (`strict ⊆ topo == full`), cost/admissibility flag table |
| audit reduction (regression) | `tests/test_baseline_audit.py` | 14 | unchanged — `strict` default keeps the baselines bit-identical |

All green: **8 + 4 + 14 = 26** (plus the 5 vectorized-attention + 10 SMC tests
from earlier today still pass). `ruff` clean; `cargo clippy` clean on the new
module (one pre-existing unused-import warning in the unrelated
`hymeko_pgraph_dump` bin, not introduced here); `rustfmt --edition 2024` clean.

## Empirical finding (refines the thesis — §11 honesty)

Leak demo (sgt, bitcoin_alpha, 60 ep, CPU), real vs shuffled per rule:

| rule | real | shuffled | drop | verdict |
|---|---|---|---|---|
| strict | 0.906 | 0.520 | 38.6 pp | clean |
| **full** | 0.905 | 0.514 | 39.0 pp | **clean (no leak!)** |

Admitting the test sign into the message-passing graph (`full`) does **not** leak
for sgt — contrast the cycle-based path (2026-06-11) that leaked at 0.73 shuffled.
**Resolution:** reachability of the label is *necessary but not sufficient*;
sufficiency needs the **readout** to expose it. Diffuse node-embedding readouts
(the whole SGCN/SiGAT/SGT/SGCL/SiGformer/SE-SGformer family) wash the test sign
out before the per-edge logit; per-edge cycle/path features expose it and leak. So
leakage factors as **reachability rule × readout locality** — a two-factor account
strictly more informative than "transductive ⇒ leak," and a candidate headline for
the standalone article. `argument.md` §5a records this.

## CORE.YAML items touched

**None.** `CORE.YAML` lists only `docs/spec/g_sphf_axioms.tex` (GGK K1–K4) — not
`hymeko_pgraph`, not `hymeko_neuro`. No dependency added. Additive throughout
(new enum, new module, new CLI flag, default-`strict` preserves behaviour),
mirroring the additive `axiom_extensions.rs` precedent.

## Files touched

| Path | Action | Lines |
|---|---|---|
| `docs/plans/2026-06-14-reachability-rules-audit-pgraph/argument.md` | new | — |
| `docs/plans/.../plan.{tex,tikz,mmd}` (+pdfs) | new | — |
| `hymeko_neuro/baselines/reachability.py` | new | 99 |
| `hymeko_neuro/tests/test_reachability.py` | new | 104 |
| `hymeko_neuro/experiments/runs/run_baseline_audit.py` | modify (rule param + CLI) | +~20 |
| `hymeko_pgraph/src/reachability.rs` | new | 188 |
| `hymeko_pgraph/src/lib.rs` | modify (+1 `pub mod`) | +1 |

## Open issues / follow-ups

1. **Phase 2 ABB threading** — `close_producible_under_rule` is implemented and
   tested; wiring it into the ABB reachability bound + a `ReachabilityRegime`
   (the `regime.rs` Strategy seam) with the admissibility test (ABB optimum
   invariant across rules) is the next step.
2. **Per-model `topo` wiring** — the `topo` neutral-sign edge needs per-model
   handling to enter signed-split adjacencies (it currently masks to a neutral
   token at the rule layer; SGCN-style pos/neg split drops a 0-sign edge). The
   `R_topo` audit cell needs this before it measures structural leakage.
3. **Article** — `argument.md` is structured as the formal core; the two-factor
   (rule × readout-locality) account, the lattice, and the soundness theorems are
   the contribution. Needs the `R_topo` empirical sweep across the readout
   families to complete the experimental section.

## Provenance

Git SHA: working tree dirty. Host: Windows 11, cargo 1.93.1, torch 2.12.0+cu132,
Python 3.12.13. Leak demo: sgt, bitcoin_alpha, seed 0, 60 ep, CPU
(`CUDA_VISIBLE_DEVICES=""`, to avoid contending the concurrent 5-seed grid).
