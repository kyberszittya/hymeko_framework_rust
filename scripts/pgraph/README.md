# P-graph examples & canonical-correctness checks

Run the Friedler / Orosz / Pimentel Losada (*P-graphs for Process Systems
Engineering*, Springer) worked examples through the HyMeKo engine and verify that
**MSG** (maximal structure) and **ABB** (cost-optimal solution) are *canonically
correct* — i.e. reproduce the book's published results.

Strategy and acceptance criteria: `docs/plans/2026-05-27-msg-abb-verification/`.
Background fixes: `reports/2026-05-27-msg-abb-canonical-fix.md`,
`reports/2026-05-27-pgraph-regime-strategy.md`.

## Prerequisites

- **Rust toolchain** (`cargo`) — builds the `hymeko_pgraph_dump` CLI and runs the tests.
- **Python ≥ 3.8** — *standard library only* (`subprocess`, `json`). **No torch/numpy/uv** needed; the engine is pure Rust.

## Files

| file | what it does |
|------|--------------|
| `run_examples.py` | Runs each book example via the CLI, prints MSG/ABB results, and checks them against the published canonical values (exit 0 iff all match). `--regimes` also shows regime effects. |
| `verify.sh` / `verify.ps1` | Full check: the Rust conformance suite (`cargo test -p hymeko_pgraph`) **and** the book-example CLI conformance. Use `.sh` on bash/WSL, `.ps1` on Windows PowerShell. |

## Quick start

```bash
# from the repo root
cargo build -p hymeko_pgraph --bin hymeko_pgraph_dump   # or: run_examples.py --build

python scripts/pgraph/run_examples.py            # book conformance table
python scripts/pgraph/run_examples.py --regimes  # + regime comparison

# full verification (tests + examples):
./scripts/pgraph/verify.sh          # bash / WSL
#   or, on Windows PowerShell:
#   .\scripts\pgraph\verify.ps1
```

### Expected `run_examples.py` output (canonical)

```
example                       MSG  exp   ABB cost      exp  ok
Chapter3/example3_2.hymeko      7    7        0.0      0.0  OK
Chapter4/example4_1.hymeko      7    7       13.0     13.0  OK
Chapter4/example4_3.hymeko     29   29        0.0      0.0  OK
Chapter5/example5_1.hymeko      6    6        0.0      0.0  OK
Chapter6/example6_1.hymeko      7    7        9.0      9.0  OK
book/example14_1.hymeko        12   12       16.0     16.0  OK
hda.hymeko                      3    3      350.0    350.0  OK
methanol_synthesis.hymeko       8    8     2940.0   2940.0  OK
ALL CANONICAL VALUES MATCH
```

## The examples

| fixture | book | maximal structure | ABB optimum |
|---------|------|-------------------|-------------|
| `Chapter3/example3_2` | Ex 3.2 | 7 units (**19 solution-structures**) | structural (no costs) |
| `Chapter4/example4_1` | Ex 4.1 | 7 units `{u2,u3,u4,u5,u6,u8,u10}` | `{u2,u4,u8}` = 13 |
| `Chapter4/example4_3` | Ex 3.3 | 29 units (**3465 solution-structures**) | structural |
| `Chapter5/example5_1` | Ex 5.1 | 6 units | structural |
| `Chapter6/example6_1` | Ex 6.1 | 7 units | `{O2,O5,O7}` = 9 |
| `book/example14_1` | Ex 14.1 | 12 units | `{u1,u4,u8,u11}` = 16 |
| `hda` | (HDA, hand) | 3 units (Disposal sink pruned) | `{Mixer,Reactor}` = 350 |
| `methanol_synthesis` | multi-objective | 8 units | scalar 2940 |

> The **19** and **3465** *solution-structure* counts come from the recursive
> decision-mapping SSG and are asserted in the Rust suite
> (`cargo test -p hymeko_pgraph --test ssg_decision_mapping` and
> `--test book_validation`) — not via this CLI, whose brute SSG cannot enumerate
> 2²⁹ subsets.

## Direct CLI use

```bash
target/debug/hymeko_pgraph_dump <file>.hymeko --algorithm msg|ssg|abb [--regime SPEC]
```

`--regime SPEC` selects the solving regime — one or more of
`canonical | no-excess | cost-dominance`, joined with `+`
(e.g. `--regime cost-dominance+no-excess`):

- **canonical** *(default)* — the textbook semantics (axioms S1–S5; no no-excess rule).
- **no-excess** — non-canonical "no-waste" filter (used by the HSiKAN architecture sweeps).
- **cost-dominance** — optimum-preserving reduction (prunes dearer interchangeable units).
- composing several stacks them on the canonical base.

## What the verification covers

`cargo test -p hymeko_pgraph` exercises the canonical-correctness suite:

- `book_validation` — the published MSG/SSG/ABB values (the spine).
- `axiom_witness` — combinatorial axioms S1–S5 on the maximal structure + ABB sub-schema.
- `ssg_decision_mapping` — exact solution-structure counts (19, 3465) + the Example 14.1 optimum.
- `relaxed_msg`, `pgraph_e2e` — MSG/ABB on the chapter examples and HDA.
- regime unit tests — Canonical / NoExcess / CostDominance / Composite.

**Out of scope here** (documented in the verification plan): the RTL golden
SystemVerilog harness (separate testbench), the strict-no-excess *non-canonical*
opt-in, operable top-k (`Reachable` refinement), and triton/`torch.compile`
paths (Linux-only).
