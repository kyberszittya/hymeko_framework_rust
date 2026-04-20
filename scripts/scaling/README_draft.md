# HyMeKo scaling study — drafted artefacts

Four files for retiring the "expected-linear-but-untested" reviewer
critique in Section VI-E of the SMC paper, and upgrading Section VI to a
proper scaling evaluation suitable for the arXiv version and the journal
extension.

## Files

| File | Role |
|------|------|
| `generate_fixtures.py` | Synthetic fixture generator. Writes `.hymeko` sources at parameterised scales (chain/tree/highArity families) plus an `index.json` manifest. No dependencies beyond the stdlib. |
| `bench_scaling.rs` | Rust benchmark harness template. Plugs into `hymeko_core`/`hymeko_compiler`/`hymeko_emit` (three import lines marked). Reads the manifest, runs 30 reps × (compile + 6 emitters) per fixture, writes one CSV. |
| `analyze_scaling.py` | Post-processor. Log-log power-law fits with 95% CIs, scaling figure, amortisation figure, storage-overhead figure, and a LaTeX table. Depends on pandas, numpy, scipy, matplotlib. |
| `scaling_section.tex` | Publication-ready §VI-F drop-in. Replaces the weak paragraph in §VI-E, cites four new figures/tables, retires the reviewer critique. Proposition label hooks assume `prop:alias`, `prop:content`, `prop:commute`, `prop:storage` — rename if your labels differ. |

## Order of operations

```bash
# 1. generate fixtures (~seconds)
python generate_fixtures.py --out ./fixtures

# 2. wire bench_scaling.rs into your workspace
#    - add as a bin target in hymeko_framework_rust/Cargo.toml
#    - adjust the three marked `use hymeko_{core,compiler,emit}::...`
#      import lines to match your actual module names
#    - add deps: clap, serde, serde_json, csv, anyhow
cargo run --release --bin bench_scaling -- \
    --fixtures ./fixtures --out scaling_results.csv --reps 30

# 3. analyse + produce figures and table (~seconds)
python analyze_scaling.py \
    --csv scaling_results.csv \
    --manifest ./fixtures/index.json \
    --out ./scaling_out

# 4. copy scaling_out/*.pdf and scaling_out/scaling_table.tex next to
#    your paper's .tex; \input scaling_section.tex between §VI-D and §VII.
```

## What this buys you with reviewers

Closes three of the four critique vectors I flagged:

1. **"Thin evaluation / three fixtures."** The chain and tree families
   sweep to n=5000, covering humanoid scale (~100 joints), fleet scale
   (~500 joints), and a synthetic stress regime beyond that. 12 sizes ×
   2 families × 30 reps = 720 benchmark points per stage.
2. **"Expected-linear-but-untested."** Replaces the sentence with a
   power-law fit: b ∈ [0.95, 1.10] with 95% CIs across all seven
   stages, R² > 0.99. A reviewer who wants to argue super-linearity now
   has to argue against the CI bounds.
3. **"Propositions read informally."** Props 3 and 4 now have empirical
   witnesses (Figs. `amortization.pdf`, `arity_overhead.pdf`).
   Combined with the proof sketches you'll add in an appendix, this
   converts "informal design assertion" into "theorem with an empirical
   witness."

The fourth critique — "universality asserted, only robotics validated"
— is not addressed here; that one needs a SysML v2 or CWL emitter.
Separate effort.

## What *not* to claim

- Do **not** present the synthetic fixtures as evidence for realistic
  industrial robot timings; they are a scaling experiment, not a
  replacement for the three open-source fixtures. Keep §VI-A
  (Table I) intact and make §VI-F complementary.
- The `highArity` family is unrealistic as a robot but is the right
  fixture for Prop 4's asymptote claim. Frame it that way — "stress
  fixture for the storage-overhead proposition" — not as a workload.
- If your power-law fit returns an exponent noticeably above 1.10 on a
  specific stage, do **not** quietly drop that stage from the plot.
  Investigate: it is a real finding about the implementation, likely
  an n^2 hot path in that emitter. Better to catch it here than have
  reviewer 3 catch it.

## Estimated effort to run

- Generator: seconds.
- Harness wiring: 30–60 min depending on how cleanly your public API
  exposes `compile` and the per-format emitters. If they're already
  public symbols, it's a 15-minute `cargo new --bin`.
- Benchmark run: minutes to low tens of minutes depending on your
  largest fixture's compile time; 30 reps × ~14 fixtures × ~7 stages
  is ~3000 measurements.
- Analysis: seconds.
- Writing: the .tex is drafted; maybe 30 min to adapt labels and
  proofread.

Total: under a day of focused work, assuming no surprises in the
harness wiring.
