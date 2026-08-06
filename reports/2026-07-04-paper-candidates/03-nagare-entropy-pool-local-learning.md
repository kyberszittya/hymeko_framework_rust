# Paper candidate 3 — NAGARE: local entropy-pool learning on hypergraph/holonomy features

**Working title:** *Local Learning on Fixed Holonomy Features: When Global Pooling Plus a Local
Readout Update Replaces Reverse-Mode Credit Assignment*
**Target venue:** Nature MI / NeurIPS if the hard discriminating tests land; workshop-tier as-is.
**Status:** consistent positive toy results with honest negative sub-results; every hard test is
still pending. This file doubles as the **consolidated NAGARE missing-point collection** requested
2026-07-04.

## Abstract seed

A local learner — fixed quaternion-holonomy vertex features, global hypergraph pooling, and a
readout updated by `W ← W + lr·gate·φ·(y−p)` with no backpropagation into features or pooling —
matches a backprop-trained counterpart on generated point-set tasks with ~47× fewer parameters and
22–33× lower forward latency, in a dependency-free Rust dataflow runtime (NAGARE, hand-coded
forward/backward pairs over SoA buffers, no autograd). A rank-6 fitted rotor/holonomy projection
gate is the first gating variant to beat both a constant gate and scalar entropy on the hardest
stress row. The claim is deliberately narrow: when structural features are already informative,
full reverse-mode credit assignment is unnecessary — and the boundary of "already informative" is
measurable.

## Central claim (as currently defensible)

> When quaternion/holonomy structural features are already informative, global entropy pooling plus
> a local output update can replace full hidden-state backpropagation on simple tasks.

(Verbatim from `reports/2026-07-02-entropy-pool-vs-backprop-hypergraph.md`.) The repo's own
consolidated report maintains an explicit **"avoid claiming"** list: "replaces backprop",
"biologically proven", "general 24× speedup", "universal accelerator". A paper draft must respect it.

## Evidence ledger

**Measured** (all single-seed, toy-scale; JSON + RSS artifacts on disk for every row):

- Local-vs-backprop: 1.000 acc parity on moons/spiral/xor; local **22.7–24.7× faster**
  (5.8–6.0 µs vs 137–143 µs), **60 vs 2,836 params**, RSS 7.59 MiB
  (`2026-07-02-entropy-pool-vs-backprop-hypergraph.md`).
- **Negative result, load-bearing but narrow:** the scalar **Shannon** entropy gate loses to a
  constant gate on **all 12** stress rows (clean/noisy/missing/few-shot) — "current entropy-gated
  update: not validated" (`2026-07-02-entropy-gate-stress-ablation.md`). Scope: one metric, one
  placement, one role — see gap 1 for the untested (metric × placement × role) space. The honest
  paper reports this cell, not a blanket verdict.
- Fitted rank-6 projection gate (228 params): ties best accuracy (0.938) and achieves best loss
  (0.2506) on the hardest row (spiral few-shot), where the fixed projector was worst (0.906);
  still loses to constant on spiral-noisy (`2026-07-02-fitted-projection-gate-holonomy-ablation.md`).
- Numerical parity vs PyTorch: max abs logit/entropy error ≤ 1.79e-7; NAGARE RSS 11.0 MiB vs
  PyTorch-tree 663 MiB (`2026-07-01-nagare-pytorch-global-pool-entropy-parity.md`).
- Chebyshev-deploy forward beats PyTorch 1.68–1.86× on shared fixtures at 5.4 vs 553 MiB RSS
  (`2026-07-01-nagare-pytorch-synthetic-cheby-compare.md`).
- Fused entropy update kernel: update-buffer bytes −75.26 %
  (`2026-07-01-nagare-holonomy-entropy-toys.md`).

**Inferred:** the unfused-path slowdown vs PyTorch (1.47–1.68×) is a kernel-fusion issue, not an
approach limitation — asserted in `2026-07-01-nagare-smaller-faster-summary.md`, and a post-fusion
parity artifact set exists (`reports/2026-07-04-nagare-fused-parity-*.json/.txt`) **without a
markdown write-up** — the first item of business below.

**Still hypothesis:** any biological claim; any accuracy or sample-efficiency win (every task
saturates at 1.000 — accuracy is a sanity check, not a discriminator); that holonomy features
*cause* the win (no matched non-holonomy control yet); generality beyond ~200-sample 2-D point sets.

## Consolidated missing points (deduplicated across all 14 NAGARE reports)

### (a) Scientific gaps — the paper-blocking ones

1. **The gating question is open — and the negative covers ONE cell of a large design space.**
   What was falsified is exactly: *Shannon entropy of the readout softmax, as a multiplicative
   scalar gate* (`0.25+H`), which shrinks as confidence grows and can only slow learning. The
   untested space is (metric × placement × role), user-flagged 2026-07-04:
   - **Metric family:** Rényi-α (α = 2 collision, α → ∞ min-entropy — tail sensitivity is tunable,
     and the α-sweep is one loop); Tsallis (non-extensive); **von Neumann / spectral entropy of the
     pooled feature covariance or the signed-Laplacian spectrum** — the natural candidate here,
     because it measures the *structure's* disorder rather than the prediction's, and connects
     directly to the frustration/holonomy invariant (candidates 1–2); relative-entropy forms
     (KL/JS to a running prior — gate on *surprise*, not absolute uncertainty); dynamical
     entropies over the training trajectory (permutation / sample / approximate entropy of the
     loss or activation series); fuzzy entropy (ties to the fuzzy-signature line).
   - **Placement:** readout update (tested) vs feature generation vs pooling weights.
   - **Role:** multiplicative gate (tested) vs homeostatic target term vs derivative/acceleration
     feedback vs class-conditional floor vs label-free inner update (the last four already
     proposed in `2026-07-02-entropy-gate-stress-ablation.md`, none run).
   The honest statement for the paper: one (Shannon, readout, gate) cell measured negative;
   the constant-gate control is now the bar every other cell must beat. A screening sweep over
   metric families at the existing 12-row stress harness is cheap (the harness exists) and should
   precede any redesign — do not conclude "entropy gating fails" from one metric.
2. **Holonomy features never shown to matter.** No matched-hidden-size comparison of holonomy vs
   non-holonomy feature paths on harder tasks. This is the NAGARE analogue of the discriminator
   toy: *does the structure carry the signal, or would any random projection do?*
3. **No task where accuracy discriminates.** All tasks saturate at 1.000. Needed: harder multi-class
   synthetics, noisy/missing hyperedges, shuffled-vs-ordered point clouds, low-data regimes with a
   sample-efficiency curve — the conditions under which "local matches backprop" can actually fail.
4. **Local rule still uses supervised error (y−p).** The delta to backprop is the *credit-assignment
   path*, not supervision. State it precisely; relate to forward-forward / feedback-alignment /
   predictive-coding literature (fresh targeted search owed — the 2026-06-29 bounded search did not
   cover the local-learning literature).
5. **Rotor/Clifford path untested under load** — bivectors identity-initialized and never move;
   the drift diagnostics have never fired on a real learned rotor.

### (b) Engineering gaps

6. **Write up the fused-parity rerun.** Artifacts exist (`2026-07-04-nagare-fused-parity-*`), no
   report. Four reports name "fuse, then re-run parity" as the open perf item — it appears to be
   done but unclaimed. Cheapest win in the whole list.
7. Native CR / Chebyshev-CR kernels not wired into the entropy/HSiKAN/FSR harnesses; parity not
   re-run against native activations.
8. Projection gate ops (`learn_projection_basis`, `project_alpha_mix`) — `project_alpha_mix.rs`
   now exists in `hymeko_nagare/src/ops/`; confirm finite-difference backward + parity tests exist
   before it is cited as a real op.
9. Sparse path (CSR/COO incidence, sparse pooling/projector) unimplemented — everything is dense
   at N=6..48; CPML routing rebuilds tier masks every forward (no cached sparse-dense matmul).
10. Rotor/FSR kernel fusion (Cayley→quaternion→rotation→signed-weighting) deferred.
11. Toolchain debt: full `cargo clippy`/`cargo fmt` blocked by pre-existing `hymeko_graph` drift.
12. Planned `nagare-holonomy-learn` sibling-repo extraction (Phases 1–5) unexecuted.

### (c) Methodology gaps

13. **Single-seed everywhere** (seeds 53/37/123, one each). §3 discipline requires ≥5-iteration
    medians/IQR for perf and multi-seed for quality claims. No error bars exist anywhere in the line.
14. Speedups are vs a "backprop-like" toy baseline, not an optimized one; PyTorch comparisons are
    sometimes non-parity architectures. A paper needs: matched-architecture PyTorch CPU baseline,
    optional GPU baseline, and an explicitly scoped speedup claim.
15. Stress conditions coarse — single magnitudes of noise/missingness/few-shot; needed: sweeps.
16. Datasets: only moons/rings/xor/spiral, ~192 train. Phase-4 of the repo report enumerates the
    honest benchmark set (blobs, noisy periodic, larger point counts, more seeds).

## On-disk artifacts

Committed at `0211128` ("NAGARE: holonomy local learning burst"). Crate: `hymeko_nagare/` with 11
ops modules (adam, catmull_rom, cayley_rotor, clifford_fir, fsr_mixer, fused_entropy_update,
linear, loss, project_alpha_mix, scatter, signed_scatter); design: deliberately **no autograd
graph** — hand-coded forward/backward pairs over SoA buffers. Fourteen reports 2026-07-01..02 with
JSON/RSS artifacts (inventory: three are synthesis docs reusing others' JSON; the rest carry their
own measurements). Unwritten: the 2026-07-04 fused-parity set.

## Suggested paper shape

Two honest papers hide in here; pick one:
- **Systems paper:** NAGARE the runtime — dependency-free Rust dataflow NN with hand-coded adjoints,
  exact PyTorch parity at 1e-7, 50–100× smaller RSS, faster at small scale. Needs gaps 6–8, 13–14.
- **Learning paper:** local learning on structural features, with the gate ablation as the story
  (including the negative entropy result — that is what makes it credible). Needs gaps 1–4, 13, 15–16.
The learning paper is the Nature-capable one; the systems paper is the safe one. Do not merge them.
