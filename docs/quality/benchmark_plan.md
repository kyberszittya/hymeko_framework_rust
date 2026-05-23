# Entropy-feedback benchmark plan

**Status:** scaffold (b) landed 2026-04-21 at `python/benches/hotswap/run_benchmark.py`. This doc plans the path from here to a real research claim.

**Goal.** Answer the honest question: **does entropy-driven structural rewriting improve learning?** The current scaffold answers only a subset of that — it measures whether the *plumbing* preserves signal, not whether the entropy signal itself drives better representations.

---

## What the three benchmark shapes measure

| Shape | What it measures | Requires | Meaningful? |
|---|---|---|---|
| **(a) Stub-infrastructure** | Correctness of the hot-swap machinery on placeholder layers | Only `ehk_torch_stub` (current) | Yes — catches regressions, but can't validate the research claim |
| **(b) Scaffold on real-but-simple MLP** | Whether weight-transfer + mid-training architecture change preserves performance relative to from-scratch baselines | `torch` + the current stub | Yes — engineering question with a clear answer |
| **(c) Real `ehk_torch` on realistic task** | Whether the signed-incidence / GGK / entropy mechanism drives better representations than a dense MLP of matched capacity | The full `ehk_torch` package (signed sparse ops, GGK kernel, SignedKAN activation) | Yes — this is the research claim |

Shape (b) — `python/benches/hotswap/run_benchmark.py` — landed this session. Shapes (a) and (c) are below.

---

## Current findings (b, 2026-04-21)

Four conditions, 5 seeds, synthetic regression (ℝ³→ℝ², 200 epochs, swap-at=100):

| Condition | Final val loss | ± std | Transfer keys |
|---|---:|---:|---:|
| `baseline_small` (hidden=5) | 0.1076 | 0.0505 | — |
| `hotswap_same` (5→5 swap)   | 0.0854 | 0.0296 | **4 / 4** |
| `hotswap_widen` (5→8 swap)  | 0.1040 | 0.0532 | **1 / 4** |
| `baseline_large` (hidden=8) | 0.0633 | 0.0237 | — |

**Key observations:**
- The plumbing is correct: `hotswap_same` transfers 4/4 tensors when shape doesn't change.
- With the spec's *compatible-subset-only* rule, a shape-changing swap (`hotswap_widen`) wipes 3/4 tensors — only the shape-invariant output bias `layer_1.bias` (2,) carries over.
- `hotswap_widen` closes only **8% of the gap** between `baseline_small` and `baseline_large`. That's accurate behaviour under the current rule, not a bug — most of the useful structure is in the hidden-width-dependent tensors.
- `hotswap_same` beats `baseline_small` (0.085 vs 0.108). The optimizer rebuild at the swap acts like a learning-rate restart (SGDR-style) — noise, not signal.

**Honest interpretation.** The plumbing works. The research claim hasn't been tested.

---

## Plan (a) — Stub-infrastructure benchmarks

Purpose: regression-test the hot-swap machinery without requiring `torch`. Fast, deterministic, runnable in CI. Lives under `python/ehk_torch_stub/tests/bench_*.py` as pytest fixtures, not a full benchmark suite.

**What to measure:**

1. **Transfer fidelity under renames.** Assert that `transfer_compatible_weights` handles common layer-name patterns produced by the `torch_dataflow` emitter: `layer_<decl_name>.linear.weight`, nested sub-modules, `register_buffer`-style non-parameter tensors. Today the test suite only exercises vanilla `nn.Linear` and `HypergraphConv`.
2. **Proposal round-trip correctness.** Take the JSON from `hymeko rewrite --json` on every fixture under `data/` and `data/nn/`, load via `load_proposal`, assert every decl name in the JSON resolves to a real vertex/edge in the original source. Catches decl-name drift between Rust and Python.
3. **Deterministic weight init.** Assert that `reinfer_structure_and_rebuild(old, factory)` with the same seed + factory produces byte-identical new-model parameters across calls. Foundation for reproducible hot-swap experiments.
4. **Proposal pathological inputs.** Empty cluster B, zero cross edges, 100% cross edges, n_cross_edges == total edges (nothing is splittable). Assert the loader and weight-transfer still return valid records.

**Estimated effort:** 4–6 hours. No new dependencies. Runs in CI under 1 second.

**Deliverable:** `python/ehk_torch_stub/tests/test_infra_bench.py` (pytest) + a line item in the repo's README under Testing.

---

## Plan (c) — Real `ehk_torch` benchmarks

This is the **research-claim benchmark**. Requires real signed-incidence hypergraph layers (§4 of `input/entropy_hypergraph_pytorch_spec.md`) which don't exist in this workspace yet.

**Prerequisites** (in order):
1. `ehk_torch.ops.sparse_signed` — signed sparse matmul with correct autograd.
2. `ehk_torch.kernels.ggk` — B-spline + RBF basis evaluators.
3. `ehk_torch.layers.hypergraph_conv` — the full forward pass (§4.3 of the spec).
4. `ehk_torch.construction.structural_min` — entropy-minimisation hypergraph construction.

None of these exist; all four are scope for the `ehk_torch` package proper. Each is a week+ of implementation + testing.

**Once those exist, the benchmark shape:**

- **Task:** one of the §7 integration examples — ShapeNet part segmentation (public) or cattle-corridor camera placement (user has infrastructure). Pick the smaller/faster one first.
- **Arms:**
  - `mlp_matched_capacity` — plain MLP with parameter count matched to the hypergraph model.
  - `hypergraph_fixed` — `HypergraphConv` + `SignedKAN` with hypergraph built by MI-threshold at start, never updated.
  - `hypergraph_entropy_hotswap` — starts like `hypergraph_fixed`, fires `reinfer_structure_and_rebuild` every N epochs when structural entropy exceeds a threshold.
- **Metrics:** final task metric (IoU for segmentation, mAP for detection), parameter count, wall-clock training time, number of hot-swap events, structural entropy trajectory.
- **Seeds:** at least 5; ideally 10. Paper-quality requires confidence intervals.
- **Controls:** ablation dropping one stage at a time (no entropy trigger, no MI construction, no SignedKAN activation).

**Acceptance criterion** — hypergraph_entropy_hotswap must beat `hypergraph_fixed` on the task metric with statistical significance (paired t-test, p < 0.05) across seeds. If it doesn't, the research claim doesn't survive contact with data — that's the real answer.

**Estimated effort:** 2–3 weeks after all four prerequisites land. ~1 week for each prerequisite means the full benchmark is ~6–8 weeks of focused work from today.

---

## Parallel: extending (b) to be more informative

While (a) and (c) are in flight, there are cheap wins that make the current (b) benchmark sharper:

1. **Partial-row/col transfer.** Extend `transfer_compatible_weights` with an opt-in mode that copies the first `min(old_dim, new_dim)` rows/cols of shape-mismatched tensors. This is the real "compatible-subset" rule the spec's §5.4 implies ("layers whose shapes match carry over") — the current strict-shape-match is one interpretation, but the other is "carry over what fits". Needs a test that a widened `layer_0.weight (5,3) → (8,3)` copies the first 5 rows verbatim.
2. **Optimizer state transfer.** Currently Adam state resets at the swap. Hugging Face and other libraries preserve optimizer state across reshapes for the compatible subset. Adding this would make `hotswap_same` trajectory-identical to `baseline_small`, making the plumbing check a proper equality rather than a key-count.
3. **Harder task.** The current synthetic regression is trivial enough that baseline_small gets within 2σ of baseline_large. A task where capacity matters more (MNIST, ShapeNet subset) gives more headroom for the hot-swap to show signal. Adds a data dependency but unblocks informative numbers.
4. **Longer training + swap timing sweep.** Sweep `swap_at` over {25, 50, 100, 150} to see when mid-training rewrites are most and least disruptive.

Items 1 and 2 are cheap (each ~2–3 hours). Items 3 and 4 are medium (each ~1 day including data wrangling).

---

## Recommended sequencing

1. **Now.** (b) scaffold landed. Ship.
2. **Next few hours.** (a) plan implemented — infrastructure benchmarks in pytest. Cheap, unblocks CI-level regression safety.
3. **Next week (optional, high-value).** Partial-row/col transfer (item 1 above). Makes the shape-change hotswap meaningful at negligible cost; turns the current 8%-gap-closure into a real number.
4. **Parallel track.** Start implementing `ehk_torch.ops.sparse_signed` and `ehk_torch.kernels.ggk`. These are the longest-pole for (c).
5. **When (c) prerequisites land.** Run the real benchmark on a small task. Publish or revise the research claim based on what the data says.

---

## Meta: honesty about scope

This codebase has infrastructure for entropy hot-swap but no evidence it improves learning. The current benchmark correctly reports that. Shipping (b) with this interpretation is better than shipping (a) with manufactured results on placeholder math — future readers will thank us for the honesty.

When (c) eventually runs and produces numbers, those numbers will mean something precisely because we didn't claim meaning at (b).
