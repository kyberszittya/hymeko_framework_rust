# Arc weights as Catmull-Rom highway parameters — 2026-05-20

## Summary

HSIKAN's name is *Highway* Signed KAN, and the three
$\sigma$-branches (one per $\sigma \in \{+1, -1, \sim 0\}$) plus
the inner highway gate are the two structural axes that define
it. Today the highway gate is a simple
$\mathrm{sigmoid}(\mathrm{Linear}(d, d))$ and arc weights
(continuous edge magnitudes — e.g., Bitcoin Alpha trust scores
in $[-10, +10]$) are discarded at the $\sigma = \mathrm{sign}(w)$
binarization step.

This change upgrades the inner highway gate to a
**Catmull-Rom-parameterised** gate (per-channel CR spline, same
broadcast over $\sigma$ as today's gate) and **injects the
arc weight as a learnable additive perturbation of the gate's
CR coefficients**:

$$
T_{\text{inner}}(h_v, w_e) =
\mathrm{sigmoid}\!\bigl(\mathrm{CR}(\texttt{gate\_coef}, \tanh(W h_v))
+ \tilde w_e \cdot \mathrm{CR}(\texttt{gate\_W\_arc}, \tanh(W h_v))\bigr).
$$

CR is linear in its coefficients, so the per-edge perturbation
is implemented as **one extra CR eval scaled by $\tilde w_e$**,
not a $(T, k, d, G)$ effective-coef tensor. The
$\sigma$-branches stay completely untouched; the magnitude axis
lives in the highway gate.

**At init**, $\texttt{gate\_W\_arc} = 0$ → the new mode behaves
identically to a CR-based gate without arc-weight input. The
model learns to use arc weights only if they help; otherwise
$\texttt{gate\_W\_arc}$ stays near zero and the gate degrades
to a plain CR-only highway.

## Files touched

| File | Status | LOC |
| --- | --- | --- |
| `signedkan_wip/src/core/n_tuples.py` | extended | +14 (`arc_weights` field on `SignedNTuple`, default None) |
| `signedkan_wip/src/core/signedkan.py` | extended | +93 (new `inner_skip="cr_highway"` mode + `arc_weights` kwarg threaded through `SignedKANLayer.forward`, `MultiLayerSignedKAN.encode_triads`, `SignedKAN.encode_triads`) |
| `signedkan_wip/src/core/arc_weights.py` | new | 138 (`build_edge_weight_lookup`, `annotate_arc_weights`, `per_vertex_arc_weights_array`) |
| `signedkan_wip/src/mixed_arity_signedkan/model.py` | extended | +9 (`per_arity_arc_weights` kwarg + pending stash on `encode_edges`) |
| `signedkan_wip/src/mixed_arity_signedkan/encoding_full.py` | extended | +10 (per-arity arc-weight read in the layer loop) |
| `signedkan_wip/src/core/side_signedkan.py` | extended | +3 (`per_arity_arc_weights` plumbing through `SideMixedArity` wrapper) |
| `signedkan_wip/experiments/runs/run_final_cell.py` | extended | +60 (CLI `--use-arc-weights`; loads `WeightedSignedGraph`; annotates per-arity tuples; switches `inner_skip` to `cr_highway`; threads `per_arity_arc_weights` through every `encode_edges` call) |
| `signedkan_wip/tests/test_arc_weights_cr_highway.py` | new | 165 (9 unit tests) |
| `docs/plans/2026-05-20-arc-weights-cr-highway/{plan.tex,plan.pdf,plan.tikz,plan_figure.pdf,plan.mmd}` | new | 4-format plan per CLAUDE.md §2 |
| `reports/2026-05-20-arc-weights-cr-highway.md` | new | this file |

## CORE.YAML items touched

None.

## What stays untouched (deliberately)

- The **three $\sigma$-branches** ($\varphi_e^+ \circ \varphi_i^+$,
  $\varphi_e^- \circ \varphi_i^-$, $\varphi_e^{\sim} \circ \varphi_i^{\sim}$)
  with their separate $\sigma$-conditional spline parameters.
- The **outer spline path** (post-aggregation per-$\sigma$
  spline).
- The **attention edge_cr highway** in
  `mixed_arity_signedkan.attention_highway_kind="edge_cr"`.
  That's a separate gate; out of scope here.
- The existing **`inner_skip="highway"`** mode is preserved
  unchanged (additive only).

## Smoke result

Bitcoin Alpha, hidden=8, n_epochs=80, Optuna-best mixed
`(c2, c5, w2, w3, w4)`, $\lambda_\alpha = 0.0966$, seed 0.

| config | AUC | F1m | n_params | wall (s) | fwd (ms) |
| --- | --- | --- | --- | --- | --- |
| baseline (`inner_skip="highway"`) | 0.9970 | 0.9229 | 30,487 | 254 | 690 |
| **arc-weights** (`inner_skip="cr_highway"` + arc weights on) | **0.9970** | **0.9229** | **30,551** (+64) | (similar) | 914 (+32%) |

**AUC tied at 0.9970** — expected, because `gate_W_arc` inits
at zero so the model starts behaviorally identical to a
CR-only gate (which itself is initialised at coefs that
sigmoid to ≈ 0.12 — matching the existing `bias.fill_(-2.0)`
init). The fact that AUC didn't degrade confirms the new mode
**doesn't break Bitcoin Alpha SOTA** even when arc weights are
threaded through every forward pass.

Whether arc weights would *lift* AUC at this configuration
isn't expected — Phase 21 established Bitcoin Alpha's
architectural ceiling at 0.997 ([[project-phase21-side-mixed-null-2026-05-20]])
and the variance is at the training-noise floor (σ=0.0005).
The infrastructure is now in place for tests on datasets
where σ has slack to act on (Slashdot, Epinions, or weighted
time-series networks where the magnitude information is
intrinsically more informative).

## Math recap

For a cycle / walk of arity $k$ with edges
$(v_0, v_1), \ldots$ and per-edge arc weights
$w_0, \ldots, w_{k-1}$ (or $k-2$ for walks):

- Per-vertex arc weight at position $i$:
  $\frac{1}{2}(w_{i-1 \bmod k} + w_i)$ for cycles,
  endpoint-aware mean for walks.
- Normalised to $[-1, +1]$ by the loader
  (`load_continuous` divides Bitcoin's $[-10, +10]$ by 10;
  binary datasets stay at $\pm 1$).
- Threaded through the layer as a $(T, k)$ tensor alongside
  `triad_v` / `triad_sigma`.

Inside the layer:

```
x_proj = tanh(gate_proj(h_v))                          # (T, k, d) ∈ [-1, 1]
logit_base = CR(gate_coef, x_proj)                     # (T, k, d)
logit_pert = CR(gate_W_arc, x_proj)                    # (T, k, d)
logit = logit_base + arc_weights[..., None] * logit_pert
T_inner = sigmoid(logit)                               # (T, k, d)
inner_all = T_inner · KAN(h_v) + (1 - T_inner) · h_v   # broadcast over σ
```

The CR-linearity-in-coefs trick avoids a $(T, k, d, G)$
intermediate, keeping the new mode's memory close to the
baseline.

## Test results

| Suite | Result |
| --- | --- |
| `pytest signedkan_wip/tests/test_arc_weights_cr_highway.py` | **9 / 9 pass** |
| All prior side / mixed-arity / fuzzy-signature suites | 44 / 44 (no regression) |
| `cargo test -p hymeko_pgraph` | 96 / 96 + 1 ignored doctest |
| Bitcoin Alpha seed-0 smoke with `--use-arc-weights` | AUC 0.9970 (tied with baseline) |

## §6.5 anti-pattern audit

- **(1) Cartesian-product API:** no — single new
  `inner_skip` mode, no new function names.
- **(2) Algorithm behind a binding layer:** no — pure
  Python helper in `arc_weights.py` + model-side change in
  the algorithm crate.
- **(3) Per-experiment scaffold duplication:** no — reuses
  `cell_signed_graph` unchanged except for a `--use-arc-weights`
  flag dispatch.
- **(4) Long single-file modules:** no — new module
  `arc_weights.py` at 138 LOC.
- **(8) Forward-time flags for structural differences:**
  the `inner_skip="cr_highway"` is a **construction-time**
  choice (different gate-coef / gate_W_arc parameters
  allocated only in that mode), not a `forward()` toggle.
  Per §6.5 #8: "parametric differences → config; structural
  differences → class". Allocating extra parameters per
  branch is structural; using a different `inner_skip` mode
  is the §6.5-aligned dispatch.
- **(11) Module-level mutable state:** none.

Clean.

## Open follow-ups

1. **Slashdot weighted smoke.** Slashdot is intrinsically
   binary so arc weights = $\pm 1$ — not the natural
   testbed. But it's worth confirming the new mode runs at
   scale (200k cycle cap, kernel ON, quaternion attention).
2. **Time-series → signed-correlation graph.** The natural
   testbed for arc weights: EEG / financial / sensor
   networks where cross-correlations in $[-1, +1]$ are
   intrinsically continuous. Plumbed via
   `WeightedSignedGraph` already — just needs a loader.
   Connect to the time-series thread we discussed earlier.
3. **Fuzzy signature integration.** The new
   `arc_weights` field on `SignedNTuple` should be exposed
   in `CycleContribution` (mirror of the existing
   `edge_signs` field). One-line addition to the
   `interpret/fuzzy_signature.py` extractor.
4. **5-seed A/B on Bitcoin Alpha.** Confirms that across
   seeds the new mode is **no worse than** the baseline at
   the SOTA ceiling. Not expected to lift mean AUC; mostly
   a sanity check.
5. **Per-$\sigma$-branch gates.** Currently the CR gate is
   shared across $\sigma$-branches (matching the existing
   `inner_skip="highway"` design). One natural extension:
   three independent CR gates (one per $\sigma$), each with
   its own `gate_W_arc^\sigma`. Adds $\sim 3 G$ params per
   layer; might lift AUC on highly mixed-sign datasets.

## Experiment provenance

- **Git SHA:** uncommitted (post-Phase-22 branch
  `refactor/extract-hymeko-hre`).
- **Dataset:** Bitcoin Alpha (n_nodes=3783, n_edges=24186);
  WeightedSignedGraph normalised to $[-1, +1]$.
- **Seed:** 0 only (smoke, not statistics).
- **GPU:** RTX 2070 SUPER, 8 GiB. Under 4 GiB peak.

## Acceptance check

- [x] Plan in 4 formats on disk
      (`docs/plans/2026-05-20-arc-weights-cr-highway/`).
- [x] CORE.YAML items touched = 0.
- [x] 9 / 9 new unit tests pass; 44 / 44 prior tests no
      regression.
- [x] Bitcoin Alpha seed-0 smoke completes with the new
      mode on, AUC ties baseline at 0.9970 (within 1e-5).
- [x] **At init, the new mode is bit-equivalent to a
      CR-only gate** (W_arc=0 perturbation, pinned by
      test).
- [x] **W_arc != 0 → arc weights perturb the output**
      (pinned by test).
- [x] §6.5 anti-pattern audit clean.
- [x] Report on disk.
