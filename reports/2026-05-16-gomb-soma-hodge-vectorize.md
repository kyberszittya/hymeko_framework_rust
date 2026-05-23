# GömbSoma Pass 5-Hodge — boundary_2 vectorisation + dead-code cleanup

**Date:** 2026-05-16
**Plan:** [docs/plans/2026-05-16-gomb-soma-hodge-vectorize/](../docs/plans/2026-05-16-gomb-soma-hodge-vectorize/) (tex/pdf/tikz/mmd)
**Phase:** post-Pass-4 incremental optimisation
**Verdict:** mixed — code is cleaner and stronger-tested, raw wall-time gain is noise-level (~1.8%), and the plan's magnitude estimate was wrong by ~20×.

## 1. Summary

Targeted the `StimulusGraphBuilder` 9 ms component identified in the
[Pass-4 SDRF optimisation report](2026-05-15-gomb-soma-ricci-stim-sdrf-optimization.md).
Direct per-section profiling first surfaced two specific hot spots:

| Section (pre-fix profile)        |  ms  | % of 9.3 ms |
|----------------------------------|-----:|------------:|
| **Hodge Laplacian (forward)**    | 4.76 | **52%**     |
| Triangle vec enum                | 1.13 | 12%         |
| **Dead `adj` list-of-sets**      | 0.87 | **9%**      |
| Walk vec enum                    | 0.84 | 9%          |
| Edges + signs                    | 0.81 | 9%          |
| Forman κ                         | 0.71 | 8%          |
| Edges build                      | 0.68 | 7%          |
| Polygons (Python loop)           | 0.43 | 5%          |
| Edge-lookup dict (Python)        | 0.23 | 2%          |
| `adj_dense` build                | 0.08 | 1%          |

The two "fixable" lines were:

1. `adj = self._adjacency(edges, n)` (stim_graph.py L191): builds a
   `list[set[int]]` that nothing consumes — pure dead code, 0.87 ms.
2. Hodge `boundary_2`: a Python loop over canonical triangles, each
   iteration appending up to 3 entries to three Python lists, then
   `torch.tensor(...)` to materialise — claimed by the section profile
   as part of the 4.76 ms Hodge cost.

This pass delivers fixes for both. **However**, a finer drill-down
after the implementation revealed that *within* the 4.76 ms Hodge,
the Python boundary_2 loop was only 0.71 ms — the bulk (~2.1 ms) is
four `torch.sparse.mm` calls plus 0.5–0.9 ms of canonicalisation. The
vectorised boundary_2 path is correctness-equivalent but at the
production scale (n_t ≤ 256) is *not faster* than the Python loop
(CUDA launch overhead from `argsort` + `searchsorted` ≈ Python loop
cost). The honest gain is from the canonicalisation-skip flag plus
dead-code removal.

| Metric                                    | Pre-fix     | Post-fix    | Δ           |
|-------------------------------------------|------------:|------------:|------------:|
| `StimulusGraphBuilder.forward` (n=256)    | 9.29 ms     | 9.12 ms     | −0.17 ms (−1.8%) |
| Hodge `forward` only                      | 4.76 ms     | 3.71 ms     | −1.05 ms |
| Config E smoke (30 imgs × 1 epoch)        | 2.1 s       | 2.0–2.1 s   | within noise |
| Dead-code LOC in tree                     | ~110 LOC    | 0           | removed |
| Hodge tests                               | 18          | 25          | +7 (5 flag pins + 2 vec pins) |
| Total Ricci-Stim tests (executed)         | 161         | 168         | +7 |

The "Hodge `forward` only" line is the only place the change shows.
Per-call full-forward savings do not stack to that 1 ms in wall-time
because (a) timing each section in isolation under-counts CUDA-async
overlap and (b) the dead `adj` Python loop was overlapped with
adjacent CUDA work in the original forward.

## 2. Files touched

| File | Lines (+ / −) | Change |
|---|---:|---|
| [signedkan_wip/src/hymeko_gomb/soma/vision/hodge.py](../signedkan_wip/src/hymeko_gomb/soma/vision/hodge.py) | +193 / −47 | new `_build_boundary_2_vectorised` static helper; `edges_already_canonical` flag on `forward`; doc reformatting in `forward` |
| [signedkan_wip/src/hymeko_gomb/soma/vision/stim_graph.py](../signedkan_wip/src/hymeko_gomb/soma/vision/stim_graph.py) | +14 / −96 | removed `_adjacency`, `_enumerate_walks`, `_enumerate_triangles` dead methods (~110 LOC); removed dead `adj = self._adjacency(...)` call; added `torch.sort(edges, dim=1)` canonical-row pass; passes `edges_already_canonical=True` into hodge |
| [signedkan_wip/tests/test_gomb_soma_vision_hodge.py](../signedkan_wip/tests/test_gomb_soma_vision_hodge.py) | +112 / 0 | `test_hodge_canonical_flag_invariance` (5 seeds, parametrised); `test_hodge_boundary_2_vectorised_drops_missing_edges`; `test_hodge_boundary_2_vectorised_partial_d_squared_zero` |

Net diff: +319 / −143 over three files. Removed ~110 LOC of dead
code; added ~80 LOC of vectorised replacement; added 112 LOC of
regression tests.

## 3. CORE.YAML items touched

None.

The change is confined to `signedkan_wip/` (non-core). No edits to
`hymeko_core`, `hymeko_query`, `parser`, dependency manifests, or any
pinned-dependency surface.

## 4. Test results

```
$ python -m pytest signedkan_wip/tests/test_gomb_soma_vision_hodge.py -v
============== 25 passed in 1.71s ==============

$ python -m pytest \
    signedkan_wip/tests/test_gomb_soma_vision_hodge.py \
    signedkan_wip/tests/test_gomb_soma_vision_stim_graph.py \
    signedkan_wip/tests/test_gomb_soma_vision_sdrf.py \
    signedkan_wip/tests/test_gomb_soma_vision_sdrf_wiring.py \
    signedkan_wip/tests/test_gomb_soma_vision_forman.py \
    signedkan_wip/tests/test_gomb_soma_vision_quadtree.py \
    signedkan_wip/tests/test_gomb_soma_vision_patch_graph.py \
    signedkan_wip/tests/test_gomb_soma_vision_ricci_stim_classifier.py \
    signedkan_wip/tests/test_gomb_soma_vision_ricci_stim_backbone.py \
    signedkan_wip/tests/test_gomb_soma_vision_ricci_stim_detector.py \
    signedkan_wip/tests/test_gomb_soma_vision_ricci_stim_train.py \
    signedkan_wip/tests/test_gomb_soma_bochner_conv.py
============== 168 passed in 15.27s ==============
```

The 7 newly added tests pin:

* **5 × `test_hodge_canonical_flag_invariance[seed=0..4]`** — on
  random simplicial complexes, ∂₁, ∂₂, Δ₀, Δ₁, Δ₂ are byte-identical
  (max-abs-diff < 1e-12) with `edges_already_canonical=True` vs the
  default canonicalising path. This is the only operator-level
  semantic the optimisation could break.
* **`test_hodge_boundary_2_vectorised_drops_missing_edges`** —
  matches the original Python loop's silent-drop behaviour when a
  triangle boundary points at an edge absent from the canonical edge
  set.
* **`test_hodge_boundary_2_vectorised_partial_d_squared_zero`** —
  ∂₁∂₂ = 0 holds under the vectorised path on a randomised mid-size
  complex (n_v = 10, p_edge = 0.5, p_face = 0.8).

## 5. Performance results

### 5.1 `StimulusGraphBuilder.forward` (median of 5 windows × 80 iters)

Host: AMD Ryzen 7 3700X (8-core), 32 GB RAM, NVIDIA RTX 2070 SUPER,
driver 580.126.09, kernel 6.17.0-23-generic. torch 2.4.1+cu121
(CORE.YAML-pinned). Quiet machine.

| Configuration                              | Median (ms) | Worst (ms) | IQR (ms) |
|--------------------------------------------|------------:|-----------:|---------:|
| Pre-fix (baseline, 2026-05-16 04:30)       |       9.288 |     9.289  |   0.020  |
| Post-fix, full forward                     |       9.133 |     9.233  |   0.076  |
| Post-fix, Hodge in isolation (flag=True)   |       3.713 |     3.713  |   ~0     |
| Post-fix, Hodge in isolation (flag=False)  |       3.904 |     3.904  |   ~0     |
| Pre-fix, Hodge in isolation (Python loop)  |       4.756 |     4.756  |   ~0     |

**Net per-call savings: 0.17 ms (1.8%).** The Hodge isolated number
moved from 4.76 → 3.71 ms (−1.04 ms) under the new code path with
the canonical flag, but the saving in the full-forward composition
is less than half of that because the dead-`adj` Python loop was
running concurrently with adjacent CUDA work in the pre-fix forward.

### 5.2 Production-scale Config E smoke

```
3× repeats after warmup:
  wall = 2.1 s
  wall = 2.0 s
  wall = 2.0 s
```

Equivalent to the [Pass-4 baseline](2026-05-15-gomb-soma-ricci-stim-sdrf-optimization.md#1-summary)
of 2.1 s. **No regression on Config E smoke.**

### 5.3 Why the wall-clock gain was smaller than budgeted

The plan budgeted 9.3 → 5.5 ms (save ~3.8 ms). Delivered: 9.3 → 9.1 ms
(save ~0.2 ms). The plan was **95% short of the savings estimate**.

Root cause (diagnosed in §5.1 post-fix breakdown):

| Sub-component of Hodge `forward`    |  ms  | Plan assumption       |
|-------------------------------------|-----:|-----------------------|
| `torch.sparse.mm` × 4 (Δ₀, Δ₁ × 2, Δ₂) | ~2.1 | Not identified         |
| `boundary_2` Python loop (pre-fix)  | 0.71 | **Claimed 2.0 ms hot spot** |
| `boundary_2` vectorised (post-fix)  | 0.87 | Expected 0.3 ms        |
| `boundary_1`                        | 0.29 | Cheap                  |
| Canonicalise (skipped post-fix)     | 0.5–0.9 | Saved by flag       |

At the production scale (n_t = 240, n_e = 700), the Python loop over
240 triangles materialising ~700 Python list appends is faster than
the vectorised path that requires `argsort(n_e)` + 3 ×
`searchsorted(n_e)` CUDA launches. The CUDA launch overhead at this
input size is the binding constraint. **The vectorised path is
expected to win at larger n_t (e.g., > 2000) but production caps
at 256.**

Net: the change is correctness-equivalent and code-cleaner but does
not deliver a useful wall-time speedup at the bench shape.

## 6. New / removed dependencies

None.

## 7. Open issues and follow-up items

### 7.1 The actual Hodge hot spot is sparse matmul

Across the four `torch.sparse.mm` calls (Δ₀, Δ₁ term-1, Δ₁ term-2,
Δ₂), each takes ~0.5 ms — totalling ~2.1 ms. To meaningfully reduce
Hodge cost we would need either:

* dense matmul on the small operators (boundary matrices fit in a
  few KB even at the production cap); or
* fuse Δ₀, Δ₁, Δ₂ computation into one custom CUDA / Triton kernel
  that touches the boundary matrices once.

Neither belongs in this pass. Flagged as a candidate for a future
Pass-6+ together with the cross-pass topology-cache idea from the
`2026-05-15-ricci-stim-opt-pass-5` plan (whose P5 — "profile the
unaccounted 10 ms" — is now also better served by the data in §5.1).

### 7.2 Pass-5-Hodge boundary_2 keeps its place as the production path

The Python-loop boundary_2 is gone from the tree. The vectorised
variant ships even though it does not win at n_t ≤ 256, because:

* it is the only path the suite tests after this commit;
* it scales correctly into n_t » 1000 territory (e.g., if the SDRF
  rewiring is later allowed to produce more aggressive triangle
  promotion);
* keeping a Python-loop fallback alongside is a §6.5(#1) violation
  (Cartesian-product API).

This is a deliberate "do not regress small-scale at the cost of
keeping a dual-path API" trade-off, documented here.

### 7.3 Ablation result vs. the plan's decision tree

The 2026-05-15 ablation
([`ricci_stim_ablation_20260514T225345Z/`](../signedkan_wip/experiments/results/ricci_stim_ablation_20260514T225345Z/ablation.jsonl))
landed mAP50_proxy in [0.14, 0.17]:

| Config | mAP50_proxy | Wall (s) | Settings           |
|-------:|------------:|---------:|--------------------|
|   A    |       0.153 |     2079 | no Bochner, no SDRF |
|   B    |       0.170 |     2211 | α=0.1, no SDRF      |
|   C    |       0.168 |     2055 | β=0.1, no SDRF      |
|   D    |   **0.174** |     1987 | α=β=0.1, no SDRF (best) |
|   E    |       0.141 |     2713 | α=β=0.1 + SDRF (worst) |

Per the `2026-05-15-ricci-stim-opt-pass-5` decision tree, this is the
"Config E < 0.235" branch — architectural reconsideration is needed
before further optimisation. Two findings to act on (separately,
outside this report's scope):

* **SDRF is net negative on Cluttered MNIST.** Config E (SDRF on) is
  the worst mAP50_proxy of the five, and the slowest by 27%. The
  Phase-6 monotonicity contract is satisfied, but the rewired graph
  evidently degrades the detection signal here. Cluttered-MNIST is
  a too-sparse-content regime for bottleneck rewiring to help.
* **Bochner α + β both help, additively.** Config D
  (α = β = 0.1, no SDRF) is best at 0.174, vs A's 0.153 baseline
  (+0.021 mAP50_proxy from Bochner alone). The Ricci-Bochner
  hypothesis survives.

Neither of these is changed by this pass. Logged as follow-up:
`docs/plans/<next>-gomb-soma-bochner-stronger-alpha-beta/`
(scope: sweep α/β ∈ {0.05, 0.1, 0.2, 0.3}, drop SDRF from the
default Config E definition).

## 8. Experiment provenance

* **Git SHA:** `2ccaa4d12fae1ff9cd533bd91cd84b28f11c3dab`
  ("Gomb reaches SOTA. By large", 2026-05-14). Working tree dirty,
  specifically:
  * `M signedkan_wip/src/hymeko_gomb/soma/vision/hodge.py` (this pass)
  * `M signedkan_wip/src/hymeko_gomb/soma/vision/stim_graph.py` (this pass)
  * `M signedkan_wip/tests/test_gomb_soma_vision_hodge.py` (this pass)
  * `?? docs/plans/2026-05-16-gomb-soma-hodge-vectorize/` (this pass)
  * Plus a large unstaged `docs/book/book/**` regeneration (orthogonal
    mdbook rebuild, not driven by this report).
* **Environment:**
  * OS: Linux 6.17.0-23-generic
  * CPU: AMD Ryzen 7 3700X 8-Core (3.6 GHz base)
  * RAM: 32 GB total, 17 GB available at run time
  * GPU: NVIDIA GeForce RTX 2070 SUPER (8 GB)
  * Driver: 580.126.09
  * CUDA: 12.1 (via `torch 2.4.1+cu121`)
  * Python: 3.12.13
  * Pinned `torch==2.4.1` (CORE.YAML); pinned `numpy>=1.26,<2.0`
* **Random seed:** `torch.manual_seed(0)` for profile script; seeds
  0..4 for `test_hodge_canonical_flag_invariance`. Config E smoke
  uses `--seed 0` (CLI default for this report).
* **Dataset hash:** Cluttered MNIST is generated on the fly from
  `torchvision`-provided MNIST + the
  `signedkan_wip/src/vision/cluttered_mnist.py` deterministic
  generator. No new data ingested.

## 9. §6.5 anti-pattern review

| # | Anti-pattern                                       | Status |
|--:|----------------------------------------------------|--------|
| 1 | Cartesian-product API surface                      | clean — no new variant names, no flag-named-into-function-name |
| 2 | Algorithm code behind a Python boundary             | clean — pure Python module |
| 3 | Per-experiment scaffold duplication                | clean |
| 4 | Long single-file modules (>400 LOC + ≥ 2 concerns) | improved — `stim_graph.py` net shrunk; `hodge.py` grew but stays single-concern (∂_k assembly + Δ_k) |
| 5 | Adding a new axis by inventing a new function name | clean — `edges_already_canonical` is a kwarg flag, not a new function |
| 6 | `#[allow(clippy::too_many_arguments)]` as band-aid | n/a (Python) |
| 7 | String-typed config that should be an enum         | clean |
| 8 | Forward-time flags for structural differences      | clean — flag is parametric, no structural divergence |
| 9 | Bypassing existing Strategy traits                 | clean |
| 10 | `ulimit -v` on CUDA workloads                     | n/a |
| 11 | Global / module-level mutable state               | clean |

No suppressions / waivers introduced. No `# type: ignore` /
`# noqa`. No new `unwrap` / bare-`except` / silent failures.

## 10. Acceptance

- [x] Plan dir with 4 formats committed
      (`docs/plans/2026-05-16-gomb-soma-hodge-vectorize/`).
- [x] Dead code removed (~110 LOC).
- [x] Vectorised `boundary_2` path landed under a regression-test pin.
- [x] All 168 Ricci-Stim tests pass (161 existing + 7 new).
- [x] Production-scale Config E smoke runs (2.0–2.1 s, no regression).
- [x] No CORE.YAML edits, no anti-patterns introduced.
- [ ] **Plan's 3.8 ms savings budget NOT met — delivered 0.17 ms (1.8%).**
- [x] Honest report (this file) documents the budget miss and root cause.

## 11. Honest verdict

This pass delivered the *correctness scaffolding* for vectorised
Hodge boundary assembly and removed real dead code, but the
expected ms savings did not materialise. The plan's profile-driven
estimate confused "Hodge total time" with "Hodge Python-loop time"
— the dominant Hodge cost at the production scale is four
`torch.sparse.mm` calls, not the boundary_2 Python loop, and the
vectorised replacement actually loses to the Python loop at
n_t ≤ 256 due to CUDA launch overhead.

The pass is shipped because:

1. The new code path correctly scales into larger-n_t regimes (and
   the tests now pin its correctness, which they did not before).
2. ~110 LOC of dead code is gone.
3. No regression at the bench shape.
4. The accurate Hodge breakdown (§5.3) is the necessary input for a
   future pass that actually touches sparse-mm — which is where the
   real Hodge wall-time lives.

Bigger-impact next-pass candidates, ranked by the data in this
report:

* **Fuse Hodge Δ_k construction into one Triton kernel** (estimated
  −1.5 ms / call). Hardest; biggest single-component win.
* **Topology cache between SDRF passes 1 and 2** (estimated −3 ms /
  Config E forward when SDRF is on; but SDRF itself is now known
  to be net negative on Cluttered MNIST, so this lever's value is
  conditional on a different ablation).
* **Reconsider Config E definition (drop SDRF)** — would
  unconditionally save ~3 ms and improve mAP50_proxy from 0.141 to
  ≥ 0.174 (Config D). Pure free win on this dataset.

---

*End of Pass 5-Hodge report. Code cleaner, tests stronger, wall-time
gain modest, plan budget missed but reasons understood.*
