# GömbSoma Phase 3-V — Vision starter (patch graph + WalkConv classifier)

**Date:** 2026-05-14
**Plan:** [docs/plans/2026-05-14-gomb-soma/](../docs/plans/2026-05-14-gomb-soma/)
**Phase:** 3-V (Vision branch; runs before continuing with PolygonConv in 3-G)
**Prior phases:** [Phase 1](2026-05-14-gomb-soma-phase1.md), [Phase 2](2026-05-14-gomb-soma-phase2.md)

## 1. Summary

Branched the GömbSoma phase sequence into a vision sub-branch to
test the *sensorimotor-stack* hypothesis directly: can Phase 1+2's
HypergraphConv + WalkConvLayer primitives already produce something
useful on a raw-image task, without polygon / triangle / Clifford-FIR
machinery?

Built:
* A `PatchGraphBuilder` that converts an image into a signed
  4-connected patch graph with all length-2 walks enumerated.
* A `WalkConvImageClassifier` that wraps WalkConvLayer end-to-end:
  image → patches → walks → walk-conv → classify.

The architecture is deliberately minimal: one walk-conv layer, no
polygons, no triangles, no Clifford-FIR transfer (those are phases
3-G/4/5 of the main plan). The point is to validate the bottom rung
of the sensorimotor stack actually runs on a vision task.

## 2. Files touched

| File | LOC | Notes |
|---|---|---|
| [signedkan_wip/src/hymeko_gomb/soma/vision/__init__.py](../signedkan_wip/src/hymeko_gomb/soma/vision/__init__.py) | 27 | subpackage init |
| [signedkan_wip/src/hymeko_gomb/soma/vision/patch_graph.py](../signedkan_wip/src/hymeko_gomb/soma/vision/patch_graph.py) | 198 | `PatchGraphBuilder` — image → signed patch graph |
| [signedkan_wip/src/hymeko_gomb/soma/vision/walk_conv_classifier.py](../signedkan_wip/src/hymeko_gomb/soma/vision/walk_conv_classifier.py) | 113 | end-to-end image classifier |
| [signedkan_wip/tests/test_gomb_soma_vision_patch_graph.py](../signedkan_wip/tests/test_gomb_soma_vision_patch_graph.py) | 124 | 11 tests on the encoder |
| [signedkan_wip/tests/test_gomb_soma_vision_walk_conv_classifier.py](../signedkan_wip/tests/test_gomb_soma_vision_walk_conv_classifier.py) | 113 | 7 tests on the classifier |

No edits to Gömb / Phase-1 ABC / Phase-2 WalkConvLayer / CORE.YAML.

## 3. Encoder architecture

```
image (C, H, W)
  │
  │   PatchGraphBuilder
  ▼
  ┌───────────────────────────────────────────────────┐
  │  patches:      (n_patches, C·P²)                  │
  │  walks:        (n_walks, 3)  — all length-2 paths │
  │  walk_signs:   (n_walks,)    — σ-product over     │
  │                edges in {-1, +1}                  │
  │  M_v:    sparse (n_patches, n_walks), 1/3 weight  │
  └───────────────────────────────────────────────────┘
```

Edge sign: σ(src→dst) = +1 iff mean brightness of src patch ≥ dst,
else −1. Walk sign: σ-product of the walk's two constituent edges.
The grid topology (edges, walks, M_v) is precomputed once at
`__init__` and shared across the batch.

For MNIST (28×28, P=4): 49 patches, 168 directed edges, 428 length-2
walks. For a 12×12 toy: 9 patches, 24 edges, 36 walks.

## 4. Classifier architecture

```
image  →  patchify  →  Linear(patch_dim → d_hidden)
                          │
                          ▼
                     WalkConvLayer(d_hidden, d_hidden, k_arity=3)
                          │
                          ▼  (sign-branched, position-aware)
                     mean over patches
                          │
                          ▼
                     Linear(d_hidden → n_classes)  →  logits
```

Param count at MNIST defaults (P=4, d_hidden=16, n_classes=10):
**2 010 params total** (patch_embed 272 + walk_conv 1 568 + head 170).

## 5. Test results

```
$ python -m pytest signedkan_wip/tests/test_gomb_soma_vision_patch_graph.py \
                   signedkan_wip/tests/test_gomb_soma_vision_walk_conv_classifier.py -v
=========== 18 passed in 3.37s ===========
```

### Patch graph tests (11/11)

| Test | What it pins |
|---|---|
| `test_grid_dimensions_for_mnist` | 28×28 / P=4 → 49 patches |
| `test_edge_count_for_grid` | 7×7 4-connected = 168 directed edges |
| `test_walks_are_length_2_no_backtracking` | walks have shape (·, 3); no a==c |
| `test_rejects_unaligned_image_size` | image not divisible by P → ValueError |
| `test_rejects_bad_patch_size` | patch_size < 1 → ValueError |
| `test_patchify_shape` | (1,28,28) → (49,16) |
| `test_patchify_preserves_content` | top-left patch contains top-left pixels |
| `test_edge_signs_polarity` | left→right dark→bright edge has σ = −1 |
| `test_walk_signs_compose_edge_signs` | walk σ = product of its two edge σ |
| `test_M_v_shape` | (n_patches, n_walks) with 1/3 weights |
| `test_encode_one_shot` | encode() returns consistent shapes |

### Classifier tests (7/7)

| Test | What it pins |
|---|---|
| `test_construction_smoke` | n_params = 2 010 for MNIST defaults |
| `test_forward_shape_batched` | (4,1,28,28) → (4,10) |
| `test_forward_shape_single_image` | bare (1,28,28) → (10,) |
| `test_gradient_flow_through_all_components` | every param has gradient |
| `test_brightness_inverted_input_changes_output` | sign-branching not dead — inverted image gives different output |
| `test_no_sign_branching_falls_back` | use_sign_branching=False still trains |
| `test_can_overfit_two_samples` | **end-to-end signal flow** — 300 steps on 2 samples reaches 100% accuracy |

The last test is the most informative: it proves the entire pipeline
(patchify → embed → walk-conv → pool → classify → backward) carries
training signal end-to-end. A learning-blocker bug anywhere in the
chain would fail it.

## 6. Performance

Tests run in 3.37 s on CPU. No GPU touched. Performance contracts
apply at the *training-run* level (separate runner script, not in
this commit).

## 7. Numerical stability

Forward path: `Linear → walk_conv → mean → Linear`. All operations
FP32-stable. No catastrophic cancellation. The overfit-2-samples
test reaches 100 % at loss < 0.01, no NaNs.

## 8. Static analysis

```
$ ruff check signedkan_wip/src/hymeko_gomb/soma/vision/
   (clean)
```

No new suppressions.

## 9. §6.5 anti-pattern review

| # | Anti-pattern | Status |
|---|---|---|
| 1 | Cartesian-product API | NO — one classifier class, one builder class |
| 2 | Algorithm code behind Python boundary | NO — pure PyTorch + numpy |
| 3 | Per-experiment scaffold duplication | NO — uses Phase 1+2 building blocks |
| 4 | Long single-file module | NO — patch_graph 198 LOC, classifier 113 LOC |
| 5 | New axis = new function name | NO |
| 6 | `#[allow(...)]` band-aid | N/A |
| 7 | String-typed config | NO — `HypergraphConvConfig` dataclass |
| 8 | Forward-time structural flags | `use_sign_branching` is parametric (inherited from WalkConvLayer); not a structural variant |
| 9 | Bypassing strategy traits | NO — uses HypergraphConv ABC |
| 10 | `ulimit -v` on CUDA | N/A |
| 11 | Globals | NO |

## 10. Phase 3-V acceptance

- [x] `PatchGraphBuilder` encodes an image as a signed patch graph.
- [x] `WalkConvImageClassifier` runs end-to-end on a batch.
- [x] 18 unit tests pass, including end-to-end overfit test.
- [x] Sign-branching not dead (brightness-inversion test).
- [x] No CORE.YAML edits.
- [x] No anti-patterns introduced.

All acceptance criteria met.

## 11. What this does NOT yet do

* No actual MNIST training run — the unit tests verify the
  architecture flows training signal, but a real benchmark
  (e.g., 5-epoch MNIST 60k training, accuracy vs CNN baseline)
  belongs in a separate runner script.
* No polygons / triangles / Clifford-FIR — those are phases 3-G /
  4 / 5 of the main plan. This branch tests walk-only.
* No HymeYOLO detection task — that requires multi-class detection
  heads on top of the graph features; deferred until walk-only
  classification is empirically validated.

## 12. Next phase options

1. **Phase 3-V-bench**: an MNIST runner script (5-seed, 5 epochs)
   that records accuracy + parameter count + walltime; gives the
   first hard number to compare against a small CNN.
2. **Phase 3-G**: PolygonConvLayer (the main-plan next phase).
3. **CIFAR-10**: same architecture on RGB.

The one-phase rule means we ship 3-V here, write the report,
and stop. The user picks next.

## 13. Reproducibility

```python
from signedkan_wip.src.hymeko_gomb.soma.vision import (
    PatchGraphBuilder, WalkConvImageClassifier,
)

m = WalkConvImageClassifier(
    image_h=28, image_w=28, patch_size=4,
    in_channels=1, d_hidden=16, n_classes=10,
)
logits = m(images)  # (B, C, H, W) → (B, 10)
```

No new dependencies.

---

*End of phase 3-V report.*
