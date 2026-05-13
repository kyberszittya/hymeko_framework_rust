# Report: CPML route maths + Highway / CapsNet / KAN unification (docs & book)

**Date:** 2026-05-12  
**Summary:** Documented **route vs pyramid** CPML; added **`tier_organization`** with **`capsule_soft`** (learned softmax cycle→tier weights, **route only**) as a **CapsNet-style** dimension orthogonal to `topology` and `aggregator_kind`; wired through **Gömb**, **smoke CLI**, tests, and handbook updates.

## Files touched

| Path | Role |
|------|------|
| `docs/book/src/research/cpml-routing-highway-capsule-kan.md` | Route maths + Highway/Capsule/KAN; **soft routing = shipped** `capsule_soft` |
| `docs/book/src/SUMMARY.md` | Research link |
| `docs/book/src/research/nn-architectures-and-layer-geometry.md` | §5 (`topology` + `tier_organization` + CLI) |
| `docs/book/src/research/gomb-orthogonal.md` | CPML sub-axes |
| `docs/book/src/research/signedkan-overview.md` | See also |
| `docs/book/src/results/mathematics.md` | CPML bullets |
| `docs/math/cpml-route-unification.md` | Pointer |
| `docs/book/src/results/abbreviations.md` | CPML route / pyramid / capsule_soft |
| `signedkan_wip/src/cpml.py` | `CPMLConfig.tier_organization`, router, forwards |
| `signedkan_wip/src/hymeko_gomb/cascade.py` | `GombConfig.cpml_tier_organization`, `cpml_capsule_route_hidden` |
| `signedkan_wip/src/hymeko_gomb/shells.py` | `InnerCPMLCore` kwargs |
| `signedkan_wip/src/hymeko_gomb/__init__.py` | Package doc |
| `signedkan_wip/src/run_gomb_smoke.py` | CLI + JSON + early exit |
| `signedkan_wip/tests/test_cpml.py` | Forward/backward grid + `pyramid`/`capsule_soft` rejection |
| `reports/2026-05-12-cpml-route-unification-highway-capsnet-kan.md` | This report |

## CORE.YAML

No items touched.

## Test results

- `PYTHONPATH=. pytest -q signedkan_wip/tests/test_cpml.py` — **23** passed (includes `capsule_soft` forward/backward + mutual exclusion test).
- `PYTHONPATH=. pytest -q signedkan_wip/tests/test_hymeko_gomb.py signedkan_wip/tests/test_sota_smoke.py` — **41** passed (includes CUDA AUROC regressions).

**64** tests across `test_cpml.py` + the two files above — all green in this session.

## Open follow-ups

- **Iterative CapsNet routing** (EM inner loops): not implemented — single softmax per forward.
- **Multi-seed AUROC** tables route vs pyramid vs capsule_soft: product gates.

## References cited in prose

- Highway networks (Srivastava et al., 2015).  
- CapsNet dynamic routing (Sabour, Frosst, Hinton, 2017).  
- KAN (Liu et al., 2024).
