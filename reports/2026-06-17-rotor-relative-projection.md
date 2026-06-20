# Projecting the relative rotation — both routes flat; bilinear already captures the sign-relevant rotor projection

**Date:** 2026-06-17
**Plan:** [docs/plans/2026-06-17-rotor-relative-projection](../docs/plans/2026-06-17-rotor-relative-projection/) (4 artifacts; PDF compiles).
**Status:** ✅ implemented + tested + planned + 5-seed ablation + leakage gate + a forced-γ discriminator. ⚠️ **negative result** — projecting the full relative rotation does not beat bilinear. Hypothesis (Dr. Hajdu: "the rotation must be projected, that's what bilinear does; we're working with complex numbers") tested and **not supported on these datasets**.

## Hypothesis
The woken geom_attn readout still lost to bilinear. The steer: bilinear `γ h_u^T W h_v` is, on rotor coordinates, a learned projection of the relative rotation `R_u^T R_v`; maybe the gap is that the rotor's rotational structure isn't being projected properly. Confirmed in source, and sharpened: `CayleyRotorEmbedding` returns only `e = R·v0` (one reference/block), and a rotation about the `v0` axis fixes `v0` — so that **DoF of the rotor is discarded** before bilinear ever sees it. Bilinear projects only the ≤2-DoF shadow of the relative rotation. Two routes to recover the lost DoF (user chose **both**, **quaternion/SO(3)** as built).

## What was built
- `embeddings/cayley_rotor.py`: general `quat_mul` (Hamilton product) + `quat_conjugate` (autograd-clean; the `pool_scatter` Hamilton kernel is private/Triton/fixed-shape — deliberately not coupled to); a `rotors(x)` accessor exposing `q` *before* the lossy `R·v0`; and `n_refs` (reference `(n_blocks, n_refs, 3)`; `embedding_dim = 3·n_blocks·n_refs`; `n_refs=1` = prior output dim).
- `core/rotor_relative_head.py` (new): `RotorRelativeHead` — projects the relative rotor `conj(q_u)⊗q_v` (full relative rotation) to the logit, **added** to bilinear, dead-start γ.
- `run_hsikan_rotor.py`: `--head rotor_rel` (route A: bilinear + relative-rotor term, fed by `inj.emb.rotors`), `--n-refs` (route B), provenance fields.

## Files touched
- `signedkan_wip/src/embeddings/cayley_rotor.py` (+~45 LOC: `quat_mul`/`quat_conjugate`/`rotors`/`n_refs`; `forward` refactor)
- `signedkan_wip/src/core/rotor_relative_head.py` (new, ~60 LOC)
- `signedkan_wip/experiments/runs/run_hsikan_rotor.py` (+~20 LOC: head, n_refs, wiring, provenance)
- `signedkan_wip/tests/test_cayley_rotor.py` (+6 tests; 1 shape test updated for the `n_refs` axis), `test_rotor_relative_head.py` (new, 5 tests), `test_hsikan_rotor.py` (+3 tests)
- `docs/plans/2026-06-17-rotor-relative-projection/plan.{tex,pdf,tikz,mmd}` (new)
- CORE.YAML items touched: **none**. (Verified `hymeko_hre` is *not* in CORE.YAML — protected crates are `hymeko_core`/`hymeko_query`/`hymeko_client`/`hymeko_daemon`/`parser`; `on_unknown_path: treat_as_non_core`. No new Cargo/pip dep — algebra is plain torch.)

## Tests
- `ruff check` (6 files): **PASS**. `pytest -p no:randomly` (cayley/rotor-head/driver/geom suites): **60 passed in ~23 s**. New tests pin: `quat_mul` = reference Hamilton expansion + identity + associativity; `conj(q)⊗q = 1` for unit `q`; `rotors` unit-norm; `n_refs` scales `embedding_dim` and `n_refs=1` unchanged; `RotorRelativeHead` is **invariant to a global rotation of both endpoints and changes when one endpoint rotates** (the relative-rotation property — the DoF bilinear lacks); dead-start; gradient flow; `rotor_rel` requires the rotor embedding.

## Results — 5-seed ablation (tuned recipe, strict triads, dedup)
`reports/rotorrel_ab_20260617.jsonl` (40 cells) + `reports/rotorrel_smoke_20260617.jsonl` (seed-0).

| dataset | bilinear | **rotor_rel (A)** | Δ | **n_refs=2 (B)** | Δ vs n1 | SiGAT | gate (shuffle) |
|---|---:|---:|---:|---:|---:|---:|---:|
| bitcoin_alpha | 0.8455 | 0.8457 | **+0.0002** | 0.8404 | −0.0051 | 0.884 | 0.524 [0.46–0.59] |
| bitcoin_otc | 0.8685 | 0.8685 | **+0.0000** | 0.8690 | +0.0005 | 0.902 | 0.521 [0.50–0.57] |

Per-seed `rotor_rel − bilinear`: alpha `[0,0,0,0,+0.0011]`, otc `[0,0,0,0,0]` — **9/10 seeds bit-identical**.

### Discriminating test (measured / inferred, per the contract)
The exact `rotor_rel == bilinear` equality was a flag (a constant logit offset leaves AUROC unchanged), so it was isolated, not assumed:
- **Rotors are not degenerate** (measured): structural-feature rotor component std **0.18** across nodes; relative-rotor std **0.24** across test edges. The relative-rotor term varies per edge — it is *not* a constant offset.
- **Forced-γ test** (measured): `RotorRelativeHead` with `gamma_init=1.0` (vs the 1e-3 dead-start) gives alpha 0.8632 / otc 0.8846 — still flat (±0.0004 vs bilinear). So the term is **live but carries no sign-predictive signal**; the optimiser correctly leaves the dead-start γ near zero.

**Inferred:** the about-`v0` rotation DoF that `e=R·v0` discards is **not sign-predictive** on these datasets — bilinear already projects everything the rotor comparison offers for the edge sign. Route A adds nothing even forced on; route B (exposing the full rotation to bilinear) does not help (slightly hurts alpha, flat otc). Note: quaternion SO(3) strictly contains the complex SO(2)/relative-phase case, so a complex variant cannot carry more signal than this — the negative generalises.

## Honest read
The hypothesis is cleanly **falsified**: projecting the relative rotation is not the missing element. This is the third readout/comparison-side lever to come back flat (geom_attn dead score → woken score → rotor-relative projection), all leakage-clean. Taken together they localise the SiGAT gap **away from the node-endpoint comparison** and toward what SiGAT actually does differently — aggregation over signed *neighbourhoods*. The new machinery (rotor algebra, `rotors`/`n_refs`, the relative-rotor head) is correct, tested, and kept (defaults off) for reuse.

## Performance
- Rotor-rel head adds `4·n_blocks+2` params (e.g. 42 at hidden=32) + one conjugate/Hamilton product/linear per edge — negligible. `n_refs=2` doubles the embedding width (still ~hundreds of dims).
- Wall: ~7 min for the 40-cell grid (~10–12 s/cell). Peak RSS not separately polled; same scale/config as the prior 1724 MB measurement (10.5 % of the 16 GB cap).

## §6.5 anti-patterns
None. Quaternion algebra centralised in `cayley_rotor.py` (canonical home; coupling to the private `pool_scatter` Triton kernel rejected). New head is its own module (structural variant → class, §6.5 #8), wired via a config head choice (no function-per-axis, no `_v2` file). `n_refs` is an additive kwarg; defaults preserve the bilinear line.

## Decision / next step
Three comparison-side hypotheses are now falsified. The evidence points to the **neighbourhood aggregation** as the gap — i.e. the **Berge cycles** substrate (richer signed neighbourhoods than triads; `hymeko_hre/src/traversal/berge.rs` foundation, confirmed non-core). Recommend pivoting there now, with the comparison line closed on evidence rather than assumption. Awaiting user confirmation.

Memory: `project-hsikan-geometric-attention-berge` (updated).
