# Link-head ablation (real→complex→quaternion→rotation) — expressivity beats algebra; the head is not the gap

**Date:** 2026-06-17
**Plan:** [docs/plans/2026-06-17-link-head-ablation](../docs/plans/2026-06-17-link-head-ablation/) (4 artifacts; PDF compiles).
**Status:** ✅ implemented + tested + 5-seed A/B (two harnesses: a first confounded one, then the corrected fair one) + gate. **Negative result** — no complex/quaternion/rotation head beats a real bilinear readout of the propagated rotors. The head algebra is not the missing piece; the gain lives in the input (propagation).

## Question
Bilinear is a real matched filter; the embeddings are rotor-derived. Does a head closer to the complex/quaternion algebra beat it, once the input is propagation-fixed? Ladder: bilinear (real), ComplEx (complex), QuatE/rotor_rel (quaternion), RotatE/geodesic (rotation). HOSVD/TuckER omitted by design — it is the *multi-relational* decomposition, and binary sign prediction has no relation mode, so it degenerates to bilinear; the complex/quaternion heads already carry the algebraic weight-sharing the idea targets.

## Two harnesses — and why the first was confounded
**(1) In-pipeline (`run_hsikan_rotor --head {complex,rotate}`, + existing rotor_rel):** confounded — ComplEx read the *real* encoder output `h_v` (no complex structure to exploit), and the rotor heads were *additive to bilinear* (their gradient flowed into the shared rotor injector; `rotate` actively hurt). 5-seed (`headabl_ab_20260617.jsonl`): every head ≈ bilinear (alpha ~0.85, otc ~0.879) — the confound *masked* the real ordering by anchoring all heads to bilinear-on-`h_v`. **Not the real test** (Dr. Hajdu flagged this: "not as straightforward to change the heads").

**(2) Fair isolation (`run_rotor_head_ablation.py`, new):** every head scores the **same** propagated rotors `q` (quaternion-native + neighbourhood-aware), standalone — no triad encoder, no additive bilinear. The only variable is the head's algebra.

## Results — fair 5-seed (all heads on the same `q`, r2 sw4 propagation, dedup)
`reports/rotorhead_fair_ab_20260617.jsonl` (+ `rotorhead_fair_smoke_`).

| dataset | **real** | complex | geodesic | quat (QuatE) | gate (quat) |
|---|---:|---:|---:|---:|---:|
| bitcoin_alpha | **0.8492** | 0.8334 | 0.7699 | 0.7161 | 0.392 |
| bitcoin_otc | **0.8741** | 0.8668 | 0.8063 | 0.7572 | 0.408 |

Ranking **real > complex > geodesic > quat**, consistent on both, 5 seeds; gaps ~0.13 (≫ seed noise). Params: real 2834 (full bilinear), complex/quat 1274, geodesic 1244 — **not a capacity artifact**: complex and quat have *identical* param counts yet complex ≫ quat. Gates ≈ chance → leakage-clean.

## Honest read
On the same quaternion representation, the **real bilinear readout wins**; the algebraically-structured heads are worse, the quaternion relative-rotor (QuatE) head most of all. The lesson:

- **Expressivity beats algebraic structure at the readout.** A full real bilinear can learn any `u`–`v` interaction (including relative-rotation terms) and more; QuatE/geodesic *restrict* it to one interaction (the relative rotation / its alignment), complex to a diagonal-in-complex bilinear. Constraining the readout to an algebra discards capacity, so it loses.
- **Manifold structure pays off where you aggregate, not where you compare.** Quaternion/slerp logic gave a real lift in the **input** (propagation report); imposing the same algebra on the **readout** hurts. Aggregate on the manifold; read out freely.

**Side finding:** real-bilinear on `q` *alone* (0.849/0.874) is within ~0.005 of the full triad-encoder pipeline (0.850/0.879) — the propagated rotor carries almost all the signal; the triad encoder adds little. A candidate model simplification.

## Files touched
- `hymeko_neuro/hyperedge/bilinear_head.py` (`ComplexDiagonalHead`), `core/rotor_relative_head.py` (`RotorGeodesicHead`)
- `hymeko_neuro/experiments/runs/run_hsikan_rotor.py` (`--head {complex,rotate}`, `RotorInjector.propagated_rotors`, propagated-rotor routing — fixes the latent bug where `rotor_rel` read un-propagated rotors)
- `hymeko_neuro/experiments/runs/run_rotor_head_ablation.py` (new fair harness, reuses `_optimise`/`_pos_weight`/`_drop_train_pairs`/`RotorInjector`)
- tests: `test_bilinear_head.py` (new), `test_rotor_relative_head.py` (+geodesic), `test_rotor_head_ablation.py` (new), `test_hsikan_rotor.py` (+head smokes)
- CORE.YAML items touched: **none**.

## Tests
`ruff`: PASS. `pytest`: green (ComplEx symmetric iff Im(w)=0 / asymmetric otherwise; geodesic invariant to global rotation; RotorHeadModel finite logits + gradient per head; dead-start; shape guards).

## §6.5 anti-patterns
The fair harness is a *distinct model topology* (no triad encoder), so per #8 it is its own assembly rather than another `forward`-time flag in `run()` — and it reuses the shared train/data helpers by import (no logic duplication). HOSVD omitted with reason (above). Confounded in-pipeline heads kept (defaults off) for the record.

## Decision
The head is not the gap — close the head line. The session's signal is the **input/interpolation** (propagation). Follow-ups: learnable propagation self-weight/gate; the "SiGAT gap" was partly a non-deduped-target protocol mismatch (record in any writeup). Memory: `project-hsikan-geometric-attention-berge`.
