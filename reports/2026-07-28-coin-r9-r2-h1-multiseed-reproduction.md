# R9 R2-under-H1 multiseed — the first learned K6 is a reproducible LEARNING result under the explicit handoff-reset (22/24)

**2026-07-28 · branch `recovery/coin-r9-causal-residual-delivery` · immutable base `41510cac` / tag `coin-r9-r2-first-learned-s1-k6-explicit-handoff-reset` · dev s1 (14250) · s4/s7 untouched · f1–f4 SEALED · TD3 · no tag moved**

## Summary

The corrected claim — *R2 is the first learned teacher-free s1 K6 transport policy, under the explicit H1 handoff-reset* — is now
established as a **reproducible learning result**, not a checkpoint property. Re-running the **R2 residual training** (frozen
architecture / reward2 / authority α = 0.15 / 100 %-cradle curriculum / safety / K6 / champion order) with the KINETIC policy
deployed through the explicit **H1 HANDOFF_RESET** contract, across 24 independent seeds, every episode the full uninterrupted
chain from the canonical cradle (`cradle → APPROACH → HANDOFF_RESET → KINETIC clone + learning-R2 → coast → K6`, **no offline
frozen-entry artifact as episode-start**):

- **22 / 24 verified strict K6** — rate **0.917**, Wilson 95 % CI **[0.741, 0.977]** — **0 safety violations**, **0 stall / 0
  clamp / 0 reversal** across all 24. Both pre-declared gates **PASS**: `R2_H1_MULTISEED_REPRODUCTION_PASS` (≥ 3) and
  `R2_H1_RELIABLE_LEARNING_PASS` (≥ 12).
- Every one of the 22 was **independently verified** on the H1 deploy: canonical `delivery_success`, dwell ≥ 6, **exactly one
  HANDOFF_RESET before the first policy step**, teacher-absent, deterministic replay, safe, clean.
- min_dtz over the K6 seeds: **min 0.98 mm**, median 6.4 mm (deeper than the frozen-policy 16.86 mm — training under H1 finds
  better residuals than the historical H0-trained champion). options-to-first-K6 median 225; dwell median 12.
- 2 non-K6 seeds (11, 22): 42.94 / 28.69 mm — did not reach the zone; not stalls/clamps/unsafe.

This confirms the resolution: the delivery is **learned by R2 under H1**, reproducibly, end-to-end from the cradle. It is neither a
lucky checkpoint nor an offline-snapshot artifact.

## Aggregate (24 seeds, R2 trained under H1)

| quantity | value |
|---|---|
| verified strict K6 | **22 / 24** (rate 0.917) |
| Wilson 95 % CI | [0.741, 0.977] |
| safety violations | 0 |
| stall / clamp / reversal totals | 0 / 0 / 0 |
| HANDOFF_RESET per verified seed | exactly 1 (before the first policy step) |
| options-to-first-K6 | median 225 (range 25–475) |
| min_dtz over K6 | min 0.98 mm, median 6.4 mm |
| K6 dwell | min 6, median 12, max 21 |
| non-K6 seeds | 11 (42.94 mm), 22 (28.69 mm) |

## Frozen contract (nothing tuned on the seeds' results)

Frozen K2 clone + per-step R2 residual (α = 0.15) + **H1 explicit handoff-reset** + TD3; reward2 (no R3-B envelope), 100 %-cradle
curriculum (no frontier starts), same safety / K6 monitor / champion order; fresh critics, replay, exploration RNG, residual head per
seed; 600-option budget with freeze on the first strict K6. Seed 0 (the historical champion) is not a training seed. Reproduces the
R3-B/R3-C reuse pattern via `collect_episode3(envelope_w = 0, make_controller = H1)`.

## What changed vs the earlier panels — and what didn't

- **The panel now trains R2 (not the C1 expansion), and deploys through H1 (not from a snapshot).** The C1 22/24 was the expansion
  over the frozen R2; this 22/24 is the *original R2 skill* learned end-to-end under the explicit reset — the correct object for
  the "first learned K6" claim.
- **Same 22/24 rate**, but now the delivery is attributed to R2 under H1, consistent with the frozen-policy intervention
  (clone+R2 load-bearing) and the handoff-reset audit (H1 delivers, H0 does not).
- **Preserved:** the R3-C 22/24 (dwell-refinement over R2) and the FULL-vs-NO_EXPANSION intervention — both on the H1-equivalent
  interface.

## Cross-host sanity — attempted, blocked by workspace env-parity (honest status)

kato14/kato15 have a **matching venv** (torch 2.12.0 / mujoco 3.10.0 / numpy 2.4.6) and the coin code is pure Python, so the
package + data (`clone_seed0.pt`, `teacher_bank.json`, a checkpoint) rsync cleanly. But `hymeko_rl/__init__.py` and the harness
builder (`load_harness → video_coin_variants._setup`) transitively pull in a **chain of sibling packages and top-level scripts**
(`hymeko_neuro` (84 MB), then `coin_balltip_proposal`, …) that are absent from the stale remote checkout; each rsync surfaces the
next. A faithful cross-host run therefore needs the **entire workspace** synced at `41510cac`, an env-parity/deployment task beyond
this turn. **Because the H1 contract and the K6 verdict are deterministic** (properties of the code + data, not the host — every
seed re-verified deterministic within host), **the Mac 24-seed panel is the reproduction result**; cross-host would add only a
"not-Mac-specific" cross-check, which remains pending a full-workspace deployment. (A partial checkout was left at
`kato14:~/hymeko_r9_h1/` for that follow-up.)

## Files (all `8a0c1c7b`/`41510cac` modules imported UNCHANGED; tags never moved)

| file | role |
|---|---|
| `hymeko_rl/coin_delivery/theta_option/kinetic_r2_h1.py` (+70) | H1 collect/eval for R2 training (`collect_episode3` with `make_controller = H1`, `envelope_w = 0` ⇒ reward2) |
| `hymeko_rl/experiments/coin_kinetic_r2_h1_multiseed.py` (+~180) | 24-seed R2-under-H1 panel: per-seed train → freeze-on-K6 → independent H1 verification → Wilson CI + gates |
| `hymeko_rl/tests/test_coin_kinetic_contract.py` (+1 test) | R2-H1 collect/eval well-formed (transitions + champion with single HANDOFF_RESET) |
| `reports/2026-07-28-coin-r9-r2-h1-multiseed/{r2_h1_multiseed_combined.json, *_01_12, *_13_24, seed_NN/record.json}` | aggregate + per-seed records |

`ruff` clean; `radon cc -a` A/B. Full `test_coin_kinetic_contract.py` — **37 tests** (see run). No new §6.5 anti-patterns. Frozen
modules `git diff` empty. New tag `coin-r9-r2-first-learned-s1-k6-explicit-handoff-reset` → `41510cac`; historical tag
`coin-r9-first-learned-s1-k6-delivery` → `8a0c1c7b` unchanged.

## Provenance

Immutable base `41510cac`. Python 3.11.15 / mujoco 3.10.0 / numpy 2.4.6 / torch 2.12.0 / macOS-26.5.2-arm64 (Apple Silicon, CPU),
`OMP=MKL=OPENBLAS=1`. Training seeds 1–24; cradle 14250; R2 residual distill-zero warm-start per seed. Half walls 166 s (1–12) /
221 s (13–24). Deterministic (per-seed replay re-verified). Peak RSS ≈ 0.3 GB/worker.

## Status & next

`R2_H1_MULTISEED_REPRODUCTION_PASS` **and** `R2_H1_RELIABLE_LEARNING_PASS` — **22/24, 0 safety, 0 stall/clamp/reversal.** The first
learned s1 K6 is a reproducible learning result under the correctly-defined H1 controller contract, run end-to-end from the cradle.
Committed on its own boundary; tags untouched. **Next (deferred for review):** (a) complete cross-host reproduction once the full
workspace is deployed on kato14/kato15; (b) the C1 dwell-refinement panel (paired R2 vs R2+expansion, 8–12 seeds) whose corrected
question is `AUTHORITY_UNLOCK_IMPROVES_K6_SETTLE_MARGIN`, not first delivery. **STOP.**
