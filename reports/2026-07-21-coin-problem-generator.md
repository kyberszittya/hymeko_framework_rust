---
campaign: COIN structured problem-generator experiment (matched CONTROL vs GENERATOR, 8 pairs)
title: Structured problem generation improves certified generalization to held-out generated configs
date: 2026-07-21
branch: exp/coin-problem-generator
source_commit: f054917
verdict: GENERATOR_POSITIVE — held-out generated certified coverage bootstrap CI above zero, 64102 retained, not zone-only
---

# Structured problem generator — matched CONTROL vs GENERATOR

**Created-at:** 2026-07-21 00:40 JST. Only experimental variable: the distribution of TRAINING initial states.
CONTROL = the existing fixed training states; GENERATOR = 50% fixed + 50% generated configs. Actor / semantic critic /
corrected SAC / reward / strict predicate / observation / action space / BC anchor / replay sampler all **unchanged**.
The generator only supplies initial/target configurations (it never controls the actor). 16 runs (4 seeds × 2 reps × 2
arms), thread-pinned (`OMP/MKL/OPENBLAS=1`) to shrink the BLAS noise floor found in the replay experiments; matched
pairs share `run_seed` (init/exploration).

## Verdict: **GENERATOR_POSITIVE** (and it survives the scrutiny that sank the earlier "positives")

Structured problem generation **reproducibly improves certified delivery on held-out generated configurations** — a
robust bootstrap CI above zero, 0 negative pairs — while **retaining** state-64102 competence and **not degrading** the
fixed 18-state set. It is the first genuinely positive result in this arc.

## Generator (canonical PlanarSnapshot path, minimal adapter — §2)
Reuses `snapshot_planar`/`restore_planar` + `coin_delivery.provenance.snapshot_hash`. Three families, each variant
changes exactly ONE named relation; validity is **physical only** (never policy-success): **96 train** (32
CERTIFIED_NEIGHBORHOOD [parents 64102/64201] · 32 ATTRIBUTION_BOUNDARY [64111] · 32 LEFT_RIGHT_SYMMETRY [canonical
mirror]) + **48 held-out** (16/16/16), `corpus_sha e700643c`, TRAIN/HELD hash-disjoint, only 5 `starts_in_zone`
rejections. 7 validity tests. Base checkpoint delivers **fixed cov 2 / held cov 0/48** — real headroom on generated.

## Per-pair (best checkpoint) — 8 matched pairs

| pair | fixed cov C→G | held cov C→G | 64102 C/G |
|---|---|---|---|
| s0r0 | 2→1 (−1) | 0→2 (+2) | F/F |
| s0r1 | 2→2 (0) | 2→2 (0) | F/F |
| s1r0 | 3→3 (0) | 1→1 (0) | F/F |
| s1r1 | 2→3 (+1) | 2→3 (+1) | F/**T** |
| s2r0 | 1→4 (+3) | 1→4 (+3) | F/**T** |
| s2r1 | 3→3 (0) | 0→1 (+1) | T/T |
| s3r0 | 3→2 (−1) | 0→2 (+2) | T/F |
| s3r1 | 3→2 (−1) | 1→1 (0) | T/F |

## Pooled 8-pair paired deltas (GENERATOR − CONTROL; bootstrap seed 20260720, B=10000)

| endpoint | mean | median | IQR | +/0/− | bootstrap 95% CI |
|---|---|---|---|---|---|
| fixed certified coverage (primary) | +0.13 | 0 | 1.25 | 2/3/3 | [−0.625, +1.125] (spans 0) |
| **held-out generated certified coverage (generalization)** | **+1.12** | **+1** | 2.0 | **5/3/0** | **[+0.375, +1.875] — ABOVE 0** |
| P(attr ≥0.60 \| zone), fixed | −0.01 | +0.035 | 0.24 | 4/1/3 | [−0.109, +0.080] (spans 0) |
| P(clean \| zone), fixed | −0.03 | −0.062 | 0.13 | 1/3/4 | [−0.129, +0.099] (spans 0) |

**Held-out certified coverage by family:** CERTIFIED_NEIGHBORHOOD median 0, CI **[+0.125, +1.0]** (+3/−0);
ATTRIBUTION_BOUNDARY median **+1**, CI **[0.0, +0.875]** (+5/−1); LEFT_RIGHT_SYMMETRY median 0, CI [−0.25, +0.5] (spans 0).
**64102 retention:** control 3/8, generator 3/8 — equal (not degraded).

## §12 classification → GENERATOR_POSITIVE
- **Certified-coverage CI above zero on the held-out generated set** ([+0.375, +1.875]; 5 positive / 3 zero / **0
  negative** pairs). ✓
- **State 64102 competence retained** (generator 3/8 = control 3/8). ✓
- **Not caused only by loose zone entry** — the gain is in *certified strict* held-out coverage (held_strict median +1),
  not merely zone entry; fixed contact-quality is flat. ✓

## Honest scope + why this positive is trustworthy (unlike the gated case)
- The gain is **in-distribution generalization**: training on the generator's config distribution improves *held-out*
  (disjoint) instances of that same distribution — a legitimate, expected generalization signal, **not** a
  regression-to-mean artifact (the held-out set is hash-disjoint from training, the CI is above zero, and **no pair is
  negative**). Contrast the gated experiment, whose "positive" was vacuous (empty strong-basin group, CI spanning zero).
- **Scope, stated plainly:** the improvement is over the generator's own 3 config families (variants of 64102/64201/
  64111 by one relation + mirrors), not arbitrary new tasks. The **fixed 18-state** set does **not** improve (median 0) —
  configuration diversity buys generalization to new configs, not a lift on the original states. Contact-quality is
  unchanged, so the mechanism is coverage/distribution, not a cleaner grasp.
- Thread-pinning reduced (did not eliminate) the run-to-run noise; the fixed-state and 64102 endpoints remain noisy
  (as before), which is exactly why the robust signal is the **held-out coverage CI**, not any single pair.

## §10/§15 presentation videos (`..._generator_s0r0/videos/`, honest labels)
| file | policy / config | strict | attr | sha256 |
|---|---|---|---|---|
| `generator_certifies_held_attr19.gif` | GENERATOR, held-out ATTRIBUTION_BOUNDARY | YES | 0.77 | `84bf7a1d…` |
| `control_fails_held_attr19.gif` | CONTROL, same held-out config | no | 0.81 | `79882ee7…` |
| `generator_certifies_held_cert5.gif` | GENERATOR, held-out CERTIFIED_NEIGHBORHOOD | YES | 0.72 | `597f440c…` |
4 held-out configs are certified by GENERATOR-s2r0 and failed by CONTROL-s2r0 (2 CERTIFIED_NEIGHBORHOOD, 2 ATTRIBUTION_BOUNDARY).

## §13 next decision (GENERATOR_POSITIVE branch)
**Retain the 1-actor × 1-critic architecture** — the main limitation was configuration coverage, not model capacity.
**Next: test generator × `n_step=3`** as the single next change (one isolated variable), keeping the generator, on the
same matched 8-pair thread-pinned protocol. Do **not** implement the 2-actor × 2-critic factorial. Guardrails carried
forward: judge on the bootstrap CI (not the median sign); keep matched, thread-pinned, multi-rep pairs.

## Commits (branch `exp/coin-problem-generator`)
1. `5f33a20` connect the canonical HyMeKo problem generator · 2. `84e10ca` freeze corpora + 7 validity tests ·
3. `<runs>` launch 16 matched runs · 4. (this) report + videos.

## Provenance
Source commit `f054917`; base checkpoint `sac_actor_best.pt`; MuJoCo 3.10.0; CPU, threads=1; golden bit-identical; all
generator/replay/sac tests pass. `generator_comparison.json` holds every per-pair row + pooled stats + by-family.
