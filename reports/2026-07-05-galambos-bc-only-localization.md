# Galambos BC-only localization — the 0.30 anchor was stale; the system is teacher-limited at ~0.21

**Date:** 2026-07-05 00:45 JST · **Branch:** `hymeko-neuro-migration` · **Commit under test:** `4320202` (dirty:
untracked checkpoints/experiments only; no source changes) · **Host:** Windows 11, local CPU run.

## Summary

Ran the handoff's decisive test (BC-only delivery, never measured before — `Campaign` goes straight from
`behaviour_clone` into off-policy training) **plus** a re-anchor of the scripted demonstrator under the *same*
dwell protocol. Verdict: **there is no upstream BC/off-policy pathology.** The demonstrator itself delivers
**≈0.21** under current physics — not the ~0.30 the handoff carried — and BC (0.12 median) and RL (0.16–0.20
peaks, 2026-07-04 runs) sit at that teacher ceiling. The wall is the **demonstrator/controller under
fingertip-only physics**, exactly the 2026-07-03 open item ("re-tune the demonstrator") that was never done.

## Measured (all: dwell rule `DwellMetric(in_zone, success_steps)`, baseline env, difficulty 0.3, 300 steps)

| policy | delivery | n episodes | both_contact |
|---|---|---|---|
| scripted demonstrator (pooled, 4 disjoint seed sets) | **0.205** (0.16 / 0.20 / 0.20 / 0.26 per 50-ep set) | 200 | 0.134 (12 eps, seed 9000) |
| demonstrator @ eval protocol (seed0 9000) | 0.160 | 50 | — |
| BC-only clone (collab `sa_hsikan` h=64, 200 demo eps → 3204 transitions, 200 epochs) | **median 0.12** [0.10, 0.12, 0.18] (seeds 0/1/2) | 50/seed | 0.011 / 0.019 / 0.000 |
| TD3+BC RL peaks (2026-07-04 A/B runs, prior measurement) | 0.16–0.20 | 50 | ~0.01 |

## Inference chain (measured → inferred → hypothesis)

- **Measured:** demonstrator ≈ 0.205 over 200 episodes; BC ≈ 0.12; RL peaks 0.16–0.20. All at the same level.
- **Measured:** fingertip-only collision is the **env default** (coin movable only by fingertip geoms,
  `planar_grasp_env.py` ~line 173); the demonstrator has **no re-tune commits** after the 2026-07-03 change that
  the 2026-07-03 report itself flagged as breaking it.
- **Inferred:** the handoff's "demonstrator ~0.30" was a small-n point estimate (0.30 on a 24-ep probe is ~25%
  likely under a true p = 0.205) and/or carried forward from pre-fingertip-change measurements (2026-07-01 era
  0.33; `galambos_bc.py` docstring "~25%"). Likewise the 2026-07-03 collab **0.40** most plausibly predates the
  fingertip-only default — the 2026-07-04 re-run under current physics reproduced only 0.16, and blamed config
  mismatch; the physics change is the more parsimonious cause.
- **Measured (grasp):** the clone does **not** reproduce the pinch — `both_contact` 0.00–0.02 vs teacher 0.134.
  It reaches ~0.12 delivery by pushes alone.
- **Hypothesis (next lever):** delivery beyond ~0.2 requires a demonstrator redesigned for fingertip-only
  manipulation (closed-loop push-from-behind / stable pinch), then re-BC. Secondary: 3204 transitions is a small
  BC corpus; more demo episodes may close the 0.12 → 0.16 clone gap, but cannot pass the 0.21 teacher ceiling.

## What this closes

- **Do not** chase a BC/off-policy trainer bug — RL ≈ BC ≈ teacher. The handoff's "RL below its own teacher"
  framing dissolves once teacher and policy are graded by the same protocol at adequate n (evaluation-metric
  integrity, CLAUDE.md §3: measure the ceiling before optimizing under it).
- Reward remains closed (per handoff). Upstream trainer remains closed (this report). The open item is the
  **controller/demonstrator for fingertip physics** (2026-07-03 next-step #2, still open).

## Presentation numbers (for Kato & Galambos)

Honest current state, one line: *"Under fingertip-only physics (arm bodies cannot touch the coin), the scripted
teacher delivers 0.21; behaviour cloning reaches 0.12 and TD3+BC RL 0.16–0.20 — the learning stack matches its
teacher, and the ceiling is the teacher, not the learner."* Artifacts: bar plot
`experiments/2026_07_05_00_16_galambos_bc_only/bc_vs_demo.png`, GIF `.../gifs/bc_only_s1.gif` (best clone),
prior RL GIFs under `experiments/2026_07_04_*_galambos_coord_ab_*/gifs/`.

## Files touched

- **No source changes.** Measurement driver: `experiments/2026_07_05_00_16_galambos_bc_only/bc_only_delivery.py`
  (one-off, reuses `collect_galambos_demos` / `behaviour_clone` / `eval_delivery` / `eval_metric` /
  `build_collaborative_offpolicy` / `render_actor_gif` — no new library code, §6.1). No §6.5 anti-patterns
  introduced; no new suppressions; no CORE.YAML items touched. No new tests required (no new/modified source).

## Provenance

- Git SHA `4320202`, branch `hymeko-neuro-migration`. Seeds: BC 0/1/2; eval seed0 9000 (+0/3000/20000 for the
  demonstrator pooling). Wall: 774 s (main run) + ~6 min (anchor checks). Peak RSS not instrumented (CPU-only,
  well under cap by construction — 3204-sample BC; same code path as the passing A/B runs).
- Run log: `experiments/2026_07_05_00_16_galambos_bc_only/run.log`; results:
  `experiments/2026_07_05_00_16_galambos_bc_only/results.json`.
