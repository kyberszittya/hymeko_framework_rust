# Galambos collaborative grasp — reward the fingertip, not the nearest body

**Date:** 2026-06-26 · **Author:** Aiko (Claude Code) for Dr. Csaba Hajdu
**Plan:** `docs/plans/2026-06-26-galambos-fingertip-reward/` (tex/pdf/tikz/mmd) ·
**Status:** code + tests done; production-scale smoke passed (§ below)

## Summary

The collaborative (two-arm Galambos) scenario's dense approach reward `grasp_approach`
(= `−½(left_tip_dist + right_tip_dist)`) was measuring the **nearest arm body origin**, not the
fingertip. `compute_planar_metrics._nearest` took the min over each arm's `{base_*, upper_*, lower_*}`
body origins. Two measured consequences on the live `from_hymeko` env (difficulty 0.3, 6 seeds):

- The **emitted** arm ships **no fingertip site at all** (only `target_zone`), so the metric returned
  the `lower_*` body *origin* — the **elbow**, ~0.14 m (one forearm) short of the grasping point.
- Seed 5 (coin near centre, `(−0.037, 0.058)`): the nearest body was `base_*`, the **immovable**
  worldbody anchor → the approach term measured a **constant** → **zero gradient** to bring the tip to
  the coin. This is the user-reported "robot position is rewarded, not the tip of the arm."

**Fix (single root):** the emitted arm is missing its fingertip sites. `with_fingertip_sites` injects a
massless `tip_{side}` site at each arm's distal-link far end (idempotent; the hand-authored scene
already declares its own), and `compute_planar_metrics` now reports a **tip-dominant blend**
`0.75·fingertip + 0.25·elbow` (user-selected: the fingertip is the grasping point; the elbow keeps a
far-field gradient when the arm is fully extended). One change repairs three things: the reward metric,
the BC demonstrator (`galambos_demo._extract_arms` fell back to `tip_site=−1`, reading `target_zone`),
and `_extract_arms`'s link-length resolution.

A site is massless and collisionless → the **compiled dynamics are bit-identical** (verified:
`nq/nv/nu/nbody` unchanged, `nsite += 2`). Both backbones are shaped by the same reward, so this does
**not** explain HSiKAN-vs-MLP; it explains the low collaborative delivery (0.208 baseline). This is part
of the §0 per-task pipeline audit (handoff `reports/2026-06-26-session-handoff.md`).

### Measured before/after (tip vs old body-min, m)

| seed | coin | old nearest (body) | true fingertip | Δ |
|---|---|---|---|---|
| 2 | (0.13, 0.07) | 0.091 (`lower_right` = elbow) | 0.220 | +0.129 |
| 4 | (0.19, 0.07) | 0.076 (`lower_right`) | 0.216 | +0.140 |
| 5 | (−0.04, 0.06) | 0.230 (`base_right` = **fixed**) | 0.310 | +0.080 |

The old metric reported the arm "close" (0.076–0.091 m) when the fingertip was 0.22 m away — it was
rewarding arm-*folding*, not reaching.

## Files touched (CORE.YAML: none — verified non-core)

- `hymeko_rl/env/planar_grasp_env.py` — new `with_fingertip_sites` (ET site injection, ~40 LOC),
  `_vec3`/`_leaf_body` helpers, `_distal_body` method, `_TIP_BLEND=0.75`; `compute_planar_metrics`
  gains `tip_sites`/`elbow_bodies`/`tip_blend` and replaces `_nearest` with the tip+elbow `_approach`;
  `__init__` injects sites + resolves tip-site/elbow-body ids; `_metrics` passes them. (~80 new LOC.)
- `hymeko_rl/env/reward.py` — `_term_grasp_approach` docstring corrected (formula unchanged).
- `hymeko_rl/tests/test_planar_grasp_env.py` — 4 new tests (+78 LOC).

No new dependencies (`xml.etree.ElementTree` is stdlib). `galambos_demo.py` unchanged — the injected
site fixes its `−1` fallback for free (pinned by a test).

## Test results

- `pytest hymeko_rl/tests/test_planar_grasp_env.py -p no:randomly` — **26 passed** (22 prior + 4 new), 11.3 s.
  - `test_fingertip_sites_injected_and_idempotent` — emitted arm gains the sites; idempotent; hand-authored intact.
  - `test_fingertip_injection_preserves_dynamics` — `nq/nv/nu/nbody` identical, `nsite += 2`.
  - `test_approach_distance_is_tip_dominant_blend_not_body_min` — exact `0.75·tip+0.25·elbow`; on seed 2
    the blend strictly exceeds the body-min (**fails against the pre-fix `_nearest` metric** — the regression).
  - `test_extract_arms_resolves_fingertip_site` — demonstrator gets `tip_site ≥ 0` (was −1).
- Collateral suite (`-k "galambos or demo or planar or bc or render or grasp"`) — **74 passed**, 94 s. No breakage.
- Static gates: `ruff check` clean; `mypy` — only the pre-existing `mujoco` `import-untyped` baseline
  (every package file hits it), no new type errors.

## Performance

- Per-step added cost: 2 `site_xpos` + 2 `xpos` reads, 4 `hypot` — O(1), no allocation.
- Cart-pole control run (negative-control audit, separate): RSS 660 MB, < 16 GB cap.
- Production-scale galambos SAC smoke (1 seed, 2k steps, fixed env): **passed** — wall 282 s, RSS 700 MB
  (< 16 GB cap), HSiKAN curve_max −228.3 / final −259.4, MLP −241.5; **finite, no divergence**. (2k-step
  smoke → deeply negative returns by construction; this is the no-crash/finite/no-divergence gate, not a
  performance claim.)

## §0 audit context — cart-pole step result (banked)

Cart-pole (negative control, structure not load-bearing): **HSiKAN 200.0/200, MLP 200.0/200** (params
matched 13826 ≈ 13910). The HSiKAN backbone learns end-to-end → **no universal backbone bug**; any §0
problem is per-task wiring. Wiring statically clean (correct signed adjacency, feature↔vertex alignment).

## Open issues / follow-up

1. **Validation run (deferred):** short BC + off-policy refine on the fixed env vs the 0.208 baseline,
   with GIF + curve (§9 three-form). The reward-weight scale may need retuning now that the approach
   distance is larger when extended — the `.hymeko` weight is the knob (no code change).
2. Continue the per-task order: arm-movement → pick-and-place → quadruped (the §0 audit).

## Provenance

- Git: branch `fix-hsikan`; working tree dirty (intentional — prior-session banked changes per the
  handoff, plus this change). Files this change: the 3 above + plan dir.
- Env: Windows 11, Python 3.12, mujoco/torch (CPU). Seeds: env audit 0–5; cart-pole seed 0; smoke seed 0.
- No persistent state mutated (no checkpoints/datasets written by the audit or tests).
