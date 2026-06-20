# Galambos: overshoot-brake attempt (negative) + best-run GIFs

*2026-06-20 · Aiko (Claude Code) for Dr. Csaba Hajdu*

## Summary

Two things: (1) the attempt to push the goal rate past 5/8 with a `settle` (overshoot
brake) reward term — **negative in both gatings (4/8 < 5/8)**, reverted out of the
canonical task; (2) a GIF renderer for the best runs, with per-run folders of rendered
rollouts.

## 1. Settle term — negative result

Diagnostic showed the 3 non-goal episodes were control-precision misses (overshoot,
wrong-direction, undershoot). The lever tried: penalise coin speed so it decelerates
into the zone instead of sailing through.

| Variant | Gate | Goals |
|---|---|---|
| Baseline (approach + pull) | — | **5 / 8** |
| `settle` weight 0.3 | near zone (`dist < 2·zone_half`) | 4 / 8 |
| `settle` weight 0.3 | inside zone (`dist < zone_half`) | 4 / 8 |

**Both worse.** The near-zone gate braked during approach → **undershoot** (coins stalling
at the zone boundary, ~0.057 vs the 0.055 threshold). The in-zone gate avoided that but
still landed at 4/8 (a different episode dropped). The overshoot brake trades one
precision failure for another; it does not raise the rate at this weight/formulation.

**Decision:** reverted `@settle` out of `galambos_task.hymeko` — the 5/8 reward is the
canonical task. The term is **kept as opt-in infrastructure**: the `settle` kind
(`meta_reward.hymeko`), its extractor (`reward.py`), and the `PlanarGraspMetrics.disk_speed`
field it reads remain, with unit tests, so a future *tuned* attempt (lower weight,
velocity-only-while-exiting, or a directional term) can use them. This mirrors how the
safety terms default to weight 0 and tasks opt in.

**Honest standing:** two levers tried to beat 5/8 — curriculum (null) and settle
(negative). The 5/8 plateau is robust to these reward-shaping tweaks. The remaining
real lever is structural (a true two-sided pinch instead of open-loop pushing —
`both_contact` is still 0), which is a larger change, not a reward knob.

## 2. Best-run GIFs

New `hymeko_rl/render_planar_gifs.py` — loads a checkpoint, builds `PlanarGraspEnv`, and
renders rollouts to GIFs with a **top-down camera** matching the table, **one folder per
run**. Reuses `evaluate.render_episode_gif` (env-agnostic offscreen renderer); no new
render infrastructure.

Rendered the two best (5/8) runs, goal seeds:

- `reports/gifs/galambos_freed/` — `seed_{1000,1003,1005,1006,1007}_goal.gif` (returns 36–40).
- `reports/gifs/galambos_curriculum/` — same seeds (returns 39–43).

The settle runs (4/8) were **not** rendered — they are not best runs.

    python -m hymeko_rl.render_planar_gifs --checkpoint <ckpt> --run <name> --seeds 1000 1003 ...

## Files touched

| File | Δ | Note |
|------|---|------|
| `hymeko_rl/env/planar_grasp_env.py` | +~6 | `PlanarGraspMetrics.disk_speed` + disk dof addrs |
| `hymeko_rl/env/reward.py` | +~12 | `settle` extractor (kept, opt-in) |
| `data/robotics/meta_reward.hymeko` | +4 | `@settle` term kind (opt-in) |
| `data/robotics/galambos_task.hymeko` | +3/−1 | settle tried then **reverted** (note left) |
| `hymeko_rl/render_planar_gifs.py` | +90 (new) | GIF renderer, per-run folders |
| `hymeko_rl/tests/{test_planar_grasp_env,test_render_planar}.py` | +~45 | settle + render tests |
| `reports/gifs/galambos_{freed,curriculum}/` | new | 10 rendered goal GIFs |

## CORE.YAML / dependencies

**None.** All `hymeko_rl/` + `data/robotics/` (non-core). GIFs via Pillow (already a dep);
no imageio/mp4.

## Test results

- Full `hymeko_rl` suite — **116 passed** (incl. settle-term, disk-speed metric,
  top-down-camera, and a GL-gated render-produces-valid-GIF test).
- `hymeko validate data/robotics/galambos_task.hymeko` — ✅.
- `ruff` + `mypy --strict` on changed code — clean (pre-existing `mujoco` import-untyped
  notes only).

## Open / follow-up

- A **true two-sided pinch** (vs the current push) is the structural lever for >5/8 and
  for `both_contact` > 0 — a model/reward change, not a knob.
- A tuned settle (velocity-only-while-exiting-the-zone, or directional) could be retried
  with the kept infrastructure, but is deprioritised after two negative knob attempts.

## Provenance

Git branch `soma-vision`; tree dirty (pre-existing). CPU MuJoCo, no GPU. Seeds fixed.
Checkpoints: `ppo_freed.pt` (5/8, best), `ppo_curriculum.pt` (5/8), `ppo_settle.pt` /
`ppo_settle2.pt` (4/8, negative).
