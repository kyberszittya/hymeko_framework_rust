# HTL-robustness as the reward — galambos proof-of-concept

**Date:** 2026-06-26 · **Author:** Aiko (Claude Code) for Dr. Csaba Hajdu
**Plan:** `docs/plans/2026-06-26-htl-reward-poc/` (tex/pdf/tikz/mmd) · **Status:** adapter + tests done; A/B smoke passed

## Summary

A working proof-of-concept that a **geometric temporal-logic formula** can drive RL reward. The thesis:
the quantitative semantics of an STL/HTL predicate `x ≥ θ` is its robustness `ρ = v(x) − θ` — a **signed
geometric margin** — so one declared formula yields *both* the dense training signal (`ρ`) and the monitor
verdict (`sign ρ`). This collapses reward shaping (`reward.py`) and runtime accountability (`hymeko_monitor`
/ HTL) into a single artifact.

**Reuse, not rebuild (§6.1).** The HTL evaluator already exists, non-core, in `signedkan_wip/src/htl/`
(`parse`, `robustness_at`, `HtlMonitor`, `HypergraphEvent`; `AND/OR/NOT/G[a,b]/F[a,b]` over `ScalarPred`,
robust-STL min/max). This change is a thin **adapter** plus a `.htl` spec — no new logic engine. The bridge
to that package mirrors the existing `hymeko_ros2_demo/.../dashboard_node.py` `sys.path` pattern. The
module-level predicate **registry is bypassed** (§6.5 #11): metrics travel via `event.scalar_signals`.

### Key design decisions (analyze, don't declare)

- **Per-step reward = `robustness_at(formula, event)`** — the *instantaneous* robustness. Temporal `G`/`F`
  collapse to their leaf at a single event, so the reward is a **Markovian** min/max composition of geometric
  margins — valid for the off-policy replay buffer. The genuinely temporal `ρ` over a rollout
  (`F[0,T](in_zone)`) is **non-Markovian** (a reward machine) and is exposed only as the per-episode
  *verdict*, not the reward — that is the FSM-structured-RL line (`docs/plans/2026-06-23-fsm-structured-rl/`),
  deliberately out of scope.
- **Dense leaves, not binary.** `AND = min`, so a binary `in_zone` leaf would be a constant −0.5 masking the
  dense terms. The reward formula uses dense margins (`disk_to_zone`, `approach_l/r`); `min` then
  auto-focuses on the currently-worst subgoal — a natural curriculum (approach first, then deliver),
  pinned by a unit test.
- **The formula's leaves are the same metrics the flat terms read** (incl. the just-fixed tip-blend
  `approach_l/r`), so the A/B isolates *composition*: STL `min/max` + temporal vs `Σ wᵢ`.

The spec (`data/robotics/galambos_spec.htl`):
```
disk_to_zone < 0.055 AND G(disk_oob < 0.5) AND approach_l < 0.06 AND approach_r < 0.06
AND (disk_to_zone > 0.055 OR disk_speed < 0.3)
```

## Files touched (CORE.YAML: none — grammar untouched; `.htl` parsed by non-core Python HTL)

- **New** `hymeko_rl/htl_reward.py` — `HtlRewardSpec` (duck-types `RewardSpec.evaluate`, drops into the env's
  `reward_spec=` seam with no env change), `signals_from_planar`, `episode_monitor`. (~115 LOC.)
- **New** `data/robotics/galambos_spec.htl` — the task formula.
- **New** `hymeko_rl/exp_htl_reward_ab.py` — flat-vs-HTL A/B (reuses SAC trainer, `evaluate`,
  `render_actor_gif`, `plot_scoreboard`; §9 three-form). (~120 LOC.)
- **New** `hymeko_rl/tests/test_htl_reward.py` — 6 tests.
- **No change** to `planar_grasp_env.py` or `signedkan_wip/src/htl/` (consumed as-is).

## Test results

- `pytest hymeko_rl/tests/test_htl_reward.py -p no:randomly` — **6 passed**, 6.2 s.
  - spec parses (comments stripped); signals extractor complete + finite, binaries 0/1.
  - robustness sign: far < 0 < delivered (directed).
  - `min` focuses on the worst subgoal (closing the fingertips raises ρ when approach is binding).
  - **duck-types into the env**: `PlanarGraspEnv(reward_spec=HtlRewardSpec())` steps with finite reward, no
    env edit (the seam accepts any `.evaluate`).
  - episode monitor verdict: satisfied on a delivering trace, unsatisfied on a never-in-zone trace.
- Static: `ruff check` clean (adapter, harness, tests); `radon cc` — no flagged complexity.

## Performance

- Per-step reward cost: build an ~8-key dict + one AST walk over a 5-leaf formula — O(formula), µs, no torch.
- A/B smoke (flat vs HTL, 2k steps each, HSiKAN SAC, seed 0, 12 eval eps): **passed** — wall 594 s, RSS
  792 MB (< 16 GB). Both 0% delivery (2k steps ≪ the ~30k the grasp needs — expected; this is the
  plumbing/finite/no-divergence gate). 3-form output written to `reports/htl_reward_ab/`
  (`htl_reward_ab.json`, `scoreboard.png`, `galambos_{flat,htl}.gif`). **Returns are NOT comparable**
  (`htl` −22.9 vs `flat` −239 is pure reward *scale* — ρ in metres vs the tuned weighted sum); delivery is
  the comparable metric and is 0/0 at smoke scale. The flat-vs-HTL delivery verdict needs the full
  multi-seed run (follow-up #1).

## Open issues / follow-up

1. **A/B result + multi-seed.** The smoke proves plumbing + 3-form output; a multi-seed full run is the
   real comparison (delivery flat vs HTL).
2. **Hard-min bottleneck → smooth robustness.** STL `AND = min` shapes only the worst conjunct; if the A/B
   shows slower learning, the fix is smooth/soft-min (log-sum-exp / AGM) — a **noted follow-up**, not
   pre-optimized (§6.3 forbids defensive optimization).
3. **Declaring the formula in `.hymeko`** (a `monitor{}` block) is the grammar step — CORE-gated, deferred
   (FSM-structured-RL plan P1). Today the formula lives in a non-core `.htl` companion.

## Provenance

- Git: branch `fix-hsikan`; working tree dirty (prior-session + this session's galambos-fingertip + this PoC).
- Env: Windows 11, Python 3.12, mujoco/torch (CPU). Seed 0. No persistent state mutated by tests.
