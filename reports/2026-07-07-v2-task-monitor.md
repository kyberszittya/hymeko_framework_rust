# Hierarchical formal task monitor — external verifier over trajectories (galambos coin delivery)

**Date:** 2026-07-07 · Git SHA `4320202` (working tree dirty; new package + tests + report). One-machine eval,
CPU rollouts, no training. RL stays frozen — the monitor is an **external evaluator**, not a reward.

## Summary

Built a **hierarchical formal task monitor** as a core ecosystem component, generated from the HyMeKo task
contract on the env (single-source-of-truth). It is the third leg of an explicit three-way separation:

> **reward** = what RL optimises · **metrics** = what we report · **formal monitor** = an external deterministic
> verifier over trajectories (a top-down camera).

The monitor consumes a **position-based trajectory tensor** (positions / velocities / contacts / distances /
role labels — *no policy internals, no reward*) and returns, per aspect: PASS/FAIL, a continuous robustness
score, a violation reason, relevant time indices, and trajectory slices. A root `TaskMonitor` composes the
submonitors into `monitor_pass` + `monitor_score` + the five aspect scores + `violation_reason`.

Evaluated all 7 requested policies through it and produced the reward-vs-monitor table + scatter and the two
cross-cutting alignment verdicts. **Headline: the monitor independently confirms both misalignments the RL
debugging hypothesised** — the dense reward ranks the body-shove exploit *above* the DAgger policy, and the
failed RL critic Q ranks the exploit *above* DAgger — while the monitor (like `fingertip_dominant_delivery`)
correctly ranks DAgger above the exploit.

## Architecture (hierarchy, not a monolith)

Package `hymeko_rl/eval/task_monitor/` (replaces the earlier monolithic `task_monitor.py`). Strategy + Composite:
each submonitor is a `TrajectoryMonitor` Strategy reading a shared, precomputed struct-of-arrays
`MonitorContext`; the root `TaskMonitor` composes them.

| # | Submonitor | Checks | Gates PASS? |
|---|---|---|---|
| 1 | `GeometryMonitor` | coin-target dist, L/R fingertip-coin dist, arm-body-coin, zone membership | base facts (no) |
| 2 | `ApproachMonitor` | fingertips approach coin; coin not pushed away; approach before displacement | yes |
| 3 | `ContactMonitor` | L / R fingertip contact, both-fingertip engagement, duration, timing | yes |
| 4 | `ProgressMonitor` | dist decreases; target-directed progress; fingertip- vs body-attributed | yes |
| 5 | `DeliveryMonitor` | coin enters zone, holds k steps, final dist, stable delivery | yes |
| 6 | `AntiExploitMonitor` | body-driven / body-assisted / arm-shove / no-engagement delivery | yes |
| 7 | `RewardConsistencyMonitor` | reward-vs-monitor + critic-vs-monitor rank inversions (cross-cutting) | — |
| 8 | `TensorContractMonitor` | obs / privileged-z / reward-feature / contact-quality schema + field-order hashing | — |

- **Single-source:** `MonitorContract.from_env` reads `zone_half` + `success_steps` from the env's HyMeKo/EnvSpec
  contract; `TensorContract.from_env` reads obs / privileged / node-feature dims. Contact roles come through the
  `ContactLegalitySpec` on the recorded trajectory. No thresholds are hand-duplicated from the env.
- **Task formula (PASS):** fingertips approach ∧ both fingertips engage ∧ coin moves toward target ∧ progress is
  fingertip-dominant ∧ body-only progress ≤ ε ∧ coin enters zone ∧ holds k steps ∧ no body-driven exploit.
- **Score convention:** each aspect is `tanh`-squashed to `>0` good / `<0` bad, so aspects are comparable;
  `monitor_score` = mean of the five gating aspects.

## Reward-vs-monitor table (7 policies, N=24, eval seeds 9000+)

| policy | reward | ft_dom | mon_pass | mon_score | appr | contact | progress | delivery | anti_exploit | top violation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| scripted_v2b | −98.8 | **0.792** | **0.500** | **0.345** | 0.035 | −0.038 | 0.355 | 0.688 | 0.684 | no_right_fingertip_contact |
| mlp_dagger_selected | −115.2 | 0.750 | 0.417 | 0.278 | 0.004 | −0.318 | 0.398 | 0.688 | 0.617 | no_right_fingertip_contact |
| mlp_dagger_best | −115.2 | 0.750 | 0.417 | 0.278 | 0.004 | −0.318 | 0.398 | 0.688 | 0.617 | no_right_fingertip_contact |
| mlp_bc0 | −209.1 | 0.208 | 0.125 | 0.086 | −0.040 | 0.085 | −0.068 | −0.125 | 0.578 | fingertips_never_approached |
| body_shove_exploit | **−111.4** | 0.125 | 0.083 | −0.202 | −0.299 | −0.844 | 0.130 | 0.312 | −0.310 | fingertips_never_approached |
| one_fingertip | −330.4 | 0.042 | 0.000 | −0.273 | −0.076 | −1.000 | 0.010 | −0.438 | 0.139 | no_left_fingertip_contact |
| failed_ctde_td3bc | −343.0 | 0.042 | 0.042 | −0.310 | 0.016 | −0.763 | −0.316 | −0.188 | −0.299 | fingertips_never_approached |

Scatter: `reports/figures/task_monitor/reward_vs_monitor_scatter.png` (reward x, monitor score y, colour = PASS
rate). JSON: `reports/figures/task_monitor/monitor_eval.json`.

## Alignment verdicts (cross-cutting monitors)

- **`TensorContractMonitor.check_env` → PASS.** obs / privileged-z(5) / node-feature(8) dims match the declared
  schema; privileged field list length matches `privileged_dim`. The tensor contract holds at the env boundary.
- **`RewardConsistencyMonitor` reward-vs-monitor → MISALIGNED** (concordance 0.714). The dense reward **prefers
  the body-shove exploit (−111.4) over both `mlp_bc0` (−209.1) and the DAgger policy (−115.2)**, while the monitor
  ranks the exploit strictly *below* them (−0.202 vs +0.086 / +0.278). The reward's dense annuity terms make a
  body-driven push look ≈ as good as a fingertip-dominant delivery; the monitor does not.
- **`RewardConsistencyMonitor` critic-vs-monitor → MISALIGNED.** Using the measured failed-RL critic Q on DAgger
  states (red-team test 3, `reports/2026-07-07-v2-rl-smoke.md`): **Q ranks `body_shove_exploit` (−4.997) above
  `mlp_dagger_selected` (−5.65)**, whereas the monitor prefers DAgger. This is the **category-B off-policy OOD
  overestimation** from the red-team, now confirmed by an *independent* external verifier rather than the ft_dom
  metric alone — the critic is misaligned exactly where the exploit lives.

**Interpretation.** The monitor score and `fingertip_dominant_delivery` agree on the policy ranking
(scripted ≳ DAgger ≫ bc0 > exploit ≈ RL ≈ one-finger); the **reward and the failed RL critic do not**. The two
misalignments are the concrete, quantified mechanism behind "RL degrades this task": both the training signal and
the learned value rank a body-driven exploit above the imitation policy.

**Monitor stricter than the metric (by design).** The scripted ceiling scores `ft_dom` 0.792 but `monitor_pass`
only 0.500 — the ContactMonitor additionally requires *simultaneous two-fingertip* engagement, which the scripted
controller does not always exhibit even when its progress is fingertip-attributed. The monitor verifies the
manipulation *mode*, not just the outcome attribution; the gap (0.792 → 0.500) is a real, useful strictness, not
a bug. `no_right_fingertip_contact` is the dominant residual violation for the two good policies.

## Files touched

| file | LOC | note |
|---|---:|---|
| `hymeko_rl/eval/task_monitor/__init__.py` | 56 | package facade + re-exports + `monitor_policy` back-compat |
| `hymeko_rl/eval/task_monitor/contract.py` | 52 | `MonitorContract`, `TensorContract` (both `from_env`) |
| `hymeko_rl/eval/task_monitor/context.py` | 130 | `record_trajectory`, `MonitorContext` (SoA precompute), `contiguous_runs` |
| `hymeko_rl/eval/task_monitor/submonitors.py` | 184 | `SubVerdict`, `TrajectoryMonitor` ABC, 6 submonitors, `default_submonitors` |
| `hymeko_rl/eval/task_monitor/consistency.py` | 117 | `RewardConsistencyMonitor`, `TensorContractMonitor` |
| `hymeko_rl/eval/task_monitor/root.py` | 105 | `TaskVerdict`, `TaskMonitor` (compose + aggregate) |
| `hymeko_rl/tests/test_task_monitor.py` | 182 | 14 unit tests (env-free synthetic trajectories) |
| `scratchpad/monitor_eval.py` | — | 7-policy eval harness (not committed; artifacts under `reports/figures/`) |

Deleted: the monolithic `hymeko_rl/eval/task_monitor.py` (superseded by the package; uncommitted, same session).

**CORE.YAML items touched:** none. New non-core package under `hymeko_rl/eval/`; no dependency changes.

## Tests

- **Unit:** `pytest hymeko_rl/tests/test_task_monitor.py -p no:randomly` → **14 passed in 2.9 s**. Covers every
  submonitor pass+fail path, `MonitorContext` progress attribution, empty-trajectory rejection, root aggregation
  (good delivery PASS / exploit FAIL / no-delivery FAIL), and both consistency monitors incl. **directional
  wording** of the reward and critic inversion messages.
- **Static:** `ruff check hymeko_rl/eval/task_monitor/` → clean.
- **Integration:** the 7-policy `monitor_eval.py` run is the real-env exercise (v2 graded scene + v2b reward,
  N=24 each), artifacts on disk.
- **Bug fixed mid-task:** the critic/reward inversion messages printed the direction backwards (said "Q ranks A
  above B" when Q ranked B above A). Fixed to compute the higher-ranked side from the actual values; locked by an
  exact-string test. The *detection* (concordance, PASS/FAIL) was correct throughout; only the explanation text
  was wrong.

## Performance

Not a perf-sensitive path (per-episode eval, ~300 steps). `MonitorContext` precomputes SoA arrays once per
trajectory; submonitors read slices (no re-walk). 7 policies × 24 episodes ran in ≈ CPU-bound minutes with peak
RSS well under the 16 GB cap (single MuJoCo env at a time). No numerical budget asserted — the monitor is a
verifier, not a hot loop.

## How the monitor is used from here (per the directive)

External evaluator first: (1) diagnostics, (2) acceptance tests, (3) reward-vs-monitor misalignment detection,
(4) critic-vs-monitor misalignment detection — all delivered above. Shielding / reward-shaping is explicitly
**later and optional**; the monitor is **not** in the reward.

**RL acceptance gate (from now on, before any TD3/SAC/residual RL is accepted):** a run must show
(a) `fingertip_dominant_delivery` ≥ the DAgger baseline, (b) `monitor_pass_rate` not below DAgger's 0.417,
(c) `monitor_score` not worse, and (d) no increase in body-driven violations (`AntiExploitMonitor`). The current
failed CTDE-TD3+BC fails all four — consistent with the frozen "RL stays frozen" verdict.

## Open items / follow-up

- `TensorContractMonitor.verify_stages` is unit-tested on synthetic field tuples but **not yet wired into the
  live rollout → replay → critic → eval pipeline**. Wiring the field-order hash into those five stages is the
  next hardening step (would have caught the scripted-vs-DAgger anchor provenance bug earlier).
- Queued research (not now): HSiKAN structured-tensor A/B; if RL is revisited, a **conservative critic (CQL) /
  residual / phase-gated** actor — and its acceptance is now the monitor gate above, not reward alone.
