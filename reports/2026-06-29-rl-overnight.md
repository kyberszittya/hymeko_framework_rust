# Overnight RL session — coin-toss + pick-and-place (HSiKAN vs MLP)

**Date:** 2026-06-29 · **Wall:** ~3 h (launched 04:17, done 07:19) · **Git:** `9aea4f6` (dirty: session work staged) · **Seeds:** 0,1,2 · **Host:** local Windows, 16 cores, CPU-only.

## Summary

A full overnight RL session, run by the `run_overnight_rl.py` orchestrator: **coin-toss** (collaborative Galambos — single-agent HSiKAN vs 2-agent CTDE) and **pick-and-place** (FANUC arm, PPO warm-started from BC, **HSiKAN vs MLP**), 3 seeds each, with the three §9 outputs (numbers + plot + gifs). **15/15 subprocesses succeeded, zero errors.** New glue (this session, tested): `pick_place_ppo --out` (saves `.pt`+`.json`) and `render_pick_place --checkpoint` (renders the *trained* policy).

## Results

### Coin-toss — delivery rate (median of 3 seeds)
| controller | delivery | params |
|---|---|---|
| HSiKAN single-agent | **0.125** | 30 025 |
| collaborative CTDE (2 agents) | 0.042 | 20 489 |

Delivery is **low** for both, and the single agent **beats** the collaborative CTDE here. This is BC + only 20 PPO-refine iters — not enough to solve the two-arm hand-off.

### Pick-and-place — lift / place success (median of 3 seeds)
| policy | after BC (lift / place) | after PPO (lift / place) | params |
|---|---|---|---|
| HSiKAN | 0.875 / **0.625** | 0.000 / **0.000** | 30 991 |
| MLP | 1.000 / 0.500 | 0.125 / 0.000 | 20 495 |

PPO's **return** improves every seed (e.g. HSiKAN s0 −415 → −92), but **task success collapses to ~0** — the dense approach→grasp→lift→place reward is being optimised while the actual lift/place rate falls.

## Findings (honest)

1. **BC is the workhorse; PPO degrades success.** Behaviour cloning reaches 0.5–1.0 lift/place; PPO then drives it to ~0 while improving the scalar return. This is the **documented Phase-2 PPO-vs-BC regression** (`[[project-hymeko-rl-phase2-debug]]`) resurfacing — the dense return is **misaligned with success** (reward-hacking signature), not a wiring bug (the truncation/critic fixes are in; this is reward shape + budget).
2. **HSiKAN ≈ MLP — no structural advantage demonstrated.** Post-BC place: HSiKAN 0.625 vs MLP 0.500 (comparable; HSiKAN slightly higher but with **50 % more params**, 31 k vs 20 k). Consistent with the standing `[[project-galambos-hsikan-tie-rootcause]]` result: structure isn't load-bearing on these objectives.
3. **Collaborative < single on coin-toss** under this budget — the CTDE split needs more than BC+light-refine to coordinate two arms.

## Artifacts
- **Plot:** `reports/figures/overnight_rl.png` (coin-toss delivery; pick-place BC-vs-PPO place success).
- **Gifs (12):** `reports/gifs/overnight_coin_toss/seed{0,1,2}/coin_toss_{single_hsikan,collab_ctde}.gif`; `reports/gifs/overnight_pick_place/pick_{hsikan,mlp}_s{0,1,2}.gif`. *(Pick-place gifs render the PPO-final policy — i.e. the degraded one; the BC policy grasps better than the gif shows.)*
- **Checkpoints:** `checkpoints/pick_place_overnight/{hsikan,mlp}_s{0,1,2}.pt` + `.json`.
- **Data/log:** `reports/overnight/rl_session.{json,log}`.

## Provenance
- Config — coin-toss: 220 demos, 220 BC epochs, 24-eval, 20 PPO-refine, `--task-graph`. pick-place: BC warm-start (16 demos) + 80 PPO iters × 2048 steps.
- RL carve-out: seeds set; quantitative claims rest on the 3-seed medians above (not single-run reproduction).
- Run was throttled/suspended/relaunched once early (user needed the box); the reported run is the clean 04:17→07:19 relaunch.

## Follow-ups
- **The real blocker is PPO-degrades-success on the dense reward.** Options, in order: (a) a **success-aligned / sparser** reward or terminal bonus so return ⟺ success; (b) **more PPO budget** (80 iters is light); (c) **DAgger** instead of PPO past BC. Until then, **lead demos with the BC policy** (it actually grasps).
- Coin-toss: give the CTDE a real PPO budget (not 20 refine) before concluding single > collab.
- No new §6.5 anti-patterns; glue is in-place edits + 2 tests, ruff clean.
