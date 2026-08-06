# HSiKAN vs MLP on turning — TIED (no structural advantage), and the residual doesn't beat the scaffold

**Date:** 2026-07-29 (JST) · **Branch:** `research/aibo-lyapunov-ph` (worktree `hymeko_aibo`)
**SIMULATION. 3 seeds, 25k.** · **Verdict: `HSIKAN_TIES_MLP_ON_TURNING; RESIDUAL_DOES_NOT_BEAT_THE_SCAFFOLD`.**

---

## The test (user's hypothesis)

The crab was a simple lateral push (HSiKAN ≈ MLP). Turning is **whole-body coordination** — route the
heading error through the body's kinematic structure into a coordinated per-leg stride correction — so
(hypothesis) HSiKAN's structural propagation might finally beat a flat MLP here. A bounded **phase**
residual (scale 0.12, preserves the limit cycle) over the working `turn_then_walk` scaffold (a=0 = 0.50),
WIDE bearings (±135°); **MLP** (flat obs) vs **HSiKAN** (signedkan over the full **33-vertex body
hypergraph**, per-joint qpos/qvel — the right structural obs for turning); same pooled head, same 12-dim
action; goal-reach on the wide grid. Selection by dense mean-min-distance (after the first run's
all-zero-VAL selection bug — see `result_turn_rl_flawed.json`).

## Result — tied, and neither beats the deterministic scaffold

| recipe | seed 0 | seed 1 | seed 2 | median | max |
|---|---|---|---|---|---|
| MLP (flat) | 0.389 | 0.444 | 0.444 | **0.444** | 0.444 |
| HSiKAN (body-hg) | 0.389 | 0.333 | 0.556 | **0.389** | **0.556** |
| scaffold (a=0 turn_then_walk) | — | — | — | **0.50** | — |

Two honest readings:

1. **The bounded residual does not beat the deterministic scaffold.** Both medians (MLP 0.444, HSiKAN
   0.389) are **below** the a=0 scaffold's 0.50 — the gentle residual slightly *disrupts* more than it
   helps. The rotational-couple `turn_then_walk` is already near the ceiling of what's reachable: the
   ±135° goals need ~135° of turning ≈ 2870 steps at the stable ~47°/1000 rate, **beyond** the 2400-step
   horizon, so they are largely unreachable regardless of the residual. The residual is chasing headroom
   that mostly isn't there.
2. **HSiKAN ≈ MLP — no clear structural advantage, even on turning.** MLP has the better median (0.444 vs
   0.389); HSiKAN has the better single seed (0.556 — the *only* run to exceed the scaffold) but also the
   worst (0.333). High variance, overlapping ranges, 3 seeds. The user's hypothesis (structure helps the
   whole-body turning coordination where it didn't for the crab) is **not confirmed**: HSiKAN ties MLP on
   turning as it did on the crab. The faint hint — HSiKAN's best run uniquely beat both the scaffold and
   the MLP — is within seed noise and not a result.

## What actually moved goal-reaching

Not the architecture and not the RL. The **deterministic rotational-couple `turn_then_walk`** (0.11 →
0.50) is the real win of this refocus; the learned residual, of either backbone, does not improve on it
here. Across the whole AIBO campaign the robust finding stands: **HSiKAN ≈ MLP on these control tasks** —
the structural prior neither helps the crab (simple push) nor the turn (coordination), and the leverage
was always in the *mechanism / action space* (the rotational couple), not the policy class.

## Ceiling test (6 seeds) — the "HSiKAN uniquely beats the scaffold" hint was seed noise

The 3-seed run had HSiKAN's best (0.556) as the *only* run to exceed the 0.50 scaffold (MLP max 0.444) —
a genuine-looking ceiling signal, worth testing. Rerun at **6 seeds**, saving checkpoints + the
`beats_scaffold` count:

| | reaches (6 seeds) | median | max | beats scaffold |
|---|---|---|---|---|
| MLP | 0.389, 0.444, 0.444, 0.5, **0.556**, 0.444 | 0.444 | 0.556 | **1/6** |
| HSiKAN | 0.389, 0.333, **0.556**, 0.5, 0.5, 0.222 | 0.444 | 0.556 | **1/6** |

**The hint dissolved.** With more seeds the MLP *also* reaches 0.556 and beats the scaffold (seed 4) — so
"HSiKAN uniquely exceeds the scaffold" was a 3-seed artifact. Both beat it **exactly 1/6**, identical
median (0.444) and max (0.556); HSiKAN is if anything **more variable** (worst 0.222 vs MLP 0.389).

**Winner analysis — the clincher.** The two 0.556 winners (HSiKAN seed 2, MLP seed 4) reach the **same
goals**: both *gain* exactly (90°, 0.5) and (−20°, 0.7) — the scaffold's near-misses (90° sat at 0.158,
just over the 0.12 threshold) — and each *loses* one far goal (40°/−40° at 0.7). The HSiKAN winner does
**nothing structurally different** from the MLP winner; both just nudge the same near-misses over the line
at the cost of a far goal. No structural mechanism, same trade, same goals.

## Honest caveats

3 seeds (below the repo's 5-iteration bar) — but the qualitative reading (tie; residual ≤ scaffold) is
consistent across seeds. The residual is bounded (0.12) over a strong scaffold with a horizon-limited
wide-bearing tail, so there is little room for *either* architecture to separate — a fairer HSiKAN test
might be a from-scratch turning policy (no scaffold, more authority), or a longer horizon so ±135° is
reachable. Recorded as the open follow-up, not run.

## Files

```
scenarios/aibo/run_aibo_turn_rl.py                          MOD  dense-mean-dist selection, phase residual (fixed setup)
reports/2026-07-29-aibo-turn-rl/result_turn_rl.json         NEW  the fixed 3-seed result
reports/2026-07-29-aibo-turn-rl/result_turn_rl_flawed.json  NEW  the first (buggy-selection) run, kept honestly
```

CPU, seeds 0-2, 25k; HSiKAN 33-vtx body hg ~52 steps/s. CORE.YAML: none. SIMULATION.
