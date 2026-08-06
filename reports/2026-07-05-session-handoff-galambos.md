# Session handoff — galambos coin-toss (2026-07-05, ~00:10 JST)

## Bottom line (read this first)

**Galambos is NOT working.** RL delivery is stuck **0.10–0.20 (noisy)**, *below* the scripted demonstrator's
**0.30**. The reward is now **correct** — farm-proof and oracle-certified — so **reward is no longer the blocker.**
The unsolved wall is **upstream** (BC / off-policy / the controller itself). The single most useful next action —
**not yet run** — is to eval the **BC-only** policy to localize where the 0.30 → 0.15 loss happens.

**Do not restart the reward work in the new session.** That loop is closed (details below).

## What is solid and committed (do NOT redo)

Branch `hymeko-neuro-migration` (off `fix-hsikan`; a WIP-snapshot commit preserves pre-session work). All green:

- **`hymeko_neuro` merge** (`e7e5835`) — signed_kan + signedkan_wip → one package, two distinct cores. Gates A–D pass.
- **Env constants ontology** (`8aa2b35`), **humanoid.hymeko** authored + validated 13-DOF (`e74f7ba`).
- **Reward is farm-proof + certified:** PBRS shaping (`e3ab7d4`), nonlinear **conjunctive** PBRS (`30f6adc`). The
  `reward_oracle` certifies whether a reward's *optimum* delivers in **ms** (P0 made it dwell-aware, `f11dcd8`).
- **Observability end-to-end:** demo progress print (`4320202`), BC per-epoch (`d553b1a`), off-policy `Q=/bc=`
  loss decomposition (`0849918`). No dark gaps.
- **Reward-alignment sphere viz** (`7f028d5`): only **~3% of reward-shape space delivers** — reward alignment is a
  needle in a haystack, which is why hand-tuning failed and the oracle matters.

## The core open problem (numbers)

| reward | delivery (seed 0, over 250k) | peak | both_contact |
|--------|------------------------------|------|--------------|
| baseline (farmable) | collapses → **0.00** | 0.16 | ~0.005 |
| PBRS (linear, farm-proof) | noisy, peak **0.20** | 0.20 | ~0.01 |
| Conjunctive (nonlinear, farm-proof) | noisy, peak **0.18** | 0.18 | ~0.01 (spiky 0.07) |
| **scripted demonstrator (the BC teacher)** | — | **~0.30** | 0.13 |

The trained RL policy delivers **worse than its own teacher.** PBRS gives a **healthy critic** (Q = +8, bounded) —
the opposite of the baseline's Q → −112 drift. So the reward is well-conditioned; the policy still doesn't learn
to reliably deliver, and `both_contact ≈ 0.01` means the two-finger grasp isn't forming.

## THE next test (do this FIRST — one measurement ends the guessing)

**Eval the BC-only policy** (collab actor, BC on ~16k demos, no off-policy), delivery on the baseline env:
- **BC-only ≈ 0.30** → the off-policy phase is **degrading** the clone → fix the trainer (stronger BC anchor /
  higher `bc_coef` / or just deploy the BC policy). The reward/RL machinery is then the culprit, not the task.
- **BC-only ≈ 0.10–0.20** → BC/arch isn't cloning the teacher, **or** the demonstrator's 0.30 is the real ceiling
  → the lever is a **better controller/demonstrator**, not reward or RL tuning.

Quick to run: `collect_galambos_demos` → `build_collaborative_offpolicy` → `behaviour_clone` → `eval_delivery`
(all in `hymeko_rl`). ~5 min. It removes the biggest unknown.

## What NOT to repeat (the circles this session ran)

- **Reward tweaking is DONE.** Farming is understood (canonical *specification gaming*), fixed (PBRS,
  Ng-Harada-Russell 1999), and certified. More reward variants will not lift delivery. The novelty search
  confirmed the reward-design/PBRS/oracle space is crowded — it's not a standalone paper; it's the reward layer
  of the HyMeKo-substrate paper.
- **The Q-negative-drift was a red herring** — I chased it hard and reframed it 3× before pinning it as benign
  (and PBRS gives healthy +Q anyway). Don't re-investigate the loss curve; watch **delivery**.
- **The demonstrator ceilings at ~0.30** — pinch-carry (0.30), shove (0.15–0.25), clamp/speed tuning all fail to
  beat it; the free-cylinder manipulation is genuinely hard. This is the suspected real wall.

## Likely levers (after the BC-only test localizes it)

1. **If off-policy degrades BC:** raise `bc_coef` (TD3+BC anchor), or deploy the BC policy directly, or shorten
   off-policy. The trainer is over-writing a good clone.
2. **If controller-bound (BC ≈ demo ≈ 0.30):** build a genuinely reliable coin-pushing controller (closed-loop:
   push from behind, correct heading, decelerate into the zone) as a stronger teacher — OR accept that two planar
   arms delivering a free cylinder to a small zone caps near 0.30, and reset the target.
3. **Sanity: is 0.30 even a fair target?** Measure the demonstrator ceiling honestly and set expectations before
   optimizing under a possibly-unreachable bar (evaluation-integrity §3).

## Honest note on this session

A lot got built and committed (merge, constants, humanoid, PBRS, oracle, conjunctive term, sphere viz,
observability), and the reward pathology is genuinely understood and fixed. But the session over-invested in the
**reward** while the delivery was capped by something **upstream** that was never measured (the BC-only number).
The next session should start there, not with more reward or arch changes. Several bugs slipped in mid-session
(a bad `--variant` arg, ruff misses committed then fixed, a stale test after a reward reshape) — real, and a sign
the reward-iteration loop had gone unproductive. Cut to the BC-only measurement first.

## Provenance
All committed through `4320202` on `hymeko-neuro-migration`. Experiment artifacts under
`experiments/2026_07_04_*_galambos_coord_ab_*/` (results.json + curves + gifs + run.log per run). Nothing running.
