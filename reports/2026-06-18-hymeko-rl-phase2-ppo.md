# hymeko_rl Phase 2 — in-repo PPO + the matched-capacity ablation (honest negative)

**Date:** 2026-06-18
**Plan:** [docs/plans/2026-06-18-mujoco-rl-grasping](../docs/plans/2026-06-18-mujoco-rl-grasping/).
**Status:** ⚠️ PPO is built, tested, and the loop works — but it is **not yet effective**
on the reaching task: it improves from a random start yet **degrades a BC-pretrained
policy**. So the HSiKAN-vs-MLP architecture ablation is **inconclusive**. Reported as the
measurement contradicts the plan's expectation (CLAUDE.md §11), with the diagnosis and
the fix, not a manufactured win.

## What was built
`hymeko_rl/ppo.py` — a minimal in-repo PPO (clipped surrogate + GAE), one file over the
pinned torch, no heavy RL dependency. It trains *either* policy on the same `ArmReachEnv`
reward (fix the algorithm, ablate the architecture); `ent_coef` is the seat reserved for
the algebraic-entropy-feedback signal. `run_ppo(..., pretrain_demos=k)` adds the
BC→PPO warm-start (imitation then RL). CLI: `--task reach-ppo --policy {hsikan,mlp}`.
Tested: GAE is correct; PPO improves return from scratch (`test_ppo_improves_return`).

## Results (matched ~14k params, seed 0, 40 PPO iters × 1024 steps)
| run | policy | params | return (init→final) | reach err (m) | floor |
|---|---|---|---|---|---|
| from-scratch | hsikan | 13 961 | −42.9 → **−34.2** (+8.7) | 0.368 | 0.397 |
| from-scratch | mlp | 14 153 | −27.4 → −30.5 (−3.2) | 0.380 | 0.396 |
| BC→PPO warm | hsikan | 13 961 | −32.4 → −41.6 | BC 0.281 → **PPO 0.351** | 0.397 |
| BC→PPO warm | mlp | 14 153 | −36.3 → −35.2 | BC 0.253 → **PPO 0.345** | 0.396 |

## The honest finding (measured / inferred)
- *Measured:* from a **random** start PPO improves the HSiKAN return (+8.7) more than the
  MLP (−3.2), but both reach only marginally below the floor (0.368/0.380 vs ~0.40). From
  a **BC-pretrained** start (reach 0.25–0.28) PPO makes **both worse** (→ 0.345–0.351).
- *Inferred (the diagnosis):* **critic cold-start under a dense-negative reward.** The
  reward is `−dist` every step (≈ −0.3 × 80 ≈ −24/episode); the critic is random-init
  (BC trains only the actor), so early advantages `return − value ≈ large-negative − 0`
  are uniformly bad and the clipped policy update pushes the *good* actor off its
  solution. Consistent with "improves from random (room to move), degrades from good."
- *Conclusion:* the **architecture ablation is inconclusive** — PPO is not yet effective
  enough to separate HSiKAN from MLP. The from-scratch return trend (HSiKAN up, MLP flat)
  is suggestive at best — single seed, marginal reach, and confounded by the same tuning
  issue. **No architecture win is claimed.**

## Files touched
**New:** `hymeko_rl/ppo.py` (+175), `hymeko_rl/tests/test_ppo.py` (+35).
**Modified (mine):** `hymeko_rl/train_robot_rl.py` (+`reach-ppo` mode + `--iters`).
**CORE.YAML:** none. **No new dependency.**

## Test results
- `hymeko_rl/tests/` **25 passed** (36 s, `pytest -p no:randomly`): the 23 prior + 2 new
  (GAE finiteness/identity; PPO improves return from scratch). `ruff` + `mypy --strict`
  clean on `ppo.py`.

## Performance
From-scratch run ~100 s (HSiKAN) / ~50 s (MLP); warm-start ~150 s / ~64 s. CPU, peak RSS
≪ 16 GB. The MLP is ~2× faster (no message passing). Single seed — not a benchmark.

## §6.5 anti-patterns
None. One PPO trains both policies (the ablation is the existing backbone Strategy, no
per-arm loops); GAE/update are pure functions; specific errors; no globals.

## Open / next (the fix path, in order)
1. **Fix the critic cold-start** — the cheapest first: during the BC warm-start, also fit
   the **critic** to the discounted return-to-go of the demos (or run a few value-only PPO
   updates before policy updates). Expectation: PPO then *preserves/improves* the BC reach
   instead of degrading it.
2. **Reward** — consider a shaped/normalised reward (success bonus + small `−dist`), and
   advantage/return normalisation, to de-risk the dense-negative scale.
3. **Then** re-run the matched-capacity ablation **multi-seed** — only after PPO reliably
   beats BC is an HSiKAN-vs-MLP comparison meaningful.
4. **Then** the **algebraic-entropy-feedback** test (`structural_inductive_entropy_note.tex`
   §5): structure-driven exploration vs vanilla `β·H(π)`, MLP as the negative control.
5. Grasping (gripper+object MJCF) remains Phase 2b; the quadruped reuses all of this.
