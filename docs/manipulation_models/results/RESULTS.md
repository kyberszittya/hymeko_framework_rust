# Manipulation-learning results (2026-06-24)

Consolidated results for the two HyMeKo manipulation tasks. Full write-up:
`2026-06-24-manipulation-learning-report.pdf`.

## Headline
1. **Pure RL fails both tasks** — it never discovers the long-horizon grasp from a scalar reward.
2. **Behaviour cloning (BC) breaks the exploration wall** — clone a scripted demonstrator, get a competent start.
3. **Architecture (HSiKAN vs params-matched MLP) is a tie** at iso-parameters once BC supplies the start.
4. **Off-policy from a BC warm-start *was* collapsing** — diagnosed as a **setup bug** (cold critic + random
   `start_steps` destroying the clone), **now fixed** (warm-start bridge). It was not a DDPG/TD3 limitation.
5. **FANUC is the working learned-control demo** (BC ≈ 50% placement); **Galambos is a hard control case**
   (~25% ceiling — a free-spinning coin rolls out of a 2-finger drag).

## Galambos — BC → PPO (greedy delivery, 3 seeds, difficulty 0.3)
| backbone | BC | →PPO (per seed) | median |
|---|---|---|---|
| HSiKAN | 0.04–0.21 | 0.21, 0.25, 0.29 | **0.25** |
| MLP    | 0.08      | 0.17, 0.21, 0.25 | **0.21** |

HSiKAN edges MLP but within the seed spread → effectively a tie; both at the ~25% control ceiling.

## FANUC — BC, then off-policy refine (placement, held-out seeds)
**BC baseline ≈ 0.50.** Off-policy refine *before the fix* (the collapse):
| | DDPG | TD3 |
|---|---|---|
| HSiKAN | 0.25 (from 0.50) | 0.00 |
| MLP    | 0.00 (NaN-unstable) | 0.00 |

## The off-policy collapse: cause + fix
- **Cause:** `train_offpolicy` ran 1000 uniform-**random** `start_steps` (ignoring the cloned actor) into a
  **cold (random) critic**, then the actor update *maximised that cold critic's Q* → pushed the clone in random
  directions and destroyed it. Exploration noise (0.1·π ≈ 0.31 rad) also shattered the precise grasp.
- **Fix (warm-start bridge, `ddpg.py`):** opt-in `warm_start` — act with the **cloned actor from step 0** (no
  random seeding) — and `critic_warmup` — update the **critic alone** first so it is sane before the actor moves;
  plus a smaller exploration noise. Default off → from-scratch behaviour unchanged.
- **Verified (Galambos BC→DDPG, warm-start):** BC 0.083 → DDPG **0.083** — **preserved, no collapse** (the cold
  critic no longer destroys the clone). Re-running the full FANUC/Galambos off-policy with the fix is the next step.

## On GPU
Little benefit for these jobs: the bottleneck is **serial MuJoCo env-stepping (CPU)** + **tiny nets** (~15k
params). DDPG/TD3 get their GPU boost in the **massively-parallel-env** regime (thousands of envs → big batches);
single-env + small structural nets are CPU/rollout-bound. Unlocking the boost would need vectorized parallel envs.

## Artifacts in this folder
- `gifs/demo_seed_*_goal.gif` — the scripted Galambos demonstrator's successful coin deliveries (timestamped).
- `2026-06-24-manipulation-learning-report.pdf` (+ `.tex` source) — the full report.
- `composition.dot` — the FANUC model's composition as a hypergraph-of-hypergraphs (render with graphviz `dot`).
- `2026-06-24-galambos-demonstrator-bc.md`, `2026-06-24-galambos-hyperedge-ab.md` — detailed notes.
- `../*.hymeko` (parent folder) — the manipulation models + their README.

Checkpoints (not copied — large): `checkpoints/{galambos,fanuc}/*.pt`.
