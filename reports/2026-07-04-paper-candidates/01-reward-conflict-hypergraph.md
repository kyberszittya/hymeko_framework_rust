# Paper candidate 1 — Reward conflict as a signed-hypergraph invariant that causes RL failure

**Working title:** *The Reward as a Hypergraph: Measured Conflict Between Reward Terms Causally
Predicts Manipulation Failure*
**Target venue:** Nature Machine Intelligence (primary); Science Robotics or NeurIPS as fallback.
**Status:** core result measured and causally confirmed on one task; breadth (multi-task,
multi-seed, prospective prediction) is the entire remaining distance to submission.

## Abstract seed

A two-arm robot that never grasped through a week of reward shaping begins grasping the moment its
reward is *simplified* from ~11–14 terms to four (grasp-fraction 0.10 → 0.615). We explain this by
treating the reward as a signed hypergraph over its terms, with edge signs given by the correlation
of term increments along a competent trajectory. The complex reward is frustrated (Harary index 2);
the simplified one is balanced (0). Re-adding exactly the two most conflicting terms causally drops
grasping to 0.333. The predictive invariant is the continuous **conflict magnitude** (total
negative-edge weight), of which the frustration index is the stricter topological special case.
Reward conflict thus becomes a computable, cheap-to-probe property of a declarative reward
specification — reward design as a spectral problem on a hypergraph.

## Central claim

Conflict between reward terms — measurable as negative co-movement edge weight on the signed term
graph, probed along a competent policy's trajectory — causes manipulation failure, and reducing it
(by term deletion) restores the behavior the task requires.

## Evidence ledger

**Measured** (source: `reports/2026-06-28-reward-conflict-hypergraph.tex`, artifacts under
`reports/overnight/`, figures under `figures/reward_conflict/`):

- Simplification effect: grasp-fraction of deliveries **0.10 → 0.615** (11-term → 4-term reward;
  MLP actor, SAC, 25k steps, difficulty 0.3, 50 eval episodes, **seed 1**).
- Structure: complex reward frustration index **2** (9 active terms, 8 negative edges); simplified
  reward index **0**. Dominant conflicts: `arm_motion × joint_velocity = −0.24` (anti-stall vs
  smoothness — opposed by definition), `grasp_approach × arm_motion = −0.20`.
- Causal test: re-adding the two conflicting terms → grasp-fraction **0.615 → 0.333**.
- Refinement (an honest negative that sharpens the law): the frustration index stayed 0 after the
  re-add (two conflict edges form a path, not an odd cycle) while behavior degraded — so the
  predictive quantity is **conflict magnitude**, not the frustration index. Three measured points
  are monotone in it.
- Consistency check from a later A/B (`reports/2026-07-04-galambos-coord-ab.md`): adding a
  coordination term (`both_approach 4.0`) to the balanced 4-term reward gave **no improvement**
  (0.12 vs 0.16 median, 3 seeds, overlapping ranges) — consistent with the thesis that once the
  reward is balanced, term-weight shaping is second-order and the remaining blockers are
  physics/demonstrator quality.

**Inferred:** the mechanism (conflicting gradients pull the policy away from contact-seeking) is a
plausible reading of the measurements, not itself instrumented.

**Still hypothesis:** generality beyond Galambos; the spectral formulation (smallest signed-Laplacian
eigenvalue = algebraic frustration) as a *prospective* design tool; any claim that conflict predicts
failure *before* training.

## On-disk artifacts

- Full technical report: `reports/2026-06-28-reward-conflict-hypergraph.tex` (compilable).
- Companion: `reports/2026-06-28-scenarios-investigation.tex`.
- Reward source of record: `data/robotics/galambos_task.hymeko` (reward edits live in `.hymeko`,
  git history is the audit trail — user rule).
- A/B counter-evidence for term-shaping: `reports/2026-07-04-galambos-coord-ab.md` +
  `experiments/2026_07_04_03_33_galambos_coord_ab_baseline/`, `…_05_47_…_coord/`.

## Prior art and delineation (search debt: HIGH)

The bounded 2026-06-29 search did **not** target this line specifically. Before drafting, a focused
search is owed on: multi-objective RL and reward-term interference; gradient-conflict methods
(PCGrad, CAGrad — those measure conflict in *policy-gradient* space, this work measures it in
*reward-term co-movement* space along a trajectory, a different and cheaper object); reward-shaping
theory (Ng's potential-based shaping — orthogonal: it characterizes *invariance*, not *conflict*);
reward-design diagnostics. The signed-graph frustration machinery is classical (Harary, Zaslavsky) —
cite, do not claim. The delineation to defend: **conflict measured on the reward's own term
hypergraph, causally tested by term reconstruction, on a physical manipulation task**.

## Missing work to reach submission

1. **Breadth (the main gap).** Repeat the protocol on ≥4 distinct tasks (candidates already in the
   repo: FANUC pick-place, quadruped jump/standing, cart-pole with distractor terms, walker) —
   complex vs simplified reward, frustration/conflict measured, ≥3 seeds each, median/IQR. §3
   benchmark discipline applies; the current headline is seed 1 only.
2. **Prospective test.** Measure conflict of a *new* reward before training and predict the failure
   mode. One confirmed prospective prediction converts the paper from post-hoc to predictive —
   the largest single value-add.
3. **Principal-axes / spectral study** (declared in progress in the source report): eigendecomposition
   of the term co-movement matrix; smallest signed-Laplacian eigenvalue as algebraic frustration;
   term-redundancy axes. This is the "reward design as spectral problem" section.
4. **Probe-policy sensitivity.** The signs are measured along a trained grasping policy's trajectory
   (random exploration is underpowered — measured). Chicken-and-egg to resolve: how competent must
   the probe be? Sweep probe checkpoints (2k/6k/12k steps) and show sign stability.
5. **Estimator robustness.** Sign = sign(corr(Δfᵢ, Δfⱼ)) — characterize threshold sensitivity,
   window length, and near-zero correlations (currently a hard sign; a weighted or thresholded
   variant may be more stable).
6. **Literature search** per item above; write the related-work delineation.
7. **Graphics** per §9: the two existing figures (grasp-by-reward bars, grasp-vs-conflict monotone)
   plus per-task conflict-vs-performance scatter across the breadth study; GIFs of knock-vs-grasp
   behavior (renderers exist: `evaluate.render_episode_gif` / `compare_gif`).

## Risks / falsifiers

- Breadth study finds tasks where high conflict coexists with success → the law needs a
  conditioning variable (e.g. conflict *on load-bearing terms only*), or is Galambos-specific.
- The gradient-conflict literature may contain a near-identical trajectory-space measurement —
  the focused search must settle this before any novelty claim.
- Baseline calibration: the 2026-07-04 A/B under-reproduced a known-good 0.40 baseline (harness,
  not env — verified). Any breadth run must first reconcile its driver against the recorded best.
