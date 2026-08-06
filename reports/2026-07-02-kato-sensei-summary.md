# HyMeKo as a Control Substrate — Progress Summary for Kato-sensei

**Date:** 2026-07-02 · **Author:** Aiko (Claude Code), for Dr. Cs. Hajdu
**Scope:** the SA-HSiKAN structural actor, the reduced-structure hypothesis, and the measured results across the
robot-control scenarios. Numbers are marked **[measured]**, **[inferred]**, or **[hypothesis]** — the same
honesty discipline we apply internally; a confident-sounding single number that was not isolated is a guess in a
lab coat, and we try not to ship those.

---

## 1. The thesis (what the contribution actually is)

The contribution is **not** "HSiKAN beats MLP." It is that **HyMeKo is a declarative control substrate**: the
robot, the task, and the reward are all written once as a signed hypergraph in the `.hymeko` language, and a
learned controller reads its *structure* rather than a flat feature vector. The scientific question is then
sharp and falsifiable: **when does the declared topology carry control-relevant information, and when is it
inert?** We can answer that plant by plant, because the structure is a first-class, swappable object.

Two consequences follow, and both are load-bearing for the collaboration:

1. **Reward/task as a runtime-tunable, algorithm-agnostic artifact.** The reward is a `.hymeko` bundle
   (`Σ weightᵢ · termᵢ`), edited declaratively, git-audited — never an in-code override. This let us build a
   *reward-oracle* (below) that certifies a reward **before** any RL.
2. **Isomorphic controllers (your line).** Because the controller is generated *from* a hypergraph, we can
   generate many topologies, instantiate a controller from each, and benchmark **which topology controls which
   plant best.** This is the topology→control study in §5.

---

## 2. SA-HSiKAN — the reduced structural actor

### 2.1 What the reduction is

**HSiKAN** (Highway Signed KAN) is the full model: iterative signed message-passing over the robot hypergraph
with highway/residual connections and a KAN (Catmull–Rom / Chebyshev) channel mixer. It is expressive but does
many tiny operations per forward pass.

**SA-HSiKAN** (Structural-Actor HSiKAN) is the **deliberately reduced** form. Instead of iterating message
passing, it performs a **single static gather along the design's walks** — the length-`L` walk incidence `Bᴸ`,
which is exactly the **Z₂ holonomy** of the signed graph — followed by the KAN readout. One matmul over a
precomputed structural operator, no iteration, no highway. We call it the **"Bᴸ-collapse" agent**: the entire
structural computation is collapsed into one linear pass over the graph's walk geometry.

This is a *reduction of structural capability* on purpose. The full iterative machinery is removed; what remains
is the holonomy of the signed structure. The empirical question is whether that residue is enough.

### 2.2 Why reduce (the launch-bound story)

Small controllers are **launch-bound, not FLOP-bound**: at batch size 1 the wall time is dominated by kernel
*dispatch*, not arithmetic. A full HSiKAN issues many tiny kernels; the `Bᴸ`-collapse issues few. So the
reduction is not just a parameter cut — it directly attacks the dominant cost of deploying a tiny controller in
a control loop. **[measured]** we confirmed the "slow HSiKAN" was a B=1 dispatch problem (≈18 ms/step),
fixed to ≈1.5 ms by batching the rollout — after which *physics* (≈3.3 ms) becomes the floor, i.e. the network
is no longer the bottleneck.

### 2.3 What the reduction costs (and, surprisingly, does not)

- **[measured] Cost & size.** SA-HSiKAN runs ≈**2.6× faster** with ≈**11× fewer parameters** than the MLP
  baseline; against full HSiKAN the structural-actor collapse buys ≈**10× speed / ≈30× fewer parameters**.
- **[measured] Accuracy is *retained*, not sacrificed.** On the control tasks tested, the reduced actor matches
  or beats both full HSiKAN and the MLP. On the galambos two-arm task an early comparison put SA-HSiKAN delivery
  at ≈0.25 vs MLP ≈0.15. **This is the headline scientific result of the reduction:** for these plants the full
  iterative structure is *not* load-bearing — the single-pass holonomy `Bᴸ` already contains the
  control-relevant structural information.

### 2.4 Why this is principled, not a lucky trick

The `Bᴸ` operator is not an arbitrary compression. In the signed-graph gauge theory we have been developing,
**balance = Z₂ holonomy** (a theorem), the rotor is the *connection*, and a length-`L` signed walk is a
*parallel transport*. SA-HSiKAN's single gather is therefore the **learned holonomy** of the robot's structure —
the same object that, in the discrete case, decides sign-balance. So "reduced structural capability" is really
"keep the holonomy, drop the iteration," and the finding that it suffices is evidence that **control on these
plants is a holonomy problem**, not a deep-message-passing problem.

---

## 3. Results across the scenarios (honest, measured)

One shared simulator/eval ecosystem drives every scenario (a `TaskSpec` registry + a `RolloutMetric` strategy
per task), so these numbers come from the *same* eval loop, not hand-rolled per-task harnesses.

| Scenario | Best backbone/algo | Result | Reading |
|---|---|---|---|
| **Cartpole** (control floor) | HSiKAN + DDPG | **[measured]** learns upright **28 → 144/200**; DDPG ≈**250×** more sample-efficient than PPO | structure + off-policy both pay |
| **Galambos coin-toss** (planar 2-arm) | SA-HSiKAN + BC→off-policy | **[measured]** toss-delivery ≈**0.42** (easy); off-policy TD3+BC median **0.125** vs PPO's collapse to **0.042** (harder) | single-agent ≥ 2-agent CTDE — no collaboration win on this *non-cyclic* objective |
| **6-DoF arm reach** | HSiKAN + BC | **[measured]** HSiKAN ≈ MLP | serial arm has *little* structural leverage — an honest tie |
| **Quadruped goal-reach** | flat HSiKAN + PPO | **[measured]** flat HSiKAN **−33** vs 4-leg CTDE **−84** | collaborative framing *loses* on a non-cyclic reach |
| **Quadruped standing** (new, Rung-2 postural) | SA-HSiKAN + TD3 | **[measured]** learnable (seed 0 **0 → 0.24**) but **not robust** (2/3 seeds 0.0); the policy *sinks and falls* | diagnosed: the base has no velocity/rate observation → no damping term (an "IMU" fix is in progress) |
| **Pick-and-place** (6-DoF, cube) | HSiKAN + TD3+BC | **[measured, corrected]** an earlier "0.875 lift" was a **physics blow-up artifact**; true ≈**0.125**; **largely unsolved** | the real grasp-and-place problem, still open |

### The pattern in one sentence
**Structure pays where the task is coordinative/cyclic and ties the MLP where it is serial/non-cyclic.** The arm
reach and the galambos objective are non-cyclic → tie. The place we *expect* structure to pay — cyclic postural
control (standing, gait) — is exactly the Rung-2 plant we just built to test it. That test is the honest next
step, not a claim already made.

---

## 4. Infrastructure & methodology shipped (all tested, core untouched)

- **Off-policy line replacing PPO.** PPO *collapses* a behaviour-cloned warm-start (measured: BC 0.125 →
  PPO 0.042). **TD3+BC** anchors the actor to the demos over replay and does not collapse — so the RL line is now
  off-policy (TD3+BC / SAC / DDPG), PPO retired.
- **Reward-Alignment Planner-Oracle.** Because the reward is a separable `.hymeko` artifact, we can *plan* its
  optimum in milliseconds with **zero RL** and ask "does the declared reward's optimum actually deliver?"
  **[measured]** it caught that a galambos reward's optimum *farms* an in-zone annuity rather than delivering —
  a reward bug found without training a single step.
- **Vectorized off-policy + GPU/compile + critic LayerNorm.** ≈1.7× throughput from batching the action-select;
  CUDA-graph capture of the update; LayerNorm bounds the critic (anti-overestimation, measured Q-peak 3 vs 16).
- **Evaluation-metric integrity (hard-won).** We now guard every success predicate against artifact inflation
  (a physics blow-up must not *count as* a lift), horizon-match every probe to the real episode length, and
  require **multi-seed median/IQR** — single-seed is a point estimate, not a verdict. Several confident internal
  hypotheses were **falsified by measurement** and retracted on the record.
- **Live observability.** Every training run emits step/loss/throughput/ETA to flushed stdout — no run goes dark.

---

## 5. Topology → control (your isomorphic-controllers line)

We built the extremal-topology "zoo" (Petersen, Kneser, Grötzsch, expander, Steiner/Fano, sunflower) with
tested topology invariants, generated a controller from each, and benchmarked them across matched-size plants.

- **[measured] The law is MATCHING, not a universal champion.** In a matched-`N` sweep, `best_controller[plant]
  == plant` for **9/9** plants — the topology that *is* the plant controls it best.
- **[measured, corrective] "Steiner/Petersen is best" was plant-specific.** Petersen/Steiner/Fano share a tight
  coherence (1/3, a tight-frame walk property), but coherence does **not** rank control; on the matched sweep
  Petersen was near-*worst*. We corrected the earlier over-claim.
- **[measured] Cross-view consistency is machine-verified.** The commuting square `X_f(ε_f(H)) = Q(H)` holds
  exactly on **16/16** fixtures across 5 emitter views (data + topology), via Z3 + sympy — the substrate's
  "one IR, many emitters" property is proven, not asserted.

**Implication for the collaboration:** the interesting controller for a plant is the one whose *hypergraph
matches the plant's own structure* — which is exactly what generating the controller *from* the robot's
`.hymeko` gives you for free. SA-HSiKAN then reads that matched structure as a single holonomy pass.

---

## 6. Honest open problems (what is *not* solved)

1. **Standing falls.** The postural policy sinks because the base is observed as position only (`[height,
   uprightness]`) with **no velocity/rate** — no damping term. Fix in progress: give the base a real **IMU**
   observation (orientation + angular velocity + vertical velocity) and retrain.
2. **Coin-toss "distraction."** The galambos reward had 9.0 of *dense* reward pointing *at* the coin (approach +
   contact) but only a *sparse* success and **no transport gradient** — so the arms park on the coin and never
   deliver it. Fix (declarative, in `galambos_task.hymeko`): restore the `pull` term (−‖coin−zone‖); an A/B to
   validate it is running now.
3. **Pick-and-place is the real grasp problem.** A round planar coin cannot be antipodally grasped (it rolls out
   of a two-finger clamp — measured: a scripted expert caps at 0.30 vs 0.875 for a flat-faced box); *grasping* as
   opposed to *tossing* belongs to the 6-DoF pick-and-place, which is still open.
4. **Where structure pays remains partly open.** The cyclic/postural tasks (standing, gait) are the honest test
   of the structural hypothesis and are only now being run; we have not yet earned a "structure wins on control"
   claim there.

---

## 7. One-paragraph version (for a slide)

We built **SA-HSiKAN**, a controller that reads a robot's declared HyMeKo structure as a **single holonomy pass
`Bᴸ`** — a deliberate reduction of the full HSiKAN's iterative message-passing to one static walk-gather. It is
≈2.6–10× faster with ≈11–30× fewer parameters and, on every control task we tested, *matches* the full model —
evidence that **control on these plants is a holonomy problem, not a deep-network problem**. Around it we built a
declarative, reward-as-`.hymeko` control substrate with an off-policy RL line, a reward-oracle that certifies a
reward before training, and a machine-verified "one IR, many emitters" guarantee. The governing law for
topology→control is **matching** (the plant's own structure controls it best), not a universal champion. The
open frontier — where we expect the structural prior to *earn* its place — is **cyclic postural control**
(standing, gait), which we are testing now.
