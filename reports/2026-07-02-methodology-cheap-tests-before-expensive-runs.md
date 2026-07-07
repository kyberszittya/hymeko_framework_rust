# Methodology: cheap discriminating tests before expensive runs

**Date:** 2026-07-02 · **Author:** Aiko (Claude Code), for Dr. Cs. Hajdu
**Subject:** the debugging methodology used across the standing / coin-toss / pick-place session, why it saved
hours of RL wall-time, and the transferable recipe. Written mid-session while a 3-seed A/B trains in the
background — itself an instance of the method (the expensive run proceeds only *after* a cheap test framed it).

---

## 0. Thesis

Reinforcement-learning debugging is slow because the feedback loop is slow: a single training run is ~50 min
and its output is noisy. The naïve loop — *guess a fix → train → look at the number → guess again* — costs
~1 hour per guess and the number often lies (it can be inflated by a physics artifact or loosened by a slack
threshold). The method that worked was to **never spend an expensive run to answer a question a cheap test can
answer first**, and to **watch the artifact, not the scalar**. Concretely, four moves, each with a measured
payoff this session.

---

## 1. Certify before you train — the reward-oracle (biggest saver)

**Move.** A reward is task-aligned *iff* its optimal trajectory delivers. RL finds that optimum slowly and
noisily; a planner finds it *exactly, in milliseconds*, by solving a small abstract phase-MDP with the **same**
declarative `RewardSpec`. So candidate rewards are **certified by planning, not by training.**

**This session.** The question "is the coin-toss failure the approach↔pull *balance*?" was answered by
certifying five reward weightings + a terminal-bonus search + an in-zone de-annuitize sweep — **all in < 1 s**:
- the balance is **second-order** (all configs return 1216–1236, all farm),
- the real reward flaw is **farming** (needs a ~1319 terminal bonus; lowering the in-zone weight doesn't fix it),
- which redirected the fix from "rebalance weights" (wrong) to "de-annuitize the in-zone reward" (right).

**Payoff.** Answering that by RL would have been ~5 reward variants × ~50 min = **~4 hours**, and the noise
(±0.06 at 24-ep eval) would likely have made the balance-comparison *inconclusive* anyway. The oracle gave a
crisp, deterministic answer in under a second. **The single highest-leverage tool in the loop.**

## 2. Smoke before you queue — never launch a run blind

**Move.** Before any multi-hour / multi-seed run touches a new code path, exercise that path at toy scale in the
same environment (§3 of the operating contract). A new branch's bugs surface in a 1-minute smoke, not a 40-minute
dark run.

**This session.**
- A 2-seed multiseed smoke confirmed the training loop completes *cell 2* before committing the overnight (the
  prior sweep had mysteriously stopped after cell 1; the smoke proved the loop was sound and the stop was
  environmental).
- A **stand-train smoke caught a `torch.compile` CUDA-graphs NaN crash** on the quadruped — *before* a 40-minute
  standing run would have died in the dark. It also let the fix (fall back to CPU) be chosen calmly.

**Payoff.** At least one avoided 40-minute blind failure, plus the confidence to launch the real runs without
babysitting them.

## 3. A discriminating test before a conclusion — isolate, don't narrate

**Move.** When a failure has several plausible causes, run the *one* test that separates them before asserting a
cause. A confident single explanation that wasn't isolated is a guess in a lab coat.

**This session.**
- **NaN in standing training:** GPU-crash vs genuine divergence? → rerun on **CPU**. It completed (finite by
  step 2000) → the NaN was a CUDA-graph artifact, not a training bug. Then an **env-finiteness probe** (0
  non-finite rewards, reward bounded [−2.2, 1.4]) ruled out the env. Two ~30 s probes replaced hours of
  wrong-tree debugging.
- **"The coin is hard to grasp — is it the shape or the weight?"** → a **2×2 shape×weight** table (cylinder/box ×
  light/heavy), 40-episode scripted-demonstrator probes, ~2 min total. Result: box−cylinder = +0.70 vs
  heavy−light = +0.10 → **shape, decisively**, and *heavier helps*. Refuted the weight hypothesis with data
  instead of argument.

**Payoff.** Each discriminating probe (~30 s–2 min) replaced a multi-run guessing spiral and, more importantly,
prevented *fixing the wrong thing*.

## 4. Watch the artifact, not the scalar — metric integrity

**Move.** A scalar success metric can be **inflated by an artifact** (a physics blow-up counted as a lift) or
**loosened by a threshold** (a slow sag counted as standing). Render the behaviour and watch it; trust the eye
over the number when they disagree.

**This session.**
- The galambos **0.42 "delivery" was a knock**, caught from the GIF — the metric graded in-zone, not grip.
- The standing **0.24 "stand-rate"** was a policy that *sinks and falls* — a trajectory probe showed the torso
  going 0.42 → 0.14 m while the loose 8 cm tolerance kept scoring it "standing."
- The **pick-place 0.875 "lift" was an explosion artifact** (earlier session) — a divergence that ejected the
  box upward.

Each of these, believed as a scalar, would have sent the optimization loop chasing a phantom. Watching + a
trajectory probe caught all three.

---

## 5. The honest anti-lesson (what cost time, and why)

Not everything was efficient. Early in the coin-toss thread I **over-engineered a grasp-gating + box-object
"fix"** — because I had *mis-framed the task*: I assumed the coin-toss required a precision grasp, when toss/push
is a valid solution (the grasp requirement belongs to the separate 6-DoF pick-and-place). The user's correction
reset it. **Lesson, and it is the same lesson one level up:** *confirm the objective before optimizing under it.*
Measure the ceiling, and equally, **verify what "success" is supposed to mean** — a mis-defined goal wastes the
same hours a mis-measured metric does. The cheap test here was one clarifying exchange, and it should have come
first.

---

## 6. Quantified payoff (this session)

| Cheap test | Cost | Replaced | Saved |
|---|---|---|---|
| Reward-oracle: 5 configs + tuning | < 1 s | ~5 RL reward-variant runs | **~4 h** (and gave a *deterministic* answer noise couldn't) |
| Stand-train smoke | ~1 min | a 40-min blind CUDA-graph crash | **~40 min** + the debug spiral |
| GPU→CPU NaN isolation + env probe | ~1 min | multi-run "why does it diverge" hunt | hours of wrong-tree debugging |
| 2×2 shape×weight table | ~2 min | training configs to guess graspability | hours, and *the wrong fix* |
| GIF / trajectory probes | seconds each | optimizing three phantom metrics | unbounded (chasing a lie has no floor) |

---

## 7. The transferable recipe

1. **Before an expensive run, write down the question it answers.** If a cheaper test answers it, run that first.
2. **Certify rewards by planning, not training** — the reward is a separable artifact; solve its abstract
   optimum in ms. Only train the survivors.
3. **Smoke every new code path** at toy scale, in the real environment, before queuing hours on it. No run goes
   dark.
4. **When a failure has ≥2 causes, run the one test that separates them** before naming a cause. Isolate one
   variable at a time.
5. **Watch the behaviour.** A metric that disagrees with the GIF is guilty until proven innocent — guard against
   artifact-inflation and threshold-slack.
6. **Confirm the objective first.** A mis-framed task wastes exactly as much time as a mis-measured metric.
7. **Label measured / inferred / hypothesis, and retract on contact with data.** Several confident calls this
   session were falsified by the next measurement; saying so early is cheaper than defending them.

The through-line: **make the feedback loop cheap.** RL's slowness is a property of *training*, not of *asking
questions* — most questions have a sub-second or sub-two-minute answer if you look for it before reaching for a
GPU.
