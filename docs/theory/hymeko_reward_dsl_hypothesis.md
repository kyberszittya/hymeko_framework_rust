# Hypothesis: HyMeKo as a declarative RL reward/task DSL — algorithm-agnostic, runtime-tunable, at zero performance cost

**Created-at:** 2026-06-27 16:00 JST · **Author:** Aiko (Claude Code) for Dr. Csaba Hajdu
**ETA (full experiment):** ~1–2 days (E1 parity-across-algorithms ~½ day machine-bound; E2 tuning-velocity is
mostly already demonstrated by the grind loop; E3 hardcoded counterfactual ~½ day).
**Sibling:** `docs/plans/2026-06-27-isomorphic-controllers-from-hypergraphs/` — same thesis, controller-topology
side. Together: **HyMeKo's contribution is a declarative substrate for control (reward terms *and* controller
topology), not a better optimizer.**

## The hypothesis

HyMeKo's value for RL is **not** "HSiKAN beats MLP" (the control results don't support that). It is that the
reward **and** task are a **declarative, runtime-tunable, algorithm-agnostic spec** (`.hymeko`, read via
`RewardSpec.from_hymeko`), so reward/term engineering happens **during simulation** — author, tune, swap, reuse —
instead of editing and recompiling Python. Formally:

- **H1 — algorithm-agnostic.** One `.hymeko` reward drives PPO, TD3, SAC, DDPG unchanged (reward ⟂ optimizer).
- **H2 — runtime-tunable / streamlined.** A reward change is a declarative `.hymeko` edit (no code, no recompile,
  git-tracked, hot-reloadable per episode) — measurably fewer touched LOC and faster iteration than Python.
- **H3 — zero performance cost.** The declarative reward computes the *identical* `Σ wᵢ·termᵢ`, so closed-loop
  performance equals a hardcoded reward — flexibility for free.
- **H4 — controlled / inspectable.** The `.hymeko` is a readable, declarative spec (terms + weights explicit,
  reusable across bundles/tasks/agents), supporting "structurally accountable" reward authoring and audit.

## What's already true (built, not claimed)

- **H1 is largely built.** `offpolicy_eval._ALGOS` wires **ppo / sac / ddpg / td3** to the *same* env, which reads
  the *same* `RewardSpec.from_hymeko` — the reward is already decoupled from the optimizer. The shared
  `RewardSpec` is consumed by `htl_reward`, `exp_reward_weight_sweep`, `reach_arch_compare`, the envs.
- **H2 is being demonstrated now.** The galambos reward grind = editing `galambos_task.hymeko` arc weights
  (e.g. `pull 2.0→4.0`, `oob 5.0→2.0`), no code touched, git-tracked. (Standing rule:
  `feedback-reward-definition-in-hymeko` — reward changes *always* edit the `.hymeko`.)
- **H3 is structural.** The reward value is the same `Σ wᵢ·termᵢ` whether the weights come from a file or a literal.

## The experiment (to turn "true by construction" into "measured")

- **E1 — parity + agnosticism (the headline figure):** PPO / TD3 / SAC on the galambos `.hymeko` reward → all run
  to comparable delivery (median/IQR), with **0 lines of reward code** (just the spec). One spec, three optimizers.
- **E2 — tuning velocity:** the grind loop as the demonstration — N reward variations as `.hymeko` diffs applied
  *uniformly* across the three algorithms; report diff size / no-recompile / git history vs the hardcoded path.
- **E3 — the counterfactual:** the *same* reward hardcoded as a Python function vs the `.hymeko` — identical
  performance (H3) and a side-by-side of the engineering cost to change a term (LOC, recompile, per-algorithm
  duplication).

## Honest framing & risks

- **The claim is software-engineering / MDSD, not RL-superiority.** Don't conflate it with the (shaky) structural-
  prior claim — keep them separate (as the gauge memory insists for HSiKAN's debuggability claim).
- **H3 risk:** if a future `.hymeko` reward needs a term the Python reward can't express as cheaply, parity could
  break — but today the term *vocabulary* is shared (`meta_reward.hymeko` ↔ `reward.py`), so it holds.
- **E1 is machine-bound** (3 algorithms × seeds) and must respect the 2-worker page-file cap.

## Tie to Kato's controller-topology program

Both are one thesis: **HyMeKo = the declarative substrate.** Reward side: declare/tune reward terms during
simulation (this doc). Controller side: declare/generate hypergraph topologies and emit isomorphic controllers
(`isomorphic-controllers-from-hypergraphs` plan, P4). The unifying contribution is *control authored as a
declarative, inspectable, runtime-tunable model* — across optimizers and across controller structures.
