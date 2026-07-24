---
title: Option-RL framework migration — coin becomes the first task adapter
date: 2026-07-24
branch: recovery/coin-hymeko-bundle-and-results
status: OPERATIONAL_EXPERIMENTAL / RL_ENABLED / PHYSICAL_SUCCESS_DEMONSTRATED / RL_SUPERIORITY_NOT_YET_ESTABLISHED
commit: d795d7f1
---

# Migrating the option-RL + proposal-search engine into the HyMeKo framework (2026-07-24 04:45 JST)

The coin push-delivery work produced a set of **task-independent** abstractions that had outgrown the experiment directory.
Per the user directive, the stable ones are now a framework engine `hymeko_rl/option_rl/`, and **coin is the first task
adapter** — not the owner of the mechanism. The RL-margin question (does RL beat the strong search-guided proposal?) does
**not** block this migration; it is being clarified by the Stage-5b campaign running in parallel (§6).

## 1. Discovery (§6.1) — what already existed vs what was new
Framework RL scaffolding already present: `hymeko_rl/agents/` (architectures), `hymeko_rl/train/option_search.py`
(`OptionSearchConfig`/`option_cem` — CEM option search), `hymeko_rl/campaign/` (ExperimentSpec + Adapter Protocol +
CoinDeliveryAdapter). **Absent** — and therefore the genuinely-new value to migrate: a general **semi-MDP option-RL engine**
(OptionTransition, OptionReplayBuffer, γ^τ target, proposal-center-as-Bellman-action + selected-candidate provenance,
semi-MDP SAC/TD3, checkpoint-wise paired evaluation, hierarchical skill routing). The migration *reuses* the existing
`option_search`/`campaign`/stats rather than forking them.

## 2. The framework engine `hymeko_rl/option_rl/` (task-independent, no task imports)
| module | contents |
|---|---|
| `core.py` | `OptionTransition`, `OptionReplayBuffer`, `smdp_target` (γ^τ), `OptionEnv`/`HandoffCertificate` Protocols, `OptionEnd` |
| `proposal.py` | `ProposalPolicy`/`CandidateGenerator`/`CandidateScorer` Protocols, `FixedBudgetSearch`, `SelectedActionProvenance` — **Bellman action = proposal center; the search-selected candidate is provenance, never the trained action** |
| `agents.py` | `QNet`/`GaussActor`/`DetActor` (obs/act dims parameterised), `SemiMDPConfig`, `train_semi_mdp(algo, env, actor, dev_eval_fn, …)` — SAC/TD3 over the proposal, actor trained through the critic only |
| `eval.py` | `bootstrap_ci`, `preregistered_select` (paired Δ → CI-lower → −exit → return), `paired_final_score`, `solved-set`, `across_seed_summary` |
| `hierarchy.py` | `SkillRoute` — upstream option → handoff certificate → frozen downstream skill (general skill composition) |

## 3. Coin as the first task adapter (non-breaking)
`hymeko_rl/coin_delivery/coin_carry_option_rl.py` is now the **coin adapter**. It keeps only the coin-specific pieces:
`execute_one_option` (push/brake/release macro through the coin physics), `OptionReward` (certificate-aligned), the coin
K6/handoff certificate, `SearchWrapperEnv` (implements the framework `OptionEnv`), `eval_policy`/`eval_paired`, and the
carry family classifier / frozen coin settling pi_0. `QNet`/`GaussActor`/`DetActor` are thin **coin-bound subclasses** of the
framework nets (identical `state_dict` keys), and `train_agent` is a thin wrapper over `train_semi_mdp`. Coin-specific but
**not** in the engine, exactly per the split: push/brake/release θ parametrization, coin certificate, K6, reward adapter,
carry classifier, frozen pi_0.

**Compatibility (verified):** prior Stage-5 SAC/TD3 checkpoints (`carry_rl_*.pt`) load unchanged into the refactored actors;
the stage5/5b/4c entries import-compile unchanged; all coin RL/proposal/option tests pass. The migration is pure addition +
re-export, revertible by deleting `option_rl/` and restoring two files.

## 4. Task-independence proof
`hymeko_rl/tests/test_option_rl.py` includes a **synthetic, non-coin** `ToyReachEnv` (a trivial 2-D reach option, no coin
import) trained end-to-end by `train_semi_mdp` — the engine learns it (best-val success > 0.6), demonstrating the mechanism
carries no coin specifics. Plus unit contracts: γ^τ target (terminal/non-terminal/tensor), replay FIFO+sample,
`FixedBudgetSearch` argmax + center≠selected provenance, pre-registered selection, `SkillRoute` routing.

## 5. Tests / lint / CORE
28 tests pass (6 framework `option_rl` + 8 coin RL + 5 coin proposal + 9 coin option). `ruff --select F` clean; E702
compact-semicolon style consistent with the arc. **CORE.YAML: empty** — `option_rl/` is new source; no dependency
added/upgraded (uses the already-pinned torch/numpy). Plan on disk at
`docs/plans/2026-07-24-option-rl-framework-migration/{plan.md,plan.mmd}` (repo `.gitignore` excludes `docs/plans`, so it is
on-disk for the audit trail but untracked — not force-added against the repo convention).

## 6. Coin pipeline status
`OPERATIONAL_EXPERIMENTAL / RL_ENABLED / PHYSICAL_SUCCESS_DEMONSTRATED / RL_SUPERIORITY_NOT_YET_ESTABLISHED`. The coin
push-delivery pipeline **brings the coin to target** on held-out carry states (~0.167 vs frozen pi_0's 0.000) via
`proposal → fixed b=8 search → committed option → frozen settling pi_0 → K6`, with correct semi-MDP RL wired in and reward
certified. It is a usable experimental component and the **canonical integration benchmark** for the framework option-RL
engine, not the final RL grand-challenge. Whether reward-driven RL *reproducibly beats* the strong search-guided update-0
proposal at fixed b=8 is the single open claim, under test in the Stage-5b variance-reduced campaign (in flight: log
`/tmp/s5b_full.log`, entry `coin_carry_option_rl_stage5b.py`, SAC 4 seeds + TD3 1 control × 2000 options; result to be
appended). Coin's essential mission is fulfilled: it surfaced the abstractions the framework needed and physically delivers.

## 7. Where real RL advantage is expected next
Simple planar manipulation may be near-solved by search/expert + BC, leaving RL marginal — a scientifically useful result,
not a failure (an explicit/strong-BC baseline gives an oracle, an upper bound, and a mechanism). Real RL headroom is
expected in harder regimes the framework engine now supports directly: varying mass/friction, coin size/shape, partial
observability, moving target, obstacles, varying contact geometry, multiple embodiments, and constraint transitions where a
single hand-written controller generalises poorly. The engine is the reusable substrate for those; coin is task adapter #1.
