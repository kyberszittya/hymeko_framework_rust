---
title: OPTION_RL_STRUCTURED_TEMPORAL_RUNTIME_V1 — runtime API + serialization contract
date: 2026-07-24
branch: feat/architectural-assimilation-v1
status: RUNTIME_CORE_FROZEN (task adapters bind to this API; changes are versioned)
---

# OPTION_RL_STRUCTURED_TEMPORAL_RUNTIME_V1

The shared, task-independent structured-temporal-multimodal execution runtime that the coin/object, 6DOF, pick-and-place,
and AIBO adapters all consume. Built by EXTENDING `hymeko_rl/option_rl/` (the single-θ `ProposalPolicy`/`FixedBudgetSearch`
remain the K=1 special case). `option_rl` imports **no** task module (asserted by `test_option_rl_runtime`).

## Pipeline
```
raw env → StructuredStateAdapter → StructuredState
        → LSTMTemporalEncoder.update(x_t, hidden) → temporal_embedding      (hidden = RUNTIME state, threaded)
        → fuse_state(structured, temporal, history) → fused representation
        → MultimodalProposalPolicy.modes(fused) → [ProposalMode]            (K strategy modes, not one θ)
        → MultimodalBudgetSearch.select(proposal, obs, rng) → MultimodalProvenance
        → committed option execution → HandoffCertificate/CertificateMonitor
        → OptionTransition (γ^τ semi-MDP) → learner/eval
```

## Task-adapter contract (what a task implements; everything else is provided)
| interface | method | supplies |
|---|---|---|
| `StructuredStateAdapter` (Protocol) | `structured(raw) → StructuredState` | entities/edges/attrs/phase/monitor/geometry/contact/metadata |
| `MultimodalProposalPolicy` (Protocol) | `modes(obs) → [ProposalMode]` | K strategy-mode centres + probs + per-mode spread (K=1 via `SingleModeProposal`) |
| `CandidateGenerator` (Protocol) | `sample(center, n, rng) → (n,d)` | task perturbation around a mode centre |
| `CandidateScorer` (Protocol) | `score(cand, rng) → (float, dict)` | roll the candidate → local-search score + outcome/certificate dict |
| `HandoffCertificate` (Protocol) | `success/handed_off(outcome)` | physical task-success predicate (independent of proposal + reward) |

A task must NOT bring its own SAC/TD3, bootstrap, search, LSTM, or eval — those are the runtime's (`train_semi_mdp`,
`eval.*`, `MultimodalBudgetSearch`, `LSTMTemporalEncoder`). Missing shared interface ⇒ file an `ASSIMILATION_REQUIREMENT`.

## Guarantees (tested)
- **Order-invariance** — reordering the mode list or candidates does not change the winning mode/θ/score/outcome: each mode
  gets an independent child rng keyed to its CANONICAL identity (`mode_id` then centre bytes), not its list position; score
  ties break on a canonical candidate key. (`test_search_mode_order_invariant_*`.)
- **K=1 ≡ FixedBudgetSearch** — bit-identical (single mode keeps the passed rng). (`test_k1_multimodal_reproduces_*`.)
- **Bellman-safety** — the Bellman action is the SELECTED MODE's centre; the concrete selected candidate is provenance only;
  budget/generator/scorer/allocation are frozen across a run (stationary env response) — same contract as the single-θ path.
- **LSTM: batch ≡ streaming**, no cross-episode hidden leakage after `initial_hidden`, mid-episode checkpoint round-trips
  exactly. (`test_lstm_*`.)
- **Domain-free** — a `pkgutil` scan asserts `option_rl` pulls in no coin/CIP/pick-place/… symbol.
- **End-to-end** — a non-coin ToyReach runs the whole pipeline to a delivered certificate + checkpoint restore
  (`test_option_rl_toy_reach`).

## Serialization / replay / checkpoint contract
- `StructuredState` — pure data (arrays + dicts); JSON/npz-serializable via its fields; `flat()` is the `FlatStateView` payload.
- `LSTMTemporalEncoder` — `state_dict()` (weights) + the returned `(h, c)` hidden tuple (mid-episode). Restore = load weights
  + pass the saved hidden into `update` ⇒ identical next embedding. The hidden state is NEVER a global side effect.
- `MultimodalProvenance.as_dict()` — the deploy record: selected_mode, n_modes, mode_probs, θ_selected, score, budget,
  per_mode_budget, k6/reached_handoff/τ. Deterministic given (proposal, obs, rng).
- A full runtime checkpoint = `{proposal/actor state_dict, LSTM state_dict + hidden, StructuredState, search config, rng seed}`.

## Public API (`from hymeko_rl.option_rl import …`)
`StructuredState`, `StructuredStateAdapter`, `FlatStateView`; `LSTMTemporalEncoder`, `fuse_state`; `ProposalMode`,
`MultimodalProposalPolicy`, `SingleModeProposal`, `MultimodalProvenance`, `MultimodalBudgetSearch`, `allocate_budget`;
(retained) `ProposalPolicy`, `CandidateGenerator`, `CandidateScorer`, `FixedBudgetSearch`, `SelectedActionProvenance`,
`OptionTransition`, `OptionReplayBuffer`, `smdp_target`, `OptionEnv`, `HandoffCertificate`, `SkillRoute`, `train_semi_mdp`,
`QNet`, `GaussActor`, `DetActor`, `SemiMDPConfig`, the paired-eval helpers.

## Not yet in V1 (deferred; do not block adapters)
- A learnable explicit-history encoder (the `fuse_state` contract accepts one; `IdentityHistoryEncoder`/`FixedWindowMLPEncoder`
  are the reference stubs — the LSTM is the learned temporal component, a second big history net is not forced now).
- The RL **mode-selection / search-control** heads (the multimodal actor that emits mode logits + per-mode centres) — the
  interfaces are in place; wiring the distributional actor is the next assimilation increment, gated by the O3 physical result.

## Version / freeze
Tag `OPTION_RL_STRUCTURED_TEMPORAL_RUNTIME_V1`. Task adapters (6D-0 → 6D-1 → pick-place → AIBO) bind to THIS API; any change
is a new version. Tests: `test_option_rl_multimodal.py`, `test_option_rl_runtime.py`, `test_option_rl_toy_reach.py` (25
task-independent) + the coin `option_rl` consumer (unregressed).
