# FULL_ACTION_BC_NOT_ESTABLISHED — a sparse marginal expert (3/9) does not support a competent full-action BC

**Created-at:** 2026-07-22 16:55 JST
**Branch:** recovery/coin-hymeko-bundle-and-results (e0cecb7 → fdbe99d)
**Bundle:** `6664ac459cca8f62` (v3 dynamics-corrected)

## Verdict

`CANONICAL_FULL_ACTION_EXPERT_DATASET_PASS` (dataset built) but `FULL_ACTION_BC_NOT_ESTABLISHED`. RL remains gated
(§11) — no critic/actor learning launched.

## What was built (real, on the canonical v3 bundle)

- **Dataset** (`EXPERT_FULL_ACTION_CANONICAL_V3_{train,val}.npz`): the frozen composed chain (E_valselect approach →
  handoff transport) rolled from true neutral on disjoint seed pools, recording `(node_features flat 48,
  u_expert_executed = inner.data.ctrl 4)` per step — no residual/scripted-delta/post-hoc. 62/200 train seeds delivered
  → 62 trajectories, 10 222 transitions (approach 5036 + transport 5186); val 16/40. All motion through `env.step`.
- **Standalone BC** (`node_features → 4`, MLP, no scripted base/carry/residual/online-expert): clones well (train loss
  0.29→2e-4, val 9e-4).
- **On-policy transport DAgger**: roll the BC to its OWN grasp state, hand off to the expert transport via `env.step`,
  relabel — targeting the observed failure phase.

## Result (competence, multi-panel, from true neutral)

| policy | headline first-contact | headline grasp | headline strict-delivery |
|---|---|---|---|
| zero-action | 0/9 | 0/9 | 0/9 |
| frozen expert | 9/9 | 4/9 | **3/9** |
| plain BC | 9/9 | 9/9 | 0/9 |
| BC + DAgger (best) | 9/9 | 9/9 | **1–2/9** (val 2/40, held-out 5/30) |

Competence ratio 0.33–0.67 < the 0.90 target.

## Causal failure distribution

- **Approach: SOLVED.** BC/DAgger reaches first-contact + grasp on 9/9 headline states from neutral — the approach
  transfers robustly.
- **Strict-settle TRANSPORT: the wall.** Strict K=6 (centered `dtz≤0.02` ∧ settled `speed<0.06`, 6 consecutive) is
  reached only 1–2/9. This is the **expert's own ceiling** (3/9 — the documented one-finger contact-mechanics limit),
  compounded by BC covariate shift in the marginal final settle. DAgger helps at first (0→2/9) but **oscillates and
  destabilises** (1,1,0,0,0) as the transport labels flood the aggregate and the sparse positive signal is drowned.

## Honest conclusion (per directive §3/§7)

The 3/9 frozen chain **establishes the mechanism but is too sparse and marginal to clone a competent full-action
policy** — exactly the caveat §3 stated and §7 warned against ("do not claim competence from 3 vs 3 on nine states").
The bottleneck is not the learning algorithm; it is **expert quality**: strict K=6 delivery is contact-mechanics-
limited to ~1/3 of states even for the oracle. Before a competent BC / RL:

1. **Strengthen the dynamic expert** — per-state exact-rollout / trajectory search on KatoLab (§3-allowed) to raise
   the delivering-state count well above 3/9, and/or a better transport policy; then regenerate the dataset.
2. Re-run BC → DAgger competence against the stronger expert.
3. Only then critic calibration → smokes → campaign.

RL is not launched (§11). The canonical v3 bundle, dynamic-expert reproduction, dataset harness, BC and DAgger code
are all committed and reusable for the stronger-expert retry.
