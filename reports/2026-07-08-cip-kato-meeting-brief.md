# CIP scenario — meeting brief for Kato-sensei

**Date:** 2026-07-08 · Aiko · branch `hymeko-neuro-migration`
**One line:** *DirectLiNGAM/CIP running over the machine-verified HyMeKo hypergraph substrate — demonstrated
end-to-end on coin-toss (de-risk) and on your MetaWorld coffee-push scenario, with every discovered causal graph
cross-view-verified against the HyMeKo engine.*

---

## 1. The idea we are testing (the joint)

The runtime monitor judges the task (reward-independent); CIP prioritizes reward↔monitor disagreement; DirectLiNGAM
proposes a causal graph over the rollout variables; **that graph is declared as a `.hymeko` signed hypergraph and
the HyMeKo engine re-derives its tensor view + a Blake3 canonical hash** — so *the causal model the agent uses is
provably the one a human audits* (Kato-LiNGAM joint #1). The science extension on top is **LiNGAM-SH**
(signed-hypergraph LiNGAM) — the contribution proper, not just integration glue.

```
rollouts ──▶ HyMeKo runtime monitor ──▶ CIP variables ──▶ DirectLiNGAM ──▶ causal DAG
                                                                              │
                                                        declare as .hymeko ◀──┘
                                                                              │
                                            HyMeKo engine: star/clique tensor + Blake3 hash
                                            ── cross-view: declared ≡ tensor ✅ every run
```

## 2. What is built and verified (this week)

| Layer | Status | Evidence |
|---|---|---|
| Runtime monitors (reward-independent) | ✅ built | `task_monitor/` (coin + coffee-push + dial-turn) |
| CIP-export bridge (verdict → CIP variables) | ✅ built | `cip_export.py` |
| DirectLiNGAM consumer (numpy/scipy, no `lingam` dep) | ✅ built | recovers synthetic ground truth, recall 1.00 |
| **`.hymeko` declaration + engine cross-view** | ✅ built | `hymeko_emit.py`; **cross-view passed in 100 % of runs** |
| Coin-toss PoC (de-risk) | ✅ done | `contact_score → total_reward`, farming candidate |
| Contact-reward ablation (intervention) | ✅ done | removing contact terms collapses the edge |
| MetaWorld coffee-push / dial-turn (template) | ✅ done | recovers the scenario story |
| MetaWorld coffee-push (real env) + **multi-seed** | ✅ done | reward is proximity-shaped (5/5 batches) |

## 3. Results across the scenario

**Coin-toss (de-risk PoC).** DirectLiNGAM over real coin rollouts: `contact_score → total_reward` (+0.69) is the
strongest edge and `total_reward` is disconnected from `delivery_score` → **`reward_farming_candidate`**. Then the
**intervention** (Stage A, no training): recomputing the reward with the contact terms removed **collapses that
edge to 0.0** and re-parents the reward onto `delivery_score` (+0.88), disagreement drops. So the diagnosis is
supported *at the reward-computation level* — the coin reward is contact-farmable. (We did **not** "solve"
coin-toss; a physics wall and an imitation gap remain. CIP *diagnosed* it.)

**MetaWorld coffee-push (your scenario).** Real `SawyerCoffeePushV3Policy` rollouts + action noise, through the
HyMeKo monitor. **Multi-seed (5 batches × 80, no training):**

- `near_fraction ↔ total_reward`: **CONFIRMED stable** — present 5/5, |w| median **0.973**, IQR [0.962, 0.981].
  The real MetaWorld reward is **proximity/contact-shaped**. *(Honest: DirectLiNGAM flips the edge direction 3/2
  across batches — the coupling is solid, the direction is not uniquely resolved.)*
- `progress_score → total_reward`: **rejected** as a direct edge (present 1/5) — `near_fraction` mediates it
  (`near ↔ progress` stable +0.870). *The single run suggested this edge; multi-seed removed it — this is the
  value of the aggregation.*
- Cross-view verification: **passed in all 5/5 batches**. pass-rate median 0.51, disagreement median 0.35.

## 4. GIFs (for the slides)

`reports/gifs/metaworld_coffee_push/` — the scripted policy acting in the real MetaWorld env:
`coffee_push_success.gif` (clean, noise 0.0 → success), `coffee_push_failure.gif` (noisy, noise 0.9 → miss),
`coffee_push_compare.gif` (side-by-side). The action-noise knob is the same observed input the CIP run uses.

## 5. Figures

- Coin causal DAGs: `reports/figures/2026_07_08_13_32_cip_lingam_coin/` (mlp, hsikan).
- Contact-reward ablation (edge collapse): `reports/figures/2026_07_08_15_12_cip_contact_ablation_stageA/`.
- MetaWorld templates: `reports/figures/2026_07_08_15_45_cip_metaworld_synthetic/`.
- MetaWorld real-env + multi-seed: `reports/figures/2026_07_08_16_39_cip_metaworld_multiseed/` (per-batch DAGs +
  `.hymeko`).

## 6. Honest scope (say this in the meeting)

- Every causal graph is **PROPOSED**; controlled ablation decides. The `.hymeko` cross-view proves *representation*
  consistency (declared ≡ tensor ≡ hash), **not** causal truth.
- MetaWorld's env randomization is **not controlled by the seed** — single-run DAG order is a point estimate;
  that is exactly why the coffee-push claim rests on the **5-batch aggregate**, not one run.
- Real-env is **coffee-push only** so far (clean obs→monitor mapping). Dial-turn real-env needs a dial-angle
  extraction (its template is done).
- No RL training was run in any of this — all read-only rollouts of cached / scripted policies.

## 7. Proposed next steps (decisions for Kato)

1. **LiNGAM-SH** — the science contribution: constrain LiNGAM's `B` to factor through the signed incidence
   (`B = A_out Σ A_inᵀ`), so non-Gaussianity identifies the *grouping* (the theorem). Nearest prior art
   (arXiv:2511.03831, CAM→acyclic hypergraphs, Gaussian) validates the direction, does not pre-empt (no
   LiNGAM/ICA regime, no signs).
2. **Real-env reward ablation on MetaWorld** — transfer the coin Stage-A method: decompose the MetaWorld reward,
   recompute contact-off offline, check the `→ total_reward` edge collapses.
3. **Multi-seed everywhere + dial-turn real-env** — for a publishable ranking claim.

## Environment / provenance

metaworld 3.0.0, mujoco 3.10.0 (pinned to avoid a downgrade), gymnasium 1.3.0, numpy 2.x, scipy 1.17.1; native
`hymeko` engine built via maturin. 90 CIP tests pass; ruff/radon/mypy(strict) clean. Commits `a39acbb` (Phase 2 +
ablation) · `b3d56d3` (MetaWorld templates) · `d9a436f` (MetaWorld real-env) · `52e3af9` (multi-seed).

**Backing reports:** `2026-07-08-cip-phase2-coin-poc.md`, `2026-07-08-cip-contact-reward-ablation-setup.md`,
`2026-07-08-cip-metaworld-templates.md`, `2026-07-08-cip-metaworld-coffee-push-multiseed.md`.
