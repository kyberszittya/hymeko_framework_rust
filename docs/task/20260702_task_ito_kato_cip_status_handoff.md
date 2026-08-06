# CIP status handoff — Ito + Kato task

**Note:** the task of record `20260702_task_ito_kato` is a file (not a directory), so this pointer lives beside it
as a same-stem sibling.

The Ito + Kato CIP scenario is built end-to-end and validated (no training run): reward-independent runtime
monitors → CIP-export → DirectLiNGAM → `.hymeko` declaration cross-view-verified against the HyMeKo engine, on
coin-toss (de-risk) and MetaWorld coffee-push (items 2–4). Robust verdicts: the coin reward-farming hypothesis is
supported at the reward-computation level; MetaWorld coffee-push `near_fraction ↔ total_reward` is stable across 5
seeds (reward is proximity/contact-shaped) while `progress_score → total_reward` was rejected as a direct edge by
multi-seed aggregation; cross-view verification passed in every run. All DAGs remain PROPOSED; policy-learning
consequences are not yet tested.

- **Full status:** `reports/2026-07-08-cip-status.md`
- **Commit range:** `a39acbb` … `49b8785` (branch `hymeko-neuro-migration`)
- **Next recommended decision:** choose the direction with Kato — **LiNGAM-SH** (the signed-hypergraph LiNGAM
  science contribution) vs. **real-env MetaWorld reward ablation** (transfer the coin Stage-A intervention to a
  real MetaWorld reward). See `reports/2026-07-08-cip-kato-meeting-brief.pdf`.
