# FROZEN selected MLP+DAgger checkpoints (deployable imitation baseline, 2026-07-07)

Validation-24-selected checkpoints (deployable view), mean fingertip_dominant = 0.452 (test N=48):
- mlp_s0_selected_d2.pt  (D2, ft_dom 0.438)  [val optimal]
- mlp_s1_selected_d3.pt  (D3, ft_dom 0.625)  [val optimal; best single]
- mlp_s2_selected_bc0.pt (BC0, ft_dom 0.292) [val missed D1=0.479]

Diagnostic best-checkpoint mean = 0.514. Scripted v2b ceiling = 0.792.
These are the RL-init checkpoints for the gated CTDE-TD3+BC re-entry. Do NOT overwrite.
Reward: galambos_task_deliver_v2b.hymeko (frozen). Scene: v2 graded contact_legality.
