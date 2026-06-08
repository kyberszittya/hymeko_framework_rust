# Fuzzy-pose config-audit — verdict report

Date: 2026-06-02T07:24:54+02:00
Plan: `docs/plans/2026-06-01-fuzzy-pose-config-audit/plan.tex`
Audit log: `/tmp/fuzzy_smoke_2026_05_30/fuzzy_pose_config_audit.log`

## Summary

>>> VERDICT: H3 CONFIRMED — architectural ceiling. All three var_expl < 0.70 (C0=0.460, C1=0.142, C2=0.395). 55% is the honest number for this synthetic regime.

Branch taken: **H3 CONFIRMED**

## Conditions tested

| Condition | Configuration | var_explained |
|---|---|---|
| C0 as-shipped | {} | 0.4603 |
| C1 validated-classifier | {'cr_input_scale': 'unit_to_grid', 'residual_kind': 'lerp', 'fuzzification_kind': 'sigmoid'} | 0.1418 |
| C2 sharper-softmax | {'soft_argmax_beta': 0.5} | 0.3953 |

## Provenance

- Git HEAD (pre-audit): `8fd8187c7dc3e9c7bda67c01c10364f416127e54`
- fuzzy_pose.py SHA256 (pre-audit): `71e64dcad2b2e995403a550ad8a656b61123a00469d69f72a234ffc43b210c1a`
- Orchestrator PID: 169665
- Orchestrator log: `/tmp/fuzzy_smoke_2026_05_30/overnight_audit_then_fix.log`
- Hardware: NVIDIA GeForce RTX 2070 SUPER

## Decision

The defaults-flip follow-up plan was NOT applied (branch: H3 CONFIRMED).
The contingent plan directory
`docs/plans/2026-06-02-fuzzy-pose-defaults-flip/` was removed per
empty-plan-dir hygiene.

The night-1 fuzzy result (var_expl 0.555) stands as the honest
architectural number for this synthetic-pose regime, pending any
further work along the H2 / H3 branches.

