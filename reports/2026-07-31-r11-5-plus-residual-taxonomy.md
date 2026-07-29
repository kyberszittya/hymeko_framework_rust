# R11.5+ Phase 1 — Complete Residual Failure Taxonomy (all 24)

**Gate:** `R11_5_PLUS_RESIDUAL_TAXONOMY_COMPLETE` — 24 uncovered, 24 unique.

## By category

| category | n | scenarios |
|---|---|---|
| `CAPTURE_SUPPORT_FAILURE` | 10 | bank_c1_+0.01_+0.02, bank_c1_+0.01_+0.03, bank_c1_+0.03_+0.02, bank_c1_+0.03_+0.03, bank_c1_-0.01_+0.00, bank_c2_+0.015_+0.015, bank_c2_+0.015_+0.025, bank_c2_+0.025_+0.015, bank_c2_+0.025_+0.025, bank_c2_-0.015_+0.000 |
| `CONTACT_LOSS_DURING_DELIVERY` | 4 | bank_c1_-0.01_+0.03, bank_c1_-0.03_+0.02, bank_c1_-0.03_+0.03, bank_c2_-0.025_+0.015 |
| `INSUFFICIENT_TRANSPORT_PROGRESS` | 10 | bank_c2_-0.015_+0.015, bank_c2_-0.015_+0.025, bank_c2_-0.025_+0.025, bank_c3_r5_a-45, bank_c3_r6_a-45, bank_c3_r7_a+15, bank_c3_r7_a-45, bank_c3_r9_a+15, bank_c3_r9_a+30, bank_c3_r9_a-45 |

## Per-scenario panel

| scenario | stage | split | rel-target | certified | kinetic | min→final dtz (mm) | max prog (mm) | lostBR/rel | dwell | zone-entry | residual class |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bank_c1_+0.01_+0.02 | c1 | train | [-0.0697, 0.0315] | False | — | —→— | — | —/— | — | — | `CAPTURE_SUPPORT_FAILURE` |
| bank_c1_+0.01_+0.03 | c1 | dev | [-0.0697, 0.0315] | False | — | —→— | — | —/— | — | — | `CAPTURE_SUPPORT_FAILURE` |
| bank_c1_+0.03_+0.02 | c1 | test | [-0.0697, 0.0315] | False | — | —→— | — | —/— | — | — | `CAPTURE_SUPPORT_FAILURE` |
| bank_c1_+0.03_+0.03 | c1 | test | [-0.0697, 0.0315] | False | — | —→— | — | —/— | — | — | `CAPTURE_SUPPORT_FAILURE` |
| bank_c1_-0.01_+0.00 | c1 | train | [-0.0697, 0.0315] | False | — | —→— | — | —/— | — | — | `CAPTURE_SUPPORT_FAILURE` |
| bank_c2_+0.015_+0.015 | c2 | train | [-0.0697, 0.0315] | False | — | —→— | — | —/— | — | — | `CAPTURE_SUPPORT_FAILURE` |
| bank_c2_+0.015_+0.025 | c2 | dev | [-0.0697, 0.0315] | False | — | —→— | — | —/— | — | — | `CAPTURE_SUPPORT_FAILURE` |
| bank_c2_+0.025_+0.015 | c2 | test | [-0.0697, 0.0315] | False | — | —→— | — | —/— | — | — | `CAPTURE_SUPPORT_FAILURE` |
| bank_c2_+0.025_+0.025 | c2 | test | [-0.0697, 0.0315] | False | — | —→— | — | —/— | — | — | `CAPTURE_SUPPORT_FAILURE` |
| bank_c2_-0.015_+0.000 | c2 | train | [-0.0697, 0.0315] | False | — | —→— | — | —/— | — | — | `CAPTURE_SUPPORT_FAILURE` |
| bank_c1_-0.01_+0.03 | c1 | train | [-0.0697, 0.0315] | True | True | 75.48→126.71 | 21.5 | 39/47 | 0 | — | `CONTACT_LOSS_DURING_DELIVERY` |
| bank_c1_-0.03_+0.02 | c1 | train | [-0.0697, 0.0315] | True | True | 79.01→160.71 | -73.3 | 44/49 | 0 | — | `CONTACT_LOSS_DURING_DELIVERY` |
| bank_c1_-0.03_+0.03 | c1 | train | [-0.0697, 0.0315] | True | True | 77.77→159.34 | -33.4 | 10/15 | 0 | — | `CONTACT_LOSS_DURING_DELIVERY` |
| bank_c2_-0.025_+0.015 | c2 | train | [-0.0697, 0.0315] | True | True | 84.88→160.55 | -69.6 | 44/49 | 0 | — | `CONTACT_LOSS_DURING_DELIVERY` |
| bank_c2_-0.015_+0.015 | c2 | train | [-0.0697, 0.0315] | True | True | 70.52→70.52 | 35.2 | 0/15 | 0 | — | `INSUFFICIENT_TRANSPORT_PROGRESS` |
| bank_c2_-0.015_+0.025 | c2 | train | [-0.0697, 0.0315] | True | True | 48.01→48.01 | 32.2 | 4/8 | 0 | — | `INSUFFICIENT_TRANSPORT_PROGRESS` |
| bank_c2_-0.025_+0.025 | c2 | train | [-0.0697, 0.0315] | True | True | 47.37→47.56 | 88.7 | 0/17 | 0 | — | `INSUFFICIENT_TRANSPORT_PROGRESS` |
| bank_c3_r5_a-45 | c3 | train | [-0.0177, 0.0468] | True | True | 31.14→31.14 | 24.4 | 0/11 | 0 | — | `INSUFFICIENT_TRANSPORT_PROGRESS` |
| bank_c3_r6_a-45 | c3 | train | [-0.0212, 0.0561] | True | True | 42.26→42.26 | 22.9 | 0/18 | 0 | — | `INSUFFICIENT_TRANSPORT_PROGRESS` |
| bank_c3_r7_a+15 | c3 | train | [-0.0691, 0.0113] | True | True | 22.36→22.36 | 61.4 | 0/8 | 0 | — | `INSUFFICIENT_TRANSPORT_PROGRESS` |
| bank_c3_r7_a-45 | c3 | train | [-0.0247, 0.0655] | True | True | 38.89→38.89 | 37.0 | 0/19 | 0 | — | `INSUFFICIENT_TRANSPORT_PROGRESS` |
| bank_c3_r9_a+15 | c3 | test | [-0.0888, 0.0146] | True | True | 35.2→35.2 | 68.1 | 0/8 | 0 | — | `INSUFFICIENT_TRANSPORT_PROGRESS` |
| bank_c3_r9_a+30 | c3 | test | [-0.0896, -0.0089] | True | True | 40.86→40.86 | 57.9 | 0/8 | 0 | — | `INSUFFICIENT_TRANSPORT_PROGRESS` |
| bank_c3_r9_a-45 | c3 | dev | [-0.0318, 0.0842] | True | True | 55.56→55.56 | 56.1 | 0/14 | 0 | — | `INSUFFICIENT_TRANSPORT_PROGRESS` |

## Bounded 12-scenario recovery pilot

| scenario | split | category | subtype |
|---|---|---|---|
| bank_c1_+0.01_+0.03 | dev | `CAPTURE_SUPPORT_FAILURE` | systematic_pp |
| bank_c1_+0.03_+0.02 | test | `CAPTURE_SUPPORT_FAILURE` | systematic_pp |
| bank_c1_+0.01_+0.02 | train | `CAPTURE_SUPPORT_FAILURE` | stochastic_regen |
| bank_c1_-0.01_+0.00 | train | `CAPTURE_SUPPORT_FAILURE` | stochastic_regen |
| bank_c1_-0.01_+0.03 | train | `CONTACT_LOSS_DURING_DELIVERY` |  |
| bank_c1_-0.03_+0.02 | train | `CONTACT_LOSS_DURING_DELIVERY` |  |
| bank_c1_-0.03_+0.03 | train | `CONTACT_LOSS_DURING_DELIVERY` |  |
| bank_c2_-0.025_+0.015 | train | `CONTACT_LOSS_DURING_DELIVERY` |  |
| bank_c3_r9_a-45 | dev | `INSUFFICIENT_TRANSPORT_PROGRESS` |  |
| bank_c3_r9_a+15 | test | `INSUFFICIENT_TRANSPORT_PROGRESS` |  |
| bank_c3_r7_a+15 | train | `INSUFFICIENT_TRANSPORT_PROGRESS` |  |
| bank_c2_-0.015_+0.015 | train | `INSUFFICIENT_TRANSPORT_PROGRESS` |  |

## Notes (trajectory ground truth)

- Classification is trajectory-derived (deterministic replay), not final-state: the coverage ledger's `dtz` is `dtz_end`, which cannot separate zone-entry from a short stall.
- **Re-diagnosis:** the 4 negative-x cases (all rel-target `[-0.070, +0.032]`) are `CONTACT_LOSS_DURING_DELIVERY`, not directional drift — the grasp loses bilateral contact for >80% of the pre-release window (`lostBR` 39–44/47) and the coin flies off to 126–161 mm. The final-state guess was `DELIVERY_DIRECTIONAL_BIAS`; the mechanism is the coin squirting out when pushed off-axis, so the ALIGN fix (which *preserves the grasp* while correcting direction) still applies.
- The 10 `INSUFFICIENT_TRANSPORT_PROGRESS` cases by contrast *hold* the grasp (`lostBR` ~0), move toward target, and stall 11–50 mm short — a transport horizon/magnitude lever.
- Empty categories (also findings): `HANDOFF_TO_KINETIC_FAILURE`=0 (every certified case moved the coin), `TARGET_ENTRY_SPEED_FAILURE`/`ZONE_ENTRY_WITHOUT_DWELL`=0 (nothing reached the 20 mm zone; closest 22.4 mm).
- Capture-support audit (candidate-hook, systematic +/+ × 2 seeds): bilateral contact **never forms** (best class SINGLE, 0 BITRANS/CERT) — an honest-negative geometry bound, not a rank-then-reject bug, so elite-diversity cannot help the systematic +/+ six.

