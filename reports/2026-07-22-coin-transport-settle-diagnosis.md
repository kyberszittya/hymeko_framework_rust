# Transport/settle diagnosis → STRENGTHENED_DYNAMIC_EXPERT_BLOCKED (needs a KatoLab suffix-search)

**Created-at:** 2026-07-22 17:26 JST
**Branch:** recovery/coin-hymeko-bundle-and-results
**Bundle:** `6664ac459cca8f62` · reclassified: `FULL_ACTION_BC_NOT_ESTABLISHED_AT_TRANSPORT_SETTLE`

## Diagnosis (§4 probes — written before any rebuild)

| probe | result | reading |
|---|---|---|
| **P1** handoff from EXPERT grasp | **3/9** (median min_dtz 0.055) | the baseline the transport was trained on |
| **P1** handoff from BC grasp | **1/9** (median min_dtz 0.048) | **SUCCESSFUL_SUFFIX_COVERAGE_GAP** — the BC reaches grasp states off the transport's success distribution |
| **P4** transport obs→action | nbhd spread **0.020** vs global std **1.23** (ratio 0.02) | labels are **unimodal** — NOT `DETERMINISTIC_ACTION_AVERAGING`, NOT `OBSERVATION_INSUFFICIENT` |
| **P3** bounded search (handoff+noise, 40 cand) from expert grasp | **2/9** (seeds 1045, 1278); 7/9 never reach strict dwell | ≤ the handoff — search-budget-limited, NOT proven physical (§4) |

## Mechanism verdict

1. **SUCCESSFUL_SUFFIX_COVERAGE_GAP** (confirmed, primary BC failure): the same frozen handoff delivers 3/9 from
   expert grasp states but only 1/9 from the BC's grasp states. The BC's grasp quality/distribution differs from the
   expert's, and the transport (trained on expert grasps) does not carry from BC grasps. This — not a learner defect —
   is why BC/DAgger caps at 1–2/9.
2. **An unresolved ~3/9 ceiling** even from expert grasps. The bounded local search (handoff + Gaussian noise, 40
   candidates) reached only 2/9 and 7/9 grasp states never reached strict dwell. **Per §4 this is a search-budget /
   method limitation, NOT a proven physical impossibility** — a stronger search (CEM / receding-horizon
   exact-rollout) is required to distinguish `WEAK_TEACHER` from `CONTACT_MECHANICS_CEILING`.

P4 rules out the representation mechanisms; the two live mechanisms are the coverage gap (confirmed) and the
teacher/ceiling question (unresolved, search-budget-limited).

## Seed-by-seed (P3, expert grasp)

- **search-deliverable:** 1045, 1278.
- **no suffix found in the 40-candidate handoff+noise budget:** 1011, 1164, 1174, 1202, 1358, 1447, 1568 — classify
  as *search-budget-limited* (weak method), to be re-tested with CEM / exact-rollout before any ceiling claim.

## Blocker: STRENGTHENED_DYNAMIC_EXPERT_BLOCKED

The §7 target (≥6/9 headline, ≥15/30 held-out, success-certified) cannot be reached with the bounded in-context search
(≤3/9). It requires a **KATO14 CPU-parallel success-certified suffix-search** (CEM / beam / receding-horizon
exact-rollout over the transport→brake→settle segment, lexicographic objective: strict-K6 → dwell → contact → dtz →
speed → effort), starting from states genuinely reached through `env.step`, with every accepted suffix **replay-
certified from the original neutral reset** (§5–§6). That search is a multi-hour parallel job — not runnable to
completion in one context — and is the exact next step; it also closes the coverage gap by generating diverse
success-certified transport suffixes from the BC's own grasp states.

- exact seeds: the 7/9 above (headline).
- first-divergence state: the post-grasp transport start (grasp reached 9/9; strict-settle is the wall).
- search budget used: 40 candidates × handoff+noise (weak); needed: CEM/exact-rollout (KatoLab).
- successful/failed suffix counts: 2 success / 7 no-suffix-in-budget on expert-grasp.
- why compute helps: a stronger parallel suffix-search directly tests the ceiling and produces the certified teacher.

## Preserved

All v3-learning artifacts + this diagnosis committed; RL still gated (§11/§13). New artifact root
`experiments/2026_07_22_coin_v3_expert_strengthening/`. Diagnostic harness `coin_delivery.transport_diagnosis`
(P1/P4 reproducible; P3 shooting inline).
