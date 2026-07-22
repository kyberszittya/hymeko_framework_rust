# STRENGTHENED_CANONICAL_DYNAMIC_EXPERT_PASS — KatoLab suffix search lifts the expert to 6/9 headline, 24/30 held-out

**Created-at:** 2026-07-22 18:10 JST
**Branch:** recovery/coin-hymeko-bundle-and-results · source commit `fc67c58` · bundle `6664ac459cca8f62`
**Compute:** KATO14 (32 cores, 125 GB), uv Python 3.11 + CPU torch 2.13 + MuJoCo 3.10, host-local NVMe `/tmp`.

## Verdict

`STRENGTHENED_CANONICAL_DYNAMIC_EXPERT_PASS`. A CPU-parallel success-certified suffix search on KatoLab, seeded from
the frozen-chain transition states, lifts the strict-K=6 delivery of the canonical-v3 expert from **3/9 → 6/9** on the
headline panel and to **24/30** on a disjoint held-out panel — both above the §10 targets (≥6/9, ≥15/30), every
accepted trajectory replay-certified from the neutral reset with no state injection.

## Method

- **Deploy (§1-2):** git-archive of `fc67c58` + checkpoints to kato14 host-local NVMe; built `hymeko_cli` (14.6 s);
  the canonical env verified on kato — semantic fingerprint `sem:469094de…` **identical to Mac**, arm mass 0.351557.
- **Search-state bank (§4):** the frozen **E_valselect approach** driven from true neutral to its transition state
  (bilateral grasp_hold OR the 160-step cap — the composed-chain handoff point).
- **Search (§5-7):** CEM over a compact **10-knot residual suffix** (piecewise-linear, applied open-loop via
  `env.step`), seeded with the handoff continuation, ranked lexicographically **strict-K6 → dwell → −dtz → −speed →
  −effort**. `robot_touched` carried honestly from the real prefix contact (not forced) so the search certificate
  matches the neutral-replay certificate.
- **Replay-certification (§9):** every accepted suffix re-run from the ORIGINAL seed — E-approach to grasp + suffix,
  all through `env.step` — required natural strict-K6 termination. This is the acceptance criterion (the snapshot
  search over-counts by ~15% vs the replay).

## Result

| panel | frozen handoff | **strengthened (handoff ∨ certified-search)** | target |
|---|---|---|---|
| headline (9) | 3/9 | **6/9** {1011,1045,1164,1278,1447,1568} | ≥6/9 ✓ |
| held-out (30) | — | **24/30** (search-certified 22, +handoff) | ≥15/30 ✓ |

Headline search 7/9 → replay-cert 4/9; held-out search 26/30 → replay-cert 22/30. The search **recovers states the
frozen handoff cannot** (headline 1164/1447/1568), resolving the earlier "unresolved 3/9 ceiling" — it was
**search-budget-limited, NOT a physical ceiling** (as §4 required us not to prematurely conclude). Per-state wall
2–68 s; 30 held-out states in ~5 min on 14 workers.

## Mechanism & non-claims

- **Weak-teacher confirmed, not contact-ceiling:** a stronger search (CEM, honest certificate) finds delivering
  transport suffixes on 22/30 held-out states where the fixed handoff did not — the frozen handoff was the limitation.
- **Residual unrecoverable states:** headline 1202/1358 and held-out's 4 non-successes hit
  `CONTACT_STATE_UNRECOVERABLE_WITH_TESTED_SEARCH` (dwell 0 — the coin never reaches the strict zone); 1174 has a
  snapshot-vs-replay dynamics gap (MuJoCo warmstart). These are the honest remaining limits.
- **This is a demonstration teacher, not a deployed policy.** The per-seed transport selection uses search provenance
  for dataset generation only; the deployed BC will run `u = policy(obs)` with no online teacher (§10, §13).
- **Not claimed:** that BC competence follows automatically — that is the next gate (§13). The held-out 24/30 shows
  the teacher generalizes, which is the prerequisite the coverage-gap diagnosis identified.

## Provenance

Source host kato14, commit `fc67c58`, bundle `6664ac459cca8f62`, search CEM(10-knot,pop64,iters25,honest robot_touched).
Immutable artifacts + SHA-256 in `experiments/2026_07_22_coin_v3_expert_strengthening/kato14_primary/`. Harness
`coin_delivery/{coin_v3_suffix_search,coin_v3_replay_certify}.py`.

## Next (§13, RL still gated §15)

Regenerate a phase-balanced success-certified full-action dataset from the strengthened teacher → phase-balanced BC →
success-certified DAgger on real BC divergences → `FULL_ACTION_BC_COMPETENCE_PASS`. No actor-critic before that gate.
