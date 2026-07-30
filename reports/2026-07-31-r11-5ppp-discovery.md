# R11.5+++ Stage A — Capture-Population Discovery Curve (5 barren INSUFFICIENT scenarios)

**Date:** 2026-07-31 (Mac; kato14 run stopped mid-way at user request and relaunched on Mac)
**Verdict:** **`R11_5_PLUS_CAPTURE_POPULATION_GROWTH_PASS`** — 3/5 barren scenarios acquire a deliverable grasp once the
capture population is grown to 40 seeds. Everything else frozen (transport bit-exact, R11.5++ ranking, capture
controller + bounds, delivery R=5; `delivered_dtz` teacher-only oracle).

## Discovery curves (deliverable K6 grasps at capture budgets {5, 10, 20, 40})
| scenario | certified@40 | unique-desc [5,10,20,40] | deliverable [5,10,20,40] | first-deliverable seed | best dtz | status |
|---|---|---|---|---|---|---|
| **c2_-0.015_+0.025** | 16 | 2,4,6,**16** | 0,0,1,**4** | **15** | 12.82 | under-sampled (climbing) |
| **c3_r5_a-45** | 14 | 3,3,7,14 | 0,0,1,**2** | **18** | 18.45 | under-sampled |
| **c3_r7_a-45** | 12 | 2,2,5,12 | 0,0,0,**1** | **37** | 19.49 | under-sampled (rare) |
| c3_r6_a-45 | 15 | 3,3,7,15 | 0,0,0,0 | — | 30.98 | **saturated** |
| c3_r9_a-45 | 14 | 3,3,7,14 | 0,0,0,0 | — | 30.46 | **saturated** |

## Findings
1. **3/5 barren scenarios were under-sampled, not barren.** Their first deliverable grasp appeared at seeds **15, 18,
   and 37** — beyond the N=10 the R11.5++ A/B sampled. The lever (grow the capture population) recovers them, all via the
   **existing single-stage transport** and the deliverability ranking. `c2_-0.015_+0.025` is still climbing (4 deliverable
   at 40, unique descriptors → 16).
2. **The proposal already generates diverse grasps** — 12–16 unique handoff descriptors per scenario by seed 40. So this
   is a *sampling depth* problem for the recoverable three, not a diversity-collapse problem.
3. **The 2 saturated a-45 scenarios (`c3_r6_a-45`, `c3_r9_a-45`) are already diverse** (14–15 unique descriptors) yet
   deliver **nothing** (best ~30 mm, curve flat). This **refutes Stage B's premise for them** — they are not
   under-diversified; the proposal's grasp *space*, even sampled broadly, contains no grasp that delivers to these
   specific a-45 targets. They are genuinely capture-**geometry**-limited (the parked frontier), not diversity-limited.

## Plan refinement (evidence-based)
- **Stage B (descriptor-diversity) is NOT warranted for the 2 saturated** — they are already diverse. Do not spend the
  diversity mechanism on them; they belong to the parked capture-geometry problem (with CONTACT_LOSS and +/+ support).
- **Next step is the re-A/B on the full 10-panel** with the enlarged (up-to-40-seed) population: current-selection vs
  deliverability-ranked, then the official coverage recount from the kato14 38/64 baseline.

## Coverage implication (provisional)
R11.5++ had 5/10 INSUFFICIENT deliverable + 5/10 barren. Discovery adds 3 of the 5 barren → **8/10 INSUFFICIENT are now
deliverable** (via more seeds + the frozen transport). If they hold under a frozen-protocol re-run, that projects
**~46/64** from the 38/64 baseline (still shy of the 47 margin; the 2 saturated a-45 + the capture-support tail remain).
Not counted coverage — teacher-only oracle, single R=5 realization; the official re-A/B + recount certifies it.

## Claims / non-claims
- **Claim:** 3/5 barren INSUFFICIENT scenarios were under-sampled — deliverable grasps exist at seeds 15–37 and the frozen
  transport delivers them to K6 (12.8–19.5 mm). Capture-population growth is a real, bounded lever.
- **Non-claim:** this is coverage (teacher-only oracle, R=5 single realization), or that the 2 saturated a-45 are
  recoverable by more seeds (they saturate at 40 with 15 diverse descriptors and 0 deliverable).
- Transport frozen + bit-identical; delivered_dtz teacher-only; no diversity mechanism, ALIGN, extra restarts, new
  controller, BC, or RL used.

## Provenance
Mac (darwin, 5-way fan-out, one process per scenario, max-seeds 40, R=5 delivery; ~46 min). kato14 run stopped at
~33 min (user request) and re-run on Mac. Deterministic; teacher-only oracle; energy diagnostic-only. Artifacts
`reports/2026-07-31-r11-5ppp-discovery-mac/` (merged.json + 5 shards with per-seed curves). Code `1f1daa43`. §2 plan
`docs/plans/2026-07-31-r11-5ppp-grasp-supply/`. No CORE.YAML, no deps.
