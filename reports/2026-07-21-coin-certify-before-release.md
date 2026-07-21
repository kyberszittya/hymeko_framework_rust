---
campaign: COIN certify-before-release oracle — isolate the settle failure before RL
title: FORCE_CLOSURE_BLOCKED — the release hypothesis is falsified; the strict "successes" are one-finger coast-to-stop, not grasps
date: 2026-07-21
branch: exp/coin-wristed-pad-delivery-integration
source_commit: 0903aa8
classification: FORCE_CLOSURE_BLOCKED (release-disturbance falsified; strict predicate over-certifies non-grasp deliveries; no RL launched)
---

# Certify-before-release oracle (§1–§9)

**Created-at:** 2026-07-21 13:24 JST. The prior report returned NO_FORCE_CLOSURE; the user rejected it and named a
specific alternative defect: *"RELEASE/WITHDRAW destroys the centered low-velocity state before the strict dwell
certificate can accumulate."* This iteration builds the four fixed oracle variants (A/B/C/D) that differ **only** in
post-transport sequencing, runs them on matched states with identical approach/grasp/transport, logs the strict
components per step, and applies one discriminating test the whole exercise turns on: *does the strict certificate,
when it fires, correspond to a genuine bilateral force closure?* No geometry / friction / actuator / reward change.

## VERDICT: FORCE_CLOSURE_BLOCKED

The release-sequencing hypothesis is **falsified**, and the reason no certificate survives release is that **no
force-closure certificate ever forms** — the strict predicate has been firing on non-grasp coast-to-stop deliveries.

## §1/§9 — the four variants on E3 (48 matched seeds, 4 clearance bands)

| variant | post-transport sequence | strict ≥+0.030 | dominant first-fail |
|---|---|---|---|
| **A** CLOSED_HOLD_CERTIFY (never release) | hold pads closed + arm fixed, wait for dwell | **0** | TRANSPORT (never reaches settle) |
| **B** CERTIFY_THEN_RELEASE | A + release after cert | **0** | TRANSPORT |
| **C** CERTIFY_RELEASE_WITHDRAW | A + release + withdraw | **0** | TRANSPORT |
| **D** CURRENT | release/withdraw immediately | **0** | TRANSPORT |

**A = B = C = D, band-for-band identical.** If release/withdrawal were destroying a certificate, variant A (which
*never* releases) would out-perform D. It does not. On E3 the coin never even reaches a held-in-zone state — the
failure is TRANSPORT, upstream of release. **Release sequencing changes nothing → RELEASE_DISTURBANCE falsified.**

- **Max pre-release dwell (E3-A):** 0 steps (dwell required = 6).
- **Min pre-release in-zone velocity (E3):** 0.0908 m/s (> settle threshold 0.06) — one seed (+0.060–0.080 band) briefly
  in-zone but too fast; first-fail there = `settle_velocity`. All other E3 seeds never enter the zone.
- **First failing strict component (E3):** the coin does not reach the zone at all (TRANSPORT); the dwell counter never
  starts.

## §5 — DoF sweep under variant A (which DoF is load-bearing?) — 48 seeds

| embodiment | strict predicate fires (≥+0.030 / all bands) | genuine bilateral force closure |
|---|---|---|
| **E0** passive concave ring (no added DoF) | **16 / 48** | **0** |
| **E1** wrist-yaw | 0 | 0 (all GRASP_FAILURE) |
| **E2** independent closure | 0 | 0 (TRANSPORT / HOLD_SETTLE) |
| **E3** wrist + closure | 0 | 0 (TRANSPORT) |

The added actuated DoF **hurt**: E1 collapses to GRASP_FAILURE, E2/E3 grasp-then-fail-to-settle. The minimally-tuned
wrist/closure controllers make the grip worse than the passive ring, not better.

## The discriminating test — the E0 "successes" are NOT grasps

E0 firing the strict predicate 16/48 looked like the first real signal in this line. But the strict certificate is
*centered + settled + no-shove* — **it does not require force closure.** So I instrumented every one of the 16 winners
for bilateral pad contact during the 6-step dwell window (deterministic; determinism confirmed across two runs):

- **0 / 16 winners have any bilateral contact during certification.** 14 are single-pad grazes (one pad, fL=0.0,
  fR ≈ 0.02–0.07 N — incidental, not a grip); 2 (negative clearance) are pure **no-contact** coast-to-stop.
- The two winners at the meaningful ≥+0.030 clearance (seed 1011 @ +0.033, seed 1045 @ +0.0386) both show the coin
  **coasting** into the zone at dtz ≈ 0.019, cvel ≈ 0.018–0.035, decreasing ~1.5 mm over the whole 6-step window —
  it drifts to a stop under friction, grazing one pad. No closure.

**Max certified clearance where the strict predicate fires: +0.0386 — but by coast-to-stop, not grasp.** Genuine
force-closure clearance: **none, any embodiment.**

## Interpretation (§6–§8)

1. **RELEASE_DISTURBANCE is falsified** — A=B=C=D; the certificate is not destroyed by release because it never forms
   as a grasp.
2. **FORCE_CLOSURE_BLOCKED** — no embodiment (E0/E1/E2/E3) forms a bilateral grasp that holds the coin. This reconfirms
   the arc's **terminal contact-mechanics wall** (memory: box/cylinder → one-finger point contact), now measured
   through the wristed-pad path rather than inferred.
3. **The strict predicate over-certifies.** It counted 16 non-grasp push/coast deliveries as STRICT_SUCCESS. To gate
   *force-closure* delivery, the predicate needs a **bilateral-contact-during-dwell** requirement — the concrete,
   actionable fix this test surfaced. (Consistent with the standing guard "loose overcounts 3–4×"; here even the
   strict predicate overcounts, because it lacks a grasp term.)
4. **Push-delivery is partially viable** (a one-finger nudge does land the box centered+settled) — consistent with the
   prior ACTOR-1 push/plow PARTIAL — but it is a *different solution* from grasp-carry-hold, and not force closure.

## §10 — RL gate

Not launched. The oracle is **not force-closure-positive** (0/… genuine grasps), so per the discipline no learned
training was started. katolab/GPU is irrelevant until a valid physical grasp oracle exists.

## Artifacts / provenance
- Oracle runner: `hymeko_rl/experiments/coin_wristed_delivery.py` (variants A/B/C/D + DoF sweep + discriminating
  bilateral-contact trace). Transport trigger fires on sustained **bilateral contact** (upstream of the certificate,
  cannot inflate it). ruff clean.
- Shared infra unchanged: `hymeko_rl/env/pad_actuation.py`; 7 integration tests still pass.
- Data: `experiments/2026_07_21_coin_wristed_pad/oracle/certify_oracle.json` (sha256 `0f4a238b`), verdict
  `HOLD_SETTLE_BLOCKED` from the E3-keyed auto-classifier — **superseded by the discriminating test to
  FORCE_CLOSURE_BLOCKED** (the auto-verdict keys on E3 alone and cannot see the E0 non-grasp artifact).
- Figures: `reports/figures/2026-07-21-certify-before-release/certify_evidence.png` (force-closure = 0 across all
  embodiments; A=B=C=D); `e0_coast_seed1045.gif` (sha `eee04d5a`) — the one-finger coast-to-stop.
- **Preserved:** transport `39551de3`, APPROACH `94601ea4`, P&P `d2da720a`, Beni `4630b537`. No CORE.YAML. No deps.
- Host Apple M5 Pro; oracle wall ≈ 45 s (48 seeds × 6 cells), RSS ~0.45 GB; threads pinned OMP/MKL/OPENBLAS=1.

## Honest scope
This is a control-limited, minimally-tuned oracle (one force target, one brake radius, hand-written phase machine), so
it does not *prove* a box cannot be parallel-jaw grasped in this sim. What it does establish, on a verified
integration: (a) the specific release-disturbance hypothesis is wrong, and (b) every strict "success" to date on this
path is a non-grasp coast, so the strict metric must gain a force-closure term before it can gate or reward RL. That
metric fix is the next concrete step — not RL.
