# R5 — adaptive coast identification: A0 reveals an OBSERVABILITY wall (RL still blocked)

**Date:** 2026-07-27 (JST) · **Branch:** `recovery/coin-r5-adaptive-coast-release` (from R4 tip `c7a09b00`) · Contract:
`reports/2026-07-27-coin-r5-adaptive-coast-release-contract.md`.

**Result in one line:** the online coast-deceleration estimator is *correct* (unit-verified — it recovers a known constant
deceleration, rejects invalid samples, is median-robust, clamps), but the **A0 de-risking gate fails the same way R4 did
(1/4)** for a *structural* reason: the released-coast dissipation that governs delivery is only observable **after** the grip
is relaxed, whereas the release decision needs it **before** — the deceleration measurable *pre*-release is the (much larger)
gripped deceleration, so `â_eff` saturates at the clamp and does not proxy the coast friction. Verdict
**`ADAPTIVE_COAST_RELEASE_HITS_AN_OBSERVABILITY_WALL`** (a sharpening of R4, not "unlearnable"). **SAC/TD3 remain BLOCKED**
(A1/A2 not reached; no RL this session).

## 1. What was built (committed `203003fd`, on the R4 substrate)

- `closed_loop_state.CoastEstimator` — online `â_eff = clamp(median{−Δv∥/Δt}, [a_lo,a_hi])` over windows with no active
  forward push, `v∥ > v_min`, and `Δv∥ < 0`; a prior below `n_min` samples. **Unit tests pass** (recovers a=0.6 exactly,
  rejects active-push/low-speed/accelerating samples, median-robust to one outlier, clamps, prior).
- `ClosedLoopController` adaptive path — feeds the estimator each step; a **grip-held coast-probe** after the push
  (`probe_brake_gain=0` ⇒ grip-hold, not brake) seeds `â_eff`; `coastin_phase` uses `â_eff` when `adaptive=True`. Reduces
  **exactly** to R4 when `adaptive=False` (R4 tests + GOLDEN unchanged; adaptive off by default).

## 2. A0 — adaptive teacher fixed point (the R4-blocker de-risking test)

Teacher own-intent + the adaptive estimator, probe-length swept:

| probe_steps | A0 teacher (s1 s3 s4 s7) | total |
|---|---|---|
| 2 | 0 0 0 0 | 0/4 |
| 3 | 0 **1** 0 0 | 1/4 |
| 4 | 0 0 0 0 | 0/4 |

`1/4` at best — **no improvement over R4's fixed-coast `1/4`.** The estimated `â_eff` (teacher intent, probe=3): s1 **1.20**,
s3 **1.20**, s4 1.20 (range 0.54–1.20), s7 **1.20** — i.e. **saturated at the clamp `a_hi=1.20`**, not the ≈0.42–0.67 coast
friction the R4 traces implied.

## 3. Why — the observability wall (the load-bearing finding)

The estimator is not wrong; the *observable signal* is. Before the release decision the coin is **gripped** (PUSH, or the
grip-held probe), and a gripped coin decelerates at the **grip/brake friction (≳1.2 m/s²)**, not the **released-coast
friction (≈0.5)**. The two are different physical quantities:

- to measure the released-coast dissipation you must **relax the grip** (enter RELEASE);
- but RELEASE **is** the decision the estimate is supposed to inform, and it is **monotone** (no re-grip) and lets a fast
  cradle (s3) escape.

So the effective coast dissipation is a **post-release** quantity, structurally unobservable at the moment it is needed. The
grip-held probe (even at brake-gain 0) measures the gripped deceleration → `â_eff` saturates → `coast_reach = v∥²/2·â_eff`
collapses → the release trigger mistimes exactly as R4's fixed model did. This is *why* a single measured `â_eff` cannot, by
itself, replace the fixed `a_friction`: there is nothing to measure it from before the commitment.

## 4. Where this points (fresh contract, not built here)

The wall is specific and suggests three concrete redirections, in order of promise:

1. **Grip-held brake-to-stop** (sidesteps the wall). Abandon coast-then-release; instead modulate a *measured* deceleration
   under a **held grip** so the coin stops in the zone (grip held throughout ⇒ s3 cannot escape, and the *gripped*
   deceleration — the thing that IS observable — is exactly what governs the stop). The estimator built here measures that
   quantity directly; only the strategy changes (stop-in-zone vs coast-in). This is the shortest fix given what is
   observable.
2. **Calibrated gripped→released map.** Learn/regress the released-coast friction from the observable gripped deceleration +
   geometry (a small, local, cross-cradle map), then use it in the coast-in release — a *bounded* learned residual over the
   analytic estimate (the contract's §3), but calibrated on the *right* (observable) input.
3. **Two-pass identify-then-execute** only if a state reset were allowed — it is not (continuous-trajectory contract), so
   this is excluded.

## 5. Recorded audit (from R4, now empirically verified)

The R4 open-loop baseline read `1/4` (vs the canonical R3 `2/4`) purely because `coin_r4_gates.py` uses a **uniform** search
seed `90000` while R3/R1/R2 use a **per-state** seed `90000+i·131+K`. Direct check this session: **s3 open-loop predictor
delivers at seed 90131 (dtz 6.7 mm) and misses at 90000 (dtz 37.1 mm)** — same decoder, predictor, option, physics; a
narrow-basin search-jitter sensitivity, not a regression. Recorded so the R4 comparison is not misread.

## 6. Tests, provenance, status

R5 estimator unit test + the 9 R4 fast tests + the slow GOLDEN — pass; ruff clean; all functions < CC 11. Adaptive is off by
default so R4 behaviour/tests are byte-unchanged. Peak RSS < 0.3 GB. **Verdict:**
`ADAPTIVE_COAST_RELEASE_HITS_AN_OBSERVABILITY_WALL`. **SAC/TD3 BLOCKED** (A1/A2 not reached). **Exact next action:** a fresh
contract for the **grip-held brake-to-stop** strategy (§4.1) — it uses the *observable* gripped deceleration the estimator
already measures, holds the fast cradle, and only then, if it clears A0→A2, the residual-intent RL (R6+) is authorised.
