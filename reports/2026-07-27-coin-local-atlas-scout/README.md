# Local-atlas reconnaissance scout (N=24) — data acquisition only

**2026-07-27 · `coin_local_atlas_scout.py --scout --n 24` · reconnaissance for the local-control-atlas reframe · HARD STOP after (no clustering / feedback-banks / BC — those are the fresh-head next steps)**

## Purpose

Cheap reconnaissance for the go-forward reframe (a **local control atlas + causal gate**, not one global KINETIC controller):
scout candidate straddle cradles, certify a frozen-CEM teacher θ per deliverable cradle, and record the **realized teacher
transport physics** so the transport *modes* can be discovered from the physics, not the θ vectors. Reuses
`scout_certified_cradles` / `acquire_snapshot` / `deliver_on_snapshot` / `rollout_primitive` / `_phys_hook` — no new algorithm.

## Counts

- **24 scouted → 9 certified → 5 dev-deliverable** (+ 2 held-out deliverable, s4/s7).
- New dev-deliverable cradles beyond s1/s3: **16500, 17750, 19500** (2 more certified but not deliverable under the frozen option: 17000, 18250).

## Realized teacher physics (the mode signal)

| seed | mean v_par | peak v_par | **mean Fn** | contact_frac | min_dtz | sign_rev |
|---|---|---|---|---|---|---|
| 14250 (s1) | 0.096 | 0.322 | **1.54** | 0.55 | 18.5 mm | 0 |
| 17750      | 0.062 | 0.257 | **1.39** | 0.40 | 16.4 mm | 0 |
| 14750 (s3) | 0.133 | 0.381 | **3.09** | 0.30 | 15.3 mm | 0 |
| 19500      | 0.188 | 0.532 | **2.99** | 0.32 | 17.0 mm | 0 |
| 16500      | 0.165 | 0.652 | **3.76** | 0.25 |  5.5 mm | 1 |

**Preliminary (NOT a clustering verdict — that is the fresh-head D2 step):** the 5 deliverable teachers split by grip force
into two apparent regimes — **LIGHT-slide** (mean Fn ≈ 1.4–1.5 N, higher contact fraction ≈ 0.40–0.55: s1, 17750) and
**FIRM-guided** (mean Fn ≈ 3.0–3.8 N, lower contact fraction ≈ 0.25–0.32: s3, 19500, 16500). Grip force and contact fraction
anti-correlate. This is the raw signal the user hypothesised (LIGHT_SLIDING vs FIRM_GUIDED); a proper clustering / expert-
consistency audit (D2) is deferred to a fresh head.

## Exclusions audit

- **s4/s7 (seeds 15000, 15750): held-out validation-only** — deliverable, but NOT profiled and NOT in the dev set (`held_out=True`).
- **f1–f4: sealed blind** — untouched.

## Caveat

`release_dtz_mm` / `release_vpar` are read at step `θ[4]`, which is the (dead) BRAKE boundary for **s1** (there `ramp 12.8 >
rel 9.2`, so the θ[4] step is still deep in PUSH → the 84.2 mm is not the coast-release point). For the other four cradles
`ramp < rel`, so the field is meaningful. The **trajectory-wide** features (mean/peak Fn, mean/peak v_par, contact fraction,
min_dtz, sign reversals) — the ones the mode clustering will use — are correct.

## Artifact

`scout_n24.json` (full per-cradle rows + θ + physics). Next (fresh head): D1 per-cradle receding-horizon feedback banks →
D2 expert-consistency / mode clustering → local KINETIC policies → causal gate. And in parallel, the **fastest first delivery**:
the s1-local learned KINETIC coin-following policy (frozen APPROACH → learned policy → G0 release-guard → frozen coast → K6).
