---
title: Upstream headroom probe — the gate-active state geometry is dichotomous, there is no in-distribution headroom region
date: 2026-07-24
branch: recovery/coin-hymeko-bundle-and-results
terminal: NO_INDISTRIBUTION_GATE_ACTIVE_HEADROOM_REGION — settling (strict 1–3) is pi_0-saturated (K6 0.95); the dominant carry region (strict-0 transport/contact_retention) is OOD for the critic AND pi_0-incompetent (K6 ~0 even at H=160)
tags: [coin, markov, headroom, state-geometry, gate-active, halt-condition, value-guided-search]
---

# UPSTREAM_HEADROOM_PREFIX_SEARCH — halted at the panel: strict 1–2 has no headroom, and the real upstream region is a different problem

The plan was to re-run the critic-guided prefix-search ablation on upstream states (strict 1–2, target_entry/braking
strata) where pi_0 has measurable headroom (K6 ~0.2–0.8). Before spending a ~20 min full run, three cheap structural
probes (§11: a measurement contradicting a plan assumption) showed the specified panel does not exist as imagined, and no
in-distribution gate-active region has the intended headroom. I halted the full run and report the geometry.

## Probe 1 — gate-active state geometry (120 held-out seeds)
Gate-ON handoffs by strict bucket: **strict-0: 8202**, strict-1: 11, strict-2: 11, strict-3+: 26. The gate-active
late-control region is overwhelmingly **strict-0**, dominated by **contact_retention (6248) and transport (1727)** — the
coin being *carried* toward delivery. At strict 1–2 the only family is **settling_dwell** (the coin is already held
in-zone; strict≥1 ⇒ in-zone). **target_entry (7) and braking (49) barely occur** as gate-active states. So the specified
"strict 1–2, target_entry/braking strata" panel is essentially empty — those families are strict-0 and rare.

## Probe 2 — pi_0 K6 by region (H=30)
| region | pi_0 K6 | mean dwell |
|---|---|---|
| settling_dwell (strict 1–2) | **0.95** | (near-optimal) |
| strict-0 transport | 0.042 | 0.25 |
| strict-0 contact_retention | 0.000 | 0.00 |
| strict-0 braking | 0.083 | 0.50 |
| strict-0 overshoot | 0.000 | 0.00 |

## Probe 3 — is the carry-region ~0 a horizon artifact? No.
pi_0 K6 vs horizon: transport {H30 0.042, H60 0.042, H100 0.042, H160 0.042}; contact_retention {H30 0.000, H60 0.000,
H100 0.042, H160 0.083}. Even at H=160 pi_0 essentially never delivers from carry states — **pi_0 is genuinely incompetent
in the carry region, not horizon-limited.**

## Structural conclusion — a dichotomy with no middle
The gate-active region is bimodal, with **no clean 0.2–0.8 headroom region**:
- **settling (strict 1–3)** — in-distribution for the critic (trained on target_entry/braking/settling), pi_0 **saturated**
  at K6 ≈ 0.95 (no room to convert — confirmed twice now: VALUE_GUIDED_PREFIX_SEARCH and this).
- **carry (strict-0 transport/contact_retention)** — the *dominant* gate-active region, but **OOD for the current critic**
  (it never trained there) AND **pi_0-incompetent** (K6 ≈ 0 even at H=160).

So "just go upstream for headroom" does not hold: there is no in-distribution gate-active panel where pi_0 both acts and
has room to improve. The settling region has no room; the carry region is a *different, harder problem* (pi_0 broadly
fails there, and the value we validated was only validated on the pi_0-saturated settling region).

## What this means for the arc
The value→policy-conversion question, framed as "beat pi_0's K6 on a headroom panel," is **not answerable with the current
pi_0 + critic**, because their competent/in-distribution region (settling) is exactly where pi_0 is already near-optimal.
This is the deep reason LOCAL improvement caps at the supervised ceiling here: pi_0 IS the ceiling on the region it and the
critic cover, and the region with room is one neither covers.

## Redirect (needs your decision)
Testing conversion in a region with genuine headroom requires **extending competence to the carry region**, not another
search on settling:
1. **Retrain pi_0 + critic to include the carry region** (transport / contact_retention as late families), giving a
   baseline with headroom AND an in-distribution critic there; then the prefix-search / chunk-critic question becomes
   answerable. This is a scope step up (new training), not a bounded audit.
2. **Or accept the settling result as final** — on the in-distribution region, pi_0 is near-optimal and the bounded value
   signal has no local headroom to convert; further conversion work is unwarranted until the baseline is extended upstream.
The bounded-search machinery is built and reusable for whichever region we choose.

## Machinery built (correct, reusable — not wasted)
- lib `hymeko_rl/coin_delivery/coin_prefix_search.py` — critic-independent generator + receding-horizon controllers:
  `sel_pi0`, `make_sel_random` (RANDOM_VALID control), `make_sel_search` (reward / value / reward_value, Q_target
  bootstrap), `candidate_outcomes` (oracle coverage), `run_controller`, `buffer_obs_sample`.
- entry `experiments/…/rl_entry/coin_upstream_headroom_prefix_search.py` — coverage / selection-quality / unconditional-ΔK6
  reported separately; (strict×family) strata; both arms; per-state hierarchical bootstrap; equivalence-band CI. Runnable;
  on strict-1–2 it correctly reports settling-only, pi_0 K6 ≈ 0.95, near-zero coverage (the panel is genuinely empty of
  headroom — the entry is not at fault).
- test `test_prefix_search_offsets_and_random_control` (offsets geometry, pi_0-select, RANDOM_VALID determinism+coverage).
  25 tests pass, ruff F-clean.

## Provenance
Branch `recovery/coin-hymeko-bundle-and-results`; pi_0 `1902454c`; probes over held-out seeds ≥6200/6300 (disjoint from
train 6000–6088 / dev 6100–6148). No new campaign, no reward/task change, no CORE.YAML items. Structural probes are
pi_0-only (no training), so seed-robust; the pi_0 K6 rates are single-pass over 12–24 states per family (a data point on
the geometry, not a tuned metric).
