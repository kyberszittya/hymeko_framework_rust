# R3-C — FIRST TEACHER-FREE LEARNED s1 K6 DELIVERY (action-preserving authority unlock)

**2026-07-28 · branch `recovery/coin-r9-causal-residual-delivery` · worktree `hymeko_coin_r9_wt` · dev s1 (14250) · s4/s7 untouched · f1–f4 SEALED · teacher-free deploy · TD3**

## Summary — the coin is in

**`FIRST_LEARNED_S1_K6_DELIVERY`.** A learned, teacher-free KINETIC coin-following policy delivers the s1 coin into the strict K6
zone: **min_dtz 7.96 mm, K6 dwell 19 frames (≥ 6 required), dtz_end 17.2 mm, clean (0 stall / 0 clamp / 0 reversal), safe, coin
still moving at release** — no teacher, no CEM, no oracle in the loop, deterministic. Independently verified against the canonical
`delivery_success(m, DELIVERY_CFG)` contract. This is the campaign goal (R9 "vidd be az érmét") reached.

The result follows directly and cleanly from the two diagnostics that preceded it: the audit localised the wall to *residual
authority*, and the torque-span diagnostic proved it was a **bound** limit (α = 0.15 clipped 94 % of the teacher's correction, all
of which lay in the current span) — α ≈ 1.0 reaches the corridor. R3-C unlocked exactly that authority, **action-preservingly**, and
TD3 learned to use it.

## The construction — action-preserving authority unlock

Not a new basis, not a direct 6-D actor, and NOT "load the R2 actor and widen α" (which would apply 6–13× larger corrections in
the first rollout). Instead: keep the frozen R2 residual at α = 0.15 and ADD a **zero-initialised** state-dependent expansion head on
the SAME 4-D per-step basis and SAME augmented state:

```
u = clip( u_clone  +  0.15·a_R2(aug)  +  β·δ_expand(aug) ,  −1, 1 )
```

- **`AUTHORITY_EXPANSION_UPDATE_ZERO_IDENTITY_PASS`** — with a bit-exact zero expansion head (final linear zeroed → `tanh(0) = 0`
  for all inputs), this is **bit-identical to the R2 champion** (`max|Δcoin_trace| = 0.00e+00`) for both families, regardless of β.
  RL opens the authority gradually from the known-good policy, never starting as a suddenly-amplified one.
- Two families: **C1 β = 0.85** (total ≈ 1.0), C2 β = 1.85 (total ≈ 2.0). The delivering seed is **C1** — total authority ≈ 1.0,
  exactly the corridor-reaching level the torque-span sweep predicted.
- Two prev-residual trackers keep the R2 head's augmented state identical to the R2 champion's (the identity depends on it) while
  the expansion head carries its own history. `aug_trace` records only the EXPANSION (state, action) → TD3 trains only δ_expand.
- **Safety is the final clip + slew + governor, independent of β** (verified: a non-zero expansion at β = 1.85 keeps every action
  in [−1, 1] and the rollout inside the motion contract). Exploration is β-scaled so applied noise is β-invariant.

## The run — production-scale smoke DELIVERED (C1, seed 0, Phase A)

Per the operating contract's "production-scale smoke before queuing a multi-seed run," the mandatory 1-seed / 1-family smoke was
run first — and it delivered strict K6 at option 375/600, so the run **froze immediately** (the panel never needed the remaining
seeds). Clean, monotone learning from the R2 baseline:

| eval option | 175 | 250 | 300 | 325 | 350 | **375** |
|---|---|---|---|---|---|---|
| min_dtz (mm) | 32.1 | 35.97 | 39.49 | 30.58 | 28.07 | **7.96** |
| clean | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| strict K6 | ✗ | ✗ | ✗ | ✗ | ✗ | **✓** |

Every eval clean (0/0/0) and safe throughout — the stall-aware champion + envelope (R3-B) held; the policy did not cheat by
stalling or clamping. It crossed the close-moving corridor (≤ 30 mm) around option 325 and reached the K6 zone by 375.

## Independent verification (from the frozen checkpoint)

Reconstructed the full policy from `first_k6_checkpoint.json` (frozen R2 champion + expansion actor state) and re-rolled
teacher-free from the frozen KINETIC entry:

| check | result |
|---|---|
| strict K6 — `delivery_success(m, DELIVERY_CFG)` | **True** (canonical contract) |
| K6 dwell | **19** frames (≥ 6) |
| min_dtz / dtz_end | **7.96 mm** / 17.2 mm (inside the 20 mm zone) |
| teacher-free | deploy = clone + frozen-R2-NN + expansion-NN; no teacher / θ / CEM attribute |
| safety | peak_qdot 2.03 ≤ 3, coin speed 0.32 ≤ 1.5 |
| deterministic replay | `max|Δcoin_trace| = 0.00e+00` (two rolls) |
| reproduces from committed code | re-run smoke → checkpoint **bit-identical** |

## Files touched

| file | change | role |
|---|---|---|
| `hymeko_rl/coin_delivery/theta_option/kinetic_authority_unlock.py` | +139 (new) | `AuthorityUnlockController` (two-head action-preserving), `zero_init_detactor` (bit-exact 0), `champion_key_unlock`, β-scaled `expl_noise_for`, `make_collect_unlock` / `make_dev_eval_unlock`, `stop_on_strict_k6` |
| `hymeko_rl/experiments/coin_kinetic_r3c_rl.py` | +190 (new) | campaign driver: R2 regen → C1/C2 families × seeds → Phase A/B (80/20 curriculum) → freeze-on-first-K6 (checkpoint + deterministic-replay + teacher-absence audit) |
| `hymeko_rl/coin_delivery/theta_option/kinetic_residual2.py` | +13/−… | extract `augmented_state` as the single bit-exact source (R2 + R3-C share it) |
| `hymeko_rl/coin_delivery/theta_option/kinetic_rl2.py` | +9 | additive `stop_when` early-stop hook in `train_perstep` (freeze-on-first-K6) |
| `hymeko_rl/coin_delivery/theta_option/kinetic_rl3.py` | +18 | additive `make_controller` factory param on `collect_episode3` / `make_collect3` (R3-C reuses the tested curriculum + reward3) |
| `hymeko_rl/tests/test_coin_kinetic_contract.py` | +115 (+5 tests) | zero-init exactness; update-zero identity (C1 & C2); β-independent safety/slew; deterministic replay; champion order + β-scaled noise |
| `reports/2026-07-28-coin-r9-r3c-rl/{smoke.json, first_k6_checkpoint.json}` | new | verdict + frozen policy checkpoint |

## Tests / static analysis

- **Full `test_coin_kinetic_contract.py` — 34 passed** (29 prior + 5 R3-C gate tests). `AUTHORITY_EXPANSION_UPDATE_ZERO_IDENTITY`
  bit-exact for both families; safety β-independent; deterministic replay; the 3 committed authority/torque-span suites unaffected
  by the additive refactors.
- `ruff check` clean on all touched files; `radon cc -a` **A**; worst function `run` = **B(8)** (refactored down from 19 via
  `_setup` / `_run_panel` / `_panel_config`). No new suppressions; no §6.5 anti-patterns (config-dispatched families, not a
  Cartesian dump; the controller is swapped by a factory, not duplicated; string-free enums).

## Provenance

Off `2f437c73` (torque-span commit). Python 3.11.15 / mujoco 3.10.0 / numpy 2.4.6 / torch 2.12.0 / macOS-26.5.2-arm64 (Apple
Silicon, CPU). Seeds: cradle 14250; R2 regen 0; expansion TD3 seed 0. β = 0.85 (C1). Peak RSS **0.36 GB**; smoke wall **51 s**.
Frozen checkpoint `reports/2026-07-28-coin-r9-r3c-rl/first_k6_checkpoint.json` (r2_champ + expansion state, β, α0). Deterministic:
committed code reproduces bit-identically.

## Status & honest scope

`FIRST_LEARNED_S1_K6_DELIVERY` — **frozen, verified, committed, tagged `coin-r9-first-learned-s1-k6-delivery`.** The first STOP
condition is met, so the full C1/C2 multi-seed panel was **not** run (the first smoke seed delivered; the run froze on it per the
freeze rule). Honest scope: this is **one seed, one family (C1, β = 0.85), the s1 dev cradle**, teacher-free strict K6. It is NOT
yet a robustness or generalisation claim — the sealed panels (f1–f4) and validation cradles (s4/s7) remain untouched by rule, and
generalisation, the clone-vs-zero ablation (how much the frozen clone/R2 still contribute at α ≈ 1.0), and the C2 family are
explicitly **post-first-K6** follow-ups for review. STOPPED.
