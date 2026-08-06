# fresh_eval_seed_confirmation_E_valselect_v2 — CONFIRMED (2026-07-08)

**Run:** Mac (arm64, torch 2.12, CPU eval) · `experiments/2026_07_08_fresh_eval_confirm/` · wall 245 s ·
**fresh eval seeds 31000/33000/35000/37000** (disjoint from val 7000, test 9000/11000/13000/15000, search 20000).
n=48 per seed. Same frozen TaskMonitor + ledgers.

## Verdict: **CONFIRMED** — E_valselect_v2 holds (and beats baseline) on unseen seeds

The POSITIVE_ROBUST artifact is not an eval-seed artifact. On four fresh seeds it passes every gate condition, and
on these seeds it is **significantly better** than baseline on ft_dom (not merely tied).

Guards **PASS/PASS** (tensor-contract PASS, policy-provenance PASS). md5: baseline `edf4fe81…`, **E_valselect
`b822a660…`**, C_anchor_repr `0971a6e6…`. v2b certified.

| metric (mean ± std, 4 fresh seeds × 48) | baseline (DAgger) | C_anchor repr | **E_valselect_v2** |
|---|---:|---:|---:|
| ft_dom | 0.458 ± 0.066 | 0.521 ± 0.069 | **0.615 ± 0.023** |
| monitor_pass | 0.344 | 0.469 | **0.521** |
| monitor_score | 0.179 | 0.277 | **0.433** |
| sustained-PUSH / ep | 0.307 | 0.563 | **1.042** |
| both-contact fraction | 0.045 | 0.060 | **0.091** |
| mean contact-window len (steps) | 11.8 | 13.4 | 13.4 |
| ft-progress-in-contact | 0.0023 | 0.0051 | **0.0093** |
| body-only progress | 0.0 | 0.0 | 0.0 |
| arm-body contact | 0.0 | 0.0 | 0.0 |
| body-driven exploit | 0.0 | 0.0 | 0.0 |
| violation_dist | fingertips_never_approached ×4 | fingertips_never_approached ×4 | never_delivered ×2, fingertips_never_approached ×2 |

## Gate (E_valselect vs baseline, fresh seeds): **PASSED**

- **ft_dom:** decision **better** (z=3.07, **p=0.0021**), baseline 0.458 → E_valselect 0.615, Δ **+0.156**. (On the
  fresh seeds the baseline drew lower, 0.458 vs its 0.568 on the v2 test seeds — the *baseline* is eval-seed
  sensitive; E_valselect is tight, std 0.023.)
- **monitor_score up:** 0.433 vs 0.179 ✓ · **sustained-PUSH up:** 1.042 vs 0.307 (3.4×) ✓ · **clean:** exploit /
  body-only / arm-body all 0 ✓.

So E_valselect remains deployable: preserves-or-exceeds ft_dom (here exceeds), improves monitor_score + sustained
contact ~3×, zero exploit, clean body/arm-body — on seeds it never saw. The ordering baseline < C_anchor <
E_valselect holds on every metric, consistent with the v2 run.

## Reporting gap (honest)

`total_reward` came back **NaN** — `measure_policy`'s coord path does not expose a total-reward key under
`_coordination_metrics`, so the reward field was not captured (it is not gate-relevant). If a reward number is
wanted it needs a small addition to the coord-metric path; flagged, not fixed here.

## Decision

Fresh-eval confirmation **passes**. Per the standing plan, `bounded_option_parameter_rl_v0` (option-level RL over
the 5 PhasePushController params, acceptance vs these E_valselect fresh-eval metrics) is unblocked — pending one
composition-design clarification (how the 5 scripted option params compose with the learned MLP base), raised
separately. No CORE edit; TaskMonitor external verifier throughout.
