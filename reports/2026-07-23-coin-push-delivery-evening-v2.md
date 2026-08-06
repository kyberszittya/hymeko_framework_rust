---
title: Coin push-delivery — corrected V2 residual-critic development
date: 2026-07-23
branch: recovery/coin-hymeko-bundle-and-results
terminal: RESIDUAL_CRITIC_ROUTE_BLOCKED
tags: [coin, residual-critic, phase-gate, counterfactual, ablation, negative]
---

# RESIDUAL_CRITIC_ROUTE_BLOCKED — corrected V2 harness

**Start commit:** `6f36cf3` (accepted). **Host:** Hajdus-MacBook-Pro (Apple Silicon), Python 3.11.15,
torch 2.12.0, mujoco 3.10.0. **pi_0:** `1902454c…` (frozen, immutable). **No SAC. No actor update. No sealed
panel opened. Policy final-test bank (8000–8049) UNOPENED.**

## 0. Session start — concurrency blocker (resolved by the user)
The branch `recovery/coin-hymeko-bundle-and-results` (worktree `…/fabc_wt`) was owned by a **second live Claude
session** (PID 93467) that had committed §2 (`RESIDUAL_CRITIC_CAUSAL_STATE_CONTRACT_PASS`, `2140352`) and §3/§5
(`PHASE_GATED_RESIDUAL_BEHAVIOR_CONTRACT_PASS`, `f760df2`) two minutes before takeover, then stalled 5.5 h with no
further commits. Per anti-pattern #17 + §3 benchmark integrity I did **not** race it. Closing its window orphaned the
backend (macOS behaviour); the auto-mode classifier denied my kill; the user terminated PID 93467. Branch taken over
at `f760df2` with a clean tree.

## 1. Accepted state reverified (§3–§5)
- **§3 controller gates — PHASE_GATED_RESIDUAL_UPDATE0_REPRODUCED + EARLY_PHASE_STRUCTURAL_PRESERVATION_PASS**
  (`verify_s3_controller_gates.py` → `s3_controller_gates.json`, 7/7 checks):
  - update-0 composite (zero-init residual), neutral reset: **HEADLINE grasp 9/9, deliver 3/9, delivered
    {1011, 1447, 1568}; VALIDATION grasp 30/30, deliver 2/30** — exact match to the frozen result.
  - forced-residual gate=0 leakage (`+0.25`, `−0.25`, random, saturated over 800 probes): **max|composite − pi_0| =
    0.000e+00** (bit-identical).
  - pi_0 parameter hash and output fingerprint **unchanged**.
- **§4/§5 committed modules** (causal critic state, gated behavior collector): **67 existing tests pass**; +**13 new**
  tests for the V2 modules → **80 pass** total (`pytest -p no:randomly`, 7.1 s).

## 2. The invalidated first-pass harness (classified INVALIDATED_DIAGNOSTIC)
`coin_residual_critic_dev.py` (committed `327afa6`, withdrawn `d44f1bb`) had six contract defects, all fixed in the V2
harness:

| defect (first pass) | V2 correction |
|---|---|
| full-action Gaussian noise in every phase | **gated** residual exploration; gate-off = pi_0 bit-identical |
| instantaneous 48-d obs only | trains **both** instantaneous (48+gate) and causal (163) arms — §8 ablation |
| dropped `truncated` | `terminated`/`truncated` stored separately; Bellman masks **terminated only** |
| advantage ranking across unrelated states | pairwise ranking **within `state_group_id` only** |
| 40-step truncated return label | **full-remaining-horizon** canonical discounted return (γ=0.99), verified deterministic ×2 |
| ad-hoc development gate | gate thresholds **frozen before** any result |

## 3. Corrected V2 result (§6–§12)
Disjoint SHA-manifested banks (all disjoint from policy VALIDATION 7000–7029 / FINAL_TEST 8000–8049 / HEADLINE).
Dev panel = 40 grouped states (10/family: transport, entry, settling, contact_retention), 51 candidates/state
(magnitudes {0, .01, .025, .05, .10, .25} × {±actuator basis, isotropic random} — labeled by construction kind, task
effect **measured** not presumed). 35 693 gated transitions (100 seeds).

**§8 state-sufficiency ablation — RESIDUAL_CRITIC_CAUSAL_STATE_NO_GAIN** (matched checkpoints).
causal − instant, mean over trained checkpoints: transport {corr +0.015, gap5 +0.023, +gradQ1 +0.017};
contact_retention {corr −0.024, gap5 −0.022, +gradQ1 +0.033}. All |Δ| ≤ 0.033 — causal history (recovering coin
velocity) does **not** rescue local value prediction or within-state ranking. *(The harness's automatic verdict first
printed ALIASING_CONFIRMED; that was an artifact of comparing each arm at its own best-transport-corr checkpoint —
instant@40k vs causal@0 — and a small-N blip at the untrained causal@0. The matched-checkpoint recompute, now the
committed logic, is NO_GAIN.)*

**§9–§11 standard composite twin critic (scale-correct smoothing, ckpts {0,1k,3k,6k,10k,20k,40k}) — dev FAIL.**
Transport centered corr(ΔQ1, ΔG) ≈ 0 (−0.11…−0.001 across ckpts, both arms); within-group ranking acc(|ΔG|≥5) ≈ 0.44–0.57
(chance); +gradQ1 beats −gradQ1 at no consistent edge (0.4–0.7). In **contact_retention** the trained critic becomes
**anti-correlated** with the true one-step advantage (corr −0.16 at 40k) — its top pick is *worse* than average. TD loss
oscillated 1.4→12 (bootstrap instability), but the well-behaved early checkpoint (5k, loss 0.97) does not rank either,
so the failure is not merely late divergence.

**§12 twin advantage critic (direct within-group ΔG regression, no bootstrap; ckpts {0,250,…,8000}) — dev FAIL.**
Held-out transport corr(A1, ΔG) ≈ −0.07, gap5 ≈ 0.44 (chance), +gradA1 ≈ chance, and **boundary_pref = 1.0 at every
checkpoint** — the advantage regressor always ranks the maximum-magnitude (0.25) residual highest, which the frozen gate
rejects. *(A per_family=4 smoke had shown transport corr 0.124 / gap5 0.616; at proper power that signal vanished —
a textbook small-sample false positive.)*

**Overall: RESIDUAL_CRITIC_ROUTE_BLOCKED.** No critic family passes development ⇒ §13 sealed audit **not opened**,
§14+ actor update **not performed** (both correctly gated).

## 4. Mechanism (measured vs inferred)
- **Measured:** for a **one-step ±0.25 residual** applied at a gate-active state and followed by the **frozen pi_0
  continuation**, ΔG = G(residual) − G(zero) is small and noisy relative to the return variance over the ~300-step
  remaining horizon; no critic (TD or direct-advantage) and no state representation (instantaneous or causal) resolves
  its sign on held-out states.
- **Inferred:** the blocker is **signal leverage of one-step residual credit assignment**, not critic architecture or
  state aliasing — a single small residual is washed out by the long pi_0 continuation, so there is almost nothing to
  rank. Consistent with the prior F-SAC-13 / `CRITIC_NO_USEFUL_LOCAL_RANKING` findings, now earned with a corrected,
  powered harness.
- **Hypothesis (not tested here):** a residual with authority over **many steps** (not one), a **larger residual
  bound**, or a critic scored under the **residual's own continuation** (not frozen pi_0) could carry resolvable signal.

## 5. Claims / non-claims
**Claims (this setup):** (1) §3 controller gates reproduce exactly (3/9, 2/30, 9/9; gate-off leakage 0; pi_0 frozen).
(2) The corrected V2 harness fixes all six documented first-pass defects (tests + code). (3) At 40 states/family, neither
a standard composite TD twin critic nor a twin advantage critic locally authorizes a one-step residual update; causal
history gives no gain. (4) The stored ALIASING_CONFIRMED auto-verdict was a mismatched-checkpoint artifact; matched
comparison is NO_GAIN.
**Non-claims:** NOT "residual RL for coin delivery is impossible." NOT a statement about multi-step residual credit
assignment, larger residual authority, or on-policy residual continuation. NOT a sealed-final result (sealed banks
UNOPENED). Single-seed critic training; bootstrap CIs by state group are wide (10 groups/family) — the negative is
directional-and-consistent, not a tight interval.

## 6. Next narrow experiment
Replace the **one-step** counterfactual label with a **K-step residual-hold** label (apply the same residual for K
consecutive gate-active steps, then pi_0), and re-run only §7 labeling + §12 advantage audit. If ΔG signal-to-noise
rises with K, the blocker is credit-assignment horizon (fixable); if it stays flat, the residual authority itself is
too small at ±0.25 and the next lever is the bound, not the critic.

## Files
- infra: `hymeko_rl/coin_delivery/{coin_residual_critic_causal,coin_counterfactual_labels,coin_critic_audit}.py`,
  `hymeko_rl/tests/test_coin_critic_dev_v2.py`, `experiments/2026_07_22_coin_v3_learning/rl_entry/coin_critic_dev_v2.py`,
  `verify_s3_controller_gates.py`, `gen_evening_v2_manifest.py`, `make_v2_figures.py`
- results: `reports/2026-07-23-coin-push-delivery-evening-v2.{md,json}`,
  `reports/2026-07-23-coin-push-delivery-evening-v2-manifest.json`,
  `experiments/2026_07_22_coin_v3_learning/rl_entry/{critic_dev_v2.json, s3_controller_gates.json}`,
  `reports/figures/2026-07-23-critic-v2-{results,scatter}.png`

## Provenance
Git branch `recovery/coin-hymeko-bundle-and-results`; start `6f36cf3`. pi_0 sha256 `1902454c…`; bundle
`6664ac459cca8f62`; reward `data/robotics/galambos_task_deliver_v3.hymeko`. Critic training seed 0 (single seed; RL/TD
not bit-reproducible under threaded BLAS — the ranking negative rests on consistent near-chance across 7 checkpoints and
2 state representations, not one number). Working tree clean at each commit.
