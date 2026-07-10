# Monitor-aligned reward repair — final synthesis (frozen)

**Date:** 2026-07-10 · Aiko · branch `hymeko-neuro-migration` · **FROZEN — no further experiments in this arc.**
Kato-facing synthesis + paper-ready material. All numbers are from committed runs; nothing new was run for this
document.

---

## 1. Narrative — detect → explain → repair → verify

- **Detect.** Stage A found a reward-computation misalignment: on MetaWorld pick-place the HyMeKo reward SoT could
  be re-parented by ablating `mw_in_place`, and BC-anchored policies showed rising reward↔monitor disagreement
  under the ablated reward — a signal that the shaped reward is not tightly bound to task success.
- **Explain.** CIP / LiNGAM-SH explained *which* reward proxies are load-bearing: `mw_in_place` (progress) is the
  dominant driver (5-seed robust), `mw_grasp` is inert, `mw_dist` a weak secondary — i.e. the reward leans on a
  progress proxy that can move without delivery.
- **Repair.** A **monitor-aligned** reward variant re-shaped the dense reward so its dominant terms (gated delivery,
  success) require real manipulation: approach (potential) + capped contact + **potential-based lift** + delivery
  **gated on grasp/lift evidence** + a large success bonus − a hover-stagnation penalty.
- **Verify.** Eight **decoupled counterfactual** trajectories (breaking the contact↔delivery collinearity of the
  scripted data) verified that the repaired reward **suppresses proxy farming ≈45×** while the original reward
  scores a pure proxy-exploit as high as true delivery. Offline recomputation (CIP) and cross-view HyMeKo checks
  confirm consistency.
- **BC-anchored smoke** suggests the repaired reward does not immediately destroy policy performance (success not
  destroyed; return learnable; disagreement 0) — single seed.
- **From-scratch learning is explicitly out of scope / gated:** the from-scratch optimizer does not yet pass its
  reach gate, so no learning-role claim is made.

## 2. Final claim set (careful wording)

- **Claim A.** The original MetaWorld-style reward can assign high reward to proxy/contact-like trajectories that do
  not represent task completion. *(Supported: proxy-exploit total 88.8 ≈ delivery 86.8, 82 % of success.)*
- **Claim B.** A HyMeKo/runtime-monitor-aligned reward can suppress such proxy farming while preserving dense
  pre-success shaping. *(Supported: proxy total 2.0; dense pre-success shaping on approach/delivery.)*
- **Claim C.** On decoupled diagnostic trajectories, the repaired reward suppresses proxy exploitation by ≈45×
  relative to the original reward. *(Supported: proxy/success 0.816 → 0.018.)*
- **Claim D.** Offline recomputation shows lower reward↔monitor disagreement for the repaired reward, with
  cross-view verification passing. *(Supported: decoupled disagreement 0.071 → 0.000; all variants cross-view ✅.)*
- **Claim E.** A BC-anchored smoke test suggests the repaired reward remains usable for policy fine-tuning, but
  multi-seed policy robustness remains future work. *(Supported, single seed: success 0.50 vs BC 0.75 / original
  0.625, within env variance.)*

**Explicit non-claims.** We do **not** claim MetaWorld is globally wrong; that from-scratch learning is solved;
that an adversarial RL policy would (or would not) discover the exploit; or any policy-learning superiority until
multi-seed RL validates it.

## 3. Result tables

### A — R2 offline recomputation (scripted rollouts, n=60, monitor = success)

| variant | disagreement | corr(delivery) | corr(progress) | reward std (density indicator) | cross-view |
|---|---:|---:|---:|---:|---|
| original | 0.000 | 0.83 | 1.00 | 472 | ✅ |
| `mw_in_place_off` | 0.001 | 0.78 | 0.96 | 84 | ✅ |
| **monitor_aligned** | 0.000 | 0.81 | 0.98 | 2566 | ✅ |

*On scripted rollouts contact/progress/delivery are collinear, so all three align (disagreement ≈ 0) — R2 shows
alignment and cross-view consistency but cannot separate farming. That separation is Table B.*

### B — Anti-farming validation (decoupled counterfactual trajectories; total reward)

| trajectory class | original | `mw_in_place_off` | **monitor_aligned** | interpretation |
|---|---:|---:|---:|---|
| far / inactive | −52.0 | −60.0 | **0.0** | all low — correct |
| approach only | −46.3 | −54.3 | **−0.6** | small pre-success shaping |
| hover-near farming | −6.4 | −30.4 | **−20.0** | monitor_aligned penalizes hover |
| bare-contact farming | 8.8 | −23.2 | **2.0** | contact capped without motion |
| grasp/lift, no delivery | 24.8 | −23.2 | **5.5** | moderate, **below** delivery |
| delivery progress | 86.8 | −1.2 | **11.8** | rewarded |
| true success | 108.8 | 4.8 | **112.4** | strongest event |
| **proxy_exploit** | **88.8** | −23.2 | **2.0** | **original farms it (≈ delivery); monitor_aligned suppresses** |

Suppression: proxy/success **0.816 → 0.018 (≈45×)**; 8-class disagreement **0.071 → 0.000**.

### C — Claim-discipline table (whole Stage-B arc)

| stage | question | result | valid claim | caveat |
|---|---|---|---|---|
| reward-computation ablation | Is `mw_in_place` load-bearing in the reward? | SUPPORTED, 5-seed | `mw_in_place` dominates the reward mechanism | reward-computation level only |
| BC-anchored fine-tune | Does ablation change policy success? | NOT robust (REINFORCE); ~equal under PPO (both ~100%) | task success unaffected in a BC-anchored policy | single→5-seed; PPO both succeed |
| from-scratch diagnostics | Was 0%-vs-0% the reward or the setup? | PPO-setup issue (harness/metrics proven correct) | not a reward result / not a wall | — |
| optimizer repair | Can from-scratch PPO learn reach? | 3 real bugs fixed; reach improved but **not robust** | reach sometimes learned; ablation still invalid from scratch | env non-determinism; original reward can't bootstrap |
| monitor-aligned repair | Can we reduce reward/monitor disagreement, keep dense? | R1–R3 pass; disagreement ↓, dense, cross-view ✅ | a monitor-aligned reward reduces disagreement while staying dense | R2 collinear; R3 single-seed |
| anti-farming validation | Does the repair suppress proxy farming? | **YES, ≈45×** on decoupled trajectories; HEALTHY (5/5) | repaired reward suppresses proxy farming, preserves shaping | counterfactual trajectories, not a learned adversary |

## 4. Important repair detail — a farming bug found *inside* the repair

The first repaired reward scored **`grasp/lift no delivery` (19.75) above `delivery` (11.80)**: the lift term
rewarded the object's *held height every step*, so holding the object aloft accumulated lift reward — a farming
vector introduced by the repair itself. **Fix:** the lift reward was changed to **potential-based lift progress**
(reward *raising* per step, gated by grasp; it telescopes to total height raised and cannot be farmed by holding).
After the fix `grasp/lift no delivery` → 5.45 (below delivery) and the 8-class disagreement → 0.000. **Why it
matters:** a monitor-aligned reward repair *must itself be audited for farming* — the same decoupled-trajectory
discipline used to catch the benchmark's proxy caught a proxy in our own fix.

## 5. Density — careful wording

`monitor_aligned` preserves dense pre-success shaping — shown by the nonzero pre-success reward fraction (approach
0.95, delivery 1.0), the per-component diagnostics (multi-component approach/delivery shaping, entropy 0.5–0.6), and
the approach/delivery shaping curves — **while suppressing proxy/contact farming through gates and penalties**. We
do **not** claim it is "dense because it has the highest variance."

## 6. Kato-facing short summary

> We found that the benchmark reward can be semantically misaligned with task success in diagnostic counterfactual
> cases. Using a HyMeKo/runtime monitor, we repaired the reward by gating proxy terms through task-progress
> evidence. The repaired reward keeps dense shaping but suppresses proxy farming, and the result is verified both by
> offline recomputation and cross-view HyMeKo checks. This supports using HyMeKo not only to audit learned policies,
> but also to repair reward functions.

## 7. Paper-ready paragraph

> These results suggest that runtime monitors can serve not only as post-hoc success checkers but also as semantic
> instruments for reward repair. In the pick-place case, the original shaped reward assigned high value to
> counterfactual proxy trajectories with contact-like evidence but no delivery progress. A monitor-aligned reward
> variant, constructed by gating contact, lift, and delivery terms through task-progress predicates, suppressed
> this proxy-farming behavior while preserving dense pre-success shaping. Cross-view verification confirmed that the
> repaired reward and its causal summaries remained consistent with the HyMeKo representation.

## 8. Future work (three options)

1. **Adversarial policy search** — train a learner to actively *discover* the proxy exploit, testing suppression
   against an adaptive adversary rather than hand-built counterfactuals (needs RL).
2. **Multi-seed BC-anchored fine-tuning** — turn the single-seed R3 into median/IQR to quantify the repaired
   reward's policy robustness.
3. **SAC / curriculum from-scratch learning** — only after the from-scratch optimizer gates (reach → grasp) are
   defined and met.

**Recommendation: stop here for the current arc.** The reward-repair result is frozen, verified in both directions
(offline + anti-farming), and Kato-ready; the three options above are larger, gated follow-ons.

## Artifacts

- R1–R3: `reports/2026-07-10-pick-place-monitor-aligned-reward-repair.md` + `figures/2026_07_10_monitor_aligned_reward/`, `…_bc_smoke/`.
- Anti-farming: `reports/2026-07-10-monitor-aligned-anti-farming-validation.md` + `figures/2026_07_10_anti_farming_validation/`.
- Code: `hymeko_rl/eval/cip/monitor_aligned_reward.py`, `hymeko_rl/eval/reward_repair/anti_farming_validation.py` (26 tests green).
