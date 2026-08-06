# Guarded RL sanity sequence — null-wrapper → critic-only → guarded CTDE-TD3+BC smoke

**Date:** 2026-07-07 · Git SHA `4320202` (dirty). Non-core. Seed 1, CPU, one seed only (not a sweep). Every step
ran through the **full safety stack** (schema ledger + provenance ledger + formal TaskMonitor) and reports the 14
required fields. v2b reward unchanged; monitor NOT in the reward; no SAC / residual / multi-seed.

## Canonical result

**The earlier RL smoke is now confirmed a genuine failure of the current CTDE-TD3+BC method/config under a fully
guarded setup — NOT a silent tensor or provenance bug.** Both ledgers PASS at every step; the guarded null-wrapper
reproduces the DAgger checkpoint exactly; the guarded smoke still collapses. The failure is in the *method*, and
the monitor's critic-vs-monitor check pinpoints *why*: the critic mis-ranks the exploit above DAgger.

## 14-field table (all three steps)

| field | 1 null-wrapper | 2 critic-only | 3 guarded smoke |
|---|---|---|---|
| reward | −115.15 | −115.15 | −442.04 |
| ft_dom | **0.75** | 0.75 | **0.0** |
| monitor_pass | 0.417 | 0.417 | 0.0 |
| monitor_score | 0.278 | 0.278 | −0.443 |
| violation_reason | no_right_fingertip_contact | no_right_fingertip_contact | coin_pushed_away_during_approach |
| reward-vs-monitor | misaligned (exploit>DAgger) | misaligned | misaligned (concord 0.667) |
| critic-vs-monitor | n/a (critic untrained) | **INVERTED** (exploit>DAgger) | **INVERTED** (one_finger>exploit) |
| tensor-contract | PASS | PASS | PASS |
| policy-provenance | PASS | PASS | PASS |
| actor ckpt hash | edf4fe81… | edf4fe81… | edf4fe81… (init) |
| anchor ckpt hash | edf4fe81… | edf4fe81… | edf4fe81… |
| selected DAgger stage | d3 | d3 | d3 |
| reward file | galambos_task_deliver_v2b.hymeko | (same) | (same) |
| env file | PlanarGraspEnv v2 graded (contact_legality, diff 0.3, 300) | (same) | (same) |
| **step verdict** | **PASS** | **PASS** | **FAIL acceptance** |

## Step 1 — null-wrapper: PASS

`actor_lr = critic_lr = 0`, no noise, 600 steps. **Param hash unchanged** (`e8ff8f82…` in = out), and eval
**reproduces the selected DAgger checkpoint exactly**: ft_dom 0.75, monitor_pass 0.417, monitor_score 0.278 —
identical to the cached `mlp_dagger_selected` reference. Schema PASS (5/5 stages), provenance PASS
(actor=anchor=selected, md5 `edf4fe81…`). The guarded harness is faithful: with no learning, nothing moves.

## Step 2 — critic-only: PASS (mechanically), critic mis-ranks

Actor frozen (`critic_warmup ≥ total_steps` + `actor_lr=0`); critic trains 4000 steps on DAgger-seeded replay.
**Actor unchanged** (ft_dom still 0.75). The trained critic Q on DAgger states:

| action | Q | monitor score |
|---|---:|---:|
| body_shove_exploit | **−5.70** | −0.202 |
| mlp_dagger_selected | −6.55 | **+0.278** |
| one_fingertip | −6.76 | −0.273 |

**critic-vs-monitor INVERTED**: `Q ranks body_shove_exploit above mlp_dagger_selected but monitor prefers
mlp_dagger_selected`. This is the **category-B off-policy OOD overestimation** — a critic trained *only* on
DAgger-seeded replay still values the out-of-distribution body-shove action above the DAgger policy — now surfaced
formally by the monitor's critic-vs-monitor consistency, not just the ft_dom metric. Q also drifts monotonically
negative through training (−1.75 → −7.05), the same divergence signature as the frozen smoke.

## Step 3 — guarded CTDE-TD3+BC smoke: FAIL acceptance

Gate (steps 1 & 2 pass) satisfied → ran one seed, 6000 steps, DAgger init + frozen-DAgger BC anchor (22 526
pairs, anchor loss `2.5e-14 ≈ 0` at init → the anchor *is* the DAgger actor), critic Huber, `critic_warmup=2000`.

**ft_dom 0.75 → 0.0** (total collapse), monitor_pass 0.417 → 0.0, monitor_score 0.278 → −0.443, reward −115 →
−442. Training trace: actor frozen through warm-up (Q −4.9 → −8.8), then the moment the actor follows the critic
(step ~2500+) it diverges — actor loss 11.7 → 22.4, Q −11.8 → −21.6 — and the policy collapses. The collapsed
actor's monitor violation is **`coin_pushed_away_during_approach`**: it moves the coin the *wrong way*.

Acceptance (7 criteria):

| criterion | result |
|---|---|
| ft_dom ≥ 0.452 | ✗ (0.0) |
| monitor_pass ≥ 0.417 | ✗ (0.0) |
| monitor_score ≥ selected (0.278) | ✗ (−0.443) |
| no rise in body-driven violations | ✓ (exploit stayed 0.0) |
| critic-vs-monitor not inverted | ✗ (inverted) |
| policy-provenance PASS | ✓ |
| tensor-contract PASS | ✓ |

→ **FAIL (4/7).** Note the collapse is a *pure delivery collapse*, **not** reward-hacking: exploit stayed 0.0 and
the graded-contact safety held — the policy got worse, it did not cheat. This single-seed collapse (ft_dom 0.0) is
more severe than the frozen smoke's 0.062–0.167; it is a single-seed point estimate (RL carve-out — rest verdicts
on multi-seed median), same *direction and mechanism* (critic divergence + mis-ranking), harsher magnitude.

## Interpretation

- The **null-wrapper reproduces** and both ledgers **PASS everywhere** → the guarded setup is correct; the earlier
  smoke's failure is not a tensor/provenance artifact.
- The **critic-only step isolates the cause**: the critic mis-ranks the exploit above DAgger on DAgger-only replay
  → **category-B OOD overestimation**, confirmed by an external verifier.
- The **guarded smoke reproduces the collapse** and fails acceptance → **CTDE-TD3+BC (this config) degrades the
  DAgger policy**. The lever past the DAgger ceiling is **not** behaviour-regularized off-policy RL as configured.

This matches the frozen v1 + v2 verdicts, now established under the complete safety stack with the full 14-field
provenance. **RL stays frozen.**

## Artifacts

- Checkpoint `experiments/v2_rl_guarded/rl_guarded_smoke_s1.pt`, results `experiments/v2_rl_guarded/results.json`,
  log `scratchpad/sanity.log`. Harness `scratchpad/v2_rl_guarded_sanity.py` (gates step 3 on 1 & 2).
- No GIF: the step-3 policy collapsed to ft_dom 0.0 (pushes the coin away) — an animation adds no discriminating
  value over the numeric verdict. The headline deliverable is the acceptance table.

## What this does NOT license

Per directive: no SAC, no residual RL, no multi-seed RL, no v2b reward change, monitor not in reward. **If** RL is
revisited (research, not now), the guarded evidence points at the **critic's OOD overestimation** as the thing to
fix first — a conservative critic (CQL / OOD penalty), a residual-only actor, or phase-gated correction — and any
such run must clear the same 7-criterion acceptance under the full safety stack. The minimum safety stack
(schema PASS + provenance PASS + monitor active + reward-vs-monitor + critic-vs-monitor) is now demonstrated
end-to-end on live RL.
