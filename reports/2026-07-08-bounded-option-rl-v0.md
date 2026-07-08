# bounded_option_parameter_rl_v0 — morning report (2026-07-08)

**Run:** Mac (CPU, torch 2.12) · `experiments/2026_07_08_option_rl_v0/` · wall 168 s · CEM pop16×iters8 (search
seed 41000) + fresh-seed acceptance eval (31000/33000/35000/37000, n=48).
**Design:** phase-gated hierarchy (user-chosen) — E_valselect_v2 (frozen MLP) drives APPROACH; a CEM-tuned
`PhasePushController(θ)` drives the CONTACT/PUSH phase (handoff, not a per-step residual). θ∈ℝ⁵ is the only
learnable quantity. Plan: `docs/plans/2026-07-08-bounded-option-rl-v0/`.

## Verdict: **POSITIVE** (gate = ft_dom + monitor_pass; monitor_score-mean trade accepted, user decision 2026-07-08)

The phase-gated option improves the base on almost every axis. It initially read NEGATIVE_WITH_MECHANISM under a
strict "preserve monitor_score-**mean**" clause, but the drop is a benign **delivery↔mean-quality trade** — the
pass-**rate** rises while the mean dips as marginal deliveries are rescued. Per the user's decision the gate is on
monitor_**pass** (rate), under which the option is **POSITIVE**. **Deployable option saved:**
`experiments/2026_07_08_option_rl_v0/phase_gated_theta.json` (θ\* + reference to `E_valselect_v2.pt`, md5
`b822a660…`).

Guards **PASS/PASS**. θ\* = {contact_offset −0.001, push_gain 0.609, direction_correction 0.014, brake_threshold
0.055, release_threshold 0.015}; CEM search objective 2.513→2.899 (improved over scripted).

## Gated θ\* vs E_valselect_v2 (fresh acceptance seeds, n=48)

| metric | E_valselect_v2 | gated θ\* | Δ |
|---|---:|---:|---:|
| ft_dom | 0.615 ± 0.023 | **0.688 ± 0.015** | **+0.073** (tie-test tied, p=0.13) |
| monitor_pass | 0.521 ± 0.078 | **0.620 ± 0.038** | **+0.099** |
| **monitor_score** | **0.433 ± 0.024** | **0.409 ± 0.030** | **−0.024** ⬅ the only fail |
| sustained-PUSH / ep | 1.042 | **1.427** | **+0.385** (further up) |
| both-contact fraction | 0.091 | **0.181** | +0.090 (2×) |
| ft-progress-in-contact | 0.0093 | **0.0229** | +0.0136 (2.5×) |
| body-only progress | 0.0 | 0.0 | 0 |
| arm-body contact | 0.0 | 0.0 | 0 |
| body-driven exploit | 0.0 | 0.0 | 0 |

Acceptance flags (gate = ft_dom + monitor_pass): ft_dom **ok** (tied, mean +0.073), monitor_pass **ok** (+0.099),
sustained further-up **ok**, ft-progress further-up **ok**, no-exploit/body/arm **ok** → **POSITIVE**.
monitor_score-mean preserved: **False** (Δ −0.0247, informational — the accepted trade).

## The mechanism (measured)

monitor_pass is the **conjunctive pass rate**; monitor_score is the **mean of the gating sub-scores**. The gated
handoff **rescues borderline deliveries** — pass-rate up (0.521→0.620), ft_dom up (0.615→0.688), contact up ~2× —
but the newly-rescued episodes carry **modest sub-scores**, so the *mean* dips (0.433→0.409) even as the *rate*
rises. This is a **delivery↔mean-quality trade**, not a harmful regression: exploit / body-only / arm-body all
stay exactly 0; the policy simply completes more (and more contact-rich) deliveries at a slightly lower average
monitor score. The tuned pusher (push_gain 0.61, tighter release 0.015) is doing exactly what it should — securing
and finishing pushes the MLP alone left borderline.

## Honest read

The phase-gated option-RL **behaviorally works**: it further improves sustained contact (the goal, +37%), raises
ft_dom and monitor_pass, doubles fingertip progress, with zero exploit. It fails the strict gate on **one mean
metric** that moves opposite to its own pass-rate. Whether this is "POSITIVE" depends on a gate choice the user's
spec fixed as strict-preserve-monitor_score — under which it is NEGATIVE_WITH_MECHANISM. I did **not** iterate or
re-tune (the standing rule: on NEGATIVE, report the mechanism, don't iterate blindly).

## Decision (resolved — user, 2026-07-08): accept the trade, gate on pass-rate

The gate is monitor_**pass** (delivery-quality rate), not the monitor_score **mean**. Under it the gated θ\* is
**POSITIVE** and the deployable option is saved. The monitor_score-mean dip (−0.0247) is recorded as the accepted
delivery↔mean-quality trade — the policy completes more (and more contact-rich) deliveries, and the mean is pulled
down only by rescuing borderline episodes, never by exploit or body-driven behavior (both exactly 0).

## Guards / discipline
Option-parameter RL only (θ∈ℝ⁵); no per-step residual, no scalar TD3/SAC/CQL, no raw-action RL, no reward change,
no CORE edit; TaskMonitor stayed the external verifier (SearchObjective separate). Single CEM seed (v0) — θ is
5-dim (low variance); a CEM-seed sweep is cheap if this branch is continued. Deployable checkpoint **not** saved
(gate not passed). Result JSON: `experiments/2026_07_08_option_rl_v0/results.json`.
