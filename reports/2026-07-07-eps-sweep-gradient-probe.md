# ε-sweep + gradient-alignment probe — the off-policy RL lever is closed (with a mechanism)

**Date:** 2026-07-07 · Git SHA `03b01c3` (dirty). Non-core. One seed, CPU. Final diagnostic, not a campaign. One
frozen DAgger base + one frozen STRONG_PASS CQL critic shared across all ε; same phase gate / seed / replay /
monitor / guards; no reward change / SAC / plain actor-critic / multi-seed.

## Verdict

**Stop off-policy RL on this task.** No ε improves the DAgger baseline; every ε>0 degrades it monotonically; and
the gradient probe shows *why*: **the critic's local action-gradient raises Q but reduces two-fingertip engagement —
it is monitor-misaligned exactly where its ranking is STRONG_PASS-correct.** Per the stop rule ("no more RL unless
ε=0.01 clearly improves or preserves all gates"), ε=0.01 does **not** preserve (ft_dom 0.75→0.708, monitor_pass
0.417→0.292) → **RL stays frozen; switch to imitation / monitor-directed / gradient-free.**

Figure: `reports/figures/eps_sweep/eps_sweep_gradient_probe.png`. Data: `experiments/v2_epsilon_sweep/results.json`.

## Part A — ε-sweep (monotone degradation, no ε improves)

| ε | ft_dom | monitor_pass | monitor_score | gate_frac | res mean/max | sat | both-contact | engagement | delivery | anti_exploit |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **0.00** | **0.750** | **0.417** | **0.278** | 0.28 | 0 / 0 | 0 | 0.025 | **4.17** | 0.792 | 0.617 |
| 0.01 | 0.708 | 0.292 | 0.250 | 0.22 | 0.017 / 0.020 | 0.86 | 0.020 | 3.12 | 0.792 | 0.582 |
| 0.02 | 0.500 | 0.250 | 0.206 | 0.18 | 0.030 / 0.040 | 0.76 | 0.015 | 2.46 | 0.708 | 0.546 |
| 0.03 | 0.417 | 0.208 | 0.157 | 0.15 | 0.037 / 0.058 | 0.67 | 0.007 | **1.12** | 0.750 | 0.442 |

- **ε=0 reproduces baseline exactly** (0.750 / 0.417 / 0.278) — control valid; `residual_norm_approach = 0.0` at
  every ε (the phase gate held; the residual never leaked into APPROACH).
- **Every ε>0 degrades, monotonically** in ft_dom, monitor_score, engagement, both-contact, anti_exploit.
- **Even ε=0.01 degrades** (ft_dom 0.708, monitor_pass 0.292) → the smallest trust region is already harmful; this
  is not "ε too large," it is a **harmful direction**.
- The tell-tale: **engagement duration collapses 4.17 → 1.12 steps/ep** as ε grows — the residual progressively
  *strips two-fingertip engagement*. (Interpretation flags: `eps0_reproduces_baseline=True`,
  `every_eps_gt0_degrades=True`, `eps001_preserves_or_improves=False`, `any_eps_improves=False`.)

## Part B — direct gradient-alignment probe (the mechanism), n=200 CONTACT/PUSH states

At fixed contact states, ∇ₐQ was computed on the frozen STRONG_PASS critic, and each candidate action was branched
**one physics step** via MuJoCo snapshot/restore:

| candidate | ΔQ | one-step coin progress | two-finger contact | arm-body contact |
|---|---:|---:|---:|---:|
| DAgger `a` | 0.000 | −0.0165 | **0.045** | 0.080 |
| **`a + ε∇Q`** | **+0.380** | −0.0157 | **0.010** | 0.095 |
| `a − ε∇Q` | −0.707 | −0.0169 | 0.025 | 0.070 |
| random | −0.439 | −0.0161 | 0.035 | 0.075 |

**KEY TEST: `GRADIENT_MONITOR_MISALIGNED = True`.** Moving along **+∇Q raises Q by +0.38 but cuts two-fingertip
contact 0.045 → 0.010 (≈4.5×) and raises arm-body contact 0.080 → 0.095.** The critic's local improvement direction
points *away* from fingertip engagement and *toward* body contact — the exact opposite of the monitor's
fingertip-dominant contract — even though the same critic *ranks* body-shove far below DAgger (STRONG_PASS margin
~12). (One-step coin progress is ≈−0.016 for all candidates — too noisy at one step to discriminate; the
discriminating signal is two-finger engagement, which Part A confirms collapses with ε.)

**Ranking is a global/ordinal property; the gradient is a local/directional one. Here they disagree: the critic
orders whole policies correctly and points the wrong way locally.** RL improvement rides on the gradient, so it
fails; the residual smoke, the CQL actor smoke, and this sweep are three views of the same misalignment.

## The complete RL arc (each attempt removed the prior failure, exposed the next)

1. **Baseline CTDE-TD3+BC** — critic **mis-ranking** (Q(exploit) > Q(dagger)).
2. **CQL actor smoke** — ranking fixed + held; **Q-scale runaway + off-manifold drift**.
3. **Residual + phase-gated** — ranking + runaway + drift all fixed; still degraded (on-manifold).
4. **ε-sweep + gradient probe (this)** — isolates + *directly measures* the root cause: **the critic gradient is
   monitor-misaligned**; no ε improves. **Root cause reached; lever closed.**

## Decision + next line (imitation / monitor-directed / gradient-free)

**Off-policy critic-gradient RL is not the lever for coin delivery.** The deployable policy remains **MLP+DAgger
(ft_dom 0.452 deployable / 0.75 this checkpoint)**. Options that do **not** rely on the critic gradient, in order
of proximity to what already works:

- **Better imitation** — more/better demonstrations, more DAgger rounds, a stronger demonstrator. Consistent with
  the whole project: the lever past a BC/DAgger ceiling has always been *imitation*, not off-policy RL.
- **Gradient-free / monitor-directed improvement over the bounded residual** — e.g. CEM/ES on the ε-bounded
  gated residual scored *directly by the monitor* (not the critic Q). The gradient probe is precisely the evidence
  that the improvement signal must be the monitor, not the value function. Reconsidering the monitor as an
  in-the-loop optimization target crosses the current "monitor stays external" contract — **do not** do it without
  explicit sign-off; this report is the justification to weigh it.
- **Accept the DAgger ceiling** for this task as a measured result and redirect effort.

## Files

- Harness `scratchpad/v2_epsilon_sweep_gradient_probe.py` (Part A sweep + Part B MuJoCo snapshot/restore probe);
  data `experiments/v2_epsilon_sweep/results.json`; figure `reports/figures/eps_sweep/eps_sweep_gradient_probe.png`;
  log `scratchpad/eps_sweep.log`. Reuses the tested `ResidualActor` / `train_residual` / `critic_repair` /
  `critic_benchmark` / `task_monitor` stack. CORE.YAML untouched, no new deps.

**Status:** the off-policy RL diagnostic thread is complete. Root cause = critic-gradient monitor-misalignment,
measured directly. RL frozen; imitation baseline stands; next branch is imitation or (with sign-off) monitor-
directed gradient-free improvement — **not** more actor-critic.
