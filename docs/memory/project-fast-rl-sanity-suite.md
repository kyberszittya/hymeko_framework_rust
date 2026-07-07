---
name: project-fast-rl-sanity-suite
description: Fast physics-free RL sanity testbeds (bandit / grid / hex / collab) — architecture sanity + deploy-latency in SECONDS instead of hour-long MuJoCo; built 2026-06-29
metadata: 
  node_type: memory
  type: project
  originSessionId: 913c706b-9719-45ca-aa85-e9cfbef27d5d
---

2026-06-29 (Hajdu asked "do we have small fast RL scenarios like k-bandit / gridworld for architecture sanity?" —
calibrated, there was a real gap: every RL env was MuJoCo, minutes-to-hours/verdict). Built a fast, physics-free,
in-process sanity suite running the REAL `build_policy` backbones (mlp/hsikan/sa_hsikan/mixture, cr vs cr_cheby):

- `hymeko_rl/sanity_rl.py` — **ContextualBandit** (1-step, signed ring; REINFORCE). `--target flat|structural`.
- `hymeko_rl/sanity_worlds.py` — **LatticeNav** grid(4-nbr)/hex(6-nbr) multi-step navigation + **CollabBandit**
  (2-agent sum-to-target = CTDE). `--world grid|hex|collab`. hex = grid-cell/path-integration substrate.
- 13 tests, ruff+mypy clean. Report `reports/2026-06-29-fast-rl-sanity-suite.md`.

**Three findings it already produced (in seconds):**
1. **Linear graph targets do NOT discriminate HSiKAN from MLP** — an MLP represents any linear graph op (A·x)
   trivially, so flat + structural-linear both tie (~-0.25). Reproduces the standing tie ([[project-galambos-hsikan-tie-rootcause]])
   fast. The genuine ACCURACY discriminator must be NONLINEAR (cycle parity / Z2 holonomy) = the holonomy-toy P1
   ([[project-hsikan-loses-possible-bug]] decisive-next; backlog 2026-06-29-holonomy-discriminator-toy).
2. **Deploy-latency (B=1) exposes the recent enhancements** ([[project-hsikan-launchbound-alternatives]]): SA-HSiKAN
   B^L collapse = ~5x faster deploy + 10x fewer params (0.91ms, 775 params) vs vanilla HSiKAN (4.67ms, 8071);
   CR-Chebyshev deploy path = ~22% faster B=1 + better accuracy (wired `set_deploy_mode` in signed_kan/splines.py,
   train-CR/deploy-Chebyshev; signed_kan is NOT core).
3. Multi-step nav re-demonstrates the launch-bound (HSiKAN ~3x slower over the rollout).

**HOLONOMY + SPIKE-ROTOR VERIFIED (2026-06-29, the decisive test — answers [[project-hsikan-loses-possible-bug]]'s
decisive-next + supports [[project-gauge-holonomy-signed-hsikan]] C1/C3).** Built `hymeko_rl/holonomy_probe.py`
(T1 parity: per-sample signed ring, label = Z2 cycle holonomy = product of edge signs; confound-guarded). 4-arm
@ ring-16: **transport(rotor) 1.00 ; additive-B^N(=HSiKAN mechanism) 0.50 ; MLP 0.50 ; linear 0.49 (confound
passes)**. KEY: **holonomy is MULTIPLICATIVE (parallel transport); HSiKAN's ADDITIVE signed-walk (B^N sum) reads
it at CHANCE — only the rotor does.** This EXPLAINS the HSiKAN-ties-MLP record (additive aggregation isn't a
holonomy reader) and is why holonomy+spike-rotor go together. **Spike-rotor verified** by running the existing
`spike_probe` (SO(3) non-abelian): spike_gated 0.0 vs order_blind 0.059 at θ=0.8 — the spike is needed to select
the time-ordered walk for non-abelian holonomy. So: rotor reads abelian holonomy, spike handles non-abelian order.

Also this session: HSiKAN+MLP **mixture-of-experts backbone** (`policy.MixtureBackbone`, gated, registered
"mixture"). NEXT: grid-cell population-structure on hex,
nav GIFs.
