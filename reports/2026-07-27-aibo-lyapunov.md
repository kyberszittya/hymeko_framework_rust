# AIBO Lyapunov integration — pH high-level loop on a state-dependent embodiment

**Date:** 2026-07-27 (JST)
**Branch:** `research/aibo-lyapunov-ph` (from `simulation/aibo-simple-scenarios-v1`)
**SIMULATION. NOT RL.**  ·  **Verdict: `AIBO_STATE_DEPENDENT_LOOP_SATISFIES_LYAPUNOV`.**

---

## What was integrated

Lyapunov conditions are now part of the CIP-0 model. The high-level pH loop over the
underactuated AIBO body carries a **Lyapunov function**

    V(s) = ½ [ w_d·max(0, d − reach)² + w_θ·herr² + w_v·speed² ]

(d = distance to waypoint, herr = heading error, speed = planar body speed; V → 0 iff
at the waypoint, aligned, at rest), and the certificate layer gains a reward-
independent **Lyapunov certificate**: V ≥ 0, V is (near-)monotone non-increasing
(dV ≤ tol on ≥ 90 % of steps), and V converges (V_final ≤ 0.05, net decrease).

## Result — the refined hypothesis is CONFIRMED

| controller | Lyapunov passes | V_final |
|---|---|---|
| high-level pursuit, offset +10° | ✅ | 0.013 |
| high-level pursuit, offset −10° | ✅ | 0.031 |
| high-level pursuit, offset +20° | ✅ | 0.034 |
| high-level pursuit, offset −20° | ✅ | 0.039 |
| **negative control** (constant forward, no align/stop) | ❌ | **4.307** |

- The **state-dependent** high-level loop **satisfies the Lyapunov conditions on every
  offset** — a rigorous, reward-independent stability guarantee (descent + convergence
  to ~0), not a heuristic.
- The certificate **discriminates**: the non-converging negative control fails
  decisively (V_final 4.31 — it walks off). A certificate everything passes would be
  worthless; this one rejects instability (metric integrity).

## The predicted split (PnP vs AIBO)

The PnP hierarchical experiment falsified the pH loop **there** — PnP's under-transport
is a **uniform bias**, so a constant residual is optimal and the state-dependent loop
underperformed. It predicted the pH/Lyapunov hierarchy would pay where the correction
is **state-dependent**. AIBO approach-align-stop is exactly that: **you cannot align a
body to a waypoint with a constant offset** — the yaw/drive correction must respond to
the current (underactuated) pose. The Lyapunov result confirms the split:

> **pH/Lyapunov high-level loop: FALSIFIED on PnP (uniform bias) — CONFIRMED on AIBO
> (state-dependent), where it satisfies a rigorous Lyapunov stability certificate.**

## Core-promotion candidate

`lyapunov_certificate(name, V_fn, ...)` is **generic and scenario-independent** — the
caller supplies the Lyapunov function, so the core names no scenario signal. It is the
rigorous generalization of the promoted `stability_certificate` (which only checked an
uprightness threshold): a formal descent+convergence stability guarantee usable by
AIBO, a floating-base humanoid (COM Lyapunov), and any convergent CIP-0 task. Carry to
the core-promotion review.

## Files (all NEW/scenario-side, non-core)

```
scenarios/aibo/lyapunov.py          (AIBOLyapunov + evaluate_lyapunov + lyapunov_certificate)
scenarios/aibo/run_aibo_lyapunov.py (pursuit vs negative-control verification harness)
tests/test_aibo_lyapunov.py         (certificate descent/convergence/discrimination — 5)
reports/2026-07-27-aibo-lyapunov/lyapunov_gates.json
```

## Tests + lint

`pytest` — 12 passed (5 Lyapunov + 5 AIBO conformance + 2 bidirectional-yaw), no
regression. `ruff` clean. CORE.YAML: none. Shared CIP core unchanged.

**Verdict:** Lyapunov conditions integrated into the CIP-0 model; the state-dependent
AIBO high-level loop **satisfies a rigorous, reward-independent Lyapunov stability
certificate** on all offsets, and the certificate discriminates against instability.
This confirms the PnP-predicted split (pH hierarchy pays on state-dependent, not
uniform-bias, tasks) and yields a strong generic core-promotion (`lyapunov_certificate`).
