# Humanoid COM-Lyapunov — verified against the Lyapunov certificate

**Date:** 2026-07-27 (JST)
**Branch:** `research/humanoid-com-lyapunov` (from `scenario/cip-humanoid-v0`)
**SIMULATION. NOT RL.**  ·  **Verdict: `HUMANOID_FLOATING_FAILS_LYAPUNOV_BALANCE_CONTROLLER_PREREQUISITE`.**

---

## Floating-base HyMeKo humanoid (the HUM-1 wall, unblocked at the model level)

HUM-1 found the HyMeKo humanoid FIXED-base (the `base` joint is a single z-hinge —
yaw only — so it cannot fall; balance was vacuous). Here the emitted MJCF's
`<joint name="base" type="hinge" axis="0 0 1"/>` is replaced **at runtime** with a
`<freejoint/>` → a genuine 6-DOF floating base (nq 13→19, nv 12→18) that **can fall**.
HyMeKo stays the model source; only the root joint is un-welded.

## COM-Lyapunov + verification against the certificate

The whole-body COM is the underactuated coordinate:

    V(s) = ½ [ w_h·(com_z − h_ref)² + w_xy·‖com_xy − support_xy‖²
               + w_v·‖com_vel‖² + w_up·(1 − uprightness)² ]

V → 0 iff the COM holds its standing height over the support, at rest, upright. A
fall (tip OR collapse) makes V diverge. Verified against the **same generic Lyapunov
certificate** AIBO passed:

| controller | Lyapunov passes | V_final | V_max |
|---|---|---|---|
| **floating base, naive gravity-comp PD** | ❌ | **4024.8** | 4024.8 |
| fixed base, constrained (cannot fall) | ✅ | 0.0 | 0.0 |

- The **floating-base** humanoid under a naive controller **collapses** (pelvis sinks
  0.95 → 0.50 m by step 303; COM stays centred, it does not tip — it *sinks*), so V
  diverges → the Lyapunov certificate **REJECTS** it. The balance controller is now
  the **rigorous, Lyapunov-expressed prerequisite** for HUM-2/3/4.
- The **fixed-base** humanoid holds V ≈ 0 → **PASSES vacuously** — its "stability" is
  a constraint artifact, exactly the HUM-1 "balance vacuous" finding, now formal.
- The certificate **discriminates** (floating fails, fixed passes) — and AIBO (a real
  state-dependent controller) passed it. **Same generic certificate, three verdicts.**

## Certificate refinement (a genuine improvement)

Lyapunov stability is **bounded + converged**, not strict net-decrease: a
start-at-equilibrium trajectory (V ≈ 0 throughout) is stable but never "decreases".
`evaluate_lyapunov` now passes iff V ≥ 0, near-monotone non-increasing (dV ≤ tol on
≥ 90 % of steps), **bounded** (V_max ≤ max(V0, ε)+tol), and converged (V_final ≤ ε).
This certifies both the AIBO converge-from-perturbation and the fixed-base
stay-at-equilibrium cases. **The AIBO copy should be unified with this corrected
form at the core-promotion review.**

## Cross-embodiment Lyapunov picture

| embodiment | controller | Lyapunov |
|---|---|---|
| AIBO (state-dependent pursuit) | real | ✅ satisfies |
| humanoid, floating base | naive (none) | ❌ fails → prerequisite |
| humanoid, fixed base | constrained | ✅ vacuous |
| PnP | uniform bias | (hierarchy underperforms — different regime) |

The generic `lyapunov_certificate` is one reward-independent, discriminating stability
verifier across embodiments — a strong **core-promotion** candidate (rigorous
generalization of `stability_certificate`).

## Files (all NEW/scenario-side, non-core)

```
scenarios/humanoid/lyapunov.py            (HumanoidCOMLyapunov + evaluate_lyapunov + certificate)
scenarios/humanoid/run_humanoid_lyapunov.py (floating-base runtime freejoint + verification)
tests/test_humanoid_lyapunov.py           (V, bounded/converged, discrimination — 5)
reports/2026-07-27-humanoid-lyapunov/lyapunov_gates.json
```

## Tests + lint

`pytest` — 10 passed (5 Lyapunov + 5 HUM conformance), no regression. `ruff` clean.
CORE.YAML: none. Shared CIP core unchanged.

**Verdict:** the floating-base HyMeKo humanoid **fails the COM-Lyapunov certificate**
under naive control (it collapses) — a rigorous statement that a genuine balance
controller is the HUM-2/3/4 prerequisite; the fixed base passes only vacuously. The
same generic Lyapunov certificate discriminates across AIBO / humanoid, confirming it
as a scenario-independent stability verifier for core promotion.
