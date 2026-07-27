# Humanoid balance — SAC-from-scratch under the Lyapunov reward + certificate gate

**Date:** 2026-07-27 (JST)
**Branch:** `research/humanoid-com-lyapunov` (worktree `hymeko_humanoid`)
**SIMULATION. Genuine RL (SAC from scratch, no certified baseline).**
**Verdict: `SAC_SURVIVES_PARTIALLY_BUT_IS_NOT_LYAPUNOV_STABLE` — 0.589 upright fraction, 0/12 Lyapunov certificate.**

---

## What was run

SAC from scratch on the floating-base humanoid balance task, driven by the COM
Lyapunov energy as the per-step reward and gated (eval-only) by the **reward-independent**
`lyapunov_certificate`. This is the `SAC-from-scratch` path from the readiness report
(`2026-07-27-humanoid-balance-sac.md`), chosen because **no certified hand-tuned or
LQR baseline exists** (both failed — see that report and `…-humanoid-lqr-attempt.md`),
so this is genuine RL (coin R14–R60 regime), not residual-over-scaffold.

- **Env** `scenarios/humanoid/balance_env.py` — freejoint base + floor, **unactuated
  base** (12 joint torques), obs 35-D (uprightness, pitch, pelvis-height error, base
  twist, joint q/v, COM offset + COM x-velocity). Action = normalised joint torque;
  applied as **gravity-comp feedforward (`qfrc_bias`) + SAC residual** (`τ = a·50 + bias`).
- **Reward** `1 − 2·V − 1e-3·‖a‖²`, V = COM Lyapunov energy. Alive-bonus + energy
  penalty + control cost. Fall (uprightness ≤ 0.6 or pelvis_z ≤ 0.55) terminates.
- **Gate (never in the reward)** `evaluate_lyapunov`: V ≥ 0, near-monotone descent
  (dV ≤ 5e-3 on ≥ 90 % of steps), converged (V_final ≤ 0.05), bounded (V_max ≤ V₀+tol).
- **SAC** MLP actor/critic (hidden 128), 150 000 steps, auto-α, seed 0.

## Result

| metric | value |
|---|---|
| final upright fraction (12 held-out seeds, greedy) | **0.589** (~290/500 steps) |
| final Lyapunov certificate pass rate | **0.0 / 12** |
| eval curve (every 15 k steps) | 0.30 · 0.51 · 0.52 · 0.44 · 0.63 · **0.93** · 0.69 · 0.67 · 0.68 · 0.59 |

**Per-condition breakdown (final policy, 3 seeds):** V₀ ≈ **0.045** (reset is near-standing,
*already below* the 0.05 convergence threshold), but the policy lets V **climb** to
V_max ≈ **1.64** and end at V_final ≈ **0.6–0.73**. It fails all three non-trivial
conditions — `descent_fraction ≈ 0.76 < 0.9`, `converged = False`, `bounded = False`
(only `nonnegative` holds, trivially).

## Honest interpretation

SAC found a **"drift-while-upright"** local optimum: it keeps the pelvis nominally
upright long enough (~59 % of the episode) to harvest the `+1` alive-bonus, while the
**COM energy grows** — it trades a rising V for staying alive rather than driving V to
rest. It does **not** learn a stabilizing (settling) policy.

The result's value is what the **reward-independent certificate** proves:

- The gate is **achievable, not impossible** — V₀ ≈ 0.045 already satisfies `converged`,
  a still-hold near the start certifies, and AIBO passed the same generic certificate.
  So `0/12` is a **genuine policy failure**, not a mis-calibrated gate.
- The certificate **discriminates survival from stability**: `upright fraction 0.589`
  looks like partial success, but the certificate correctly refuses to certify a
  drifting COM. This is exactly the campaign discipline — *the metric the RL run
  optimizes (survival + energy penalty) is not the same as the certificate it must
  satisfy (Lyapunov descent to rest)*, and only the latter is trusted.

This is a **partial positive on learning, negative on the objective**: SAC beats the
hand-tuned/gravity-comp controllers on survival (they tip in ~1.2–1.4 s ≈ 240–280
steps; SAC holds ~290) but, like them, **fails Lyapunov**. Balance-to-rest for this
underactuated floating humanoid is not solved.

## Why (mechanism, not excuse)

The reward `1 − 2V` is dominated by the `+1` alive term when V is small; once upright,
the marginal cost of a slowly-growing V (2·ΔV per step) is less than the cost of a
control effort large enough to actively *descend* V against the unactuated base mode.
The policy therefore prefers cheap survival over expensive stabilization. Candidate
fixes (future work, not run): reward `−dV/dt` (descent, not level), a larger fall/energy
penalty, terminate on `V > V₀ + margin` (make drift a failure, not just a cost), or a
longer horizon with curriculum. None is claimed here.

## Files touched (all NEW, scenario-side, non-core)

```
scenarios/humanoid/balance_env.py            115 LOC  (HumanoidBalanceEnv — SAC gym env)
scenarios/humanoid/run_humanoid_sac.py        97 LOC  (SAC harness + Lyapunov eval)
tests/test_humanoid_balance_env.py            96 LOC  (6 integration tests)
reports/2026-07-27-humanoid-sac/{sac_gates.json,sac_train.log,humanoid_sac_actor.pt}
reports/2026-07-27-humanoid-sac.md            (this report)
```

## Tests

`ruff` clean. **16/16 humanoid tests pass** (0.72 s): 5 Lyapunov unit + 5 CIP
conformance + **6 new** balance-env integration (gym contract, deterministic reset,
fall-termination reachable, gravity-comp-alone-does-not-balance invariant, SAC eval
helper end-to-end). The new `balance_env.py` and `run_humanoid_sac._eval_balance` are
covered; the certificate's discrimination is regression-tested against the drift trace.

## Performance

- Wall time ≈ **7 min** (150 k steps at ~365 steps/s + 10 evals + final 12-seed eval).
- **Peak RSS ≈ 0.27 GB** — far under the 16 GB cap (§4). MLP + 100 k-transition buffer.
- Single-shot wall (diagnostic only, not a benchmark per §10).

## Provenance

- Git SHA `26306271` (working tree: the 4 new files above, untracked pre-commit).
- Seed 0; eval seeds 2000–2005 (curve), 3000–3011 (final); deterministic reset verified.
- Host: Apple-Silicon Mac, torch CPU. MuJoCo model emitted by `target/release/hymeko`.
- Env: `hymeko_rl` native venv (`…/hymeko_framework_rust/.venv`).

## CORE.YAML / protocol notes

- **CORE.YAML items touched: none.** Shared CIP core (`hymeko_control/`) unchanged; the
  env reuses the promoted `HumanoidCOMLyapunov` / `lyapunov_certificate` read-only.
- **§2 plan artifacts:** this is an exploratory RL probe within the humanoid-Lyapunov
  arc (the user directed "let reality override the plan"); the design/pre-registration
  lives in `2026-07-27-humanoid-balance-sac.md` (task, reward, gate, regime — written
  *before* this run). The formal 4-format plan was **not** produced for this probe; I am
  flagging that as a deliberate deviation rather than back-dating artifacts after the run.
- **§6.5 anti-patterns:** none introduced (single env class, single harness, config via
  `SACConfig`; no Cartesian dump, no globals, no string-typed modes).

## Bottom line

SAC-from-scratch under the Lyapunov reward **partially survives (0.589) but is not
Lyapunov-stable (0/12)** — a drift-while-upright local optimum. The reward-independent
certificate did its job: it separates survival from stability and refuses the drifting
policy. No balance baseline is claimed. The honest routes forward are a descent-shaped
reward (`−dV/dt` + drift-as-failure) or the contact-consistent-equilibrium LQR that
would yield a *certified* scaffold for residual RL — both deferred.
