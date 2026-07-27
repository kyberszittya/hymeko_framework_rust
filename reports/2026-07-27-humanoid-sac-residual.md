# Humanoid balance — certified PD scaffold + residual SAC extends the certified envelope

**Date:** 2026-07-27 (JST)
**Branch:** `research/humanoid-com-lyapunov` (worktree `hymeko_humanoid`)
**SIMULATION. Genuine RL (bounded residual over a certified scaffold, coin-R8 regime).**
**Verdict: `RESIDUAL_EXTENDS_CERTIFIED_ENVELOPE` — PD-hold certifies 0/12 on the hard envelope, residual SAC certifies 12/12 (Δ +1.0).**

---

## How we got here (measurement over guessing)

The first SAC-from-scratch run (`…-humanoid-sac.md`, commit edae7b0e) survived partially
(0.589) but was **not Lyapunov-stable (0/12)**. Asked whether the observation or action
model needed retuning, I **measured** the bottleneck rather than guessed and found it was
neither primarily — it was two bugs plus a secondary authority limit:

1. **`h_ref` mis-referenced (highest leverage).** The reward/Lyapunov reference height was
   0.818 (a pelvis-top height); the **measured standing COM is ~0.645**. The old reward
   asked the humanoid to lift its COM 14 cm above where it stands — an ill-posed target
   that put V₀ exactly on the 0.05 convergence cliff. Fixed: `h_ref = 0.645`.
2. **Action authority (the retune the user flagged).** SAC saturated the ±50 N·m torque on
   40 % of steps. Replaced the torque action with a **position-servo (PD-to-target)**:
   `τ = clip(kp·(q0 + a·δ − q) − kv·q̇ + qfrc_bias, ±τ_max)`, `kp=60, kv=10` (the stable
   ceiling for this model's Euler @ 1 ms integrator — `kp≳150` diverges, measured).
3. **Observation** was adequate (sagittal signals present); not the bottleneck.

## Result 1 — a *certified* balance baseline (was the whole arc's missing piece)

With those two fixes, **a = 0 is the PD-hold-`q0` controller**, and it is a **certified
balance controller**: it passes the **unchanged** Lyapunov certificate.

| pitch-rate perturbation | 0.0 | 0.1 | 0.2 | 0.3 | 0.6 |
|---|---|---|---|---|---|
| PD-hold certified? | ✅ | ✅ | ✅ | ✅ | ❌ (V_max 0.10 overshoot) |
| V_final | 0.0002 | 0.0013 | 0.0024 | 0.0044 | 0.0116 |

The certificate is **respected, not loosened** — it passes for small perturbations
(V converges to rest, bounded) and fails at 0.6 on `bounded` (the recovery transient
overshoots V_max > 0.055). This is a genuine robustness boundary. The LQR attempt
(`…-humanoid-lqr-attempt.md`) failed to produce this; **position-servo energy-shaping
(PD-to-nominal-pose) succeeds** where model-based LQR stalled on the contact-consistent
equilibrium — and it fits the port-Hamiltonian framing (potential shaping to a nominal
configuration).

## Result 2 — residual SAC *extends* the certified envelope (coin-R8, cross-embodiment)

On a **harder envelope** (pitch-rate 0.4–0.8, both signs) the certified scaffold no longer
certifies (it survives but overshoots). A **bounded residual** trained over it (the coin-R8
pattern) extends the *certified* region:

| policy | envelope | certified (held-out 12 seeds) | upright fraction |
|---|---|---|---|
| PD-hold scaffold (a = 0) | 0.4–0.8 | **0 / 12** | 1.0 (survives, overshoots) |
| SAC residual (best-val ckpt) | 0.4–0.8 | **12 / 12** | 1.0 |
| SAC residual (last ckpt) | 0.4–0.8 | 12 / 12 | 1.0 |

**Δ certified rate = +1.0** on held-out seeds → `RESIDUAL_EXTENDS_CERTIFIED_ENVELOPE`.
Baseline and residual are evaluated on **identical per-seed perturbations**, so the delta
is the pure RL value-add. This confirms the campaign's coin-R8 finding — *a bounded residual
over a working (here, certified) scaffold generalizes* — now on the floating humanoid.

## Honest process note — the first residual run was unstable (fixed, documented)

The first residual run used SAC's **AUTO entropy** (classic α-tuning). It **worked then
collapsed**: val certified curve `1.0 → 0.33 → 0 → … → 0.5`, α collapsed to 5e-4 → the
near-deterministic policy oscillated. The naive last checkpoint still beat the scaffold
(test 0.333 > 0.0, `auto_naive_gates.json`), but it was not reliable. Two measured fixes:

- **ANNEAL entropy** (`init 0.1 → 0.005` over 60 % of steps — the campaign's corrected
  schedule): val certified curve became `1.0` on **14/15** evals (only the anneal-endpoint
  90 k dipped to 0.125, then recovered).
- **Best-validation-checkpoint** (val seeds select, test seeds reported once — a clean
  val/test split; matches the coin arc's `bestval` practice): a safety belt against any
  residual instability.

Root cause (interesting, honest): the **reward (−2V, an integral) and the certificate
(peak V ≤ 0.055 + convergence) are subtly misaligned** — a high-reward policy can still
spike V on a seed. ANNEAL + best-ckpt closes the gap operationally; a peak-V-aware reward
is the principled future fix.

## Robustness frontier (measured, push-to-fall)

Escalating the pitch-rate kick maps the honest boundary (seed 0, both signs sampled):

| kick | PD-hold | SAC residual |
|---|---|---|
| 0.8–1.6 | survives, not certified | survives, **CERTIFIED** |
| 2.0–2.6 | survives, not certified | survives, not certified |
| 3.2 | survives (500) | falls (496) |
| ≥ 4.0 | **FALLS** (~200 steps ≈ 0.2 s) | **FALLS** (~200 steps) |

Two honest limits: (1) the residual extends the **certified** envelope to ~1.6 (from
~0.3), generalizing well past its 0.4–0.8 training band; (2) beyond ~4.0 the humanoid
**tips over** — a hard physical limit, and far out-of-distribution (3.2) the residual can
even slightly *reduce* raw survival (it optimizes certified stability, not survival). The
`humanoid_balance_frontier.mp4` video shows this: residual CERTIFIED (1.6) → SURVIVES with
a big sway (2.6) → **FELL** (4.5, a genuine tip-over).

## Files touched (all scenario-side, non-core)

```
scenarios/humanoid/balance_env.py        rewrite  161 LOC  (position-servo, h_ref=0.645, BalanceConfig)
scenarios/humanoid/run_humanoid_sac.py   rewrite  137 LOC  (residual + PD baseline + best-val ckpt + ANNEAL)
scenarios/humanoid/lyapunov.py           1-line   h_ref default 0.818 -> 0.645 (measured correction)
tests/test_humanoid_balance_env.py       rewrite   6 tests (certified-scaffold + envelope-boundary invariants)
tests/test_humanoid_lyapunov.py          2-line   equilibrium fixtures 0.818 -> 0.645
reports/2026-07-27-humanoid-sac-residual/{sac_residual_gates.json,auto_naive_gates.json,
    sac_train.log,humanoid_sac_residual_best.pt,humanoid_sac_residual_actor.pt}
```

## Tests

`ruff` clean. **16/16 humanoid tests pass** (0.99 s): 5 Lyapunov unit + 5 CIP conformance +
6 balance-env integration. New invariants regression-tested: `a=0` passes the certificate on
the nominal envelope (certified scaffold); at perturbation 1.0 the scaffold **survives but
fails** the certificate (survival ≠ stability); the fall path terminates; determinism holds.

## Performance

- Wall ≈ 9 min (150 k steps ~400 steps/s + 15 val evals × 8 seeds + final test evals).
- Peak RSS ≈ **0.30 GB** — far under the 16 GB cap (§4).

## Provenance

- Parent SHA `edae7b0e` (new work uncommitted at write time). Seeds: train 0; val 2000–2007
  (checkpoint selection); test 3000–3011 (reported once, held out). Deterministic reset.
- Host: Apple-Silicon Mac, torch CPU. MuJoCo model emitted by `target/release/hymeko`;
  timestep 1 ms, Euler integrator.
- Checkpoint `humanoid_sac_residual_best.pt` (best-val); `…_actor.pt` (last).

## CORE.YAML / protocol notes

- **CORE.YAML items touched: none.** Shared CIP core unchanged; `hymeko_rl.train.sac` used
  read-only (best-ckpt via an `eval_fn` closure — a closure capture, not global state, §6.5).
- The Lyapunov certificate was **not modified** — it passes/fails on its own terms; only the
  scenario's physical `h_ref` reference was corrected (measured).
- **§2 plan artifacts:** measurement-driven retune within the humanoid-Lyapunov arc (the user
  directed the scope: h_ref + reset + action authority). The design crystallized through the
  documented diagnostics above; the formal 4-format plan was not produced for this probe —
  flagged, not back-dated.
- **§6.5 anti-patterns:** none. `BalanceConfig` dataclass (no arg explosion), single env class,
  single harness, enum-typed `AlphaMode`.

## Bottom line

The floating humanoid **is now balanced to certificate two ways**: (1) a **certified
PD-to-nominal scaffold** (energy shaping) for perturbations ≤ ~0.3 — the certified baseline
the LQR attempt could not deliver; (2) a **residual SAC** that extends the certified envelope
to 12/12 on the harder 0.4–0.8 band where the scaffold overshoots. The SAC-from-scratch
negative was an artifact of a mis-set reference height and a weak action model, both fixed by
measurement. The certificate stayed fixed throughout and did its job — separating survival
from stability at every step.
