# R8 — corrected RL-authorization gate + tip-referenced residual scaffold: STAGE-0 CONTRACT

**Created-at:** 2026-07-27 14:10 JST · **Branch:** `recovery/coin-r8-tip-referenced-residual-rl` (from R7 tip `81efd831`).
**Author of the gate correction:** the user (Dr. Hajdu), on the record; this contract freezes that decision. **CORE.YAML:
none.**

## 0. Why the old gate was circular (the correction)

The prior gate — *RL is authorised only after a cradle-agnostic deterministic update-zero controller reaches K6 4/4* — is
**circular**. R1→R7 established (each for a distinct, measured reason) that the missing element is exactly the
**soft-frictional, cradle-specific, history-dependent torque→tip→coin map** that learning is meant to acquire; if a fixed
controller already delivered it 4/4, RL would have almost nothing left to do. The old gate demanded *learning performance*
as a *precondition for learning*.

**Correction (frozen):** a **safe deterministic scaffold IS required, but deterministic 4/4 is NOT a justified prerequisite
for RL.** Relax the baseline's *success* requirement; keep every *physics / safety / honesty* constraint hard.

## 1. Kept HARD (nothing relaxed — the integrity constraints)

- no teleport / coin-state edit; no hidden force; no teacher fallback at deploy;
- exact Bellman action + provenance (`θ_0`/residual is the label, never θ_exec/post-search);
- reward and the external K6 certificate are SEPARATE (no reward-hacking the metric);
- slew / torque / contact / motion limits unchanged; frozen K6 unchanged;
- **held-out s4/s7 never enter training, replay, normalisation, critic/actor fit, seed/hyperparameter selection, or early
  stopping** — one-shot frozen eval only;
- **update-zero (δu = 0) reproduces the scaffold bit-/tolerance-identically** (so RL's contribution is measurable);
- **oracle stays 4/4** (S0 — physical solvability already established).

## 2. The corrected gate (SAFE · BOUNDED · CAUSAL · LEARNABLE)

- **S0 — physical solvability:** oracle / search / teacher = 4/4. **HAVE.**
- **S1 — safe scaffold:** the tip-referenced torque-target regulator stays within torque/slew limits, is numerically stable,
  does not drive tip velocity unbounded, anti-windup works, the stop injects no further energy, no illegal contact/state
  edit. **K6 4/4 NOT required.**
- **S2 — update-zero identity:** δu = 0 reproduces the scaffold to bit/tolerance.
- **S3 — learnability signal (dev):** the residual has a measurable trajectory effect; +δ and −δ give different returns;
  safe candidates better than the baseline exist; a (difference-)critic can at least partially rank them.
- **S4 — reward-driven RL (dev panel):** SAC and TD3 identical (replay, actor arch, seeds, budget, reward, residual bounds,
  certificate). First goal: **better dev K6 / return than the baseline with no safety regression** — not immediately 4/4.
- **S5 — frozen held-out eval:** s4/s7 one-shot; no re-tuning, no threshold edits.

RL (bounded residual) is **AUTHORISED** once S1–S3 pass. The final *claim* may still require 4/4; **development** is judged
on multi-seed K6 rate / min-dtz / dwell / peak-velocity / sign-reversal / safety-violations / return median-IQR — not the
brittle binary 4/4 (a single search seed moved s3 from 6.7 mm to 37.1 mm; §R7 audit).

## 3. What RL learns (a bounded residual over a SAFE base — not raw torque)

Base = the **tip-referenced velocity/torque-target controller** (regulate the *directly-actuated tip/joint* velocity; the
coin follows through the grip — R7 §4.1) + anti-windup + saturating non-reversing stop + the R6 release **certificate as a
SHIELD** (it decides *when* release is safe; it does NOT carry the coin to goal). The actor adds a small clipped residual:

    u_t = u_safe_base(s_t) + clip(δu_π(h_t), −Δ_max, +Δ_max)

over: δ tip-velocity target, δ torque target, δ squeeze target, δ stop-timing, δ release-readiness. The **history** `h_t` is
load-bearing (tip- & coin-velocity history, torque accumulation, contact force/slip, relative tip–coin motion,
squeeze/preload, remaining distance) — precisely the nonlinear soft-contact map R7 showed a fixed law cannot capture.

## 4. This session's scope (gated; RL training NOT started here)

**Did:** froze this corrected gate. **Building next in-session:** the **tip-referenced scaffold** + the **S1 safety
de-risk** (does regulating the joint/tip velocity keep the coin bounded — no blow-up / reversal — where R7's coin-velocity
servo could not). **Then (next phase):** S2 update-zero adapter, S3 learnability probe, S4 matched SAC/TD3 on dev, S5 frozen
held-out. RL training begins only after S1–S3 pass. Every §1 constraint holds throughout.

## 5. Honest verdict carried in

R1→R7 do **not** prove all deterministic control hopeless; they prove a *safe deterministic scaffold is required and a
deterministic 4/4 is not a justified RL prerequisite*. Under §1's intact safeguards, bounded residual RL over the
tip-referenced scaffold is the scientifically-justified next axis.
