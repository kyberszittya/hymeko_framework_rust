# R9 resolution — the handoff-reset made explicit: R2 IS the first learned K6 under an explicit hybrid handoff-reset

**2026-07-28 · branch `recovery/coin-r9-causal-residual-delivery` · immutable base `85c5eca6` · dev s1 (14250) · s4/s7 untouched · f1–f4 SEALED · NO RL · no tag moved**

## Summary — the frozen entry was H1, and it's legitimate

The prior boundary showed the frozen KINETIC entry is the natural handoff **advanced by one servo KINETIC step** that existed only
in offline entry-creation. Rather than reorder the capture to cherry-pick K6, this step makes the reset map **explicit** as a
first-class **online** controller mode and audits it end-to-end. Two boundary contracts (never mixed in one run):

- **H0 — DIRECT_HANDOFF**: APPROACH → guard crossing → the KINETIC policy acts immediately (the plain controller; the audit's E1).
- **H1 — EXPLICIT_HANDOFF_RESET**: APPROACH → guard crossing → **one explicit frozen transition-servo step** (mode `HANDOFF_RESET`,
  its own trace, clear pre/post semantics, clone does not act) → the KINETIC policy.

**Verdict: `R2_IS_FIRST_LEARNED_K6_UNDER_EXPLICIT_HANDOFF_RESET`.** All three mandated gates pass. Under H1, run entirely online
from the cradle (no snapshot injection), **clone + R2 delivers strict K6 end-to-end** — and the online post-reset state is
**bit-exactly** the offline frozen entry. The frozen-entry K6 is therefore real; it was only ever mis-labelled as an offline-only
interface. The corrected claims:

- **R2 (clone + R2) is the first learned teacher-free s1 K6 transport policy — under the explicit HANDOFF_RESET contract.**
- **R3-C's authority-unlock contribution is dwell / settle-margin refinement, not delivery creation** (FULL dwell > clone+R2 dwell
  under both interfaces).
- **The handoff-reset step is load-bearing, not cosmetic**: H0 (direct) does *not* deliver (39.58 mm); the one servo step is what
  sets up the deliverable KINETIC state.

## Gates (all from the canonical s1 cradle, teacher-free, no interruption)

| gate | result |
|---|---|
| `HANDOFF_RESET_EXPLICIT_PASS` | ✅ exactly one `HANDOFF_RESET` event, before the first `KINETIC_CLONE`, with its own trace |
| `ONLINE_FROZEN_ENTRY_EQUIVALENCE_PASS` | ✅ online post-reset ≡ frozen entry — **dtz Δ0.0, qpos Δ0.0, prev_tau Δ0.0** (bit-exact) |
| `END_TO_END_R2_K6_UNDER_EXPLICIT_RESET_PASS` | ✅ clone+R2 delivers strict K6 end-to-end, safe |

## End-to-end runs (min_dtz / K6 / dwell)

| controller | interface | min_dtz | K6 | dwell |
|---|---|---|---|---|
| clone + R2 | H0 DIRECT_HANDOFF | 39.58 mm | ❌ | 0 |
| clone + R2 | **H1 EXPLICIT_HANDOFF_RESET** | **16.86 mm** | **✅** | 13 |
| FULL (clone+R2+β·exp) | H0 DIRECT_HANDOFF | 38.19 mm | ❌ | 0 |
| FULL | **H1 EXPLICIT_HANDOFF_RESET** | **19.14 mm** | **✅** | 15 |
| clone + R2 | E0 frozen-entry snapshot (reference) | 16.86 mm | ✅ | 17 |

H1 clone+R2 reproduces the frozen-entry delivery (min_dtz 16.86 mm, identical) as **one continuous end-to-end controller**. The
dwell (13/15 online vs 17/27.5 offline) is a horizon-budget effect — the online chain spends part of the fixed horizon on APPROACH —
not a trajectory difference; the closest approach is identical. Under both interfaces FULL's dwell exceeds clone+R2's, so the
expansion's dwell-refinement role holds under the explicit reset.

## Which of the three stories

This is the user's **case 1**: clone+R2 achieves end-to-end K6 under H1. Consequently:

- **Corrected historical claim:** R2 was the first learned K6 transport policy — with an explicit hybrid handoff-reset. It was
  never delivering under the DIRECT handoff (H0), and the earlier "R2 saturates ~36 mm" was the H0 reading.
- **R3-C:** genuine reward-driven RL whose contribution is **margin/settle robustness (dwell)**, not the creation of delivery.
- No new RL was needed to establish this — the *existing* frozen R2 and FULL policies deliver under H1; the fix was purely the
  controller's boundary semantics.

## Preserved / retracted / not-yet-claimed

- **Preserved:** the 22/24 multiseed reproduction and the FULL-vs-NO_EXPANSION causal intervention — now understood to hold on the
  H1 (explicit-reset) interface, which is bit-exactly the frozen entry.
- **Retracted (unchanged from `85c5eca6`):** "R3-C authority unlock produced the first learned K6" — corrected to "R2 delivers under
  H1; R3-C refines dwell."
- **Not claimed:** K6 under H0 (DIRECT_HANDOFF). Whether a policy can be *trained* to deliver under H0 is the separate
  `NATURAL_HANDOFF_RL_CLOSURE` question (only relevant if one wants delivery without any handoff-reset).

## The scientific through-line (two mode boundaries, two corrections)

The same hybrid-boundary principle — *the same continuous state is not the same hybrid control situation* — has now bitten twice:
at the KINETIC→contact-loss frontier (R3-B, it changed **replay**) and at the APPROACH→KINETIC handoff (here, it changed **causal
attribution** and the "first learned K6" claim). Making the reset map an explicit, traced, online mode (H1) with a bit-exact
online↔offline equivalence is the general fix: a hybrid controller must not have events that exist only in offline artifact
construction.

## Files (all `8a0c1c7b`/`85c5eca6` modules imported UNCHANGED; no tag moved)

| file | role |
|---|---|
| `hymeko_rl/coin_delivery/theta_option/kinetic_handoff_reset.py` (+67) | `ExplicitHandoffResetMixin` + `HandoffResetTemporalController` (H1 clone+R2) + `HandoffResetUnlockController` (H1 FULL) |
| `hymeko_rl/experiments/coin_kinetic_handoff_reset_audit.py` (+164) | H0/H1 × clone+R2/FULL end-to-end + 3 gates + online↔frozen equivalence |
| `hymeko_rl/tests/test_coin_kinetic_contract.py` (+2 tests) | online≡frozen bit-exact + reset-before-policy; H0≠H1 distinct contracts |
| `reports/2026-07-28-coin-r9-handoff-reset/handoff_reset_audit.json` | the runs, equivalence, gates, verdict |

`ruff` clean; `radon cc -a` A/B. Full `test_coin_kinetic_contract.py` — see run log (36 tests). No new §6.5 anti-patterns; the
mixin replicates the frozen `dtau_for_step` only because the phase machine must not be double-invoked (documented). Frozen modules
untouched (`git diff` empty).

## Provenance

Immutable base `85c5eca6` (tag `coin-r9-first-learned-s1-k6-delivery` still → `8a0c1c7b`, unchanged). Python 3.11.15 / mujoco 3.10.0
/ numpy 2.4.6 / torch 2.12.0 / macOS-26.5.2-arm64 (Apple Silicon, CPU). Thread-pinned; frozen R2/expansion from the committed
multiseed checkpoint (seed_02); deterministic (online↔frozen Δ = 0.0). Peak RSS < 0.4 GB; audit wall 21.9 s.

## Status & next

`R2_IS_FIRST_LEARNED_K6_UNDER_EXPLICIT_HANDOFF_RESET` — the controller semantics is now fixed (explicit online HANDOFF_RESET,
bit-exactly the frozen entry), and both R2 and FULL deliver end-to-end under it. Per the plan, only *after* this result does a new
panel start. The greenlightable next step is the **kato14/kato15 multiseed reproduction under the H1 contract** (the C1 recipe now
run end-to-end from the cradle, not from a snapshot) — deferred for review. **STOP.**
