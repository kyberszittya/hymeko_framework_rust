# R9 corrective boundary — the K6 was a displaced-privileged-handoff artifact, and it's carried by clone+R2, not the authority unlock

**2026-07-28 · branch `recovery/coin-r9-causal-residual-delivery` · immutable base `2478a35d` · dev s1 (14250) · s4/s7 untouched · f1–f4 SEALED · NO RL · no tag moved**

## Summary — two corrections to the R9 K6 narrative

A component load-bearing probe on the reproduced C1 policy, and a handoff semantic-equivalence audit it triggered, together
**retract the "R3-C authority unlock delivered the first learned s1 K6" claim** — twice over:

1. **Component attribution (frozen-policy intervention).** On all 22 verified K6 checkpoints, masking one action term at a time
   (the clone GRU hidden that a_R2 and δ_expand condition on is always kept, so a single causal factor changes): **removing the
   expansion does not remove K6** (22/22 → 22/22, dwell 27.5 → 17). Removing R2 (→ 6/22) or the clone (→ 5/22) does. **The K6 from
   the frozen entry is carried by the clone + R2 heritage; the authority-unlock expansion refines dwell margin, it is not
   load-bearing for delivery.**

2. **Handoff attribution (equivalence audit).** The reason clone+R2 "delivers" at all is a **mode-boundary off-by-one**: the frozen
   KINETIC entry is the natural cradle→APPROACH→KINETIC handoff **advanced by one servo KINETIC step**. From the natural
   uninterrupted chain, clone+R2 does **not** deliver K6 (39.58 mm). **The learned K6 is valid only from a displaced, privileged
   entry interface.**

**Verdict: `FROZEN_ENTRY_IS_A_DISPLACED_PRIVILEGED_HANDOFF`.** This is the second time the hybrid mode boundary has changed the
causal reading of a learning result (cf. the R3-B frontier aliasing) — the same continuous state is not the same hybrid control
situation.

## Frozen-policy intervention (22 verified K6 checkpoints, from the frozen entry; no training)

| mode | K6 | dwell (median) | reading |
|---|---|---|---|
| FULL (clone + R2 + expansion) | 22/22 | 27.5 | control |
| **NO_EXPANSION (clone + R2)** | **22/22** | 17.0 | **expansion not load-bearing for delivery** |
| NO_R2 (clone + expansion) | 6/22 | 0 | R2 load-bearing |
| NO_CLONE (R2 + expansion) | 5/22 | 0 | clone load-bearing |
| EXPANSION_ONLY | 4/22 | 0 | expansion alone barely reaches |

All modes clean (0 stall/clamp/reversal) and safe. Artifact `frozen_intervention.json` (sha256 `5d235126…`, committed).

## Handoff equivalence audit — E0/E1/E2/E3 from the canonical s1 cradle

Comparison at the **first KINETIC pre-step** (before the first KINETIC action, to avoid pre/post aliasing):

| run | start | first KINETIC step | dtz at clone's first action | outcome |
|---|---|---|---|---|
| E0 | frozen-entry artifact | t=1 (of the resumed roll) | **75.75 mm** | **K6, min_dtz 16.86 mm** |
| E1 | cradle → APPROACH → natural KINETIC (uninterrupted) | t=4 | **77.99 mm** | **no K6, min_dtz 39.58 mm** |
| E2 | E1 paused/serialised at the pre-step, reconstructed, resumed | — | — | one-step **0.0**, continuation **0.0**, no K6 |
| E3 | fresh controller from E1's exact handoff state | — | — | identical to E2 |

| gate | result |
|---|---|
| `NATURAL_HANDOFF_CAPTURE_PASS` | ✅ E1's first-KINETIC pre-step fully & causally captured |
| `PAUSE_RESUME_ONE_STEP_IDENTITY_PASS` | ✅ **0.0** (bit-exact — cleaner than R3-B's 7 µm, the entry hidden is fresh) |
| `PAUSE_RESUME_CONTINUATION_PASS` | ✅ **0.0** continuation; E2 K6 == E1 K6 |
| `FROZEN_ENTRY_EQUIVALENCE_PASS` | ❌ E0 ≠ E1 (dtz Δ2.24 mm, qpos Δ0.018) |
| `END_TO_END_R2_K6_PASS` | ❌ the natural chain does not reach K6 |

**Localisation — it is physical, not a lost controller state.** At the first KINETIC pre-step the **controller/hybrid state is
identical** between E0 and E1: same mode (KINETIC), `kinetic_steps=0`, `prev_res=0`, `obs_hist_len=0`, clone hidden fresh (None).
The divergence is **purely physical**: `prev_tau` differs by 0.3 (a full slew unit), dtz by 2.24 mm, qvel by 0.63, contacts and
spin differ. E0's dtz (75.75) matches E1's **post-step** (75.50), not its pre-step (77.99). Cause: `freeze_kinetic_entry` uses
`roll_until` with `stop_when(phase==KINETIC)` checked **after** `step_ablation`, so it captures the state **after** the plain
APPROACH controller's servo has already taken the first KINETIC step. The learned policies (K2 clone, R2, R3-C) were all
trained/evaluated from that servo-warmed post-step; the natural chain hands off to the clone one step earlier, and from there they
do not deliver. Since the pause/resume identity is bit-exact (0.0), the snapshot contract is **not** the defect — this is a genuine
handoff-semantics displacement (the user's case 1 → case 2), not case 3.

## Corrected causal attribution — revised claims and non-claims

**Retracted:**
- "R3-C authority unlock produced the first learned s1 K6." The K6 is (a) carried by the clone+R2 heritage, and (b) reachable only
  from the servo-displaced frozen entry.
- The implicit "R2 saturates ~36 mm and cannot deliver" **and** "R2 delivers 16.86 mm" were both partial: the first was measured
  from the natural path, the second from the privileged entry. Neither is a natural-handoff K6.

**Still true:**
- The frozen-policy intervention (clone+R2 load-bearing, expansion = dwell refinement) — a within-interface causal fact.
- The R3-C update-zero identity, the multiseed determinism, and the 22/24 reproducibility — all real, but scoped to the frozen
  entry interface.
- No safety/cleanliness regression anywhere (0 stall/clamp/reversal throughout).

**Not claimed (the honest gap):**
- No genuine natural-handoff teacher-free strict K6 has been demonstrated. The next real objective is **natural-handoff → K6
  closure** — train/evaluate the policy on the uninterrupted cradle→APPROACH→KINETIC chain (or redefine the handoff so the frozen
  entry equals the natural one) — **not** the F0/F1/F2 retraining ablation, which is correctly paused.

## Files (corrective boundary; all `8a0c1c7b`/`2478a35d` modules imported UNCHANGED; no tag moved)

| file | role |
|---|---|
| `hymeko_rl/coin_delivery/theta_option/kinetic_ablation.py` (+106) | `AblationUnlockController` (frozen-policy action-term mask), F2 `DirectKineticController` (built, unused pending the paused retraining) |
| `hymeko_rl/experiments/coin_kinetic_ablation.py` (+139) | frozen-policy intervention over the 22 K6 checkpoints |
| `hymeko_rl/experiments/coin_kinetic_handoff_audit.py` (+198) | E0/E1/E2/E3 handoff equivalence audit + 5 gates |
| `reports/2026-07-28-coin-r9-ablation/frozen_intervention.json` (+`.sha256`) | preserved intervention artifact (hashed) |
| `reports/2026-07-28-coin-r9-handoff-audit/handoff_audit.json` | the four runs, full state diff, gates, verdict |
| `reports/2026-07-28-coin-r9-handoff-equivalence-and-causal-attribution.md` | this report |

`ruff` clean; `radon cc -a` A/B — one waiver: `_capture_at_first_kinetic` = C(12) (an audit capture loop mirroring the frozen
kernel; < the 15 fail ceiling). No new §6.5 anti-patterns. Frozen-module sanity tests (update-zero identity, teacher-adapter) still
pass. The F0/F1/F2 retraining panel and KatoLab cross-host work remain **paused** by the decision above.

## Provenance

Immutable base `2478a35d`, tag `coin-r9-first-learned-s1-k6-delivery` (verified unchanged, still → `8a0c1c7b`). Python 3.11.15 /
mujoco 3.10.0 / numpy 2.4.6 / torch 2.12.0 / macOS-26.5.2-arm64 (Apple Silicon, CPU). Thread-pinned; frozen-policy R2 champion from
the shared seed-0 regen (checkpoint seed_02); deterministic (pause/resume 0.0). Peak RSS < 0.4 GB; audit wall 22.5 s.

## Status

`FROZEN_ENTRY_IS_A_DISPLACED_PRIVILEGED_HANDOFF`. The R9 K6 line is corrected: the delivery is clone+R2 heritage, not the authority
unlock, and it holds only from a one-servo-step-displaced privileged entry — the natural uninterrupted handoff does not deliver.
Committed as a corrective scientific boundary alongside (not replacing) the historical tags. **STOP** — the next objective for
review is natural-handoff → K6 closure, not component retraining.
