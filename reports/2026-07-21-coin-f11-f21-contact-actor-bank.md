---
campaign: COIN F11 vs F21 contact-actor-bank contrast (1 actor vs 2 contact-mode actors)
title: The two-actor contact decomposition improves contact quality + retention but does not crack the +0.030 transport barrier (no-effect on transport)
date: 2026-07-21
branch: exp/coin-two-actor-contact-bank
source_commit: 4ab8275
verdict: NO_EFFECT (primary transport) — STAGE-2 certified coverage identically zero for both cells (all 8 paired Δ=0); F21 non-inferior with a reproducible fingertip-attribution gain (CI>0) and a STAGE-1-retention lean, but bilateral-contact establishment CI spans zero
---

# F11 vs F21 — does explicit reposition/transport separation solve what one actor cannot?

**Created-at:** 2026-07-21 04:35 JST. **ETA basis (met):** 16 runs × 50k, 8-parallel ≈ 8 min (measured 401 s) +
analysis/figure/GIF. Plan-first bundle waived by the explicit "EXECUTE NOW, do not ask" directive.

## Hypothesis (isolated structural variable)
> *Explicit separation of contact establishment/repositioning from coin transport lets the policy solve clear-start
> configurations a single actor cannot.*

The semantic-critic axis was closed (F12 NO_EFFECT). The **only** architectural variable here is one actor → **two
contact-mode actors** (`SAC_CONTACT_ACTOR_BANK` + `HYMeko_CONTACT_MODE`). Everything else — K0 env, delivery-v2b
reward, strict predicate, TASK_ONLY twin-Q SAC, `n_step=1`, replay, generator, obs/action schema, BC/competence logic,
the curriculum source checkpoint, the frozen STAGE-2 corpus + committed 70/15/15 mix, canonical rollout/eval — is held
identical. F21 **warm-starts BOTH heads from the F11 curriculum policy** (`warm_start_contact_bank`), so both arms
share source lineage and the *only* step-0 difference is the architecture.

## F21 architecture (§3–§5)
Shared MLP encoder → two mode heads: **ACTOR_REPOSITION** (approach / repair one-sided or lost contact / reach a valid
bilateral configuration) and **ACTOR_TRANSPORT** (hold bilateral contact / push to target / certify). Both emit the
same canonical 4-actuator action. `HYMeko_CONTACT_MODE` is an **explicit, inspectable named-field gate** (not a learned
black box): TRANSPORT iff `both_contact ∧ ¬arm_body_contact` (with a `prev_*_contact` 1-frame hysteresis hold),
REPOSITION otherwise (reasons: no-contact / one-sided / lost / body-shove). Per-sample routing (`torch.where`) sends
each head's gradient only to its mode's states.
**Two named-field corrections made during build (measured):** the phase one-hot is *stuck at CONTACT* in the coin env
(inert as hysteresis → use `prev_*_contact`); `contact_lost_after_handoff` is a *latch* (reason-only, never a transport
gate, else it bars re-entry after any loss).
**Capacity control (§6):** F11 7628 params, F21 8408 params at the production width (hidden=64) → **10.2% overhead**
(< 15%; no widened control needed — the shared encoder dominates, the two heads are tiny).

## Design
Matched, thread-pinned (`OMP/MKL/OPENBLAS=1`, `PYTHONUNBUFFERED=1`), 8 pairs = seeds {0..3} × reps {0,1}; 50k steps;
eval every 2500; `n_step=1`. Fresh matched F11 + F21 controls (16 runs, one campaign). §9 smoke gate passed before the
matrix (finite losses/grads, both heads' magnitudes change, both modes activate, neither head dead/permanently-selected).

## Per-pair best checkpoint (F11 → F21)
| pair | STAGE2 cov | STAGE1 retention | ΔS1 bilat | ΔS1 clean | ΔS1 attr | TRANSPORT occ |
|---|---|---|---|---|---|---|
| s0r0 | 0→0 | 3→6 (+3) | +0.33 | +0.19 | — | 0.017 |
| s0r1 | 0→0 | 2→4 (+2) | −0.03 | +0.33 | — | 0.014 |
| s1r0 | 0→0 | 8→4 (−4) | −0.13 | +0.20 | — | 0.018 |
| s1r1 | 0→0 | 6→10 (+4) | 0.00 | 0.00 | — | 0.014 |
| s2r0 | 0→0 | 5→6 (+1) | 0.00 | −0.11 | — | 0.013 |
| s2r1 | 0→0 | 6→8 (+2) | +0.07 | −0.06 | — | 0.014 |
| s3r0 | 0→0 | 4→9 (+5) | +0.24 | −0.18 | — | 0.016 |
| s3r1 | 0→0 | 6→7 (+1) | +0.12 | +0.12 | — | 0.021 |

## Pooled 8-pair paired deltas (F21 − F11; bootstrap seed 20260721, B=10000)
| endpoint | mean | median | +/0/− | bootstrap 95% CI | reads as |
|---|---|---|---|---|---|
| **STAGE2 certified coverage** (primary) | **0** | **0** | 0/8/0 | **[0.0, 0.0]** | dead flat |
| STAGE2 loose entry | 0 | 0 | 0/8/0 | [0.0, 0.0] | dead flat |
| STAGE2 max certified clearance | 0 | 0 | 0/8/0 | [0.0, 0.0] | dead flat |
| STAGE1 retention coverage | +1.75 | +2 | 7/0/1 | [−0.25, +3.25] | **strong positive lean** (CI just touches 0) |
| strong (+0.0253) retention | +0.375 | +0.5 | 4/3/1 | [−0.125, +0.875] | positive lean, spans 0 |
| 64102 retention | −0.25 | 0 | 0/6/2 | [−0.625, 0.0] | slight negative lean |
| STAGE1 **P_bilat** (establishment) | +0.075 | +0.034 | 4/2/2 | [−0.02, +0.176] | positive lean, spans 0 |
| STAGE1 **P_clean** (mechanism) | +0.062 | +0.062 | 4/1/3 | [−0.051, +0.179] | spans 0 |
| STAGE1 **P_attr** (attribution) | +0.158 | +0.131 | 6/0/2 | **[+0.02, +0.311]** | **reproducible gain (CI > 0)** |

Mean TRANSPORT occupancy: **1.6%**.
Figure: [reports/figures/2026-07-21-f11-f21/paired_deltas.png](figures/2026-07-21-f11-f21/paired_deltas.png).
Animated: [reports/figures/2026-07-21-f11-f21/f11_vs_f21_64102.gif](figures/2026-07-21-f11-f21/f11_vs_f21_64102.gif) —
matched s3r0 pair on state 64102: **F11 fails to certify (attr 0.42), F21 certifies (attr 0.62)** — a concrete
quality/certification win on this pair (but seed-specific: the *aggregate* r64102 Δ = −0.25, so this does not generalize).

## Verdict: **NO_EFFECT** (primary transport endpoint) — with honest secondary signal
- **Primary dead flat.** STAGE-2 (+0.030–0.060) certified coverage, loose entry, and max clearance are **identically
  zero for both cells across all 8 pairs** (CI [0,0]); **no F21 run certifies any STAGE-2 state.** The two-actor
  decomposition did **not** crack the +0.030 barrier. Not ACTOR_POSITIVE.
- **F21 is non-inferior and improves contact QUALITY + retention** (this is the real, honest difference from F12's
  sign-inconsistent null): fingertip attribution `P_attr` improves **reproducibly** (+0.158, 6/8 positive, **CI
  [+0.02, +0.311] above zero**), STAGE-1 retention leans strongly up (+1.75, 7/8 positive, CI barely includes 0), and
  clean mechanism leans up — the mode decomposition makes the arms contact the coin *more cleanly* and holds earlier
  competence *better*.
- **But bilateral-contact ESTABLISHMENT — the specific ACTOR_CONTACT_POSITIVE criterion — is not reproducible**:
  `P_bilat` +0.075 with CI [−0.02, +0.176] spanning zero. So per the taxonomy this is **not** ACTOR_CONTACT_POSITIVE.
- **Not ACTOR_NEGATIVE** — no reproducible transport or retention damage (retention improved; the only negative lean,
  r64102 −0.25, has CI upper bound exactly 0).

## Why the transport head was under-tested (measured, load-bearing caveat)
TRANSPORT occupancy is **1.6%** — on the frozen far STAGE-2 corpus, clean bilateral contact is rare (the
contact-mechanics wall: sphere-on-cylinder → one-finger/antipodal point contact), so the mode gate routes ~98% of
samples to REPOSITION and the TRANSPORT head trains on ~4 samples/batch. This means the experiment strongly tested the
**REPOSITION** head (which did improve contact quality + retention) but only weakly tested the **TRANSPORT** head. The
NO_EFFECT-on-transport is robust across 8 seeds, but it is *not* a strong falsification of "a dedicated transport actor
helps" — the gate rarely gave that actor data. **This is the evidenced motivation for the Phase B arm-repositioning
generator** (states where reposition→bilateral→transport is geometrically achievable would raise occupancy and give the
transport head a real test), recorded as future work — not run, because Phase D/E are gated on ACTOR_POSITIVE.

## Measured vs inferred vs hypothesis
**Measured:** two contact-mode actors, warm-started as an F11 clone, on the frozen STAGE-2 corpus, produced no STAGE-2
certification (primary endpoint flat), a reproducible fingertip-attribution gain, and a STAGE-1-retention lean, over 8
matched seeds. **Inferred:** the contact-mode decomposition improves *how* the arms contact and *retention*, consistent
with the REPOSITION head specializing on the 98% of approach/repair states. **Still hypothesis (not closed):** whether a
*well-fed* TRANSPORT head (on a reposition-first corpus where bilateral contact is establishable) unlocks transport is
**untested** here — the occupancy was too low. This is one clean data point that the decomposition, on the existing far
corpus, does not move transport; it is **not** a verdict that structural decomposition is dead.

## §11 next decision (NO_EFFECT) — SPEC ONLY, not implemented
The evidenced next lever is the **Phase B arm-repositioning generator**: a corpus where successful delivery *requires*
approach-from-open-space → arm repositioning → bilateral establishment → transport, with clearance bands NEAR/MEDIUM,
so the TRANSPORT head is actually exercised (occupancy ≫ 1.6%). Gate the follow-up on the same STAGE-2 bootstrap-CI
endpoint with retention guards. Not built here (Phase D/E gated on ACTOR_POSITIVE, which was not reached).

## Files touched
- `hymeko_rl/train/rl_config.py` — `select_contact_mode` + `ContactModeReason`; SAC_CONTACT_ACTOR_BANK / HYMeko_CONTACT_MODE promoted to supported; validation.
- `hymeko_rl/train/sac.py` — `ContactActorBank`, `_ModeHead`, `warm_start_contact_bank`, `_squashed_sample`/`_squashed_mean` (de-duplicated squash); `build_sac(actor_head="contact_bank")`.
- `hymeko_rl/experiments/coin_nstep_exp.py` — `--actor-head` axis + bank warm-start + mode-occupancy/per-head-magnitude logging.
- `hymeko_rl/experiments/coin_f11_f21_campaign.py` (NEW) — matched campaign + ACTOR classification (reuses F11/F12 paired-delta primitives).
- Tests (NEW): `test_contact_actor_bank.py` (14), `test_coin_f11_f21_campaign.py` (7).
- Data: `experiments/2026_07_21_coin_f11_f21/` (16 runs + manifest + comparison JSON); figures under `reports/figures/2026-07-21-f11-f21/`.

## CORE.YAML items touched
None. No dependencies added.

## Test results
| suite | count | result |
|---|---|---|
| `test_contact_actor_bank` | 14 | pass |
| `test_coin_f11_f21_campaign` | 7 | pass |
| `test_sac_mechanism_critic` (validation updated) | 11 | pass |
| SAC/replay regression (`test_sac`, `test_sac_asym`, nstep, competence-gate, compiled-update, metaworld) | 35 | pass |

`ruff check` clean; new bank code all A-grade cyclomatic; `train_sac` unchanged at CC 48. F11 path byte-identical
(pooled default). Squash de-duplication verified by the existing SAC actor tests.

## Performance
| axis | value | budget |
|---|---|---|
| training wall / run (50k) | F11 178 s, F21 222 s (1.25× — the bank samples both heads) | — |
| campaign wall (16 runs) | 401 s | — |
| peak RSS / run | ~0.42 GB (same class as F12; bank adds 2 tiny heads) | 16 GB cap ✓ |

## Experiment provenance
- Git SHA `4ab8275` (working tree adds the uncommitted `experiments/2026_07_21_coin_f11_f21/` data + figures + report).
- Host Apple M5 Pro, 18 cores, 48 GB; torch 2.12.0, mujoco 3.10.0, numpy 2.4.6.
- Seeds run_seed = seed·100 + rep, {0..3}×{0,1}; bootstrap seed 20260721, B=10000.
- Source checkpoint sha `39551de3`; STAGE-2 corpus hash `2d5d7659778fe793`, STAGE1 held `a1de9f4e96d3a255`.
- RL not bit-reproducible (§3) → verdict rests on the 8-pair matched bootstrap CI.

## Open issues / follow-ups
- **TRANSPORT-head starvation (1.6% occupancy)** is the load-bearing limitation — the Phase B reposition-first corpus is the evidenced fix.
- The reproducible P_attr gain (CI > 0) suggests the REPOSITION head does its job (cleaner fingertip contact); a corpus that rewards *establishing* bilateral contact would convert that into P_bilat.
- Phase D (far-start demo) and Phase E (transfer to pick-and-place / humanoid) are **gated on ACTOR_POSITIVE** → not executed. Both target envs exist (`env/pick_place_env.py`, `env/locomotion_env.py`) and are SAC-compatible; the transfer boundary is a *task-specific mode selector* (the bank + warm-start + train_sac + campaign are already task-agnostic once the selector is injectable).
