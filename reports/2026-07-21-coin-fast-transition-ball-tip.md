---
campaign: COIN neutral delivery — accelerate transitions + single spherical-tip transfer
title: TRANSITION_SPEED_POSITIVE + BALL_ZERO_SHOT_POSITIVE — the confirmation delay is removed (−34%) and the learned chain transfers zero-shot to one sphere per arm (4/9)
date: 2026-07-21
branch: exp/coin-fast-transition-ball-tip
source_commit: c38676b
classification: TRANSITION_SPEED_POSITIVE + BALL_ZERO_SHOT_POSITIVE
---

# Accelerate learned transitions + single spherical-tip transfer

**Created-at:** 2026-07-21 17:03 JST. Two isolated parts on the frozen NEUTRAL_DELIVERY_POSITIVE baseline (`56e0324`/
`c38676b`). Successful checkpoints kept read-only: E approach `E_valselect_v2.pt` (**b822a660**), handoff transport
`handoff_best.pt` (**8955e8db**), frozen transport `learned_delivery_positive.pt` (**8bd73d8c**). control_dt = 0.010 s,
real-time = 100 fps.

## PART A — transition acceleration → **TRANSITION_SPEED_POSITIVE**

### §1–§2 Timing audit (headline states, exact chain)
The chain is E-approach (until handoff) → handoff transport → certify. Two removable/irremovable latencies:
- **Bilateral-grasp confirmation window** = `grasp_hold` control steps (was **3**) — a STATE_MACHINE_DELAY.
- For states where a full grasp never forms (intermittent one-sided contact, push-delivery), the E-approach ran to its
  **160-step cap** before handing off — a HANDOFF wait, not physical.
- `handoff → first targetward motion` ≈ 14–23 steps — policy ramp-up (not touched; a weight fine-tune would risk the
  fragile 3/9 delivery, so per §4 it was not done).

### §3–§4 Change: immediate, safe handoff (weights unchanged)
Handoff now fires on **bilateral contact confirmed for 1 step** OR **one-sided contact held for a short window (20
steps)** — the latter removes the wait-to-cap dead time for push-delivery states. Acceptance gate met:

| headline (CONCAVE_CLAMP ring) | before (grasp_hold 3) | after (accelerated) |
|---|---|---|
| strict delivery, 3 states | 10/10 each | **10/10 each (no regression)** |
| seed 1045 total duration | 211 steps = 2.11 s | **139 steps = 1.39 s (−34%)** |
| seed 1447 bilateral→handoff | 2 steps | **0 steps** |
| seed 1278 | 3.36 s | unchanged (physical approach duration; contact never sustained) |

**Verdict: TRANSITION_SPEED_POSITIVE** — real phase-transition latency decreases (1045 −34%, 1447 confirmation 2→0)
with **no delivery regression** and no weight change. The remaining 1278 duration is the E policy's genuine approach,
not a removable state-machine delay; the §5 optional speed fine-tune was declined (would risk the fragile delivery).

## PART B — single spherical tip per arm → **BALL_ZERO_SHOT_POSITIVE**

### §6 Geometry resolved — POINT *is* one sphere per arm (no `BALL_TIP` built)
- **CONCAVE_CLAMP** (current E0, "large pad"): **12 spheres** — a ring of 6 (r=0.012) per arm.
- **POINT** (canonical golden): **exactly one sphere per arm** (`fingertip_left`/`fingertip_right`, r=0.014,
  symmetric, no prongs/cup/pad/suction/weld, friction [1.0, 0.05, 0.001]).

POINT already satisfies the single-spherical-tip requirement, so it is **reused** (no duplicate geometry). Same
action layout (nu=4) and obs schema (41-dim ACTOR_FIELDS + `node_features`) as the ring → schemas compatible → zero-shot
is valid. Matched problems: same seeds, same true-neutral condition (arm [0,0,0,0], no contact, no prefix, coin
outside); the coin/target placement is geometry-independent, so the neutral clearances are identical.

### §8–§9 Zero-shot transfer (unchanged learned policies)
| embodiment | first-contact | bilateral | **strict delivery** | winners |
|---|---|---|---|---|
| CURRENT_PAD (CONCAVE_CLAMP ring) | 9/9 | 7/9 | 3/9 | 1045, 1278, 1447 |
| **SPHERICAL_TIP (POINT, 1 sphere/arm)** | 7/9 | 4/9 | **4/9** | **1011, 1045, 1174, 1447** |

The single-sphere tip has *lower* contact rates (smaller target) but **delivers one more state** (4/9 vs 3/9). Per §9,
BALL_ZERO_SHOT_POSITIVE requires ≥1 headline state at ≥8/10 with the spheres — **met on four** (each deterministic →
10/10). The dominant spherical-tip difference is BILATERAL_CONTACT (7→4), not a transport failure — transport transfers.

### §12 Causal (POINT, per state; deterministic → 10/10)
| POINT headline | clearance | E + handoff (zero-shot) | frozen transport | zero-action |
|---|---|---|---|---|
| **1011** | **+0.079** | **10/10** | 0/10 | 0/10 |
| 1045 | +0.011 | 10/10 | 0/10 | 0/10 |
| 1174 | +0.014 | 10/10 | 0/10 | 0/10 |
| 1447 | +0.039 | 10/10 | 0/10 | 0/10 |

Headline POINT state **1011 (+0.079): E+handoff 10/10, frozen transport 0/10, zero-action 0/10** — the unchanged
learned chain certifies neutral delivery on a single spherical tip per arm. **No §10 carry oracle or §11 adaptation
needed** (zero-shot already delivers).

## §13 Verdicts
- **Transition:** TRANSITION_SPEED_POSITIVE.
- **Embodiment transfer:** BALL_ZERO_SHOT_POSITIVE.

## §14 Videos (`reports/figures/2026-07-21-coin-delivery-e0/`, 100 fps real-time / 50 fps slow)
- `coin_delivery_fast_transition_pad_real_time.mp4` (e8a3ee14) — accelerated pad chain, 1045, 1.51 s.
- `coin_delivery_ball_tip_zero_shot_real_time.mp4` (932d8d5b) — POINT single-sphere zero-shot, state 1011 (+0.079).
- `coin_delivery_ball_tip_zero_shot_slow_motion.mp4` (e6e534bd, 50 fps) + `.gif` (b9ed3ffd).
- `coin_delivery_pad_vs_ball_tip.mp4` (f8e00ead) — ring vs single-sphere on the same state (1447), both deliver.

## §15 Provenance
- Checkpoints (all read-only, unchanged): E approach `b822a660`, handoff transport `8955e8db`, frozen transport
  `8bd73d8c`.
- Current fingertip geometry: CONCAVE_CLAMP = 12 spheres (6/arm, r=0.012). Spherical tip: POINT = 1 sphere/arm
  (r=0.014). Same compiled DOF (nu=4, nq=7); geometry differs (own model).
- Code: `hymeko_rl/experiments/coin_neutral_start.py` (`eval_composed` gains `contact_window`/accelerated handoff;
  `neutral_env(geom=)` builds the POINT variant through the same builder). ruff clean.
- Problem/state hashes: true-neutral seeds; POINT winners 1011/1045/1174/1447, clearances +0.079/+0.011/+0.014/+0.039.
- Verify command (POINT zero-shot):
  ```
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. python -c \
  "from hymeko_rl.experiments.coin_neutral_start import neutral_env, eval_composed; \
  from hymeko_rl.experiments.coin_delivery_e0_stabilize import build_sac; import torch; \
  a,_=build_sac('mlp',obs_dim=41,flat_dim=41,action_dim=6,action_scale=1.0); \
  a.load_state_dict(torch.load('experiments/2026_07_21_coin_neutral_handoff/handoff_best.pt',weights_only=True)); \
  print(eval_composed(a,[1011],grasp_hold=1,contact_window=20,env_cf=neutral_env(prefix_steps=0,geom='POINT')))"
  ```
- Commits: `0200df1` (code), this report additive. Branch `exp/coin-fast-transition-ball-tip` from `c38676b`.
  Preserved read-only: b822a660, 8955e8db, 8bd73d8c, P&P d2da720a, Beni 4630b537.
- Host Apple M5 Pro; threads pinned; RL not bit-reproducible (BLAS); claims rest on deterministic per-state eval.

## Bottom line
The transition is genuinely faster (1045 −34%) with no delivery loss, and — more strikingly — the learned neutral
chain **transfers zero-shot from a 12-sphere clamp to a single sphere per arm**, delivering 4/9 (one more than the pad)
with certificates where the frozen transport and zero-action both fail. The learned approach + handoff transport are
robust to fingertip geometry.
