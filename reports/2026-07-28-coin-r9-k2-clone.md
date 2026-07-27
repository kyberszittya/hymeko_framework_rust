# K2 — the KINETIC feedback clone: teacher-free closed-loop `LOCAL_KINETIC_FEEDBACK_SKILL_PASS`

**2026-07-28 · branch `recovery/coin-r9-causal-residual-delivery` · worktree `hymeko_coin_r9_wt` · dev s1 (14250) · s4/s7 untouched · f1–f4 SEALED · NO RL / DAgger started**

## Summary

K2-A behaviour-clones the 32 K1-A feedback labels into a small recurrent actor (GRU-64 + MLP head 64→32→4, tanh); K2-B
deploys it **teacher-free** in the frozen chain — frozen APPROACH → learned KINETIC clone → frozen G0/release → frozen coast →
frozen K6 — and grades by the closed-loop behaviour, not action-R².

**Gate: `LOCAL_KINETIC_FEEDBACK_SKILL_PASS`** (the intermediate gate, distinct from delivery). With no teacher in the loop the
clone keeps the coin moving (v_par positive throughout, 0 stalls, 0 sign-reversals), does **not** clamp or drift into a
terminal-failure state, and gets **measurably past the 48–51 mm hand-tuning plateau** (min_dtz **46.2 mm**). It does **not** yet
reach the 20–30 mm corridor or strict K6 — as expected: the K1-A bank covers only the 55–70 mm entry neighbourhood, so below
~55 mm the clone extrapolates. **Next per the pre-registered decision: `TD3_VS_SAC`** (the clone maintains transport but sits at
20–50 mm ⇒ bounded-residual RL now, not another BC bank). **Stopped here — no RL/DAgger started.**

## K2-A — the clone fit

- Model: `GRU(41→64)` + `Linear(64→32)→ReLU→Linear(32→4)→tanh`; input normalisation **frozen from the K1-A bank**; weighted
  Huber loss (delivering ×1.5, near-slip ×1.3; terminal-failure states never enter the loss).
- Fit: loss **0.0315 → 0.00062** (400 epochs, ~0.6 s). Seed audit (seeds 0/1/2) final loss 0.00062 / 0.00064 / 0.00041 — stable.
- **Streaming determinism (control):** batch == streaming max gap **1.5e-7** (a sequence processed at once equals step-by-step
  with the carried hidden state); deterministic hidden-state reset (bit-identical replay); actions tanh-bounded.

## K2-B — teacher-free closed-loop smoke (event trace)

The clone owns only the KINETIC transport Δτ; the frozen release / contact-risk / horizon guards and the coast/K6 downstream
are unchanged. **No teacher/CEM in the loop.**

| t | phase | dtz mm | v_par | fn_L | fn_R | note |
|---|---|---|---|---|---|---|
| 1–3 | FROZEN APPROACH | 76–79 | ≤ 0 | 2.5–4.0 | 2.7–8.7 | momentum build |
| 4 | **KINETIC clone** | 77.99 | **+0.195** | 3.08 | 2.68 | clone takes over |
| 5 | KINETIC clone | 75.52 | +0.280 | 1.79 | 0.78 | grip fading |
| 6 | KINETIC clone | 72.40 | +0.328 | 0.87 | 0.41 | light contact |
| 7 | KINETIC clone | 68.90 | +0.355 | 0.27 | 0.16 | sliding transport |
| 8 | KINETIC clone | 65.18 | +0.370 | 0.37 | 0.31 | momentum held |
| 9 | KINETIC clone | 61.42 | +0.366 | 0.24 | 0.21 | last gripped step → coast |
| coast | frozen | → **46.2** | — | 0 | 0 | passive landing |

**The clone reproduced the teacher's mechanism** — build forward momentum, **fade the grip** (fn 3.08 → 0.24), light-contact
slide-transport while holding **positive v_par (0.195 → 0.370)** — carrying the coin 78 → 61 mm gripped, then coasting to 46.2 mm.
It is not a firm-grip clamp and not a stall: it is the intended light-contact kinetic transport, learned from the bank.

## The report's required signals

| signal | value |
|---|---|
| BC fit (loss) | 0.0315 → 0.00062 |
| streaming determinism | batch==streaming gap 1.5e-7; replay bit-identical |
| min_dtz (closed-loop) | **46.2 mm** (past the 48–51 mm plateau) |
| v_parallel (transport) | **positive throughout**, [0.195, 0.370], 0 sign-reversals |
| force / contact profile | fn 3.08 → 0.24 N (grip fades — light-contact sliding, no clamp) |
| terminal-failure entries | **0** (0 stalls; the clone never drove the coin into a stalled/clamped state) |
| K6 delivered | **False** (expected — bank is local; clone alone not required to deliver) |
| safety | peak_qdot 2.0 ≤ 3.0; peak coin speed 0.38 ≤ 1.5 ✅ |
| seed audit (min_dtz) | 46.2 / 47.8 / 56.2 mm (seeds 0/1/2) |

## Gate reasoning (the two gates are distinct)

- **`LOCAL_KINETIC_FEEDBACK_SKILL_PASS` — reached.** Teacher-free; kept the coin moving (no clamp/stall); no terminal-failure
  drift; **past the 48–51 mm plateau** (46.2 mm). A positive gate even without final K6.
- **`FIRST_LEARNED_S1_K6_DELIVERY` — not reached** (clone alone does not deliver; not a prerequisite for RL).

## Next (pre-registered — not started here)

The clone maintains transport and brings the coin closer but sits at 46 mm (the 20–50 mm band), so the next round is **bounded
residual RL, TD3 vs SAC** — not another large BC bank, not (necessarily) DAgger (the clone does not stall). RL contract to use:
`u_t = clip(u_clone(o≤t) + α·tanh δ_ψ(o≤t), u_min, u_max)` with update-zero (δψ=0 ⇒ bit-identical to the clone), RL modifying
only the KINETIC segment, task-tied reward (progress / positive v_par / close+moving release / K6; penalise stiction / clamp /
early contact-loss / sign-reversal / safety), and a matched TD3-vs-SAC comparison (same replay / demo-seed / residual bound /
interaction budget / eval seeds; TD3 the first main branch, SAC the mandatory comparator).

## Files touched (all new / additive; K0/K1 modules untouched)

| file | role |
|---|---|
| `hymeko_rl/coin_delivery/theta_option/kinetic_clone.py` | `KineticClone` (GRU+head), `CloneActor`, `train_clone`, `KineticCloneController` |
| `hymeko_rl/experiments/coin_kinetic_k2_clone.py` | K2-A fit + K2-B teacher-free closed-loop smoke + seed audit |
| `hymeko_rl/tests/test_coin_kinetic_contract.py` | +2 tests (batch==streaming/bounded; frozen-APPROACH match + slew bound) |
| `reports/2026-07-28-coin-r9-k2-clone/{k2_clone.json, clone_seed0.pt}` | metrics + event trace + the checkpoint |

## Tests / static analysis

- `pytest test_coin_kinetic_contract.py` — **13 passed** (11 prior + the 2 clone tests).
- `ruff check` clean; `radon cc -a` on `kinetic_clone.py` = **A (2.16)**; no new suppressions; no §6.5 anti-patterns.

## Provenance

Git `f23eba77` (K1-A commit; K2 files uncommitted at run time). Python 3.11.15 / mujoco 3.10.0 / numpy 2.4.6 / torch 2.12.0 /
macOS-26.5.2-arm64. Seeds: cradle 14250; clone train seeds 0 (main) + 0/1/2 (audit); CEM seed 20260727 (frozen, only used to
build the committed bank). Bank `reports/2026-07-28-coin-r9-k1a-bank/feedback_labels.json` (hash `bc36e0521982bd36`). Peak RSS
0.31 GB; total 22.4 s. Deterministic (fit + closed-loop reproduce).
