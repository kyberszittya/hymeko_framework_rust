# Agile AIBO embodiment — the semantic stance lever, and its propulsion/turning tradeoff

**Date:** 2026-07-28 (JST) · **Branch:** `research/aibo-lyapunov-ph` (worktree `hymeko_aibo`)
**SIMULATION. Model (`.hymeko`) change — semantics, not control.** ·
**Verdict: `WIDER_STANCE_FIXES_THE_TURN_WALL_BUT_CRIPPLES_PROPULSION` (morphology tradeoff).**

---

## Why

The residual-RL (3 designs) and richer-turn-primitive arcs both hit the AIBO **turn/stability wall**
(stable turn ~15°/1000 steps; faster turning tips). That wall is a *model* property, so — as the user
noted — the lever is a **semantic (`.hymeko`) change**, not more control. The clearest lever: widen
the **stance** (the hip-abduction lateral offset) for a longer yaw moment arm + a wider base.

## The semantic change (`agile_embodiment.py`)

`widen_stance(hymeko_text, half_width)` moves the four `@hip_abduct_{leg}` lateral offsets from the
canonical **±0.062** outward. Committed variant `data/robotics/quadruped_agile.hymeko` uses **±0.11**
(same 12 leg actuators, same DOF — only the stance widens). The env loads it via `hymeko_path`; the
canonical model is untouched.

## Measured — the lever works for the wall …

| | stable turn (deg/1000) | tipping onset | 
|---|---|---|
| baseline (y=0.062) | 15.4 | tips at \|turn\|=0.9 (upright → −0.48) |
| **agile (y=0.11)** | **70.7 (~4.6×)** | **never tips** (stable through a full spin, upright ≥ 0.81) |

Widening the stance **4.6×'s the turn authority and eliminates tipping** — exactly what the wall
needed. Facing a ±40° goal drops from ~2600 steps of stable turning to ~570.

## … but it cripples forward propulsion (the honest tradeoff)

The *same* wide stance breaks the trot's forward walk:

| | forward walk (700 steps) |
|---|---|
| baseline (y=0.062) | +0.487 m |
| agile (y=0.11) | **~0.18 m (~2.7× weaker)**, and the net direction **flips with stance width** |

A stance sweep (0.062 / 0.075 / 0.085 / 0.095) showed the trot's net walk direction flip
chaotically (+0.41 / −0.30 / +0.41 / −0.11 m) — the gait sits at a marginal operating point,
**fragilely coupled to the exact geometry**. Even with an adapted (reversed-stride) gait the agile
model walks only ~0.18 m and reaches **0/10** multi-position goals (it turns well and stays upright,
but can't close distance). So a wide stance trades **forward propulsion for turning agility**.

## Conclusion — a morphology tradeoff, needs co-design

The AIBO's multi-position reach is bottlenecked by a **coupled propulsion/turning tradeoff** in the
morphology: a single geometric parameter (stance width) that buys turn authority costs forward
propulsion, and the gait doesn't robustly transfer across the change. A genuinely agile AIBO needs
**co-designed morphology *and* a gait tuned/trained for it** — multiple coordinated changes (stance,
leg geometry, mass, contact) plus a re-optimised controller — not a one-parameter edit. This closes
the "reach multiple positions" arc honestly at the embodiment level: the semantic lever is real (it
fixes the turn/stability wall), but agility is a *joint* model+controller design problem.

## Files

```
scenarios/aibo/agile_embodiment.py     NEW  (widen_stance semantic transform + measure_forward_propulsion)
data/robotics/quadruped_agile.hymeko   NEW  (wider-stance ±0.11 variant; canonical untouched)
tests/test_aibo_agile_embodiment.py    NEW  4 tests (transform correctness + the turn↑/walk↓ tradeoff)
```

## Tests / provenance

`ruff` clean. **4/4** agile-embodiment tests (transform sets the four offsets; rejects non-positive;
the variant emits+builds; and the regression-locked tradeoff — agile turns > 1.5× baseline AND walks
< 0.7× baseline). Full AIBO suite green. CPU, seed 0, MuJoCo, base=free; `.hymeko` is non-core
(CORE.YAML protects only the Rust crates/spec/rtl).

## Bottom line

The user's instinct was right — the fix is in the **model semantics**. Widening the stance
(a `.hymeko` edit) **does** fix the turn/stability wall (4.6× turn, no tipping). But it **cripples
forward propulsion** (2.7× slower, direction fragile), so multi-position reach stays 0/10. The AIBO
morphology has a **propulsion/turning tradeoff**: a truly agile AIBO is a **joint model+controller
co-design**, not a single-parameter change — the honest embodiment-level close of the arc.
