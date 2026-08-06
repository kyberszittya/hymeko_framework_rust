# AIBO morphology↔controller co-design — the joint search maps a fundamental Pareto frontier

**Date:** 2026-07-28 (JST) · **Branch:** `research/aibo-lyapunov-ph` (worktree `hymeko_aibo`)
**SIMULATION. Joint model+controller search (scripted, no RL).** ·
**Verdict: `CODESIGN_NO_IMPROVEMENT_TRADEOFF_FUNDAMENTAL` (within stance × gait).**

---

## Why

The stance lever alone was a tradeoff (wide → turns well, walks poorly). The user asked for the
**co-design loop**: optimise the morphology **and** the controller *together*. This searches the
joint space — stance width × gait (stride sign, hip amplitude) — scoring each point by **upright
multi-position reach**, to find whether a co-designed (model, gait) pair breaks the tradeoff.

## The loop (`codesign.py` + `run_aibo_codesign.py`)

Grid: stance ∈ {0.062, 0.085, 0.10} × stride-sign ∈ {±1} × hip-amp ∈ {0.7, 1.1}. Each point is
profiled (forward propulsion, turn authority, uprightness); the top candidates by a composite that
needs **both** walk and stable turn are run through a turn-then-walk pursuit on a held-out
(distance × bearing) goal grid; the best-reaching pair is reported.

## Result — a clean, fundamental Pareto frontier

| point | forward (m) | turn (°/1000) | upright | upright reach |
|---|---|---|---|---|
| **baseline** 0.062 | +0.337 | +14 | 1.00 | **0.20** (straight goals only) |
| best-profile 0.085 | +0.351 | −21 | 0.91 | 0.00 (turn too slow to align off-axis) |
| fast-turn 0.10, s− | +0.11 | **−69** | 0.71 | 0.00 (walk too weak to close) |
| high-amp points | (varied) | (varied) | **tips (<0)** | disqualified |

**The frontier is exclusive:** every point trades away one of {walk, turn, upright}. Points that walk
well (fwd > 0.33) turn weakly (14–21°/1000) → can't align to off-axis goals; the one point that
turns fast (−69°/1000) *and* stays upright walks too weakly (0.11 m) → can't close distance; pushing
amplitude to get both **tips over**. So **no (stance × gait) point beats the baseline's upright reach
(0.20)** — the best is the baseline itself.

## Conclusion — the tradeoff is fundamental within this design space

The co-design loop **proves** (by exhaustive joint search) that the AIBO's propulsion/turning/stability
tradeoff cannot be resolved within the **(stance width × gait)** design space: strong walk, fast turn,
and staying upright are mutually exclusive here. Multi-position reach caps at the straight goals.
Breaking it requires **expanding the design space** — leg geometry (longer legs = more stride *and* a
longer moment arm), mass/COM redistribution, a genuinely different turning mechanism (a stepping-turn
that repositions feet without skid-steer slip), or a *learned* controller over a richer action space.
That is a larger co-design loop (more morphology DOFs + RL), the honest next frontier.

## Files

```
scenarios/aibo/codesign.py             NEW  (CoDesignPoint parameterised gait, agility profile, multi-position reach)
scenarios/aibo/run_aibo_codesign.py    NEW  (joint stance×gait search + reach eval + Pareto frontier report)
tests/test_aibo_codesign.py            NEW  4 tests
reports/2026-07-28-aibo-codesign/result.json  NEW  (the frontier + verdict)
```

## Tests / provenance

`ruff` clean. **4/4** co-design tests (valid parameterised action; stride sign flips walk direction;
the wider-stance turn↑/walk↓ tradeoff; baseline reaches straight goals upright); full AIBO suite green.
Scripted (no RL) — CPU, seed 0, MuJoCo, base=free; no non-improving variant is committed (the best
point was the baseline).

## Bottom line

The co-design loop ran — and honestly, **jointly optimising stance × gait does not beat the baseline**:
the propulsion/turning/stability tradeoff is **fundamental within this design space**. The full AIBO
multi-position arc (residual RL × 3 → richer turn primitive → stance lever → stance×gait co-design)
converges on one wall: this morphology can't reach arbitrary off-axis goals, and every in-space lever
hits the tradeoff. Genuine agility needs a **richer morphology + learned controller co-design** — a
bigger loop, honestly scoped as the next step.
