# Humanoid balance certified by the SAME Vukobratović-ZMP certificate as the AIBO — multi-embodiment core

**Date:** 2026-07-29 (JST) · **Branch:** `research/humanoid-com-lyapunov` (worktree `hymeko_humanoid`)
**SIMULATION.** · **Verdict: `VUKOBRATOVIC_ZMP_CERTIFICATE_IS_EMBODIMENT_AGNOSTIC; HUMANOID_SUPPORT_MARGIN_NOW_GENUINE`.**

---

## Why

The AIBO turning work produced a proper **Vukobratović-ZMP-in-support** certificate (the full ZMP with the
angular-momentum-rate term ``Ḣ``, under the motion contract). The humanoid's own `support_margin`
certificate was **vacuous** (welded-base era) and its balance was known to `SURVIVE_PARTIALLY_NOT_STABLE`
(the PD scaffold stays upright without genuine stability). The ZMP support criterion is **embodiment-
agnostic** — whole-body CoM + momentum + a support polygon — so the *same* certificate makes the
humanoid's `support_margin` **genuine**.

## What was built (`scenarios/humanoid/zmp_stability.py`)

`vukobratovic_zmp` (identical physics to `hymeko_aibo/scenarios/aibo/turn_stability.py` — MuJoCo
`subtree_com/linvel/angmom`, ``a``/``Ḣ`` by finite difference) + `foot_support_margin` (ZMP vs the
`foot_l/foot_r` support box) + `zmp_balance_certificate` (CIP-0 **SAFETY**: PASS iff the ZMP stays in
support). The AIBO used paws; the humanoid uses feet — the only embodiment-specific line.

## Result — genuine, monotone, and it distinguishes STABILITY from SURVIVAL

PD scaffold (a=0) balance at increasing pitch-rate perturbation:

| pitch-rate | min ZMP margin | ZMP certificate | pelvis upright | fallen? |
|---:|---|---|---|---|
| 0.0 | +0.084 | **PASS** | 1.00 | no |
| 1.0 | +0.040 | **PASS** | 0.99 | no |
| 2.0 | **−0.062** | **FAIL** | 0.96 | **no** |
| 3.0 | −0.193 | **FAIL** | 0.87 | no |
| 4.0 | −0.348 | **FAIL** | 0.69 | (degrading) |

The margin is **monotone** in the perturbation and **predictive** — at pitch-rate 2.0 the ZMP has already
left support (**FAIL**) while the pelvis is still upright (0.96). This is exactly the humanoid's
`SURVIVES_PARTIALLY_NOT_STABLE` distinction, now **formalized**: a certificate for genuine capturability
(ZMP in support), not mere survival (not-yet-fallen). The certified-stable region is pitch-rate ≤ 1.

## The multi-embodiment point

The certificate is **one embodiment-agnostic core** — the same `vukobratovic_zmp` certifies the AIBO's
turn (ZMP in support ≤61°/1000) and the humanoid's balance (ZMP in support ≤ pitch-rate 1). It is the
CIP-0 stability layer shared across embodiments; the humanoid's PMP/LIPM balance core reciprocally
transfers to the AIBO (per `2026-07-27-humanoid-pmp.md`). Both now share a formal Vukobratović-ZMP /
Lyapunov capturability boundary. Follow-up: lift the shared core into `hymeko_control` so both worktrees
import one module (here it is mirrored to keep the worktrees independent).

## Files / tests

```
scenarios/humanoid/zmp_stability.py         NEW  vukobratovic_zmp + foot_support_margin + zmp_balance_certificate (CIP-0 SAFETY)
tests/test_humanoid_zmp_stability.py        NEW  5 tests: stable PASS / large FAIL / monotone / ZMP finite / certificate fn
```

`ruff` clean; humanoid tests green. CORE.YAML: none. SIMULATION. Reuses `hymeko_control` Certificate.
