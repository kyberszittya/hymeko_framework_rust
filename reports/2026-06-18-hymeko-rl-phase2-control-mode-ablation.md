# hymeko_rl Phase 2 — control-mode × architecture ablation (clean ordering; PPO still unreliable)

**Date:** 2026-06-18
**Status:** ✅ test suite repaired + control-mode ablation run; ⚠️ **PPO still degrades the
BC policy in 4/6 cells** even after the truncation fix — a *second* PPO issue remains. The
clean takeaway is the control-mode difficulty ordering. Debug journey:
`reports/2026-06-18-hymeko-rl-phase2-debug-checkpoint.md`.

## Results (BC → PPO, matched ~15k params, seed 0, 80 PPO iters, value-warmup 5)
| control mode | policy | BC reach (m) | PPO reach (m) | floor | PPO Δ |
|---|---|---|---|---|---|
| position | hsikan | **0.220** | 0.254 | 0.395 | −0.034 (worse) |
| position | mlp | 0.262 | 0.255 | 0.399 | +0.007 |
| velocity | hsikan | 0.269 | 0.364 | 0.399 | −0.095 (worse) |
| velocity | mlp | 0.296 | 0.335 | 0.403 | −0.039 (worse) |
| torque | hsikan | 0.363 | **0.347** | 0.421 | +0.016 |
| torque | mlp | 0.352 | 0.368 | 0.424 | −0.016 (worse) |

## Finding 1 (clean) — control-mode difficulty ordering
By BC reach (how well the policy imitates the closed-loop expert):
**position (0.22–0.26) < velocity (0.27–0.30) < torque (0.35–0.36)**. Torque is the
principled / real-robot interface but the hardest to imitate (tiny torque errors compound
on the light arm); position is easiest (the actuator's internal PD does the work). This is
the expected, informative ordering — and it argues that for a *working demo*, position or
velocity control reaches far better than raw torque.

## Finding 2 (honest negative) — PPO is still unreliable
Even with the truncation-bootstrap fix, **PPO improved over BC in only 2/6 cells**
(torque-HSiKAN, position-MLP) and *degraded* the other 4. So the truncation fix was
necessary but **not sufficient**; the earlier "monotonic improvement" was specific to
torque-HSiKAN. **The HSiKAN-vs-MLP architecture comparison is therefore inconclusive**
(mixed sign across modes, single seed) — no architecture claim is made.

**Prime remaining suspect (from the checkpoint, hypothesis 2): the shared actor-critic
backbone.** The value-loss gradient corrupts the actor's features; the critic warm-up only
partially mitigates it. The standard fix is **separate actor and critic networks** (or
stop-gradient the value head into the backbone). That is the next step before any
architecture or entropy-feedback comparison is meaningful.

## UPDATE — separate actor/critic networks: PPO is now STABLE (not yet a strong improver)
The shared-backbone suspect was addressed: actor and critic now have **independent
backbones** (a regression test proves a value-loss backward leaves the actor untouched).
Re-running this exact ablation (BC→PPO, 80 it):

| | worst BC→PPO degradation | spread of all 6 changes |
|---|---|---|
| shared backbone | **−0.095** (velocity-HSiKAN) | up to ±0.095 |
| separate networks | **−0.020** | all within ±0.03 |

Eval noise is ~1 SE ≈ 0.025 (24 episodes), so with separate networks **every BC↔PPO change
is within noise** — PPO *preserves* the BC policy in all 6 cells instead of wrecking it.
So the separation **fixed the instability** (the hypothesis was right about *stability*),
but the headline "improved in 2/6" is unchanged because the real remaining fact is: **PPO
does not meaningfully improve past BC in 80 iterations** — the reaching task is hard and the
gains are noise-level, not a bug. HSiKAN-vs-MLP stays inconclusive (mixed within noise).

**Phase-2 verdict:** PPO went from *catastrophically broken* (always 0.31→0.38) to *stable
and correct* (preserves BC ± noise) via four fixes this session (critic warm-up, coherent
control interface, mass-matrix torque expert, truncation bootstrap) + separate networks.
Beating BC clearly is now a **compute/scale** problem (more iters, multi-seed, maybe DAgger),
not a debugging one. 36 tests pass, ruff + mypy clean.

## What is solid now
- The full PPO + BC + control-mode-ablation infrastructure works end to end; the env is
  coherent (torque default, inverse-dynamics expert, 3 control modes); 35 tests pass, ruff
  + mypy clean.
- Verified experts: torque 0.069 / position 0.082 / velocity 0.134.
- Fixed this session: critic cold-start, the incoherent delta-on-position action, the
  missing mass-matrix scaling in the torque expert, the truncation bootstrap. PPO went from
  *always degrading* to *degrading in 4/6* — progress, not done.

## Files touched
`hymeko_rl/ppo.py` (run_ppo `control_mode`), test suite repaired + new coverage
(`test_arm_world.py`, `test_reach_bc.py`, `test_ppo.py` — control modes, per-mode expert,
truncation path). No CORE.YAML, no new dependency.

## Next (in order)
1. **Separate actor/critic networks** (hypothesis 2) — the prime remaining PPO suspect.
   Re-run this exact ablation; expect PPO to then improve over BC across cells.
2. Multi-seed once PPO is reliable, for the real HSiKAN-vs-MLP claim.
3. The algebraic-entropy-feedback test on top.
4. For a *demo*, consider leading with position/velocity control (reaches far better) while
   keeping torque as the principled-interface result.
