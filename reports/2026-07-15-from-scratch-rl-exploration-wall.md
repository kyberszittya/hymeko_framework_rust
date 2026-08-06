---
title: From-scratch RL exploration wall + the pick-place RL-space capstone
date: 2026-07-15
scope: the one untested RL lever (from-scratch, non-residual) — and the full map of RL on pick-place
status: experiment (negative, confirmatory) + capstone synthesis
core_touched: none
---

# From-scratch RL — the exploration wall, confirmed across optimizers

**Question.** After residual RL (F-PP-009/010/011) and precision (cached 2026-07-14) were characterized, the one RL
lever left was **from-scratch, non-residual** RL — not bound to the base setpoint, so the only path with headroom in
principle. Cached from-scratch **PPO** (2026-07-09/10) failed the exploration wall (never grasps; "reward can't
bootstrap pre-grasp"; `pick_place_ppo` warm-starts from BC *because* "from-scratch PPO would never discover a grasp
by random exploration"). Untested: off-policy **SAC** (replay + AUTO-entropy exploration) + the dense aligned reward.

**Result (from-scratch SAC, 80k steps, symmetric critic, `pick_place_aligned.hymeko` reward, live probe):**

```text
UNTRAINED   reach=0.00 grasp=0.00 lift=0.00 place=0.375
@ every eval (146s … 3491s):  reach=0.00 grasp=0.00 lift=0.00 place≈0.38
FINAL n=16  reach=0.000 grasp=0.000 lift=0.000 place=0.375
```

**Grasp never leaves 0.00. Reach never leaves 0.00.** SAC's replay + entropy exploration + a dense reward did **not**
clear the reach→grasp wall — it did not even learn to *approach*. The `place≈0.375` is a **metric-inflation
artifact** (flagged before the run): boxes spawn near the target and an idle arm trips `placed_stable`; grasp=0.00
proves nothing was picked-and-placed. Judging on grasp (the un-fakeable metric) the verdict is unambiguous.

**Conclusion.** The from-scratch exploration wall is **optimizer-agnostic** — PPO (cached) and SAC (here) both fail
to discover a grasp by random exploration on the 7-DOF contact task. This is expected: off-policy replay cannot learn
a grasp that never appears in the buffer, and an asymmetric critic stabilizes *value learning*, not *exploration*
(symmetric SAC never even reached, so asymmetry cannot be the fix). A vastly larger budget is not justified: the
cached PPO failed with more tuning, reach never started here, and the scripted expert already dominates.

*Caveat (§3):* single 80k SAC run. But it converges with the cached multi-run PPO failure and the theoretical
argument; the claim is "from-scratch does not learn in a practical budget," not a proof of impossibility.

---

## Capstone — the pick-place RL space, fully mapped

| RL approach | setpoint-bound? | outcome | evidence |
|---|---|---|---|
| **Residual PPO** (structured/mode-gated) | yes | **holds** base 0.875, cannot beat | cached 2026-07-14; F-PP-011 |
| **Residual SAC** (off-policy) | yes | collapses to 0.458 (critic over-estimation) | **F-PP-009** |
| Residual clone (FF/history/recurrent) | yes | 0.75–0.83, < base (execution-hard at contact) | **F-PP-010** |
| Mode-gated residual | yes | ceiling = base (grasp-separable, lift not) | **F-PP-011** |
| Precision (recurrent clone) | — | closed: 5–6 cm, worse than base; 2.16 cm un-clonable | cached 2026-07-14 |
| **From-scratch PPO** | no | exploration wall: never grasps | cached 2026-07-09/10 |
| **From-scratch SAC** | no | exploration wall: never reaches/grasps | **this report** |
| Imitation→RL (BC warm-start → RL) | — | = the deployed base (0.875) | `pick_place_ppo`, hybrid-DAgger |

**Two structural walls bound every learned policy:**
1. **Setpoint wall** (residual RL): a bounded correction on the frozen base can only *hold* base, never beat it — and
   the oracle's 1.000 needs the un-clonable grasp/lift discontinuities.
2. **Exploration wall** (from-scratch RL): random exploration cannot discover a grasp on a 7-DOF contact task, so
   from-scratch RL never gets a signal — optimizer-agnostic (PPO + SAC).

Between them, **imitation→RL is the only viable learned path, and it *is* the deployed base (0.875 / 4.69 cm).** No RL
lever with real headroom remains. The scripted **v3 expert (0.958 / 2.16 cm) dominates every learned policy** on both
success and precision.

**Deployment recommendation:** for best performance, deploy the scripted v3 expert. For a learned/differentiable
policy: the FF-DAgger base (0.875 / 4.69 cm, best learned precision) or the LSTM-BC clone (0.917 / 6.09 cm, best
learned reliability). Residual RL can be added only to *hold* base safely (structured + mode-gated + conservative
critic), never to beat it.

## Provenance
- Branch `integration/fanuc-pick-place-canonical`, audit checkpoints `6b90ca8`/`b7e85f0`/`a4e6a9a`.
- Experiment: `scratchpad/from_scratch_sac.py` (NormActionEnv [-1,1]→[lo,hi]; `build_sac` symmetric AUTO-alpha;
  `train_sac`; `pick_place_aligned.hymeko` reward; live reach/grasp/lift/place probe). No warm-start.
- No CORE. No persistent state mutated. No kato15 (away). Local CPU, 80k steps ≈ 58 min wall.
