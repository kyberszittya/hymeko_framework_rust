# Quadruped standing (Rung-2 postural plant)

SA-HSiKAN TD3 (pure, dense STAND_REWARD), 60000 vec-8 steps, 3 seeds. Success = held
upright-at-height for >=200/250 consecutive steps (DwellMetric). Base = free (the torso can fall).

| seed | stand_rate (untrained -> trained) | survival frames (u -> t) | gif |
|---|---|---|---|
| 0 | 0.0 -> **0.24** | 230.9 -> 247.3 | gifs/stand_s0.gif |
| 1 | 0.0 -> 0.0 | 248.5 -> 226.8 | gifs/stand_s1.gif |
| 2 | 0.0 -> 0.0 | 203.0 -> 204.7 | gifs/stand_s2.gif |

**stand-rate median = 0.000** (seeds 0.24 / 0.0 / 0.0), survival-frames median = 226.8/250.

## Reading

- **The scenario works and standing is learnable**: seed 0 rises from a 0.0 untrained baseline to **0.24**
  sustained-stand rate (held upright-at-height >=200 consecutive steps in ~1/4 of eval episodes). This is a
  positive existence result for the Rung-2 postural plant.
- **Not yet robust across seeds** at 60k steps: seeds 1 and 2 stay at 0.0. High seed variance = undertrained /
  unstable, not a broken task (single-seed is a point estimate, not a verdict, CLAUDE.md).
- **Survival is near-saturated** (203-248 frames even untrained): the free base rarely fully inverts under
  `flip_cos=-0.2`, so survival does NOT discriminate; **stand_rate is the discriminating metric** (0.24 vs 0.0).
- **Next levers** (measure, don't declare): more steps (100k-200k), a PD-hold-q0 BC warm-start -> TD3+BC
  (galambos anti-collapse recipe), or a tighter `flip_cos` so falling is easier to trigger (sharpen the signal).
  Restoring the GPU 5-6x (the `torch.compile` CUDAGraphs crash on the quadruped) makes longer runs cheap.

Artifacts: `policies/stand_s{0,1,2}.pt` (source of truth), `gifs/stand_s{0,1,2}.gif` (seed 0 = the standing
policy; watch it hold upright), `results.json`.
