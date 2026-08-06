# Quadruped standing (Rung-2 postural plant)

SA-HSiKAN TD3 (pure, dense STAND_REWARD), 150000 vec-8 steps, 3 seeds. Success = held
upright-at-height for >=200/250 consecutive steps (DwellMetric).

| seed | stand_rate (untrained -> trained) | survival frames (u -> t) | gif |
|---|---|---|---|
| 0 | 0.0 -> 0.0 | 230.9 -> 216.5 | gifs/stand_s0.gif |
| 1 | 0.0 -> 0.02 | 248.5 -> 244.6 | gifs/stand_s1.gif |
| 2 | 0.0 -> 0.0 | 203.0 -> 244.0 | gifs/stand_s2.gif |

**stand-rate median = 0.000**, survival-frames median = 244.0/250.