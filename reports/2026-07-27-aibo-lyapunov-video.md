# AIBO Lyapunov — informative video (pursuit vs negative control, HyMeKo embedded)

**Date:** 2026-07-27 (JST)
**Branch:** `research/aibo-lyapunov-ph` (worktree `hymeko_aibo`)
**SIMULATION. NOT RL.** · **Verdict: video renders the certified state-dependent loop vs the negative control.**

---

## What

The same informative-video treatment built for the humanoid, applied to the AIBO
(22-DOF Aibo ERS-1000) Lyapunov result. Two side-by-side rollouts to an **off-axis
waypoint** (bearing 28°), framed by the HyMeKo pipeline and carrying live telemetry:

| panel | control | Lyapunov certificate |
|---|---|---|
| approach-align-stop (pursuit) | `yaw = clip(1.1·herr, ±0.8)`, drive ∝ align·dist, stop at reach | ✅ **CERT PASS** (V → 0: aligned, reached, stopped) |
| constant-forward (negative) | `yaw=0, drive=1` (no align/stop) | ❌ **CERT FAIL** (misses the off-axis goal, V stays high) |

Both use the identical waypoint, so the difference is purely the control law — the
metric-integrity check (a certificate everything passes is worthless): the reward-independent
`lyapunov_certificate` **passes the state-dependent pursuit and rejects the negative control**.

## What's in the frame

- **Top strip** — the pipeline: HyMeKo model → MJCF (freejoint + floor) → `SteeredTrotGait`
  control → Lyapunov certificate.
- **Per-panel live telemetry** — step, distance-to-goal, heading error (deg), body speed,
  V(t), running V_max, the live Lyapunov checklist (descent / converged), a **V(t) sparkline**
  (with the 0.05 convergence line), and a **top-down minimap** of the trajectory → goal
  (with the reach circle). The minimap makes the pursuit's curve-to-goal vs the negative's
  straight miss immediately legible.
- **Bottom strip** — the actual AIBO HyMeKo model excerpt (torso/rear-body via `waist(AXIS_Z)`,
  4× 3-axis legs `hip_abduct(X)→hip_flex(Y)→knee(Y)`, head/neck/tail/ears/mouth = 22 DOF),
  the gait + pursuit law, the Lyapunov V, and the certificate definition.

Representative mid-frame (step 1801): pursuit d 0.46 m, herr +10°, V 0.019, descent+conv OK;
negative herr +125°, V 2.40, descent+conv no.

## Second video — the AIBO **reaches the goal** (`aibo_reach_goal.mp4`)

A single-panel, fixed-camera cut framing the whole start→goal traverse, with the **goal
rendered in the 3D scene** (a translucent green reach-zone disk + pole via `mjv_initGeom`)
so the viewer watches the dog trot up to a visible target. It **reaches at step 2025**,
settles to **final d = 0.42 m** inside the reach zone (`GOAL REACHED` badge, minimap dot
inside the circle), **CERT PASS** — V decays to 0.039. The certified pursuit genuinely
arrives, aligns, and stops at the waypoint.

## Files

```
scenarios/aibo/render_lyapunov_video.py   NEW  (rollout + telemetry/minimap panels + HyMeKo strip)
tests/test_aibo_render.py                 NEW  (guarded render smoke)
reports/2026-07-27-aibo-lyapunov/aibo_lyapunov_compare.{mp4,gif}  NEW
```

Reuses `QuadrupedGoalEnv`, `SteeredTrotGait`, `heading_error`, `AIBOLyapunov`,
`evaluate_lyapunov` — no new dynamics, no certificate change.

## Tests / lint

`ruff` clean. AIBO render smoke + Lyapunov tests pass (6/6, 0.77 s). The render test skips
gracefully where no offscreen GL context exists (headless CI).

## Provenance

- MuJoCo model emitted by `target/release/hymeko` from `data/robotics/quadruped.hymeko`.
  Seed 0; waypoint bearing 28°, goal 0.9 m, reach tolerance 0.42 m. Peak RSS well under cap.
- Certificate **unchanged** (`AIBOLyapunov` / `lyapunov_certificate`); visualization only.

## Bottom line

The AIBO's certified state-dependent approach-align-stop loop is now a legible video: the
minimap + V(t) sparkline show it **curving to the off-axis goal and converging (CERT PASS)**,
while the constant-forward negative control **misses and does not converge (CERT FAIL)** —
the same reward-independent certificate, discriminating, with the HyMeKo model embedded.
Cross-embodiment: the identical treatment now exists for the humanoid balance and the AIBO
navigation.
