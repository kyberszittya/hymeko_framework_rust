# Closed-loop capturability MPC — receding-horizon feedback rejects disturbances

**Date:** 2026-07-28 (JST) · **Branch:** `research/aibo-lyapunov-ph`
**SIMULATION.** · **Verdict: `CLOSED_LOOP_MPC_REJECTS_MID_RUN_PUSH`.**

The open-loop running plan is a fixed force profile; a mid-run push drifts it. `ClosedLoopRunningMPC`
re-solves the stance force **from the measured state each stride** (receding horizon), driving the
centroid back onto the nominal running orbit while keeping the capturability Lyapunov bounded.

## Result — a +0.4 m/s push at stride 3

| controller | vx after push | recovers to 0.6? | V_cap max |
|---|---|---|---|
| **closed-loop MPC** | 0.97 → **0.60 next stride** (err → 0.000) | ✅ | 0.030 (< 0.15) |
| open-loop plan (same push) | 1.00 | ❌ drifts | — |

`closed_loop_mpc.png`: the closed-loop speed spikes then returns to target in one stride while the
open-loop stays drifted; the capturability Lyapunov stays a bounded sawtooth through the push.
Feedback MPC rejects the disturbance; the fixed plan does not.

## Files

```
scenarios/aibo/closed_loop_mpc.py     NEW  (ClosedLoopRunningMPC: replan-each-stance receding horizon)
scenarios/aibo/render_closed_loop.py  NEW  (closed-loop vs open-loop under a push + V_cap plot)
tests/test_aibo_closed_loop_mpc.py    NEW  3 tests (rejects push, V_cap bounded, open-loop drifts)
reports/2026-07-27-aibo-hop/closed_loop_mpc.png  NEW
```

## Bottom line

The receding-horizon capturability MPC closes the loop: it rejects a mid-run push (vx → target in
one stride) where the open-loop plan drifts, keeping the run inside the capturability region. This
is the feedback layer over the certified centroidal plan — the honest "MPC" the user asked for.
