---
name: feedback-throttle-rl-runs-on-shared-box
description: "Don't launch full-core RL/compute while the user may be on the machine — throttle by default or confirm the box is free; one-time affinity caps don't hold for subprocess orchestrators (children escape) — suspend instead"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 913c706b-9719-45ca-aa85-e9cfbef27d5d
---

2026-06-29: launched an overnight multi-core RL session (`run_overnight_rl.py`) full-throttle (~6+ cores,
multithreaded BLAS) while the user was still on the Windows box → they **couldn't log in / lost access**.

**Why:** CLAUDE.md's RL carve-out ("multithreading encouraged for overnight runs") assumes an **idle** machine. At
~04:00 the user was still using it. Launching heavy compute without confirming the box is free locks them out.

**How to apply:**
1. Before any multi-core/overnight compute run, either confirm the machine is free for the night, or launch it
   **pre-throttled** (limit BLAS threads via `OMP_NUM_THREADS`/`MKL_NUM_THREADS` at process start + a core cap),
   not full-throttle by default.
2. **A one-time `ProcessorAffinity`/priority cap does NOT hold for a subprocess orchestrator** — the parent keeps
   spawning new children that escape the cap (observed: capped pids, then a fresh subprocess at ~8 cores). To cap
   such a run you must cap the *parent* before it spawns AND limit thread env, or just thread-limit at launch.
3. The reliable in-place pause is **suspend**, not throttle: PowerShell P/Invoke
   `ntdll!NtSuspendProcess($_.Handle)` over `Get-Process python` → 0 CPU instantly, machine freed, work preserved;
   `NtResumeProcess` to continue. (Caveat: a >timeout suspend can trip the orchestrator's per-task `subprocess`
   timeout on resume — relaunch fresh if paused long.)
Ties [[project-morning-queue-viz-ddpg]] (the RL run lines). The user values the box being usable over a run
finishing fast — default to considerate, not greedy, with shared resources.
