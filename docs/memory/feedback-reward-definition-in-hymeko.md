---
name: feedback-reward-definition-in-hymeko
description: "RULE (user, 2026-06-27): reward changes ALWAYS go into the .hymeko file (the declarative source) — never an in-memory weight override that leaves the .hymeko stale."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 913c706b-9719-45ca-aa85-e9cfbef27d5d
---

**Rule:** when varying ANY reward, modify the **`.hymeko` reward definition file** directly
(`data/robotics/galambos_task.hymeko`, `pick_place_task.hymeko`, `meta_reward.hymeko`) — edit the arc weights
`(+ term weight, …)` / add terms there. Do **not** vary the reward via an in-memory weight override (e.g. a
sweep tool that bypasses the file), which leaves the `.hymeko` inconsistent with what is actually trained.

**Why:** the `.hymeko` is the single authoritative source of the reward (the MDSD / single-source thesis —
[[project-mdsd-reuse-and-docs]], [[project-xprofile-instance-refs]]). Training reads it via
`RewardSpec.from_hymeko`, so editing the file keeps the declared reward and the trained reward in lockstep;
git tracks each variation as the audit trail and makes revert clean.

**How to apply:** reward iteration loop = edit the `.hymeko` weights/terms → train (reads from it) → eval →
keep or `git checkout` the file. The `.hymeko` must always reflect the active/best reward. A quick in-memory
scan to *find* a good weight is fine, but the chosen value is then **written into the `.hymeko`** before any
run that counts. Reports cite the `.hymeko` reward bundle, not loose override flags.
