---
name: project-galambos-reward-fixed-rl-below-demo
description: "2026-07-05 corrected: push-controller teacher is useful scripted evidence (~0.80-0.84) and BC gives a real learned floor, but TD3+BC/SAC-style refinement degrades the clone. Next lever is Q-term/critic diagnosis, not more refine or scenario redesign."
metadata: 
  node_type: memory
  type: project
  originSessionId: d2ccb45c-9c6f-4422-a725-08dd14fe9109
---

Reports: `reports/2026-07-05-galambos-bc-only-localization.md` (part 1) and
`reports/2026-07-05-push-controller-demonstrator-hybrid-fsm.md (historical filename)` (part 2 — supersedes the handoff's framing).

**Part 1 — the stale anchor (measured):** under one `DwellMetric` protocol (difficulty 0.3, 300 steps) the OLD
pinch-carry teacher was **0.205** pooled (not 0.30 — that was a small-n / pre-physics-change number), BC clone
0.12, RL peaks 0.16–0.20 → **learner ≈ teacher**; no reward or trainer pathology. The 2026-07-03 fingertip-only
collision default broke the old teacher and it was never re-tuned; the 07-03 collab 0.40 likely predates it.

**Part 2 — the push-controller teacher (built + measured same night):** don't grasp the unstable cylinder — PUSH-CONTROLLER
it (k-tip slot fan behind the coin, closed-loop plow, brake into the zone). Delivery **0.205 → 0.80–0.84**
(3×50 eps; press_max 0.012 measured, 0.018 ≈ same). BC clone from its 27,475 dwell-filtered transitions:
median **0.34** [0.28, 0.34, 0.44], and the clone now inherits the contact behaviour (both_contact ≈0.10 vs
old ~0.01). **Correction after the July-5 refine logs:** the next lever is not more TD3+BC refine. Every
off-policy continuation seen so far degrades the BC floor. The live suspect is the Q-term/critic/update
machinery; diagnose that first with frozen-clone probes.

**Structure (user directives, all shipped):** controller = declarative hybrid FSM in
`data/robotics/galambos_push.hymeko` (vocabulary `meta_controller.hymeko`, reader
`hymeko_rl/control/controller_spec.py`) — phases/`on event to phase` transitions/gait scalars in the file,
Python only binds named guards/laws; rotation-group slot indexing (no left/right, k-arm general);
`PushObs` dataflow snapshot + `demo.events` transition log. Demo filter now dwell-consistent (§3
filter≡grading violation removed); teacher injectable, push controller default. `behaviour_clone` gained `device=`
(cuda b512 = 3.9×; b128 launch-bound — defaults unchanged).

**Monitors EXTRACTED (02:35–02:55, user order 2→3→1):** guards are now STL-robustness monitors (ρ>0 fires;
min/max combinators; `PushEvent.margin` + `last_margins` expose graded margins) — verdict-identical on
all 9 press-sweep cells; the Rust `hymeko_monitor` crate (`observe()` todo, no py bindings) is the future
seam. **Hybrid per-mode learned laws: TIED with flat BC** (0.28 [0.20–0.44] vs 0.34 [0.28–0.44]) despite
near-perfect per-mode fits (plow MSE 5e-6) → the clone gap is trajectory-level compounding + the swing
cw/ccw multi-modality, NOT mode-mixing. **Do not cite the overnight TD3+BC refine as success:** the follow-up
logs show refinement degrades delivery/contact; use it as negative evidence for the Q-term collapse audit.
RV-paper case study lives in `hymeko_monitor/paper/`. Pairs with [[project-fsm-structured-rl]] (RL in
declared modes = hybrid dynamical system). Related: [[project-hymeko-as-control-substrate]],
[[project-collab-ctde-substrate-galambos]].
