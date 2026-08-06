# From Clearance-Aware Waypoints to Hypergraph-Based Parallel Planning

**2026-07-07 · design roadmap (documentation only).** Origin: the FANUC pick-place v2 clearance failure. Scope
of this note: the planning *roadmap*. It does **not** implement the v2 hotfix (the parallel track owns that);
it records the architectural insight and stages the planner from a minimal deterministic waypoint sequence to a
HyMeKo hypergraph planner.

## The insight — a missing planner layer

The v2 clearance problem is not a tuning miss; it exposes an **architecture gap**: the expert is a **one-target
IK controller**, not a **planner**. A single `env._ik.solve(q_now, far_target, down=True)` from `arm_home`
descends into a folded local minimum and the gripper dips below the table *en route* — even though the final
target pose is valid (verified: `grasp_z+0.06` is IK-valid and clears the table; the path to it is not).

> **Path validity ≠ final-pose validity. A valid IK target is not enough — the whole path to the target must be
> valid.**

The fix is therefore not "a better hover height" but **introducing a planning layer** that *generates* a safe
sequence of waypoints and *validates every segment* step-by-step.

## Immediate scope — a MINIMAL deterministic clearance-aware waypoint planner (v2 hotfix)

Not A*/RRT*/hypergraph yet. A hand-designed, deterministic waypoint sequence with per-segment validation. The
purpose is narrow: generate a safe waypoint chain, validate each segment, keep the gripper/fingers at positive
clearance during transit, and prevent table-edge strikes before the gripper is above the object.

**Segments (states):**
1. `HOME_SAFE_RISE` — rise straight up from home to a safe transit altitude.
2. `TRANSIT_ABOVE_TABLE` — lateral move at the transit altitude, clearance maintained.
3. `ABOVE_OBJECT_ALIGN` — arrive above the object still clear of the table.
4. `VERTICAL_DESCENT` — descend straight down only once centred over the object.
5. `GRASP` — close + settle.
6. `LIFT` — straight up from the captured grasp xy.
7. `PLACE_TRANSIT` — traverse at clear height to the target.
8. `PLACE_DESCEND_RELEASE` — lower and release.

**Per-segment validation (every segment must pass):**
- IK success at each step,
- stepwise path tracking (short hops, un-folded — see the two levers below),
- positive finger/table clearance throughout,
- no gripper/table-edge strike,
- no finger/table contact during transit,
- high enough lift/place.

**Two levers that make the path valid** (from the waypoint-planner design note): (A) **IK seeding** — seed each
solve from a good, un-folded config (home / high-elbow) rather than the drifting `q_now`; (B) **short Cartesian
waypoints** — each solve is a short hop from the current pose, so the solver never folds. Measured constraint:
the valid clearance-hover ceiling is ~`grasp_z+0.06` (above → self-collision); prefer a radius-scheduled
transit altitude.

**Acceptance gate (UNCHANGED) → workflow.** No table-edge strike before over-object · no finger/table transit
contact · positive minimum clearance during transit/approach · high enough lift/place. **Only then:** regenerate
demos (v2) → recompute expert ceiling (v2) → BC v2 → DAgger v2 (`algorithm "dagger";` TrainingSpec path). Label
`pick_place_v2_clearance_aware`; never overwrite `v1_dirty`. **No BC/DAgger until the expert passes the gate.**

## Future planner roadmap

**Suggested planner stack (Phases 1–5):**
1. deterministic clearance-aware waypoint planner (the immediate v2 hotfix above),
2. A* over validated waypoint hyperedges,
3. RRT* / RRT-Connect for continuous path segments,
4. HyMeKo hypergraph planner with parallel edge expansion,
5. **RL-bounded hypergraph search** — RL learns expansion priority, cost-to-go, feasibility estimates, and
   sampling-budget allocation, **on top of** (never replacing) the hard validators.

### Phase 2 — A* over validated waypoint hyperedges
- **nodes** = candidate safe Cartesian/joint waypoints,
- **edges** = validated IK/path segments (an edge exists iff the segment passes the per-segment validation above),
- **cost** = path length + clearance penalty + joint motion + failure risk,
- A* selects the least-cost **safe** waypoint chain. Generalises the hand-designed 8-state sequence into a search
  over candidate waypoints — useful when the deterministic chain is too rigid for cluttered / edge-case scenes.

### Phase 3 — RRT* / RRT-Connect for continuous path segments
- continuous joint-space or task-space sampling,
- collision/clearance checked **per edge**,
- for when hand-designed waypoints (and even the A* graph) are insufficient — dense clutter, tight envelopes,
  non-monotone approaches.

### Phase 4 — HyMeKo hypergraph planner (with parallel edge expansion — ties to the declarative-substrate thesis)
- **nodes/hyperedges** represent robot state, object state, constraints, phases, and resources;
- **hyperedges encode actions** — `rise`, `transit`, `align`, `descend`, `grasp`, `lift`, `place` — as first-class
  declared entities (the same hypergraph substrate the framework already uses for scene/reward/strategy);
- **edge validity** = IK feasibility + collision + clearance + contact constraints + task preconditions
  (the per-segment validation, promoted to a declared edge predicate);
- **expansion is parallelisable** over candidate hyperedges / waypoint samples (fan out edge-validity checks);
- the result is a **MoveIt-like planning layer augmented by HyMeKo's hypergraph representation** — planning
  expressed in, and validated against, the declarative substrate, consistent with
  `project-hymeko-as-control-substrate`.

## Phase 5 — RL-Bounded Hypergraph Search

**Long-term objective.** The goal is **not to replace classical planning with reinforcement learning, but to place
learned guidance inside a validated hypergraph search loop.** In this architecture, **symbolic and geometric
validators preserve correctness**, while learned models improve **expansion order, sampling-budget allocation, and
recovery selection**. A learned model may make the search *slower* when wrong — **but it must not make an invalid
path legal.** This is the single invariant the whole phase is built around.

**Core idea.** The hypergraph planner should not rely only on blind expansion. RL can learn to **bound and
prioritise** the search — but it must **not** replace the hard validators. RL *focuses* the planner; it does not
own its correctness.

**Architecture.**

1. **Hypergraph planner.**
   - nodes / hyperstates represent robot state, object state, phase, contact state, clearance state, and remaining
     budget;
   - hyperedges represent actions — `rise`, `transit`, `align`, `descend`, `grasp`, `lift`, `place`, `recover`;
   - each hyperedge carries **preconditions**, **effects**, and **hard validity checks**.

2. **Hard validators** (the non-negotiable layer — same predicates as Phases 1–4):
   - IK success,
   - collision-free path,
   - positive clearance,
   - no forbidden contact,
   - task-specific phase constraints.

3. **RL-learned search guidance** (models trained *around* the validators, never instead of them):
   - `V(h)` — expected success / cost-to-go from hyperstate `h`,
   - `Q(h, e)` — value of expanding hyperedge `e` from `h`,
   - `π(e | h)` — hyperedge expansion prior,
   - `P_valid(h, e)` — predicted validity / feasibility,
   - `B(h)` — suggested sampling / planning budget.

4. **What RL is used for:**
   - frontier ranking,
   - beam-search ordering,
   - budget allocation,
   - early rejection of low-probability branches,
   - choosing which hyperedges to expand in parallel,
   - deciding when to fall back to RRT* / A* (Phases 2–3).

5. **What RL is NOT used for.** RL is **not the final safety oracle.** It may guide the search, but **every
   proposed edge/path must still pass the hard geometric and physics validation** of layer (2).

**The load-bearing distinction — soft bound vs hard proof.**
- **Soft bound (RL):** "this edge is unlikely or expensive" → expand it *later*, or with *less* budget. Reversible,
  advisory, never removes a valid branch permanently.
- **Hard proof (validator):** IK / collision / clearance / contact validity, *confirmed*. This is the only thing
  allowed to declare an edge legal or illegal.

**Initial safe design.** Use RL **only as a priority function, not as a pruning oracle.** The planner remains
complete over the validated edge set; RL merely reorders and budgets exploration. Nothing an RL model says can, on
its own, make a valid path unreachable.

**Later research direction — uncertainty-aware / calibrated learned bounds:**
- conservative thresholds,
- fallback expansion when uncertainty is high,
- **no irreversible pruning unless confidence is validated,**
- calibration checks and comparison against exhaustive / classical-planner baselines (a pruned run must not miss
  a solution the exhaustive planner finds).

**Key principle.** *RL accelerates and focuses the planner; it does not replace the planner's correctness checks.*
The safety property (only validated edges are legal, the planner stays complete over them) is a **hard invariant**,
independent of how well or badly the learned models perform — a mistrained `π`/`Q`/`P_valid` costs *speed*, never
*correctness*.

## Why this is the right arc

The clearance bug is a small symptom of a real architectural gap — **control without planning**. Staging the fix
as *minimal waypoints → A* graph → RRT* → hypergraph planner → RL-bounded hypergraph search* turns a hotfix into a
principled planning layer whose endpoint is native to HyMeKo: actions and their feasibility as declared
hyperedges, validated per edge, expanded in parallel, and — finally — *focused* by learned search guidance that
never overrides the validators. The manipulation planner becomes another view of the same declarative substrate
rather than a bolted-on external planner, and RL enters as an accelerator layered on top of a provably-correct
core rather than as the safety authority itself.

## CIP and DirectLiNGAM for RL Diagnostics

> **Orthogonal layer — NOT the planner.** CIP / DirectLiNGAM is **not** the low-level waypoint planner
> (Phases 1–4) and **not** the RL search-guidance of Phase 5. It is the **causal diagnostic and
> experiment-prioritization layer** that sits *above* the whole stack: it decides *which* failure modes,
> variables, and interventions deserve experimental budget first — especially for RL and imitation-learning
> failures. Keep it distinct from the planner in every discussion.

**Purpose.** Use **CIP — Causal Information Prioritization** — to decide which variables, failure modes, and
interventions deserve experimental budget first. Use **DirectLiNGAM** *only* as an **exploratory**
causal-discovery tool over logged continuous variables. **Do not present DirectLiNGAM as proof of causality by
itself.**

**Candidate variables.**
- **Coin-collab:** delivery · final coin-to-target distance · dist_delta · coin velocity toward target ·
  fingertip contact · both-contact · approach error · contact error · push error · brake error · phase ·
  BC action error · DAgger round · architecture · seed.
- **Pick-place:** lift success · place success · final object-to-target distance · gripper/table contact ·
  finger/table contact · minimum clearance · first-collision timestep · over-object timestep · grasp success ·
  BC loss · DAgger round · selected checkpoint · seed.
- **RL-specific:** critic loss · Q mean · Q drift · TD error · actor BC deviation · behavior-cloning
  regularization strength · reward components · contact metrics · off-distribution action magnitude ·
  success/lift/place/delivery.

**CIP workflow.**
1. Log structured rollout metrics.
2. Group variables into: **task-outcome** · **phase** · **contact/clearance** · **learning** ·
   **architecture/method**.
3. Run DirectLiNGAM on **suitable continuous variables only**.
4. Handle categoricals (method, architecture, stage, seed) carefully — **prefer stratification**, or use them as
   **grouping variables**; **do not naively mix them into a linear causal model.**
5. Use the discovered candidate causal ordering to **prioritize interventions**.
6. **Confirm any proposed causal claim with controlled ablations.**

**Hard rule.** *DirectLiNGAM proposes candidate causal structure; controlled interventions decide whether the
structure is real.* (The causal-discovery instance of the framework's discriminating-test rule — a proposed
ordering is a hypothesis, not a conclusion.)

**Expected outputs.** candidate causal ordering · strongest edges / dependencies · failure-mode ranking ·
recommended next intervention · ablation plan.

**Worked examples.**
- **Imitation chain** — if DirectLiNGAM suggests
  `approach error → both-contact → target-directed velocity → distance reduction → delivery`,
  CIP prioritizes interventions that **reduce approach error and improve both-contact** *before* trying new RL
  algorithms.
- **RL chain** — if the chain is
  `Q drift → actor deviation from BC → contact collapse → delivery collapse`,
  the intervention priority is: **freeze RL → strengthen behavior regularization → use DAgger / residual RL →
  or restrict RL to phase-gated residual corrections.** (Consistent with the on-record TD3+BC value-drift
  collapse: DAgger stays in the imitation regime, so there is no Q to drift.)

**Status.** Roadmap/handoff placeholder only. **Do not run CIP/DirectLiNGAM now unless explicitly requested.**
Science sibling (distinct concern — the *contribution*, not the diagnostic use): signed-hypergraph LiNGAM
(LiNGAM-SH), `project-kato-lingam-cip-hymeko`.

## Coordination

- The **minimal deterministic waypoint planner** (immediate v2 hotfix) is owned by the **parallel track**.
- The **full A*/RRT*/hypergraph planner is NOT built inside the v2 hotfix.**
- This note is the roadmap; this thread is design/documentation only unless implementation is explicitly handed
  over.
