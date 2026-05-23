# HymeKo task description — design note

**Date:** 2026-05-16
**Status:** design + minimal vocabulary + one concrete example
**Companion artefacts:**
[`data/robotics/meta_task.hymeko`](../../data/robotics/meta_task.hymeko)
+ [`data/robotics/sim/dual_fanuc/handover_task.hymeko`](../../data/robotics/sim/dual_fanuc/handover_task.hymeko)

## TL;DR

**Yes — HymeKo describes tasks naturally.** The hypergraph
substrate that already carries `meta_kinematics` (links / joints /
controllers / sensors) and `meta_topic` (pub-sub / nodes) is the
same one that carries actions, behavior-tree control flow,
preconditions, and dual-arm coordination. A new `meta_task.hymeko`
library defines the vocabulary; concrete `.hymeko` files describe
specific tasks the same way they describe specific robots.

The headline property is the same as the dual-FANUC demo: one
`.hymeko` is the **single source of truth**. A task description
references the kinematic vertices (joints, end effectors, sensors)
defined by the robot's `.hymeko`; the *same* IR carries both.
Cross-domain queries become first-class:

* "Which tasks use joint J3?" → graph query.
* "Which actions have unsatisfied preconditions?" → graph query.
* "Which two actions can run in parallel given their joint
  resource sets?" → graph query (disjoint shared-resource sets).

Tasks emit to multiple downstream formats via the existing
template engine: BehaviorTree.CPP XML, PDDL, ROS 2 action
servers, MoveIt 2 planning scenes, executable Python harnesses.
This note proposes the design; one example proves the
description side parses. The emitters are each their own
follow-up template.

## Why HymeKo specifically (vs PDDL / BehaviorTree XML / SkillML)

Four reasons the hypergraph IR is the right substrate, not a
re-skin of existing formats:

1. **Shared vertex space with the kinematics.** PDDL's "objects"
   and BT.CPP's `<Action ID="...">` arguments live in their own
   namespace; HymeKo task hyperedges reference the *same vertex
   identifiers* (links, joints, sensors, frames) that the robot
   description introduces. No serialization mismatch between the
   robot description and the task description.

2. **Hyperedge arity is variable.** A pick-and-place action
   touches `(arm, gripper, object, source_pose, target_pose)` —
   5-ary. Behavior-tree `sequence` over N children is N+1-ary.
   Coordination among 2+ arms is 2..K-ary. Hypergraphs handle
   all of these natively. Conventional graph languages need
   intermediate "argument" nodes for each.

3. **Compositionality via `isa`.** The `@rev_joint: + <isa>
   elements.joint` pattern in `meta_kinematics` lets every
   revolute joint inherit `elements.joint` fields. The same
   pattern works for tasks: `pick_and_place: + <isa> sequence`
   reuses the sequence semantics, adding task-specific fields.

4. **One emit pipeline.** The
   [`template-driven codegen path`](https://...) in
   `hymeko_formats` already supports six downstream formats
   (URDF / SDF / MJCF / Mermaid / DOT / Gazebo world / SysML).
   Adding a task emitter is *another template directory*, not
   another tool. A single HymeKo file emits *both* the robot's
   URDF *and* the task's BehaviorTree XML.

## The minimal vocabulary

The `meta_task.hymeko` library (companion file, this commit)
defines four families:

### 1. Actions

Atomic motion / interaction primitives. Concrete `.hymeko`
descriptions instantiate these per-task.

| Type                | Hyperedge shape                                | Notes |
|---------------------|------------------------------------------------|-------|
| `move_to`           | `(+ effector, - target_pose)`                  | Cartesian-space goal |
| `joint_move`        | `(+ joint_set, - target_config)`               | Joint-space goal |
| `grip_open`         | `(+ gripper)`                                  | Side-effect on gripper state |
| `grip_close`        | `(+ gripper, - object?)`                       | Optional `object` says what the close is grabbing |
| `wait`              | `(+ duration)`                                 | Temporal padding |
| `apply_force`       | `(+ effector, - force_vector)`                 | Force-control primitive |

### 2. Control flow

The behavior-tree backbone. Composite, each `isa`-derived from a
base `composite` type.

| Type        | Hyperedge shape                              | Semantics |
|-------------|----------------------------------------------|-----------|
| `sequence`  | `(+ children: list[node])`                   | Run in order; abort on first failure |
| `parallel`  | `(+ children: list[node])`                   | Run concurrently; success policy in field |
| `fallback`  | `(+ children: list[node])`                   | Try in order until one succeeds |
| `loop`      | `(+ body: node, + condition: predicate)`     | While `condition` holds |

### 3. Conditions

Predicates that gate actions or compose into guards.

| Type             | Hyperedge shape                          | Notes |
|------------------|------------------------------------------|-------|
| `precondition`   | `(+ action, - predicate)`                | Predicate must hold before action |
| `postcondition`  | `(+ action, - predicate)`                | Predicate must hold after |
| `at_pose`        | `(+ effector, - pose, + tolerance)`      | Spatial predicate |
| `holding`        | `(+ gripper, - object)`                  | Possession predicate |

### 4. Coordination

The dual-arm primitive — the reason this design exists.

| Type            | Hyperedge shape                              | Semantics |
|-----------------|----------------------------------------------|-----------|
| `synchronize`   | `(+ point_a: node, + point_b: node)`         | Barrier: both must reach their point before either proceeds |
| `handover`      | `(+ from_gripper, - to_gripper, + object)`   | Sequential object transfer; emits as a 3-step sub-tree at compile time |
| `lock`          | `(+ resource)`                               | Mutex on a shared resource (a workspace, a tool) |
| `release`       | `(+ resource)`                               |  |

### 5. World objects (lightweight)

Tasks reference world objects (parts, trays, fixtures) that the
robot's kinematic description doesn't carry. The simplest design
reuses `frame` (already in `meta_kinematics.elements`) and adds a
thin marker type for "this is a manipulable thing":

| Type           | Hyperedge shape                  | Notes |
|----------------|----------------------------------|-------|
| `scene_object` | `(+ frame)`                      | A named pose with a "this is a thing" tag |
| `pickup_pose`  | `(+ frame, + scene_object)`      | "where to put the gripper to grasp this" |

## Concrete example — dual-FANUC handover

The companion file
[`data/robotics/sim/dual_fanuc/handover_task.hymeko`](../../data/robotics/sim/dual_fanuc/handover_task.hymeko)
describes the canonical dual-arm handover for the FANUC cell:

1. `fanuc_left` reaches the input tray, grips the part.
2. Both arms move to a known handover pose (sync barrier).
3. `fanuc_right` grips, `fanuc_left` releases (atomic with the
   `handover` primitive).
4. `fanuc_right` places on the output tray.

The structure is a single `pick_handover_place` sequence with
seven children, one of which is a `parallel` block containing
each arm's path to the handover point and a `synchronize`
barrier that gates the grip swap.

This `.hymeko` *parses and validates* (the
`hymeko validate` invocation in the companion file's footer
returns zero diagnostics beyond the standard `world` parent
warning). It does **not yet emit** to BehaviorTree.CPP / PDDL /
ROS 2 — those are separate templates.

## How task → executable is done (the missing emitter)

A `transforms/behavior_tree/` directory mirroring
`transforms/sdf/`:

* `transforms/behavior_tree/queries.hymeko` — extract the four
  vocabulary families above (actions, composites, conditions,
  coordination) from the IR.
* `transforms/behavior_tree/template.bt.xml` — render
  BehaviorTree.CPP XML. A `sequence` becomes
  `<Sequence>...</Sequence>`; a `move_to` becomes
  `<MoveTo target="{{target_pose}}" />`; a `handover` expands
  into the three-step sub-tree.

PDDL emission, ROS 2 action-server scaffolding, and a Python
runtime harness are analogous — each is one template directory.

The hard part is *coordination semantics*, not syntax: a HymeKo
`synchronize` between two arms maps cleanly to a BehaviorTree
parallel-with-sync-policy, but PDDL has no native sync barrier
and would need a workaround via auxiliary state predicates. This
mismatch is *features of the target format leaking through*; the
HymeKo IR remains the same.

## What this design does NOT cover

* **Geometric planning.** Actions name target poses but don't
  resolve "what trajectory gets me there." That's MoveIt 2 /
  OMPL territory, called from the emitted ROS 2 action server.
* **Force/torque profiles.** `apply_force` is a primitive; the
  full force-control profile (impedance gains, contact-search
  strategy) lives at the controller level, referenced by name.
* **Visual/sensor-conditioned tasks.** Adding a "see object, then
  pick" loop wants a `vision_detection` action and a
  `detected_at` predicate; both fit the framework but are open
  follow-ups.
* **Time / scheduling.** No deadlines / soft real-time semantics
  in this v1. `wait` is the only temporal primitive.

These are extensions, not blockers. The v1 vocabulary handles
the dual-FANUC handover end to end at the description level.

## Acceptance

* [x] `meta_task.hymeko` parses via `hymeko validate`.
* [x] `handover_task.hymeko` parses + references kinematic
      vertices defined in `fanuc_lrmate200id.hymeko`.
* [x] Design note written (this file).
* [ ] BehaviorTree.CPP emitter — separate follow-up
      (`transforms/behavior_tree/`).
* [ ] PDDL emitter — separate follow-up.
* [ ] Demo that the emitted BT runs against the dual-FANUC
      Gazebo world — separate follow-up requiring `gz_ros2_control`
      + BehaviorTree.CPP integration.

The follow-ups are each well-scoped to a single
template-directory commit, sized like
`transforms/sdf/template.sdf.xml` (~150 LOC of template + ~20
LOC of queries). None requires new core machinery.

## Bottom line

HymeKo's hypergraph IR + template-driven emit was already the
substrate for describing robots *and* their pub-sub topology
*and* (since v0.1 of `meta_topic`) their ROS topic structure.
Adding tasks is *additive vocabulary*: a new library file
+ optional new emit templates. The fundamental machinery — the
parser, the IR, the query engine, the transform pipeline — does
not change.

The dual-arm coordination case is where this pays off most
clearly: a `synchronize` hyperedge spanning two arms' execution
points is *one line* in the task description and exactly the
sort of N-ary relation hypergraphs were invented for. PDDL
expresses it indirectly via state predicates; BehaviorTree.CPP
expresses it via parallel-with-policy XML; HymeKo expresses it
directly as the relation it actually is.
