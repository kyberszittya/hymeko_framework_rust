# §0 audit — is "HSiKAN loses to MLP" a wiring bug? (verdict: NOT a wiring bug)

**Date:** 2026-06-26 · **Author:** Aiko (Claude Code) for Dr. Csaba Hajdu
**Context:** the #1 open question (`project-hsikan-loses-possible-bug`, handoff `2026-06-26-session-handoff.md`):
the uniform "HSiKAN underperforms a params-matched MLP on every RL task" pattern was suspected to be an
obs/hypergraph **wiring bug** ("the policy fights its own structure"), not an architecture verdict.

## Method

Task-by-task audit of the obs→HSiKAN-structure mapping (the §0 epicentre), order
cartpole → collaborative → arm → pick-place → quadruped. Per task: vertex labels + signed adjacency, per-vertex
in-degree (isolated/degenerate-row check), feature↔vertex alignment, all-zero feature rows, and a behavioural
HSiKAN forward (`build_policy` + `hsikan_diagnose`) on a real reset obs.

## Results — every task is wiring-clean

| Task | N (vtx) | Topology | Isolated / all-zero rows | Forward finite + diagnose | Verdict |
|---|---|---|---|---|---|
| cartpole | 2 | cart↔pole chain | none | ✅ (and learns **200/200**) | clean |
| arm6dof | 7 | base→link_0..4→tool chain | none | ✅ | clean |
| quadruped | **9** (not 14) | torso → 4×(thigh→shin), **branching correct** | none | ✅ | clean |
| pick-place | 9 | base→link_0..4→tool→{finger_l,finger_r} | none | ✅ | clean |
| fanuc pick | 9 | same as pick-place | none | ✅ | clean |

Notes:
- **Quadruped (the prime suspect) is 9 vertices, not 14** (the handoff figure was wrong). Its **branching
  topology is correctly encoded** — the torso has exactly 4 down-chain children (`thigh_fl/fr/bl/br`), each
  `thigh→shin` — i.e. the exact structure C3 was meant to reward. No degenerate rows (the §0 #2 "all-zero
  hub" suspect is **absent**); degrees `[1,2,1,2,1,2,1,2,1]` are correct chain/leaf boundaries.
- The feature scatter (`jnt_bodyid − 1` → child vertex) matches `HypergraphState.from_mjcf`'s convention in
  every env (`_scatter_joint_scalar`, the inline cart-pole/quadruped loops) — features align with vertices.
- Cartpole learns perfectly (HSiKAN **200/200**, MLP 200/200), confirming the backbone runs end-to-end.

**Separately:** the collaborative task had a genuine **reward** bug (the dense approach term measured the
nearest body origin / immovable base, not the fingertip) — found, fixed, reported
(`2026-06-26-galambos-fingertip-reward.md`). That is a reward-shaping defect, not a graph-wiring bug, and it
shapes both backbones equally (so it does not explain the HSiKAN-vs-MLP gap).

## Verdict

**The wiring-bug hypothesis is falsified across every audited task.** No obs/hypergraph misalignment, no
degenerate adjacency rows, no NaN/inf, healthy backbone forward; the graphs faithfully encode the kinematics.
"HSiKAN loses/ties to MLP" is **not** a wiring bug.

Two hypotheses remain, and they split cleanly:
- **(a) genuine parity / structure-not-load-bearing** — the robot *objectives* (reach a goal, balance,
  deliver a coin) are **not graph-structural properties**, so message-passing over the kinematic graph has no
  leverage a params-matched flat MLP (which sees the same obs) lacks. Aligns with
  `project-galambos-hsikan-tie-rootcause` (the objective isn't a graph property).
- **(b) silent-wrong-but-finite backbone forward bug** — one that `hsikan_diagnose` (which catches blow-ups,
  not finite-but-wrong outputs) cannot detect.

## Decisive next step (the discriminating test)

Build a **toy task whose answer IS a graph property the MLP structurally cannot see** (e.g. reward = a cycle
parity, or reaching along a hidden path defined over the hypergraph). Then:
- HSiKAN **wins** there → the backbone is correct, and the robot-task ties are **real** (verdict **a**): the
  structural prior simply isn't load-bearing for these objectives — not a bug.
- HSiKAN **loses** there → the backbone forward is silently buggy (verdict **b**) — localise in
  `hymeko_neuro/core/backbone.py` against a hand-computation.

One test collapses the question. This is the handoff's own proposed discriminating test, now the *correct*
next step precisely because the wiring is cleared: "a real architecture wins somewhere" must be tested where
structure is load-bearing, not on a robot task where it isn't.

## Provenance

- Git: branch `fix-hsikan`; tree dirty (prior-session + this session's changes). Audit is read-only
  introspection (no persistent state mutated); cartpole numbers from the SAC run in
  `2026-06-26-galambos-fingertip-reward.md`. Env: Windows 11, Python 3.12, mujoco/torch CPU, seed 0.
