# Handoff packet — Mac transition (2026-07-07)

**Purpose:** carry state across the machine transition. **Documentation only — no experiments run, no v2 code
edited to produce this note.** All numbers below are quoted from frozen reports on disk; nothing here is a new
measurement.

---

## 1. Current frozen branches (canonical numbers + report anchors)

| # | branch | frozen result | report |
|---|---|---|---|
| 1 | **coin-collab (CTDE/TD3), HSiKAN vs MLP** | scripted 2-fingertip **0.854** (best controller); HSiKAN BC0 **0.417** > MLP BC0 **0.208**; TD3 refinement **degrades both** (HSiKAN 0.417→0.292, MLP 0.208→0.125). Architecture signal PASS; learned-RL success FAIL. | `reports/2026-07-05-coin-collab-mlp-vs-hsikan.md` |
| 1b | **coin-collab v2b graded-contact BC0** (parallel track, separate scene) | verdict by `fingertip_dominant`: scripted ceiling **0.792**; **HSiKAN BC0 0.361** > MLP BC0 0.194 (both clean, exploit 0.0); body-shove 0.542 raw is an EXPLOIT, not success. DAgger may proceed only after this BC0. | `reports/2026-07-07-v2-bc0.md` |
| 2 | **pick_place_v1_dirty_expert** | expert ceiling **0.875 place / 0.958 lift** (n=24) — but classification **C, physically dirty**: 32/32 strike the table edge in transit (~step 51, over-object only ~step 275), 32/32 negative min clearance (~2.6 cm below tabletop), no clean episode. Succeeds *despite* an invalid trajectory. | `reports/2026-07-06-pick-place-clearance-forensics.md` |
| 3 | **pick-place declarative DAgger** (v1_dirty) | DAgger best-checkpoint **median 0.792 / mean 0.833** (n_eval=24, 3 seeds); BC0 floor 0.542; TD3+BC **0.0 (value-drift collapse)**. Best-ckpt is the deployable artifact; final ckpt (0.792/0.736) reported separately. Substantially closes the cloning gap; does not cleanly match ceiling. | `reports/2026-07-06-dagger-declarative-strategy.md` |
| 4 | **clearance forensics** | classification **C — physically dirty, requires trajectory revision** (N=32). Evidence frozen. | `reports/2026-07-06-pick-place-clearance-forensics.md` |
| 5 | **v2 waypoint-planner design** | static IK clearance poses DO exist (hover ≈ `grasp_z+0.06`); blocker is the rate-limited DLS-IK **path** (dips ~11 cm from `arm_home`), not the static reach. Fix = waypoint plan + IK seeding, not a hover-height tweak. **Gate NOT passed.** | `reports/2026-07-06-pick-place-v2-waypoint-planner-design-note.md`, `…-v2-reach-envelope-verification.md`, `…-pick-place-clean-expert-v2-attempt.md` (parallel track) |
| 6 | **HyMeKo planner roadmap** | 5-phase stack: deterministic waypoints → A* over validated waypoint hyperedges → RRT*/RRT-Connect → HyMeKo hypergraph planner → **RL-bounded hypergraph search** (RL prioritises, validators keep correctness). | `reports/2026-07-07-hymeko-planner-roadmap.md` |

> **Do not confuse the two "v2" lines.** FANUC **pick-place v2** = the clearance-aware waypoint planner (this
> handoff's subject; gate NOT passed). Coin-collab **planar-grasp v2b** (`2026-07-07-v2-*`) = a separate
> graded-contact scene owned by the parallel track.

---

## 2. Current ownership

- **Parallel track owns the v2 implementation** (`expert_version=2`, `_expert_action_v2` in
  `hymeko_rl/env/pick_place_env.py`).
- **This thread is design/documentation only** — no v2 code edits unless implementation is explicitly handed over.
- **No BC/DAgger on pick-place v2 until expert v2 passes the clearance gate.**

---

## 3. Next executable tasks (in order)

> **Phase-1 plan bundle now on disk (design-only, 2026-07-07):**
> [docs/plans/2026-07-07-pick-place-v2-waypoint-planner/](docs/plans/2026-07-07-pick-place-v2-waypoint-planner/) —
> `README.md` · `waypoint_state_machine.md` (8 segments, entry/exit/failure/diagnostics each) · `clearance_gate.md`
> (N=24/32 acceptance) · `implementation_notes.md` (files, API, keep-v1-frozen, coin-collab-v2b disambiguation) ·
> `next_commands.md` (exact gated commands a–e). Implementation is parallel-track-owned; the bundle is the spec.

1. **Implement the deterministic clearance-aware waypoint planner v2** — 8 segments: HOME_SAFE_RISE →
   TRANSIT_ABOVE_TABLE → ABOVE_OBJECT_ALIGN → VERTICAL_DESCENT → GRASP → LIFT → PLACE_TRANSIT →
   PLACE_DESCEND_RELEASE. Levers: IK seeding from an un-folded config + short Cartesian hops.
2. **Expert-only clearance gate** — N=24/32: no table-edge strike before over-object; transit finger↔table ≈ 0;
   positive minimum clearance during approach; lift/place stay high (place ≥ 0.80, lift ≥ 0.90).
3. **Regenerate v2 demos** (only after the gate passes) — label `pick_place_v2_clearance_aware`.
4. **BC v2** — new baseline; do not compare against v1_dirty numbers as if the same benchmark.
5. **DAgger v2** — `algorithm "dagger"` via TrainingSpec; best-checkpoint deployable.
6. **(Optional) route Galambos through TrainingSpec** — unify the coin-collab training loop under the declared
   `algorithm` field, same dispatch as pick-place.

Gate ordering is strict: step *k* does not start until step *k−1* passes.

---

## 4. Mac transition checklist

- **Clone / sync the repo** to the Mac; confirm the working tree matches (branch, uncommitted v2 WIP owned by the
  parallel track).
- **Verify the Python environment** — MuJoCo importable, torch present; run the unit tests before anything else.
- **Verify SSH to kato15** — `ssh kato15` should be passwordless (ed25519 key installed; password retired). If the
  key did not travel, re-install the public key on kato15 first.
- **Remote MuJoCo render runs: use an explicit `env MUJOCO_GL=egl <cmd>`** on kato15 (RTX 6000 Ada, headless — no
  display). Do not rely on it being exported.
- **Do not assume remote shell profile persistence** — kato15 is tcsh + noclobber; each non-login `ssh host cmd`
  starts fresh. Set `MUJOCO_GL`, `PYTHONPATH`, and any venv activation *inline per command* (or via a heredoc);
  `>>` / `2>&1` misbehave under noclobber (use `|&`, `tee -a`).
- **Run only smoke tests locally** (Mac) — unit tests, `--smoke` entrypoints, bounded diagnostics.
- **Run heavy experiments on kato15** (multi-seed training, DAgger rounds, long rollouts) — not on the Mac.

---

## 5. Important invariants

- **`v1_dirty_expert` numbers are never overwritten** — re-version, do not overwrite. The prior expert / BC /
  TD3+BC / DAgger pick-place numbers are `v1_dirty` results only, not clean pick-place learning.
- **v2 is a new benchmark version** — its expert ceiling, BC baseline, and DAgger result are measured fresh; they
  do not inherit v1 numbers.
- **Static IK validity is not enough; path validity is required** — a valid final IK target does not imply a valid
  trajectory to it. The whole path must clear.
- **RL guidance can prioritise the hypergraph search but cannot replace the hard validators** — symbolic/geometric
  validators preserve correctness; a wrong learned model may make the search slower, but it must never make an
  invalid path legal.

---

## 6. CIP / DirectLiNGAM — causal diagnostic & experiment-prioritization layer (do not lose)

**Orthogonal to the planner.** Decides *which* failure modes / variables / interventions get experimental budget
first — especially for RL and imitation-learning failures. Not the waypoint planner, not the Phase-5 search
guidance.

- **CIP (Causal Information Prioritization):** rank interventions by causal leverage before spending budget.
- **DirectLiNGAM:** exploratory causal discovery over logged **continuous** rollout variables only — proposes
  **candidate** structure, **not** proof. Categoricals (method / architecture / stage / seed) → **stratify or
  group, never mix into the linear model**.
- **Variable groups:** task-outcome · phase · contact/clearance · learning (critic loss / Q mean / Q drift / TD
  error / actor-BC deviation / BC-reg strength / reward components / off-distribution action magnitude) ·
  architecture/method. (Full variable lists per scene in the roadmap report.)
- **Hard rule:** DirectLiNGAM proposes; **controlled ablations decide.**
- **Intervention templates:** IL chain (`approach error → both-contact → … → delivery`) ⇒ fix approach/contact
  before new RL; RL chain (`Q drift → actor-BC deviation → contact collapse → delivery collapse`) ⇒ freeze RL,
  strengthen BC-reg, DAgger/residual, phase-gated residual only.
- **Status:** **do NOT run now.** Present in [2026-07-07-hymeko-planner-roadmap.md](reports/2026-07-07-hymeko-planner-roadmap.md)
  and memory `project-cip-lingam-rl-diagnostics`. Science sibling: LiNGAM-SH (`project-kato-lingam-cip-hymeko`).

---

**Related memory:** `project-hymeko-planner-roadmap`, `project-cip-lingam-rl-diagnostics`,
`project-pick-place-gripper-collision`, `project-dagger-declarative-strategy`, `project-fanuc-offpolicy-collapse`,
`project-kato-lingam-cip-hymeko`, `reference-katolab-gpu-kato15`.
