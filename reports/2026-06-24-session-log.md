# Session log — 2026-06-24 (Kato HSiKAN / RL manipulation + edge-weighted hypergraphs)

**Branch:** soma-vision · **Scope:** docs polish → FANUC off-policy negative result → Kato's collaborative/dual-discriminator
plan → edge-weighted hyperedges (reward + HSiKAN incidence) → decision to unify HSiKAN across the sparse/dense lines.

---

## 1. Main points / timeline

1. **Manipulation-models docs package** (`docs/manipulation_models/`) brought to self-contained state: 16 `.hymeko`
   models, `README.md`, `FILES.md` (per-file detail), `results/`.
2. **Architecture diagrams** added (README): RL policy pipeline (HSiKAN/MLP backbone switch → shared actor/critic
   heads), HSiKAN internals (signed message passing + Catmull-Rom + mean-pool), plus the two composition diagrams.
   All Mermaid validated, black text, labeled edges.
3. **Per-policy GIFs** embedded (demonstrator / hsikan_bc / mlp_bc, same seeds, outcome in filename).
4. **Result plots** added (`reports/figures/`, copied into the package): `bc_breaks_wall`, `galambos_bc_ppo`,
   `offpolicy_collapse`; reusable generator `hymeko_rl/plot_manipulation_results.py` (ruff/mypy clean).
5. **Numerical tables** added to the README (the data behind the plots).
6. **Full HyMeKo source** embedded in the README as collapsible `<details>` blocks (generated from the files).
7. **FANUC warm-start runs finished → NEGATIVE RESULT** (see §3). README updated honestly; user chose **"Both"**
   (TD3+BC anchor + long run) as the fix path.
8. **Kato's next ideas** → discovery sweep + 4-artifact plan (collaborative k-agent Galambos + dual-discriminator
   + edge weights).
9. **Edge-weighted hyperedges** directive (weights on arcs, not node attributes) + **arm template** directive
   (k-scaling). Confirmed arc weights already parse → **no CORE edit**. Reward scenarios reworked + tested.
10. **HSiKAN weighted incidence** (`incidence="weighted"`) added — real arc weights on the fixed structural mask.
11. **Insight:** binary {0,±1} incidence defeats the signed-hypergraph premise (real weights) — a candidate cause
    for the weak vision results, and the motivation for weighted incidence.
12. **Highway clarification:** HSiKAN = *Highway* Signed KAN (Srivastava/Greff/**Schmidhuber** 2015). The canonical
    `hymeko_neuro` has the highway gate; the **RL backbone does not** (it is a truncated HSiKAN).
13. **Decision: generalize.** The sparse/transductive (`hymeko_neuro`) and dense/inductive (`hymeko_rl`) HSiKANs
    are variations of one abstraction → unify under a single signed-KAN core. User chose **full cross-package
    unification** ("all these"). Plan in progress.

---

## 2. Achievements (implemented, tested, on disk)

- **Edge-weighted reward hyperedges** — weights moved from term-node attributes to bundle **arcs**
  (`(+ approach 4.0, …)`). Files: `data/robotics/{galambos_task,pick_place_task,arm_reach_task}.hymeko`;
  reader `hymeko_rl/env/_profile.py` (`read_bundle` → 4-tuple with arc weight) + `env/reward.py`
  (arc weight, body fallback, then 1.0); 4 other `read_bundle` callers updated. **Parity-exact** (pick-place
  parity test unchanged). Report: `reports/2026-06-24-edge-weighted-reward-arcs.md`.
- **`meta_reward.hymeko` vocab cleanup** — dropped the redundant `weight 0.0` node defaults (weights now live on arcs).
- **HSiKAN `incidence="weighted"`** — `hymeko_rl/policy.py`: the signed incidence A± now supports three modes —
  `fixed` (binary structure; hsikan), `learned` (full trainable; signedkan), `weighted` (free real weights on the
  *fixed* structural arcs, init 1.0 = parity). New tests pin init parity + that the learned weights stay masked to
  real arcs. (Replaced the internal `learn_incidence` bool with the `incidence` mode — no `hsikan` rename; HSiKAN
  stays HSiKAN.)
- **Test/quality status:** the reward + policy + caller suites pass (51 + 28 across runs); `ruff` and
  `mypy --strict --ignore-missing-imports` clean on all changed modules. No CORE edits.

---

## 3. Key findings

- **FANUC off-policy collapse is real and not a wiring bug.** BC→DDPG/TD3 (warm-start fix on) drove `refine_place`
  to **0.0 in all four cells** from a working BC. Root cause: **gross under-budgeting** — `refine=12000` steps on a
  620-step episode ≈ ~19 episodes, far below off-policy norms (1e5–1e6). The clone *is* carried into the off-policy
  actor (verified). Fix (user: "Both"): **TD3+BC anchor** (`λ·MSE(actor, demo)`) **+ ≥1e5 steps**. Lever stays
  **BC / BC→PPO**. Memory: `project-fanuc-offpolicy-collapse`.
- **Binary incidence defeats signed hypergraphs.** `dense_signed_adj` builds `+=1.0` then row-normalises → arc
  weights are effectively {0, ±1}, not free reals. Lifting this is the point of `incidence="weighted"`.
- **The RL HSiKAN is a truncated HSiKAN.** The robot-policy backbone dropped the **highway gate** and the
  inter-layer **residual** (and used binary incidence). So the Galambos "HSiKAN ≈ MLP" tie was measured on a
  backbone missing HSiKAN's defining pieces — a confound, not a verdict on HSiKAN as designed.
- **Arc weights already parse** in HyMeKo (`RefAtom.anno.value` admits `Value::Num`) — no parser/CORE change.
- **The two HSiKANs are one abstraction.** Only deep difference = aggregation backend (sparse-scatter/transductive
  vs dense-einsum/inductive/batched); spline/skip/incidence/pool are shared config. `hymeko_neuro` already
  dispatches `catmull_rom` and has `cr_highway` (highway gate + arc weights) — the design is validated there.

---

## 4. Plans on disk

- `docs/plans/2026-06-24-kato-collab-dual-discriminator/` — 4 artifacts (tex/pdf/tikz/mmd), built + validated.
  Collaborative k-agent Galambos (arm template ×k + CTDE), dual-discriminator (HSiKAN deliberative / MLP+CliffordFIR
  reflexive), edge-weighted hyperedges. 3 open decisions for Kato; arc-weight CORE risk **resolved** (already parses).
- `docs/plans/2026-06-24-unify-hsikan-signed-kan-core/` — **IN PROGRESS** (this is the current task). One signed-KAN
  core + pluggable aggregation backend (dense-batched ‖ sparse-scatter ‖ Triton); phase 1 extract core, phase 2
  migrate `hymeko_rl` (gets highway + weighted incidence for free), phase 3 migrate `hymeko_neuro` **gated on the
  OTC AUC ≥ 0.8738 regression**, Triton kept as an optional backend.

---

## 5. Open items / gates / next steps

- **HSiKAN unification (current):** finish the 4-artifact plan, then implement phase 1→2→3 (phase 3 OTC-gated).
- **Kato's 3 decisions** (gate the dual-discriminator/collaborative implementation): MARL scheme (rec. CTDE),
  fusion topology (rec. lateral), CliffordFIR placement (rec. reflexive branch + frame-stack).
- **FANUC "Both":** implement the TD3+BC anchor (`bc_coef` in `ddpg.py`) + a ≥1e5-step run (smoke one cell first).
  The anchor is also what the CTDE refine needs — shared primitive.
- **Highway gate for the RL backbone:** subsumed into the unification (the gate exists once in the shared core).

---

## 6. Memories written/updated

- `project-kato-dual-discriminator-plan` — the collaborative/dual-discriminator direction + scaffolding map +
  arc-weight-parses confirmation + the 3 Kato decisions.
- `project-fanuc-offpolicy-collapse` — the negative result + under-budget diagnosis + TD3+BC fix.

(A unification memory will be added once that plan lands.)
