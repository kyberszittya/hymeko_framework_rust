# HSiKAN / Gömb-Soma structural-leverage hypothesis — preparation

**Date:** 2026-07-10 · Aiko · branch `hymeko-neuro-migration` · **preparation only — no experiments run.**
Formalizes the hypothesis "HSiKAN / Gömb-Soma leverage structurally-heavy tasks better than MLP / general nets,"
grounded in a repo map, designed to be falsifiable, and wired to **reuse the existing capacity-matched harnesses**
rather than reinvent them.

---

## What HSiKAN / Gömb / Soma actually are (so the claim is precise)

- **HSiKAN** = Highway Signed KAN: per-vertex features message-passed over a **signed** adjacency `B = A⁺ − A⁻`
  with a Catmull-Rom KAN spline edge nonlinearity + a highway skip (`hymeko_neuro/core/layer.py:68`,
  `backbone.py:20`). **Structure is mandatory** — `build_policy` raises `TypeError` if `hsikan` is built without
  `hg_state` ("the structure is the whole point", `hymeko_rl/agents/policy.py:349`). The MLP control **flattens the
  same per-vertex obs** (`policy.py:162-169`), so the comparison is a pure backbone swap on identical inputs.
- **Gömb** = a 3-shell cascade (Clifford-FIR ⊃ **HSiKAN** ⊃ CPML) → edge-sign predictor over signed *cycles*
  (`hymeko_neuro/models/hymeko_gomb/cascade.py:100`); ablations are separate classes (`GombNoMiddle`, …).
- **Soma** = a reflex/vision lane (patch-hypergraph + signed walk-conv, `data/nn/soma_vision.hymeko`).
- **Gömb-Soma as an online RL backbone is a *plan*, not wired** (`docs/memory/project-gomb-soma-rl-remap.md`; CPML
  + spatial tree are "the missing pieces"). ⇒ the RL Gömb-Soma arm is **gated on building those**; the
  **supervised** Gömb (signed-link) is available today.

## Why the naïve hypothesis is already known to fail here (this is the key context)

The repo has **run this comparison and debunked the easy version twice**:
- **Cart-pole:** a params-matched MLP (26.7k ≈ HSiKAN 26.2k) **ties** HSiKAN; the earlier "HSiKAN win" was an
  **under-parameterized baseline artifact** — *"Structure is NOT load-bearing on cart-pole"* (`DONE.md:17-19`).
- **Coin/Galambos arch A/B:** the HSiKAN>MLP delivery gap was a **BC-step-0 peak** that collapses under RL
  (hsikan→0.04, mlp→0.0) — not a clean win.
- **Where structure *is* load-bearing:** signed-link prediction — HSiKAN-Optuna beats DADSGNN by **+8.57 pp AUC**
  on Bitcoin-Alpha, Gömb-strict +19.77 pp accuracy vs SE-SGformer on Epinions (`docs/GOMB_SOTA_COMPARISON_2026_05_17.md`).

**Consequence:** any acceptable hypothesis must survive capacity-matching, must NOT rely on RL-fragile single peaks,
and must explain *why* structure helps on signed-link but not on cart-pole. That forces the **scaling + causal**
form below.

## The hypothesis (falsifiable, two forms)

> **H1 (scaling).** For **capacity-matched** networks trained identically, the (HSiKAN or Gömb) − MLP performance
> gap is **≈0 on structure-poor tasks and increases monotonically with a measured structure-richness** of the task.
>
> **H2 (causal).** That gap is **caused by the task's structure**: destroying the structure while preserving the
> data (permuting the signed incidence `A±`) collapses the gap to ≈0.

H1 alone is confoundable (a wide enough MLP, a lucky task). H2 is the discriminator — and it goes **beyond** the
existing controls (see below).

## The structure-richness ladder (anchored by known results)

Ordered by how much the optimal function depends on graph structure. The two ends are already measured — the
hypothesis is that the gap **interpolates monotonically** between them.

| rank | task | structural object | richness proxy | expected HSiKAN−MLP gap | status |
|---:|---|---|---|---|---|
| 1 | **signed-link** (Bitcoin-Alpha/OTC, Epinions, Slashdot) | signed graph; label = edge sign | max (task *is* the graph) | **large** (measured: +8–20 pp) | ✅ genuine win on record |
| 2 | **holonomy / structural probe** | fixed signed graph; label = `Σtanh(B²x)` / Z₂ cycle holonomy | max (label is a *pure* structural invariant) | large, cleanest | harness exists (`structural_probe.py`, `holonomy_probe.py`) |
| 3 | **WL-hard synthetic** (`is_3_regular`, `has_triangle`) | signed hypergraph properties | high | large | `hymeko_gnn_experiment/` (GNN baselines are stubs) |
| 4 | **quadruped** goal-reach | branched kinematic hg (torso+4 legs) | high branching | medium | env ready (`quadruped_env.py:161`) |
| 5 | **galambos** bimanual grasp | two arms + coin (task-hyperedges) | medium-high w/ `with_task_hyperedges` | medium | env ready; RL-fragile |
| 6 | pick-place / gripper | chain + gripper fork | medium | small-medium | env ready |
| 7 | arm reach (4-/6-DOF) | serial chain | low | small | `reach_arch_compare.py` |
| 8 | **cart-pole** | 2-vertex chain | minimal | **≈0** | ✅ measured tie |
| 9 | **MetaWorld pick-place** (control) | flat obs, no `.hg` | **0** | **0** (or slight MLP win) | flat control |

Richness proxies to log per task (so the x-axis is quantitative, not a vibe): `n_vertices`, branching factor, mean
hyperedge arity, signed-arc fraction, and — for supervised — a boolean *label-is-structure-determined*.

## Three controls (two exist; one is the new falsifier)

1. **Capacity-match** *(exists)* — binary-search MLP width to the HSiKAN param count (`exp_v2_hsikan.py:129
   _match_mlp_hidden`); the fairness gate is "beat MLP-**on-structured-obs**, not just flat" (`exp_v2_hsikan.py:206`).
2. **Structure-independent target / `bag` control** *(exists)* — `structural_probe.py` trains on `structural`
   (`B²x`) **and** `bag` (per-node sum, structure-free); the decisive signature is
   `MSE_struct(HSiKAN) ≪ MSE_struct(MLP)` while `MSE_bag(HSiKAN) ≈ MSE_bag(MLP)`. Plus the **linear confound guard**
   in `holonomy_probe.py` (a linear reader must be at chance).
3. **Incidence-scramble ablation** *(NEW — the H2 falsifier)* — run HSiKAN on the *same* task/data with `A±`
   **permuted** (shuffle the vertex↔arc incidence, preserving degree). Prediction: HSiKAN's advantage **vanishes**
   under scramble. This is subtly stronger than the `bag` control: `bag` changes the *target*; scramble keeps the
   target and destroys only the *given structure* — so it isolates "HSiKAN uses the *correct* structure" from
   "HSiKAN is just a better function class." I did **not** find an existing incidence-scramble harness; it is the
   one genuinely new piece to build (small — a permutation of `HypergraphState.dense_signed_adj`).

## Metrics

Supervised: held-out MSE / accuracy / **Spearman** / AUC (`exp_v2_hsikan.py`, `GOMB_SOTA`). RL: delivery-rate,
return, EE-error (`reach_arch_compare.py`, `exp_pernode_actor_ab.py`). Always: **`n_params` (matched)**,
**forward latency** (`holonomy_probe.py` — HSiKAN must not win by being slower/bigger), multi-seed **median/IQR**,
and **the gap as a function of the richness proxy** (the H1 slope is the headline figure).

## Falsification criteria (state before running)

The hypothesis is **rejected** if any hold: (a) HSiKAN wins as much on the flat/cart-pole end as on signed-link
(→ capacity/optimization, not structure); (b) the gap does **not** increase with the richness proxy (H1 fails);
(c) **scrambling `A±` does not hurt** HSiKAN (H2 fails — the "structure" is decorative); (d) a params-matched MLP
on the *same structured obs* closes the gap (the `exp_v2_hsikan` fairness gate).

## Reuse map (no new trainer; §6.1)

| need | reuse |
|---|---|
| capacity-matched HSiKAN vs MLP, supervised | `hymeko_rl/experiments/structural_probe.py`, `exp_v2_hsikan.py` |
| pure structural-invariant discriminator | `structural_probe.py` (`B²x` vs `bag`), `holonomy_probe.py` |
| backbone-swap on an RL/BC task | `reach_arch_compare.py`, `exp_pernode_actor_ab.py`, `pick_place_bc.py` |
| signed-link SOTA (genuine-win anchor) | `hymeko_neuro/core/graph_model.py` (SignedGraphHSiKAN), Gömb `cascade.py` |
| structural augmentation for RL envs | `HypergraphState.star_expansion` / `with_task_hyperedges` (`hypergraph_state.py:112,148`) |
| **new** incidence-scramble ablation | a degree-preserving permutation wrapper on `dense_signed_adj` (to build) |

## Staged pilot (each stage gated; supervised first, low-variance)

- **Stage 0 — build the scramble ablation + richness logger** (small, offline). No training.
- **Stage 1 — supervised pilot (2 rungs):** `structural`/`bag` probe (rung 2, structure-max) **vs** a flat control
  (rung 9-like), capacity-matched HSiKAN/MLP, 3 seeds. *Does the gap appear on structured and vanish on flat, and
  does scramble kill it?* Cheap, fast, decisive — this is the whole hypothesis in miniature.
- **Stage 2 — the ladder:** add signed-link (rung 1) + holonomy + arm-reach + cart-pole to plot the **H1 slope**
  (gap vs richness), 5 seeds, median/IQR. Add **Gömb** on the signed-link rung.
- **Stage 3 — RL rungs:** quadruped / galambos backbone-swap (structure genuinely branched), reusing
  `exp_pernode_actor_ab.py`; RL last because of its variance.
- **Stage 4 (gated on missing code):** online **Gömb-Soma** RL backbone — requires building CPML + the Soma spatial
  tree (`project-gomb-soma-rl-remap.md`); out of scope until Stages 1–3 confirm the supervised signal.

## What is NOT ready / honest caveats

- Gömb-Soma as an **online RL** backbone does not exist yet (plan only) — so the headline "Gömb-Soma on RL" claim
  is **not** testable without a build. Supervised Gömb (signed-link) **is** testable now.
- The naïve "HSiKAN > MLP" has been a **capacity/BC-step-0 artifact** twice here — the scaling+scramble framing
  exists precisely to not repeat that.
- `hymeko_gnn_experiment` GNN baselines are **stubs** — a fair "vs general networks" claim needs a real GNN/attention
  baseline wired in (currently only MLP is a live control).

## Recommendation

Green-light **Stage 0 + Stage 1** (build the scramble ablation + run the 2-rung supervised pilot) — it is cheap,
low-variance, reuses existing harnesses, and either shows the structure-scaling signature immediately or falsifies
it before any RL compute. If Stage 1 holds, I promote this to the full four-format plan bundle (§2) for Stages 2–4.
No experiments run for this prep.
