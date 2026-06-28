# Session handoff — 2026-06-26 (END): §0 resolved + HTL reward + fingertip fix

**Read first.** This supersedes the start-of-session handoff (git has the old one). The #1 open question
(§0: "is HSiKAN-loses-to-MLP a bug?") is now **effectively resolved**. Working tree is dirty (intentional).

---

## 0. THE HEADLINE — §0 is resolved: NOT a bug

The "HSiKAN ties/loses to MLP on every RL task" pattern is **not** a wiring bug and **not** a backbone bug.

- **Wiring audit** (`reports/2026-06-26-hsikan-wiring-audit.md`): cartpole / arm6dof / quadruped / pick-place
  / fanuc are ALL wiring-clean — no obs↔vertex misalignment, no degenerate rows, healthy forward. (Quadruped
  is **9 vtx not 14** as the old handoff claimed; its branching topology is correctly encoded. Cartpole
  HSiKAN learns **200/200**.)
- **Structural probe** (`reports/2026-06-26-structural-probe.md`, `hymeko_rl/structural_probe.py`): supervised
  RL-free probe, params-matched HSiKAN vs MLP, data-scaling sweep. Backbone is healthy. **Surprise:** HSiKAN's
  edge is **POOLING-first** (per-node-activation + mean-pool = Deep-Sets prior): **up to 52×** on a separable
  `Σtanh(x_v)` target; the signed-2-hop advantage is secondary (**~1.5–3×, data-hungry**). Both advantages
  GROW with data (representational, not sample-efficiency).
- **Reinterpretation (user's reframe, vindicated):** the robot tie is a **bias↔objective mismatch**, not
  incompetence — robot value/policy under current rewards isn't a pooled/signed-structural function. Don't drop
  HSiKAN; give it structure to cash in.

**THE DECISIVE NEXT TEST (start here): the readout ablation.** New hypothesis: HSiKAN's **mean-pool readout is
the liability for control** — it helps separable aggregates (the 52×) but *collapses* the cross-joint
coordination control needs (which a flat MLP preserves). Swap mean-pool → per-node-concat / attention pool,
re-run ONE robot task (e.g. galambos or quadruped). If the tie flips, pooling (not the backbone) was the
bottleneck. This is the single experiment that closes §0. (Pool mode is a param on `SignedKANBackbone`,
`pool=` in `POOL_MODES` — check what non-collapsing options exist; may need a concat readout added.)

---

## 1. Banked this session (DONE, tested, all non-core, CPU)

| Work | Files | Report / Plan |
|---|---|---|
| **Fingertip reward fix** (collaborative): dense approach measured nearest body/base, not the tip → inject `tip_{side}` sites + tip+elbow blend | `hymeko_rl/env/planar_grasp_env.py`, `reward.py`, tests | `reports/2026-06-26-galambos-fingertip-reward.md`, plan `…-galambos-fingertip-reward` |
| **HTL-robustness-as-reward PoC**: one `.htl` formula → dense reward + monitor verdict; reuses `signedkan_wip/src/htl` | `hymeko_rl/htl_reward.py`, `data/robotics/galambos_spec.htl`, `exp_htl_reward_ab.py`, tests | `reports/2026-06-26-htl-reward-poc.md`, plan `…-htl-reward-poc` |
| **Wiring audit** (read-only) | (introspection) | `reports/2026-06-26-hsikan-wiring-audit.md` |
| **Structural probe** + data-scaling sweep | `hymeko_rl/structural_probe.py`, tests | `reports/2026-06-26-structural-probe.md`, plan `…-structural-probe` |

All tests pass (26 planar, 6 htl_reward, 7 structural_probe + collateral). ruff clean. No CORE.YAML edits.

---

## 2. Interrupted / to redo (background runs DIED on reboot)

- **HTL full A/B** (`bbyd75usd`): `python -m hymeko_rl.exp_htl_reward_ab --mode full --n-eval 20 --out-dir
  reports/htl_reward_ab_full` — never finished (30k×2 SAC). **Rerun.** (Smoke passed: plumbing OK, both 0%
  delivery at 2k steps as expected.)
- **Structural-probe sweep regen post determinism-fix** (`bb23th7jq`): `python -m hymeko_rl.structural_probe
  --sweep 16,32,64,128,256,512 --hidden 32 --seeds 5 --epochs 300 --out-dir reports/structural_probe` —
  **rerun** so `structural_probe_sweep.{json,png}` and the report's sweep table are post-fix (single-point
  n=256 IS post-fix: struct 1.46×, bag 17.3×; trend robust either way).

---

## 3. Pending (priority order)

1. **[#1] Readout ablation** (§0 above) — the decisive test. Mean-pool vs concat/attention on a robot task.
2. **Structural robot reward** — give HSiKAN structure to cash in: `galambos_taskgraph` hyperedges +
   the HTL structural predicates (`exp_htl_reward_ab.py --mode full`). Tie to the probe's finding.
3. **Collaborative fingertip-fix validation** — delivery vs the 0.208 baseline (BC + off-policy refine).
4. **Vary-the-graph probe** (`incidence="learned"`) — expose structural value the fixed-graph ceiling hides.

---

## 4. Orientation (key files this session)
- Probe: `hymeko_rl/structural_probe.py` (`run_probe`, `sweep_n_train`, `build_toy_graph`).
- HTL reward: `hymeko_rl/htl_reward.py` (`HtlRewardSpec` duck-types `RewardSpec.evaluate`); evaluator reused
  from `signedkan_wip/src/htl/` (bridge = `sys.path` insert, cf `dashboard_node.py`).
- Backbones: `signed_kan/backbone.py` (`SignedKANBackbone`, `pool=` mode — the readout-ablation lever),
  `hymeko_rl/policy.py` (`mlp_backbone`, `build_policy`).
- Reward/metrics: `hymeko_rl/env/planar_grasp_env.py` (`with_fingertip_sites`, `compute_planar_metrics`),
  `hymeko_rl/env/reward.py`.
- Memory: `project-hsikan-loses-possible-bug` (the §0 resolution), `project-fsm-structured-rl` (HTL reward).

## 5. Process notes
- Launch long runs with `exec env … python …` (no trailing `; echo`) so TaskStop reaps the child.
- One heavy job at a time; 16 GB RSS cap (runs stayed 0.7–0.8 GB).
- `structural_probe` is deterministic only AFTER the build-seed fix (seed torch before each model build).
