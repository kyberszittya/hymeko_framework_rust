# CIP contact-reward ablation — Stage A (cached-rollout recomputation, no training)

**Date:** 2026-07-08 15:15 JST
**Author:** Aiko
**Status:** Stage A harness built, tested, and **run** (no training). Stage B (training smoke) **proposed, NOT
run** — awaiting explicit authorization. Phase-2 result stays labelled PROPOSED at the policy-learning level;
Stage A upgrades it to **SUPPORTED at the reward-computation level**.

---

## Goal

Turn the Phase-2 DirectLiNGAM result from a proposed causal hypothesis into intervention-backed evidence.

**Hypothesis under test:** `contact_score → total_reward` (w≈+0.69), while `total_reward` is disconnected from
`delivery_score` — i.e. the coin reward is *contact-shaped* and optimizable without improving delivery (the
BC→RL-collapse signature).

## Phase-2 baseline artifact

- Discovery run: `reports/figures/2026_07_08_13_32_cip_lingam_coin/` (`summary.json`, `causal_{mlp,hsikan}.hymeko`,
  `discovered_dag_{mlp,hsikan}.png`).
- Phase-2 report: `reports/2026-07-08-cip-phase2-coin-poc.md`.
- **Not overwritten** — Stage A writes to a fresh timestamped dir.

## Is cached-rollout recomputation available? — **Yes (verified bit-exact)**

The coin reward is `RewardSpec = Σ weight · term(state)` (`hymeko_rl/env/reward.py`), a linear combination of
Strategy term-extractors. Re-evaluating every term on the **post-`step` env** with the same `(disk_to_zone,
action)` the env used reproduces the env's own reward **exactly** (max |recomputed − env_reward| = **0.0** over a
162-step episode). So each rollout's per-step *unweighted* term vector is recorded once, and any reward *variant*
is a pure offline reweighting — **no env re-stepping, no training**. The harness asserts this per episode
(`_record_policy`, tol 1e-6) and **halts** (§11) on any mismatch.

Default coin reward (the base spec, parsed from `data/robotics/galambos_task.hymeko`):
`grasp_approach 4.0 · both_contact 5.0 · finger_contact 1.5 · in_zone 10.0 · out_of_bounds 2.0 ·
arm_body_collision 0.5`. The **contact annuity** = `both_contact` + `finger_contact`; the **delivery** term =
`in_zone`.

## Ablation variants (`build_variants`)

| Variant | Change vs base | Rationale |
| --- | --- | --- |
| `original` | none | reproduces the deployed reward (bit-exact) |
| `contact_off` | `both_contact`, `finger_contact` → 0 | remove the contact annuity |
| `contact_downweighted` | contact terms × 0.25 | soften, don't remove |
| `delivery_aligned` | contact → 0, `in_zone` ×2, `+grasp_deliver 2.5` | redirect the budget to grasp-gated delivery |

## Is retraining required for a delivery delta? — **Yes (Stage B)**

Stage A holds the **policy fixed**, so `delivery_score`, `contact_score`, and every monitor sub-score are
**identical across variants** — only `total_reward` (and its reward↔monitor disagreement) changes. Stage A
therefore tests **whether the reward computation causes the causal signature**, and *cannot* claim a delivery
delta. A delivery/collapse delta requires **retraining** under the contact-off reward — that is Stage B.

---

## Stage A results (measured, N=30 episodes/arch)

Run dir: `reports/figures/2026_07_08_15_12_cip_contact_ablation_stageA/`. Every DAG cross-view-verified
(`agree=True`, engine backend, canonical hash) — declared ≡ engine tensor view.

| Arch | Variant | `contact_score→total_reward` edge | reward↔monitor disagreement | corr(reward, delivery) |
| --- | --- | --- | --- | --- |
| mlp | original | **+0.688** | 0.589 | −0.042 |
| mlp | **contact_off** | **0.000** (collapsed) | **0.432** ↓ | **+0.855** ↑ |
| mlp | contact_downweighted ×0.25 | +0.707 (persists) | 0.437 ↓ | +0.433 |
| mlp | delivery_aligned | 0.000 | 0.428 ↓ | +0.938 ↑ |
| hsikan | original | **+0.691** | 0.566 | +0.343 |
| hsikan | **contact_off** | **0.000** (collapsed) | **0.276** ↓↓ | **+0.802** ↑ |
| hsikan | contact_downweighted ×0.25 | +0.447 (weakened) | 0.253 ↓↓ | +0.645 |
| hsikan | delivery_aligned | 0.000 | 0.271 ↓↓ | +0.901 ↑ |

**The re-parenting is explicit in the declared DAGs.** In `causal_mlp_contact_off.hymeko` the reward's parent
edge flips from `contact_score → total_reward` to **`delivery_score →(+0.88) total_reward`** — with the contact
annuity gone, the recomputed reward is caused by delivery, not contact.

### Decision-rule evaluation

> *If removing/downweighting the contact term collapses the `contact_score → total_reward` edge and reduces the
> reward↔monitor disagreement, the reward-farming hypothesis is supported at the reward-computation level.*

- **Collapse:** `contact_off` drives the edge to **exactly 0.0** in both arches; the reward re-parents onto
  `delivery_score`. ✅
- **Disagreement drop:** mlp 0.589→0.432, hsikan 0.566→0.276. ✅
- **Alignment:** corr(reward, delivery) rises from −0.04/+0.34 to **+0.86/+0.80**. ✅ (a bonus check beyond the rule)

**Verdict: the reward-farming hypothesis is SUPPORTED at the reward-computation level.**

**Honest caveats (measured, not smoothed):** (1) `contact_downweighted ×0.25` does **not** collapse the mlp edge
(+0.707) — for mlp even a quartered contact weight dominates the reward's causal signature; only full removal
collapses it (hsikan does weaken, 0.691→0.447). Downweighting is not sufficient; removal is. (2) `delivery_aligned`
aligns trivially by construction (it *is* a delivery-weighted reward) — its real test is Stage B, not Stage A.
(3) N=30 is a point estimate; the direction is consistent across both arches but a multi-seed pass would quantify
edge-weight IQR before any public claim.

---

## Exact commands

**Stage A (this run — no training):**
```
PYTHONIOENCODING=utf-8 python -m hymeko_rl.eval.cip.contact_reward_ablation --n 30 --out reports/figures
```
(`--downweight 0.25` and `--seed 0` are the defaults; seed base is the 9000 eval base.)

**Stage B (proposed, DO NOT run without authorization).** A bounded 1-seed training smoke, `original` vs
`contact_off`, from the same BC start, comparing delivery pass-rate / delivery_score / contact_score /
total_reward / reward_progress_disagreement / collapse. Because a **training** reward must be declared in
`.hymeko` (no in-memory term surgery for training runs), Stage B first needs a reward file
`data/robotics/galambos_task_contact_off.hymeko` (base spec with `both`/`fingertouch` arcs removed), then:
```
# 1. certify the contact-off training reward BEFORE launch (§reward-oracle gate); log delivers=… in the summary
python -c "from hymeko_rl.eval.reward_oracle import certify; from hymeko_rl.env.reward import RewardSpec; \
           print(certify(RewardSpec.from_hymeko('data/robotics/galambos_task_contact_off.hymeko')))"
# 2. 1-seed bounded A/B smoke (original vs contact_off), ~3k steps — SMOKE, not final evidence
python -m hymeko_rl.experiments.exp_galambos_coord_ab data/robotics/galambos_ab_contact_off.hymeko --smoke
```
Neither the `.hymeko` reward file nor the Stage-B A/B profile is created yet — they are authored when Stage B is
authorized (the user gated it behind Stage A support). Stage B is a **smoke**, explicitly not final evidence; a
production ablation (Stage C) is not proposed until Stage B.

## What upgrades PROPOSED → SUPPORTED

| Level | Evidence | Status |
| --- | --- | --- |
| Reward-computation | contact-off collapses the edge + cuts disagreement (this Stage A) | ✅ **SUPPORTED** |
| Policy-learning | a 1-seed contact-off training smoke improves delivery / reduces collapse vs original | ⏳ Stage B (not run) |
| Production | multi-seed contact-off ablation confirms the delivery/collapse delta with median/IQR | ⏳ Stage C (not proposed yet) |

The Phase-2 causal claim stays **PROPOSED at the policy level** until Stage B; Stage A only isolates the reward
computation. `contact_downweighted`'s non-collapse for mlp is itself a finding to carry into Stage B (test removal,
not just downweight).

---

## Files touched

| File | LOC | Role |
| --- | --- | --- |
| `hymeko_rl/eval/cip/__init__.py` | 33 (new) | package exports |
| `hymeko_rl/eval/cip/contact_reward_ablation.py` | 300 (new) | Stage-A harness: variants, offline recompute, per-arch/variant diagnosis + `.hymeko` cross-view |
| `hymeko_rl/tests/test_contact_reward_ablation.py` | 110 (new) | 9 pure-logic tests (recompute, variants, edge/alignment extractors) |

**No files modified** in existing modules. **CORE.YAML: none touched.** No new dependency. FANUC v2 / MetaWorld
monitors / Phase-2 artifacts untouched.

## Test + static results

- **19 passed** (`test_contact_reward_ablation.py` 9 + `test_causal_hymeko_emit.py` 10), 0.7 s. The full causal +
  export suite (45) remains green.
- `ruff check` clean · `radon cc` no block ≥ C (largest A; `_diagnose_variant`/`run_stage_a` refactored below the
  warn threshold via `_variant_frame`/`_fit_and_declare`/`_diagnose_arch`) · `mypy --strict` clean on the new
  module (native `import hymeko` scope-ignored, per Phase 1/2).
- **No §6.5 anti-patterns.** Discovery pass ran (no existing `eval/cip/` or ablation harness); variants are a
  config list not a Cartesian API; the per-step correctness gate is a raise, not a hope; shared Phase-2
  loaders/plotting reused (`_load_coin_actor`, `render_dag`, `cross_view_verify`), not duplicated.

## Experiment provenance

- Git SHA at start: `5b53a92` (branch `hymeko-neuro-migration`, working tree dirty).
- Cached checkpoints (no training): `coin_arch_ab_mlp_s0.pt`, `coin_arch_ab_hsikan_s0.pt`.
- Env `PlanarGraspEnv(robot=None, max_steps=300, difficulty=0.3)`; eval seeds 9000–9029/arch; deterministic greedy.
- No persistent state mutated. Wall ≈ 90 s (both arches × 4 variants), RSS env-bound ≪ 16 GB.

**Related:** `reports/2026-07-08-cip-phase2-coin-poc.md`; memory `project-cip-lingam-rl-diagnostics`.
