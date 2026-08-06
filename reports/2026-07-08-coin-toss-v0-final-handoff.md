# Coin-toss v0 — final handoff (2026-07-08, FROZEN)

**Status:** frozen. Nothing running. This is the single entry point for the coin-toss / Galambos delivery result
as of 2026-07-08. Read this before touching the coin-toss RL area.

## 1. Final deployable artifacts + hashes

| role | artifact | md5 |
|---|---|---|
| **base learned policy** (robust MLP) | `experiments/2026_07_08_seed_stabilized/E_valselect_v2.pt` | `b822a6608f6a626aedef49f4d1ee379a` |
| **option layer** (5 bounded params + base ref) | `experiments/2026_07_08_option_rl_v0/phase_gated_theta.json` | `06b162d0c64338081cc5b4ff64762c39` |
| original baseline (frozen DAgger) | `experiments/v2_dagger/FROZEN_selected/mlp_s1_selected_d3.pt` | `edf4fe81f04bbda26393ca9f230828b9` |

- **Reward:** `data/robotics/galambos_task_deliver_v2b.hymeko` — unchanged; oracle-certified `delivers=True`,
  optimal_return 25.404.
- **Verifier:** frozen `TaskMonitor` (external, never in a learning objective).
- **Guards (both final runs):** PipelineSchemaLedger **PASS**, PolicyProvenanceLedger **PASS**. Zero
  exploit / body-only / arm-body behavior across every reported cell. **CORE.YAML untouched.**
- `phase_gated_theta.json` θ\* = {contact_offset −0.00109, push_gain 0.60931, direction_correction 0.01379,
  brake_threshold 0.05461, release_threshold 0.01508}; references base MLP `E_valselect_v2.pt` (`b822a660…`).
- **The deploy stack is E_valselect_v2.pt + phase_gated_theta.json** wired by `PhaseGatedPolicy`
  (`hymeko_rl/agents/phase_gated.py`): frozen MLP drives APPROACH; `PhasePushController(θ)` drives the two-finger
  PUSH once fingertip contact exists (per-phase handoff, NOT a per-step residual).
- **Animated proof (§9):** `experiments/2026_07_08_option_rl_v0/deploy_stack_success.gif` — the deploy stack
  making a held, fingertip-dominant delivery (seed 31004, 640px); `…/E_valselect_only.gif` — the base MLP alone at
  the same seed for comparison. Rendered from the frozen artifacts (no training/search).

## 2. Final metrics (fresh eval seeds 31000/33000/35000/37000, n=48, v2b, frozen TaskMonitor)

| metric | original baseline | E_valselect_v2 | **E_valselect_v2 + phase_gated_theta** |
|---|---:|---:|---:|
| ft_dom (fingertip-dominant delivery) | 0.458 | 0.615 | **0.688** |
| monitor_pass (rate — the accepted gate) | 0.344 | 0.521 | **0.620** |
| monitor_score (mean — informational) | 0.179 | 0.433 | 0.409 |
| sustained-PUSH windows / ep | 0.31 | 1.04 | **1.43** |
| both-fingertip contact fraction | 0.045 | 0.091 | **0.181** |
| fingertip-progress-in-contact | 0.0023 | 0.0093 | **0.0229** |
| body-only progress / arm-body / exploit | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |

Note: the frozen-DAgger baseline is eval-seed-sensitive (0.568 on the v2 test seeds 9000–15000 vs 0.458 on these
fresh seeds); E_valselect_v2 is tight (ft_dom std 0.023) and exceeds it on unseen seeds. The deploy stack improves
delivery **rate** and sustained contact ~3–4× over the original baseline with zero exploit.

## 3. The negative arcs and their measured mechanisms (do not re-walk)

1. **Scalar TD3+BC / CQL / residual critic-gradient — FAILED (mechanism measured).** The STRONG_PASS critic's
   local ∂Q/∂a is monitor-misaligned: `a+ε∇Q` raises Q while cutting two-fingertip contact (0.045→0.010). No ε>0
   improves; TD3+BC, CQL-actor, and frozen-critic residual all degrade the clone.
   (`reports/2026-07-08-vector-critic-projected-gradient.md`, `2026-07-08-fair-vector-critic-retest.md`.)
2. **Vector-critic / local residual — FAILED (root cause measured).** With action-diverse replay + Monte-Carlo
   component critics (critics now *calibrated*), the vector-projected gradient still cuts contact identically to
   scalar; and `best_sampled` (best of 8 local perturbations) < DAgger — **the DAgger action is locally
   contact-optimal**, so no bounded local move (scalar, vector, or sampled) improves the monitor.
   (`reports/2026-07-08-fair-vector-critic-retest.md`.)
3. **Demo-mix — NOT_ROBUST (the single-seed POSITIVE was a lucky training seed).** A single-training-seed mix_25
   read POSITIVE, but across 8 training seeds the median ft_dom fell below baseline and the pooled ft_dom was
   statistically *worse*. The variance was in the **training seed / checkpoint**, not the behavioral lever.
   (`reports/2026-07-08-option-msdm-trainseed-robustness.md`, which supersedes `2026-07-08-option-msdm.md`.)
4. **Seed-stabilized rescue — POSITIVE_ROBUST.** DAgger-anchor (`+λ·MSE(π, π_DAgger)`) gives the most stable
   basin; **val-selected checkpointing** gives the best behavior; both fix the variance. Critically, **validation
   predicts test** (Spearman val↔test: ft_dom 0.55, monitor_score 0.68, sustained-PUSH 0.63) — checkpoint
   selection is legitimate, not test leakage. Deployable = `E_valselect_v2.pt`.
   (`reports/2026-07-08-seed-stabilized-demo-mix-v2.md`; confirmed on fresh seeds in
   `2026-07-08-fresh-eval-confirm.md`.)
5. **Option-RL layer — POSITIVE (phase-gated).** CEM over 5 bounded PhasePushController params on top of
   E_valselect_v2 further improves delivery + contact. (`reports/2026-07-08-bounded-option-rl-v0.md`.)

## 4. Accepted caveat (option layer)

The option layer's **monitor_score MEAN decreases slightly** (0.433→0.409, −0.025) while **monitor_pass (rate)
rises** (0.521→0.620) along with ft_dom, sustained-PUSH, and fingertip progress. Mechanism: the handoff **rescues
borderline deliveries** — more episodes pass (rate up), but the rescued episodes carry modest sub-scores, so the
*mean* dips. This is a benign **delivery↔mean-quality trade**, never exploit or body-driven (both exactly 0). Per
user decision (2026-07-08) the gate is on monitor_**pass** (rate); the mean dip is recorded as informational.

## 5. Do-NOT-rerun-blindly notes

- **No per-step TD3 / SAC / CQL actor updates.** Measured-failed and mechanism-understood (§3.1–3.2). The DAgger
  action is locally contact-optimal; per-step critic-gradient refinement cannot beat it.
- **No raw per-step residual actor.** Same closed lever.
- **No monitor-as-reward.** The `TaskMonitor` is the external verifier only; any learning objective must be a
  separate `SearchObjective` (kept separate throughout).
- **No reward change, no CORE.YAML edit.**
- **Future improvement direction (if any):** (a) **v0→v1 option handoff refinement** — hand off to the pusher only
  *after* the MLP secures two-finger contact, to preserve the monitor_score mean *and* the delivery gains; and/or
  (b) **reshape the external SearchObjective** to penalise low-sub-score deliveries. Both are option-parameter /
  handoff-level changes, not new per-step RL. A CEM-seed sweep is cheap (θ is 5-dim) if robustness of θ\* is
  questioned.

## 6. Exact reproduction commands

Run from repo root with the Mac venv (`.venv/bin/python`) and `PYTHONPATH=.` (workspace is `package = false`).
All eval is deterministic CPU MuJoCo; no GPU needed.

```bash
# (a) fresh-seed eval — baseline vs C_anchor repr vs E_valselect_v2 (writes experiments/2026_07_08_fresh_eval_confirm/)
PYTHONPATH=. .venv/bin/python -m hymeko_rl.experiments.exp_fresh_eval_confirm

# (b) option-θ eval — CEM over θ (phase-gated) + fresh-seed acceptance vs E_valselect_v2
#     (writes experiments/2026_07_08_option_rl_v0/results.json + phase_gated_theta.json on POSITIVE)
PYTHONPATH=. .venv/bin/python -m hymeko_rl.experiments.exp_option_rl_v0 --stage full

# (c) guard checks — the code-level safety stack (unit tests) + the ledgers assert PASS inside every run above
PYTHONPATH=. .venv/bin/python -m pytest -p no:randomly -q \
  hymeko_rl/tests/test_phase_gated.py hymeko_rl/tests/test_stabilized_bc.py \
  hymeko_rl/tests/test_demo_mix.py hymeko_rl/tests/test_multiseed.py hymeko_rl/tests/test_push_audit.py \
  hymeko_rl/tests/test_option_search.py
# guards per run (PipelineSchemaLedger / PolicyProvenanceLedger / v2b certify) are recorded in each results.json:
python3 -c "import json;d=json.load(open('experiments/2026_07_08_option_rl_v0/results.json'));print(d['guards'],d['provenance'],d['v2b_reward']['delivers'])"
```

kato15 (GPU, training-seed layer) reproduction: safe separate dir
`/home/hajdu/hymeko_framework_rust_option_msdm_20260708`, `.venv_stand` (uv; scipy/matplotlib added),
`MUJOCO_GL=egl`, `--device auto`; `exp_seed_stabilized.py --stage trainseeds` / `--stage full`. See
`project-seed-stabilized-v2-positive-robust` memory. The Mac cannot `uv sync --group ml` (cu132 pin); it uses a
standalone `uv pip install --no-sources` CPU/MPS venv (`project-mac-rl-venv-migration`).

## 7. Abstract (for paper/report reuse)

> We study learned two-arm coin (cylinder) delivery under fingertip-only physics with an external
> temporal-logic-style monitor as the verifier. Per-step critic-gradient refinement (TD3+BC, CQL, scalar and
> vector-projected residuals) is shown, with measured mechanism, to be mis-specified for this contact task: the
> demonstrator action is locally contact-optimal, so bounded local action moves reduce two-fingertip engagement
> regardless of the critic. We instead generate sustained-contact demonstrations from a bounded scripted push
> option and imitate them; naive imitation is high-variance across training seeds (a single-seed "success" fails to
> replicate), but a DAgger-anchored objective and validation-selected checkpointing both make it
> training-seed-robust, and — crucially — a held-out validation gate predicts test performance (Spearman
> 0.55–0.68), so checkpoint selection is principled. The resulting learned policy preserves delivery while roughly
> tripling sustained two-finger contact with zero exploit, and holds on unseen evaluation seeds. A bounded
> five-parameter option-RL layer (CEM) composed as a phase-gated handoff on top of the learned base further raises
> delivery rate and contact, at a small, characterized decrease in the mean monitor sub-score while its pass-rate
> rises. Throughout, the reward and monitor are frozen and separated, a pipeline/provenance safety stack passes,
> and no core framework component is modified.

---

**End of handoff. Frozen. Nothing scheduled or running.**
