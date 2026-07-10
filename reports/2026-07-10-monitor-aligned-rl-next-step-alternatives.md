# Monitor-aligned reward-repair / RL-audit — next-step alternatives map

**Date:** 2026-07-10 · Aiko · branch `hymeko-neuro-migration` · **planning doc — no experiments run.**
Maps the plausible directions after the frozen reward-repair arc, compares cost/risk/value, and recommends the next
gated step.

---

## Where we are (inputs to the decision)

Completed & frozen: reward-computation audit (Stage A) → CIP/LiNGAM-SH identified the load-bearing progress proxy →
`monitor_aligned` reward implemented → anti-farming validation showed the original reward scores a proxy-exploit ≈
true delivery (82 % of success) and the repair suppresses it **≈45×** while preserving dense pre-success shaping →
BC-anchored smoke did not destroy policy performance (1 seed). From-scratch learning remains blocked by optimizer
robustness. **Option A (final synthesis) is essentially already done** (commit `250c6b9`); the live decision is
B/C/D/E/F.

## Quick comparison matrix

| opt | direction | question in one line | compute | risk | value | novelty | RL? |
|---|---|---|---|---|---|---|---|
| **A** | Stop & synthesize | Is it Kato/paper-ready now? | ~none (done) | low | anchors the arc | low | no |
| **B** | Cross-task generalization | Does repair generalize beyond pick-place? | low–med | low–med | **very high** (patch→method) | med–high | mostly no |
| **C** | Adversarial farming search (no full RL) | Can a search *find* high-reward/low-monitor trajectories? | low–med | low | **very high** (direct reward-hacking test) | med–high | no (search, not learning) |
| **D** | Multi-seed BC-anchored validation | Does repair preserve policy robustly? | med | med (seed variance) | med–high | low | yes (BC-anchored PPO) |
| **E** | From-scratch SAC / curriculum | Does repair help learning from scratch? | **high** | **high** | high if it works | med | yes (heavy) |
| **F** | LiNGAM-SH causal-hypergraph theory | Generalize the audit into a mechanism-discovery method | med (compute) + high (thought) | med | high (scientific) | **high** | no |

## Option cards

### A — Stop and synthesize *(already delivered)*
- **Question:** Is the current result strong enough for Kato-facing discussion / a paper seed?
- **Hypothesis:** Yes — the audit→repair→verify loop with 45× proxy suppression + cross-view is a self-contained result.
- **Minimal implementation:** final synthesis + figures + claim-discipline + paper paragraph — **done** (`2026-07-10-*final-synthesis.md`).
- **Expected artifact:** synthesis reports (exist).
- **Compute:** none. **Novelty:** low (packaging). **Risk:** low.
- **Claim supported:** "runtime monitors can audit *and repair* dense shaped rewards."
- **Stopping criterion:** already met; nothing further to do for A.

### B — Cross-task generalization
- **Question:** Does monitor-aligned reward repair work beyond pick-place?
- **Hypothesis:** The gate-proxy-through-task-progress recipe transfers; ≥2 of {coffee-push, dial-turn, another wired task} show proxy-farming suppression or improved reward↔monitor alignment.
- **Minimal implementation (per task, mostly offline):** (1) map the task's info signals to monitor-aligned components (approach/contact/lift-or-actuate/delivery-or-goal/success); (2) build 6–8 decoupled diagnostic trajectories; (3) score original / ablated / monitor_aligned; (4) run the offline CIP/LiNGAM-SH audit + cross-view. **No RL** unless a task needs it. Reuse `monitor_aligned_reward.py` + `anti_farming_validation.py` (generalize the component map; coffee-push has no grasp, so gate on push-contact + goal progress).
- **Expected artifact:** a cross-task anti-farming table (task × variant × suppression) + per-task CIP audit.
- **Compute:** low–med (offline scoring + a few scripted rollouts per task). **Novelty:** med–high (a *method*, not a patch). **Risk:** low–med (some tasks lack clean proxy/delivery decoupling; dial-turn has no "delivery").
- **Claim supported:** "monitor-aligned reward repair is a general recipe, not a pick-place-specific patch" — **the strongest general-method claim.**
- **Stopping criterion:** ≥2 tasks show suppression ⇒ method claim; else report which task structures it does/doesn't fit (still informative).

### C — Adversarial farming search *without* full RL
- **Question:** Can an optimizer deliberately find high-reward / low-monitor trajectories under the original reward — and does monitor_aligned deny them?
- **Hypothesis:** Under `original`, a short trajectory-level search surfaces high-reward/zero-delivery candidates; under `monitor_aligned` the same search cannot (its high-reward set coincides with delivery/success).
- **Minimal implementation (no learning):** CEM / random-shooting / short-horizon action optimization over open-loop action sequences (or a small scripted-adversary family) on the real MetaWorld env, **maximizing the reward under evaluation**; measure the resulting monitor-success/delivery of the top-reward trajectories. Compare `original` vs `monitor_aligned`. This is *search*, not RL — bounded horizon, no replay, no gradient policy. Reuses the env wrapper + `_eval_trained`-style metric collection.
- **Expected artifact:** "reward-hacking gap" table — for each reward, `max reward found` vs `delivery/success of that trajectory`; a farming GIF under `original` that the repair denies.
- **Compute:** low–med (a few thousand env steps of shooting; no training). **Novelty:** med–high (an *active* reward-hacking probe, stronger than hand-built counterfactuals). **Risk:** low (bounded search; main risk is the search being too weak to find the exploit — mitigated by seeding it from the known proxy pattern).
- **Claim supported:** "the original reward is *actively* hackable and the repair resists it" — **the sharpest reward-hacking evidence.**
- **Stopping criterion:** search finds ≥1 high-reward/zero-delivery trajectory under `original` and none under `monitor_aligned` (matched budget).

### D — Multi-seed BC-anchored policy validation
- **Question:** Does monitor_aligned preserve policy performance robustly (not just single-seed)?
- **Hypothesis:** Over 3–5 seeds, monitor_aligned success ≈ original (within IQR) and disagreement ≤ original.
- **Minimal implementation:** the existing R3 BC-anchored PPO smoke, 3–5 seeds, original vs monitor_aligned; median/IQR of success/grasp/near/delivery/disagreement/return. Reuses `run_monitor_aligned_bc_smoke` + the multi-seed aggregation already built for Stage B.
- **Expected artifact:** median/IQR R3 table + panel (like the Stage-B multiseed panel).
- **Compute:** med (5 seeds × 2 profiles × BC+fine-tune ≈ minutes on the Mac). **Novelty:** low (robustness pass). **Risk:** med — env randomization is seed-uncontrolled (we saw success swing run-to-run), so the result may be "within noise, indistinguishable," which is a weaker outcome.
- **Claim supported:** "the repaired reward does not degrade BC-anchored policy performance (multi-seed)."
- **Stopping criterion:** success gap not significant (IQRs overlap) AND disagreement ≤ original ⇒ claim; a large robust success drop ⇒ the repair over-shapes and needs re-weighting.

### E — From-scratch RL / SAC / curriculum
- **Question:** Does monitor_aligned improve *learning from scratch*?
- **Hypothesis:** A sample-efficient off-policy method (SAC + replay) and/or a reach→grasp→delivery curriculum lets the original reward learn, and monitor_aligned learns comparably-or-better without farming.
- **Minimal implementation:** SAC + replay, curriculum object-init, staged reward, larger budget — **substantial** (a proper trainer, not the flat-obs PPO). Gated on a written optimizer plan + compute budget.
- **Expected artifact:** from-scratch learning curves + success + a farming check on the learned policy.
- **Compute:** **high** (1–2 M steps, hours). **Novelty:** med. **Risk:** **high** — from-scratch MetaWorld is hard; the optimizer-repair pass showed our current stack can't robustly learn even reach.
- **Claim supported:** "monitor_aligned enables/does-not-harm from-scratch learning" (only if it actually learns).
- **Stopping criterion:** original reward learns reach→grasp→delivery ≥ some threshold first (the gate); otherwise the comparison is uninformative. **Do not start without an approved plan + budget.**

### F — LiNGAM-SH / causal-hypergraph theory
- **Question:** Can the causal reward audit be generalized into a hypergraph mechanism-discovery *method*?
- **Hypothesis:** Mechanism stability scoring + group-sparse factorization + multi-seed mechanism ranking yields a reusable, ranked, cross-view-verified mechanism set — a method, not a one-off.
- **Minimal implementation:** extend the existing LiNGAM-SH factorization (`mechanism_factorization.py`) with stability scoring across seeds/subsamples + group-sparse selection; a spec/theorem writeup; validate on the pick-place + cross-task data.
- **Expected artifact:** a mechanism-stability method + writeup + a validated ranking on existing data.
- **Compute:** med (re-uses cached data); high *thinking*. **Novelty:** **high** (scientific), but a **novelty check is required first** (nearest areas: signed/causal graph learning, LiNGAM variants, spectral hypergraph methods, the user's own Hajdu & Hegyi 2025 precursor) before any novelty claim.
- **Claim supported:** "a machine-verified causal-hypergraph mechanism-discovery method over a signed IR."
- **Stopping criterion:** a stability-ranked mechanism set that is reproducible across seeds and cross-view-consistent; else report the instability honestly.

## Recommendation

Evaluating your stated preference (freeze A → then B or C → prefer C for sharpest reward-hacking, B for strongest
method claim, defer E):

1. **A is done** — the arc is frozen and Kato-ready. No action needed there.
2. **Do C next.** It is the sharpest, lowest-risk, highest-marginal-value step: an *active* adversarial search
   (bounded, no RL) directly upgrades the anti-farming result from "hand-built counterfactuals" to "an optimizer
   *cannot* find the exploit under the repair" — closing the exact caveat the final synthesis flagged. Cheap,
   bounded, and it produces a compelling farming-vs-denied GIF for Kato.
3. **Then do B.** Once C shows the repair resists an active search on pick-place, generalize to coffee-push +
   dial-turn to turn the patch into a *method* — the strongest paper claim. (B and C compose: run C's adversarial
   search per task in B.)
4. **D** is worth doing opportunistically (cheap-ish) but its likely outcome is "within noise," so it strengthens
   E-readiness more than the core claim.
5. **Defer E** until a written optimizer plan + budget exist (the optimizer-repair pass showed the current stack
   can't robustly learn reach). **F** is a parallel, longer scientific track — pursue only if the goal shifts from
   "empirical method" to "theory," and only after a novelty search.

**Single next gated step: Option C (adversarial farming search, no RL).** Second choice if the goal is breadth over
sharpness: Option B.

## Decision guidance by priority

- **If you want *low risk*:** **Option C** (bounded search, no training) — or the opportunistic **D** for a quick
  robustness datapoint. Both are cheap and safe.
- **If you want *highest novelty / strongest method claim*:** **Option B** (cross-task → "general recipe"), with
  **F** as the longer scientific extension (after a novelty search).
- **If you want *strongest RL / reward-hacking evidence*:** **Option C first** (active exploit search — the direct
  reward-hacking test without heavy RL), then **Option E** only once its gate (original reward learns the task) is
  met under an approved compute plan.

## Novelty caveat (per operating discipline)

Relative-novelty labels above are estimates for planning, **not** claims. Before asserting novelty for B/C/F, run a
focused literature check — nearest areas are reward hacking / specification gaming (Skalse et al.; Krakovna et al.),
potential-based reward shaping (Ng et al. 1999), runtime verification / shielding, and causal/signed-graph learning
(incl. the user's Hajdu & Hegyi 2025 precursor) — and cite a concrete instance or state "focused search found none."

**No experiments were run for this document.** CORE.YAML / `pyproject.toml` / FANUC / coin-collab untouched.
