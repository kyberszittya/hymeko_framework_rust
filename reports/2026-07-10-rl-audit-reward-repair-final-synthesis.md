# RL-audit & reward-repair — final synthesis (FROZEN)

**Date:** 2026-07-10 · Aiko · branch `hymeko-neuro-migration` · **ARC FROZEN — no further experiments.**
Kato-facing + paper-seed synthesis of the whole MetaWorld pick-place RL-audit / reward-repair arc, including the
Option C adversarial search. All numbers are from committed runs; nothing new was run for this document.

**Central narrative: detect → explain → repair → verify → scope honestly.**

---

## Stages

### 1. Reward-computation audit
- **Finding:** the `mw_in_place` / progress proxy is load-bearing at the reward-computation level (5-seed robust);
  ablating it re-parents the reward toward delivery.
- **Claim:** reward computation can be causally audited against task-monitor variables (CIP / LiNGAM-SH).

### 2. BC-anchored fine-tune
- **Finding:** BC-anchored PPO preserves task success under both reward variants (≈100% under PPO), but the
  reward↔monitor disagreement is higher under the ablated reward (4–5/5 seeds).
- **Claim:** policy success alone can hide a reward/monitor mismatch — the disagreement surfaces it.

### 3. From-scratch PPO diagnostics & optimizer repair
- **Finding:** the 0%-vs-0% from-scratch result was **invalid as a reward-learning result** — PPO was
  optimizer-limited, not the reward. Three real PPO defects were found and fixed: (a) observation normalization was
  applied *between* rollout collection and the PPO update (corrupting the ratio); (b) no obs-norm warmup; (c) an
  obs-std floor of 1e-3 amplified a near-constant object dimension (~1000×) into noise.
- **Claim:** the from-scratch **policy-learning role remains gated and unresolved** (reach not robust after repair).

### 4. Monitor-aligned reward repair
- **Finding:** a monitor-aligned reward was implemented: potential-based approach, capped/gated contact,
  potential-based lift, delivery **gated on grasp/lift evidence**, success as the strongest term, and a
  stagnation/hover penalty.
- **Claim:** runtime monitors can be used as **semantic instruments for reward repair**, not only post-hoc
  evaluation.

### 5. Decoupled anti-farming validation
- **Finding:** on counterfactual diagnostic trajectories that break the scripted contact↔delivery collinearity, the
  original reward scores a proxy-exploit highly while monitor_aligned suppresses it. **Proxy/success ratio:
  original ≈ 0.816, monitor_aligned ≈ 0.018 — ≈45× suppression.**
- **A bug in our own repair:** the first repaired reward accumulated lift reward while *holding* the object aloft,
  scoring `grasp/lift no delivery` (19.75) **above** `delivery` (11.80). Fixed by making lift **potential-based**
  (reward raising, not holding) → `grasp/lift no delivery` = 5.45 (below delivery).
- **Claim:** reward repair itself must be audited — monitor-aligned design is an **iterative audit/repair** process.

### 6. Real-env adversarial search (Option C, bounded CEM, no RL)
- **Finding:** warm-started CEM maximizing **either** reward still **delivers** (success 1.0). So the original
  reward is **not globally hackable** in this bounded search.
- **But:** scripted zero-success proxy-adversaries receive **positive** reward under the original reward —
  `hover` +9.9 (small), `grasp_hold` +241 (≈28% of expert median, up to ~79% on some layouts). Under
  monitor_aligned, `hover` is penalized (−115.8) and `grasp_hold` is ≈0 (+0.5).
- **Final scoped interpretation:**

  > The original reward's global optimum remains task completion under bounded CEM search, but the reward admits
  > semantically misaligned local plateaus: failed grasp/hold or proximity behaviors can receive positive reward
  > despite zero task success. The monitor-aligned repair suppresses these local plateaus while preserving
  > completion as the reward-maximizing behavior.

## 7. Final claim set

**Supported:**
- **A.** Runtime monitors can expose reward/monitor disagreement invisible from task success alone.
- **B.** Causal reward audit can identify load-bearing proxy reward structure.
- **C.** A monitor-aligned reward can suppress proxy/contact farming while preserving dense pre-success shaping.
- **D.** The repaired reward preserves the global optimum in bounded CEM search: maximizing it still delivers.
- **E.** The original reward is not globally hackable under bounded CEM search, but it contains positive-reward
  zero-success local plateaus.
- **F.** From-scratch learning superiority is not claimed and remains future work.

**Explicit non-claims:** MetaWorld is not globally wrong; the original reward is not globally hackable; an RL learner
would not *necessarily* exploit the plateau; monitor_aligned does not solve from-scratch learning; no
policy-learning superiority without future multi-seed RL.

## 8. Tables

### A — Stage summary

| stage | question | method | result | valid claim | caveat |
|---|---|---|---|---|---|
| audit | Which reward proxies are load-bearing? | CIP / LiNGAM-SH, 5-seed | `mw_in_place` dominant | reward computation is causally auditable | reward-computation level |
| BC fine-tune | Does ablation change success? | BC-anchored PPO | success ≈ equal; disagreement higher under ablated | success hides reward/monitor mismatch | single→5-seed |
| from-scratch diagnostics | reward or setup? | 6-probe sanity suite | PPO-setup issue (harness proven correct) | 0%-vs-0% not a reward result | — |
| optimizer repair | Can PPO learn reach? | std sweep/anneal + 3 bug-fixes | reach improved, not robust | learning-role gated | env non-determinism |
| reward repair | Reduce disagreement, stay dense? | monitor-aligned gated reward | R1–R3 pass; cross-view ✅ | monitors can repair rewards | R2 collinear; R3 1-seed |
| anti-farming validation | Suppress proxy farming? | 8 decoupled counterfactuals | ≈45× suppression; HEALTHY 5/5 | repair suppresses proxy farming | counterfactual, not a learner |
| adversarial search | Can search hack the reward? | warm-started CEM + scripted adversaries | not globally hackable; local plateau removed | see distinction §6 | magnitude layout-variable |

### B — Anti-farming counterfactual trajectories (total reward)

| class | original | mw_in_place_off | monitor_aligned |
|---|---:|---:|---:|
| far | −52.0 | −60.0 | 0.0 |
| approach | −46.3 | −54.3 | −0.6 |
| hover-farm | −6.4 | −30.4 | −20.0 |
| bare-contact-farm | 8.8 | −23.2 | 2.0 |
| grasp/lift no delivery | 24.8 | −23.2 | 5.45 |
| delivery | 86.8 | −1.2 | 11.8 |
| success | 108.8 | 4.8 | 112.4 |
| **proxy_exploit** | **88.8** | −23.2 | **2.0** |

Proxy/success: original 0.816 → monitor_aligned 0.018 (**≈45×**).

### C — Real-env adversarial search (median, 5 layout seeds)

| controller | success | original reward | monitor_aligned reward | interpretation |
|---|---:|---:|---:|---|
| expert | 1.0 | 862.5 | 4106.8 | delivers; both reward it |
| CEM-max original | 1.0 | 885.1 | 4162.9 | maximizing original → delivers (no global hack) |
| CEM-max monitor_aligned | 1.0 | 811.6 | 4263.2 | maximizing repair → delivers (global optimum preserved) |
| hover | 0.0 | +9.9 | −115.8 | proxy: small + under original, penalized under repair |
| grasp_hold | 0.0 | +241.1 | +0.5 | **failed near-completion: credited by original (~28%), ≈0 under repair** |

## 9. Kato-facing summary

> We found that a benchmark reward can have the correct global optimum while still containing local incentives that
> are not aligned with task semantics. HyMeKo/runtime monitors let us detect these local reward plateaus, explain
> them causally, and repair the reward by gating dense shaping through task-progress evidence. The repaired reward
> suppresses zero-success proxy behaviors while still rewarding successful completion.

## 10. Paper-ready abstract seed

> We present a monitor-guided reward audit and repair workflow for reinforcement learning. Using runtime task
> monitors as semantic references, we identify reward components whose dense shaping signals become decoupled from
> task progress. In a MetaWorld pick-place case study, causal reward audit and counterfactual validation reveal that
> the original shaped reward admits positive-reward zero-success plateaus, although its global optimum remains
> successful task completion under bounded CEM search. A monitor-aligned reward repair suppresses these local proxy
> incentives while preserving dense pre-success shaping and successful completion as the reward-maximizing behavior.
> These results suggest that runtime monitors can serve not only as evaluation tools, but as semantic instruments
> for reward design.

## 11. Future work (three directions)

1. **Learned adversary / RL exploit search** — does a *policy* actually discover the local plateau? (needs RL.)
2. **Cross-task generalization** — apply the audit/repair workflow to coffee-push, dial-turn, and ≥1 more
   manipulation task (mostly offline).
3. **Policy-learning validation** — multi-seed BC fine-tuning first; only later SAC / curriculum from scratch.

**Recommendation: stop here for the current arc and prepare the Kato-facing discussion package.** The reward-repair
result is verified three ways (offline recomputation, decoupled anti-farming, real-env adversarial search) with
honest scoping; the three directions above are larger, gated follow-ons.

## Report index

| topic | report |
|---|---|
| reward SoT → LiNGAM-SH; Stage A ablation / positive control / multi-seed | `reports/2026-07-09-metaworld-reward-*.md` |
| Stage B setup / result / multi-seed / PPO; Kato brief / index / claim discipline | `reports/2026-07-09-metaworld-stage*.md` |
| from-scratch + sanity diagnostics + optimizer repair | `reports/2026-07-{09,10}-*from-scratch*.md`, `…-ppo-optimizer-repair.md` |
| monitor-aligned repair (R1–R3) + anti-farming + adversarial search | `reports/2026-07-10-*monitor-aligned*.md`, `…-adversarial-farming-search.md` |
| prior syntheses + next-step alternatives | `reports/2026-07-10-*final-synthesis.md`, `…-next-step-alternatives.md` |

No experiments were run for this document. CORE.YAML / `pyproject.toml` / FANUC / coin-collab untouched.
