---
title: SAC-residual abstraction/interface audit (pick-place)
date: 2026-07-15
scope: hymeko_rl residual RL — diagnose the repeated 0.458 collapse as a possible space/interface mismatch
status: audit (no kato15, no multi-seed sweep, line NOT closed)
core_touched: none
---

# SAC-residual abstraction audit — is the actor optimizing the right residual in the right space?

**Directive (user, 2026-07-15).** Do **not** close the SAC-residual line. Treat the repeated 0.458 collapse first as
a possible abstraction/interface failure — between observation space, teacher/base-policy space, residual action
space, and executed env action space — not as evidence that RL fails. Determine whether the SAC residual actor is
optimizing the *correct residual problem in the correct space*. No kato15, no multi-seed sweep, no "line closed".

**One-line verdict.** The abstraction is **sound** — zero-residual reproduces the base (0.875), obs are aligned, the
residual is a function of obs, and the teacher residual executed in closed loop scores **1.000**. A proper feedforward
imitation of that target reaches **0.792**; SAC collapses to **0.458** because its critic **overvalues its own
out-of-distribution large-residual actions** (§8) and the `−Q` loss pulls the actor off the imitation basin — an
**off-policy value-overestimation** failure, *not* an interface mismatch. Allowed claim stands: *"simple
anchor-strength scaling did not rehabilitate the current SAC residual implementation."* The forbidden claims (SAC
dead / residual RL dead / line closed) are **not** supported and are **not** made. Fix path: BC the reactive teacher
(0.792 FF, → recurrent) and refine with a **BC-dominant / conservative-critic** method (TD3+BC/CQL/IQL), not vanilla
SAC + weak anchor.

---

## 1. Action-space alignment — exact executed-action formula

Traced through `train/sac.py` + `env/residual_pick_env.py`. The SAC actor is a `SquashedGaussianActor` built with
`action_scale = 1.0`, so its output is **already the normalized residual**, not an env-scaled action:

```text
# actor (rollout, stochastic):   sac.py:74-85
a_sample   = action_scale · tanh(μ(o) + σ(o)⊙ε) = tanh(μ+σε)          ∈ [-1,1]^7   (action_scale = 1.0)
# actor (eval / anchor, greedy): sac.py:87-89
a_mean     = action_scale · tanh(μ(o))          = tanh(μ(o))          ∈ [-1,1]^7

# env receives the actor output AS the residual:  residual_pick_env.py:141-149
r          = clip(a, -1, 1)                                            # a already in [-1,1] → identity
r          = gate(r)                                                   # zeroed outside active modes; modes="all" → no-op
executed   = clip( base(o) + delta ⊙ r , lo, hi )                     # delta = [0.25×6 joints, 0.02 grip]
env.step(executed)

# teacher label (rollout-state anchor):  pick_place_residual_rl.py:88-92
ro_act     = clip( (expert(o) − base(o)) / delta , -1, 1 )            ∈ [-1,1]^7
# anchor loss:  sac.py:341-345
L_anchor   = mean( ( a_mean(o) − ro_act )² )                          # residual [-1,1] space
```

**Executed-action, deterministic:** `a_exec = clip( base(o) + delta · tanh(μ(o)), lo, hi )`.

**Is the anchor in the same space the env receives?** Yes. The env receives `r` (the [-1,1] residual) and maps it to
`executed = clip(base + delta·r)` internally. The anchor compares `a_mean = tanh(μ)` (the residual the actor emits)
against `ro_act = clip((expert−base)/delta)` (the residual that reproduces the expert) — **both in the same [-1,1]
residual space**. When `a_mean = ro_act`, `executed = clip(base + delta·(expert−base)/delta) = clip(expert) = expert`.
The spaces are **consistent**. (Verified numerically in §5/§7.)

## 2. Residual semantics — which form is implemented?

```text
a_exec = clip( a_base + delta · tanh(actor) , lo, hi )
```

The tanh squash is **inside** the residual (the actor output is squashed, then delta-scaled and added to the base),
i.e. the correct bounded form `clip(base + delta·tanh(actor))`. **Not** `tanh(base+δ)`, **not** `scale(tanh(actor))`
replacing the base. Zero residual (`tanh(0)=0`) → `clip(base)` → base. Correct by construction, confirmed in §5.

## 3–4. Teacher/SAC observation & action-space alignment

For 600 states from a base rollout, teacher, base and actor **all read the same `renv._obs`** (there is no
normalization wrapper in `ResidualPickEnv` — raw env obs to all three).

```text
obs: shape (9,10)=90 · no NaN · range [-3.04, 5.72] · identical array to teacher/base/actor
CHECK3 (actor at init, residual≈0):
  anchor-space  |a_mean − ro_act|         = 0.2006     (actor=0, teacher≈0.20 → expected)
  exec-space    |executed(actor) − expert| = 0.0564    ≈ |base − expert| = 0.0558  (actor≈base at init) ✓
```

Teacher and SAC decide from the **same semantic state**. No frame/order/normalization/stale-running-stat mismatch.

## 5. Base-policy preservation (HARD GATE) — PASS

```text
zero-residual actor (μ head zeroed) stable-place = 0.875
explicit zero-residual (_ZeroResidual)           = 0.875
Δ = 0.000  → PASS (the residual chain reproduces the base exactly; base+residual plumbing is correct)
```

## 6. Anchor-only training (no Q) — 0.500, the diagnostic split

Fitting the actor's `a_mean` to the teacher residual on **base-trajectory** states (600 steps supervised, no Q term):

```text
residual-fit MSE            = 0.00066      (the network fits the teacher residual almost perfectly)
|fit_res − teacher_res|     = 0.0106
|executed(fit) − expert|    = 0.0161       (executed ≈ expert to 0.016 rad on the fit states)
EVAL stable-place           = 0.500        ← fits the target perfectly, yet SCORES 0.500 (base = 0.875)
```

The actor reproduces the expert action to 0.016 rad **on the states it was fit on**, yet scores 0.500 when run. That
is the classic **covariate-shift signature**: fit on the base trajectory, executed on the base+residual trajectory
(different state distribution). It is *not* a space/formula error — §5 and §1 rule those out.

## 7. Residual / action-space calibration (per-dim: 6 joints + grip)

```text
mean|expert action|    = [0.194 0.260 2.044 0.003 0.859 0.189 0.027]
mean|base action|      = [0.182 0.241 2.133 0.002 0.821 0.171 0.025]
mean|expert − base|    = [0.027 0.115 0.159 0.002 0.056 0.027 0.004]
mean|teacher residual| = [0.109 0.341 0.450 0.007 0.215 0.111 0.171]   overall 0.201
teacher-res |r|>0.99   = [0.00  0.13  0.24  0.00  0.05  0.00  0.02 ]   (joints 2,3 saturate 13–24%)
|exec(teacher)−expert| = [0.0014 0.0337 0.0500 0.0001 0.0049 0.0014 0.0056]  (target achievable to ≤5 crad)
executed clip@bounds   = [0 0 0 0 0 0 0.993]                            (grip is bang-bang; expected, benign)
```

**Reading.** Residuals are well-scaled (overall 0.20, not tiny, not blown out). Joints 2–3 saturate 13–24 % of the
time — δ=0.25 rad is a touch small there, mildly capping how exactly the residual can match the expert, but
`|exec(teacher)−expert|` is ≤ 0.05 rad everywhere, so the target is essentially achievable. Grip clips at a bound 99 %
of the time because grip is open/closed bang-bang (base 0.025 ± δ0.02 vs range [0,0.035]); benign, not a pathology.

## Part 2 — the teacher target is not weak, and reading it is (here) side-effect-free

Two code-read hypotheses were **tested and refuted** (discriminating-test discipline):

```text
A  reading env.expert_action mutates model gains?   max|Δgainprm| = 0.000   → NO (idempotent; v3 holds a
   joint-1 kp over an episode: [60,60,60,60]            consistent kp60, no phase-flip). Gain-mutation code
                                                        exists (_v2_set_arm_gains) but is inert for v3's path.
B  scripted v3 expert closed-loop stable-place       = 0.958   → ABOVE the base; a strong target, not weak.
C  reactive-teacher-RESIDUAL closed-loop (r=teacher   = 1.000   → executing clip(base+δ·teacher(state)) every
   recomputed live every step, executed)                        step is a PERFECT controller on this metric.
D  base (zero residual)                               = 0.875
```

**C is the crux.** The exact residual target the anchor pulls toward, executed **reactively** (recomputed from the
live state every step), scores **1.000** — better than base (0.875) and expert-own (0.958). So the residual
abstraction, the teacher target, and the action space are not merely correct — they define a **1.000 controller**.
The entire gap is between that reactive oracle and a **network** that must generalize obs→residual off its training
distribution.

## 6b — DAgger discriminator (relabel on the actor's OWN greedy states)

If the collapse is covariate shift, relabeling the teacher on the actor's own greedy trajectory and refitting
(DAgger, aggregated) should climb back toward the 1.000 oracle. If instead it stalls near 0.5, a deeper
capacity/representation issue would remain.

```text
base(zero-residual)=0.875   reactive-teacher(oracle)=1.000   (targets)
DAgger round 0: dataset= 1997  fit-MSE=0.00270  eval stable-place=0.500
DAgger round 1: dataset= 4017  fit-MSE=0.00296  eval stable-place=0.500
DAgger round 2: dataset= 6641  fit-MSE=0.00446  eval stable-place=0.500
DAgger round 3: dataset= 8661  fit-MSE=0.00355  eval stable-place=0.458
DAgger round 4: dataset=11671  fit-MSE=0.00458  eval stable-place=0.500
DAgger round 5: dataset=14681  fit-MSE=0.00502  eval stable-place=0.583
DAgger round 6: dataset=16701  fit-MSE=0.00459  eval stable-place=0.583
```

**DAgger did NOT recover the base** — it stalled at 0.58 (≪ base 0.875 ≪ oracle 1.000), with the fit MSE floored at
~0.004 (RMS ≈ 0.06 in [-1,1] residual space). So the failure is **deeper than covariate shift**: relabeling on the
actor's own states, aggregated over 7 rounds, does not close it. The residual is fit to ~0.06 RMS yet the executed
policy caps at 0.58 — the task is hypersensitive to residual error, and the network cannot drive the error lower.
Two hypotheses remain, resolved in §6c: **H-a** underfitting (capacity/training) vs **H-b** the teacher residual is
**history-dependent** (a latched/integral expert not representable by a memoryless actor).

## 6c — H-a (underfitting) vs H-b (history-dependence): the representability test

```text
reactive-teacher dataset: 4126 states, obs dim 90, residual std/dim=[0.131 0.196 0.101 0.009 0.149 0.135 0.243]
(1) HISTORY-DEPENDENCE probe:
  overall residual pair-gap median                        = 0.012
  near-identical-obs (<10th pct ||Δobs||=0.010) gap median= 0.001     ← near-obs → near-residual
  residual full-vector norm median                        = 0.197
(2) HIGH-CAPACITY FF BC (hidden=128, 4000 steps, reactive-teacher distribution):
  final train MSE=0.00246 (RMS=0.050)   eval stable-place = 0.792     ← far above DAgger 0.58 / SAC 0.458
```

**H-b refuted, H-a supported.** Near-identical observations map to near-identical residuals (gap 0.001 ≪ residual
norm 0.197), so the residual is essentially a **function of obs** — a memoryless actor is not fundamentally blocked.
(Caveat: a state's nearest neighbor is often its temporal neighbor, so (1) is suggestive; (2) is the decisive
evidence.) A **high-capacity feedforward BC of the reactive teacher reaches 0.792** — the earlier DAgger 0.58 was
underpowered (hidden 64, 300 steps/round). So a feedforward residual *can* be learned to ~0.79; the residual
0.79→1.0 gap is the ordinary clone gap on a hypersensitive task (recurrence may narrow it — open, not required).

**The reframed question this answers:** a proper imitation of the residual target reaches 0.792, yet **SAC collapses
to 0.458**. The bottleneck is therefore the **SAC reward-optimization (−Q) pulling the actor off the imitation
basin**, not the abstraction and not the observation space. §8 tests that mechanism directly.

## 8. Critic interface sanity — the collapse mechanism, confirmed

Gated on §1–7 passing (they do). A short SAC (rehab, no anchor) was trained 12k steps to the collapse (→0.500),
then its critic's Q was probed on reactive-teacher states:

```text
Q(collapsed actor residual) = -0.687   ← HIGHEST; the critic's favorite    (|actor residual| mean = 0.707)
Q(teacher residual)         = -1.164
Q(near-teacher +0.05 noise) = -1.164
Q(zero residual = base)     = -1.181
Q(uniform random)           = -1.221                                        (|teacher residual| mean = 0.094)
```

**The critic ranks the collapsed actor's large residual ABOVE the teacher's** (−0.687 ≫ −1.164), and assigns the
teacher, the base, and *random* nearly the **same low Q** (~−1.16 to −1.22). This is classic **off-policy value
overestimation**: the critic inflates Q on the actor's own out-of-distribution, near-saturated residuals (|r|=0.707)
and fails to credit the good small teacher/base residuals (|r|=0.094). The `−Q` actor loss therefore drives the
policy *toward* the large destructive residual and *off* the imitation basin — and the rollout anchor at coef 1/5/20
cannot outweigh a −0.5-nat Q advantage. **This — not the abstraction — is the collapse.**

## 9. Decision

- **Executed-action formula:** `a_exec = clip(base(o) + delta·tanh(μ(o)), lo, hi)`, delta=[0.25×6, 0.02]. (§1)
- **Teacher/SAC obs:** identical raw `renv._obs`, aligned, no normalization mismatch. (§3–4)
- **Zero-residual base score:** 0.875 = base (HARD GATE **PASS**). (§5)
- **Anchor-only:** fits the target to 0.016 rad but scores 0.500 → covariate shift, not space error. (§6)
- **Calibration:** residuals well-scaled; joints 2–3 mildly δ-capped; grip bang-bang (benign). (§7)
- **Target quality:** reactive-teacher-residual closed-loop = **1.000**; v3 expert = 0.958 (both > base). (Part 2)
- **Collapse attribution:** **off-policy critic value-overestimation (an RL-optimization failure), NOT an
  abstraction/interface mismatch.** Every interface check passes (§1–7); a proper feedforward imitation of the target
  reaches 0.792; the reactive oracle reaches 1.000. SAC collapses because its critic overvalues its own OOD
  large-residual actions (§8) and the `−Q` loss pulls the actor off the imitation basin. The earlier "anchor-strength"
  reading (F-SAC-3) was a symptom: the anchor can't win against an overestimating critic, no matter the coefficient.
- **kato15 justified yet?** **No.** No config has been *locally vetted* to beat the base (0.875). A vanilla-SAC
  multi-seed sweep would reproduce the 0.458 collapse on the GPU. kato15 also needs a full env build (no venv, no
  base ckpt, stale `main`) that must not be stood up for an unvetted config. kato15 becomes justified once a local
  config **beats 0.875**.

### Recommended next step (local, not anchor-strength, not closing the line)
The collapse is a **critic-distribution** problem, so the fix is a method that constrains the critic to the data —
exactly the codebase's own coin-toss lesson, applied correctly:
1. **Imitation already works** — a feedforward BC of the reactive teacher = **0.792** (and the oracle = 1.000). Wire
   the reactive teacher as a proper demonstrator and BC-clone it (optionally a **recurrent** actor to chase the
   0.79→1.0 clone gap — ties to the existing LSTM pick clone).
2. **If RL refines past imitation, use a BC-dominant / conservative-critic method** already present in `train/sac.py`
   (`bc_coef` demo anchor = TD3+BC-style) or add CQL/IQL/AWAC — **not** vanilla SAC with a weak rollout anchor. Warm-
   start the actor from the 0.792 BC clone and keep a strong demo anchor so the critic cannot drag it off.
3. Only after a local config **beats 0.875** on 24-ep eval → kato15 multi-seed.

**Allowed claim (only):** *simple anchor-strength scaling did not rehabilitate the current SAC residual
implementation; the abstraction is sound and the failure is off-policy critic overestimation.* **Not** claimed: SAC
cannot work / residual RL cannot work / the line is closed.

### What would justify closing *this exact implementation* (none yet satisfied together)
zero-residual reproduces base ✅ · obs aligned ✅ · anchor in executed/justified space ✅ · anchor-only
preserves/recovers base ❌ (covariate shift, §6/§6b decides) · residual/clip/tanh sane ✅ · critic clean (§8) ·
collapse persists under bounded local tests after all the above. The line is **not** closed.

---

## Provenance
- Git branch `integration/fanuc-pick-place-canonical`; base ckpt `experiments/hybrid_dagger_gif/policies/hybrid_dagger_hsikan_s0_best.pt`.
- Probes: `scratchpad/audit_sac_residual{,2,3}.py` (seeded, deterministic; N=24 eval @ seed0=20000).
- Env: `fanuc_pick_env(expert_version=3, require_settle=True, max_steps=1000)`, δ=0.25/grip0.02, reward=settle.
- No CORE touched. No persistent state mutated. No kato15 launch. No multi-seed sweep.
