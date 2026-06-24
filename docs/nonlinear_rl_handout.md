# Nonlinear reward, strategy & action models — a handout

*HSiKAN / Kato collaboration · 2026-06-24 · hymeko_rl + signed_kan*

## The one idea

The framework already owns the right nonlinear primitive — the **KAN univariate edge function** (a learnable
Catmull-Rom spline, `signed_kan.CatmullRomActivation` / `EdgeActivation`). Today it lives **only inside the
message-passing backbone**. Everywhere a value is *read out* or *combined*, the pipeline is still **linear**:

| site | today (linear) | the gap |
|---|---|---|
| **reward** | `r = Σ wᵢ·termᵢ` (scalar weights) | no nonlinear shaping of a term, no nonlinear interaction *between* terms |
| **action model** | `aₘ = Linear(features)` | nonlinear backbone, **linear head** (SAC adds only a tanh squash) |
| **strategy** | scalar constants (`ent_coef`, `lr`, …) | no declared nonlinear schedules |

Individual reward terms *can* be nonlinear in state (`pick_lift = min(lifted, thresh)`), but the **combination**
is linear. The fix is one principle applied at three sites: **push the KAN nonlinearity from the backbone out to
the reward arcs, the action head, and the strategy schedule.**

> Same lesson as before: binary incidence broke *signed* hypergraphs; a missing highway broke *HSiKAN*; a linear
> readout/combination under-uses the *KAN* premise. The nonlinear primitive exists — extend its reach.

---

## 1. Nonlinear reward

Two complementary, **declarative** mechanisms (both extend the just-landed arc-weight reward — weights on the
hyperedge):

**(a) Per-arc nonlinearity (KAN reward).** Generalise the arc payload from a scalar weight `wᵢ` to a univariate
op `φᵢ`:
```
r = Σ φᵢ(termᵢ)            # linear weight is the special case φ(x) = w·x
@grasp_reward {
    (+ approach square 4.0,    // φ = 4·(·)²        — sharper near-contact gradient
     + zone   gate   10.0,     // φ = 10·𝟙[·>τ]     — sparse cliff
     + center 5.0);            // linear (back-compat)
}
```
`φ` = a **named op** (`square`, `abs`, `clip`, `tanh`, `gate`) for an interpretable, readable reward — **or** an
opt-in **learnable Catmull-Rom spline** (reuse `signed_kan`) for a true KAN-reward. The reward becomes a
Kolmogorov–Arnold sum of univariate nonlinearities over the term hyperedge.

**(b) Cross-term combinator (structural).** A reward node that combines *several* terms nonlinearly:
```
@grasp_success: product(both_contact, in_zone);   // AND, not "either"
```
The linear sum gives partial credit for *either* contact *or* in-zone; a **product / gate / min** requires
*both* — which is what "a successful grasp" actually means. This is a small computation graph on the term
hyperedge (product / min / max / gated-sum).

**Reuse:** the arc-weight reader (`_profile.read_bundle`, `reward.read_reward_terms`) — the payload generalises
from one number to `(op, weight)`; `signed_kan.CatmullRomActivation` for the learnable variant.

---

## 2. Nonlinear action model

- **KAN actor head.** Replace `actor_mean = Linear(feat, action_dim)` with a **Catmull-Rom readout**
  (`signed_kan` edge functions feat→action). The action map becomes nonlinear end-to-end, not just the trunk.
  (This is the "defuzzification head" idea — a KAN/rotor readout that collapses features to a bounded action.)
- **Squash / bounded transform.** SAC already squashes (`a = scale·tanh(μ+σε)`, with the tanh change-of-vars).
  Generalise the option to PPO and the deterministic actors so the action model is bounded + nonlinear uniformly.

**Reuse:** `signed_kan.CatmullRomActivation`/`EdgeActivation`; SAC's `SquashedGaussianActor` (the bounded
nonlinear transform already exists off-policy).

---

## 3. Nonlinear strategy

- **Declared schedules.** Let the strategy spec carry a **schedule** for `ent_coef` / `lr` / `log_std`
  (cosine, exp-anneal, staged-curriculum) instead of a constant — `curriculum_iters` is the seed of this. The
  schedule itself can be a spline (a KAN over training progress).
- **Richer action distribution.** Squashed-Gaussian (have it), then optionally mixture / normalising-flow
  policies for multi-modal control (heavier; later).

**Reuse:** `strategy_spec` (parse a schedule block); the PPO/DDPG loops (apply the scheduled value per iter).

---

## Where it lands (all non-core)

```
signed_kan/            ← the nonlinear primitive lives here (CatmullRomActivation, EdgeActivation)
  ├─ used by: backbone (today)
  └─ extend to ──► reward φ ops      (hymeko_rl/env/reward.py + _profile.py)
                ► KAN actor head     (hymeko_rl/policy.py ActorCritic)
                ► strategy schedule  (hymeko_rl/strategy_spec.py + ppo/ddpg)
```

## The one design choice (for Kato)

**Declarative named ops + combinators** (interpretable — a reward/strategy you can *read* and audit) **vs.
learnable KAN splines** (more expressive, but a partly *learned* reward is unusual and harder to trust in RL).
**Recommendation:** declarative ops + combinators as the default; learnable-spline as an explicit opt-in per arc.

## Suggested order

1. **Nonlinear reward** (bounded, testable, continues the arc-weight work) — named ops + product/gate combinator.
2. **KAN actor head** (reuse `signed_kan`, parity-init so it starts ≈ the linear head).
3. **Strategy schedules** (declarative anneal/curriculum).

Each ships with parity-preserving defaults (linear weight / linear head / constant schedule reproduce today's
behaviour), so the nonlinearity is opt-in and A/B-able against the current results.
