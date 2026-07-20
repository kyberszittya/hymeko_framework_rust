# Next experiment — minimal HyMeKo-native actor×critic factorial (SPEC ONLY, per §13 NO_EFFECT)

The generator (GENERATOR_POSITIVE) extended coverage and the curriculum extended certified transport to +0.0253
(SHORT_TRANSPORT_ONLY), but the longer credit horizon did **not** crack the +0.030 STAGE2 barrier: matched n_step=3 vs
n_step=1 left STAGE2 coverage/loose/max-clearance identical across all 8 seeds (all Δ=0, no STAGE2 certification by
either arm). **Distribution + credit-horizon interventions on a single-actor/single-critic policy are exhausted for
long-range transport.** The remaining lever is representational, per §13.

## Factorial to run (NOT implemented here)
A 2×2 over actor count × semantic-critic count, all cells sharing the K0 env, delivery-v2b reward, strict predicate,
observation schema, action semantics, the frozen curriculum corpora, the committed 70/15/15 mix, thread-pinned matched
seeds, and the canonical rollout eval:

| cell | actors | critics |
|---|---|---|
| A | 1 actor | 1 semantic critic (the current baseline) |
| B | 2 actors (bilateral-delivery + contact-recovery, structural variants) | 1 semantic critic |
| C | 1 actor | 2 semantic critics (task-delivery `Q_task` + mechanism-validity `Q_mechanism`) |
| D | 2 actors | 2 semantic critics |

**Primary endpoint:** STAGE2 (+0.030–0.060) reproducibly certified coverage (≥8/10, footprints disjoint), with the
STAGE1 +0.0253 retention and 64102 retention as guards (a cell that gains STAGE2 by destroying earlier competence loses).
**Hypotheses to discriminate:** does a *second actor* (a dedicated transport/recovery mode) or a *second critic* (a
mechanism-validity value that stops the policy trading clean bilateral contact for reach) unlock STAGE2 where neither
config-distribution nor credit-horizon did — and is the 2×2 interaction (D) super-additive over B and C?

## Guardrails carried forward
- Matched, thread-pinned, multi-seed (≥4 seeds × 2 reps); judge on the bootstrap CI, not the median sign.
- Keep the reward/predicate/env fixed; `Q_mechanism` is a *value head*, never a new reward term.
- Structural actor variants are distinct classes (not a `forward()`-time flag), per the anti-pattern rules.
- Treat the first factorial result as a data point, not a verdict (the whole arc needed 8–12 matched seeds to resolve
  each intervention).
