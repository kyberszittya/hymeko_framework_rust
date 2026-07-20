# Next experiment specification — minimal HyMeKo-native 2-actor × 2-critic contact-mode (SPEC ONLY, per §13 NEGATIVE)

The single-variable replay-sampling experiment measured **NEGATIVE** (contact-stratified replay reduced certified
competence vs uniform on the matched seed; see `../..reports/2026-07-20-coin-contact-stratified-replay.md`). Per §13,
**do not continue tuning replay ratios.** The measured blocker is *contact strategy* (P(fingertip-attribution|zone) and
clean-mechanism), and reweighting the SAME single-actor/single-critic replay did not fix it — it made contact quality
worse. This points to a representational limit, not a data-distribution one.

## Specification (NOT implemented in this task)

**Two actors (structural variants, not a config toggle — §6.5 anti-pattern #8):**
- `A_delivery` — bilateral delivery actor: drive the coin into the zone with a clean two-finger push (the CERTIFIED_BILATERAL competence).
- `A_recovery` — contact-recovery actor: re-establish bilateral fingertip contact after loss (the A4 pulse→recontact competence), invoked when contact degrades.
- Composition through the existing option/switch interface (a contact-state gate decides which actor acts), NOT a new monolithic policy.

**Two critics (distinct value heads):**
- `Q_task` — task-delivery value (the delivery-v2b return).
- `Q_mechanism` — mechanism-validity value: scores fingertip-attribution / clean-bilateral-contact / ¬body-shove (the strict predicate's contact conditions), computed from the PUBLIC rollout trace — NOT a new reward term added to the environment.
- The actor objective trades off `Q_task` and `Q_mechanism` (e.g. lexicographic: maximize `Q_task` subject to `Q_mechanism` above a floor), so the policy cannot buy zone entry with a one-finger bulldoze.

**Invariants to keep:** K0 env, four canonical actuators, delivery-v2b reward (unchanged; `Q_mechanism` is a *value head*, not a reward term), the strict predicate, explicit state splits, the canonical rollout eval path.

**Success criterion:** raise P(fingertip-attribution ≥ 0.60 | zone entry) above the CONTROL 0.67 and lift strict-state coverage above 4/18 while retaining state 64102 — multi-seed (≥3), median/IQR.

## Guardrail before building it
The NEGATIVE is a **single matched seed** (though internally consistent across coverage, mechanism metrics, 64102 retention, and attribution margin). A 2–3-seed replication of the CONTROL vs STRATIFIED comparison should confirm the direction before committing to the 2-actor × 2-critic build — a fresh idea is not decisively dead from one seed.
