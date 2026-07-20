# Next experiment — competence-gated replay sampler (SPEC ONLY, per the 12-seed BASIN_DEPENDENT result)

The 12-seed replication (`../../reports/2026-07-20-coin-contact-replay-12seed.md`) showed fixed contact-stratified
replay has **no average effect** (strict median −0.5, bootstrap CI [−0.83, +0.42] spanning zero, no compensating
contact-quality gain) but is **basin-dependent**: Spearman(CONTROL strict, Δstrict) = −0.825; weak basins (CONTROL
strict ≤2) improve by median +1.0, strong basins (≥4) degrade by median −1.0. Contact-stratified replay is a
**mean-reverting regularizer** — it helps only when uniform replay lands poorly.

**Do NOT** build the N-actor × k-critic architecture on this evidence. **Do NOT** tune the always-on stratified ratios
further (the effect is basin-dominated, not ratio-dominated).

## The single next experiment: gate stratification on measured competence

**Idea:** apply the stratified sampler *only while contact competence is weak*, and revert to uniform replay once
certified/contact competence is established — so the regularizer acts in the basins where it helps and is removed from
the basins where it hurts.

**Reuse the existing machinery (no new framework):**
- The competence state `comp` already tracks `progress_ok` / `first_strict` / `consec_strict` and already drives
  `bc_coef_fn`. Add a `demo_frac_fn` that returns the stratified fraction **while `comp` indicates weak competence**
  (e.g. `not comp["first_strict"]`) and **0.0 (→ uniform `buf.sample`) once competence is established**. The existing
  `train_sac` `_stratified` switch already falls through to uniform when `demo_frac_fn` yields no demo draw; extend it so
  a per-step gate can turn stratification off mid-run (a small, in-place `train_sac` change, guarded by a regression
  test — not a new sampler).
- Keep the corpus, strata definitions, BC anchor, reward, strict predicate, K0 env, state splits, eval path unchanged.

**Matched comparison:** UNIFORM (control) vs COMPETENCE-GATED-STRATIFIED (treatment), the SAME 12 seeds, same source
checkpoint. Primary endpoint: 12-seed **mean/median paired Δ strict + bootstrap CI**, and — critically — the
**strong-basin subgroup** (CONTROL strict ≥4): the gate is successful iff it removes the strong-basin degradation
(strong-group median Δ ≥ 0) **without** losing the weak-basin gain (weak-group median Δ > 0). I.e. break the −0.825
inverse coupling toward a non-negative average.

**Classification to pre-register:** GATE_POSITIVE (strong-basin harm removed AND weak-basin gain kept AND 12-seed median
Δ ≥ 0) / GATE_NEUTRAL (average still ~0, basin coupling weakened but no net gain) / GATE_NEGATIVE (no improvement over
always-on). Only GATE_POSITIVE would justify retaining a (gated) sampler; GATE_NEUTRAL/NEGATIVE retires fixed
replay-distribution interventions and moves to the minimal HyMeKo-native contact-mode specification.

**Guardrail:** this is the FIRST test of the competence-gated variant — treat its first 12-seed result as a data point,
not a verdict, exactly as the always-on version required 12 seeds to resolve.
