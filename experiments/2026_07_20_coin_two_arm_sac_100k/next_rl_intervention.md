# Next RL intervention — single specification (CONTACT_STRATEGY)

Derived **only** from the measured strict-gap classification (`strict_gap_results.json`): blocker = **CONTACT_STRATEGY**
(stabilization group — dwell/settle — passes 100 % of zone entries with large margins; every strict miss is
`ATTRIBUTION + MECHANISM`: fingertip attribution 0.47–0.50 vs the 0.60 bar, coupled with a one-finger bulldoze).

**This intervention is specified, NOT implemented in the evaluation task.**

## Do
1. **Do NOT add hold/dwell shaping.** Post-entry stabilization is already solved (dwell median +3.67, settle +0.75) —
   shaping it would optimize a non-bottleneck.
2. **Stratify the demo replay toward high-fingertip-attribution, low-body-shove A1/A4 contact sequences.** Extend the
   existing `stratify_seed` phase weighting with a per-transition *quality* weight = f(fingertip force balance, ¬body
   contact), so the seeded demos over-represent clean bilateral pushes, not one-finger bulldozes. Reuse the existing
   `RolloutStep` fields (`fl`, `fr`, `body_contact`); no new metric.
3. **Implement the explicit demo/online sampling ratio in the existing `ReplayBuffer` sampler** (the §2/§3 ratio that is
   currently only approximated by demo-seeding + the separate BC anchor): 50 % demo / 50 % online for the first 25k
   steps, then 25 % / 75 %, never below 10 % demo before 3 consecutive strict deliveries. One focused regression test on
   the sampler's batch composition.
4. **Retain the BC anchor** (competence-gated `bc_coef` hook, already wired).

## Do not
- Do not change the reward (`galambos_task_deliver_v2b.hymeko` certified), the actor architecture, the strict predicate
  `_valid_delivery`, or the K0 env.
- Do not add a symmetry reward or any term that could overpower target delivery (the L/R attribution signal is for replay
  weighting + acceptance only, per the standing rule).

## Success criterion (unchanged)
Three consecutive deterministic strict deliveries on at least one normal-target state with genuine two-arm participation.
The reproducible certified delivery on state 64102 (10/10) is the existence proof that the configuration *can* reach
strict; the intervention aims to raise fingertip-attribution above 0.60 on the currently-near states (64111 at 0.47).
