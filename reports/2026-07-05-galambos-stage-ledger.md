# Galambos Stage Ledger

**Date:** 2026-07-05 JST  
**Purpose:** quarantine ambiguous "learned/RL" claims after the coin-toss cleanup pass.

## Stage Labels

Use these labels for all Galambos / k-arm coin-toss artifacts:

| Stage | Meaning | Current best delivery | Honest interpretation |
|---|---:|---:|---|
| `scripted_controller` | Hand-designed push/plow controller, declared in HyMeKo and executed by Python bindings | `0.80-0.84` | Useful controller/demo fallback; not learned policy evidence |
| `bc_clone` | Behaviour-cloned policy trained from scripted-controller demonstrations | `0.44-0.52` | Best current learned artifact; imitation, not RL improvement |
| `rl_refined` | Policy after TD3+BC/SAC/off-policy updates beyond BC | worse than BC in measured runs | Negative result; refinement damages the clone |
| `framework_substrate` | General dataflow event + FSM + monitor machinery usable by multiple scenarios | not implemented by Galambos-specific FSM alone | Required architecture work remains |

## Required Reporting Sentence

Every future result summary should include this sentence shape:

> Scripted controller: X. BC clone: Y. RL-refined policy: Z. Best saved checkpoint came from stage S.

For the current Galambos state:

> Scripted controller: 0.80-0.84. BC clone: 0.44-0.52. RL-refined policy: below BC in measured TD3+BC/SAC continuations. Best saved checkpoint came from `bc_clone` / step-0, not from RL refinement.

## Quarantined Claims

Do not use these claims without qualification:

- "RL achieved 0.52."
- "The framework is now FSM/dataflow based."
- "The declarative Galambos FSM satisfies the framework architecture goal."
- "TD3+BC improved the coin-toss policy."
- "Best learned policy" without saying whether it is `bc_clone` or `rl_refined`.

Preferred wording:

- "The scripted controller delivers around 0.8."
- "The BC clone reaches around 0.5."
- "Off-policy refinement currently degrades the clone."
- "The Galambos FSM is a scenario-local prototype, not the framework-level dataflow/FSM substrate."

## Preservation Rule

Preserve the push/plow controller as a useful reference scenario and demo fallback. Do not let that preservation blur the scientific claim: it is scripted control. Its role is to provide a strong teacher, a physical baseline, and a worked example for the future framework substrate.
