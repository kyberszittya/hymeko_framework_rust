# EPIGENE-16: Detailed Architecture Description

**Date:** 2026-07-06 JST  
**Status:** design description / terminology lock-in  
**Related systems:** HyMeKo, HIVE, HOTARU, AKOIRE, Galambos controller work  
**Core idea:** learning should regulate *structured expression knobs*, not seize
raw control of the whole system.

## Executive Summary

EPIGENE-16 is a proposed HyMeKo control layer for expressing adaptive behavior as
a small, typed, auditable vector of regulatory parameters. It is "epigenetic" in
the computational sense: it does not rewrite the organism's genotype, grammar, or
hard physical law. Instead, it modulates which already-valid structural programs,
controllers, monitors, and repair paths are expressed under the current context.

The practical motivation comes from the July Galambos / k-arm coin-toss work. Raw
or weakly constrained neural control could clone useful behavior, but refinement
often degraded contact behavior. The successful direction was not "give the
network more authority"; it was to expose a bounded high-level controller
interface and put a phase controller plus safety arbiter around it. EPIGENE-16
generalizes that lesson:

> A learned system may tune bounded expression parameters, but HyMeKo keeps the
> grammar, transactions, monitors, safety gates, and acceptance criteria.

The "16" denotes a fixed-width expression profile: sixteen named regulatory
channels. Each channel is small enough to inspect, log, diff, clamp, replay, and
verify, but expressive enough to steer planning, learning, controller selection,
repair, and experiment scheduling.

## Position In The Stack

EPIGENE-16 is not a replacement for HIVE, HOTARU, or AKOIRE. It sits between the
cognitive/planning layer and concrete runtime execution.

```text
Intent / task / experiment goal
        |
        v
AKOIRE cognitive loop
        |
        v
HOTARU hypergraph planner
        |
        v
EPIGENE-16 expression profile
        |
        v
HIVE transactions + HyMeKo parser/IR gate
        |
        v
FSM / dataflow runtime / monitors / controllers
```

AKOIRE decides what kind of refinement is being attempted. HOTARU proposes
candidate hypergraph deltas or plans. EPIGENE-16 determines how strongly each
class of behavior is allowed to express itself in this context. HIVE commits only
valid transactional state. The runtime executes only accepted structure, subject
to monitors and acceptance gates.

## What EPIGENE-16 Is

EPIGENE-16 is a typed regulatory profile:

```text
Epigene16 {
  channels: [ExpressionChannel; 16],
  parent_hash: HiveHash,
  context: task / scenario / phase / actor / monitor scope,
  provenance: planner id, model id, seed, evidence id,
}
```

Each channel has:

- a stable name,
- a normalized value, usually in `[0, 1]` or `[-1, 1]`,
- a semantic type such as gain, threshold, priority, budget, confidence, or
  permission,
- a hard clamp,
- an owner layer,
- a monitor contract,
- a reason trail explaining why the current value was selected.

The important distinction is that a channel is not just a floating point number.
It is a signed piece of governance: "this subsystem may express this much of this
capability under these conditions."

## What EPIGENE-16 Is Not

EPIGENE-16 is not:

- a raw action vector,
- a hidden neural latent with no names,
- an unconstrained optimizer state,
- a replacement for `.hymeko` source or HIVE transactions,
- a way to let an LLM edit core state directly,
- a proof that a controller is safe,
- a substitute for monitors or acceptance tests.

If the vector cannot be explained, diffed, clamped, or rejected, it is not
EPIGENE-16. It is just another latent.

## The Sixteen Channels

The first useful EPIGENE-16 layout should be domain-general enough for HyMeKo
workflows, while still mapping cleanly onto robotics/RL experiments.

| # | Channel | Type | Meaning |
|---:|---|---|---|
| 1 | `expression_gain` | gain `[0,1]` | How strongly learned/adaptive behavior may influence the active program. |
| 2 | `safety_margin` | threshold `[0,1]` | Extra distance from constraint boundaries before monitors intervene. |
| 3 | `exploration_budget` | budget `[0,1]` | How much novelty/search is allowed before falling back to known-safe behavior. |
| 4 | `repair_priority` | priority `[0,1]` | Bias toward fixing existing structure rather than generating new structure. |
| 5 | `reuse_bias` | priority `[0,1]` | Preference for existing HIVE artifacts, demos, controllers, and profiles. |
| 6 | `mutation_temperature` | gain `[0,1]` | Breadth of HOTARU's candidate-delta search. |
| 7 | `transaction_strictness` | threshold `[0,1]` | How conservative HIVE commit acceptance should be. |
| 8 | `monitor_sensitivity` | threshold `[0,1]` | How early monitors fire on drift, leakage, unsafe contact, or protocol failure. |
| 9 | `phase_adherence` | gain `[0,1]` | Strength of FSM/phase-controller authority over learned outputs. |
| 10 | `contact_conservatism` | threshold `[0,1]` | Bias toward preserving established contact/manifold behavior. |
| 11 | `baseline_guard` | threshold `[0,1]` | Required retention of scripted/known baseline performance. |
| 12 | `credit_assignment_scope` | budget `[0,1]` | How far reward/blame may propagate across graph, time, and actor boundaries. |
| 13 | `llm_authority` | permission `[0,1]` | Maximum authority of LLM-generated refinements before formal gates. |
| 14 | `formalization_pressure` | priority `[0,1]` | Bias toward turning soft descriptions into typed HyMeKo/HIVE structures. |
| 15 | `evidence_requirement` | threshold `[0,1]` | Required measurement strength before promoting a result or plan. |
| 16 | `rollback_readiness` | budget `[0,1]` | Amount of state/provenance retained for replay, rollback, and quarantine. |

This layout is intentionally not task-specific. A Galambos controller may bind
channels 9-11 strongly. A seminar-deck generator may use channels 5, 14, and 15.
A HIVE query planner may use channels 4, 6, 7, and 16. The profile remains one
language for expression control.

## Channel Capacity And Compression Rate

EPIGENE-16 is a small language, so it needs an explicit capacity model. The
important point is that its useful capacity is not the same as the raw number of
floating-point states. A 16-float vector has many possible bit patterns, but most
of those patterns are meaningless, unsafe, redundant, or unverifiable. EPIGENE-16
should therefore be measured as a **semantic control channel**.

### Raw Capacity

If each of the 16 channels is quantized into `q_i` admissible levels, the maximum
raw capacity is:

```text
C_raw = sum_i log2(q_i) bits
```

For uniform quantization:

| Levels per channel | Bits / channel | Total raw capacity |
|---:|---:|---:|
| 2 | 1 | 16 bits |
| 4 | 2 | 32 bits |
| 8 | 3 | 48 bits |
| 16 | 4 | 64 bits |
| 32 | 5 | 80 bits |
| 64 | 6 | 96 bits |
| 256 | 8 | 128 bits |
| 65,536 | 16 | 256 bits |

The practical default should not be 32-bit floats. For governance and
verification, 4 to 8 bits per channel is usually more useful than continuous
precision. A profile with 16 channels at 8 bits each is only 128 raw bits, but
already has far more resolution than most monitor policies can justify.

### Effective Capacity

The effective capacity is lower than the raw capacity because channel values are
constrained by policy, monitors, and cross-channel invariants:

```text
C_effective = log2(|A|)
```

where `A` is the set of admissible profiles after all constraints are applied.
Examples of constraints that shrink `A`:

- `llm_authority <= transaction_strictness`,
- high `expression_gain` requires high `evidence_requirement`,
- high `mutation_temperature` requires nonzero `rollback_readiness`,
- low `phase_adherence` is forbidden in contact-critical controller scopes,
- `baseline_guard` must map to a concrete measurable floor,
- any profile with high `monitor_sensitivity` must attach monitor relations.

This shrinkage is a feature. EPIGENE-16 is supposed to remove unsafe degrees of
freedom from the communication channel. A 128-bit raw profile might have only
40-70 bits of admissible capacity in a safety-critical robotics scope, and that
is preferable to a larger ungoverned latent.

### Semantic Capacity

The most important capacity is not `C_raw` or even `C_effective`, but semantic
capacity:

```text
C_semantic = log2(number of behaviorally distinguishable admissible profiles)
```

Two profiles are behaviorally indistinguishable if they produce the same accepted
plan/controller/monitor behavior under the same context. For example, if the
phase controller is locked, changing `expression_gain` from `0.21` to `0.22` may
not change any runtime decision. Those two raw states should count as one
semantic state.

This suggests an empirical measurement:

1. Sample or enumerate admissible EPIGENE-16 profiles.
2. Lower each profile into the target subsystem: HOTARU planning, HIVE commit
   strictness, or Galambos controller refinement.
3. Hash the resulting accepted behavior class: chosen plan, accepted/rejected
   delta set, controller parameter bucket, monitor firing pattern.
4. Count distinct behavior classes.

Then:

```text
C_semantic ~= log2(distinct_behavior_classes)
```

This is the capacity that matters for LLM-to-HIVE communication: how many
different *auditable intentions* the language can express, not how many numeric
vectors it can store.

### Compression Rate

EPIGENE-16 is also a compression language. It compresses a large, messy
decision context into a small expression profile:

```text
compression_ratio = source_description_bits / epigene_profile_bits
compression_rate  = epigene_profile_bits / source_description_bits
```

The source description may include:

- an LLM plan,
- a controller state,
- a replay-buffer diagnosis,
- monitor traces,
- HIVE query results,
- a scenario report,
- a set of candidate HOTARU deltas.

Approximate examples:

| Source state | Rough source size | EPIGENE-16 payload | Compression ratio |
|---|---:|---:|---:|
| short LLM instruction, 1 KB text | ~8,000 bits | 128 bits | ~62:1 |
| experiment summary, 10 KB text | ~80,000 bits | 128 bits | ~625:1 |
| compact monitor trace, 100 KB | ~800,000 bits | 128 bits | ~6,250:1 |
| neural checkpoint, 1 MB | ~8,000,000 bits | 128 bits | ~62,500:1 |

This is not lossless compression. It is **policy compression**: it throws away
details that should not be used as direct control authority and keeps only the
small set of regulatory decisions that the runtime can verify.

### Rate-Distortion View

The right question is not "how many bits can EPIGENE-16 carry?" but:

> How few bits can preserve the decisions that matter?

For a task family `T`, define distortion as the behavioral loss caused by using
an EPIGENE-16 profile instead of the full source context:

```text
D = loss(full_context_policy, epigene_policy)
```

Possible distortions:

- lower delivery rate in Galambos,
- extra rejected HIVE transactions,
- slower HOTARU convergence,
- more false-positive monitor triggers,
- missed reuse of an existing artifact,
- failure to quarantine an unsafe import.

The target is a low-rate, low-distortion control code. For many governance
decisions, 16 channels at 4-8 bits each should be enough because the runtime only
needs coarse regimes:

```text
reject / cautious / normal / exploratory
locked / bounded / adaptive
reuse / repair / regenerate
untrusted / provisional / accepted
```

That is why EPIGENE-16 should prefer calibrated buckets over continuous values.
The profile is meant to be robust under small numerical noise.

### Noise And Error Correction

The language should assume noisy producers: LLMs, heuristic planners, partial
experiments, and stale reports. EPIGENE-16 gets reliability from redundancy and
constraints:

- `llm_authority` and `transaction_strictness` cross-check each other,
- `expression_gain` and `baseline_guard` cross-check learned-control authority,
- `mutation_temperature` and `rollback_readiness` cross-check exploration risk,
- `monitor_sensitivity` and `evidence_requirement` cross-check claim promotion,
- `phase_adherence` and `contact_conservatism` cross-check robotics contact
  safety.

These pairs are deliberate redundancy. They reduce capacity, but increase
recoverability. A corrupted or overconfident profile is more likely to violate a
cross-channel invariant and be rejected before execution.

### Capacity Budget By Use Case

Different contexts should expose different effective capacities:

| Context | Recommended quantization | Effective capacity target | Reason |
|---|---:|---:|---|
| quarantine / untrusted import | 2-4 levels/channel | ~16-32 bits | only coarse safety regimes should be expressible |
| robotics contact refinement | 4 levels/channel | ~24-40 bits after constraints | preserve phase/contact authority |
| exploratory design | 8 levels/channel | ~48-80 bits | allow wider planning variation |
| offline analysis / report classification | 8-16 levels/channel | ~64-128 bits | no direct runtime authority |
| formal proof profile | symbolic buckets | small finite state set | easier model checking |

This gives EPIGENE-16 a useful design rule:

> Runtime authority should decrease as channel capacity increases, unless formal
> verification also increases.

High-capacity profiles are good for offline analysis. Low-capacity profiles are
better for direct control.

### Compression Quality Metrics

A first implementation should log enough data to measure compression quality:

- `raw_bits`: sum of channel quantization widths,
- `admissible_bits`: estimated `log2(|A|)` under active constraints,
- `semantic_bits`: estimated `log2(distinct_behavior_classes)`,
- `source_bits`: approximate input size being compressed,
- `compression_ratio`: `source_bits / raw_bits`,
- `distortion`: task-specific loss versus full-context or baseline behavior,
- `rejection_rate`: fraction of profile deltas rejected by HIVE/monitors,
- `stability`: behavior-class change under one-bucket perturbations.

The stability metric is especially important. A good EPIGENE-16 language should
be locally stable: small channel perturbations should usually preserve behavior
class, except near explicit thresholds.

### Design Consequence

The EPIGENE-16 language should be specified as **quantized typed buckets first**
and floats second. Floats may be useful internally, but the committed HIVE form
should expose canonical buckets:

```text
off < low < medium < high < locked
```

or fixed integers:

```text
0..15   # 4-bit channel
0..255  # 8-bit channel
```

The bucketed representation gives the system finite capacity, finite model
checking, stable diffs, and meaningful compression-rate accounting. It also keeps
the LLM interface honest: the model chooses a named regime, not a fake-precise
continuous value.

## SymPy / Z3 Theorem-Checking Start

The first theorem layer should be deliberately small. Do not start by trying to
prove the whole controller, planner, or LLM loop. Start with the finite language:
channel count, bucket bounds, admissibility constraints, and compression algebra.

The repo already has the right pattern:

- `verification/propositions/p4_storage_overhead.py` uses SymPy for symbolic
  algebra and monotonicity;
- `verification/cross_view_consistency/commute_z3.py` uses Z3 to prove an
  architectural implication by showing the negation is unsatisfiable, then proves
  the guard is load-bearing by finding a counterexample when the guard is removed.

EPIGENE-16 should follow the same split.

### SymPy: Algebraic Claims

Use SymPy for claims with closed-form expressions:

```text
C_raw = n_channels * bits_per_channel
compression_ratio = source_bits / C_raw
compression_rate = C_raw / source_bits
```

The starter SymPy obligations are:

1. derive raw capacity from uniform quantization,
2. prove capacity increases with channel count,
3. prove capacity increases with bucket width,
4. prove compression ratio increases with source-context size,
5. prove compression ratio decreases as the committed profile gets wider,
6. reproduce witness values such as 16 channels × 8 bits = 128 bits and
   8,000 / 128 = 62.5:1 compression.

This is not deep mathematics, but it is useful because it prevents later reports
from drifting between "capacity", "compression ratio", and "compression rate".

### Z3: Finite Bucket Invariants

Use Z3 for the finite governance language. Start with 4-bit buckets:

```text
channel_i in {0, ..., 15}
LOW = 4
MEDIUM = 8
HIGH = 12
```

The first admissibility predicate should include:

```text
0 <= channel_i <= 15
llm_authority <= transaction_strictness
expression_gain >= HIGH -> evidence_requirement >= HIGH
mutation_temperature >= MEDIUM -> rollback_readiness >= MEDIUM
monitor_sensitivity >= HIGH -> monitor_attached
contact_critical -> phase_adherence >= HIGH
contact_critical -> contact_conservatism >= HIGH
contact_critical -> baseline_guard >= HIGH
```

Positive theorems:

- no admissible profile lets `llm_authority` bypass
  `transaction_strictness`;
- no contact-critical admissible profile has low `phase_adherence`;
- no contact-critical admissible profile has low `contact_conservatism`;
- no contact-critical admissible profile has low `baseline_guard`;
- no high-expression profile has low `evidence_requirement`.

Negative/load-bearing checks:

- remove the contact-critical implication and Z3 should find an unsafe contact
  model;
- remove the authority constraint and Z3 should find an LLM-authority bypass;
- remove rollback readiness and Z3 should find high mutation without replay
  support.

This negative half matters. It tells us which constraints are actually carrying
the safety argument.

### Starter Files

The initial executable scaffold is:

```text
verification/epigene16/README.md
verification/epigene16/capacity_sympy.py
verification/epigene16/invariants_z3.py
verification/epigene16/tests/test_epigene16_verification.py
```

Run:

```powershell
uv run python verification/epigene16/capacity_sympy.py
uv run python verification/epigene16/invariants_z3.py
uv run pytest verification/epigene16/tests -q
```

2026-07-06 local verification note: `uv run` is currently blocked by an
unrelated workspace metadata issue (`signedkan_wip/signedkan_native` editable
package path is missing). The same checks were run through the existing local
virtualenv:

```powershell
.\.venv\Scripts\python.exe verification\epigene16\capacity_sympy.py
.\.venv\Scripts\python.exe verification\epigene16\invariants_z3.py
.\.venv\Scripts\python.exe -m pytest verification\epigene16\tests -q
```

Result:

```text
capacity_sympy.py: EPIGENE-16 capacity algebra verified: True
invariants_z3.py: EPIGENE-16 finite-bucket invariants verified: True
pytest: 3 passed
```

The goal of this first layer is not to prove that EPIGENE-16 makes a robot safe.
The goal is to prove that the EPIGENE-16 *language* is finite, bounded,
load-bearing, and rejects obvious governance violations before runtime.

### Next Theorem Boundary

Once the finite profile checker exists, the next boundary is HIVE:

```text
Epigene16 profile
  -> HIVE relation encoding
  -> HIVE transaction commit
  -> loaded Epigene16 profile
```

The theorem shape should be:

```text
If a profile is admissible and its HIVE transaction commits,
then loading the committed HIVE state recovers the same canonical profile.
```

That theorem is not pure SymPy/Z3 anymore; it needs property tests over the real
HIVE encoder/decoder plus Z3 for the finite admissibility predicate. Keep those
layers separate:

- Z3 proves profile admissibility properties;
- property tests prove encoder/decoder round-trip over generated admissible
  profiles;
- HIVE transaction tests prove stale-parent and idempotent replay behavior.

## Relation To The Galambos Controller Gate

The current Galambos lesson is the clearest concrete example. The accepted policy
is:

- neural output may tune bounded high-level controller parameters,
- the phase controller owns the state machine,
- the safety arbiter may reject bad parameter targets,
- delivery below the floor is rejected, not interpreted as a trade-off.

EPIGENE-16 lifts that pattern into a reusable profile:

| Galambos mechanism | EPIGENE-16 equivalent |
|---|---|
| `PushControllerParams` | task-local expression outputs |
| `PhasePushController` | high `phase_adherence` |
| safety arbiter | high `safety_margin`, `contact_conservatism`, `monitor_sensitivity` |
| delivery floor | high `baseline_guard` |
| post-fix rerun/probe requirement | high `evidence_requirement` |
| quarantine of untrusted Fable artifacts | high `rollback_readiness`, high `transaction_strictness` |

This matters because it prevents the common failure mode where learning degrades
a narrow contact behavior while reporting partial reward progress. EPIGENE-16
makes the acceptance boundary explicit: the profile may tune expression, but the
runtime owns admissibility.

## Relation To HIVE

HIVE is the canonical database-like store for signed typed hypergraphs.
EPIGENE-16 should be persisted as HIVE state, not as an untracked side file.

A profile can be represented as:

- one `EpigeneProfile` node,
- sixteen `ExpressionChannel` nodes,
- one relation per channel binding profile, channel name, value, bounds, owner,
  and monitor contract,
- provenance relations to the parent HIVE hash, planner, experiment, and evidence
  artifacts.

This makes EPIGENE-16 queryable:

```text
node.type:EpigeneProfile
node.type:ExpressionChannel
relation.endpoint_type:+:Monitor
relation.attr:baseline_guard
```

It also makes it transactional. A HOTARU planner may propose a profile delta, but
HIVE still checks parent hashes, idempotence, and atomic commit behavior. In
distributed workflows this is the difference between "a model changed a knob" and
"a typed, replayable transaction updated expression state."

## Relation To HOTARU

HOTARU should treat EPIGENE-16 as a planning bias and as a candidate artifact.

As planning bias:

- `mutation_temperature` controls search breadth,
- `repair_priority` biases patch-vs-regenerate decisions,
- `reuse_bias` makes existing artifacts attractive,
- `transaction_strictness` controls how risky a proposed delta may be,
- `formalization_pressure` pushes vague language toward typed structures.

As candidate artifact:

- HOTARU may propose a new EPIGENE-16 profile,
- propose a delta to one or more channels,
- split one profile into phase-specific profiles,
- attach a profile to a scenario, controller, experiment, or query plan.

HOTARU still does not commit directly. It proposes. HIVE commits. Monitors
validate. AKOIRE incorporates success/error feedback.

## Relation To AKOIRE

AKOIRE is the cognitive loop: intent, objectives, constraints, ambience,
refinement, parse/evaluation feedback. EPIGENE-16 becomes part of the ambience.

AKOIRE should see:

- the active profile,
- which channels are saturated,
- which monitors fired,
- which channels changed since the last accepted state,
- whether the current objective is failing because of bad structure, bad
  expression settings, or an impossible task.

That gives the LLM-facing loop a fast interface without granting direct authority.
An LLM may say "increase repair priority and formalization pressure"; AKOIRE/HOTARU
lower that into a typed profile delta; HIVE and monitors decide whether it is
admissible.

## Transaction And Distributed Workflow Semantics

EPIGENE-16 must obey the same discipline as other HIVE state:

1. Every profile delta names a `parent_hash`.
2. Every delta has a transaction id.
3. Replaying the same transaction id is idempotent.
4. A stale parent hash is rejected.
5. Channel bounds are checked before commit.
6. Monitor contracts are evaluated after commit in a deterministic order.
7. Failed monitor evaluation produces feedback, not silent mutation.
8. Accepted profiles are content-addressable and diffable.

For distributed workflows, the profile is the small object that lets actors
coordinate without sharing large neural states. One actor may perform a robotics
probe, another may run query planning, another may formalize HyMeKo source. They
can still communicate through profile deltas:

```text
actor A: evidence_requirement ↑ after detecting leakage risk
actor B: mutation_temperature ↓ because HIVE rejects too many deltas
actor C: rollback_readiness ↑ before importing untrusted artifacts
```

The shared state is not "trust me, I changed behavior." It is a typed,
transactional expression profile over a known parent hash.

## Formal Verification Layer

EPIGENE-16 is not itself the formal proof layer, but it gives the proof layer a
small surface to verify.

Useful invariants:

- channel count is exactly 16,
- channel names are unique and canonical,
- values respect declared bounds,
- authority channels cannot exceed policy caps,
- `llm_authority` cannot bypass `transaction_strictness`,
- `expression_gain` cannot override phase adherence when `phase_adherence` is
  locked,
- `baseline_guard` maps to a concrete acceptance criterion,
- high `rollback_readiness` implies provenance retention,
- monitor-sensitive profiles must attach monitor relations,
- profile deltas preserve HIVE transaction semantics.

This is the "fast interface / hard layer" split: LLMs and planners can describe
profile changes quickly; the formal layer verifies the small typed object and its
hypergraph relations.

## Minimal Implementation Plan

The smallest useful implementation is not a neural model. It is a typed profile
and tests.

1. Add a non-core `Epigene16` struct in the layer that owns HIVE/AKOIRE policy,
   not in locked core until the contract stabilizes.
2. Define `ExpressionChannel`, `ChannelKind`, `ChannelBounds`, and
   `ProfileScope`.
3. Implement HIVE lowering: `Epigene16 -> Vec<HiveDelta>`.
4. Implement HIVE loading: query profile/channel relations back into `Epigene16`.
5. Add transaction tests: add profile, update channel, stale parent rejection,
   idempotent replay.
6. Add monitor tests: invalid bounds, over-authority LLM delta, baseline-floor
   violation.
7. Bind Galambos controller gate to a profile read, but keep the existing
   hard-coded safe defaults as the fallback.
8. Only then expose an LLM/HOTARU profile-edit interface.

## Suggested Default Profiles

**Quarantine / untrusted import**

```text
expression_gain        = 0.10
safety_margin          = 0.90
exploration_budget     = 0.05
repair_priority        = 0.80
reuse_bias             = 0.70
mutation_temperature   = 0.10
transaction_strictness = 0.95
monitor_sensitivity    = 0.95
phase_adherence        = 0.90
contact_conservatism   = 0.90
baseline_guard         = 0.95
credit_assignment_scope= 0.20
llm_authority          = 0.05
formalization_pressure = 0.85
evidence_requirement   = 0.95
rollback_readiness     = 1.00
```

**Exploratory design session**

```text
expression_gain        = 0.45
safety_margin          = 0.65
exploration_budget     = 0.70
repair_priority        = 0.45
reuse_bias             = 0.55
mutation_temperature   = 0.65
transaction_strictness = 0.70
monitor_sensitivity    = 0.70
phase_adherence        = 0.65
contact_conservatism   = 0.50
baseline_guard         = 0.70
credit_assignment_scope= 0.55
llm_authority          = 0.25
formalization_pressure = 0.75
evidence_requirement   = 0.65
rollback_readiness     = 0.80
```

**Robotics controller refinement**

```text
expression_gain        = 0.25
safety_margin          = 0.85
exploration_budget     = 0.20
repair_priority        = 0.60
reuse_bias             = 0.80
mutation_temperature   = 0.25
transaction_strictness = 0.90
monitor_sensitivity    = 0.90
phase_adherence        = 0.95
contact_conservatism   = 0.95
baseline_guard         = 0.95
credit_assignment_scope= 0.35
llm_authority          = 0.10
formalization_pressure = 0.70
evidence_requirement   = 0.90
rollback_readiness     = 0.90
```

## Naming Note

The name EPIGENE-16 is strong because it says exactly what the layer should do:
regulate expression without rewriting the underlying organism. In HyMeKo terms,
the "genome" is the typed hypergraph grammar, canonical IR, HIVE transaction
semantics, and controller/monitor contracts. EPIGENE-16 is the context-sensitive
expression mask over that substrate.

The key sentence to preserve:

> EPIGENE-16 is not the controller; it is the typed expression profile that tells
> the controller, planner, learner, and monitors how much adaptive behavior may
> be expressed in the current context.
