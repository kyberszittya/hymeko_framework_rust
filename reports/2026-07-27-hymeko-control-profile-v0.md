# HyMeKo Control Language + CIP-0 Runtime Profile — v0

**Date:** 2026-07-27 (JST)
**Branch:** `integration/hymeko-control-profile-v0`  (worktree `../hymeko_control_profile`)
**Base commit:** `819f35fc` (tag `OPTION_RL_STRUCTURED_TEMPORAL_RUNTIME_V1`)
**Stage:** Campaign Stage 0 + Stage 1 (shared contract). No scenario, no RL.

---

## Summary

Froze the smallest **common contract** for the sequential multi-embodiment
campaign (pick-and-place → humanoid → AIBO): a declarative HyMeKo control
**language** (schema v0 + IR + validator) and the **CIP-0** runtime lifecycle
expressed as typed value types, an eight-method adapter Protocol, and a runtime
driver that enforces the contract at every step.

The package `hymeko_control/` is a **stdlib-only, torch-free, scenario-agnostic**
shared core. Dependency direction is one-way: *scenario adapter → core*, never
*core → adapter*. This is machine-checked (conformance test 10).

The lifecycle:

```
OBSERVE → IDENTIFY MODE → FORM INTENT → MEASURE AUTHORITY
→ DECODE → EXECUTE OPTION → MEASURE RESPONSE → CERTIFY → TRANSITION
```

## Stage 0 — base selection (recorded)

- **Selected base:** `OPTION_RL_STRUCTURED_TEMPORAL_RUNTIME_V1` (`819f35fc`).
- **Why:** it is an ancestor of the current coin HEAD (a genuine common base);
  it carries the frozen task-independent `hymeko_rl/option_rl` runtime
  (StructuredState, option transitions, replay/provenance, multimodal
  proposal/search) and the torch-free `hymeko_rl/control` vocabulary
  (`ControllerSpec`, `HybridAutomaton`); and it includes a **non-coin** ToyReach
  end-to-end integration proving the runtime is scenario-independent.
- **Included generic infrastructure:** `option_rl` (semi-MDP option runtime +
  provenance), `control.controller_spec`/`control.hybrid_automaton` (modes,
  guards, transitions), structured state + history contracts.
- **Excluded unfinished task branches:** all coin decision-representation
  experiments R1–R6 (`recovery/coin-r{1..6}-*`), which post-date the tag.
- Created `integration/hymeko-control-profile-v0` as a new worktree from the
  tag; no existing branch pointer moved. `git config rerere.enabled true`.

## Design — reuse vs new (discovery pass done first)

A discovery pass (`find`/`grep`/`ls` + a read-only vocabulary map of
`hymeko_rl/control` and `hymeko_rl/option_rl`) established what already exists so
the profile **extends** rather than duplicates:

| Concept | Existing (reused by contract / duck-type) | Net-new in `hymeko_control` |
|---|---|---|
| structured state | `option_rl.state.StructuredState` (numpy) | `StructuredStateLike` Protocol + torch-free `ControlState` |
| modes / guards / transitions | `control.controller_spec` / `hybrid_automaton` | declarative `Mode`/`Transition` IR + legality/terminal checks |
| option semantics | `option_rl.core.OptionEnd`/`OptionTransition` | first-class `ExecutableOption` (initiation+policy+termination) |
| authority handoff | `option_rl.hierarchy.SkillRoute` | `AuthorityMap` + `AuthorityChannel` (typed provenance) |
| **physical intent** | — (gap) | `PhysicalIntent` (bounded) |
| **deterministic decode** | — (only stochastic search existed) | `Decoder` Protocol + `AffineAuthorityDecoder` |
| **certificate value type** | only `HandoffCertificate` Protocol | composable `Certificate`/`CertificateSuite`/`CertificateResult` |

The shared core is stdlib-only because importing `option_rl.state` transitively
pulls `torch` via that package's `__init__`; re-declaring the *contract* torch-free
(and staying compatible by duck-type) is the correct boundary, not a fork.

## Files touched (all NEW, non-core)

```
hymeko_control/__init__.py                         36
hymeko_control/_frozen.py                          28
hymeko_control/language/{__init__,schema_v0,ir,validator}.py   48/105/193/223
hymeko_control/cip/{__init__,protocol,structured_state}.py     67/56/78
hymeko_control/cip/{physical_intent,authority,option}.py       85/80/146
hymeko_control/cip/{certificate,runtime}.py                    120/192
hymeko_control/conformance/{battery,toy}.py                    141/189
hymeko_control/conformance/tests/test_cip0_conformance.py      259
reports/2026-07-27-hymeko-control-profile-v0.md  + schema_contract.json
                                                  + cip_contract.json
                                                  + conformance.json
docs/plans/2026-07-27-hymeko-control-profile-v0/  (plan.tex/pdf/tikz/mmd; gitignored)
```
Total package: **2047 LOC** (incl. tests + toy reference).

## CORE.YAML items touched

**None.** `hymeko_control/` is a new top-level path (`on_unknown_path:
treat_as_non_core`). No pinned dependency added — stdlib only; `numpy`/`torch`
are deliberately **not** imported on the core path (`yaml` is a lazy, optional
import inside `validator.load_yaml`).

## Test results

`pytest -p no:randomly hymeko_control/conformance/tests`

| # | Conformance requirement | Test | Result |
|---|---|---|---|
| 0 | full lifecycle → certified HOLD (integration) | `test_00_full_lifecycle_reaches_certified_hold` | ✅ |
| 1 | schema validation | `test_01_schema_validation` | ✅ |
| 2 | causal observation | `test_02_causal_observation` | ✅ |
| 3 | legal mode transitions | `test_03_legal_mode_transitions` | ✅ |
| 4 | bounded physical intent | `test_04_bounded_physical_intent` | ✅ |
| 5 | authority provenance | `test_05_authority_provenance` | ✅ |
| 6 | deterministic decoding | `test_06_deterministic_decoding` | ✅ |
| 7 | option execution provenance | `test_07_option_execution_provenance` | ✅ |
| 8 | certificate independence from reward | `test_08_certificate_independent_of_reward` | ✅ |
| 9 | no hidden state modification | `test_09_no_hidden_state_modification` | ✅ |
| 10 | shared-core import isolation | `test_10_shared_core_import_isolation` | ✅ |

**11 passed in ~0.06 s.** Each negative path (bad spec, illegal transition,
unbounded intent, empty provenance, non-deterministic decode, broken option
provenance) is exercised, not just the happy path.

- `ruff check hymeko_control` — **all checks pass.**
- `radon cc -a -nc` — no function rated C or worse (all A/B; under the §6.2 gate).

## Performance

Contract logic on toy fixtures: full suite **0.023 s** (junit-measured), peak RSS
negligible (< 100 MB, pure Python). No GPU. Well under all §4 budgets. This is a
correctness contract, not a throughput target; no benchmark discipline applies.

## Two real bugs found and fixed during test bring-up

1. **Early termination at TOUCH.** `run()` stopped when a self-looping mode's
   success certificate passed — but the `reached` predicate also holds in the
   TOUCH band. Fix: added `ControlModel.is_terminal_mode` and gated completion on
   reaching a *terminal* mode (or an `OptionEnd.COMPLETED`), so an episode
   completes only in HOLD. This is a genuine contract sharpening, not a test hack.
2. **Over-strict reward check.** test_08 first grepped the source string
   `reward`, which the module legitimately *documents*. Fix: AST check that no
   function parameter or `Name` node is `reward` — docstrings may discuss it.

## New / removed dependencies

None.

## §6.5 anti-patterns

None introduced. No Cartesian API surface, no string-typed config crossing into
internals (enums used), no globals, no `_v2` files, no algorithm behind a
binding boundary. Discovery pass performed before creating the package.

## Open issues / follow-ups

- `validator.load_yaml` needs PyYAML at call time; scenarios lacking it pass a
  dict to `validate` directly. Not a core dependency.
- Scenario adapters reuse `conformance.battery.run_positive_lifecycle` +
  `import_isolation_violations` + the schema asserts; the negative-path core
  enforcement tests stay here (they test the core, not the scenario).

## Provenance

- Git base SHA: `819f35fcee6d643e6346c7728776ab6277531d2f` (clean worktree from tag).
- Python 3.11.15, pytest 8.4.2, numpy 2.4.6 (unused on core path), ruff 0.15.20.
- Host: darwin 25.5.0 (arm64), `.venv` at the main repo root.
- No randomness in the core or tests; no seed needed (deterministic contract).

**Verdict:** CIP-0 profile v0 **PASSES** its 10 conformance guarantees. Ready to
tag `hymeko-control-profile-v0` and branch the three scenario worktrees from it.
