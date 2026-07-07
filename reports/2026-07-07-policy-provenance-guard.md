# Policy-provenance guard — identity companion to the tensor-contract guard

**Date:** 2026-07-07 · Git SHA `4320202` (working tree dirty). Non-core (`hymeko_rl`). RL stays frozen — this is
an identity guard on the RL setup, not an RL run.

## Summary

Built `PolicyProvenanceLedger` (in the `task_monitor` package) as the **identity** sibling of the tensor-contract
guard. The distinction is the whole point, and the user flagged it precisely:

> `PipelineSchemaLedger` catches tensor schema / field-order / dimension drift. It does **not** catch wrong policy
> identity, wrong checkpoint, wrong anchor actor, or scripted-vs-learned provenance if the tensor schema is still
> valid.

The provenance ledger closes that gap. For every policy entering a run it records checkpoint path + md5,
architecture, seed, DAgger stage, role, **post-load parameter hash**, and **action checksum on a fixed canonical
observation batch**, and asserts the identities the run claims. It is wired into `train_offpolicy(provenance=…)`
and aborts at RL init — before a single gradient step — if the actor/anchor/checkpoint is not what the run says.

## Validated on the real frozen artifacts

`scratchpad/provenance_validate.py` against the actual checkpoints:

| checkpoint | file md5 | param hash |
|---|---|---|
| frozen selected DAgger (`mlp_s1_selected_d3.pt`) | `edf4fe81f04bbda26393ca9f230828b9` | `e8ff8f82bcbb5d47233fed55db97a686` |
| BC0 (`mlp_s1.pt`) | `a274888cbceaa1925e0842fcb5903202` | `43dc18aba59d5c8d6e3576f92e1e138a` |
| failed RL smoke actor (`rl_smoke_mlp_s1.pt`) | `a2774902a154c30a4044c4d8e06ae471` | `c563d2fddb84f388a9f5bc5f6fd15d6f` |

The selected md5 **`edf4fe81…`** exactly matches the value the earlier RL-smoke provenance audit recorded → the
ledger reproduces the true checkpoint identity. The four decisive checks:

1. **Genuine RL init from the frozen selected DAgger → PASS.** actor = anchor = selected: md5 matches, param
   hashes equal, actor-vs-anchor **action MSE = 0** on the fixed obs batch.
2. **Drifted failed-RL actor → REJECTED.** `rl_actor param hash c563d2fd… != selected e8ff8f82…`. This is exactly
   the point-5 anchor-bug class the schema guard cannot see: a policy that ran but was **not** the claimed one.
3. **Scripted callable as learned actor → REJECTED.** A scripted controller has no `state_dict`, so it cannot be
   registered as a learned checkpoint (assertion 4, enforced structurally).
4. **BC0 as the "selected DAgger" anchor → REJECTED.** Different param hash — catches a wrong-DAgger-stage anchor.

5. **`train_offpolicy` integration:** given a failing provenance ledger (drifted actor claimed as init-from-
   selected), `train_offpolicy` **aborted at RL init** with `PolicyProvenanceError` before any step.

## The five required assertions (all implemented + tested)

1. checkpoint claimed "= frozen selected DAgger" has the frozen file md5 (`assert_checkpoint_matches`).
2. anchor claimed "= frozen DAgger actor" has the selected checkpoint's parameter hash (`assert_rl_init`).
3. at RL init, actor & anchor param hashes both equal the selected's, and actor-vs-anchor action MSE ~0
   (`assert_rl_init`).
4. a scripted controller / teacher wrapper is never accepted as a learned actor (structural: no `state_dict`).
5. report block: provenance pass/fail + actor/anchor checkpoint hashes + selected DAgger stage + reward/env files
   (`report_fields`).

Expectations queue via `expect_*` and run at RL init through `verify_or_abort` (what `train_offpolicy` calls), or
call the `assert_*` methods directly in a launch harness.

## Files touched

| file | LOC | note |
|---|---:|---|
| `hymeko_rl/eval/task_monitor/provenance.py` | **NEW 226** | `PolicyRole`, `PolicyProvenance`, `PolicyProvenanceLedger`, `PolicyProvenanceError`, `file_md5`, `param_hash`, `action_checksum`, `canonical_obs_batch` |
| `hymeko_rl/eval/task_monitor/__init__.py` | — | export provenance symbols + docstring |
| `hymeko_rl/train/ddpg.py` | ~5 | `provenance` param + `verify_or_abort` at RL init |
| `hymeko_rl/tests/test_policy_provenance.py` | **NEW 134** | 8 unit tests |
| `scratchpad/provenance_validate.py` | — | real-checkpoint validation (not committed) |

**CORE.YAML:** none. No new dependencies (`hashlib` is stdlib).

## Tests

- **Unit:** `pytest hymeko_rl/tests/test_policy_provenance.py` → **8 passed**. Cover param-hash/checksum
  determinism, deterministic `canonical_obs_batch`, RL-init pass (actor=anchor=selected, MSE 0), RL-init fail
  (wrong actor param hash), checkpoint md5 mismatch, scripted-cannot-be-learned (both the role guard and the
  no-`state_dict` guard), and `report_fields`.
- **Full monitor + provenance suite:** `pytest test_task_monitor.py test_policy_provenance.py` → **36 passed**
  (coexists with the linter-added `StagnationMonitor`).
- **Real-artifact validation + live abort:** `provenance_validate.py` → the four identity checks behave correctly
  and `train_offpolicy` aborts at RL init on a failing ledger. `ruff` clean.

## Performance

Negligible: hashing three small state-dicts + one forward on a 32-row obs batch, once at setup. No effect on the
training loop.

## Updated required report fields (binding, every future experiment)

- tensor-contract pass/fail (live schema guard);
- **policy-provenance pass/fail** (new);
- reward, ft_dom, monitor_pass, monitor_score, violation_reason;
- reward-vs-monitor consistency;
- critic-vs-monitor consistency (if a critic exists);
- **actor checkpoint hash, anchor checkpoint hash, selected DAgger stage, reward file, env file** (new,
  produced by `PolicyProvenanceLedger.report_fields`).

## Minimum safety stack before any future RL

1. TensorContractMonitor / **PipelineSchemaLedger: PASS**;
2. **PolicyProvenanceLedger: PASS**;
3. Formal **TaskMonitor: active**;
4. reward-vs-monitor consistency reported;
5. critic-vs-monitor consistency reported if a critic exists.

RL remains frozen until a run satisfies all five. The monitor is still an external verifier — **not in the
reward.**
