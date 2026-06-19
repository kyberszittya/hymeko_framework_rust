# Report — cross-profile instance references in HyMeKo (phase 1, qualified ref)

**Date:** 2026-06-19 · **Plan:** `docs/plans/2026-06-19-xprofile-instance-refs/` (justification + 4 artifacts)
**CORE edit:** **`APPROVED-CORE-EDIT: xprofile-instance-refs`** (2026-06-19 chat) — quoted per CORE.YAML §1.
**Status:** ✅ **Core enabler done + verified.** Downstream consumption (shared reward model + import-aware
profile reader + `AgentSpec.from_hymeko`) is phase 2, not in this change.

## Summary
Enabled one profile to reference another profile's **instance** declarations (qualified ref), so a
shared reward/observation model can be authored once and reused across agent descriptions. Empirical
investigation (three failing `hymeko inspect` experiments) pinned the *minimal* gap: `compile()`
applied `apply_usings` for the **root** AST only, never for imported profiles — so an imported
profile's own `using … as el` never resolved, its arcs/bases failed to lower (`UnresolvedRef target
"el"`), and its decls were unreachable. The grammar requires the `_description` wrapper (imports
cannot live in a content node — a parse error confirms it), so decls sit at `[ns, content, decl]`
and the importer reaches them by full path or a local `using <desc>.<content> as arr` alias. **No
grammar change, no path-doubling surgery, no transitive-import machinery** (the importer already
imports the meta vocab the shared profile uses).

## Change (CORE, `hymeko_core`, `lockdown: full`)
- `hymeko_core/src/module_store/module_store.rs` — after the root `apply_usings`, a loop applies
  **each imported profile's own `using` aliases**, best-effort and per-statement (an alias that does
  not resolve against the available import namespaces is skipped — a genuinely-needed one surfaces
  later as a clear `UnresolvedRef`). Strictly **additive**: meta-only imports have no usings → no-op;
  no existing program's resolution or canonical hash changes. (~9 lines incl. comment.)
- `parser`, `resolve.rs` internals, foundational types: **untouched**.

## Tests
- **New core regression** `hymeko_core/.../test_import_graphs.rs::check_xprofile_instance_ref` +
  fixtures `data/minimal_examples/import_examples/xprofile_{shared,importer}.hymeko`: an importer
  references a shared *profile*'s instance decl (`xs.shared_thing`); asserts it's indexed
  cross-profile and referenced by an arc. **Would fail against the prior code** (the shared profile's
  `el.operand` base raised `UnresolvedRef` at compile).
- **Full regression (the lockdown:full gate):** `cargo test -p hymeko_core` **133 passed**;
  `-p hymeko_query --test integration` **212 passed** (1 ignored); `-p hymeko_formats` clean. The
  canonical-hash suite (`ir_test_hash_pass`) passes byte-identical — **no hash drift**.
- **Manual:** `hymeko inspect` on a cross-profile importer resolves `reward_spec → +…dist` and the
  shared profile's internal aliases.

## CORE.YAML / dependencies
- Touched `hymeko_core` (`lockdown: full`) under the quoted token. No dependency change. No other
  core item touched. The commit footer must carry the token.

## §6.5 anti-patterns
None. Additive loop reusing `apply_usings` (no duplication); no new flags/wrappers; the design was
pinned by experiment (no superstition), and the highest-risk path (hash drift) is gated by the
parity suite with an explicit "stop + revert on drift" rule.

## Performance
Compile-time only: one extra `apply_usings` pass per imported profile (a handful of aliases),
no-op for meta-only programs. No runtime path; benchmark-relevant resolution for existing files is
byte-identical (proven by hash parity).

## Open / phase 2 (downstream consumption)
The core capability exists; consuming it is the next, separate effort:
1. **Import-aware `_profile.read_bundle`** — the Python shim still parses one file; to read a
   cross-profile `reward_spec` it must follow `@"…"` imports and resolve the `using alias.member`
   the way the engine now does. (Now *correct* to do — it would match `hymeko inspect`, not diverge.)
2. **Shared `arm_reach_reward.hymeko`** + rewire `arm_reach_task` / `arm_reach_safe_task` to import
   and reference it (deferred until #1, else the existing Python reward tests break).
3. **`AgentSpec.from_hymeko`** — compose obs + reward + action/vertex into one MDP spec.
4. **Phase-later:** mixin/`include` (bare-name splice) "when performance requires it" (user's words).

## Provenance
- Git SHA `7d16ad0` (working tree dirty; this is an uncommitted increment). MuJoCo/torch per CORE
  pins. Host: Windows 11. Rust debug build. Fixtures + test deterministic (no seed).
