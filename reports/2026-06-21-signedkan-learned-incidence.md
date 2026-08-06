# A learned-incidence (`signedkan`) policy — the trained weights ARE the star edges

**Date:** 2026-06-21 · **Author:** Aiko (Claude Code) for Dr. Csaba Hajdu
**Plans:** [signedkan](../docs/plans/2026-06-21-signedkan-learned-incidence/) ·
[binary tensor storage (plan-only)](../docs/plans/2026-06-21-binary-tensor-storage/)

## Summary
The P4 storage prototype stored *dense* learned weights over a *fixed* kinematic incidence. The deepest form
of "weights as the star expansion of a HSiKAN description" is when the **incidence itself is learned** — then
the trained parameters *are* the weighted star edges. This adds a `signedkan` policy variant and demonstrates
it end-to-end on the cart-pole.

`HSiKANBackbone` gained `learn_incidence`: when set, `a_pos`/`a_neg` become `nn.Parameter`s initialised from
the kinematic signed adjacency (a parametric flag, not a structural fork — the `einsum` forward is identical).
A new `"signedkan"` policy kind maps to it through the existing backbone registry.

**Verified:**
- `signedkan` `a_pos`/`a_neg` are trainable parameters, init = kinematic adjacency; `hsikan` stays a **buffer**
  (regression guard). `+16` params total (the 2×2 incidence × 2 signs × 2 backbones).
- After 80 vec-iters the incidence **drifted 0.026** from the kinematic init (init `a_pos[pole,cart]=1.00` →
  learned `1.01`, with small learned off-diagonals) — the star edges were actually learned, not frozen.
- The learned-incidence policy is stored as a **valid HyMeKo** file
  ([data/nn/cartpole_signedkan_policy.hymeko](../data/nn/cartpole_signedkan_policy.hymeko)) and round-trips
  **bit-exact**, the learned incidence included.

**Honest result:** `signedkan` 39.9 vs `hsikan` 56.8 upright-steps (single seed, 80 iters) — **parity-ish, no
structural gain**, exactly as expected on a 2-vertex graph where there is almost no incidence to learn. This is
a *mechanism* demonstration (learned star edges, stored as such), not a performance claim. The params-matched-
control rule and "judge structure only on a real-topology task" still stand.

## Figure
`docs/figures/2026-06-21-policy-storage/07-learned-incidence.{png,svg}` — the kinematic-init incidence vs the
learned incidence (per-sign heatmaps) and the learned incidence drawn as signed star edges (blue +, pink −).

## Files touched
| File | Δ |
|---|---|
| `hymeko_rl/policy.py` | +18 (`learn_incidence` flag, `signedkan_backbone`, `"signedkan"` kind) |
| `hymeko_rl/train_inverted_pendulum.py` | `_make_balance_policy` dispatches `hsikan`/`signedkan` via the hg path |
| `hymeko_rl/tests/test_policy.py` | +2 tests (learned-vs-buffer incidence; incidence receives gradient) |
| `scripts/render_policy_storage_figures.py` | +fig7 (learned incidence) |
| `data/nn/cartpole_signedkan_policy.hymeko`, `reports/cartpole_signedkan_policy.pt` | new artifacts |

**CORE.YAML / deps:** none.

## Test results
- `test_policy.py` — **13 passed** (2 new: `signedkan` incidence is a `requires_grad` parameter initialised to
  the kinematic adjacency while `hsikan` stays a buffer; a loss yields a finite non-zero gradient on the
  incidence). `ruff` clean; `mypy --strict` clean on `policy.py`.
- `hymeko validate` ✅ on the stored signedkan policy.

## Binary tensor storage — plan + **T1 implemented**
[docs/plans/2026-06-21-binary-tensor-storage/](../docs/plans/2026-06-21-binary-tensor-storage/) (4 artifacts).
Three tiers behind one chooser: **T0** inline-decimal (auditable); **T1** inline-binary `data_b64` (base64 of
float32 bytes — a *string*, so no `e`-lexer issue); **T2** content-addressed blob (planned). `policy_to_hymeko`
gained `tier="t0"|"t1"|"auto"`; `hymeko_to_policy` dispatches on the present field. **T1 built + tested**:
cart-pole policy **332 KB → 142 KB** (T1/T0 = 0.43), `auto` 148 KB (small/structural tensors stay readable
decimals, bulk binary), both valid HyMeKo and bit-exact. The two stored policies are now written with `auto`.
T2 (external blob + `sha256`) remains plan-only. **18 `test_policy_store` tests** (T0/T1/auto round-trip,
codec, size, validity).

## §6.5 anti-patterns
None. `learn_incidence` is a parametric flag (identical forward), not a forked class; `signedkan` rides the
existing backbone registry (one dispatch). The binary tiers are one Strategy chooser, not per-format wrappers.

## Open issues / follow-ups
1. Implement the binary storage plan (T1/T2) — drops the 332 KB inline policy to ~110 KB.
2. Multi-seed `signedkan` vs `hsikan` vs MLP on a **real-topology** task (6-DOF arm) — the only setting that
   can show whether learning the incidence helps.
3. Load a stored `.hymeko` policy back **in the editor** and see the learned star edges on the attribute-HUD.
