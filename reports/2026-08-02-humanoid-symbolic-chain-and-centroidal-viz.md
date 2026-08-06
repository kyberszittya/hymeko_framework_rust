# Humanoid port-Hamiltonian — symbolic N-link chain M(q) + centroidal-running visualization

**Date:** 2026-08-02
**Branch/worktree:** `research/humanoid-com-lyapunov` @ `hymeko_humanoid` (head `38016af1`, working tree dirty — see below)
**Scope:** the (b)+(c) increment of the user's `a → b → c` ordering. (a) [source-level pH ROLE vocabulary] shipped in `38016af1`.

---

## Summary

Two deliverables, both in service of the HyMeKo goal *"generate + control a complete dynamical-system description"*:

- **(b) Centroidal-running pH panel** added to the IDA-PBC visualization artifact. It renders the momentum-level
  Hamiltonian `H_c = ½|p|²/m + ½L²/I + m g z` and simulates the **angular-momentum channel L** live: with
  *regulate L* ON the foot-placement / arm-swing ports drive `L → 0` and the torso stays upright; with it OFF the
  stance torque bias integrates, `L` grows, and the torso pitches over — the exact failure the linear-only run
  planner produced (~50° pitch). This makes the diagnosed under-actuation legible as an energy-port channel.
- **(c) Symbolic humanoid M(q) — the tractable core.** The full 22-DOF `M(q)` is intractable in closed form, but
  the humanoid's **sagittal building block — a planar N-link chain (hip–knee–ankle)** — is not. Added
  `planar_chain_ph(masses, lengths)` which derives `M(q) = Σ mᵢ JᵢᵀJᵢ + Σ Iᵢ (∂φᵢ/∂q)ᵀ(∂φᵢ/∂q)` and the gravity
  generalised force `G = ∂V/∂q` by CoM-Jacobian composition, for any n. The 3-link leg builds in ~2 s; its M is
  symmetric, SPD, and configuration-dependent (hip–knee–ankle coupling via `cos q₂`, `cos q₃`, `cos(q₂+q₃)`).

**Honesty note (why not the *full* M):** the closed-form momentum Hamiltonian needs `M⁻¹`, whose symbolic form for
n ≥ 3 explodes (the earlier attempt via `mechanical_ph`, which inverts + `simplify`s, timed out at 2 min). The
symbolic deliverable is therefore `M(q)`, `V(q)`, `G(q)` and `H = ½q̇ᵀMq̇ + V` (velocity form, no inverse); the
momentum-form `M⁻¹` is evaluated numerically (MuJoCo `mj_fullM`) — stated in the function's returned `note`. The
full humanoid M is the **block composition** of two such legs + two arms + torso-about-base; that structure is
documented, not expanded.

---

## Files touched

| File | Δ | notes |
|---|---|---|
| `scenarios/humanoid/symbolic_ph.py` | +36 | new `planar_chain_ph()` (CoM-Jacobian M(q), V, G; trig-simplified entries, no inverse) |
| `tests/test_humanoid_port_hamiltonian.py` | +24 | 2 new regression tests (SPD+config-dependent; n=2 cross-validation vs hand-derived leg) |
| `scratchpad/idapbc_viz.html` (Artifact) | +~85 | centroidal-running pH card: `H_c` equations, regulate-L toggle, live L/z/pitch canvas |

Artifact republished in place — URL unchanged: <https://claude.ai/code/artifact/bfeedd7c-7093-471c-a276-8b96ea95cf7c>.

**Working tree also shows** unrelated untracked files (seminar decks under `docs/seminar/`, `experiments/*`,
regenerated report GIFs/MP4s) — **not part of this change**; excluded from any commit.

## CORE.YAML items touched

None. `symbolic_ph.py` and the test are non-core Python; the `.hymeko` vocabulary (a) is a non-core data module.
No dependency added (SymPy already present).

---

## Test results

`pytest tests/test_humanoid_port_hamiltonian.py -p no:randomly -q` → **12 passed in 14.6 s** (10 prior + 2 new).
`ruff check` on both touched files → clean.

New regression tests (each would have failed before this change — the function did not exist):
- `test_planar_chain_leg_mass_matrix_is_spd_and_configuration_dependent` — 3-link leg M is 3×3, symmetric,
  `cos`-coupled off-diagonal, numerically SPD at a representative config (eigs `[1.1e-3, 4.0e-2, 1.36]`).
- `test_planar_chain_reduces_to_the_hand_derived_two_link_leg` — **cross-validation:** the general N-link M(q) at
  n=2 equals the independently hand-derived `two_link_leg_ph` M(q) to `1e-9` at three configurations. Two
  derivations from different code paths agreeing is the strongest available correctness signal here.

## Performance

Symbolic build (not a hot path): 3-link leg M(q) in **~2.0 s** (per-entry `trigsimp`, no matrix inverse). The
prior `mechanical_ph`-based route (symbolic `M⁻¹` + `simplify(H)`) **timed out > 120 s** for n=3 — the redesign is
the fix, not a regression. No numeric/RSS budget applies (pure-SymPy, single expression, < 100 MB).

## §6.5 anti-patterns

None introduced. `planar_chain_ph(masses, lengths, …)` is the *general* entry that **subsumes** the special-case
`pendulum_ph` (n=1) and `two_link_leg_ph` (n=2) rather than adding another `_3link_` / `_4link_` wrapper — the
opposite of a Cartesian-product surface. Error path: relies on SymPy raising on shape/derivative errors; no new
`unwrap`/broad-except equivalent.

## Open issues / follow-up

- The visualization's centroidal sim is **schematic** (a principled SLIP-bounce + L-integration illustration), not
  the collocation solver's actual trajectory — it demonstrates the L-channel qualitatively. Wiring the real
  `centroidal_run` plan trace into the panel is a possible follow-up.
- (c) delivers the *chain* M(q). Assembling the **floating-base block M** symbolically (6-DOF base ⊕ limb blocks
  via the composite-rigid-body recursion) is the natural next symbolic step if the full closed form is wanted;
  flagged, not attempted.
- The pitch-over failure now has a control target (regulate L via foot placement / arm swing) — the WBC plan
  should carry an angular-momentum reference, which the current linear-only planner omits.

## Provenance

Git SHA at start: `38016af1` (working tree dirty; dirty files listed above, only the two source files are part of
this change). Env: HyMeKo `.venv` (Python 3.11, SymPy 1.14, NumPy, MuJoCo 3.10), macOS (darwin 25.5). Deterministic
(symbolic; no seed needed). No experiment data produced.
