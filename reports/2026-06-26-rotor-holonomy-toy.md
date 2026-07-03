# Rotor / holonomy toy — the rotor connection sees what the signed Z₂ link cannot

**Date:** 2026-06-26 · **Author:** Aiko (Claude Code) for Dr. Csaba Hajdu
**Plan/design:** `docs/plans/2026-06-26-rotor-spikes-ablation/` (tex/pdf/tikz/mmd) · **Status:** built, tested, run.

## The claim (gauge picture)

A signed edge `s_ij ∈ {±1}` is a **Z₂ connection**; structural balance is its trivial **holonomy** (sign-product
around a cycle = +1). A **rotor** replaces the sign with a continuous rotation `R_ij ∈ SO(2)` — a genuine
connection that parallel-transports a node's feature vector; its holonomy `∏_cycle R_ij ∈ SO(2)` is *continuous*,
and balance is only its Z₂ quotient. So the signed model sees a **shadow** of what the rotor sees.

## The toy (the small example)

A cycle with hidden per-edge rotations summing to a holonomy `Φ`. A source vector `x ∈ R² ~ N(0,I)` is
transported around the loop: target `y = R(Φ) x`. Two models learn their (single, effective) holonomy:
- **rotor** — a learnable angle `α`; predicts `R(α) x` (can match any Φ);
- **signed** — a learnable real **scalar** `c` (Z₂ = `c=±1`); predicts `c·x` (can *scale*, never *rotate*).

## Result (3 seeds, 400 epochs, CPU)

| Φ | rotor MSE | signed MSE | analytic sin²Φ |
|---|---|---|---|
| 0 | 0.0000 | 0.000 | 0.000 |
| π/4 | 0.0000 | 0.511 | 0.500 |
| **π/2** | **0.0000** | **1.022** | **1.000** |
| 3π/4 | 0.0000 | 0.872 | 0.854 |
| π | 0.0000 | 0.000 | 0.000 |
| 3π/2 | 0.0000 | 1.022 | 1.000 |

Signed/rotor ratio at Φ=π/2 ≈ **10⁹**. Figure: `reports/rotor_probe/rotor_probe.png` (rotor flat ~0; signed =
sin²Φ; analytic overlay matching). JSON: `reports/rotor_probe/rotor_probe.json`.

### Reading

- **Measured = predicted.** The signed model's optimal scalar is `c* = cos Φ` (the projection of `R(Φ)x` onto
  `x`), giving residual `sin Φ · x⊥` and **`MSE_signed = sin²Φ`** — the measured curve matches this closed form to
  ~1% (finite-sample). The toy is *verified against theory*, not merely suggestive.
- **The signed link captures exactly the Z₂ quotient.** It hits zero error only at Φ ∈ {0, π} — the holonomy
  values that *are* a sign. Everywhere else it is blind to the rotation; the rotor is not.
- This is the cleanest possible statement of "balance = Z₂ holonomy; rotor = continuous holonomy": the signed
  model lives on the cosine shadow, the rotor on the full circle.

## Implication & next step

Motivates **rotor-as-connection** in HSiKAN: where a task's signal lives in the continuous holonomy of the
signed graph (not just its balance), a rotor message-passing layer can represent it and the signed conv cannot.
The production realization already exists (quaternion/SO(3) `RotorInjector` / `CayleyRotorEmbedding` /
`SignedRotorPropagation` in `hymeko_neuro/.../run_hsikan_rotor.py`) — this toy isolates *why* it should help.

**Spikes (next toy, designed not built):** SO(2) is abelian, so holonomy is order-independent — spikes do
nothing here. They earn their keep with a **non-abelian** (SO(3)/quaternion) connection, where a walk's holonomy
depends on traversal **order**; **spike timing** selects the time-ordered walk. Toy sketch in the plan: a diamond
graph (two paths, `R_a R_b ≠ R_b R_a`), target = the spike-ordered path's transport; a static aggregator averages
the paths and loses it, a spike-gated reader selects the timed one.

## Files (CORE.YAML: none)
- **New** `hymeko_rl/rotor_probe.py` — `rot_matrix`, `make_holonomy_data`, `RotorTransport`/`SignedTransport`,
  `run_rotor_probe`, `plot_rotor_probe`, CLI. **New** `hymeko_rl/tests/test_rotor_probe.py` — 5 tests pass.
- Reuse/cite (no edit): the production quaternion rotor in `hymeko_neuro`. ruff clean. Supervised A/B on fixed
  data → strict-deterministic (CLAUDE.md §3 carve-out).

## Provenance
Reproduce: `python -m hymeko_rl.rotor_probe --seeds 3 --epochs 400`. Git `fix-hsikan`, tree dirty. Windows 11,
Python 3.12, torch CPU.
