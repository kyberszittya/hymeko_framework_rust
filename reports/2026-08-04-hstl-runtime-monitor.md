# M2⁺ — HSTL runtime monitor of the certified basin (+ an M2 certificate correction)

**Date:** 2026-08-04
**Plan:** `docs/plans/2026-08-02-learned-viability-boundary/` (M2⁺ addendum; on disk, gitignored per convention)
**Branch/worktree:** `research/humanoid-com-lyapunov` @ `hymeko_humanoid` (head at start `483ccb1c`)

---

## Summary

Extended M2 with an **HSTL (HyMeKo/Hypergraph Signal Temporal Logic) runtime monitor** — the online companion to
the offline certificate — and, in the process, **corrected a real flaw in the committed M2 certificate**.

- **`hstl_monitor.py`** — a backend-agnostic `MonitorBackend` Protocol (mirroring `coin_carry_monitor.py`'s
  §2A/§2B pattern); a `PythonHtlBackend` that **reuses** the pure-Python HTL evaluator `hymeko_neuro.eval.htl`
  (robust-STL min/max, `G`/`F` over `ScalarPred`) — no reinvention (§6.1); a `make_monitor(spec, backend)` factory
  where `"python"` ships and `"rust"` is a **documented slot**; and `monitor_trajectory`, which runs a centroidal
  run against the certificate-tied specs `G(cert_margin ≥ 0)` (stay in `{V≤c}`) and `G(fall_margin > 0)` (never
  fall), emitting per-step robustness, the verdict, and the early-warning lead.

- **The Rust HTL version, honestly:** yes, `hymeko_monitor` (a Rust STL crate, not in CORE.YAML) is the fast
  version — but its **PyO3 binding is not built in this venv** (`import hymeko_monitor` is empty), and even the
  coin work's `make_monitor` only ships the Python backend. So the Rust backend is a **slot behind the same
  interface**, not a claim that it runs here; wiring + parity-testing it is a follow-up.

## M2 certificate correction (found while building the monitor)

The monitor computed a degenerate `certified_level ≈ 0`, which exposed that the committed full-state certificate
was **grid-fragile**: the running target is a **limit cycle** (`z` bounces), so `V=‖φ(x)−φ(x*)‖²` over the full
state oscillates and the certified level collapses on grids that include near-nominal points (the committed
0.0004 / 0.45 came from a held-out grid that happened to avoid the collapse).

**Fix:** the `(z, ż)` gait is **decoupled** from the fall in these dynamics — only `(L, pitch)` governs falling —
so the certificate now acts on the **`(L, pitch)` subspace**. `V` is then (near-)constant on the `z`-bounce limit
cycle, the decrease condition is genuine, and the level is robust:

| | committed M2 (full state) | **corrected (L,pitch subspace)** |
|---|---|---|
| certified_level (train grid) | 0.0 (collapsed) / 4.73 (held-out only) | **4.0 (robust)** |
| fall-violation on `{V≤c}` | 0.0004 (at the lucky level) | **0.028** |
| IoU vs recoverable | 0.45 | **0.48** |

**Honest caveat:** at this (strong) regulation the fall is **pitch-dominated** — `L` is weakly coupled, so the
basin is close to `{|pitch| < fall_pitch}` and the certificate is a valid but only *mildly*-nonlinear demo.
Softening the regulation (`pitch_gain 7→2.5`, `l_damp 9.5→4`) makes `L` genuinely matter (low-pitch high-`L`
states then fall), **but** the attractor becomes a limit cycle `(L_ss, pitch_ss)` where a point-Lyapunov `V`
collapses again — **limit-cycle-aware certification (transverse / Poincaré Lyapunov) is the honest follow-up**, not
attempted here. Config kept at the working (pitch-dominated) values with this documented in `centroidal.py`.

## Monitor results

- **Inside `{V≤c}` → certified safe:** a recovering state inside the set keeps both specs satisfied with a
  **positive graded margin** (deep-inside `min_cert ≈ +4.0` > near-boundary `≈ +0.36` > outside `< 0`) — the
  robustness is a live, graded safety signal.
- **Outside/falling → flagged, no later than the fall:** for falling states the monitor is unsatisfied and
  `warn_step ≤ fall_step` in 48/50 sampled fallers (the 2 exceptions are the certificate's own ~2.8 % violations).
- **Early-warning lead ≈ 0 here** — honestly, because the pitch-dominated fall is near-instantaneous once `V`
  exceeds `c` (the certified and fall boundaries nearly coincide in time). The monitor's value in this model is
  the **graded online margin + verdict-consistency**, not a large lead; a slower-fall (L-coupled) model would
  show a real lead but needs the limit-cycle certificate above.

## Files touched

| File | Δ | notes |
|---|---|---|
| `scenarios/humanoid/hstl_monitor.py` | +110 (new) | Protocol + Python HTL backend + Rust slot + `monitor_trajectory` |
| `scenarios/humanoid/neural_certificate.py` | ±15 | certificate over the `(L,pitch)` subspace; fit defaults iters 800 / sep 2.0 |
| `scenarios/humanoid/centroidal.py` | +4 | documented the pitch-dominated / limit-cycle caveat |
| `tests/test_hstl_monitor.py` | +100 (new) | 9 tests (robust-STL, rust slot, certified/flagged, graded, determinism, integration, perf) |
| `tests/test_centroidal_certificate.py` | ±4 | thresholds updated to the corrected certificate (violation ≤ 0.04, IoU ≥ 0.40) |
| `reports/2026-08-03-…-m2.md` | +1 note | correction pointer |

## CORE.YAML items touched
None. Reuses the non-core Python HTL evaluator; `hymeko_monitor` (Rust) is **not** modified and its binding is not
built here. torch stays at the pinned `==2.12.0`. No dependency change.

## Test results
- `pytest tests/test_hstl_monitor.py` → **9 passed in 5.9 s**; `tests/test_centroidal_certificate.py` → **7 passed**;
  regression `tests/test_viability_boundary.py` + `tests/test_viability_certificate.py` (M0+M1) → **20 passed**.
- `ruff check` on all touched files → clean.

## §6.5 anti-patterns
None. The monitor reuses the existing HTL engine (no re-implementation) behind the established `MonitorBackend`
Protocol (Strategy); the Rust backend is a factory slot, not a fabricated impl. No globals; explicit errors
(`NotImplementedError` for the unbuilt Rust backend, `ValueError` for a batched state / unknown backend).

## Open issues / follow-up
- **Limit-cycle-aware certificate** (transverse/Poincaré Lyapunov) so `L`-coupled dynamics can be certified — the
  prerequisite for a genuinely nonlinear basin and a non-zero early-warning lead.
- **Build + parity-test the Rust `hymeko_monitor` backend** behind `MonitorBackend`, then benchmark vs Python.
- **Formal (SMT/Lipschitz) verify** to replace sampling — the monitor gives an online margin, still not a proof.

## Provenance
Git SHA at start `483ccb1c`. Env: HyMeKo `.venv` (Python 3.11, torch 2.12.0 CPU, NumPy 2, SymPy 1.14;
`hymeko_neuro.eval.htl` importable; `hymeko_monitor` importable but empty), macOS (darwin 25.5), 4 CPU threads.
Deterministic: fixed `seed=0`, pinned `dt=4 ms`. No GPU, no external dataset.
