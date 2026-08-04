# Rust HSTL monitor backend (parity + benchmark) & exact formal verification

**Date:** 2026-08-04
**Branch/worktree:** `research/humanoid-com-lyapunov` @ `hymeko_humanoid` (head at start `58643959`)
**Follow-ups (2) and (3)** of the viability arc.

---

## Correction (important)

The prior M2⁺ report claimed "the Rust binding is not built in this venv." **That was wrong** — I searched for a
Python module named `hymeko_monitor` (which is only the Rust *crate* directory, imported as an empty namespace
package under `PYTHONPATH=.`). The **actual built binding is `hymeko.HtlMonitor`** — a compiled pyo3 extension
`hymeko/hymeko.cpython-311-darwin.so` (from the `hymeko_py` crate, pyo3 0.28.2), importable and working. The user
flagged this; verified and corrected. **No §1 escalation is needed** (nothing to build, no dependency added).

## (2) Rust monitor backend — wired, parity-tested, benchmarked

- **`RustHtlBackend`** (in `hstl_monitor.py`) wraps `hymeko.HtlMonitor` behind the same `MonitorBackend` Protocol
  as `PythonHtlBackend`; `make_monitor(spec, "rust")` returns it (falling back with a clear `RuntimeError` only if
  the `hymeko` extension is absent, e.g. a headless build).
- **Parity: 0 mismatches** — Rust and the Python `hymeko_neuro.eval.htl` engine return **bit-identical** robustness
  and satisfaction on the same event stream (`G(m ≥ 0)`, mixed-sign margins).
- **Benchmark: ~93× faster** — 130 µs/obs (Rust) vs 12 140 µs/obs (Python) at 2000 observations. (The Python
  engine recomputes `G` over its history each call — O(n) per observe; the Rust monitor is the one to use for long
  runs / real-time.)

## (3) Formal verification — exact for M1, no new dependency

Replaced sampling with an **exact algebraic guarantee** for the M1 quadratic certificate. Inside the well the
pendulum closed loop is linear, `z⁺ = M z` (M the H-step semi-implicit flow on `z=(u,θ̇)`), so a quadratic
`V=zᵀPz` decreases **everywhere** iff the discrete-Lyapunov inequality `Q = MᵀPM − P ⪯ 0` holds — an eigenvalue
check, not a sample. `LyapunovCertificate.formal_verify` returns that verdict plus the largest sublevel provably
inside the well, `c = π²/(P⁻¹)₀₀`.

- **Seeded (H_d) certificate:** `Q ⪯ 0` (max eig −0.14), **`formal_level = 118.45 = c*`** (the exact barrier),
  vs the conservative **sampling level 110.2** — the formal guarantee is both *sound* and *tighter*.
- **Non-Lyapunov detection:** an unfit identity metric (`P ≈ I`) is *not* a Lyapunov function for the semi-implicit
  flow (it transiently expands) — `formal_verify` catches it (`decreasing = False`, max eig +2.98).
- **SMT would be a §1 dependency** (dReal/Marabou are new packages); the LMI needs none and is *exact* for the
  linear case. The **M2 neural** certificate is nonlinear, so its formal guarantee needs a **Lipschitz-bounded
  sampling** (weight-norm bound on `V` × flow Lipschitz) or SMT — flagged as the next step, not claimed here.

## Files touched

| File | Δ | notes |
|---|---|---|
| `scenarios/humanoid/hstl_monitor.py` | +25 −8 | `RustHtlBackend`; `make_monitor("rust")` wired; corrected docstring |
| `scenarios/humanoid/viability.py` | +22 | `LyapunovCertificate.formal_verify` (exact LMI) |
| `tests/test_hstl_monitor.py` | ±8 | rust-vs-python parity (replaces the "documented slot" test; `importorskip` for headless) |
| `tests/test_viability_certificate.py` | +16 | 2 formal-verify tests (exact + non-Lyapunov rejection) |
| `reports/2026-08-04-hstl-runtime-monitor.md` | +1 note | correction pointer |

## CORE.YAML items touched
None. `hymeko` (pyo3) is already built and installed; using it adds no dependency. `hymeko_monitor` (the Rust
crate) is not modified. No new Python package. §1 not triggered — the earlier escalation was based on the wrong
module name and is withdrawn.

## Test results
- `pytest tests/test_hstl_monitor.py` → **9 passed** (incl. Rust parity, skipped only if `hymeko` absent).
- `pytest tests/test_viability_certificate.py` → **8 passed** (incl. 2 formal-verify).
- `ruff check` → clean. (M0/M2 suites unaffected — `formal_verify` is an additive method; the monitor change is
  behind the same interface.)

## §6.5 anti-patterns
None. The Rust backend is a second `MonitorBackend` impl behind the existing Protocol (Strategy), not a fork of
the logic; `formal_verify` is an additive method; no globals; explicit error on a missing extension.

## Open issues / follow-up
- **Lipschitz/SMT verify for the M2 neural certificate** (the nonlinear case the LMI does not cover).
- **Limit-cycle-aware certificate** so `L`-coupled dynamics can be certified (from the M2⁺ report).

## Provenance
Git SHA at start `58643959`. Env: HyMeKo `.venv` (Python 3.11; `hymeko` pyo3 ext built, pyo3 0.28.2;
`hymeko_neuro.eval.htl` importable; NumPy 2, torch 2.12.0 CPU), macOS (darwin 25.5). Deterministic; no GPU.
Benchmark on this quiet host (single process); latency numbers are median-of-run indicative, not a criterion
benchmark.
