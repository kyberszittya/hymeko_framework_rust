# EPIGENE-16 Verification Starter

This folder is the first mechanized theorem-checking spine for EPIGENE-16.

It follows the existing HyMeKo verification pattern:

- use **SymPy** for algebraic/channel-capacity claims,
- use **Z3** for finite bucket invariants and counterexample search,
- keep each theorem executable as a small Python function,
- make positive and negative claims explicit.

## Scripts

| Script | Role |
|---|---|
| `capacity_sympy.py` | Symbolically derives raw capacity and policy-compression equations; proves monotonicity facts. |
| `invariants_z3.py` | Models a 16-channel, 4-bit EPIGENE profile and checks safety/authority invariants with Z3. |
| `tests/test_epigene16_verification.py` | Pytest bridge for both scripts. |

## Run

From the repository root:

```powershell
uv run python verification/epigene16/capacity_sympy.py
uv run python verification/epigene16/invariants_z3.py
uv run pytest verification/epigene16/tests -q
```

## Initial Theorem Split

SymPy starts with algebra:

- `C_raw = n_channels * bits_per_channel`,
- `compression_ratio = source_bits / C_raw`,
- raw capacity increases with channel count and bits per channel,
- compression ratio decreases as the committed profile width increases,
- compression ratio increases as the source context grows.

Z3 starts with finite-governance logic:

- channel values are 4-bit buckets `0..15`,
- `llm_authority <= transaction_strictness`,
- high expression requires high evidence,
- high mutation temperature requires rollback readiness,
- contact-critical profiles require high phase adherence, contact conservatism,
  and baseline guard,
- the contact guard is load-bearing: if it is removed, Z3 finds an unsafe model.

This is intentionally small. The purpose is to establish the contract before
binding it to HIVE transactions or Galambos runtime code.

