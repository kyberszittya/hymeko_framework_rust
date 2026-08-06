"""SymPy starter proofs for EPIGENE-16 channel capacity and compression.

This is algebraic verification, not empirical measurement. It proves the simple
closed-form facts that should stay true regardless of the later HIVE encoding:

* raw capacity is channels * bits_per_channel for uniform quantization;
* compression ratio is source_bits / raw_capacity;
* capacity increases with channel count and quantization width;
* compression ratio decreases as the committed profile gets wider;
* compression ratio increases as the source context gets larger.
"""
from __future__ import annotations

import sympy as sp


def derive_formulas() -> dict[str, sp.Expr]:
    n, b, source_bits = sp.symbols("n b source_bits", positive=True)
    raw_bits = sp.simplify(n * b)
    compression_ratio = sp.simplify(source_bits / raw_bits)
    compression_rate = sp.simplify(raw_bits / source_bits)
    return {
        "n": n,
        "b": b,
        "source_bits": source_bits,
        "raw_bits": raw_bits,
        "compression_ratio": compression_ratio,
        "compression_rate": compression_rate,
    }


def theorem_raw_capacity_uniform() -> bool:
    f = derive_formulas()
    n, b = f["n"], f["b"]
    raw_bits = f["raw_bits"]
    expected = n * b
    proved = sp.simplify(raw_bits - expected) == 0
    print(f"T1 raw capacity: {raw_bits} == n*b -> {proved}")
    return bool(proved)


def theorem_monotonic_capacity() -> bool:
    f = derive_formulas()
    n, b = f["n"], f["b"]
    raw_bits = f["raw_bits"]
    d_dn = sp.diff(raw_bits, n)
    d_db = sp.diff(raw_bits, b)
    proved = d_dn == b and d_db == n
    print(f"T2 capacity monotonic: dC/dn={d_dn}, dC/db={d_db} -> {proved}")
    return bool(proved)


def theorem_compression_ratio_monotonic() -> bool:
    f = derive_formulas()
    n, b, source_bits = f["n"], f["b"], f["source_bits"]
    ratio = f["compression_ratio"]
    d_dsource = sp.diff(ratio, source_bits)
    d_db = sp.diff(ratio, b)
    d_dn = sp.diff(ratio, n)

    # Under positive symbols: source growth increases ratio; profile widening
    # decreases ratio. SymPy leaves signs symbolic, so prove by substituting the
    # positive-form expressions we derived.
    source_positive = sp.simplify(d_dsource - 1 / (n * b)) == 0
    bits_negative = sp.simplify(d_db + source_bits / (n * b**2)) == 0
    channels_negative = sp.simplify(d_dn + source_bits / (n**2 * b)) == 0
    proved = source_positive and bits_negative and channels_negative
    print(
        "T3 compression monotonic: "
        f"dR/dsource={d_dsource}, dR/db={d_db}, dR/dn={d_dn} -> {proved}"
    )
    return bool(proved)


def theorem_epigene16_witnesses() -> bool:
    f = derive_formulas()
    raw_bits = f["raw_bits"]
    ratio = f["compression_ratio"]
    n, b, source_bits = f["n"], f["b"], f["source_bits"]

    cases = [
        (16, 4, 8_000, 64, 125.0),
        (16, 8, 8_000, 128, 62.5),
        (16, 8, 80_000, 128, 625.0),
    ]
    ok = True
    for nv, bv, sv, expected_bits, expected_ratio in cases:
        bits_v = int(raw_bits.subs({n: nv, b: bv}))
        ratio_v = float(ratio.subs({n: nv, b: bv, source_bits: sv}))
        case_ok = bits_v == expected_bits and abs(ratio_v - expected_ratio) < 1e-9
        print(
            "T4 witness: "
            f"n={nv}, b={bv}, source={sv} -> raw={bits_v}, ratio={ratio_v:.1f} -> {case_ok}"
        )
        ok = ok and case_ok
    return ok


def run() -> bool:
    print("EPIGENE-16 SymPy capacity checks:\n")
    checks = [
        theorem_raw_capacity_uniform(),
        theorem_monotonic_capacity(),
        theorem_compression_ratio_monotonic(),
        theorem_epigene16_witnesses(),
    ]
    ok = all(checks)
    print(f"\nEPIGENE-16 capacity algebra verified: {ok}")
    return ok


if __name__ == "__main__":
    import sys

    sys.exit(0 if run() else 1)

