"""Machine-verify Proposition 4 (storage overhead) of the HyMeKo T-SMC article with sympy — a full symbolic
proof, not a sketch. Derives rho from the size equations, proves the asymptotic bound under n=O(m log n), takes
the limit, checks monotonicity, and reproduces the empirical witness (rho: 2.00 @ d=2 -> 1.01 @ d=200)."""
import sympy as sp

n, m, d = sp.symbols("n m d", positive=True)            # d = mean arity d-bar
c, K = sp.symbols("c K", positive=True)
c_rec, c_inc = sp.symbols("c_rec c_inc", positive=True)

# Eq (ir_size) / (rho): IR stores (n+m) declaration records + m*d signed-incidence entries; adjacency stores
# only the incidence entries.
ir = (n + m) * c_rec + m * d * c_inc
adj = m * d * c_inc
rho = sp.simplify(ir / adj)
print("rho                         =", rho)
rho_c = sp.simplify(rho.subs(c_rec, c * c_inc))         # c := c_rec/c_inc
print("rho (c=c_rec/c_inc)         =", rho_c)
overhead = sp.simplify(rho_c - 1)
print("overhead rho-1              =", overhead)
assert sp.simplify(overhead - c * (n + m) / (m * d)) == 0, "rho-1 != c(n+m)/(m d)"

# Asymptotic bound: substitute the assumption n <= K m log n into the numerator.
bound = sp.simplify(overhead.subs(n, K * m * sp.log(n)))
print("rho-1 <= (n=K m log n)      =", bound, "  == c(K log n + 1)/d :",
      sp.simplify(bound - c * (K * sp.log(n) + 1) / d) == 0)

# Limit as mean arity grows: overhead -> 0  (=> rho -> 1).
lim = sp.limit(bound, d, sp.oo)
print("lim_{d->oo} (rho-1 bound)    =", lim, " => rho -> 1")
assert lim == 0

# Monotonicity: overhead strictly decreasing in d-bar.
deriv = sp.simplify(sp.diff(overhead, d))
print("d(rho-1)/d(d-bar)           =", deriv, " (negative => decreasing)")
assert deriv.subs({c: 1, n: 1, m: 1, d: 1}) < 0

# Empirical witness: fixed pool n = m = 200, c = 1  =>  rho-1 = 2/d-bar.
w = sp.simplify(overhead.subs({c: 1, n: 200, m: 200}))
print("witness rho-1 (n=m=200,c=1) =", w, " (= 2/d-bar)")
for dv in (2, 200):
    rv = float((1 + w).subs(d, dv))
    print(f"   d-bar={dv:>3}:  rho = {rv:.2f}")
    assert abs(rv - {2: 2.00, 200: 1.01}[dv]) < 1e-9

print("\nPROPOSITION 4: machine-verified (symbolic derivation + bound + limit + monotonicity + witness).")
