"""Machine-check the provable half of Proposition 2 (content-addressability): the collision-resistance bound.

P2 has two parts. (a) h depends only on the canonical IR -- this is the same mechanism as P1 (denotation-equal
sources => same canonical IR => same digest), already property-tested at 800/800. (b) Distinct IRs collide with
negligible probability -- a BIRTHDAY BOUND on the 256-bit Blake3 digest. (b)'s arithmetic is provable with sympy;
the 'treat Blake3 as a random oracle' premise is a cryptographic ASSUMPTION, not a theorem (stated as such)."""
import sympy as sp

N, b = sp.symbols("N b", positive=True)

M = 2 ** b                                   # digest space of a b-bit hash
per_pair = 1 / M                             # P(two distinct random IRs share a digest)
birthday_upper = N ** 2 / (2 * M)            # P(any collision among N IRs) <= N(N-1)/2M <= N^2/2M

print("per-pair collision probability        =", per_pair, "  (= 2^-b)")
print("birthday bound  P(any among N)        <=", sp.simplify(birthday_upper), "  (= N^2 / 2^(b+1))")

b256 = birthday_upper.subs(b, 256)
# Solve N^2 / 2^257 <= 2^-128  =>  N <= 2^64.5  : the work factor for a collision is 2^(b/2)=2^128.
crossover = sp.solve(sp.Eq(b256, sp.Rational(2) ** -128), N)
crossover = [s for s in crossover if s.is_positive][0]
print(f"\nfor b=256: P(collision) <= 2^-128  while  N <= {sp.nsimplify(crossover)}  (= 2^64.5)")
print("=> the birthday work factor is 2^(b/2) = 2^128 hashes; below that, collisions are negligible.")
for logN in (32, 64, 100):
    p = b256.subs(N, sp.Integer(2) ** logN)
    print(f"   N = 2^{logN:<3}:  P(collision) <= 2^{float(sp.log(p, 2)):.1f}")

# sanity: the bound is monotone increasing in N and decreasing in b (more items worse, longer digest better).
assert sp.diff(birthday_upper, N).subs({N: 1, b: 256}) > 0
assert sp.diff(birthday_upper, b).subs({N: 2, b: 256}) < 0
print("\nPROPOSITION 2 (collision bound): machine-verified symbolically.")
print("  Caveat (honest): the random-oracle premise for Blake3 is a cryptographic ASSUMPTION, not proved here;")
print("  and a 256-bit digest cannot be injective (pigeonhole) -- equality-by-hash is practical, not absolute.")
