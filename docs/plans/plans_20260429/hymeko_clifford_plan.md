# `hymeko_clifford` — development plan

Clifford algebra autograd backend for G-SPHF, integrated into the HyMeKo workspace.

---

## Context and constraints

### What exists

| Crate | Relevant content |
|---|---|
| `autograd/` | Tape-based reverse-mode AD, scalar/tensor values |
| `gsphf/` | G-SPHF field operations, Laplacian, incidence (scalar weights) |
| `hymeko_core` | Full crate infrastructure, workspace config |

### What is fundamentally new

The existing autograd tape assumes **commutative multiplication**. In Clifford algebra
$Cl(p,q)$, for $F = AB$, the reverse-mode adjoints are:

$$\bar{A} \mathrel{+}= \bar{F}\,\tilde{B}, \qquad \bar{B} \mathrel{+}= \tilde{A}\,\bar{F}$$

where $\tilde{B}$ is the **grade reverse** (reverses factor order within each blade).
The backward pass is structurally different — the scalar tape cannot be shimmed.
A new crate is required.

### Design invariant

`hymeko_clifford` must not depend on `hymeko_core`. Dependency goes the other way
eventually. Keep it self-contained.

---

## Crate structure

```
hymeko_clifford/
├── Cargo.toml
└── src/
    ├── lib.rs
    ├── algebra/
    │   ├── mod.rs
    │   ├── multivector.rs      # Multivector<const N: usize>, dense repr
    │   ├── blade.rs            # blade bitmask, grade, canonical sign
    │   ├── products.rs         # geometric, outer, inner, scalar products
    │   ├── grade.rs            # grade_proj, reverse, conjugate, involution
    │   ├── hodge.rs            # Hodge dual, pseudoscalar I_n
    │   └── sparse.rs           # SparseMv trait (stub — Phase 5)
    ├── autograd/
    │   ├── mod.rs
    │   ├── tape.rs             # MvTape: arena of nodes
    │   ├── var.rs              # MvVar<N>: VarId + tape ref
    │   ├── ops.rs              # MvOp enum, all Clifford-aware backward arms
    │   └── backward.rs         # reverse pass, adjoint accumulation
    ├── gsphf/
    │   ├── mod.rs
    │   ├── incidence.rs        # CliffordIncidence<N>, blade-valued B_s
    │   ├── kernel.rs           # GgkKernel trait, B-spline + RBF impls
    │   ├── laplacian.rs        # CliffordLaplacian<N> = δ*δ via MvVar ops
    │   └── field.rs            # field energy ℰ(Φ), gradient flow
    └── dynamics/
        ├── mod.rs
        ├── integrator.rs       # FieldIntegrator, Euler + RK4
        ├── optimizer.rs        # MvAdam, grade-masked moments
        └── level.rs            # GsphfLevel enum, gates active terms
```

---

## Phase 1 — Clifford algebra foundation

**Goal:** Complete `algebra/` module. No autograd. All products correct,
grade arithmetic enforced at compile time where possible.

### 1.1 Representation

```rust
/// Dense multivector in Cl(p,q). Components indexed by blade bitmask.
/// Blade e_1∧e_3∧e_4 → index 0b1101. Grade = popcount(index).
pub struct Multivector<const N: usize> {
    pub components: [f64; 1 << N],
}

pub struct Signature {
    pub p: usize,  // positive squares
    pub q: usize,  // negative squares
    // r (null) deferred to Phase 5
}
```

Upper bound: $N \leq 16$ gives $2^{16} = 65{,}536$ components per multivector.
For typical G-SPHF hypergraphs ($N \leq 12$) this is fine.
Sparse representation is stubbed as a trait in `sparse.rs` — do not implement yet,
but design the trait boundary so Phase 5 can swap without touching Phase 3/4.

### 1.2 `blade_product` — the load-bearing function

```rust
/// Returns (result_blade_index, sign) for the product of two basis blades.
fn blade_product(a_idx: usize, b_idx: usize, sig: &Signature) -> (usize, f64);
```

Implementation:
1. `result_idx = a_idx ^ b_idx`
2. `sign = canonical_reorder_sign(a_idx, b_idx)` — counts adjacent transpositions
   to sort the combined factor sequence; each swap contributes $-1$
3. For indices in both `a_idx` and `b_idx` (i.e. `a_idx & b_idx != 0`): apply
   metric. Basis vector $e_i^2 = +1$ if $i \leq p$, $-1$ if $p < i \leq p+q$.

`canonical_reorder_sign` is the single most error-prone function in the codebase.
Write it, then immediately write exhaustive unit tests before touching anything else.

### 1.3 Products

```rust
impl<const N: usize> Multivector<N> {
    pub fn geo(&self, rhs: &Self, sig: &Signature) -> Self;   // geometric product AB
    pub fn outer(&self, rhs: &Self) -> Self;                   // outer (wedge) A∧B
    pub fn inner(&self, rhs: &Self, sig: &Signature) -> Self;  // inner A·B = ⟨AB⟩_{|r-s|}
    pub fn scalar_product(&self, rhs: &Self, sig: &Signature) -> f64; // ⟨AB⟩_0
    pub fn reverse(&self) -> Self;                             // Ã: reverses blade factors
    pub fn grade_involution(&self) -> Self;                    // Â: (-1)^k on grade-k part
    pub fn conjugate(&self, sig: &Signature) -> Self;          // A† = Ã with metric
    pub fn grade_proj(&self, k: usize) -> Self;                // ⟨A⟩_k
    pub fn norm_sq(&self, sig: &Signature) -> f64;             // ⟨AÃ⟩_0
}
```

### 1.4 Hodge dual

```rust
/// Pseudoscalar I_n = e_1 e_2 … e_n, index = (1 << N) - 1
pub fn pseudoscalar<const N: usize>(sig: &Signature) -> Multivector<N>;

/// Hodge dual: ★A = A · I_n^{-1}
pub fn hodge<const N: usize>(a: &Multivector<N>, sig: &Signature) -> Multivector<N>;
```

### 1.5 Unit tests (required before Phase 2)

| Test | Condition |
|---|---|
| Basis squares | $e_i^2 = +1$ for $i \leq p$, $-1$ for $p < i \leq p+q$ |
| Anticommutativity | $e_i e_j = -e_j e_i$ for $i \neq j$ |
| Reverse involution | $\widetilde{(AB)} = \tilde{B}\tilde{A}$ |
| Outer nilpotency | $e_i \wedge e_i = 0$ |
| Hodge involution | $\star\star A = \pm A$ (sign depends on $n, k$, signature) |
| Grade projection | $\sum_k \langle A \rangle_k = A$ |
| `canonical_reorder_sign` | exhaustive check for $N=4$: all $2^4 \times 2^4 = 256$ pairs |

---

## Phase 2 — Multivector autograd tape

**Goal:** `MvTape` and `MvVar<N>` with correct reverse-mode for all Clifford
operations. This is the critical phase. Everything downstream depends on it.

### 2.1 Tape structure

```rust
pub type VarId = usize;

pub struct TapeNode<const N: usize> {
    pub value: Multivector<N>,
    pub op: MvOp,
}

pub struct MvTape<const N: usize> {
    nodes: Vec<TapeNode<N>>,
}

pub struct MvVar<'t, const N: usize> {
    pub id: VarId,
    tape: &'t MvTape<N>,
}
```

Reuse the arena pattern from the existing `autograd/` crate. Do not copy the
backward closure approach — store `MvOp` variants instead.

### 2.2 Op enum

```rust
pub enum MvOp {
    Leaf,                                           // input variable, no parents
    GeomProd   { lhs: VarId, rhs: VarId },          // F = A * B (geometric)
    OuterProd  { lhs: VarId, rhs: VarId },          // F = A ∧ B
    InnerProd  { lhs: VarId, rhs: VarId },          // F = A · B
    Add        { lhs: VarId, rhs: VarId },
    ScalarMul  { src: VarId, scalar: f64 },
    GradeProj  { src: VarId, grade: usize },
    Reverse    { src: VarId },
    HodgeDual  { src: VarId },
    Negate     { src: VarId },
}
```

### 2.3 Backward pass — critical rules

**`GeomProd`:**

```rust
MvOp::GeomProd { lhs, rhs } => {
    let f_bar = &adjoints[out_id];
    let b_rev = tape.value(rhs).reverse();
    let a_rev = tape.value(lhs).reverse();
    // ∂/∂A: F̄ * B̃
    adjoints[lhs] += f_bar.geo(&b_rev, sig);
    // ∂/∂B: Ã * F̄
    adjoints[rhs] += a_rev.geo(f_bar, sig);
}
```

Order matters. `f_bar * b_rev` not `b_rev * f_bar`.

**`GradeProj`:**

```rust
MvOp::GradeProj { src, grade } => {
    // backward is grade projection of the adjoint — identity on target grade, zero elsewhere
    adjoints[src] += adjoints[out_id].grade_proj(*grade);
}
```

**`Reverse`:**

```rust
MvOp::Reverse { src } => {
    // reverse is self-adjoint up to sign: ∂/∂A of tr(F̄ᵀ Ã) = F̄ reversed
    adjoints[src] += adjoints[out_id].reverse();
}
```

**`HodgeDual`:**

```rust
MvOp::HodgeDual { src } => {
    // ★ is linear; adjoint is ★ applied to the incoming gradient
    adjoints[src] += hodge(&adjoints[out_id], sig);
}
```

### 2.4 Gradient check infrastructure

Write this **before** implementing any backward arm:

```rust
pub fn finite_diff_check<const N: usize, F>(
    f: F,
    inputs: &[Multivector<N>],
    sig: &Signature,
    eps: f64,
    tol: f64,
) -> bool
where
    F: Fn(&[MvVar<N>]) -> MvVar<N>,
```

Perturb each component of each input by `±eps`, compare to autograd gradient.
Run this on `GeomProd`, `OuterProd`, `GradeProj`, `HodgeDual` before declaring
Phase 2 complete. A failing check here means a sign error in `canonical_reorder_sign`
or a wrong adjoint formula — catch it here, not in Phase 3.

---

## Phase 3 — G-SPHF primitives in Clifford

**Goal:** `gsphf/` submodule implementing G-SPHF operations over `MvVar`,
automatically differentiable through the tape.

### 3.1 Clifford incidence

$B_s$ is not stored as a matrix. Each hyperedge $e$ is encoded as a blade:

$$b_e = \bigwedge_{v \in e,\, b_{ve} \neq 0} \text{sign}(b_{ve}) \cdot \mathbf{e}_v$$

```rust
pub struct CliffordIncidence<const N: usize> {
    /// One blade per hyperedge, encoding sign and vertex membership.
    pub edge_blades: Vec<Multivector<N>>,
    pub sig: Signature,
}

impl<const N: usize> CliffordIncidence<N> {
    /// Discrete coboundary δ⁰Φ: for each edge e, returns Σ_{v∈e} b_{ve} Φ(v)
    pub fn coboundary<'t>(&self, field: &[MvVar<'t, N>]) -> Vec<MvVar<'t, N>>;
}
```

The coboundary is a sum of `ScalarMul` + `Add` operations over `MvVar` — fully
tracked by the tape.

### 3.2 Clifford Laplacian

$$(\mathcal{L}_s \Phi)(v) = \sum_{e \ni v} b_{ve} \cdot w(e) \cdot (\delta^0 \Phi)(e)$$

```rust
pub struct CliffordLaplacian<const N: usize> {
    pub incidence: CliffordIncidence<N>,
    pub kernel: Box<dyn GgkKernel<N>>,
}

impl<const N: usize> CliffordLaplacian<N> {
    pub fn apply<'t>(&self, field: &[MvVar<'t, N>]) -> Vec<MvVar<'t, N>>;
}
```

Each step is a `MvVar` operation — `apply` returns a field whose gradient flows
back through `GeomProd` and `Add` tape nodes.

### 3.3 GGK kernel trait

```rust
pub trait GgkKernel<const N: usize>: Send + Sync {
    /// Returns scalar weight w(e) for hyperedge blade b_e given current field.
    fn weight(&self, edge_blade: &Multivector<N>, field_at_edge: &Multivector<N>) -> f64;

    /// Inner product on edge space (discrete Hodge star G).
    fn inner(&self, a: &Multivector<N>, b: &Multivector<N>) -> f64;
}
```

Implement:
- `BSplineGgk` — B-spline basis, Gram matrix as $G$, knot measure as $\mu$.
  This is the TP model transformation instantiation.
- `RbfGgk` — radial basis function, Gaussian kernel.

### 3.4 Field energy

```rust
/// ℰ(Φ) = Φᵀ ℒₛ Φ + λ ‖Φ‖²_K  (returns scalar MvVar at grade 0)
pub fn field_energy<'t, const N: usize>(
    field: &[MvVar<'t, N>],
    laplacian: &CliffordLaplacian<N>,
    lambda: f64,
) -> MvVar<'t, N>;
```

This is the primary loss function for G-SPHF learning.

### 3.5 Integration test

Construct a 4-vertex, 3-hyperedge test hypergraph. Run gradient flow for 100 steps.
Verify convergence to harmonic field: $\|\mathcal{L}_s \Phi\|^2 < \varepsilon$.

---

## Phase 4 — Field dynamics and optimizer

**Goal:** Integrable gradient flow with level-switchable dynamics and
a multivector-aware optimizer.

### 4.1 Field integrator

```rust
pub enum Stepper { Euler, Rk4 }

pub struct FieldIntegrator<const N: usize> {
    pub stepper: Stepper,
    pub dt: f64,
    pub level: GsphfLevel,
}

impl<const N: usize> FieldIntegrator<N> {
    /// One step of  dΦ/dt = -ℒₛ Φ + u
    /// Tape is rebuilt per step (dynamic graph). Static graph opt deferred to Phase 5.
    pub fn step(
        &self,
        field: &mut Vec<Multivector<N>>,
        laplacian: &CliffordLaplacian<N>,
        input: &[Multivector<N>],
    );
}
```

Tape is rebuilt each step — correct but not maximally efficient.
Static graph optimization (fixed topology) is a Phase 5 item.

### 4.2 GsphfLevel enum

```rust
pub enum GsphfLevel {
    L0,  // plain field diffusion: dΦ/dt = -ℒₛ Φ
    L1,  // + GGK kernel weighting
    L2,  // + cochain complex terms (δ¹ contribution)
    L3,  // + Hodge decomposition projection
    L4,  // + full state-space dynamics with input u
    L5,  // + U(1) gauge connection (complex weights)
    L6,  // + RKHS functional lift (λ‖Φ‖²_K regularizer)
}
```

Each level gates additional terms in the field update.
Levels 0–4 use real `f64` weights. Level 5 requires promoting to `Complex<f64>` —
plan the type boundary now so the promotion is additive, not a rewrite.

### 4.3 MvAdam optimizer

```rust
pub struct MvAdam<const N: usize> {
    pub lr: f64,
    pub beta1: f64,   // default 0.9
    pub beta2: f64,   // default 0.999
    pub eps: f64,
    m: Vec<Multivector<N>>,   // first moment
    v: Vec<Multivector<N>>,   // second moment (componentwise)
    t: usize,
}
```

Moment updates are **componentwise on the $2^N$ component array**.
Do not mix grades in moment updates — gate each component's update by its
grade membership. This is a one-line bitmask gate per component: only update
moments for components where `popcount(idx) == target_grade`, or do it flat
and accept that grade mixing in moments is benign for convergence (simpler,
probably fine in practice — decide empirically in Phase 4).

### 4.4 End-to-end test

- 6-vertex hypergraph, random initialization of $\Phi$
- `GsphfLevel::L0`, `Stepper::Rk4`, 500 steps
- Assert: $\mathcal{E}(\Phi) < 0.01 \cdot \mathcal{E}(\Phi_0)$
- Assert: $\|\mathcal{L}_s \Phi\|_F < \varepsilon$ (harmonic convergence)

---

## Phase 5 — SIMD and sparse (deferred)

Design the trait boundary now. Implement after Phase 4 is stable.

### Sparse multivector

```rust
/// Sparse multivector: sorted list of (blade_index, coefficient) pairs.
pub struct SparseMv {
    pub terms: SmallVec<[(u32, f64); 8]>,  // sorted by blade index
}
```

Geometric product of two sparse multivectors = merge of two sorted lists,
$O(|a| \cdot |b|)$ in the worst case. For G-SPHF fields (mostly grade-1 and grade-$k$
for a fixed small $k$) this is much better than dense $2^N$.

### Swappable trait boundary (design now, Phase 1)

```rust
pub trait MvRepr<const N: usize>: Clone + Send + Sync {
    fn get(&self, blade_idx: usize) -> f64;
    fn set(&mut self, blade_idx: usize, val: f64);
    fn geo(&self, rhs: &Self, sig: &Signature) -> Self;
    fn grade_proj(&self, k: usize) -> Self;
    fn reverse(&self) -> Self;
    // ... etc
}
```

`Multivector<N>` and `SparseMv` both implement `MvRepr`.
All Phase 3/4 code is generic over `R: MvRepr<N>`.
Phase 1 stub: add `sparse.rs` with trait definition and a `todo!()` impl.

### SIMD

AVX2 on the $2^N$ component array. Grade-$k$ projection = bitmask select
(`vpand` + masked load). Geometric product inner loop = scatter-accumulate
over blade pairs. Profile before implementing — the sparse repr may make SIMD
unnecessary for realistic hypergraph sizes.

---

## Critical path

```
Phase 1: canonical_reorder_sign correct + exhaustively tested
    ↓
Phase 2: GeomProd backward correct + gradient check passing
    ↓
Phase 3: CliffordLaplacian differentiable, field energy computes
    ↓
Phase 4: end-to-end gradient flow converges
    ↓ (deferred)
Phase 5: sparse + SIMD
```

Phase 2 is the only genuinely hard step. If `canonical_reorder_sign` has a sign
error it will propagate silently through Phases 3 and 4 — gradient checks will
fail in non-obvious ways. The finite-difference check in Phase 2 is not optional.

---

## Open questions to resolve before Phase 3

1. **Level 5 complex promotion.** Does `MvVar<N>` become `MvVar<N, F: Field>` where
   `F` is `f64` or `Complex<f64>`? Or is Level 5 a separate type? Decide before
   writing Phase 3 generics.

2. **Tape lifetime.** `MvVar<'t, N>` borrows the tape. For RK4, the tape is rebuilt
   per step — field values must be extracted to owned `Multivector<N>` between steps.
   This is the right design but adds a copy per step. Benchmark against arena reset.

3. **Nilpotency at Level 2.** The cochain complex requires $\delta^1 \circ \delta^0 = 0$.
   For general hyperedges (cardinality $> 2$) this is a constraint on admissible patch
   decompositions $\mathcal{P}$, not a free theorem. Define the admissibility check and
   add it as a validation step in `CliffordIncidence::new()`.

4. **Grade-masked Adam.** Empirically test whether grade-mixed moment updates hurt
   convergence before adding complexity of grade gates. If convergence is fine without
   gates, remove the gate logic.
