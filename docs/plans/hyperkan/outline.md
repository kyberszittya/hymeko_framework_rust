# Learned Forward Kinematics via Topology-Compiled HyperKAN

**Created:** 2026-03-08
**Goal:** First end-to-end empirical validation of the Hymeko stack — from `.hymeko` robot description through compiled sparse tensors to a trained NURBS-based HyperKAN that learns forward kinematics.
**Estimated timeline:** 2–3 weeks of focused Python-side work.
**Rust changes required:** None. Everything needed from the core is already built.

---

## Thesis Statement

Current learned kinematics methods treat robot structure as either a flat vector (MLP) or a tree (GNN on URDF). We show that compiling kinematic structure into a hypergraph — where each joint is a multi-body hyperedge connecting parent link, child link, axis, and frame — and learning on the incidence topology with geometrically-motivated NURBS activations yields better sample efficiency, interpretable internal representations, and structural generalization. The entire pipeline is driven by a single `.hymeko` source file compiled through a Rust topological engine.

---

## Phase 0: Robot Description (Day 1)

### Task 0.1 — Author a 6-DOF manipulator in `.hymeko`

Write `data/robotics/arm_6dof.hymeko` describing a serial 6-revolute arm. Reuse and extend the existing `meta_kinematics.hymeko` schema.

**Key design decision:** Each joint must be modeled as a **hyperedge** (not a tree edge). A single `@joint` hyperedge connects:

- `+parent_link` (source, outgoing)
- `-child_link` (target, incoming)
- `-axis` (consumed axis definition)
- `~frame` (neutral, the joint frame transform)

This 4-ary incidence is the entire point — URDF would flatten this into a parent-child pair with axis and origin as XML attributes. Here, the topology captures the multi-body relationship natively.

```
arm_6dof
  import "meta_kinematics.hymeko"

arm use kinematics.elements use kinematics.axes
{
    base_link: link { mass 5.0; }
    link1: link { mass 3.0; }
    link2: link { mass 3.0; }
    link3: link { mass 2.0; }
    link4: link { mass 1.5; }
    link5: link { mass 1.0; }
    link6: link { mass 0.5; }
    ee_link: link { mass 0.1; }

    // Each joint is a hyperedge with 4-ary incidence
    @j1: kinematics.rev_joint {
        +base_link, [[0,0,0.1],[0,0,0]] -link1,
        -AXIS_Z
    }
    // ... j2 through j6 follow the same pattern
    @j6: kinematics.rev_joint {
        +link6, [[0,0,0.05],[0,0,0]] -ee_link,
        -AXIS_Z
    }
}
```

### Task 0.2 — Compile and verify the star expansion

Load `arm_6dof.hymeko` through `PyHypergraphEngine`, produce the star expansion COO tensor, and verify the shape. Expected topology: 6 hyperedges (joints), ~14 nodes (7 links + frames/axes), star expansion dimension `|V| + |E|`, 6 slices.

```python
engine = PyHypergraphEngine()
ir = engine.load_file("data/robotics/arm_6dof.hymeko")
coo = engine.compile_star_expansion(ir)
print(coo.shape, coo.nnz)
# Verify: each joint-hyperedge should produce 4 incidence entries × 2 (bidirectional for neutral)
```

**Checkpoint:** Print the dense slices and confirm the incidence structure matches the kinematic chain. This is a hard gate — if the topology is wrong, everything downstream is meaningless.

---

## Phase 1: Data Generation (Day 2)

### Task 1.1 — Analytical FK oracle

Implement a standard DH-parameter forward kinematics function. No external dependency needed — just rotation matrices and homogeneous transforms.

**File:** `py/hyperkan/fk_oracle.py`

```python
def fk_6dof(theta: np.ndarray, dh_params: np.ndarray) -> np.ndarray:
    """
    theta: (6,) joint angles in radians
    dh_params: (6, 4) -> [a, alpha, d, theta_offset] per joint
    returns: (4, 4) homogeneous transform of end-effector
    """
```

This function is ground truth. It must be numerically verified against a known reference (e.g., a textbook Puma 560 configuration) before any training begins.

### Task 1.2 — Dataset generator

Sample uniform random joint configurations within joint limits, compute FK analytically, and produce training/validation/test splits.

**File:** `py/hyperkan/fk_dataset.py`

- **Input representation:** `theta ∈ R^6` (joint angles)
- **Output representation:** `(position ∈ R^3, rotation ∈ R^6)` where rotation uses the 6D continuous representation (Zhou et al., 2019) — avoids gimbal lock and discontinuities of Euler angles, avoids the antipodal ambiguity of quaternions. This is important because NURBS activations learn smooth curves; the output space must also be smooth.
- **Dataset sizes:** 1K, 5K, 10K, 50K, 100K samples. We need the full range to plot sample efficiency curves.
- **Joint limits:** `[-π, π]` for all joints initially. Later experiments can restrict to `[-π/2, π/2]` and test generalization to the full range.
- **Storage:** Simple `.pt` files via `torch.save`. No overengineering.

---

## Phase 2: NURBS Activation Layer (Days 3–6)

This is the novel component. It must be correct, differentiable, and numerically stable.

### Task 2.1 — Cox-de Boor NURBS evaluation (forward pass)

**File:** `py/hyperkan/nurbs.py`

Implement a differentiable NURBS evaluation layer. One NURBS curve per incidence connection in the star expansion.

**Core computation:**

$$f(x) = \frac{\sum_{i=0}^{n} w_i \cdot c_i \cdot N_{i,p}(x)}{\sum_{i=0}^{n} w_i \cdot N_{i,p}(x)}$$

Where:
- $N_{i,p}(x)$ are the B-spline basis functions of degree $p$ (Cox-de Boor)
- $w_i$ are the rational weights (learnable, must be strictly positive)
- $c_i$ are the control point values (learnable)
- $x$ is the input scalar (one node feature flowing through one incidence connection)

**Learnable parameters per incidence edge:**
- Control points: `c ∈ R^(n_control)` — initialized small random
- Rational weights: `w_raw ∈ R^(n_control)` — stored as unconstrained, passed through softplus before use

**Knot vector:** Uniform clamped knot vector, fixed (not learned). Degree `p = 3` (cubic) as default. The knot span defines the input domain. Inputs are normalized to `[0, 1]` via a per-edge running min/max or a fixed sigmoid pre-activation.

**Critical stability details:**

- Rational weights must pass through softplus + epsilon: `w = softplus(w_raw) + 1e-7`
- The denominator `Σ w_i N_i(x)` can still underflow if `x` is outside the knot support. Clamp `x` to `[knot[p], knot[n+1]]` before evaluation.
- Cox-de Boor recurrence is numerically stable if implemented iteratively (not recursively). Use the triangular table approach.
- Degree `p = 3` means `n_control + 1 + p + 1` knots. For `n_control = 8`, that's 12 knots.

**Implementation strategy:** Pure PyTorch, no custom CUDA. The batch evaluation vectorizes across all incidence edges simultaneously. Shape: `(batch, nnz)` where `nnz` is the number of non-zero incidence connections.

### Task 2.2 — Unit tests for NURBS layer

Before wiring anything to the network, verify in isolation:

- **Partition of unity:** For uniform weights `w_i = 1`, the denominator equals 1 everywhere in the support → the NURBS reduces to a standard B-spline.
- **Endpoint interpolation:** With clamped knots, `f(0) = c_0` and `f(1) = c_n`. Verify numerically.
- **Gradient flow:** `torch.autograd.gradcheck` on `c`, `w_raw`, and `x`. All three must have non-zero gradients.
- **Exact circle test:** A degree-2 NURBS with specific weights can represent an exact circular arc. Verify that the layer can reproduce this when weights are set manually. This proves the "conic section" claim is not just theoretical.

---

## Phase 3: HyperKAN Forward Pass (Days 7–10)

### Task 3.1 — Sparse incidence wiring

**File:** `py/hyperkan/hyperkan.py`

The HyperKAN layer takes the star expansion COO tensor from the Rust engine and uses it to wire the NURBS activations.

**Architecture for one HyperKAN layer:**

```
Input: x_v ∈ R^(num_nodes × feat_dim)

1. Inner pass (node → edge):
   For each incidence (v, e) in the star expansion:
     h_{v→e} = NURBS_inner_{v,e}(x_v)      # learnable per-incidence curve
   h_e = Σ_{v ∈ e} h_{v→e}                  # aggregate into hyperedge

2. Outer pass (edge → node):
   For each incidence (e, v) in the star expansion:
     m_{e→v} = NURBS_outer_{e,v}(h_e)       # learnable per-incidence curve
   x'_v = Σ_{e ∈ E(v)} m_{e→v}             # aggregate back to node
```

**Sparse execution:** The COO indices from the Rust engine define exactly which `(v, e)` pairs exist. The NURBS evaluation operates *only* on these non-zero incidence connections. Use `torch.index_select` and `torch.scatter_add` for the gather/scatter — no dense matrix multiplication.

**Parameter count per layer:** `nnz × n_control × 2` (control points + raw weights) for inner, same for outer. For a 6-DOF arm with ~24 incidence entries and 8 control points, that's `24 × 8 × 2 × 2 = 768` parameters per layer. Very lightweight.

### Task 3.2 — Full network architecture

```
Input: θ ∈ R^6 (joint angles)

1. Embedding:
   Scatter joint angles onto the node feature vectors.
   x_v[joint_node_i] = θ_i for joint-associated nodes
   x_v[other] = 0 (or learned positional embedding)

2. HyperKAN layers × L (L = 2 or 3):
   x = HyperKANLayer(x, incidence_coo)

3. Readout:
   Gather the end-effector node's feature vector.
   output = Linear(x_v[ee_node]) → R^9 (position + 6D rotation)
```

**Key detail:** The incidence COO tensor is compiled **once** from the `.hymeko` file and reused for every forward pass. The topology is static; only the NURBS parameters and node features change during training.

### Task 3.3 — Integration test

Before training, verify the forward pass runs without errors:

- Random `θ` input → output has correct shape `(batch, 9)`
- Backward pass produces non-zero gradients on all NURBS parameters
- No NaN values in forward or backward pass for 1000 random inputs

---

## Phase 4: Training Loop (Days 10–12)

### Task 4.1 — Training script

**File:** `py/hyperkan/train_fk.py`

- **Loss:** MSE on position + geodesic loss on rotation (convert 6D representation back to rotation matrix, compute `||I - R_pred^T R_true||_F`)
- **Optimizer:** Adam, lr=1e-3 with cosine annealing
- **Batch size:** 256
- **Epochs:** 500 (should be more than enough for FK regression)
- **Logging:** Loss, position error (mm), rotation error (degrees) per epoch. Save to CSV for plotting.

### Task 4.2 — Baseline implementations

All baselines must use approximately the same parameter count as the HyperKAN for fair comparison.

**Baseline A — MLP:**
`θ → Linear(6, 64) → ReLU → Linear(64, 64) → ReLU → Linear(64, 9)`
Standard architecture. Treats the robot as a flat vector. ~5K parameters.

**Baseline B — Standard KAN (B-spline, no hypergraph):**
Same architecture as HyperKAN but with standard B-spline activations (rational weights fixed to 1.0) and operating on a flat fully-connected graph (all nodes connected to all nodes). This isolates the NURBS contribution.

**Baseline C — GNN on kinematic tree:**
A 2-layer GCN/GAT operating on the standard tree topology (parent-child edges only, as URDF would represent it). This isolates the hypergraph contribution.

**Baseline D — HyperKAN (ours):**
NURBS activations on the compiled star expansion topology.

---

## Phase 5: Evaluation & Interpretability (Days 13–16)

### Task 5.1 — Sample efficiency curves

Train all four models on dataset sizes {1K, 5K, 10K, 50K, 100K}. Plot test error vs. training set size. The hypothesis: HyperKAN should achieve lower error at smaller dataset sizes because the topology constrains the learning problem.

### Task 5.2 — Structural generalization

- **Test A (Joint removal):** Remove joint 6 (wrist roll) from the `.hymeko` file, recompile the star expansion, and test the HyperKAN trained on the 6-DOF arm. The MLP cannot do this at all (input dimension changed). The GNN tree baseline requires retraining. The HyperKAN should degrade gracefully because the NURBS functions on joints 1–5 are still valid.
- **Test B (Link parameter change):** Modify link lengths in the DH parameters without retraining. Measure how prediction error scales with the magnitude of the perturbation. The topology-aware model should be more robust because it has learned the *functional form* of the FK mapping, not just a memorized lookup table.

### Task 5.3 — NURBS activation visualization (the interpretability result)

This is the most important qualitative result. For each incidence connection in the trained HyperKAN:

- Plot the learned NURBS curve `f(x)` over the input domain `[0, 1]`
- On the same axes, plot the corresponding analytical function from the FK chain (e.g., the `sin(θ)` or `cos(θ)` that the DH formulation uses for that joint)

**Hypothesis:** The NURBS curves on joint-to-link incidence connections should visually approximate trigonometric functions. If they do, this demonstrates that:
1. The network is learning physically meaningful representations, not arbitrary mappings
2. The NURBS parameterization (with rational weights) is capturing the geometry
3. The hypergraph topology is routing information correctly through the kinematic chain

**Visualization script:** `py/hyperkan/visualize_nurbs.py`
- One subplot per incidence connection
- Overlay: learned curve, analytical reference, and the residual
- Export as PDF for publication

### Task 5.4 — Rational weight analysis

Extract the learned rational weights `w_i` after training. For incidence connections where the underlying function is linear (e.g., a prismatic joint), the weights should converge toward uniform (recovering a standard B-spline). For connections where the function is trigonometric (revolute joints), the weights should deviate — specifically, the circular arc representation requires `w_i = cos(Δθ/2)` at the interior control points.

Report the weight distributions and correlate them with joint types.

---

## Phase 6: Packaging & Documentation (Days 17–18)

### Task 6.1 — Results table

| Model | Params | 1K MSE | 10K MSE | 100K MSE | Pos. Error (mm) | Rot. Error (°) |
|---|---|---|---|---|---|---|
| MLP | ~5K | — | — | — | — | — |
| KAN (B-spline) | ~5K | — | — | — | — | — |
| GNN (tree) | ~5K | — | — | — | — | — |
| **HyperKAN (ours)** | ~5K | — | — | — | — | — |

### Task 6.2 — Figure list

1. The `.hymeko` → star expansion → HyperKAN pipeline diagram
2. Sample efficiency curves (4 models × 5 dataset sizes)
3. NURBS activation plots vs. analytical FK functions (the interpretability figure)
4. Structural generalization: error under joint removal / link perturbation
5. Rational weight distributions by joint type

### Task 6.3 — README update

Update the repository README with:
- A "Quick Start" showing the full pipeline from `.hymeko` to trained model in ~10 commands
- The results table
- One hero figure (the NURBS activation plot)

---

## File Structure

```
py/hyperkan/
├── fk_oracle.py              # Analytical FK (DH parameters)
├── fk_dataset.py             # Data generation and splitting
├── nurbs.py                  # Differentiable NURBS layer (PyTorch)
├── hyperkan.py               # HyperKAN layer and full network
├── baselines.py              # MLP, KAN, GNN baselines
├── train_fk.py               # Training loop with all models
├── eval_fk.py                # Evaluation, tables, generalization tests
├── visualize_nurbs.py        # NURBS curve visualization
└── README.md                 # Module-level docs

data/robotics/
├── meta_kinematics.hymeko    # (existing)
└── arm_6dof.hymeko           # New: 6-DOF manipulator description
```

---

## Hard Gates

These are binary pass/fail checkpoints. Do not proceed past a gate until it passes.

- [ ] **Gate 0:** `arm_6dof.hymeko` compiles and produces a star expansion COO tensor with the correct shape and nnz count matching the kinematic topology.
- [ ] **Gate 1:** FK oracle reproduces a known textbook result (e.g., Puma 560 home position) to < 1e-10 numerical error.
- [ ] **Gate 2:** NURBS layer passes `torch.autograd.gradcheck` for all learnable parameters.
- [ ] **Gate 3:** HyperKAN forward + backward pass runs without NaN on 1000 random inputs.
- [ ] **Gate 4:** HyperKAN achieves < 1mm position error on the 100K dataset (proving the pipeline works at all).
- [ ] **Gate 5:** At least one NURBS activation visually correlates with the analytical FK function (proving interpretability claim is real).

---

## What This Does NOT Require

- No changes to the Rust core
- No URDF/SDF writer
- No daemon or shared memory streaming
- No entropy module
- No Triton kernels
- No new lexer features
- No CBOR/QR serialization

Everything is Python-side, consuming the existing Rust API through `PyHypergraphEngine`.