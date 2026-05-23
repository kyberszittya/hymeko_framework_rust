# Entropy-guided signed hypergraph → PyTorch

**Scope.** Implementation spec for two parallel tracks that share a common entropy stage:

1. A native PyTorch package (`ehk_torch`) implementing entropy-driven signed-hypergraph construction, `HypergraphConv`, and SignedKAN activations parameterised over the GGK kernel.
2. A HyMeKo codegen backend (`hymeko_torch`) that emits `torch.nn.Module` Python files from `.hmk` descriptions, reusing the runtime from (1).

**Non-goals.** CUDA kernels, FPGA integration, distributed training. Keep first version single-GPU. Do not touch the HSMM path.

**Deliverables.** Two Python packages + one Rust crate (codegen) + one integration example on a dataset Csaba already has infrastructure for (cattle-corridor or a public mesh-segmentation benchmark — whichever is faster to wire up).

---

## 0. Conventions

- Python ≥ 3.11, PyTorch ≥ 2.3, Rust edition 2021.
- Type hints mandatory on all public functions. `mypy --strict` clean.
- Sparse tensors: `torch.sparse_coo_tensor`. Do not materialise dense `B` above 2k nodes.
- Signed incidence `B ∈ {−1, 0, +1}^{|V|×|E|}`. Store structure (pattern) and sign (values) separately in the sparse COO layout so autograd on edge weights does not collide with sign flips.
- Every public function that returns a gradient-bearing tensor documents the gradient path in its docstring.
- No silent dtype coercion. Fail loudly on float32/float64 mismatches.

---

## 1. Package layout

```
ehk_torch/                          # native path
├── pyproject.toml
├── src/ehk_torch/
│   ├── __init__.py
│   ├── entropy/
│   │   ├── shannon.py              # differential + discrete H
│   │   ├── mutual_info.py          # MI estimators
│   │   └── structural.py           # H_struct over signed hypergraphs
│   ├── construction/
│   │   ├── mi_threshold.py
│   │   ├── structural_min.py
│   │   └── learned.py              # Gumbel-softmax construction
│   ├── layers/
│   │   ├── hypergraph_conv.py
│   │   └── signed_kan.py           # GGK-parameterised activation
│   ├── kernels/
│   │   └── ggk.py                  # K = (B, G, μ, r) implementation
│   ├── ops/
│   │   └── sparse_signed.py        # signed sparse matmul helpers
│   └── io/
│       └── checkpoint.py           # saves (state_dict, B, K-params)
└── tests/
    └── ...                         # pytest, see §8

hymeko_torch/                       # codegen path (lives in hymeko workspace)
├── Cargo.toml                      # new crate in existing workspace
├── src/
│   ├── lib.rs
│   ├── codegen.rs                  # .hmk IR → Python source emission
│   ├── lowering.rs                 # HyMeKo IR → torch-compatible IR
│   └── runtime_binding.rs          # references to ehk_torch runtime
└── tests/

hymeko_torch_runtime/               # thin Python shim
├── pyproject.toml
└── src/hymeko_torch_runtime/
    ├── __init__.py
    └── factory.py                  # from_hmk(path) -> nn.Module
```

`hymeko_torch_runtime` depends on `ehk_torch`. `ehk_torch` has no HyMeKo dependency — it must be usable standalone.

---

## 2. Stage 1 — Entropy computation

### 2.1 Shannon / differential entropy (`entropy/shannon.py`)

```python
def shannon_entropy(
    x: torch.Tensor,          # (N,) or (N, d)
    method: Literal["binning", "knn", "kde"] = "knn",
    k: int = 3,
    bins: int = 64,
) -> torch.Tensor:
    """
    Per-channel entropy estimate.

    Returns (d,) tensor. Differentiable only for method="kde".
    For method="knn" uses Kozachenko-Leonenko; not differentiable,
    suitable for structure inference but not as a loss term.
    """
```

- Discrete inputs (integer dtype) → binning. Continuous → knn default.
- kNN via `torch.cdist` with `k+1` neighbours; drop self (distance 0) deterministically.
- KDE via Gaussian kernel with Silverman bandwidth; this is the only differentiable path and the only one usable inside a loss.

### 2.2 Mutual information (`entropy/mutual_info.py`)

Three estimators, selected by argument:

```python
def mutual_information(
    x: torch.Tensor,          # (N, d_x)
    y: torch.Tensor,          # (N, d_y)
    method: Literal["ksg", "mine", "infonce"] = "ksg",
    **kwargs,
) -> torch.Tensor | tuple[torch.Tensor, nn.Module]:
    """
    KSG: non-differentiable, fast, use for candidate edge generation.
    MINE: differentiable, returns (mi_estimate, critic_network).
    InfoNCE: differentiable, contrastive lower bound, returns scalar.
    """
```

- KSG via `sklearn.feature_selection.mutual_info_regression` for cheap non-differentiable passes. Wrap with a tensor conversion, do not reinvent.
- MINE critic: 2-layer MLP, 128 hidden, Donsker-Varadhan representation, EMA stabilisation on the denominator.
- InfoNCE: standard symmetric form.

### 2.3 Structural entropy (`entropy/structural.py`)

Csaba's ten-metric specification is the source of truth. Implement each as a separate function returning a scalar `torch.Tensor`:

```python
def h_struct_v1(B: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor: ...
def h_struct_v2(...) -> torch.Tensor: ...
# ... through v10
```

One dispatcher:

```python
def structural_entropy(
    B: torch.Tensor,
    sigma: torch.Tensor,
    variant: Literal["v1", ..., "v10"] = "v1",
) -> torch.Tensor:
```

Tests must confirm each variant matches the reference values from Csaba's existing Python evaluation pipeline for the SISY 2026 paper. Variants must not be refactored into a "unified" form until that parity is demonstrated — the variants are the research object.

---

## 3. Stage 2 — Hypergraph construction

### 3.1 MI-threshold clustering (`construction/mi_threshold.py`)

```python
def build_from_mi(
    x: torch.Tensor,          # (N, d)
    tau: float,
    max_edge_size: int = 16,
    sign_from: Literal["correlation", "gradient"] = "correlation",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (B_sparse, sigma).
    B_sparse: torch.sparse_coo_tensor of shape (N, E), values in {-1,+1}.
    sigma: dense (nnz,) for direct access.
    """
```

- Compute pairwise MI matrix (N × N) via KSG.
- Threshold at `tau`, extract connected components of the resulting graph as hyperedge candidates, cap cardinality at `max_edge_size`.
- Sign: positive if correlation within cluster > 0, negative otherwise. Mixed-sign clusters get split.

### 3.2 Structural-entropy minimisation (`construction/structural_min.py`)

Greedy refinement. Initial partition = MI-threshold output. At each step:

- Propose a merge, split, or move of a node between hyperedges.
- Accept if `H_struct(B')` < `H_struct(B)`; optionally simulated-annealing acceptance with cooling schedule.
- Stop on convergence or max iterations.

This function is slow and runs pre-training or periodically, not per batch.

### 3.3 Learned construction (`construction/learned.py`)

A small module predicting edge membership:

```python
class LearnedHypergraphBuilder(nn.Module):
    def __init__(self, d_in: int, n_edges: int, max_size: int, tau_gumbel: float = 1.0):
        ...
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (B_soft, sigma_soft). Use Gumbel-softmax."""
```

- Output is a soft incidence matrix during training, hardened at eval.
- Sign head is a separate tanh → {−1, +1} via straight-through estimator.
- Structural entropy enters the total loss as a regulariser on `B_soft`.

---

## 4. Stage 3a — Native PyTorch layers

### 4.1 GGK kernel (`kernels/ggk.py`)

```python
@dataclass
class GGKSpec:
    basis: Literal["bspline", "bezier", "rbf", "hermite", "wavelet"]
    support: Literal["interval", "simplex", "point_set", "manifold_patch"]
    degree: int
    n_knots: int           # or n_centres for RBF
    regularity: int        # C^k

class GGKBasis(nn.Module):
    """
    Evaluates basis functions of type K = (B, G, mu, r) at input points.
    mu (weights) and optionally r (regularity controls) are nn.Parameters.
    """
    def __init__(self, spec: GGKSpec, learnable_mu: bool = True, learnable_r: bool = False):
        ...
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (..., 1) in support domain -> (..., n_basis)"""
```

- Implement B-spline and RBF first. Bézier and Hermite reduce to special cases of B-spline with appropriate knot vectors — implement via that reduction, not as separate code.
- Wavelet (Daubechies-4 minimum) last; needed for representation completeness, not for the first paper.
- `mu` parameterisation must preserve partition-of-unity when the spec demands it. Use a softmax over raw parameters, not unconstrained weights.

### 4.2 SignedKAN activation (`layers/signed_kan.py`)

```python
class SignedKAN(nn.Module):
    """
    Per-edge learned activation over GGK basis.
    One basis per edge OR shared basis with per-edge mixing coefficients —
    selected by `mode`.
    """
    def __init__(
        self,
        n_edges: int,
        ggk_spec: GGKSpec,
        mode: Literal["per_edge", "shared"] = "shared",
    ): ...
    def forward(self, edge_features: torch.Tensor) -> torch.Tensor: ...
```

Name check: the module is `SignedKAN`, never `HyperKAN`. Fang et al. 2025 holds that name.

### 4.3 HypergraphConv (`layers/hypergraph_conv.py`)

```python
class HypergraphConv(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        ggk_spec: GGKSpec,
        dropout: float = 0.0,
    ): ...
    def forward(
        self,
        x: torch.Tensor,              # (N, d_in)
        B: torch.Tensor,              # sparse (N, E), values = signs
    ) -> torch.Tensor:                # (N, d_out)
```

Forward:

1. Apply signs node→edge: `signed_x = sparse_signed_matmul(B.T, x)` via `ops/sparse_signed.py`.
2. Linear projection: `edge_pre = signed_x @ W` where `W` is `(d_in, d_out)`.
3. GGK-parameterised activation: `edge_post = SignedKAN(edge_pre)`.
4. Aggregate edge→node: `y = sparse_signed_matmul(B, edge_post)`.
5. Normalise by degree (node-wise); dropout; return.

### 4.4 Sparse signed ops (`ops/sparse_signed.py`)

- `sparse_signed_matmul(B_sparse, x)` — respects `values` as signs.
- Autograd: custom `torch.autograd.Function` so gradients flow to dense `x` but *not* to the sign values (signs are discrete).
- If learned construction is used, signs are continuous relaxations in `[-1, 1]` and autograd flows through them normally. Detect this via `values.dtype` and `values.requires_grad`.

---

## 5. Stage 3b — HyMeKo codegen backend

### 5.1 `.hmk` extensions

Add a `torch_network` node type to the HyMeKo grammar:

```hmk
torch_network MyModel {
    nodes: V
    hyperedges: E with signed_incidence
    layers: [
        HypergraphConv { d_in: 64, d_out: 128, ggk: bspline(degree=3, n_knots=8) },
        HypergraphConv { d_in: 128, d_out: 64, ggk: rbf(n_centres=16) },
    ]
    readout: mean_pool
}
```

Grammar additions go through the existing LALR(1) path. The `using ... as` alias feature should be reusable for importing GGK specs.

### 5.2 Codegen (`hymeko_torch/src/codegen.rs`)

Emits a single `.py` file per `torch_network` declaration. Structure of the emitted file:

```python
# AUTO-GENERATED by hymeko_torch v{version} from {source}.hmk
# Do not edit. Regenerate via: hymeko compile --backend torch {source}.hmk

from hymeko_torch_runtime import HypergraphConv, SignedKAN, GGKSpec, build_incidence
import torch
import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("B", build_incidence(...))
        self.layer_0 = HypergraphConv(...)
        self.layer_1 = HypergraphConv(...)

    def forward(self, x):
        x = self.layer_0(x, self.B)
        x = self.layer_1(x, self.B)
        return x.mean(dim=0)
```

Rules:

- Emitted file must be human-readable. No one-liners, no minified output. This is a debugging surface.
- No logic in the emitted file beyond composition. All real code lives in `hymeko_torch_runtime`, which imports from `ehk_torch`.
- Deterministic output: same input `.hmk` → byte-identical `.py`. Required for caching and for diff review.

### 5.3 Factory (`hymeko_torch_runtime/factory.py`)

```python
def from_hmk(path: Path, recompile: bool = False) -> nn.Module:
    """
    Invokes the hymeko compiler (via PyO3 binding) to emit a .py next to
    the .hmk, imports it, returns an instantiated nn.Module.
    Caches based on .hmk mtime unless recompile=True.
    """
```

### 5.4 Hot-swap capability

One function — this is the capability that distinguishes the HyMeKo path from everything else:

```python
def reinfer_structure_and_rebuild(
    model: nn.Module,
    x: torch.Tensor,
    hmk_path: Path,
) -> nn.Module:
    """
    1. Compute entropy metrics on current activations.
    2. Run HyMeKo's query-driven rewriting with new structure.
    3. Emit new .hmk, recompile, load new model.
    4. Transfer compatible weights from old model.
    Returns new model. Old model remains valid and can be discarded.
    """
```

Weight transfer is compatible-subset only: layers whose shapes match carry over, others initialise fresh. Document this clearly; users will be surprised otherwise.

---

## 6. Stage 4 — Training loop

Provide a reference `train.py` but do not over-engineer — users have their own trainers.

```python
def compute_loss(
    y_hat: torch.Tensor,
    y: torch.Tensor,
    B: torch.Tensor,
    sigma: torch.Tensor,
    task_loss_fn: Callable,
    lambda_struct: float = 0.0,
    lambda_mi: float = 0.0,
    struct_variant: str = "v1",
) -> dict[str, torch.Tensor]:
    """Returns dict with 'total', 'task', 'struct', 'mi' for logging."""
```

Checkpointing via `io/checkpoint.py`:

```python
def save_checkpoint(
    path: Path,
    model: nn.Module,
    B: torch.Tensor,
    sigma: torch.Tensor,
    optimizer_state: dict,
    epoch: int,
    metrics: dict,
) -> None: ...

def load_checkpoint(path: Path) -> CheckpointBundle: ...
```

Incidence and sign tensors are saved alongside `state_dict`. A model file without its hypergraph is useless — prevent accidental partial saves.

---

## 7. Integration example

Pick **one** of:

- **Cattle corridor camera placement** (existing Python baseline) — compare against the classical formulation Csaba already has. This is the most realistic first test because the data and baseline exist.
- **ShapeNet part segmentation** — public, standard, reviewer-familiar. Use this if a public benchmark is needed for the first paper.

Do not attempt both in v1.

Deliverable: one Jupyter notebook `examples/cattle_corridor.ipynb` or `examples/shapenet_seg.ipynb` that:

1. Loads data.
2. Builds `B` via MI-threshold + structural-entropy refinement.
3. Trains a 2–3 layer `HypergraphConv` model with `SignedKAN` activations.
4. Reports accuracy and structural-entropy curves.
5. Repeats with the HyMeKo-generated model from an equivalent `.hmk` spec, demonstrates bit-identical behaviour.

---

## 8. Testing

`pytest` with the following categories:

- **Unit**: each entropy estimator against known closed-form values (uniform → `log N`, Gaussian → `0.5 log(2πe σ²)`, etc.).
- **Shape**: every layer's forward pass with symbolic shapes via `torch.fx` or manual assertions.
- **Gradient**: `torch.autograd.gradcheck` on every differentiable path. Allow tolerance `1e-4` for KDE-based estimators, `1e-6` elsewhere.
- **Codegen determinism**: compile the same `.hmk` twice, assert byte-identical output.
- **Parity**: `.hmk`-generated model and hand-written equivalent produce identical forward outputs given identical weights. Seed-controlled.
- **Numerical stability**: structural entropy on edge cases (empty hypergraph, single hyperedge, disconnected components) must not return NaN or Inf.

Minimum coverage target: 85% line coverage on `ehk_torch`. Not a hard requirement on `hymeko_torch` yet — codegen coverage is better measured by integration tests.

---

## 9. Documentation

- Every public class has an example in its docstring.
- `docs/architecture.md` — the four-stage pipeline (entropy → construction → layer → training) with one figure.
- `docs/ggk.md` — a concise statement of K = (B, G, μ, r) and axioms K1–K4, with the mapping to B-spline, Bézier, RBF. This doubles as the paper's background section.
- `docs/hymeko_codegen.md` — worked example of `.hmk` → `.py` with both files shown side by side.

Do not cite HyperKAN anywhere. Use SignedKAN or IncidenceKAN exclusively.

---

## 10. Build order

Priority ordering — do not parallelise prematurely:

1. `ehk_torch.entropy` (all three modules). Tested against closed-form.
2. `ehk_torch.kernels.ggk` with B-spline + RBF only.
3. `ehk_torch.ops.sparse_signed` and `ehk_torch.layers.hypergraph_conv` + `signed_kan`.
4. `ehk_torch.construction.mi_threshold` and `structural_min`.
5. Integration notebook on cattle corridor *or* ShapeNet — whichever is faster. Stop here and verify the native path works end-to-end before starting codegen.
6. `hymeko_torch_runtime.factory` — pure Python, can be stubbed before the Rust crate exists.
7. `hymeko_torch` crate — codegen proper. Start with a single layer type, expand.
8. `reinfer_structure_and_rebuild` — hot-swap. Last. This is the paper demo, not the MVP.

Stages 1–5 constitute v0.1 and are the honest scope for the first publishable result. Stages 6–8 are v0.2.

---

## 11. Out of scope for v0.1

- CUDA custom kernels (Python PyTorch ops only).
- Multi-GPU / distributed training.
- FPGA / HSMM integration.
- C# bindings.
- Quantisation or export to ONNX.
- Any cattle-automation (robovaxx) integration — proprietary; do not include in public code or demos.

---

## 12. Acceptance criteria

v0.1 is done when:

1. All tests pass, coverage ≥ 85% on `ehk_torch`.
2. The integration notebook runs end-to-end in under 30 minutes on a single consumer GPU.
3. Structural entropy curves are logged and monotonically non-increasing when structural-entropy minimisation is used as the construction strategy (sanity check — if this is violated, the estimator or the optimiser is wrong).
4. The HyMeKo-generated path produces forward outputs bit-identical to a hand-written equivalent with matched seeds.

Attach the notebook output, test report, and a short `RESULTS.md` to the v0.1 tag.
