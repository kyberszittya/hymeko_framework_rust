# CogInfoCom paper figures

Figures rendered for the CogInfoCom article on the learnable
Catmull-Rom fuzzy-signature framework.

## Legend description (read this first)

Every panel in `cr_per_layer_grid.png` and `cr_init_vs_trained.png`
plots the same five quantities, on the same x-axis ($x \in [0, 1]$
= the fuzzy input domain), with the same colours. Each item is a
named primitive of classical fuzzy systems realised as a directly
trainable component:

| Curve / Colour | Symbol | What it is | Fuzzy-systems reference |
|---|---|---|---|
| **Solid red** | $\mu^+(x)$ | The **Atanassov *membership* function** for this (layer, channel) cell. Output of $\sigma \circ \mathrm{CR}^+(x)$: a Catmull-Rom spline with 8 learnable control points, post-composed with $\sigma$ so the output stays in $[0,1]$. **Interpretation**: "to what degree does the input $x$ belong to this channel's concept?" | Atanassov (1986), §III of the math background |
| **Solid blue** | $\mu^-(x)$ | The **Atanassov *non-membership* function**, learned as a second, independent CR spline. Same shape family as $\mu^+$ but with separate control points. **Interpretation**: "to what degree does $x$ *not* belong to the concept?" Under the asymmetric-ramp init, $\mu^-$ is a Gaussian-bump-like decreasing shape, distinct from a literal complement of $\mu^+$. | Atanassov (1986); see also background §3 |
| **Dashed green** | $g(x)$ | The **Zadeh sigmoidal hedge gate**: $g(x) = \sigma(\tau \cdot (x - c))$. The parameters $\tau$ (per-channel learnable steepness) and $c$ (per-channel learnable centre) decide where $g$ crosses $\tfrac12$ and how sharply. **Interpretation**: a per-channel decision boundary in fuzzy-input space, controlling which of $\mu^+$ or $\mu^-$ dominates at each $x$. $\tau \to \infty$ approximates a crisp Heaviside; $\tau \to 0$ degenerates to $g \equiv \tfrac12$ (no preference). | Zadeh (1972) linguistic hedges; background §6 |
| **Solid purple** | $\mu(x)$ | The **IFS mix** $\mu(x) = g(x)\,\mu^+(x) + (1 - g(x))\,\mu^-(x)$. This is the *layer's actual fuzzy output* for that channel and input value: a convex combination of the membership and non-membership functions, governed by the gate. Stays in $[0,1]$ by construction. | Background §3, §7 |
| **Gray fill** (under $1 - \mu$) | $\pi(x)$ | The **hesitancy** (or "undecidedness") at input $x$. Two definitions are common: the *mix-based* hesitancy $\pi = 1 - \mu(x)$ (plotted here, gray fill) and the *strict-IFS* hesitancy $\pi = 1 - \mu^+(x) - \mu^-(x)$ (where $\mu^+ + \mu^- \le 1$ by definition). The plotted version is the simpler, layer-output-based quantity. **Interpretation**: at inputs where $\mu(x)$ is far from 1, the channel is hesitant about declaring "belongs"; the more area filled, the more hesitant the channel is in expectation. | Atanassov (1986) hesitancy index; background §3.1 |

**Per-cell label**: each panel's title shows the channel index; the
left-column $y$-label shows the layer index and the channel-0 value
of $\tau$. The hesitancy fill is rendered at low opacity so it does
not obscure the curves.

**Reading the figure top-to-bottom**: rows correspond to deeper
layers (layer 0 sees the fuzzification output; layer L−1 feeds the
defuzzification head). The IFS mix's shape can specialise per layer:
early layers tend to acquire steeper, edge-like membership shapes;
late layers tend to acquire smoother, decision-like shapes.

**Reading the figure left-to-right**: columns correspond to
different channels of the same layer. Different channels learn
*different* membership functions for the same fuzzy input — this is
the framework's analogue of a CNN's multi-channel feature
detectors, except that here each channel is an explicit, plottable
membership function rather than an opaque filter.

## Vanilla CR figures (no fuzzy interpretation)

Companion figures rendered from a **vanilla HSiKAN** model
(`HSiKANPoseModel`), showing the *same* learnable Catmull-Rom
splines **without** the Atanassov / Zadeh / hesitancy interpretive
overlay. The same architecture admits both readings; this pair of
figures lets a reviewer compare the two side-by-side and decide
whether the fuzzy interpretation adds something or just relabels
generic neural primitives.

| File | Content |
|---|---|
| `cr_vanilla_per_layer_grid.png` | 4-layer × 4-channel grid of the raw two-branch CR splines from `HSiKANVisionLayer.convs[0].activation`. No σ-clamp, no gate, no mix, no hesitancy — just the unbounded CR output on its native [-3, 3] domain. Control points overlaid as dots. |
| `cr_vanilla_init_vs_trained.png` | Before/after-training comparison for layer 0 channel 0. |
| `cr_vanilla_data.json` | Numerical dump. |

### Legend description (vanilla version)

Every panel in the vanilla figure plots two curves, on the same
x-axis ($x \in [-3, 3]$ = CR's native input domain), with two
optional sets of control-point markers:

| Curve / Colour | Symbol | What it is | HSiKAN reference |
|---|---|---|---|
| **Solid red** | CR$^+(x)$ | The **above-mean signed branch** of the Catmull-Rom activation. This is `CRActivation.forward(x, branch_idx=0)` — a cubic Catmull-Rom interpolation through $m=8$ learnable control points on the uniform grid $\{-3, -3+\frac{6}{7}, \ldots, +3\}$. Unbounded (no σ-clamp). **Interpretation in vanilla HSiKAN**: the activation applied when the pixel's polarity is positive (above the patch mean). | `SignedBranchConv.activation`, `branch_idx=0`; the "above-mean" branch of HSiKAN-vision. |
| **Solid blue** | CR$^-(x)$ | The **below-mean signed branch**: `CRActivation.forward(x, branch_idx=1)`. Separate set of $m=8$ control points; same spline basis. Unbounded. **Interpretation in vanilla HSiKAN**: the activation when the pixel's polarity is negative (below the patch mean). | `SignedBranchConv.activation`, `branch_idx=1`. |
| **Red dots** | $(x_i, p^+_i)$ | The $m=8$ **control points** of the above-mean spline. The Catmull-Rom interpolant passes through (or near) these points; they are exactly the trainable parameters. Plotted at the same x-grid as the spline domain. | `CRActivation.cpts[0, c, :]` and `CRActivation.x_grid`. |
| **Blue squares** | $(x_i, p^-_i)$ | The $m=8$ control points of the below-mean spline. | `CRActivation.cpts[1, c, :]`. |
| **Gray dotted** (horizontal) | $y = 0$ | Reference line. The CR output is signed (positive or negative), which under HSiKAN's downstream sum-aggregation gets combined with the per-edge weight $W_e$ and the spatial filter $W_{\mathrm{pos}}$. | --- |

The vanilla figure communicates: *"this is what a 2-branch
Catmull-Rom activation looks like under the standard HSiKAN
reading — just two learnable splines per channel, no claim of
fuzzy semantics, output unbounded."*

### How the two readings differ

| Aspect | Fuzzy reading (`cr_per_layer_grid.png`) | Vanilla reading (`cr_vanilla_per_layer_grid.png`) |
|---|---|---|
| Domain shown | $x \in [0, 1]$ (fuzzy input space) | $x \in [-3, 3]$ (CR native grid) |
| Output range | $[0, 1]$ (σ-clamped) | unbounded |
| Two branches mean | μ⁺ (membership) and μ⁻ (non-membership) of an Atanassov IFS | above-mean / below-mean signed activations |
| Gate present | yes ($g(x) = \sigma(\tau(x - c))$), Zadeh hedge | no |
| Mix curve | yes ($\mu(x) = g \mu^+ + (1{-}g)\mu^-$) | no |
| Hesitancy | yes ($\pi = 1 - \mu$) | no |
| Number of curves/panel | 4 + filled region | 2 + control-point markers |
| Same architecture | yes (same model family) | yes (same model family) |
| Same learnable parameters | yes (m=8 CP per channel per branch + τ, c, $W_e$) | yes (the same parameters, viewed differently) |

The architecture is identical; the interpretation is the choice
the framework makes. The vanilla figure is what a reviewer skeptical
of the fuzzy reading would draw; the fuzzy figure is what the
framework's authors claim is the right reading. The paper can show
both and argue *why* the fuzzy reading is the more useful one
(uncertainty calibration via hesitancy, named primitives, alignment
with Kóczy fuzzy signatures).



## Files

| File | Content | Provenance |
|---|---|---|
| `cr_per_layer_grid.png` | 4-layer × 4-channel grid of learned Atanassov μ⁺/μ⁻ pairs + Zadeh hedge gate g(x) + IFS mix μ(x) + hesitancy π fill. The main "interpretability evidence" figure. | Generated 2026-06-01 from a FuzzySignaturePoseModel trained 15 epochs on 500-sample SyntheticPoseDataset (loss 32 → 20). CPU run, ~5 min. |
| `cr_init_vs_trained.png` | Before/after-training comparison for layer 0 channel 0. Shows what gradient descent moved. | Same training run as above. |
| `cr_data.json` | Numerical dump of x_fuzzy, μ⁺, μ⁻, gate, mix, τ, c per (layer, channel). | Same. |

## How to regenerate

```bash
cd /home/kyberszittya/hakiko-ws/hymeko/hymeko_framework_rust
CUDA_VISIBLE_DEVICES=""  PYTHONPATH=. python \
    hymeko_neuro/examples/cr_spline_render.py \
    --n-epochs 15
# Outputs to /tmp/cr_spline_render/ — copy to docs/coginfocom_paper_figures/
```

To get **higher-quality, fully-converged** figures (use these for the
final paper submission, not the 15-epoch draft):

```bash
# After the overnight smoke chain frees the GPU:
PYTHONPATH=. python hymeko_neuro/examples/cr_spline_render.py \
    --n-epochs 60 \
    --channels 0 1 2 3 4 5 \
    --n-layers-show 8
# ~3 minutes on GPU, ~30 minutes on CPU
```

## Caption suggestions for the article

### Figure (cr_per_layer_grid.png)

> Learned Catmull-Rom splines per (layer, channel) in
> FuzzySignaturePoseModel after partial training. Each panel shows
> the Atanassov membership pair $(\mu^+, \mu^-)$ (red, blue), the
> learnable Zadeh hedge gate $g(x) = \sigma(\tau(x - c))$ (green
> dashed), the IFS mix $\mu(x) = g(x)\mu^+(x) + (1-g(x))\mu^-(x)$
> (purple), and the hesitancy $\pi(x) = 1 - \mu(x)$ (gray fill)
> for one (layer, channel) cell of an 8-layer multi-arity stack at
> $d=16$. The horizontal axis is the fuzzy input $x \in [0,1]$;
> the vertical axis is the membership value. The learnable
> dilation $\tau$ (gate steepness) and centre $c$ are reported per
> channel. Different (layer, channel) cells specialise to
> different membership shapes via gradient descent; every plotted
> curve is a primitive of classical fuzzy systems (Catmull-Rom
> membership function, Atanassov IFS pair, Zadeh hedge, hesitancy
> index) realised as a directly trainable component.

### Figure (cr_init_vs_trained.png)

> Effect of training on a single learnable membership function
> (layer 0, channel 0). Left: at initialisation under the ramp
> scheme, $\mu^+$ is a monotone increasing ramp $\sim[0.5, 0.82]$,
> $\mu^-$ is a symmetric decreasing ramp, and $g(x)$ is the
> default Zadeh hedge with $\tau = 4$, $c = 0.5$. Right: after
> 15 epochs on the synthetic pose task, the spline control points
> have moved off the ramp init, $\tau$ has begun to drift, and the
> hesitancy $\pi$ profile has acquired task-specific structure.
> The framework's claim that "every learnable scalar is a named
> fuzzy-system primitive" is operationalised by this figure:
> gradient descent is moving the same parameters that classical
> fuzzy designers would set by hand.

## Related artifacts

- Code: [`hymeko_neuro/examples/cr_spline_render.py`](../../hymeko_neuro/examples/cr_spline_render.py)
- Mathematical background: [`docs/plans/2026-05-30-fuzzy-signature-layer/background.tex`](../plans/2026-05-30-fuzzy-signature-layer/background.tex)
- Pose model definition: [`hymeko_neuro/experiments/vision/fuzzy_pose.py`](../../hymeko_neuro/experiments/vision/fuzzy_pose.py)
- Pose detection example app: [`hymeko_neuro/examples/pose_demo.py`](../../hymeko_neuro/examples/pose_demo.py)
