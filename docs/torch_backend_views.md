# Torch backend — hierarchical hypergraph in existing HyMeKo idiom

**Replaces §5.1 of `input/entropy_hypergraph_pytorch_spec.md`.**

The original spec proposed a new `torch_network` top-level decl with surface syntax (`layers: [ ... ]`, anonymous nested records, `with signed_incidence`, function-call kwargs) that would require LALR grammar changes. This rewrite expresses the same semantics in **existing HyMeKo grammar**, with no parser modifications, by formulating a neural network as a **hierarchical hypergraph** in which **each layer is a hypervertex** containing its own sub-hypergraph of neurons and factors. Two natural projections of this hierarchy — the **factor view** (project to leaves) and the **dataflow view** (project to top level) — are both expressible as signed-incidence hypergraphs and both compile through the existing HyMeKo pipeline + template dispatcher.

---

## The hierarchical hypergraph

A neural network is a single canonical IR with a two-level structure:

- **Top level (dataflow):** vertices are *tensors* and *layer hypervertices*; hyperedges are *dataflow triples* `(input-tensor, layer-hypervertex, output-tensor)` connecting them.
- **Inside each layer hypervertex (factor):** vertices are *neurons* (or activation units, ports, parameter cells); hyperedges are *factors* aggregating over subsets of those neurons. Each layer's inner sub-hypergraph is private — outer dataflow hyperedges reference the layer through its *ports*, not through its internal neurons.

In set-theoretic terms: the IR is a hypergraph `H = (V, E, I, σ)` where some `v ∈ V` are themselves hypergraphs `H_v = (V_v, E_v, I_v, σ_v)` with their own ports `P_v ⊂ V_v` exposed to the outer level. Standard hierarchical-hypergraph machinery (Bourbaki-style, also called "nested hypergraphs" or "hypergraphs over hypergraphs"); HyMeKo's existing nested-decl IR (`NodeInner.body: Option<Vec<HyperItem>>`) already realises it concretely — every node decl with a body *is* a hypervertex containing a sub-hypergraph.

This formulation gives one source description from which both granularities are recoverable as projections:

- `π_dataflow(H)` — strip every hypervertex's interior, retaining only the top-level structure. This is the **dataflow view**.
- `π_factor(H)` — flatten every hypervertex into its interior, with port-incidences resolved against outer-scope hyperedges. This is the **factor view**.

The two projections commute with the emitter (a candidate **Proposition 5** for the journal version): emitting from the projected IR equals projecting then emitting, byte-for-byte.

---

## Why two views (as projections of the same hierarchy)

A neural network can be analysed and emitted at two natural granularities, and engineers reach for different ones for different tasks:

- **Factor view** (`π_factor(H)` projection) — vertices are *neurons / activation units*; hyperedges are *factors* aggregating over subsets of neurons. This is the Forney-style factor graph representation. It makes fine-grained analysis natural (per-neuron entropy, structural entropy of the factor graph itself, edge-wise receptive fields) but bloats fast.
- **Dataflow view** (`π_dataflow(H)` projection) — vertices are *layer hypervertices* and *tensors*; hyperedges are *dataflow triples* `(input-tensor, layer-hypervertex, output-tensor)`. This is the computational-graph / ONNX representation. It makes coarse-grained emission natural (what the PyTorch code actually looks like) and fits typical ML reader mental models.

Both projections produce well-formed signed-incidence hypergraphs; both compile through the existing HyMeKo pipeline; the codegen lets any of the six existing emitters (URDF/SDF/MJCF/DOT/Mermaid/Gazebo) plus two new PyTorch ones consume either. **Proposition 1 (alias invariance)** applies: two surface variants denoting the same hierarchical network must emit byte-equal PyTorch in either view. **Proposition 2 (content-addressability)** extends naturally: the canonical hash is computed over the hierarchical IR and is invariant under rewrites that preserve the hierarchy. **Proposition 5 (projection commutativity, candidate for journal version)** asserts the new claim that view projection commutes with emission.

---

## Factor view

### Intent

Vertices are neurons (or groups of neurons, at any granularity the user chooses). Hyperedges are factors — each one aggregates some set of source neurons and contributes to some set of sink neurons, with `+` marking source participation and `-` marking sink participation. A dense layer is one hyperedge per output neuron; a convolutional layer is one hyperedge per output position.

The factor view is normally **derived as the projection `π_factor(H)` of the hierarchical source** (see §"Dataflow view" below for the hierarchical authoring surface). The example below is what the projection produces for the simple-network case; it is also a valid authoring surface in its own right for users who want flat factor-graph descriptions without layer scoping (e.g. for entropy experiments where layer boundaries are irrelevant).

### Example

Simple 2-layer network with 3 input neurons, 5 hidden neurons, 2 output neurons:

```hymeko
simple_net_description {
    @"meta_nn.hymeko";
    using nn.neurons as neu;
    using nn.factors as fac;
    using nn.ggk as ggk;
}

simple_net: neu, fac, ggk {

    // Shared GGK specs referenced by multiple factors.
    shared_bspline: + <isa> ggk.bspline {
        degree 3;
        n_knots 8;
    }

    // Input neurons.
    x0: neu.input { dim 1; }
    x1: neu.input { dim 1; }
    x2: neu.input { dim 1; }

    // Hidden neurons with their activation specification.
    h0: neu.hidden { activation "relu"; }
    h1: neu.hidden { activation "relu"; }
    h2: neu.hidden { activation "relu"; }
    h3: neu.hidden { activation "relu"; }
    h4: neu.hidden { activation "relu"; }

    // Output neurons.
    y0: neu.output { dim 1; }
    y1: neu.output { dim 1; }

    // Hidden-layer factors: each hidden neuron receives from all inputs.
    // The `+` incidences mark source neurons; `-` marks the sink.
    @fac_h0: fac.hypergraph_conv {
        ggk -> shared_bspline;
        (+ x0, + x1, + x2, - h0);
    }
    @fac_h1: fac.hypergraph_conv {
        ggk -> shared_bspline;
        (+ x0, + x1, + x2, - h1);
    }
    @fac_h2: fac.hypergraph_conv {
        ggk -> shared_bspline;
        (+ x0, + x1, + x2, - h2);
    }
    @fac_h3: fac.hypergraph_conv {
        ggk -> shared_bspline;
        (+ x0, + x1, + x2, - h3);
    }
    @fac_h4: fac.hypergraph_conv {
        ggk -> shared_bspline;
        (+ x0, + x1, + x2, - h4);
    }

    // Output-layer factors: each output neuron receives from all hidden.
    @fac_y0: fac.hypergraph_conv {
        ggk -> shared_bspline;
        (+ h0, + h1, + h2, + h3, + h4, - y0);
    }
    @fac_y1: fac.hypergraph_conv {
        ggk -> shared_bspline;
        (+ h0, + h1, + h2, + h3, + h4, - y1);
    }
}
```

### What's going on here syntactically

Everything in this source is existing HyMeKo:

- `using nn.neurons as neu;` — the aliasing mechanism already landed (see `data/robotics/anthropomorphic_arm_using.hymeko`).
- `neu.input { ... }`, `neu.hidden { ... }`, `neu.output { ... }` — standard node declarations, typed against the imported `meta_nn` namespace.
- `+ <isa> ggk.bspline { ... }` — existing type-inheritance syntax; `shared_bspline` inherits from `ggk.bspline` and overrides `degree` and `n_knots`.
- `@fac_h0: fac.hypergraph_conv { ... (+ x0, + x1, + x2, - h0); }` — standard hyperedge decl with signed incidences.
- `ggk -> shared_bspline;` — existing ref arrow, pointing at the shared GGK spec decl.

No grammar modifications. The `meta_nn.hymeko` library would be new — it defines the `nn.neurons`, `nn.factors`, `nn.ggk` namespaces analogously to `meta_kinematics.hymeko` — but that's a fixture-level addition, not a parser-level one.

### Semantics

- Each hyperedge is a **factor** in the factor-graph sense: it aggregates signal from `+` neurons, applies a transformation (the GGK kernel + learned activation), and injects into `-` neurons.
- The signed-incidence hypergraph formalism of Propositions 1–4 applies directly. Structural entropy over the factor graph is `H_struct(B_factor, σ_factor)` — exactly the expression the original spec's §2.3 targets.
- Repetition is real (five factors for five hidden neurons) but compact: each is a two-line hyperedge decl. Tier C macros (when landed) would subsume this to one `wire_dense_layer` invocation — good incremental value but not required for v0.1.

### PyTorch codegen from factor view

The codegen runs queries over the IR analogous to URDF's: find all neurons, find all factors, group factors by `{sinks, ggk_ref}`, emit PyTorch. Template skeleton:

```python
# AUTO-GENERATED from simple_net.hymeko (factor view)
import torch; import torch.nn as nn
from hymeko_torch_runtime import HypergraphConv, SignedKAN, GGKSpec, build_incidence

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        # B and σ derived from the factor-view hypergraph.
        self.register_buffer("B", build_incidence(
            shape=(10, 7),  # 3+5+2 neurons, 5+2 factors
            nnz=7*(3+5),    # computed from the IR
            pattern=...,    # COO row/col indices from query result
            signs=...,      # +1 / -1 from incidence signs
        ))
        self.ggk = GGKSpec(basis="bspline", degree=3, n_knots=8)
        self.conv = HypergraphConv(d_in=1, d_out=1, ggk_spec=self.ggk)
        # ... additional weights per factor-group ...

    def forward(self, x):  # x: (batch, 3)
        # Pack input into (batch, 10) placing x into neuron slots 0..2.
        h = pad_to_neurons(x, slots=[0, 1, 2], total=10)
        # Apply all factors in one HypergraphConv call over B.
        h = self.conv(h, self.B)
        # Extract output neurons 8, 9.
        return h[:, [8, 9]]
```

This is verbose by design — the factor view is the research surface, not the production surface. For production emission, use the dataflow view.

---

## Dataflow view (layers as hypervertices)

### Intent

Vertices are *tensors* and *layer hypervertices*. Each layer hypervertex has a body — its private sub-hypergraph of neurons and factors — plus *port* decls that expose the layer's interface to the outer dataflow. Hyperedges at this level are *dataflow triples* `(input-tensor, layer-hypervertex, output-tensor?)` connecting layer hypervertices through their tensors. For chained layers, each step is one arity-3 hyperedge; for skip connections, the extra source adds another `+` incidence.

The key contract: **outer dataflow hyperedges reference layer hypervertices by name, not by reaching into their internal neurons.** A layer's interface is its declared `lyr.input_port` and `lyr.output_port` decls. The factor-view projection later resolves port-incidences against the outer-scope hyperedges, so port semantics let the two projections compose cleanly.

### Example

The same 2-layer network, expressed as a single hierarchical source. Each layer hypervertex carries its inner factor sub-hypergraph (ports + neurons + intra-layer factors) inside its body:

```hymeko
simple_net_description {
    @"meta_nn.hymeko";
    using nn.layers as lyr;
    using nn.tensors as ten;
    using nn.ports as port;
    using nn.neurons as neu;
    using nn.factors as fac;
    using nn.ggk as ggk;
}

simple_net: lyr, ten, port, neu, fac, ggk {

    // Module-scope GGK specs (referenced by inner factors of multiple
    // layers; canonical-IR hashing dedupes inherited specs).
    layer_0_ggk: + <isa> ggk.bspline { degree 3; n_knots 8; }
    layer_1_ggk: + <isa> ggk.rbf     { n_centres 16; }

    // Top-level tensor vertices — the data carried between layers.
    x: ten.input      { shape [3]; }
    h: ten.activation { shape [5]; }
    y: ten.output     { shape [2]; }

    // ── Layer 0 hypervertex ──────────────────────────────────────────
    layer_0: lyr.hypergraph_conv {
        d_in 3;
        d_out 5;
        ggk -> layer_0_ggk;

        // Input ports — exposed to the outer dataflow.
        in0: port.input {}
        in1: port.input {}
        in2: port.input {}

        // Internal neurons.
        h0: neu.hidden { activation "relu"; }
        h1: neu.hidden { activation "relu"; }
        h2: neu.hidden { activation "relu"; }
        h3: neu.hidden { activation "relu"; }
        h4: neu.hidden { activation "relu"; }

        // Output ports — exposed to the outer dataflow.
        out0: port.output {}
        out1: port.output {}
        out2: port.output {}
        out3: port.output {}
        out4: port.output {}

        // Intra-layer factors: each output neuron aggregates all inputs
        // and writes to its corresponding output port.
        @factor_h0: fac.hypergraph_conv { (+ in0, + in1, + in2, - h0, ~ out0); }
        @factor_h1: fac.hypergraph_conv { (+ in0, + in1, + in2, - h1, ~ out1); }
        @factor_h2: fac.hypergraph_conv { (+ in0, + in1, + in2, - h2, ~ out2); }
        @factor_h3: fac.hypergraph_conv { (+ in0, + in1, + in2, - h3, ~ out3); }
        @factor_h4: fac.hypergraph_conv { (+ in0, + in1, + in2, - h4, ~ out4); }
    }

    // ── Layer 1 hypervertex (similar structure, narrower) ────────────
    layer_1: lyr.hypergraph_conv {
        d_in 5;
        d_out 2;
        ggk -> layer_1_ggk;

        in0: port.input {}
        in1: port.input {}
        in2: port.input {}
        in3: port.input {}
        in4: port.input {}

        n0: neu.output {}
        n1: neu.output {}

        out0: port.output {}
        out1: port.output {}

        @factor_y0: fac.hypergraph_conv { (+ in0, + in1, + in2, + in3, + in4, - n0, ~ out0); }
        @factor_y1: fac.hypergraph_conv { (+ in0, + in1, + in2, + in3, + in4, - n1, ~ out1); }
    }

    // ── Readout (degenerate hypervertex — pure compute, no neurons) ──
    readout: lyr.mean_pool {
        in0: port.input {}
        in1: port.input {}
    }

    // ── Outer dataflow hyperedges ────────────────────────────────────
    // Connect tensors through layer hypervertices via their ports.
    // `+` reads from, `-` writes to, `~` marks the layer hypervertex
    // (neutral-role; participates in the edge but is itself neither
    // source nor sink of the data — the data flows through its
    // ports, not through the hypervertex itself).
    @flow_0: lyr.dataflow {
        (+ x, ~ layer_0, - h);
    }
    @flow_1: lyr.dataflow {
        (+ h, ~ layer_1, - y);
    }
    @flow_readout: lyr.dataflow {
        (+ y, ~ readout);
    }
}
```

### What's going on here syntactically

Same story as before: every construct is existing HyMeKo.

- Each `layer_0: lyr.hypergraph_conv { ... }` is a node decl with a body — a *hypervertex* in the hierarchical-hypergraph sense. HyMeKo's `NodeInner.body: Option<Vec<HyperItem>>` field already holds the inner sub-hypergraph.
- Port decls (`in0: port.input {}`, `out0: port.output {}`) are ordinary nested decls with a port-typed type. The `meta_nn.hymeko` library defines `port.input` and `port.output` as standard node types.
- Inner factors (`@factor_h0: fac.hypergraph_conv { (+ in0, + in1, + in2, - h0, ~ out0); }`) are ordinary hyperedge decls. Their incidences reference port and neuron decls in the *enclosing layer's* scope — HyMeKo's resolver already walks lexical scope through nested-decl bodies.
- Outer dataflow hyperedges (`@flow_0: lyr.dataflow { (+ x, ~ layer_0, - h); }`) reference the *whole layer hypervertex* by name. The neutral-role `~` incidence already exists in the grammar (`SignedRef::Neutral`, `Sign::Neutral`); dataflow gives it the first-class role of "the hypervertex through which the data passes."

The neutral-role `~` is the lynchpin. It distinguishes "is this a `+` data source" from "is this the compute hypervertex itself" — three-role signed incidence (`+ / − / ~`) is precisely what hierarchical dataflow needs and is what HyMeKo's IR already provides.

### Semantics

- Each layer hypervertex's inner factors operate over its own port and neuron decls — strictly local to the layer's scope.
- Each outer dataflow hyperedge `(+ in_tensor, ~ layer, − out_tensor)` says: data flows from `in_tensor` through `layer` (a hypervertex) into `out_tensor`. Skip connections: extra `+` incidences. Multi-output layers: extra `−` incidences (one per output tensor).
- The hierarchical hypergraph is the canonical form. The two views are projections:
    - `π_dataflow(simple_net)` strips the layer-hypervertex bodies, leaving 3 tensor vertices + 3 hypervertex shells + 3 dataflow hyperedges.
    - `π_factor(simple_net)` flattens each layer's body into the parent scope, resolving port-incidences against the outer dataflow hyperedges. The port `layer_0.in0` becomes the tensor `x` (resolved through `flow_0`'s `+ x` source); the port `layer_0.out0` becomes the tensor `h` (through `flow_0`'s `− h` sink). Result: 10 neuron vertices + 7 factor hyperedges.
- The two projections are pure IR-level rewrites — no surface re-authoring. **Proposition 5 (candidate)**: emitting from `π_view(H)` equals projecting then emitting. The byte-equal-output property of the existing six emitters lifts to PyTorch under either projection.

### PyTorch codegen from dataflow view

Runs queries to enumerate layers in topological order, emit each as an `nn.Module` attribute, and compose them in `forward`:

```python
# AUTO-GENERATED from simple_net.hymeko (dataflow view)
import torch; import torch.nn as nn
from hymeko_torch_runtime import HypergraphConv, GGKSpec

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_0 = HypergraphConv(
            d_in=3, d_out=5,
            ggk_spec=GGKSpec(basis="bspline", degree=3, n_knots=8),
        )
        self.layer_1 = HypergraphConv(
            d_in=5, d_out=2,
            ggk_spec=GGKSpec(basis="rbf", n_centres=16),
        )

    def forward(self, x):              # x: (batch, 3)
        h = self.layer_0(x)            # (batch, 5)
        y = self.layer_1(h)            # (batch, 2)
        return y.mean(dim=0)           # readout
```

Direct, idiomatic PyTorch. The dataflow view is what a user reads when debugging.

---

## One source, three IRs

The hierarchical source above compiles through the same `ModuleStore::compile` pipeline as every other HyMeKo description. The result is **one canonical hierarchical IR** from which three things are recoverable:

- The **hierarchical IR itself** — used directly for entropy analysis that needs to span both layer-level and neuron-level structure (e.g. the hot-swap path in §5.4 of the original spec, where structural-entropy on the *whole* hierarchy guides architectural rewrites).
- **`π_dataflow(H)`** — the dataflow projection. The layer-hypervertex bodies are stripped; only port shells remain (or the bodies are erased entirely if the emitter doesn't need port-level structure). Result: ≈ 3 tensor decls + 3 layer hypervertex shells + 3 dataflow hyperedges. This is what the dataflow PyTorch template consumes.
- **`π_factor(H)`** — the factor projection. Each layer hypervertex is flattened into its parent scope; port incidences are resolved against the outer dataflow hyperedges. Result: ≈ 10 neuron decls + 7 factor hyperedges, all flat. This is what the factor PyTorch template consumes (and what `H_struct(B_factor, σ_factor)` is computed over).

Both projections are pure IR-level rewrites that any HyMeKo `transforms/<name>/queries.hymeko` query bundle can express. **Proposition 1 (alias invariance)** guarantees that surface variants denoting the same hierarchical IR produce byte-equal projections (and therefore byte-equal PyTorch). **Proposition 5 (candidate)** says the projections themselves commute with emission: `ε_torch(π_view(H)) = ε_torch_view(H)` byte-for-byte. The factor view and the dataflow view are not two independent codegen paths — they are two emitters reading two projections of the same canonical IR.

This is the cleanest possible Proposition-1 witness on a new target: two views of one PyTorch model, same source, byte-equal emission.

### Why hierarchy buys more than just structure

Three things follow from the hierarchical formulation that flat dual-view authoring would not:

1. **Encapsulation.** A layer's internal neurons and factors are accessible only through its ports from outside. Changing a layer's internal factor structure (e.g., entropy-guided rewriting of the inner sub-hypergraph) doesn't perturb the outer dataflow. The hot-swap path in §5.4 can rewrite layers individually while preserving the dataflow hyperedges around them.
2. **Compositional entropy.** Structural entropy at each level of the hierarchy is independently meaningful: `H_struct(outer)` measures dataflow complexity; `H_struct(layer_i.body)` measures per-layer factor-graph complexity. Csaba's ten-metric specification can be evaluated at any level without conflating the granularities.
3. **Re-use of layer hypervertices.** A layer hypervertex with declared ports is a *typed module*: its name + port arity + port types form an interface. Multiple top-level networks can `+ <isa>` the same layer-hypervertex declaration to reuse it (sharing weights or just structure). The canonical-IR hash makes this content-addressable: identical layer bodies hash identically, and the codegen can emit them once and reference them many times.

---

## What stays unchanged from the original spec

Everything in §2 (entropy stage: Shannon, MI, structural), §4 (PyTorch layers: GGK kernel, SignedKAN, HypergraphConv, sparse signed ops), §6 (training loop), §7 (integration example), §8 (testing), §9 (documentation), §10 (build order), §11 (out of scope), §12 (acceptance criteria) is unchanged. This rewrite touches only §5.1 (surface syntax) and §5.2 (codegen) — the parts that required grammar additions.

The `ehk_torch` PyTorch package — the native runtime — is completely unchanged. It consumes `B` and `σ` tensors at runtime; it doesn't care what surface syntax generated them.

---

## What changes from the original spec

### §5.1 — `.hmk` extensions

**Original:** add a `torch_network` keyword + a `layers: [ ... ]` list of anonymous nested records + function-call kwargs + `with` modifier.

**Replacement:** one template library shipped in `data/nn/`:

- `meta_nn.hymeko` — defines `nn.neurons.{input, hidden, output}`, `nn.ports.{input, output}`, `nn.factors.hypergraph_conv`, `nn.layers.{hypergraph_conv, mean_pool, dataflow}`, `nn.tensors.{input, activation, output}`, `nn.ggk.{bspline, rbf, bezier, hermite, wavelet}`. All standard HyMeKo nested decls. The `nn.ports` namespace is what makes layer hypervertices first-class.

That's the entire grammar story. **No LALR changes.** No new file extension (`.hymeko`, not `.hmk`). Hierarchical structure rides on the existing `NodeInner.body: Option<Vec<HyperItem>>` IR field.

### §5.2 — Codegen

**Original:** a new Rust crate `hymeko_torch` emits `.py` files bespoke from the IR.

**Replacement:** three template directories under `transforms/`, all reusing the existing template dispatcher:

- `transforms/torch_dataflow/{queries.hymeko, template.py}` — primary emit path. Consumes the hierarchical IR (or its `π_dataflow` projection); emits direct, idiomatic PyTorch with one `nn.Module` attribute per layer hypervertex and a `forward()` body that follows the dataflow hyperedges.
- `transforms/torch_factor/{queries.hymeko, template.py}` — secondary emit path. Consumes the `π_factor` projection (or a flat factor source); emits the `B` / `σ` tensor representation and a single `HypergraphConv` call. Useful for the entropy-research workflow.
- `transforms/torch_hierarchy/{queries.hymeko, template.py}` (optional v0.2) — emits both at once: a structured PyTorch module where the outer class follows dataflow and each layer hypervertex's body is emitted as a nested submodule with its own factor structure. This is what `reinfer_structure_and_rebuild` consumes.

Three new transforms registered in `hymeko_formats::register_defaults` (or in a sibling `hymeko_formats_torch` crate if the user wants PyTorch out of the core format set). No new dispatch machinery; no new Rust crate required. The `π_dataflow` and `π_factor` projections are themselves expressed as `transforms/torch_*/queries.hymeko` query bundles — they are HyMeKo queries, not Rust passes.

The factory `hymeko_torch_runtime.from_hmk` stays as spec'd — it invokes `hymeko compile --format torch_factor` or `--format torch_dataflow`, imports the emitted `.py`, returns an `nn.Module`.

### §5.4 — Hot-swap capability

**Original:** `reinfer_structure_and_rebuild(model, x, hmk_path)` recomputes entropy, rewrites the `.hmk`, recompiles.

**Replacement:** same capability, now with a choice: the rewrite may target the factor view (for per-neuron entropy-guided edge adjustment) or the dataflow view (for per-layer architectural changes). Both paths go through `ModuleStore::compile` → canonical IR → template dispatch to emitted `.py`. The weight-transfer subset-match logic is unchanged.

---

## Why this is a better shape

Three reasons:

1. **No parser risk, no blocker.** The original spec introduces grammar work that would compete with Tier C macros for the same parts of the LALR. This rewrite avoids both — the PyTorch backend ships independent of any grammar roadmap.
2. **Two views, one IR: a Proposition-1 witness on a new target.** The same underlying network described in the factor view vs. the dataflow view must emit byte-equal PyTorch under the view-transformation query. This is a genuinely new witness for the paper's cross-view-consistency claim: it's not just six robotics formats anymore, it's also two views of a PyTorch model reducing to the same emission.
3. **Reuses the codegen path we just cleaned up.** `hymeko_formats` was extracted from `hymeko_query` specifically to enable this kind of per-target family. PyTorch emission lands as two template directories, not as a new language.

---

## One decision point for Csaba

The spec's §5.1 assumes the surface syntax matters for the SISY paper's figures (a PyTorch reader expects `layers: [ ... ]`). This rewrite argues that matters less than paper readers might think — HyMeKo's native idiom is legible enough with `using` aliases, and figures can always present pseudocode when needed.

If the `torch_network` surface is a hard figure-level requirement: then Tier C macros is the right vehicle, not direct LALR additions. Tier C's parametric-macro mechanism lets `torch_network MyModel { layers: [...] }` desugar to the factor view or dataflow view at parse time, preserving both the reader-friendly surface and the no-new-IR-shape property. This is strictly additive with Tier C.

If native HyMeKo idiom is acceptable for the paper: then this rewrite is the final word, and §5.1 of the original spec ships as-is with the substitutions above, no grammar work at all.

The fallback is strictly safer than the original: nothing forecloses.
