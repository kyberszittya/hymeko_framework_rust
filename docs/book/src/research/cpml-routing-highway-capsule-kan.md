# CPML tiers as **routes**: Highway, Capsule, and KAN views

This note updates the **mathematics** and **design story** for **Concentric‑Pyramid Multi‑Layer (CPML)** inside `signedkan_wip/src/cpml.py` and **`InnerCPMLCore`** (`hymeko_gomb/shells.py`). The default readout is **`topology="route"`**; **`topology="pyramid"`** keeps the legacy inward widening stack.

---

## 1. Objects and notation

- **Graph:** vertices \(v\in V\), directed signed edges, and a multiset of **signed** simple cycles (or tuples) \(c\) with vertex set \(V(c)\subseteq V\) and per‑corner signs \(\sigma_{c,i}\in\{\pm 1\}\).
- **Tier map:** \(\tau:V\to\{0,\ldots,L-1\}\) from **degree percentiles** (`TierSpec.cuts`). Tier \(0\) = periphery (low degree mass), \(L-1\) = core/hubs — same *stratification* in both topologies.
- **Base vertex signal:** \(X^{(0)}\in\mathbb{R}^{N\times d_{\mathrm{in}}}\) — in Gömb this is the **concatenation** of embed, outer, and middle shells; in bare CPML it is `node_features`.
- **Per‑tier aggregator:** \(\mathrm{Agg}_\ell\) (MLP stub, **SignedKAN** / spline aggregator, or Clifford‑FIR) maps **corner features** of cycles in tier \(\ell\)’s pool to a **per‑cycle** vector, then **scatter‑mean** yields \(H_\ell\in\mathbb{R}^{N\times d_{\mathrm{layer}}}\).

**Tier‑restricted cycle pool**

\[
\mathcal{C}_\ell \;=\; \{\, c \;:\; \exists\, v\in V(c),\; \tau(v)=\ell \,\}\,.
\]

So **tier is a routing predicate on cycles**, not an “inward only” pyramid by default.

---

## 2. Route topology (default): parallel routes + one concat

For each \(\ell=0,\ldots,L-1\):

1. Restrict to \(\mathcal{C}_\ell\).
2. **Always** read corner features from the **same** base \(X^{(0)}\) (not from a widening state).

\[
H_\ell \;=\; \mathrm{ScatterMean}\Bigl(\mathrm{Agg}_\ell\bigl(X^{(0)}[c],\sigma_c\bigr)_{c\in\mathcal{C}_\ell}\Bigr)\in\mathbb{R}^{N\times d_{\mathrm{layer}}}\,.
\]

**Final vertex embedding** before the edge head:

\[
X^{\mathrm{final}} \;=\; \bigl[\,X^{(0)} \,\Vert\, H_0 \,\Vert\, \cdots \,\Vert\, H_{L-1}\,\bigr]
\quad\in\;\mathbb{R}^{N\times\,(d_{\mathrm{in}} + L\,d_{\mathrm{layer}})}}\,.
\]

**Edge logits** (as implemented): MLP on \(\bigl[X^{\mathrm{final}}_u \,\Vert\, X^{\mathrm{final}}_v\bigr]\).

**Properties**

- **Width of deep state per tier is fixed** at inputs to \(\mathrm{Agg}_\ell\): always \(d_{\mathrm{in}}\) on the corner branch for the MLP stub — no widening matmuls tier‑over‑tier.
- **Parameter count** of tier MLPs is **smaller** than pyramid when \(L>1\) (each \(\mathrm{Agg}_\ell\) sees \(d_{\mathrm{in}}\), not \(d_{\mathrm{in}}+\ell d_{\mathrm{layer}}\)).
- **Peak activations** avoid the growing concat tensor of pyramid mode — the main **VRAM** win in joint‑slot Gömb.

---

## 3. Pyramid topology (legacy): inward widening

Let \(x^{(0)}=X^{(0)}\). For \(\ell=0,\ldots,L-1\):

\[
H_\ell \;=\; \mathrm{ScatterMean}\Bigl(\mathrm{Agg}_\ell\bigl(x^{(\ell)}[c],\sigma_c\bigr)_{c\in\mathcal{C}_\ell}\Bigr),\qquad
x^{(\ell+1)} \;=\; \bigl[\,x^{(\ell)} \,\Vert\, H_\ell\,\bigr]\,.
\]

So \(\mathrm{Agg}_\ell\) sees **increasing** corner width \(d_{\mathrm{in}} + \ell\,d_{\mathrm{layer}}\) — the original “concentric pyramid” **inward funnel** picture: each tier’s aggregator lives in a larger space.

**Same** \(X^{\mathrm{final}} = x^{(L)}\) width \(d_{\mathrm{in}} + L d_{\mathrm{layer}}\) as route mode, so the **edge head interface is unchanged**; only **internals** differ.

---

## 4. Unifying lens A — **Highway networks**

Srivastava et al. (*Highway Networks*, arXiv:1505.00387) and the closely related **LSTM** gating idea: keep a **carry** path and add **transformed** content with learned gates.

**Structural analogy (route CPML):**

| Highway idea | Route CPML instantiation |
|--------------|---------------------------|
| **Carry** of raw inputs | \(X^{(0)}\) is **never replaced**; it is concatenated **once** at the end and is the **sole** vertex signal read by every \(\mathrm{Agg}_\ell\) on the corner path. |
| **Transform** branches | Each \(H_\ell\) is an additive “view” of cycle evidence **at tier \(\ell\)’s routing** — different **gates** are implicit in *which cycles fire* (\(\mathcal{C}_\ell\)), not only in scalar gates on channels. |
| **Parallel** then **merge** | Tiers are **parallel routes** (no widening concat between tiers); merge = final \(\Vert\) — same spirit as multi‑lane highway fusion before readout. |

We are **not** claiming a literal identity to elementwise \(T,C\) gates; we claim a **design isomorphism**: **identity path for base features** + **multiple transformed pathways** that are **fused** for downstream prediction.

---

## 5. Unifying lens B — **Capsule networks** (CapsNet)

Sabour, Frosst, Hinton (*Dynamic Routing Between Capsules*, NeurIPS 2017): **capsules** group neurons; **routing** sends lower‑level capsule votes to parents using **agreement** (iterative softmax “routing by agreement”).

**Structural analogy (route CPML):**

| CapsNet idea | Route CPML instantiation |
|--------------|---------------------------|
| **Groups / capsules** | Each tier \(\ell\) defines a **group of cycles** \(\mathcal{C}_\ell\) and a **group output** \(H_\ell\) — one “capsule map” per tier after scatter. |
| **Routing coefficients** | **Hard, structure‑given** routing: cycle \(c\) participates in tier \(\ell\) iff it **touches** a vertex with \(\tau(v)=\ell\). So \(r_{c\to \ell}\in\{0,1\}\) from **graph + degree stratification**, not from EM iterations. |
| **Agreement** | “Agreement” is **topological**: a cycle **agrees** to route to tier \(\ell\) by **incidence** to that tier’s vertices — analogous to low‑level capsules voting for parents they geometrically support. |
| **Pose / object parts** | Replace “part–whole” in images by **periphery–hub** in the graph: tiers are **scales of centrality**, cycles are **local witnesses** routed to the scales they touch. |

**CapsNet‑inspired soft routing (implemented as `tier_organization="capsule_soft"`):** a small MLP maps each cycle’s **mean‑pooled corner features** of \(X^{(0)}\) to logits over the \(L\) tiers; **softmax** yields \(R_{c,\ell}\). Every tier’s aggregator runs on **all** cycles; contributions are scaled by \(R_{c,\ell}\) before scatter‑mean. **`topology` must be `route`** (pyramid widening is rejected). CLI / config: `CPMLConfig.tier_organization`, `GombConfig.cpml_tier_organization`, `run_gomb_smoke --cpml-tier-organization …`.

The default **`structural`** mode keeps **interpretable** hard masks from \(\tau\) (stratified structural agreement). Iterative EM routing (CapsNet inner loops) is **not** implemented — one softmax pass per forward.

---

## 6. Unifying lens C — **KANs** (Kolmogorov–Arnold networks)

Liu et al. (*KAN: Kolmogorov–Arnold Networks*, 2024): replace fixed linear layers by **learnable univariate functions** on edges of the **computational graph** (here: **corners** of cycles).

**Where KANs sit in CPML**

- **`aggregator_kind="hsikan"`** uses **spline / Catmull–Rom** activations on **cycle corners** — the HSiKAN recipe is a **geometric KAN** on the **hypergraph incidence** (not an MLP on flattened adjacency).
- **Clifford‑FIR** tiers are **minimal‑parameter** signed filters — another **basis‑limited** nonlinearity on corners.

So each tier is a **KAN‑style** nonlinear readout on a **routed** sub‑hypergraph; **route vs pyramid** only changes **which width** that readout sees at each depth, not whether the block is KAN‑flavoured.

---

## 7. One sentence **unification thesis**

> **Route CPML** treats **degree tiers as routing slots** over cycles: **Highway‑like** preservation of a **base carry** \(X^{(0)}\), **Capsule‑like** **structural routing** of witnesses into tier groups, and **KAN‑like** **corner nonlinearities** inside each \(\mathrm{Agg}_\ell\) — with **learned α** (JointMix / mixed arity) as a **second routing stage** across **tuple kinds**.

**Pyramid** mode remains the **deepening representational stack** analogue (closer to classical **depth** on a widening state).

---

## 8. Gömb cascade placement

- **Outer / middle:** Clifford‑FIR volume and HSiKAN‑CR already implement **multi‑bank** and **spline** geometry on cycles.
- **Inner CPML:** consumes \(X^{(0)}=[\texttt{embed}\Vert\texttt{outer}\Vert\texttt{middle}]\); **`GombConfig.cpml_topology`** selects `route` vs `pyramid`; **`GombConfig.cpml_tier_organization`** selects `structural` vs `capsule_soft` (latter **route only**).
- **`JointMixGomb`:** four stacks (c3, c4, w2, w3) each with its own inner core — memory stress is where **route** pays off most.

---

## 9. References (external)

- Highway: Srivastava, Greff, Schmidhuber — *Training Very Deep Networks* / Highway Networks (2015).
- CapsNet: Sabour, Frosst, Hinton — *Dynamic Routing Between Capsules* (2017).
- KAN: Liu et al. — *KAN: Kolmogorov–Arnold Networks* (2024).

## 10. In‑repo pointers

| Artifact | Path |
|----------|------|
| CPML implementation | `signedkan_wip/src/cpml.py` (`CPMLConfig.topology`, `tier_organization`, `capsule_route_hidden`) |
| Gömb wiring | `signedkan_wip/src/hymeko_gomb/cascade.py`, `shells.py` |
| Smoke CLI | `python -m signedkan_wip.src.run_gomb_smoke --cpml-topology {route,pyramid} --cpml-tier-organization {structural,capsule_soft}` |
| CPML × XHC plan (historical “pyramid” framing) | `docs/plans/2026-05-11-cpml-xhc-architectures/` |
| NN field guide | [NN variants & layer geometry](./nn-architectures-and-layer-geometry.md) |

---

*Last updated: 2026‑05‑12 — `topology`, `tier_organization` (`structural` \| `capsule_soft`), Gömb `cpml_*` fields.*
