# Hymeko Fast State V2: Zero-Copy Topological Pipeline Architecture

## Abstract
This document outlines the end-to-end architecture of a high-performance, real-time hypergraph processing and visualization engine. The system solves the standard $\mathcal{O}(|E| \cdot d^2)$ combinatorial bottleneck of graph-projected n-ary relationships. It achieves this by implementing a zero-copy Inter-Process Communication (IPC) pipeline that feeds an $\mathcal{O}(|E| \cdot d)$ Star Expansion (incidence structure) directly into an interactive, hardware-accelerated 3D OpenGL manifold.

---

## 1. Zero-Copy IPC Memory Architecture
The backbone of the system is built on **Iceoryx2**, eliminating serialization latency and memory allocation overhead between the Rust daemon and the Python presentation layer.

### 1.1 Topic Topology and Subscription
The pipeline broadcasts on strictly separated topics to isolate the sparse incidence data from the dense adjacency projections:
* **Ingress:** `HymekoFastStateV2/query/src` (or `/ir`) accepts the raw query payloads.
* **Star Expansion Stream:** `HymekoFastStateV2` transmits the 3D incidence tensors.
* **Clique Expansion Stream:** `HymekoFastStateV2/tensor/clique` transmits the 2D projected adjacency tensors.

### 1.2 Memory Loan Injection
Query payloads (either CBOR-compressed data or UTF-8 strings) are injected into the daemon without standard buffer copying.
* The publisher requests a bounded memory segment via `publisher.loan_slice_uninit(len(payload))`.
* Bytes are written directly into shared memory.
* A cross-process event is triggered via `notifier.notify()`, consistently achieving injection speeds under $100$ microseconds.

### 1.3 Arrow IPC Decoding & The Latched Event Loop
Tensors are passed as continuous Apache Arrow Record Batches.
* The Python subscriber reads the payload directly from the shared memory pointer via `pyarrow.ipc.RecordBatchStreamReader(io.BytesIO(...))`.
* To prevent asynchronous rendering race conditions, the receiver utilizes a non-blocking `receive()` poll that explicitly *latches* both the Star and Clique dataframes. The GPU rendering sequence is strictly locked until both tensor projections populate the shared memory space.

---

## 2. Mathematical Dimensionality Reduction
The engine processes two simultaneous mathematical perspectives of the same topology to benchmark and prove computational efficiency.

### 2.1 The Star Monolith (Sparse Incidence)
Represents hyperedges strictly as bipartite associations. The schema utilizes a 3D coordinate system `(i, j, k)` combined with a float `val`.
* **Complexity:** $\mathcal{O}(|E| \cdot d)$
* **Scale:** At a grid configuration of $50$ nodes, $50$ edges, and a $0.30$ density factor, the Star Expansion remains highly sparse, requiring only $1,498$ Non-Zero (NNZ) matrix entries.
* **Latency:** The pipeline extracts and delivers this tensor in $79.72$ ms on average under heavy load.

### 2.2 The Clique Floor (Projected Adjacency)
A traditional 2D graph projection where every hyperedge becomes a fully connected subgraph (clique). The schema maps directly to `(i, j)` coordinates.
* **Complexity:** $\mathcal{O}(|E| \cdot d^2)$
* **Scale:** At the same $50/50/0.30$ configuration, the combinatorial explosion forces the tensor to $10,991$ NNZ entries.
* **Latency:** The heavy Clique transformation incurs up to $496.04$ ms of compute delay, highlighting the necessity of the Star model for Hypergraph Neural Networks (HGNNs) scaling.

!

---

## 3. High-Fidelity Volumetric Rendering
The visualization layer discards standard 2D plotting in favor of a custom, PyQtGraph-based volumetric OpenGL coordinate space, allowing for real-time topological inspection.

### 3.1 Coordinate Truth Mapping
Discrete PyTorch sparse COO indices are mapped directly to continuous 3D spatial grids ($X, Y, Z$).
* A dynamic centering offset calculation, specifically `off_i = (max_i + 1) / 2` and `off_j = (max_j + 1) / 2`, ensures the tensor remains perfectly balanced on the camera's focal point regardless of topological size.

### 3.2 Stratified Z-Layering
* **Incidence Monolith:** The Star Expansion floating cubes are positioned at their true $Z$-index (mapped from tensor slice $k$).
* **Adjacency Shadow:** The Clique connections are flattened to a $Z = -0.5$ ground plane. This thin topological gap visually reinforces the concept of higher-dimensional incidence casting a lower-dimensional projection.

### 3.3 Volumetric Shading Engine
* Primitive wireframes are replaced with `gl.GLMeshItem` geometries utilizing Phong reflection (`shader='shaded'`) and `glOptions='opaque'`.
* To combat default ambient light dropoff on white backgrounds (`#FDFDFD`), high-luminance colormaps (`spring` for Star, `viridis` or cyan for Clique) are applied. This guarantees that every slice of the structure remains vibrant and distinct.

### 3.4 Kinematic Interaction
Default PyQtGraph camera controls restrict lateral movement. By subclassing `gl.GLViewWidget` into a `PanViewWidget`, vector translations are mapped to `mouseMoveEvent` via `QtCore.Qt.MidButton`. This enables fluid, CAD-style spatial panning essential for large-scale structure inspection.

!

---

## 4. Ingress and Verification Modalities
The system supports both physical and procedural ingress routes to guarantee testing viability across different hardware setups.

### 4.1 Physical Optical Ingress
* Utilizes OpenCV (`cv2`) and `pyzbar` to capture and decode dense CBOR-compressed QR payloads from live video feeds.
* Implements `zlib` decompression before injecting the raw bytes into the Iceoryx2 shared memory layer.

### 4.2 Procedural "Pinnacle" Ingress
* A fully synthetic procedural generator (`generate_hymeko_file`, `generate_hymeko_graph_text`) constructs valid structural strings with precise node, edge, and density parameters.
* Proves parallel rendering stability by triggering both a NetworkX 2D bipartite plot (using `block=False`) and the PyArrow-backed 3D OpenGL application simultaneously on the same hardware thread from a single payload event.