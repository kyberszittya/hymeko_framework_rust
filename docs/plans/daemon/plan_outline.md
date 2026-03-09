# Hymeko Daemon: Master Implementation Plan

## Architectural Outline

### Phase 1: Core Engine Refactoring (The Clean Break)
* **Task 1.1:** Create the `hymeko_daemon` workspace and migrate network logic.
* **Task 1.2:** Replace SipHash with `rustc-hash::FxHashMap` for local tensor loops.
* **Task 1.3:** Convert the compiler's `Index` to a `BTreeMap` for deterministic canonical hashing.
* **Estimated Time:** 2 - 3 hours.

### Phase 2: The Data Plane (Memory & Transport)
* **Task 2.1:** Integrate `iceoryx2` for pure Rust zero-copy shared memory allocation.
* **Task 2.2:** Define the strict Apache Arrow `RecordBatch` schema (`k`, `i`, `j` as `Int64`, `val` as `Float32`).
* **Task 2.3:** Write the translation layer packing `FxHashMap` data into contiguous Arrow memory blocks.
* **Estimated Time:** 4 - 6 hours.

### Phase 3: The Control Plane (Concurrency & Networking)
* **Task 3.1:** Initialize a `moka` concurrent LRU cache mapping Blake3 ETags to `iceoryx2` memory handles.
* **Task 3.2:** Spin up the Tokio asynchronous I/O reactor for Zenoh/CBOR traffic.
* **Task 3.3:** Build the Tokio-to-Rayon bridge using `oneshot` channels to offload heavy math.
* **Estimated Time:** 4 - 5 hours.


## Phase 4: The Event-Driven Control Plane
**Objective:** Build a deterministic, asynchronous event router separating heavy data payloads (Shared Memory) from lightweight control signals (Pub/Sub) to prevent torn reads and schema mismatches.

* **Layer 1: The Dispatcher (Rust Core)**
  * **Topology Hashing:** Computes the $O(1)$ cryptographic representation of the B-Tree index.
  * **Event Classification:** Decides mathematically whether to emit a `TopologyShift` (requires Arrow schema rebuild) or a `WeightStream` (pointer bump).
* **Layer 2: The Atomic State Gate (Daemon Layer)**
  * **Double-Buffering:** Manages the `u8` front/back buffer toggle, allowing concurrent read/writes.
  * **Read-Lock Enforcement:** Prevents the Daemon from overwriting memory while a Subscriber holds a `Sample`.
* **Layer 3: The Schema Registry (FFI Contract)**
  * **Version Handshake:** Forces Python to verify its Arrow schema layout matches the Daemon before DLPack ingestion.
* **Layer 4: The FFI Reactor (Python/Tokio Bridge)**
  * **Zero-Polling Listeners:** Parks the thread using `iceoryx2` waitsets (0% CPU) until the Dispatcher fires.

## Phase 5: Hypergraph Rewriting & Traversal Engine (HRE)
**Objective:** Implement a functional visitor pattern and query engine for topological pattern matching, structural analytics, and domain transformation (e.g., URDF/SDF generation).

* **Task 1: The Motif Matcher (Query Engine)**
  * Implement subgraph isomorphism to efficiently query the `BTreeMap` AST (e.g., identifying kinematic chains or dense cliques).
* **Task 2: The Functional Visitor (Read-Only Functors)**
  * Create the `HypergraphVisitor` trait to execute closures on matched subgraphs for domain translation or analytics without mutating the core.
* **Task 3: Graph Rewriting (Mutation & Trigger)**
  * Implement safe mutation methods (node pruning, edge routing).
  * Wire rewrites to automatically invalidate the `topology_hash`, instantly triggering a `TopologyShift` event in the Phase 3 control plane.

### Phase 6: The Dual-State Query Engine (Schmidhuber/Schönhage)
* **Task 4.1:** Build the Python programmatic API (AST to CBOR serialization; no string parsing).
* **Task 4.2:** Implement the Datalog structural filter in Rust (modifying the `Int64` Slow State).
* **Task 4.3:** Implement the STL evaluator on the `Float32` Fast State for PyTorch.
* **Estimated Time:** 10 - 14 hours.

### Phase 7: The Fuzzy Signature Engine (Spectral Similarity)
* **Task 5.1:** Generate spectral embeddings via stochastic trace of the Normalized Scaled Laplacian.
* **Task 5.2:** Implement a Sparse MinHash Layer over the `k` and `i` Arrow arrays.
* **Task 5.3:** Apply Gödel T-norm thresholding for cache retrieval fallback.
* **Estimated Time:** 6 - 8 hours.

### Phase 8: The LLM Verification Interface (The Sandbox)
* **Task 6.1:** Connect LLM generation strictly to your Python Datalog/STL API objects.
* **Task 6.2:** Enforce strict AST validation and type-checking in Python before CBOR serialization.
* **Task 6.3:** Implement a Rust dry-run mechanism to verify structural legality before memory allocation.
* **Estimated Time:** 4 - 6 hours.

### Phase 9: The OpenGL Tensor View (Zero-Copy Visualization)
* **Task 7.1:** Configure the OpenGL client as a third subscriber to the `iceoryx2` shared memory.
* **Task 7.2:** Map Arrow `Int64` arrays directly to SSBOs for native instanced rendering (spatial coordinates).
* **Task 7.3:** Map the `Float32` fast-state array to dynamic buffers for real-time shader color/width updates.
* **Estimated Time:** 8 - 12 hours.
