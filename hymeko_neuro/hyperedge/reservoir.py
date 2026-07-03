"""Memory-bounded reservoir sampler — Vitter (1985), Algorithm R.

Used by the Python-fallback paths of cycle / walk enumeration to cap
peak memory at ``O(cap)`` items regardless of how many items the DFS
visits. Mirrors the Rust ``hymeko_graph::unsigned_cycles::sink::Sink::
Reservoir`` semantics so that the Python fallback and the Rust path
produce uniformly-sampled results from the same population.

Why this exists: the Komondor probe 13883886 OOM-killed at 31.97 GB
on a bitcoin_alpha walk_len=4 enumeration. Root cause: the
``hymeko_signedkan.sif`` Singularity image did NOT include the
``hymeko`` Rust wheel, so every cycle / walk enumeration silently
fell through to the pure-Python ``_python_walks`` / ``enumerate_k_cycles``
which built a full Python ``list`` of every visited tuple — billions
on a power-law-degree graph. Adding the reservoir caps that list at
``cap`` (typically 100k) regardless.

The reservoir is also a defensive net: if a future Singularity build
loses the wheel again, the Python fallback still terminates within
budget instead of OOM-killing the job.
"""
from __future__ import annotations

import math
import random
from typing import Generic, Iterable, TypeVar

import numpy as np

T = TypeVar("T")


class ReservoirSampler(Generic[T]):
    """Bounded-memory uniform sample of a stream of unknown length.

    Vitter (1985), Algorithm R. After ``offer()`` has been called
    ``n`` times, ``self.items`` holds a uniform random sample of size
    ``min(cap, n)`` from the full input stream — but peak memory was
    only ``O(cap)``.

    Parameters
    ----------
    cap
        Maximum number of items to retain. ``None`` ⇒ unbounded
        (degenerates to ``list.append``), kept as an explicit opt-out
        for callers that genuinely want every visited item.
    seed
        Seed for the per-item replacement RNG. Two samplers with the
        same seed offered the same stream return byte-identical
        ``items`` lists, so the Python and Rust paths agree on which
        rows survive when the same seed is used.

    Notes
    -----
    The Vitter R algorithm:

      i ∈ [0, cap):       always retain item i (fill phase)
      i ∈ [cap, n):       pick j ∈ [0, i] uniformly; if j < cap,
                          replace ``items[j]`` with the new item

    The resulting ``items`` list is a uniform sample, but its row
    order is NOT a uniform shuffle of the original positions — it
    reflects the survival order under random replacement. Downstream
    consumers should treat the sample as an unordered set.
    """

    def __init__(self, cap: int | None, seed: int = 0):
        if cap is not None and cap < 0:
            raise ValueError(f"cap must be >= 0 or None, got {cap}")
        self.cap = cap
        self.items: list[T] = []
        self.seen = 0
        self.rng = random.Random(int(seed))

    def offer(self, item: T) -> bool:
        """Submit one item. Returns True so callers may use the same
        ``while sampler.offer(...)`` shape they use against an
        early-stop sink — this reservoir never asks the producer
        to halt (an unbiased sample requires seeing the whole stream).
        """
        if self.cap is None or self.seen < self.cap:
            self.items.append(item)
        else:
            j = self.rng.randrange(self.seen + 1)
            if j < self.cap:
                self.items[j] = item
        self.seen += 1
        return True

    def extend(self, stream: Iterable[T]) -> None:
        """Offer every item in ``stream``. Convenience for the common
        case of sampling an existing iterable."""
        for item in stream:
            self.offer(item)

    def __len__(self) -> int:
        return len(self.items)

    def __repr__(self) -> str:
        return (
            f"ReservoirSampler(cap={self.cap}, "
            f"len(items)={len(self.items)}, seen={self.seen})"
        )


class NumpyReservoirSampler:
    """Fixed-shape numpy-backed reservoir sampler. Vitter Algorithm L.

    For the DFS-of-fixed-arity-integer-tuples case (cycle / walk
    enumeration), this is dramatically more efficient than the
    object-based :class:`ReservoirSampler`:

    1. **No per-offer Python-object allocation.** The reservoir is a
       preallocated ``(cap, k)`` numpy array; ``offer(seq)`` copies the
       sequence directly into a row via numpy's C-level row assignment.
       Eliminates the ``tuple(path)`` per DFS visit that the previous
       Python-tuple reservoir paid for every node (billions of
       allocations on a power-law-degree graph like bitcoin_alpha
       walk_len=4).

    2. **Vitter Algorithm L instead of R.** Algorithm R generates a
       random index PER VISIT; Algorithm L computes the next
       reservoir-replacement index in closed form using a
       geometric-distribution-like skip. For a stream of length N
       with cap K, the expected number of RNG draws is
       ``K · (1 + log(N/K))`` instead of N — typically 1000× fewer
       on the production-scale workloads (bitcoin_alpha walk_len=4
       has ~4 × 10⁹ stream length, K=10⁵ → ~10⁶ RNG calls instead of
       4 × 10⁹). The reservoir-replacement target ``j`` is uniform
       over ``[0, cap)``, matching Algorithm R's distribution.

    The two efficiency tricks compose: the hot path is a single
    branch + integer compare + ``seen += 1`` on rejected offers.

    Same uniform-sample guarantee as the object-based class.
    Caller must know the row width ``k`` and dtype up front.
    """

    def __init__(
        self, cap: int, k: int,
        dtype: np.dtype | type = np.int32,
        seed: int = 0,
    ):
        if cap < 0:
            raise ValueError(f"cap must be >= 0, got {cap}")
        if k < 0:
            raise ValueError(f"k must be >= 0, got {k}")
        self.cap = int(cap)
        self.k = int(k)
        self.dtype = np.dtype(dtype)
        self.buf = np.zeros((max(self.cap, 0), self.k), dtype=self.dtype)
        self.seen = 0
        self.rng = random.Random(int(seed))
        # Algorithm L state. ``W`` shrinks across selections; the next
        # row to overwrite is at stream index ``self.next_select``.
        # The expressions are deferred to first-real-use to keep the
        # cap=0 / cap=None degenerate path zero-cost.
        if self.cap > 0:
            self.W = math.exp(math.log(self.rng.random()) / self.cap)
            self.next_select = (
                self.cap - 1
                + 1
                + int(math.log(self.rng.random()) / math.log(1 - self.W))
            )
        else:
            self.W = 0.0
            self.next_select = 0

    def offer(self, seq) -> None:
        """Submit one length-``k`` sequence (list / tuple / 1-D numpy).

        Hot path on rejection: one integer compare + one increment. No
        allocation, no RNG call. Only when ``seen == next_select`` (or
        in the pre-fill phase) does the sampler do real work.
        """
        if self.cap == 0:
            self.seen += 1
            return
        if self.seen < self.cap:
            # Pre-fill phase: always accept, direct row assignment.
            # numpy handles the list/tuple → row copy in C.
            self.buf[self.seen] = seq
        elif self.seen == self.next_select:
            # Algorithm L selection: replace a uniform-random reservoir
            # row, then advance ``W`` and compute the next stream index
            # to land on. Two RNG draws regardless of stream length.
            j = self.rng.randint(0, self.cap - 1)
            self.buf[j] = seq
            self.W *= math.exp(math.log(self.rng.random()) / self.cap)
            # Skip to the next selection in O(log) expected draws.
            try:
                step = int(math.log(self.rng.random()) / math.log(1 - self.W))
            except (ValueError, ZeroDivisionError):
                # ``log(1 - W)`` is -inf when W -> 1 (sample exhausted);
                # set step to a huge value so we never select again.
                step = 1 << 62
            self.next_select += 1 + step
        # else: between selections — just bump the counter, no work.
        self.seen += 1

    def to_array(self) -> np.ndarray:
        """Return the ``(min(cap, seen), k)`` view of the reservoir.
        Slice of the preallocated buffer — no copy."""
        n_kept = min(self.cap, self.seen)
        return self.buf[:n_kept]

    def __len__(self) -> int:
        return min(self.cap, self.seen)

    def __repr__(self) -> str:
        return (
            f"NumpyReservoirSampler(cap={self.cap}, k={self.k}, "
            f"dtype={self.dtype}, seen={self.seen}, kept={len(self)})"
        )
