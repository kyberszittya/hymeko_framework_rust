"""Reusable, metric-equipped embeddings for the HyMeKo learning lines.

Currently exposes the Cayley-rotor embedding (see ``cayley_rotor``): a
Clifford/quaternion rotor feature whose rotation is parameterised by the Cayley
map, so unconstrained SGD optimises *on the rotor manifold* by construction. It
is the inductive, leakage-free replacement for the transductive ``nn.Embedding``
lookup used across the signed-link models (plan:
``docs/plans/2026-06-16-soma-structural-highway/``).
"""
from .cayley_rotor import CayleyRotorEmbedding, cayley_to_unit_quat, quat_rotate

__all__ = ["CayleyRotorEmbedding", "cayley_to_unit_quat", "quat_rotate"]
