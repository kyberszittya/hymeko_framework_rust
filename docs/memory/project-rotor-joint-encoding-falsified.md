---
name: project-rotor-joint-encoding-falsified
description: "S¹ (cos,sin) rotor encoding of robot joint angles in hymeko_rl is FALSIFIED — revolute joints are range-limited, never wrap; do not re-propose"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 7f41ec08-5aea-42ef-8f98-796f5ab83727
---

Proposed (2026-06-19) encoding `hymeko_rl` joint angles as the 2-D S¹ rotor θ↦(cos θ,sin θ)
(`Cl(0,1)⁺`, smallest [[project-cayley-rotor-idea]] member) as a third arm of the reach
ablation. **User falsified it and told me to drop it.**

**Why:** real revolute joints are range-limited (`range="-2.5 2.5"` < π in arm_world.py) — they
**never reach the ±π wrap**, so the periodicity the rotor exploits buys nothing. The 1-seed
smoke confirmed it (rotor arm *worse*: 0.329 vs hsikan 0.240 m). My floated "fix" — drop the
joint limit to make a continuous joint — is **physically improper**: that is not how arms work.

**How to apply:** the Cayley/quaternion rotor line stays valuable for its real role (leakage-free
inductive *embeddings* / SO(3) composition, [[project-cayley-rotor-idea]]). It does **not**
transfer to RL joint-coordinate encoding. Don't re-suggest cos/sin or continuous-joint variants
for `hymeko_rl`. Fully reverted (no trace left in hymeko_rl / meta_observation.hymeko).
