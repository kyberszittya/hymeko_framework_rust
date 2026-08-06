# HyMeKo WASM Emergent Agent

Browser demo for emergent-agent reinforcement-learning witnesses: a 4x4 grid,
a larger 6x6 grid, a radius-3 hexaworld, generated custom worlds, an
adversarial cellular-automaton arena, and a virus cellular-automaton gameplay
arena.

Build the WASM bundle into the shared editor package directory:

```bash
cd hymeko_wasm
wasm-pack build --target web --release --out-dir ../docs/editor/pkg
```

Serve the docs folder over HTTP:

```bash
cd docs
python -m http.server 8000 --bind 127.0.0.1
```

Open:

```text
http://127.0.0.1:8000/wasm_game/
```

The page calls `solved_world_json(world_id)` for presets and
`generated_world_json(topology, width, height)` for custom worlds. Square custom
worlds use width and height directly; hex custom worlds use the larger dimension
as radius. The adversarial preset calls `adversarial_world_json()`, which adds
Kage as a competitive agent and an evolving cellular automaton hazard layer. The
virus preset calls `virus_world_json()`, where Hikari fights an infection-style
cellular automaton with local cleanse pulses while trying to reach victory. The
Rust/WASM solver gates a generated HyMeKo agent description and returns the
agent-state/observation/action/goal vocabulary, simulation trace, source text,
and Mermaid policy graph as JSON.

Use `Randomize` to call the seeded variants: `randomized_world_json(...)`,
`generated_random_world_json(...)`, or `adversarial_world_seed_json(...)`. The
single-agent variants move the EGO/Hikari start; the adversarial variant also
moves Kage and the initial cellular-automaton hazard cells. The virus variant
uses `virus_world_seed_json(seed)` to move Hikari and the starting infection
front.

The generated HyMeKo source intentionally exercises richer parser features:
comments, a document header, nested model spaces, typed inheritance via
`+ <isa>`, scalar/vector-like attributes, and signed hyperedge arc references.
The concrete actions are nested under the agent-owned `hikari_policy`, and
`@agent_policy_binding` links `agent_hikari` and `hikari_state` to that policy.
The Evolution tab visualizes learning milestones: the initial scaffold, early
value sweeps, and the converged policy, repainting the board and showing the
HyMeKo source that corresponds to each stage.
