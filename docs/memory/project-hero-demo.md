---
name: project-hero-demo
description: "HyMeKo \"hero demo\" plan — one model → many targets, validation-gated, framed as structurally-accountable AI"
metadata: 
  node_type: memory
  type: project
  originSessionId: 51a9ecdd-e07f-4cb9-b952-6f107a810e81
---

Hero-demo concept (from a ChatGPT proposal the user brought 2026-06-13, my plan
on it). Four-format plan saved at
`docs/plans/2026-06-13-hero-demo/` (plan.tex/pdf/tikz/mmd, PDFs compiled clean).

**Thesis:** one authored `.hymeko` model → parse → signed-typed hypergraph IR →
query → **many faithful targets** (urdf/sdf/mjcf/dot/mermaid/gazebo/ros2_launch/
torch_dataflow/sysml — all already in `transforms/`), with a **validation gate**
(`hymeko_pgraph` axioms A1–A5) between IR and every emitter, content-addressable
Blake3 reproducibility, and an **optional** NL front-end whose output must pass
the same gate. Framing pivot (agreed): *"We make AI systems structurally
accountable,"* NOT "beat the LLM."

**Honesty inventory — ~80% already exists.** Net-new / risky, labeled as such in
the plan: (1) the NL→`.hymeko` front-end is **not built** (grep hits were
notes/reports, no code) — demo-able version is thin: NL→source→compile→validate→
reject-if-invalid; the gate is the real pgraph path, only the text generator is
stubbable behind a typed iface (a real LLM client = §1 dep). (2) Live
MuJoCo/Gazebo execution is **machine-gated** (`gz`/`mujoco`/`xacro` absent here);
laptop runs analytic-fallback with explicit note; live sim is a separate run on
the ROS/Gazebo machine.

**Three open decisions** (plan §"Open decisions"): demo input model (reuse a
robotics fixture vs author `hero_*.hymeko` pick-and-place cell); NL provider
(offline generator vs gated LLM client = §1); on-stage target subset (all 8 vs
curated 3–4).

**Why:** new strategic thread the user is actively shaping; distinct from the
[[project-seminar-demos-and-hymeyolo-plan]] inference demos and the
[[project-smc-paper-additions-queue]] paper work.
**How to apply:** no code yet — plan-only per CLAUDE.md §2. Resolve the three
open decisions before implementing `demos/hero/` (one orchestrator over existing
CLI/WASM, §6.5 #3 — no re-implementation of parse/query/transform).

**PHASE 1 BUILT 2026-06-15** (report `reports/2026-06-15-hero-demo-phase1.md`, plan
`docs/plans/2026-06-15-hero-demo-phase1/`). User decided "go with #1 (FANUC robotics spine),
but plan the others." Shipped `demos/hero/` orchestrator (thin over the `hymeko` CLI, §6.5 #3):
`hero_demo.py` (pure `parse_validate_output`+GateVerdict, HeroDemo driver, data-driven SCENARIOS,
BROKEN_TWIN) + `test_hero.py` (7 pytest: 5 unit + 2 integration) + README + .gitignore. **Verified:**
FANUC arm + anthropomorphic arm each emit 5 targets (urdf/sdf/mjcf/dot/mermaid); broken twin
(dangling joint ref `ghost_link`) REJECTED by the gate. ruff+mypy --strict+radon clean. Decision #1
RESOLVED = robotics spine; #2 (NL provider) deferred/stubbable; #3 (targets) = 5 kinematic formats.
**CORRECTED 2026-06-15:** earlier "validate/emit exit 0 on failure" was WRONG (read `$?` after a pipe
`| head` = head's code). CLI ALREADY exits non-zero on hard failure (validate/emit broken → 1; warnings
keep 0). Phase 1.5 CLI-hardening was a NON-ISSUE; instead the orchestrator now uses the exit code as the
authoritative gate via `verdict_from_run(text, code)` (report `reports/2026-06-15-hero-demo-gate-exitcode.md`,
12 pytest, ruff/mypy clean). Lesson: never read `$?` after a pipe. Roadmap:
Phase 3 Gömb/Soma perception (needs Soma vision round-tripped through .hymeko — currently Python-only
in signedkan_wip); editor profile for a hero cell (imports feature already loads the multi-file FANUC
cell). Gömb=3-shell neural cascade (Clifford-FIR/HSiKAN/CPML), Soma=reflex/vision lane — cognitive
stack, NOT robot kinematics.

**PHASE 2 (HYBRID) DONE 2026-06-15** (report `reports/2026-06-15-hero-demo-phase2.md`, plan
`docs/plans/2026-06-15-hero-demo-phase2/`). Same gated pipeline now emits robots AND learners: added
`Target.via` ("emit" vs template "transform"), pure `emit_args`, `HeroScenario.kind` (robot/learner),
NEURAL_TARGETS (torch_dataflow+dot), 2 learner scenarios (simple_net, mnist_resmlp_3), `cwd=REPO_ROOT`
in `_run`. Demo: 2 robots×5 targets + 2 learners×2 (torch_dataflow runnable nn.Module 4205/6224 B) +
broken twin rejected, all one gate. 11 pytest (+4) / ruff / mypy --strict clean. NO torch dep (emits+gates
module SOURCE only). NO CORE edit. Thesis "one accountable IR → robots and learners" now concrete.
**EDITOR HERO-CELL PROFILE DONE 2026-06-15** (report `reports/2026-06-15-editor-robot-arm-profile.md`):
added a "Robot arm (imported kinematics)" editor profile — `data/profiles/{meta_kinematics,robot_arm}.hymeko`
(compact kinematics vocab + arm root that `@"meta_kinematics.hymeko"`-imports it) embedded in
`docs/editor/views/profiles.js` (consistency-tested) + Rust multi-file compile case. Browser-verified
(?profile=robot_arm&select=spin_joint): arm compiles 31 nodes/6 edges with imported vocab, kinematics
palette, joint's ARC-REFS editor shows editable origin transform [[0,0,0.1],...]. Ties together
imports+profiles+palette+arc-editing+hero in the live editor. Editor cache now v=25 (profiles.js v=25).
(Phase 1.5 resolved — see correction above.)

**PHASE 3 (GÖMB STRUCTURAL PARITY + MINIMAL SOMA) DONE 2026-06-15** (report
`reports/2026-06-15-hero-demo-phase3.md`, plan `docs/plans/2026-06-15-hero-demo-phase3/`).
User decisions: Gömb cascade first then Soma; STRUCTURAL parity (gate-framed, NOT runnable); Soma
included; torch-free. KEY FINDING: the Gömb/HSiKAN architecture was ALREADY authored in
`data/nn/hsikan_mixed.hymeko` (per-arity signedkan_layer ×4 + arity_mixer αₖ + signed_classifier) on
meta_nn vocab; the torch_dataflow template already emits its kinds. Built `demos/hero/learner_parity.py`
(pure: parse_hymeko_layers / parse_torch_attrs / structural_parity — emit faithful iff every IR layer →
a self.<name> submodule; NO torch import) + new `data/nn/soma_vision.hymeko` (patch hypergraph_conv →
walk_layer → signed_classifier; minimal Soma using only emittable meta_nn kinds) + Gömb & Soma learner
scenarios in hero_demo with a parity line. VERIFIED: Gömb faithful 6/6, Soma faithful 3/3, simple_net 2/2,
mnist 5/5; broken twin rejected; 15 pytest, ruff/mypy clean, plan PDF, Rust multi_file 4/4. Soma's faithful
Hodge/stim/patch internals (signedkan_wip) + numeric round-trip vs cascade.py remain (latter pulls torch).
NOTE: most of this phase was written while the harness command-safety classifier was temporarily down (all
shell exec blocked); held unverified, batch-verified on recovery.
