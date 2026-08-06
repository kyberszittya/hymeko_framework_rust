//! HOTARU planning over HIVE-delta space.
//!
//! HOTARU does not commit structure directly. It proposes named deltas over the
//! current HIVE state, and this adapter lowers each delta to a HyQL refinement
//! that must still pass through AKOIRE's gatekeeper.

use std::collections::VecDeque;

use crate::context::{CognitiveContext, Kyosei, Objectives};
use crate::engine::{preview_edge_names, preview_edges, preview_graph, GraphView};
use crate::search::{solve, AstarResult, SearchProblem};
use crate::synthesize::{CognitiveSynthesizer, Refinement};

/// A named candidate mutation in HIVE-delta space.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HiveDelta {
    name: String,
    op: HiveDeltaOp,
}

impl HiveDelta {
    /// Replace the whole rendered HIVE state with `source`.
    #[must_use]
    pub fn replace_source(name: impl Into<String>, source: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            op: HiveDeltaOp::ReplaceSource(source.into()),
        }
    }

    /// Append a HyQL source fragment to the current rendered HIVE state.
    #[must_use]
    pub fn append_source(name: impl Into<String>, fragment: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            op: HiveDeltaOp::AppendSource(fragment.into()),
        }
    }

    /// A *semantic* graph operation: add a hyperedge named `name` over `nodes` (a top-level
    /// `@name : n0, n1, … { }`). The delta HOTARU's graph planner composes — grounded in real node
    /// names — so a plan is a sequence of edges over the model's own elements.
    #[must_use]
    pub fn add_edge(name: impl Into<String>, nodes: &[&str]) -> Self {
        let name = name.into();
        let fragment = format!("@{name} : {} {{ }}", nodes.join(", "));
        Self::append_source(name, fragment)
    }

    /// Stable planner-facing delta label.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    fn lower(&self, current_source: &str) -> Refinement {
        match &self.op {
            HiveDeltaOp::ReplaceSource(source) => Refinement(source.clone()),
            HiveDeltaOp::AppendSource(fragment) => {
                if current_source.trim().is_empty() {
                    Refinement(fragment.clone())
                } else {
                    Refinement(format!("{}\n{}", current_source.trim_end(), fragment))
                }
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum HiveDeltaOp {
    ReplaceSource(String),
    AppendSource(String),
}

/// Chooses the next HOTARU delta from the current cognitive context.
pub trait HotaruPlanner {
    /// Return the next candidate delta, or `None` when planning is exhausted.
    fn next_delta(&mut self, ctx: &CognitiveContext<'_>) -> Option<HiveDelta>;
}

/// Deterministic HOTARU stand-in for tests, demos, and benches.
#[derive(Debug, Clone, Default)]
pub struct ScriptedHotaru {
    queue: VecDeque<HiveDelta>,
}

impl ScriptedHotaru {
    /// Build a planner from a fixed delta sequence.
    pub fn new<I>(deltas: I) -> Self
    where
        I: IntoIterator<Item = HiveDelta>,
    {
        Self {
            queue: deltas.into_iter().collect(),
        }
    }

    /// Number of candidate deltas still queued.
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.queue.len()
    }
}

impl HotaruPlanner for ScriptedHotaru {
    fn next_delta(&mut self, _ctx: &CognitiveContext<'_>) -> Option<HiveDelta> {
        self.queue.pop_front()
    }
}

/// A HOTARU planner that *derives* its delta sequence by A\* search over the implicit HIVE-delta
/// space, instead of replaying a fixed script.
///
/// [`SearchHotaru::plan`] runs the [`SearchProblem`] search (via [`crate::search::solve`]) from a
/// seed source over a menu of candidate deltas: a node is an accumulated HyQL source, an edge is an
/// applied [`HiveDelta`] (cost one), a successor is kept only if it *parses* **and** respects the
/// [`Kyosei`] arity bound (the parser is the feasibility oracle, so ordering constraints such as
/// "the host block before an edge that references it" are enforced for free), and the goal is a
/// state whose edge names satisfy the [`Objectives`]. The planner then streams the resulting delta
/// sequence through [`HotaruPlanner::next_delta`], so it drops into the existing
/// [`HotaruSynthesizer`]/[`crate::CognitiveLoop`] unchanged — kicking the loop off with a *derived*
/// plan rather than a hand-written script.
#[derive(Debug, Clone, Default)]
pub struct SearchHotaru {
    plan: VecDeque<HiveDelta>,
}

impl SearchHotaru {
    /// Plan the shortest delta sequence from `seed` that reaches a state satisfying `objectives`,
    /// choosing from `menu`, exploring at most `max_expansions` nodes.
    ///
    /// # Preconditions
    /// `max_expansions >= 1`.
    ///
    /// # Postconditions
    /// `Some(planner)` whose queued deltas, applied in order from `seed`, reach a parseable state
    /// whose edge names satisfy `objectives` (and, the heuristic being admissible, via a
    /// minimum-length delta sequence); `None` if no such sequence exists within `max_expansions`
    /// (the caller then gets [`crate::Termination::Exhausted`] rather than a spinning loop).
    #[must_use]
    pub fn plan(
        seed: &str,
        menu: &[HiveDelta],
        objectives: &Objectives,
        kyosei: &Kyosei,
        max_expansions: usize,
    ) -> Option<Self> {
        Self::search(seed, menu, objectives, kyosei, max_expansions)
            .actions
            .map(|deltas| Self {
                plan: deltas.into_iter().collect(),
            })
    }

    /// The raw A\* search behind [`plan`](Self::plan): returns the full [`AstarResult`] (path +
    /// effort stats). Kept separate so the perf test can assert the deterministic expansion budget.
    fn search(
        seed: &str,
        menu: &[HiveDelta],
        objectives: &Objectives,
        kyosei: &Kyosei,
        max_expansions: usize,
    ) -> AstarResult<HiveDelta> {
        debug_assert!(max_expansions >= 1, "max_expansions must be >= 1");
        solve(
            &HiveDeltaProblem {
                seed,
                menu,
                objectives,
                kyosei,
            },
            max_expansions,
        )
    }

    /// Plan a *semantic graph plan*: a sequence of grounded `add_edge` deltas that transforms the model
    /// `seed` describes into one satisfying `goal`, choosing from `candidate_edges`.
    ///
    /// This is HOTARU planning over the HyMeKo-**described graph elements**: it reads the seed's nodes,
    /// admits only `(name, nodes)` candidates whose nodes are *actually declared* in the seed graph
    /// (grounding), and searches toward a topological [`GraphGoal`]. `candidate_edges` + `goal` are
    /// exactly the specification an LLM would produce from intent — HOTARU turns it into a verified plan.
    ///
    /// # Preconditions
    /// `max_expansions >= 1`.
    /// # Postconditions
    /// `Some(planner)` whose queued `add_edge` deltas, applied from `seed`, reach a graph satisfying
    /// `goal` (minimum length — the heuristic is admissible); `None` if `seed` does not parse or the
    /// goal is unreachable from the grounded candidates within `max_expansions`.
    #[must_use]
    pub fn plan_graph(
        seed: &str,
        candidate_edges: &[(&str, &[&str])],
        goal: &GraphGoal,
        kyosei: &Kyosei,
        max_expansions: usize,
    ) -> Option<Self> {
        debug_assert!(max_expansions >= 1, "max_expansions must be >= 1");
        let graph = preview_graph(seed)?; // the seed must be a parseable model
        let menu: Vec<HiveDelta> = candidate_edges
            .iter()
            .filter(|(_, nodes)| nodes.iter().all(|n| graph.has_node(n))) // ground against real nodes
            .map(|(name, nodes)| HiveDelta::add_edge(*name, nodes))
            .collect();
        solve(
            &GraphProblem {
                seed,
                menu: &menu,
                goal,
                kyosei,
            },
            max_expansions,
        )
        .actions
        .map(|deltas| Self {
            plan: deltas.into_iter().collect(),
        })
    }

    /// Number of planned deltas still to stream.
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.plan.len()
    }
}

impl HotaruPlanner for SearchHotaru {
    /// Streams the pre-computed plan; ignores the context (the plan is fixed by the search).
    fn next_delta(&mut self, _ctx: &CognitiveContext<'_>) -> Option<HiveDelta> {
        self.plan.pop_front()
    }
}

/// HOTARU's structure-synthesis search expressed as a [`SearchProblem`] (so it runs through the same
/// [`crate::search::solve`] framework the motion/grid problem uses). A node is an accumulated HyQL
/// source, an edge is an applied [`HiveDelta`] (cost one). A successor is kept only if it parses
/// **and** every edge respects the [`Kyosei`] arity bound (the parser's arity is the exact oracle —
/// not a heuristic); the goal is a committed state whose edge names satisfy the [`Objectives`]; the
/// heuristic is the admissible missing-edge count.
struct HiveDeltaProblem<'a> {
    seed: &'a str,
    menu: &'a [HiveDelta],
    objectives: &'a Objectives,
    kyosei: &'a Kyosei,
}

impl SearchProblem for HiveDeltaProblem<'_> {
    type Node = String;
    type Edge = HiveDelta;

    fn start(&self) -> String {
        self.seed.to_string()
    }

    fn neighbours(&self, src: &String) -> Vec<(HiveDelta, String, f64)> {
        feasible_successors(src, self.menu, self.kyosei)
    }

    fn is_goal(&self, src: &String) -> bool {
        match preview_edge_names(src) {
            Some(names) => !src.trim().is_empty() && self.objectives.satisfied_edges(&names),
            None => false,
        }
    }

    fn heuristic(&self, src: &String) -> f64 {
        preview_edge_names(src).map_or(self.objectives.required_edges.len() as f64, |names| {
            self.objectives.missing_edges(&names) as f64
        })
    }
}

/// Successors of a HIVE state: each menu delta applied, kept only if the result parses **and** every
/// edge is within the [`Kyosei`] arity bound. The shared feasibility relation for both the edge-name
/// planner ([`HiveDeltaProblem`]) and the graph-element planner ([`GraphProblem`]) — one oracle, no
/// duplicate DFS (CLAUDE.md §6.1).
fn feasible_successors(
    src: &str,
    menu: &[HiveDelta],
    kyosei: &Kyosei,
) -> Vec<(HiveDelta, String, f64)> {
    let mut succ = Vec::new();
    for delta in menu {
        let next = delta.lower(src).0;
        if next == *src {
            continue; // no-op delta
        }
        if let Some(edges) = preview_edges(&next) {
            if edges.iter().all(|(_, arity)| *arity <= kyosei.max_arity) {
                succ.push((delta.clone(), next, 1.0));
            }
        }
    }
    succ
}

/// A *semantic* goal over the graph a HyMeKo model describes (not over opaque source strings).
///
/// This is what makes HOTARU's output a semantic plan: it targets a property of the *graph elements*
/// (an edge by name, or a topological relation between nodes), which an LLM-produced specification can
/// express directly.
#[derive(Debug, Clone)]
pub enum GraphGoal {
    /// Every named edge must exist in the graph.
    RequiredEdges(Vec<String>),
    /// The two named nodes must lie in one connected component (via the hyperedges).
    Connect(String, String),
}

impl GraphGoal {
    /// Whether `graph` satisfies this goal.
    #[must_use]
    pub fn satisfied(&self, graph: &GraphView) -> bool {
        match self {
            GraphGoal::RequiredEdges(reqs) => reqs.iter().all(|r| graph.has_edge(r)),
            GraphGoal::Connect(a, b) => graph.connected(a, b),
        }
    }

    /// An admissible remaining-cost estimate: missing required edges, or 0/1 for a connectivity goal.
    #[must_use]
    fn heuristic(&self, graph: &GraphView) -> f64 {
        match self {
            GraphGoal::RequiredEdges(reqs) => {
                reqs.iter().filter(|r| !graph.has_edge(r)).count() as f64
            }
            GraphGoal::Connect(a, b) => {
                if graph.connected(a, b) {
                    0.0
                } else {
                    1.0
                }
            }
        }
    }
}

/// HOTARU's *graph-element* search as a [`SearchProblem`]: a node is an accumulated HyQL model, an edge
/// is a grounded `add_edge` [`HiveDelta`] (cost one). Successors are the arity-respecting parseable
/// applications of the menu (shared with [`HiveDeltaProblem`]); the goal is a topological property of
/// the *graph the source describes* ([`GraphGoal`]), read via [`preview_graph`].
struct GraphProblem<'a> {
    seed: &'a str,
    menu: &'a [HiveDelta],
    goal: &'a GraphGoal,
    kyosei: &'a Kyosei,
}

impl SearchProblem for GraphProblem<'_> {
    type Node = String;
    type Edge = HiveDelta;

    fn start(&self) -> String {
        self.seed.to_string()
    }

    fn neighbours(&self, src: &String) -> Vec<(HiveDelta, String, f64)> {
        feasible_successors(src, self.menu, self.kyosei)
    }

    fn is_goal(&self, src: &String) -> bool {
        preview_graph(src).is_some_and(|g| !src.trim().is_empty() && self.goal.satisfied(&g))
    }

    fn heuristic(&self, src: &String) -> f64 {
        preview_graph(src).map_or(1.0, |g| self.goal.heuristic(&g))
    }
}

#[cfg(test)]
mod search_tests {
    use super::*;

    /// The delta menu: a base host block, two wanted edges, one distractor. Edge deltas are
    /// `append_source`; base is `replace_source` (idempotent target).
    fn menu() -> Vec<HiveDelta> {
        vec![
            HiveDelta::replace_source("base", "Rig {\n  a;\n  b;\n  c;\n}"),
            HiveDelta::append_source("e_ab", "@e_ab : a, b { }"),
            HiveDelta::append_source("e_bc", "@e_bc : b, c { }"),
            HiveDelta::append_source("dist", "@dist : a, c { }"),
        ]
    }

    fn wants(edges: &[&str]) -> Objectives {
        Objectives {
            required_edges: edges.iter().map(|s| s.to_string()).collect(),
        }
    }

    /// Fold a delta sequence from `seed`, returning the final rendered source.
    fn apply_all(seed: &str, deltas: &[HiveDelta]) -> String {
        deltas
            .iter()
            .fold(seed.to_string(), |src, d| d.lower(&src).0)
    }

    #[test]
    fn feasibility_oracle_forces_base_before_edges() {
        // Measured (not assumed): the parser rejects a top-level edge with no host block, so the
        // search cannot place an edge before the base — ordering is enforced by the parse oracle.
        assert!(
            preview_edge_names("@e_ab : a, b { }").is_none(),
            "bare edge must not parse"
        );
        assert!(
            preview_edge_names("Rig {\n  a;\n  b;\n  c;\n}").is_some(),
            "base block parses"
        );
        let planner = SearchHotaru::plan(
            "",
            &menu(),
            &wants(&["e_ab", "e_bc"]),
            &Kyosei::default(),
            256,
        )
        .unwrap();
        assert_eq!(
            planner.plan.front().map(HiveDelta::name),
            Some("base"),
            "base is planned first"
        );
    }

    #[test]
    fn plan_reaches_goal_and_skips_distractor() {
        let m = menu();
        let obj = wants(&["e_ab", "e_bc"]);
        let planner =
            SearchHotaru::plan("", &m, &obj, &Kyosei::default(), 256).expect("goal is reachable");
        let deltas: Vec<&str> = planner.plan.iter().map(HiveDelta::name).collect();
        assert_eq!(
            deltas,
            ["base", "e_ab", "e_bc"],
            "optimal base-first plan, distractor skipped"
        );
        // Applying the plan from empty yields a parseable state with both required edges.
        let final_src = apply_all("", planner.plan.as_slices().0);
        let names = preview_edge_names(&final_src).expect("planned state parses");
        assert!(
            obj.satisfied_edges(&names),
            "planned state meets the objectives"
        );
    }

    #[test]
    fn search_meets_expansion_budget() {
        // Deterministic performance budget (CLAUDE.md §3): the two-edge goal is found within a
        // bounded number of node expansions, and the returned path is the 3-delta optimum.
        let r = SearchHotaru::search(
            "",
            &menu(),
            &wants(&["e_ab", "e_bc"]),
            &Kyosei::default(),
            256,
        );
        let actions = r.actions.expect("goal reachable");
        assert_eq!(actions.len(), 3, "optimal plan length");
        assert!(
            r.expansions <= 16,
            "expansions {} within budget 16",
            r.expansions
        );
        assert!(
            r.frontier_peak <= 16,
            "frontier peak {} within budget",
            r.frontier_peak
        );
    }

    #[test]
    fn plan_prefixes_are_all_feasible() {
        // A*'s feasibility oracle guarantees every intermediate state parses (the invariant that
        // makes "base before edge" hold when the grammar demands it).
        let m = menu();
        let obj = wants(&["e_ab", "e_bc"]);
        let planner = SearchHotaru::plan("", &m, &obj, &Kyosei::default(), 256).unwrap();
        let deltas: Vec<HiveDelta> = planner.plan.iter().cloned().collect();
        for k in 1..=deltas.len() {
            let prefix = apply_all("", &deltas[..k]);
            assert!(
                preview_edge_names(&prefix).is_some(),
                "prefix of length {k} must parse: {prefix:?}"
            );
        }
    }

    #[test]
    fn plan_none_when_goal_unreachable() {
        // No menu delta can produce an edge named "nope" ⇒ the search exhausts and returns None.
        let m = menu();
        let obj = wants(&["nope"]);
        assert!(SearchHotaru::plan("", &m, &obj, &Kyosei::default(), 256).is_none());
    }

    #[test]
    fn plan_respects_expansion_budget() {
        let m = menu();
        let obj = wants(&["e_ab", "e_bc"]);
        // A tiny budget cannot reach the two-edge goal ⇒ None (no spin, bounded work).
        assert!(SearchHotaru::plan("", &m, &obj, &Kyosei::default(), 1).is_none());
    }

    #[test]
    fn kyosei_arity_bound_prunes_high_arity_edges() {
        // The only route to the goal needs a 3-ary edge (`@e_abc : a, b, c`).
        let m = vec![
            HiveDelta::replace_source("base", "Rig {\n  a;\n  b;\n  c;\n}"),
            HiveDelta::append_source("e_abc", "@e_abc : a, b, c { }"),
        ];
        let obj = wants(&["e_abc"]);
        // max_arity 2 ⇒ the 3-ary edge is infeasible ⇒ goal unreachable (filter, not just heuristic).
        assert!(SearchHotaru::plan("", &m, &obj, &Kyosei { max_arity: 2 }, 256).is_none());
        // max_arity 3 ⇒ the edge is admissible ⇒ the goal is reached.
        let planner = SearchHotaru::plan("", &m, &obj, &Kyosei { max_arity: 3 }, 256).unwrap();
        assert_eq!(planner.remaining(), 2, "base, e_abc");
    }

    // ---- graph-element (semantic) planning --------------------------------

    const RIG: &str = "Rig {\n  a;\n  b;\n  c;\n}"; // three nodes, no edges

    #[test]
    fn preview_graph_extracts_nodes_edges_connectivity() {
        let g = preview_graph("Rig {\n  a;\n  b;\n  c;\n}\n@e_ab : a, b { }").unwrap();
        assert_eq!(g.nodes(), ["a", "b", "c"]);
        assert_eq!(g.edges().len(), 1);
        assert_eq!(g.edges()[0].name, "e_ab");
        assert_eq!(g.edges()[0].nodes, ["a", "b"]);
        assert!(g.connected("a", "b")); // joined by e_ab
        assert!(!g.connected("a", "c")); // c is isolated
        assert!(g.connected("c", "c")); // reflexive on a real node
        assert!(!g.connected("a", "zzz")); // unknown node
    }

    #[test]
    fn plan_graph_connects_two_nodes_via_minimal_edges() {
        // No direct a–c candidate ⇒ the plan must chain e_ab + e_bc to connect a and c.
        let cands: [(&str, &[&str]); 2] = [("e_ab", &["a", "b"]), ("e_bc", &["b", "c"])];
        let goal = GraphGoal::Connect("a".into(), "c".into());
        let planner = SearchHotaru::plan_graph(RIG, &cands, &goal, &Kyosei::default(), 256)
            .expect("a and c are connectable");
        let names: Vec<&str> = planner.plan.iter().map(HiveDelta::name).collect();
        assert_eq!(names, ["e_ab", "e_bc"]);
        let src = apply_all(RIG, planner.plan.as_slices().0);
        assert!(preview_graph(&src).unwrap().connected("a", "c")); // the plan achieves the goal
    }

    #[test]
    fn plan_graph_prefers_the_direct_edge() {
        // With a direct a–c candidate, the shortest semantic plan is that single edge.
        let cands: [(&str, &[&str]); 3] = [
            ("e_ab", &["a", "b"]),
            ("e_bc", &["b", "c"]),
            ("e_ac", &["a", "c"]),
        ];
        let goal = GraphGoal::Connect("a".into(), "c".into());
        let planner =
            SearchHotaru::plan_graph(RIG, &cands, &goal, &Kyosei::default(), 256).unwrap();
        assert_eq!(planner.remaining(), 1); // just e_ac
    }

    #[test]
    fn plan_graph_grounds_candidates_against_real_nodes() {
        // The only candidate references a node "zzz" absent from the model ⇒ grounded out ⇒ unreachable.
        let cands: [(&str, &[&str]); 1] = [("e_az", &["a", "zzz"])];
        let goal = GraphGoal::Connect("a".into(), "c".into());
        assert!(SearchHotaru::plan_graph(RIG, &cands, &goal, &Kyosei::default(), 256).is_none());
    }

    #[test]
    fn plan_graph_required_edges_goal() {
        let cands: [(&str, &[&str]); 2] = [("e_ab", &["a", "b"]), ("e_bc", &["b", "c"])];
        let goal = GraphGoal::RequiredEdges(vec!["e_bc".into()]);
        let planner =
            SearchHotaru::plan_graph(RIG, &cands, &goal, &Kyosei::default(), 256).unwrap();
        let names: Vec<&str> = planner.plan.iter().map(HiveDelta::name).collect();
        assert_eq!(names, ["e_bc"]);
    }

    #[test]
    fn plan_graph_unparseable_seed_is_none() {
        let cands: [(&str, &[&str]); 1] = [("e_ab", &["a", "b"])];
        let goal = GraphGoal::Connect("a".into(), "b".into());
        assert!(
            SearchHotaru::plan_graph("Rig { a }", &cands, &goal, &Kyosei::default(), 256).is_none()
        );
    }
}

/// AKOIRE synthesizer adapter for a HOTARU planner.
#[derive(Debug, Clone)]
pub struct HotaruSynthesizer<P: HotaruPlanner> {
    planner: P,
    last_delta_name: Option<String>,
}

impl<P: HotaruPlanner> HotaruSynthesizer<P> {
    /// Wrap a planner so it can drive [`crate::CognitiveLoop`].
    #[must_use]
    pub fn new(planner: P) -> Self {
        Self {
            planner,
            last_delta_name: None,
        }
    }

    /// Name of the most recently lowered delta, if any.
    #[must_use]
    pub fn last_delta_name(&self) -> Option<&str> {
        self.last_delta_name.as_deref()
    }
}

impl<P: HotaruPlanner> CognitiveSynthesizer for HotaruSynthesizer<P> {
    fn synthesize(&mut self, ctx: &CognitiveContext<'_>) -> Option<Refinement> {
        let delta = self.planner.next_delta(ctx)?;
        self.last_delta_name = Some(delta.name().to_string());
        Some(delta.lower(ctx.ambience.source()))
    }
}
