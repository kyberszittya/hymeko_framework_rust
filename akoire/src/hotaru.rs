//! HOTARU planning over HIVE-delta space.
//!
//! HOTARU does not commit structure directly. It proposes named deltas over the
//! current HIVE state, and this adapter lowers each delta to a HyQL refinement
//! that must still pass through AKOIRE's gatekeeper.

use std::collections::VecDeque;

use crate::context::{CognitiveContext, Objectives};
use crate::engine::preview_edge_names;
use crate::search::{astar, AstarResult};
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
/// [`SearchHotaru::plan`] runs [`crate::search::astar`] from a seed source over a menu of candidate
/// deltas: a node is an accumulated HyQL source, an edge is an applied [`HiveDelta`] (cost one), a
/// successor is kept only if it *parses* (via [`preview_edge_names`] — the parser is the feasibility
/// oracle, so ordering constraints such as "the host block before an edge that references it" are
/// enforced for free), and the goal is a state whose edge names satisfy the [`Objectives`]. The
/// planner then streams the resulting delta sequence through [`HotaruPlanner::next_delta`], so it
/// drops into the existing [`HotaruSynthesizer`]/[`crate::CognitiveLoop`] unchanged — kicking the
/// loop off with a *derived* plan rather than a hand-written script.
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
        max_expansions: usize,
    ) -> Option<Self> {
        Self::search(seed, menu, objectives, max_expansions)
            .actions
            .map(|deltas| Self {
                plan: deltas.into_iter().collect(),
            })
    }

    /// The raw A\* search behind [`plan`](Self::plan): returns the full [`AstarResult`] (path +
    /// effort stats). Kept separate so the perf test can assert the deterministic expansion budget
    /// without a second copy of the neighbour/goal/heuristic closures.
    fn search(
        seed: &str,
        menu: &[HiveDelta],
        objectives: &Objectives,
        max_expansions: usize,
    ) -> AstarResult<HiveDelta> {
        debug_assert!(max_expansions >= 1, "max_expansions must be >= 1");
        astar(
            seed.to_string(),
            // neighbours: apply each delta; keep only distinct, parseable successors.
            |src: &String| {
                let mut succ: Vec<(HiveDelta, String, f64)> = Vec::new();
                for delta in menu {
                    let next = delta.lower(src).0;
                    if next != *src && preview_edge_names(&next).is_some() {
                        succ.push((delta.clone(), next, 1.0));
                    }
                }
                succ
            },
            // is_goal: a committed (non-empty, parseable) state with all required edges present.
            |src: &String| match preview_edge_names(src) {
                Some(names) => !src.trim().is_empty() && objectives.satisfied_edges(&names),
                None => false,
            },
            // heuristic: required edges still missing (admissible — see `Objectives::missing_edges`).
            |src: &String| {
                preview_edge_names(src).map_or(objectives.required_edges.len() as f64, |names| {
                    objectives.missing_edges(&names) as f64
                })
            },
            max_expansions,
        )
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
        let planner = SearchHotaru::plan("", &menu(), &wants(&["e_ab", "e_bc"]), 256).unwrap();
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
        let planner = SearchHotaru::plan("", &m, &obj, 256).expect("goal is reachable");
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
        let r = SearchHotaru::search("", &menu(), &wants(&["e_ab", "e_bc"]), 256);
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
        let planner = SearchHotaru::plan("", &m, &obj, 256).unwrap();
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
        assert!(SearchHotaru::plan("", &m, &obj, 256).is_none());
    }

    #[test]
    fn plan_respects_expansion_budget() {
        let m = menu();
        let obj = wants(&["e_ab", "e_bc"]);
        // A tiny budget cannot reach the two-edge goal ⇒ None (no spin, bounded work).
        assert!(SearchHotaru::plan("", &m, &obj, 1).is_none());
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
