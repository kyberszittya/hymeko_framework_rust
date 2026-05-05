//! Accelerated Branch-and-Bound (ABB) for cost-minimising synthesis.
//!
//! Given a P-graph, raw set, product set and per-unit costs $c(u)$,
//! ABB returns the feasible solution structure of minimum total cost.
//!
//! The "accelerated" Friedler ABB exploits structural lower bounds
//! drawn from the maximal structure to prune large parts of the
//! $2^{|O_{\max}|}$ subset lattice. The variant implemented here uses
//! two bounds:
//!
//! 1. **Inclusion bound** — once a partial selection has cost $\geq$
//!    the current best, abandon.
//! 2. **Reachability bound** — at every branch point we check that
//!    the *remaining* (still-undecided plus already-included) units
//!    suffice to produce every required product. If not, prune.
//!
//! Branching is by include / exclude on a fixed unit ordering. Ties
//! are broken in favour of including, so the search hits a feasible
//! incumbent quickly.

use std::collections::BTreeSet;

use hymeko::common::ids::DeclId;

use crate::lowering::LoweredPGraph;
use crate::msg::{close_producible, MaximalStructure};
use crate::ssg::{is_feasible, SsgOptions};

/// Output of [`solve`]: optimal selection plus its cost.
#[derive(Debug, Clone)]
pub struct AbbSolution {
    /// Operating units in the optimal solution.
    pub units: BTreeSet<DeclId>,
    /// Total cost.
    pub cost: f64,
    /// Number of nodes explored (diagnostic).
    pub explored: u64,
    /// Number of nodes pruned by the inclusion bound (cost ≥ incumbent).
    pub pruned_by_inclusion: u64,
    /// Number of nodes pruned by the reachability bound
    /// (`producible(R, optimistic) ⊉ products`).
    pub pruned_by_reachability: u64,
}

impl AbbSolution {
    /// Total prunes (inclusion + reachability) — back-compat shim.
    pub fn pruned(&self) -> u64 {
        self.pruned_by_inclusion + self.pruned_by_reachability
    }
}

/// ABB knobs.
#[derive(Debug, Clone, Copy)]
pub struct AbbOptions {
    /// Whether to enforce the strict no-excess P-graph rule
    /// (mirrors [`SsgOptions::strict_no_excess`]).
    pub strict_no_excess: bool,
    /// Hard cap on explored nodes (safety net for pathological
    /// inputs). 0 = unlimited.
    pub max_explored: u64,
}

impl Default for AbbOptions {
    fn default() -> Self {
        Self {
            strict_no_excess: true,
            max_explored: 1_000_000,
        }
    }
}

/// Solve the cost-minimising synthesis problem.
///
/// Returns `None` if no feasible solution exists.
pub fn solve(p: &LoweredPGraph, msg: &MaximalStructure) -> Option<AbbSolution> {
    solve_with_options(p, msg, AbbOptions::default())
}

/// Solve with explicit options.
pub fn solve_with_options(
    p: &LoweredPGraph,
    msg: &MaximalStructure,
    opts: AbbOptions,
) -> Option<AbbSolution> {
    let order: Vec<DeclId> = msg.units.iter().copied().collect();
    let mut state = SearchState {
        order,
        included: BTreeSet::new(),
        excluded: BTreeSet::new(),
        cost: 0.0,
        best: None,
        explored: 0,
        pruned_by_inclusion: 0,
        pruned_by_reachability: 0,
        opts,
    };
    branch(p, msg, &mut state, 0);
    state.best.map(|(units, cost)| AbbSolution {
        units,
        cost,
        explored: state.explored,
        pruned_by_inclusion: state.pruned_by_inclusion,
        pruned_by_reachability: state.pruned_by_reachability,
    })
}

struct SearchState {
    order: Vec<DeclId>,
    included: BTreeSet<DeclId>,
    excluded: BTreeSet<DeclId>,
    cost: f64,
    best: Option<(BTreeSet<DeclId>, f64)>,
    explored: u64,
    pruned_by_inclusion: u64,
    pruned_by_reachability: u64,
    opts: AbbOptions,
}

fn branch(
    p: &LoweredPGraph,
    msg: &MaximalStructure,
    s: &mut SearchState,
    depth: usize,
) {
    if s.opts.max_explored != 0 && s.explored >= s.opts.max_explored {
        return;
    }
    s.explored += 1;

    // ── Bound 1: inclusion bound.
    if let Some((_, best_cost)) = &s.best {
        if s.cost >= *best_cost {
            s.pruned_by_inclusion += 1;
            return;
        }
    }

    // ── Bound 2: reachability bound.
    //
    // The optimistic remaining-units set is `included ∪ undecided`.
    // If even with everything still on the table we can't produce
    // every required product, this branch is infeasible.
    let mut optimistic: BTreeSet<DeclId> = s.included.clone();
    for u in &s.order[depth..] {
        if !s.excluded.contains(u) {
            optimistic.insert(*u);
        }
    }
    let producible = close_producible(p, &optimistic, &p.raws);
    if !p.products.iter().all(|m| producible.contains(m)) {
        s.pruned_by_reachability += 1;
        return;
    }

    // ── Leaf: decide.
    if depth == s.order.len() {
        let opts = SsgOptions {
            strict_no_excess: s.opts.strict_no_excess,
            require_at_least_one_unit: false,
        };
        if is_feasible(p, &s.included, opts) {
            let candidate = (s.included.clone(), s.cost);
            match &s.best {
                None => s.best = Some(candidate),
                Some((_, bc)) if s.cost < *bc => s.best = Some(candidate),
                _ => {}
            }
        }
        return;
    }

    // ── Branch: include.
    let u = s.order[depth];
    let cu = p.costs.get(&u).copied().unwrap_or(1.0);
    s.included.insert(u);
    s.cost += cu;
    branch(p, msg, s, depth + 1);
    s.included.remove(&u);
    s.cost -= cu;

    // ── Branch: exclude.
    s.excluded.insert(u);
    branch(p, msg, s, depth + 1);
    s.excluded.remove(&u);
}
