//! Serialize MSG / SSG / ABB results for tooling (Python drivers, CI).
//!
//! [`analyze_source`] parses HyMeKo P-graph text, lowers it, runs MSG,
//! then optionally full SSG or cost-minimal ABB.  SSG is omitted when
//! `|O_MSG| > 30` (see [`ssg::enumerate_with_options`]).
#![allow(missing_docs)] // JSON DTO field names are self-describing for tooling.

use std::str::FromStr;

use serde::Serialize;

use crate::abb::{AbbOptions, AbbSolution, solve_with_options};
use crate::lowering::{LoweredPGraph, lower};
use crate::msg::maximal_structure;
use crate::ssg::{SolutionStructure, SsgOptions, enumerate_with_options};
use parser::parse_description;

/// Which analysis stages to emit beyond MSG.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DumpAlgorithm {
    /// Maximal structure (unit id set) only.
    Msg,
    /// All combinatorially feasible solution structures inside MSG.
    Ssg,
    /// Cost-minimal feasible structure (ABB).
    Abb,
}

/// Parse CLI / config spelling: `msg`, `ssg`, `abb`.
impl FromStr for DumpAlgorithm {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "msg" => Ok(DumpAlgorithm::Msg),
            "ssg" => Ok(DumpAlgorithm::Ssg),
            "abb" => Ok(DumpAlgorithm::Abb),
            other => Err(format!("unknown algorithm `{other}` (use msg|ssg|abb)")),
        }
    }
}

#[derive(Debug, Serialize)]
pub struct AbbJson {
    /// Operating unit names in the optimal solution.
    pub units: Vec<String>,
    pub cost: f64,
    pub explored: u64,
    pub pruned_by_inclusion: u64,
    pub pruned_by_reachability: u64,
}

#[derive(Debug, Serialize)]
pub struct PgraphAnalysisJson {
    /// True when parse + lower + MSG succeeded.
    pub ok: bool,
    pub description: String,
    pub algorithm: String,
    pub parse_error: Option<String>,
    pub lower_error: Option<String>,
    /// Unit names in the maximal structure (post-MSG), sorted.
    pub msg_units: Vec<String>,
    /// Feasible SSG structures as name lists (sorted within each structure).
    pub ssg_structures: Option<Vec<Vec<String>>>,
    /// Set when SSG was skipped (e.g. MSG too large).
    pub ssg_note: Option<String>,
    pub abb: Option<AbbJson>,
}

fn unit_names(p: &LoweredPGraph, sol: &SolutionStructure) -> Vec<String> {
    let mut names: Vec<String> = sol
        .units
        .iter()
        .map(|id| p.decl_to_name[id].clone())
        .collect();
    names.sort();
    names
}

fn abb_to_json(p: &LoweredPGraph, abb: &AbbSolution) -> AbbJson {
    let mut names: Vec<String> = abb
        .units
        .iter()
        .map(|id| p.decl_to_name[id].clone())
        .collect();
    names.sort();
    AbbJson {
        units: names,
        cost: abb.cost,
        explored: abb.explored,
        pruned_by_inclusion: abb.pruned_by_inclusion,
        pruned_by_reachability: abb.pruned_by_reachability,
    }
}

/// Parse `hymeko_src`, lower to a P-graph, run MSG, then `algorithm` stages.
pub fn analyze_source(hymeko_src: &str, algorithm: DumpAlgorithm) -> PgraphAnalysisJson {
    let algo_label = match algorithm {
        DumpAlgorithm::Msg => "msg",
        DumpAlgorithm::Ssg => "ssg",
        DumpAlgorithm::Abb => "abb",
    };
    let desc = parse_description(hymeko_src);
    let d = match desc {
        Ok(d) => d,
        Err(e) => {
            return PgraphAnalysisJson {
                ok: false,
                description: String::new(),
                algorithm: algo_label.into(),
                parse_error: Some(format!("{e:?}")),
                lower_error: None,
                msg_units: vec![],
                ssg_structures: None,
                ssg_note: None,
                abb: None,
            };
        }
    };
    let description = d.name.to_string();
    let p = match lower(&d) {
        Ok(p) => p,
        Err(e) => {
            return PgraphAnalysisJson {
                ok: false,
                description,
                algorithm: algo_label.into(),
                parse_error: None,
                lower_error: Some(e.to_string()),
                msg_units: vec![],
                ssg_structures: None,
                ssg_note: None,
                abb: None,
            };
        }
    };

    let m = maximal_structure(&p);
    let mut msg_units: Vec<String> = m
        .units
        .iter()
        .map(|id| p.decl_to_name[id].clone())
        .collect();
    msg_units.sort();

    let mut out = PgraphAnalysisJson {
        ok: true,
        description,
        algorithm: algo_label.into(),
        parse_error: None,
        lower_error: None,
        msg_units,
        ssg_structures: None,
        ssg_note: None,
        abb: None,
    };

    let n = m.units.len();
    match algorithm {
        DumpAlgorithm::Msg => {}
        DumpAlgorithm::Ssg => {
            if n > 30 {
                out.ssg_note = Some(format!(
                    "MSG has {n} units (>30); SSG exponential enumeration omitted"
                ));
                out.ssg_structures = Some(vec![]);
            } else {
                let sols = enumerate_with_options(&p, &m, SsgOptions::default());
                let structures: Vec<Vec<String>> = sols.iter().map(|s| unit_names(&p, s)).collect();
                out.ssg_structures = Some(structures);
            }
        }
        DumpAlgorithm::Abb => {
            let msg = maximal_structure(&p);
            if let Some(abb) = solve_with_options(&p, &msg, AbbOptions::default()) {
                out.abb = Some(abb_to_json(&p, &abb));
            }
        }
    }

    out
}
