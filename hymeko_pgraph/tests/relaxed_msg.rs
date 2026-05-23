//! Tests for the relaxed-MSG option (Stage 2026-05-19, post-P-mo).
//!
//! When `MaximalStructureOptions::strict_no_excess = false`, MSG's
//! backward pass keeps a unit if *at least one* output is useful
//! (product or consumed by some surviving unit). The strict default
//! requires *every* output to be useful — the Friedler 1992 form.
//!
//! The textbook P-graph Studio chapter examples
//! (`data/pgraph/Chapter*/example*.pgip`) use the relaxed semantics
//! by default. Chapter4/example4_3 is the canonical case where
//! strict-MSG cascade-drops every unit while relaxed-MSG keeps the
//! 29-unit maximal structure.

use std::collections::BTreeSet;
use std::path::PathBuf;

use hymeko_pgraph::msg::{
    MaximalStructureOptions, maximal_structure, maximal_structure_with_options,
};
use hymeko_pgraph::abb::{AbbOptions, solve_with_options};
use hymeko_pgraph::lower;
use parser::parse_description;

fn fixture(name: &str) -> String {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let p = PathBuf::from(manifest)
        .parent()
        .unwrap()
        .join("data/pgraph")
        .join(name);
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("read {p:?}: {e}"))
}

#[test]
fn strict_msg_default_byte_identical() {
    // Pre-2026-05-19 callers used `maximal_structure(p)`. That entry
    // point must return the same result as
    // `maximal_structure_with_options(p, default)`.
    let src = fixture("hda.hymeko");
    let desc = parse_description(&src).unwrap();
    let p = lower(&desc).unwrap();
    let strict_default = maximal_structure(&p);
    let strict_explicit =
        maximal_structure_with_options(&p, MaximalStructureOptions::default());
    assert_eq!(strict_default.units, strict_explicit.units);
    assert_eq!(strict_default.materials, strict_explicit.materials);
}

#[test]
fn relaxed_keeps_more_or_equal_units() {
    // Mathematical invariant: the relaxed criterion is a *weakening*
    // of strict, so for any P-graph the relaxed MSG superset-equals
    // the strict MSG.
    let src = fixture("hda.hymeko");
    let desc = parse_description(&src).unwrap();
    let p = lower(&desc).unwrap();
    let strict = maximal_structure_with_options(
        &p,
        MaximalStructureOptions { strict_no_excess: true },
    );
    let relaxed = maximal_structure_with_options(
        &p,
        MaximalStructureOptions { strict_no_excess: false },
    );
    let s: BTreeSet<_> = strict.units.iter().collect();
    let r: BTreeSet<_> = relaxed.units.iter().collect();
    assert!(s.is_subset(&r),
            "strict MSG must be a subset of relaxed MSG; strict={:?} relaxed={:?}",
            strict.units, relaxed.units);
}

#[test]
fn chapter4_3_strict_collapses_relaxed_does_not() {
    // The Chapter 4 example 3 from P-graph Studio's textbook: 35
    // units. Under strict, MSG cascade-drops every unit (no strictly-
    // feasible structure exists). Under relaxed, MSG keeps 29 units.
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("data/pgraph/Chapter4/example4_3.hymeko");
    if !path.exists() {
        // The .hymeko is generated from the .pgip by
        // scripts/pgip_to_hymeko.py and committed alongside the
        // .pgip. If missing, skip.
        return;
    }
    let src = std::fs::read_to_string(&path).unwrap();
    let desc = parse_description(&src).unwrap();
    let p = lower(&desc).unwrap();

    let strict = maximal_structure_with_options(
        &p,
        MaximalStructureOptions { strict_no_excess: true },
    );
    let relaxed = maximal_structure_with_options(
        &p,
        MaximalStructureOptions { strict_no_excess: false },
    );

    assert_eq!(strict.units.len(), 0,
               "strict MSG must collapse on Chapter4_3; got {} units",
               strict.units.len());
    assert_eq!(relaxed.units.len(), 29,
               "relaxed MSG must keep 29 units on Chapter4_3; got {}",
               relaxed.units.len());
}

#[test]
fn abb_strict_chapter6_unchanged_at_18() {
    // The pre-2026-05-19 ABB result on Chapter6 (the costed twin of
    // Chapter3) was units {O1, O3, O6} at cost 18.0. The relaxed-MSG
    // refactor must preserve this byte-identical under the default
    // strict semantics.
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("data/pgraph/Chapter6/example6_1.hymeko");
    if !path.exists() {
        return;
    }
    let src = std::fs::read_to_string(&path).unwrap();
    let desc = parse_description(&src).unwrap();
    let p = lower(&desc).unwrap();
    let m = maximal_structure(&p); // strict default
    let abb = solve_with_options(&p, &m, AbbOptions::default())
        .expect("strict ABB must find a Chapter6 optimum");
    let names: BTreeSet<_> = abb
        .units
        .iter()
        .map(|d| p.decl_to_name[d].clone())
        .collect();
    let expected: BTreeSet<_> = ["O1", "O3", "O6"].iter().map(|s| s.to_string()).collect();
    assert_eq!(names, expected, "Chapter6 strict ABB units must be {{O1,O3,O6}}");
    assert!((abb.cost - 18.0).abs() < 1e-9, "Chapter6 ABB cost must be 18.0, got {}", abb.cost);
}
