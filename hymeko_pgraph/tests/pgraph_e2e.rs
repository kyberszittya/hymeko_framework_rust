//! End-to-end: parse a P-graph written in HyMeKo grammar, lower it,
//! then run MSG / SSG / ABB.

use std::collections::BTreeSet;

use hymeko_pgraph::{AbbOptions, SsgOptions, abb_solve, lower, maximal_structure, ssg_enumerate};
use parser::parse_description;

const HDA_SRC: &str = include_str!("../../data/pgraph/hda.hymeko");

#[test]
fn parses_hda_pgraph() {
    let d = parse_description(HDA_SRC).expect("parser must accept HDA file");
    assert_eq!(d.name, "HDA");
    // The lowering walks the `context { ... }` wrapper one level.
    let p = lower(&d).expect("lowering must succeed");
    // 5 materials.
    assert_eq!(p.materials.len(), 5);
    // 4 operating units.
    assert_eq!(p.units.len(), 4);
    // Raws: Toluene, H2.
    let raw_names: BTreeSet<&str> = p.raws.iter().map(|d| p.decl_to_name[d].as_str()).collect();
    assert_eq!(raw_names, BTreeSet::from(["Toluene", "H2"]));
    // Required products: Benzene.
    let prod_names: BTreeSet<&str> = p
        .products
        .iter()
        .map(|d| p.decl_to_name[d].as_str())
        .collect();
    assert_eq!(prod_names, BTreeSet::from(["Benzene"]));
}

#[test]
fn unit_signatures_lower_correctly() {
    let d = parse_description(HDA_SRC).unwrap();
    let p = lower(&d).unwrap();

    let mixer = p.name_to_decl["Mixer"];
    let reactor = p.name_to_decl["Reactor"];
    let direct = p.name_to_decl["DirectSynth"];
    let disposal = p.name_to_decl["Disposal"];
    let toluene = p.name_to_decl["Toluene"];
    let h2 = p.name_to_decl["H2"];
    let mix = p.name_to_decl["Mix"];
    let benzene = p.name_to_decl["Benzene"];
    let methane = p.name_to_decl["Methane"];

    assert_eq!(p.unit_inputs[&mixer], BTreeSet::from([toluene, h2]));
    assert_eq!(p.unit_outputs[&mixer], BTreeSet::from([mix]));

    assert_eq!(p.unit_inputs[&reactor], BTreeSet::from([mix]));
    assert_eq!(p.unit_outputs[&reactor], BTreeSet::from([benzene, methane]));

    assert_eq!(p.unit_inputs[&direct], BTreeSet::from([toluene, h2]));
    assert_eq!(p.unit_outputs[&direct], BTreeSet::from([benzene]));

    assert_eq!(p.unit_inputs[&disposal], BTreeSet::from([methane]));
    assert!(p.unit_outputs[&disposal].is_empty());

    assert_eq!(p.costs[&mixer], 100.0);
    assert_eq!(p.costs[&reactor], 250.0);
    assert_eq!(p.costs[&direct], 800.0);
    assert_eq!(p.costs[&disposal], 50.0);
}

#[test]
fn msg_keeps_every_unit_for_hda() {
    // Every operating unit in the HDA file is forward+backward
    // feasible, so MSG should preserve all four.
    let d = parse_description(HDA_SRC).unwrap();
    let p = lower(&d).unwrap();
    let m = maximal_structure(&p);
    assert_eq!(m.units, p.units, "MSG must preserve every unit");
    assert!(m.materials.contains(&p.name_to_decl["Benzene"]));
}

#[test]
fn msg_drops_a_forward_unreachable_unit() {
    // A unit whose input is *not* producible by any chain from the
    // raw set must be dropped by the forward pass.
    let bad = r#"
        Bad{}
        context {
            R <material, raw>;
            P <material, product>;
            X <material>;       // never produced, never raw
            @U_good <unit> {
                (-R, +P);
            }
            @U_bad <unit> {
                (-X, +P);       // input X is unreachable
            }
        }
    "#;
    let d = parse_description(bad).unwrap();
    let p = lower(&d).unwrap();
    let m = maximal_structure(&p);
    assert!(m.units.contains(&p.name_to_decl["U_good"]));
    assert!(
        !m.units.contains(&p.name_to_decl["U_bad"]),
        "U_bad consumes X which no chain from raws produces"
    );
}

#[test]
fn msg_drops_a_backward_useless_unit() {
    // A unit whose output is neither product nor consumed by any
    // other unit must be dropped by the backward pass.
    let bad = r#"
        Bad{}
        context {
            R <material, raw>;
            P <material, product>;
            Junk <material>;
            @U_good <unit> {
                (-R, +P);
            }
            @U_useless <unit> {
                (-R, +Junk);    // Junk has no consumer
            }
        }
    "#;
    let d = parse_description(bad).unwrap();
    let p = lower(&d).unwrap();
    let m = maximal_structure(&p);
    assert!(m.units.contains(&p.name_to_decl["U_good"]));
    assert!(
        !m.units.contains(&p.name_to_decl["U_useless"]),
        "U_useless produces a dead-end material — MSG drops it"
    );
}

#[test]
fn ssg_finds_known_feasible_structures() {
    let d = parse_description(HDA_SRC).unwrap();
    let p = lower(&d).unwrap();
    let m = maximal_structure(&p);

    // SSG with strict no-excess: every produced non-product must be
    // consumed.  Methane is produced by Reactor but it is neither raw
    // nor a product — so any structure containing Reactor *must*
    // also contain Disposal.
    let solutions = ssg_enumerate(&p, &m);

    let mixer = p.name_to_decl["Mixer"];
    let reactor = p.name_to_decl["Reactor"];
    let direct = p.name_to_decl["DirectSynth"];
    let disposal = p.name_to_decl["Disposal"];

    let by_units = |us: &[_]| {
        let s: BTreeSet<_> = us.iter().copied().collect();
        solutions.iter().any(|sol| sol.units == s)
    };

    // Two-stage route with disposal — feasible.
    assert!(
        by_units(&[mixer, reactor, disposal]),
        "Mixer + Reactor + Disposal must be feasible (strict)"
    );
    // One-shot direct synthesis — feasible (no methane produced).
    assert!(by_units(&[direct]), "DirectSynth alone must be feasible");
    // Two-stage WITHOUT disposal — infeasible under strict rule.
    assert!(
        !by_units(&[mixer, reactor]),
        "Mixer + Reactor without Disposal violates strict no-excess"
    );
    // Disposal alone — infeasible (no Benzene produced).
    assert!(!by_units(&[disposal]));
}

#[test]
fn ssg_relaxed_includes_excess_byproduct() {
    let d = parse_description(HDA_SRC).unwrap();
    let p = lower(&d).unwrap();
    let m = maximal_structure(&p);

    let opts = SsgOptions {
        strict_no_excess: false,
        require_at_least_one_unit: true,
    };
    let solutions = hymeko_pgraph::ssg::enumerate_with_options(&p, &m, opts);

    let mixer = p.name_to_decl["Mixer"];
    let reactor = p.name_to_decl["Reactor"];

    let pair = BTreeSet::from([mixer, reactor]);
    assert!(
        solutions.iter().any(|s| s.units == pair),
        "relaxed SSG must accept Mixer+Reactor with excess Methane"
    );
}

#[test]
fn abb_finds_minimum_cost_route() {
    let d = parse_description(HDA_SRC).unwrap();
    let p = lower(&d).unwrap();
    let m = maximal_structure(&p);

    let mixer = p.name_to_decl["Mixer"];
    let reactor = p.name_to_decl["Reactor"];
    let direct = p.name_to_decl["DirectSynth"];
    let disposal = p.name_to_decl["Disposal"];

    let sol = abb_solve(&p, &m).expect("ABB must find a solution");

    // Strict path costs:
    //   {Mixer, Reactor, Disposal} = 100 + 250 + 50 = 400
    //   {DirectSynth}              = 800
    // Optimum = the two-stage route.
    assert_eq!(
        sol.units,
        BTreeSet::from([mixer, reactor, disposal]),
        "expected the {{Mixer, Reactor, Disposal}} route"
    );
    assert!(
        (sol.cost - 400.0).abs() < 1e-9,
        "minimum cost is 400, got {}",
        sol.cost
    );

    // Negate the strict rule: with excess byproducts allowed,
    // {Mixer, Reactor} (350) becomes feasible and beats 400.
    let relaxed = hymeko_pgraph::abb::solve_with_options(
        &p,
        &m,
        AbbOptions {
            strict_no_excess: false,
            max_explored: 10_000,
        },
    )
    .expect("relaxed ABB must find a solution");
    assert_eq!(relaxed.units, BTreeSet::from([mixer, reactor]));
    assert!((relaxed.cost - 350.0).abs() < 1e-9);

    // ABB diagnostics.
    assert!(sol.explored > 0);

    // Sanity: DirectSynth should never be the minimum.
    assert!(!sol.units.contains(&direct));
}

#[test]
fn abb_returns_none_when_infeasible() {
    // Same file but with the products demand redirected to a material
    // that no unit can produce.  Lowering doesn't enforce reachability
    // — that's MSG / SSG / ABB's job.
    let bad = r#"
        Bad{}
        context {
            X <material, raw>;
            Z <material, product>;
        }
    "#;
    let d = parse_description(bad).expect("parses");
    let p = lower(&d).expect("lowers");
    let m = maximal_structure(&p);
    let sol = abb_solve(&p, &m);
    assert!(
        sol.is_none(),
        "no unit produces Z, so ABB must report infeasible"
    );
}
