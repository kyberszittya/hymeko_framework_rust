//! MDSD single-source-of-truth reuse regression.
//!
//! Proves that the ROS2 demo robot can be authored by *importing* the
//! canonical shared vocabularies (`data/robotics/meta_kinematics.hymeko` +
//! `data/robotics/meta_context.hymeko`) instead of re-declaring every type
//! inline, and that the imported form flows through the full kinematic
//! extraction + emission pipeline exactly like the WAM/DRC-Hubo fixtures
//! (see `test_imported_real.rs`).
//!
//! What this locks in:
//!   - The reused scenario compiles and lowers (cross-file import + `using`
//!     aliases for BOTH a node vocabulary and an edge vocabulary).
//!   - `extract_kinematic_model` recovers the 5-link / 4-revolute-joint
//!     backbone, with real axis vectors carried in from the imported axes.
//!   - URDF + SDF emit non-empty and contain `<joint>` elements — the
//!     semantic win over the bare-type baseline, whose generic `joint`
//!     type is not one of the four typed joint kinds the extractor queries.
//!   - The reused source is strictly shorter than the paper-faithful
//!     baseline (the MDSD length-reduction selling point), measured here so
//!     a future edit that re-inlines vocabulary fails the test.

#[cfg(test)]
mod test_mdsd_reuse {
    use crate::test_helpers::load_and_lower;
    use hymeko_formats::sdf::generate_sdf;
    use hymeko_formats::urdf::generate_urdf;
    use hymeko_query::engine::QueryEngine;
    use hymeko_query::kinematics::joints::JointType;
    use hymeko_query::kinematics::kinematic::extract_kinematic_model;

    const REUSE: &str =
        "../hymeko_ros2_demo/hymeko_ros2_demo/scenarios/hymeko_robot_reuse.hymeko";
    const BASELINE: &str =
        "../hymeko_ros2_demo/hymeko_ros2_demo/scenarios/hymeko_robot.hymeko";
    const ROBOT: &str = "hymeko_robot";

    #[test]
    fn reused_scenario_compiles_extracts_and_emits() {
        let (store, c) =
            load_and_lower(REUSE).unwrap_or_else(|e| panic!("compile {REUSE}: {e:?}"));

        let engine = QueryEngine::new(&c.ir, &store.it);
        let km = extract_kinematic_model(&engine, ROBOT);

        // 5 links, 4 revolute joints — the imported backbone.
        assert_eq!(km.links.len(), 5, "expected 5 links, got {}", km.links.len());
        let revolute: Vec<_> = km
            .joints
            .iter()
            .filter(|j| j.joint_type == JointType::Revolute)
            .collect();
        assert_eq!(
            revolute.len(),
            4,
            "expected 4 revolute joints, got {}",
            revolute.len()
        );

        // Axis vectors must have been carried in from the imported `axes`
        // profile (the bare-type baseline cannot express these).
        for j in &revolute {
            let ax = j
                .axis
                .unwrap_or_else(|| panic!("joint {} has no axis vector", j.name));
            let norm: f64 = ax.iter().map(|v| v * v).sum();
            assert!(
                (norm - 1.0).abs() < 1e-9,
                "joint {} axis {:?} is not a unit vector",
                j.name,
                ax
            );
        }

        // Emission: URDF + SDF non-empty and carry real <joint> elements.
        let urdf = generate_urdf(&c.ir, &store.it, ROBOT);
        let sdf = generate_sdf(&c.ir, &store.it, ROBOT);
        assert!(
            urdf.contains(&format!("<robot name=\"{ROBOT}\"")),
            "URDF missing robot header"
        );
        assert!(urdf.contains("<joint"), "URDF emitted no <joint> elements");
        assert!(!sdf.is_empty(), "SDF emit empty");
    }

    #[test]
    fn reused_scenario_is_shorter_than_baseline() {
        let reuse = std::fs::read_to_string(REUSE).expect("read reuse scenario");
        let baseline = std::fs::read_to_string(BASELINE).expect("read baseline scenario");
        let loc = |s: &str| s.lines().count();
        assert!(
            loc(&reuse) < loc(&baseline),
            "MDSD reuse ({} lines) should be shorter than baseline ({} lines)",
            loc(&reuse),
            loc(&baseline)
        );
    }
}
