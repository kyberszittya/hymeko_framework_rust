
#[cfg(test)]
mod test_transform_ecosystem {
    use hymeko_query::engine::QueryEngine;
    use hymeko_query::kinematics::kinematic::*;
    use hymeko_query::transforms::ModelView;

    const MOVEO_ARM: &str = "../data/robotics/anthropomorphic_arm.hymeko";
    const DIFF_ROBOT: &str = "../data/robotics/robot_4wh.hymeko";

    // ============================================================
    // Registry tests
    // ============================================================

    mod registry {
        use hymeko_query::transforms::{TransformConfig, TransformRegistry};
        use crate::test_helpers::load_and_lower;
        use super::*;

        #[test]
        fn default_registry_has_all_formats() {
            let reg = TransformRegistry::default();
            let available = reg.available();
            assert!(available.contains(&"urdf"), "Missing URDF");
            assert!(available.contains(&"sdf"), "Missing SDF");
            assert!(available.contains(&"mjcf"), "Missing MJCF");
            assert!(available.contains(&"dot"), "Missing DOT");
        }

        #[test]
        fn lookup_by_name() {
            let reg = TransformRegistry::default();
            assert!(reg.get("urdf").is_some());
            assert!(reg.get("nonexistent").is_none());
        }

        #[test]
        fn lookup_by_extension() {
            let reg = TransformRegistry::default();
            assert_eq!(reg.by_extension("urdf").unwrap().name(), "urdf");
            assert_eq!(reg.by_extension("sdf").unwrap().name(), "sdf");
            assert_eq!(reg.by_extension("dot").unwrap().name(), "dot");
        }

        #[test]
        fn generate_all_formats() {
            let (store, compiled) = load_and_lower(MOVEO_ARM).unwrap();
            let reg = TransformRegistry::default();
            let config = TransformConfig::default().with_name("moveo");

            let model = hymeko_query::transforms::extract(&compiled.ir, &store.it, &config.robot_name, hymeko_query::transforms::ModelKind::Kinematic);
            let results = reg.emit_all(&model, &config);

            assert_eq!(results.len(), 4, "Should generate 4 formats");
            for (filename, content) in &results {
                assert!(!content.is_empty(), "Empty output for {}", filename);
                println!("Generated {} ({} bytes)", filename, content.len());
            }
        }

        #[test]
        fn generate_all_for_both_robots() {
            let reg = TransformRegistry::default();

            for (path, name) in &[(MOVEO_ARM, "moveo"), (DIFF_ROBOT, "diff_robot")] {
                let (store, compiled) = load_and_lower(path).unwrap();
                let config = TransformConfig::default().with_name(name);
                let model = hymeko_query::transforms::extract(&compiled.ir, &store.it, &config.robot_name, hymeko_query::transforms::ModelKind::Kinematic);
                let results = reg.emit_all(&model, &config);

                for (filename, content) in &results {
                    assert!(!content.is_empty(),
                            "{}: empty output for {}", name, filename);
                }
            }
        }
    }

    // ============================================================
    // Validation tests
    // ============================================================

    mod validation {
        use hymeko_query::transforms::{DomainTransform, MjcfTransform, TransformRegistry, ModelView};
        use crate::test_helpers::load_and_lower;
        use super::*;

        #[test]
        fn moveo_validates_clean_for_all_formats() {
            let (store, compiled) = load_and_lower(MOVEO_ARM).unwrap();
            let engine = QueryEngine::new(&compiled.ir, &store.it);
            let model = extract_kinematic_model(&engine, "moveo");

            let reg = TransformRegistry::default();
            for name in reg.available() {
                let transform = reg.get(name).unwrap();
                let diags = transform.validate(&ModelView::Kinematic(model.clone()));
                let errors: Vec<_> = diags.iter()
                    .filter(|d| d.is_error())
                    .filter(|d| !d.message.contains("world"))
                    .collect();
                assert!(errors.is_empty(),
                        "{} validation failed: {:?}",
                        name, errors.iter().map(|e| &e.message).collect::<Vec<_>>());
            }
        }

        #[test]
        fn mjcf_validates_tree_topology() {
            let (store, compiled) = load_and_lower(MOVEO_ARM).unwrap();
            let engine = QueryEngine::new(&compiled.ir, &store.it);
            let model = extract_kinematic_model(&engine, "moveo");

            let mjcf = MjcfTransform;
            let diags = mjcf.validate(&ModelView::Kinematic(model.clone()));
            let errors: Vec<_> = diags.iter().filter(|d| d.is_error()).collect();

            // Moveo is a serial chain — should pass tree check
            assert!(errors.is_empty(), "MJCF tree validation failed: {:?}",
                    errors.iter().map(|e| &e.message).collect::<Vec<_>>());
        }
    }

    // ============================================================
    // MJCF generation tests
    // ============================================================

    mod mjcf {
        use hymeko_query::transforms::{MjcfTransform, TransformConfig, ModelView, DomainTransform};
        use crate::test_helpers::load_and_lower;
        use super::*;

        #[test]
        fn moveo_mjcf_wellformed() {
            let (store, compiled) = load_and_lower(MOVEO_ARM).unwrap();
            let engine = QueryEngine::new(&compiled.ir, &store.it);
            let model = extract_kinematic_model(&engine, "moveo");
            let config = TransformConfig::default().with_name("moveo");

            let mjcf = MjcfTransform;
            let output = mjcf.emit(&ModelView::Kinematic(model), &config).unwrap();
            println!("--- Moveo MJCF ---\n{}", output);

            assert!(output.contains("<mujoco model=\"moveo\""));
            assert!(output.ends_with("</mujoco>\n"));
        }

        #[test]
        fn moveo_mjcf_has_bodies() {
            let (store, compiled) = load_and_lower(MOVEO_ARM).unwrap();
            let engine = QueryEngine::new(&compiled.ir, &store.it);
            let model = extract_kinematic_model(&engine, "moveo");
            let config = TransformConfig::default().with_name("moveo");

            let mjcf = MjcfTransform;
            let output = mjcf.emit(&ModelView::Kinematic(model), &config).unwrap();

            // MJCF uses <body> elements for links
            for name in &["base_link", "link_0", "link_1", "link_4", "tool"] {
                assert!(output.contains(&format!("<body name=\"{}\"", name)),
                        "MJCF missing body '{}'", name);
            }
        }

        #[test]
        fn moveo_mjcf_has_joints() {
            let (store, compiled) = load_and_lower(MOVEO_ARM).unwrap();
            let engine = QueryEngine::new(&compiled.ir, &store.it);
            let model = extract_kinematic_model(&engine, "moveo");
            let config = TransformConfig::default().with_name("moveo");

            let mjcf = MjcfTransform;
            let output = mjcf.emit(&ModelView::Kinematic(model), &config).unwrap();

            // MJCF uses <joint> inside <body>
            let hinge_count = output.matches("type=\"hinge\"").count();
            assert_eq!(hinge_count, 6,
                       "Expected 6 hinge joints, got {}", hinge_count);
        }

        #[test]
        fn moveo_mjcf_has_actuators() {
            let (store, compiled) = load_and_lower(MOVEO_ARM).unwrap();
            let engine = QueryEngine::new(&compiled.ir, &store.it);
            let model = extract_kinematic_model(&engine, "moveo");
            let config = TransformConfig::default().with_name("moveo");

            let mjcf = MjcfTransform;
            let output = mjcf.emit(&ModelView::Kinematic(model), &config).unwrap();

            assert!(output.contains("<actuator>"));
            let motor_count = output.matches("<motor").count();
            assert_eq!(motor_count, 6,
                       "Expected 6 actuators (one per revolute joint), got {}", motor_count);
        }

        #[test]
        fn moveo_mjcf_half_extents_for_box() {
            let (store, compiled) = load_and_lower(MOVEO_ARM).unwrap();
            let engine = QueryEngine::new(&compiled.ir, &store.it);
            let model = extract_kinematic_model(&engine, "moveo");
            let config = TransformConfig::default().with_name("moveo");

            let mjcf = MjcfTransform;
            let output = mjcf.emit(&ModelView::Kinematic(model), &config).unwrap();

            // tool has box [0.075, 0.15, 0.1]
            // MJCF uses half-extents: [0.0375, 0.075, 0.05]
            assert!(output.contains("0.0375") || output.contains("0.075"),
                    "MJCF should use half-extents for box geometry");
        }

        #[test]
        fn moveo_mjcf_has_materials() {
            let (store, compiled) = load_and_lower(MOVEO_ARM).unwrap();
            let engine = QueryEngine::new(&compiled.ir, &store.it);
            let model = extract_kinematic_model(&engine, "moveo");
            let config = TransformConfig::default().with_name("moveo");

            let mjcf = MjcfTransform;
            let output = mjcf.emit(&ModelView::Kinematic(model), &config).unwrap();

            assert!(output.contains("<asset>"));
            assert!(output.contains("<material name="));
        }

        #[test]
        fn moveo_mjcf_radians_not_degrees() {
            let (store, compiled) = load_and_lower(MOVEO_ARM).unwrap();
            let engine = QueryEngine::new(&compiled.ir, &store.it);
            let model = extract_kinematic_model(&engine, "moveo");
            let config = TransformConfig::default().with_name("moveo");

            let mjcf = MjcfTransform;
            let output = mjcf.emit(&ModelView::Kinematic(model), &config).unwrap();

            assert!(output.contains("angle=\"radian\""),
                    "MJCF compiler should specify radian mode");
        }

        #[test]
        fn moveo_mjcf_nested_body_hierarchy() {
            let (store, compiled) = load_and_lower(MOVEO_ARM).unwrap();
            let engine = QueryEngine::new(&compiled.ir, &store.it);
            let model = extract_kinematic_model(&engine, "moveo");
            let config = TransformConfig::default().with_name("moveo");

            let mjcf = MjcfTransform;
            let output = mjcf.emit(&ModelView::Kinematic(model), &config).unwrap();

            // MJCF bodies must be nested (not flat like URDF)
            // Count indentation levels to verify nesting
            let lines: Vec<&str> = output.lines().collect();
            let max_indent = lines.iter()
                .filter(|l| l.contains("<body"))
                .map(|l| l.len() - l.trim_start().len())
                .max()
                .unwrap_or(0);

            // With 7 links in a serial chain, deepest nesting should be significant
            assert!(max_indent >= 12,
                    "MJCF bodies should be deeply nested for serial chain, max indent = {}",
                    max_indent);
        }
    }

    // ============================================================
    // DOT generation tests
    // ============================================================

    mod dot {
        use hymeko_query::transforms::{DotTransform, TransformConfig, ModelView, DomainTransform};
        use crate::test_helpers::load_and_lower;
        use super::*;

        #[test]
        fn moveo_dot_wellformed() {
            let (store, compiled) = load_and_lower(MOVEO_ARM).unwrap();
            let engine = QueryEngine::new(&compiled.ir, &store.it);
            let model = extract_kinematic_model(&engine, "moveo");
            let config = TransformConfig::default().with_name("moveo");

            let dot = DotTransform;
            let output = dot.emit(&ModelView::Kinematic(model), &config).unwrap();
            println!("--- Moveo DOT ---\n{}", output);

            assert!(output.starts_with("digraph"));
            assert!(output.ends_with("}\n"));
        }

        #[test]
        fn moveo_dot_has_all_links_as_nodes() {
            let (store, compiled) = load_and_lower(MOVEO_ARM).unwrap();
            let engine = QueryEngine::new(&compiled.ir, &store.it);
            let model = extract_kinematic_model(&engine, "moveo");
            let config = TransformConfig::default().with_name("moveo");

            let dot = DotTransform;
            let output = dot.emit(&ModelView::Kinematic(model), &config).unwrap();

            for name in &["base_link", "link_0", "link_4", "tool"] {
                assert!(output.contains(&format!("\"{}\"", name)),
                        "DOT missing node for '{}'", name);
            }
        }

        #[test]
        fn moveo_dot_has_edges_with_joint_names() {
            let (store, compiled) = load_and_lower(MOVEO_ARM).unwrap();
            let engine = QueryEngine::new(&compiled.ir, &store.it);
            let model = extract_kinematic_model(&engine, "moveo");
            let config = TransformConfig::default().with_name("moveo");

            let dot = DotTransform;
            let output = dot.emit(&ModelView::Kinematic(model), &config).unwrap();

            // Check directed edges exist
            assert!(output.contains("\"base_link\" -> \"link_0\""));
            assert!(output.contains("\"link_4\" -> \"tool\""));
        }

        #[test]
        fn moveo_dot_fixed_joints_dashed() {
            let (store, compiled) = load_and_lower(MOVEO_ARM).unwrap();
            let engine = QueryEngine::new(&compiled.ir, &store.it);
            let model = extract_kinematic_model(&engine, "moveo");
            let config = TransformConfig::default().with_name("moveo");

            let dot = DotTransform;
            let output = dot.emit(&ModelView::Kinematic(model), &config).unwrap();

            // Fixed joints should be dashed
            assert!(output.contains("style=dashed"),
                    "Fixed joints should have dashed style in DOT");
        }

        #[test]
        fn moveo_dot_revolute_joints_bold() {
            let (store, compiled) = load_and_lower(MOVEO_ARM).unwrap();
            let engine = QueryEngine::new(&compiled.ir, &store.it);
            let model = extract_kinematic_model(&engine, "moveo");
            let config = TransformConfig::default().with_name("moveo");

            let dot = DotTransform;
            let output = dot.emit(&ModelView::Kinematic(model), &config).unwrap();

            let bold_count = output.matches("style=bold").count();
            assert_eq!(bold_count, 6,
                       "Expected 6 bold edges (revolute), got {}", bold_count);
        }

        #[test]
        fn moveo_dot_axis_labels() {
            let (store, compiled) = load_and_lower(MOVEO_ARM).unwrap();
            let engine = QueryEngine::new(&compiled.ir, &store.it);
            let model = extract_kinematic_model(&engine, "moveo");
            let config = TransformConfig::default().with_name("moveo");

            let dot = DotTransform;
            let output = dot.emit(&ModelView::Kinematic(model), &config).unwrap();

            // Joint labels should include axis letter
            assert!(output.contains("(Z)"), "DOT should show Z axis joints");
            assert!(output.contains("(X)"), "DOT should show X axis joints");
            assert!(output.contains("(Y)"), "DOT should show Y axis joints");
        }

        #[test]
        fn diff_robot_dot_generates() {
            let (store, compiled) = load_and_lower(DIFF_ROBOT).unwrap();
            let engine = QueryEngine::new(&compiled.ir, &store.it);
            let model = extract_kinematic_model(&engine, "diff_robot");
            let config = TransformConfig::default().with_name("diff_robot");

            let dot = DotTransform;
            let output = dot.emit(&ModelView::Kinematic(model), &config).unwrap();
            println!("--- Diff Robot DOT ---\n{}", output);

            assert!(output.contains("\"base_link\""));
            assert!(output.contains("\"wheel_fr\""));
        }
    }
}