# T13–T16 — Tests

**Status:** ❌ NOT DONE  
**Priority:** CRITICAL — test output populates Paper 2 Table II  
**Estimated:** ~250 lines across 4 files

---

## Setup

### `hymeko_core/tests/query/mod.rs`

```rust
mod test_query_meta;
mod test_query_robot;
mod test_urdf;
```

### `hymeko_core/tests/mod.rs` — add at the end:

```rust
mod query;
```

---

## T13 — Meta-Kinematics Schema Queries

**File:** `hymeko_core/tests/query/test_query_meta.rs`

```rust
#[cfg(test)]
mod test_query_meta {
    use hymeko::query::engine::QueryEngine;
    use hymeko::query::predicate::*;
    use crate::test_helpers::load_and_lower;

    const META: &str = "./data/robotics/meta_kinematics.hymeko";

    #[test]
    fn find_all_nodes() {
        let (store, compiled) = load_and_lower(META).unwrap();
        let engine = QueryEngine::new(&compiled.ir, &store.it);
        let r = engine.query(&Predicate::node());
        assert!(r.len() > 0);
        println!("All nodes ({}): {:?}", r.len(), r.names());
    }

    #[test]
    fn find_all_edges() {
        let (store, compiled) = load_and_lower(META).unwrap();
        let engine = QueryEngine::new(&compiled.ir, &store.it);
        let r = engine.query(&Predicate::edge());
        assert!(r.len() > 0);
        println!("All edges ({}): {:?}", r.len(), r.names());
    }

    #[test]
    fn find_axes_by_inheritance() {
        let (store, compiled) = load_and_lower(META).unwrap();
        let engine = QueryEngine::new(&compiled.ir, &store.it);
        let r = engine.query(&Predicate::node().and(Predicate::inherits("axis_definition")));
        let names = r.names();
        println!("Axes: {:?}", names);
        assert_eq!(r.len(), 4);
        assert!(names.contains(&"AXIS_X"));
        assert!(names.contains(&"AXIS_Y"));
        assert!(names.contains(&"AXIS_Z"));
        assert!(names.contains(&"AXIS_M_Z"));
    }

    #[test]
    fn transitive_inheritance_joint_types() {
        let (store, compiled) = load_and_lower(META).unwrap();
        let engine = QueryEngine::new(&compiled.ir, &store.it);
        let r = engine.query(&Predicate::edge().and(Predicate::inherits("joint")));
        let names = r.names();
        println!("Edges inheriting from 'joint': {:?}", names);
        assert!(names.contains(&"fixed_joint"));
        assert!(names.contains(&"rev_joint"));
        assert!(names.contains(&"conti_joint"));
        assert!(names.contains(&"prismatic_joint"));
    }

    #[test]
    fn transitive_inheritance_meta_element() {
        let (store, compiled) = load_and_lower(META).unwrap();
        let engine = QueryEngine::new(&compiled.ir, &store.it);
        let r = engine.query(&Predicate::node().and(Predicate::inherits("meta_element")));
        let names = r.names();
        println!("Inherits meta_element: {:?}", names);
        assert!(names.contains(&"link"));
        assert!(names.contains(&"frame"));
        assert!(names.contains(&"sensor"));
        assert!(names.contains(&"control"));
    }

    #[test]
    fn controllers_by_inheritance() {
        let (store, compiled) = load_and_lower(META).unwrap();
        let engine = QueryEngine::new(&compiled.ir, &store.it);
        let r = engine.query(&Predicate::node().and(Predicate::inherits("meta_controller")));
        assert_eq!(r.len(), 5, "Should find 5 controller types");
    }

    #[test]
    fn sensors_by_inheritance() {
        let (store, compiled) = load_and_lower(META).unwrap();
        let engine = QueryEngine::new(&compiled.ir, &store.it);
        let r = engine.query(&Predicate::node().and(Predicate::inherits("sensor")));
        assert_eq!(r.len(), 2, "rgb_camera and laser_scanner");
    }

    #[test]
    fn not_predicate() {
        let (store, compiled) = load_and_lower(META).unwrap();
        let engine = QueryEngine::new(&compiled.ir, &store.it);
        let all = engine.query(&Predicate::node());
        let non_link = engine.query(&Predicate::node().and(Predicate::named("link").not()));
        assert_eq!(non_link.len(), all.len() - 1);
    }

    #[test]
    fn or_predicate() {
        let (store, compiled) = load_and_lower(META).unwrap();
        let engine = QueryEngine::new(&compiled.ir, &store.it);
        let r = engine.query(&Predicate::node().and(
            Predicate::named("link").or(Predicate::named("frame"))
        ));
        assert_eq!(r.len(), 2);
    }
}
```

---

## T14 — Robot Cross-Import Queries

**File:** `hymeko_core/tests/query/test_query_robot.rs`

```rust
#[cfg(test)]
mod test_query_robot {
    use hymeko::query::engine::QueryEngine;
    use hymeko::query::predicate::*;
    use hymeko::query::urdf::{urdf_queries, validate_robot_schema};
    use crate::test_helpers::load_and_lower;

    const ROBOT: &str = "./data/robotics/robot_4wh.hymeko";

    #[test]
    fn find_robot_links() {
        let (store, compiled) = load_and_lower(ROBOT).unwrap();
        let engine = QueryEngine::new(&compiled.ir, &store.it);
        let r = engine.query(&Predicate::node().and(Predicate::inherits("link")));
        let names = r.names();
        println!("Robot links: {:?}", names);
        assert_eq!(r.len(), 6);
        for expected in &["base_link", "wheel_fr", "wheel_fl", "wheel_rr", "wheel_rl", "camera_link"] {
            assert!(names.contains(expected), "{} missing", expected);
        }
    }

    #[test]
    fn find_robot_joints_transitive() {
        let (store, compiled) = load_and_lower(ROBOT).unwrap();
        let engine = QueryEngine::new(&compiled.ir, &store.it);
        // joint_fr → conti_joint → joint (2 levels)
        let r = engine.query(&Predicate::edge().and(Predicate::inherits("joint")));
        let names = r.names();
        println!("Robot joints: {:?}", names);
        assert!(names.contains(&"joint_fr"));
        assert!(names.contains(&"camera_joint"));
    }

    #[test]
    fn continuous_joints_specifically() {
        let (store, compiled) = load_and_lower(ROBOT).unwrap();
        let engine = QueryEngine::new(&compiled.ir, &store.it);
        let r = engine.query(&Predicate::edge().and(Predicate::inherits("conti_joint")));
        assert_eq!(r.len(), 4);
    }

    #[test]
    fn fixed_joints() {
        let (store, compiled) = load_and_lower(ROBOT).unwrap();
        let engine = QueryEngine::new(&compiled.ir, &store.it);
        let r = engine.query(&Predicate::edge().and(Predicate::inherits("fixed_joint")));
        assert_eq!(r.len(), 1, "Only camera_joint is fixed");
    }

    #[test]
    fn joints_with_plus_ref_to_link() {
        let (store, compiled) = load_and_lower(ROBOT).unwrap();
        let engine = QueryEngine::new(&compiled.ir, &store.it);
        let r = engine.query(
            &Predicate::edge()
                .and(Predicate::inherits("joint"))
                .and(Predicate::HasPlusRef(Box::new(Predicate::inherits("link"))))
        );
        println!("Joints with +link: {:?}", r.names());
        assert!(r.len() >= 5);
    }

    #[test]
    fn joints_with_minus_ref_to_link() {
        let (store, compiled) = load_and_lower(ROBOT).unwrap();
        let engine = QueryEngine::new(&compiled.ir, &store.it);
        let r = engine.query(
            &Predicate::edge()
                .and(Predicate::inherits("joint"))
                .and(Predicate::HasMinusRef(Box::new(Predicate::inherits("link"))))
        );
        println!("Joints with -link (child): {:?}", r.names());
        assert!(r.len() >= 5);
    }

    #[test]
    fn heavy_links() {
        let (store, compiled) = load_and_lower(ROBOT).unwrap();
        let engine = QueryEngine::new(&compiled.ir, &store.it);
        let r = engine.query(
            &Predicate::node()
                .and(Predicate::inherits("link"))
                .and(Predicate::ChildValue("mass".into(), ValuePredicate::NumGt(10.0)))
        );
        assert_eq!(r.len(), 1);
        assert!(r.names().contains(&"base_link"));
    }

    #[test]
    fn name_prefix_wheels() {
        let (store, compiled) = load_and_lower(ROBOT).unwrap();
        let engine = QueryEngine::new(&compiled.ir, &store.it);
        let r = engine.query(&Predicate::node().and(Predicate::name_prefix("wheel")));
        assert_eq!(r.len(), 4);
    }

    #[test]
    fn link_weight_annotations_accessible() {
        let (store, compiled) = load_and_lower(ROBOT).unwrap();
        let engine = QueryEngine::new(&compiled.ir, &store.it);
        let joints = engine.query(
            &Predicate::edge()
                .and(Predicate::inherits("conti_joint"))
                .and(Predicate::HasPlusRef(Box::new(Predicate::named("base_link"))))
        );
        assert_eq!(joints.len(), 4);
        // Verify weight annotations exist on plus refs
        for (did, name) in &joints.matches {
            let eid = compiled.ir.as_edge(*did).unwrap();
            let edge = &compiled.ir.edges[eid.0];
            for &aid in &edge.arcs {
                let arc = &compiled.ir.arcs[aid.0];
                for sref in &arc.refs {
                    if sref.sign() == 1 {
                        let atom = sref.atom();
                        println!("Joint {} +ref weights: {:?}", name, atom.weights);
                        assert!(atom.weights.is_some(),
                            "Plus ref on {} must have origin weights", name);
                    }
                }
            }
        }
    }

    #[test]
    fn schema_validation_passes() {
        let (store, compiled) = load_and_lower(ROBOT).unwrap();
        let errors = validate_robot_schema(&compiled.ir, &store.it);
        if !errors.is_empty() {
            for e in &errors { eprintln!("  Schema error: {}", e); }
        }
        assert!(errors.is_empty(), "Got {} schema errors", errors.len());
    }

    #[test]
    fn batch_query_all_urdf() {
        let (store, compiled) = load_and_lower(ROBOT).unwrap();
        let engine = QueryEngine::new(&compiled.ir, &store.it);
        let results = engine.query_all(&urdf_queries());
        println!("--- URDF query results ---");
        for (label, qr) in &results {
            println!("  {}: {} matches → {:?}", label, qr.len(), qr.names());
        }
        let links = results.iter().find(|(l, _)| l == "links");
        assert!(links.unwrap().1.len() >= 6);
    }
}
```

---

## T15 — URDF Generation

**File:** `hymeko_core/tests/query/test_urdf.rs`

```rust
#[cfg(test)]
mod test_urdf {
    use hymeko::query::urdf::generate_urdf;
    use crate::test_helpers::load_and_lower;

    const ROBOT: &str = "./data/robotics/robot_4wh.hymeko";

    #[test]
    fn generate_urdf_structure() {
        let (store, compiled) = load_and_lower(ROBOT).unwrap();
        let urdf = generate_urdf(&compiled.ir, &store.it, "diff_robot_4wh");
        println!("--- Generated URDF ---\n{}", urdf);

        assert!(urdf.contains("<robot name=\"diff_robot_4wh\""));
        assert!(urdf.ends_with("</robot>\n"));
        assert!(urdf.contains("<link name=\"base_link\""));
        assert!(urdf.contains("<link name=\"wheel_fr\""));
        assert!(urdf.contains("<link name=\"camera_link\""));
    }

    #[test]
    fn joint_types_correct() {
        let (store, compiled) = load_and_lower(ROBOT).unwrap();
        let urdf = generate_urdf(&compiled.ir, &store.it, "test");
        assert!(urdf.contains("type=\"continuous\""));
        assert!(urdf.contains("type=\"fixed\""));
    }

    #[test]
    fn degree_to_radian_conversion() {
        let (store, compiled) = load_and_lower(ROBOT).unwrap();
        let urdf = generate_urdf(&compiled.ir, &store.it, "test");
        // -90° = -1.5708 rad
        assert!(urdf.contains("-1.5708"), "Expected -90° → -1.5708 radians");
    }

    #[test]
    fn axis_on_continuous_not_fixed() {
        let (store, compiled) = load_and_lower(ROBOT).unwrap();
        let urdf = generate_urdf(&compiled.ir, &store.it, "test");
        let axis_count = urdf.matches("<axis xyz=").count();
        assert!(axis_count >= 4, "At least 4 axes for continuous joints");
    }

    #[test]
    fn geometry_types() {
        let (store, compiled) = load_and_lower(ROBOT).unwrap();
        let urdf = generate_urdf(&compiled.ir, &store.it, "test");
        assert!(urdf.contains("<box size=\"0.7 0.5 0.2\""));
        assert!(urdf.contains("<cylinder"));
    }

    #[test]
    fn link_and_joint_counts() {
        let (store, compiled) = load_and_lower(ROBOT).unwrap();
        let urdf = generate_urdf(&compiled.ir, &store.it, "test");
        let link_count = urdf.matches("<link name=").count();
        let joint_count = urdf.matches("<joint name=").count();
        assert!(link_count >= 6, "Got {} links", link_count);
        assert!(joint_count >= 5, "Got {} joints", joint_count);
    }

    #[test]
    fn xml_well_formed() {
        let (store, compiled) = load_and_lower(ROBOT).unwrap();
        let urdf = generate_urdf(&compiled.ir, &store.it, "test");
        assert!(urdf.starts_with("<?xml"));
        assert!(urdf.ends_with("</robot>\n"));
    }
}
```

---

## How to Run

```bash
cd hymeko_core

# All query tests with output
cargo test query -- --nocapture

# Specific test
cargo test test_query_robot::find_robot_links -- --nocapture

# URDF generation test (prints the full XML)
cargo test test_urdf::generate_urdf_structure -- --nocapture
```

The `--nocapture` output from `batch_query_all_urdf` gives you the numbers for Paper 2's Table II directly.
