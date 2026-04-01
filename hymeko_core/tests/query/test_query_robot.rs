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