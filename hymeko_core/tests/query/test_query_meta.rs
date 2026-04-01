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