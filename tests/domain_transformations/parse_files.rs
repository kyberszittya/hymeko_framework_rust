#[cfg(test)]
mod test_parse_files {
    use hymeko::tensor::aggregation::{AggCfg, SignAgg, WeightAgg};
    use hymeko::tensor::tensor_val::{EdgeWScalar, ScalarWeightExtractor};
    use hymeko::traversal::hypergraphview::HyperGraphView;
    use crate::test_helpers::load_and_lower;

    #[test]
    fn test_load_meta_geometry() {
        let (_store, compiled) =
            load_and_lower("./data/robotics/meta_kinematics.hymeko").unwrap();

        let aggcfg = AggCfg { weight: WeightAgg::Sum, sign: SignAgg::PreferNonNeutral, clamp01: false };
        let ex = ScalarWeightExtractor::default();
        let hg = HyperGraphView::<f32, EdgeWScalar<f32>, f32>::from_ir(&compiled.ir, &aggcfg, &ex);

        let n = hg.num_nodes();
        let m = hg.num_edges();
        let use_abs = true;

    }
}