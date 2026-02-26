#[cfg(test)]
mod tests {
    use hymeko_framework::tensor::aggregation::{agg_sign, agg_weight, clamp01, AggCfg, SignAgg, WeightAgg};

    fn cfg(weight: WeightAgg, sign: SignAgg, clamp01: bool) -> AggCfg {
        AggCfg { weight, sign, clamp01 }
    }

    #[test]
    fn test_clamp01_basic() {
        let c = AggCfg { weight: WeightAgg::Sum, sign: SignAgg::PreferNonNeutral, clamp01: true };

        assert_eq!(clamp01::<f32>(-0.5), 0.0);
        assert_eq!(clamp01::<f32>(0.2), 0.2);
        assert_eq!(clamp01::<f32>(1.2), 1.0);

        // clamp01 flag only affects agg_weight output, clamp01() itself is unconditional
        let out = agg_weight(&c, -0.5_f32, 0.2_f32);
        assert_eq!(out, 0.0); // Sum = -0.3 -> clamp -> 0
    }

    #[test]
    fn test_weight_agg_sum() {
        let c = cfg(WeightAgg::Sum, SignAgg::PreferNonNeutral, false);
        let out = agg_weight(&c, 0.2_f32, 0.3_f32);
        assert!((out - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_weight_agg_max() {
        let c = cfg(WeightAgg::Max, SignAgg::PreferNonNeutral, false);

        let out1 = agg_weight(&c, 0.2_f32, 0.3_f32);
        assert!((out1 - 0.3).abs() < 1e-6);

        let out2 = agg_weight(&c, 0.9_f32, 0.1_f32);
        assert!((out2 - 0.9).abs() < 1e-6);

        // extra: equal values
        let out3 = agg_weight(&c, 0.42_f32, 0.42_f32);
        assert!((out3 - 0.42).abs() < 1e-6);
    }

    #[test]
    fn test_weight_agg_prob_sum01() {
        let c = cfg(WeightAgg::ProbSum01, SignAgg::PreferNonNeutral, false);

        // 1 - (1-a)(1-b)
        let out = agg_weight(&c, 0.2_f32, 0.3_f32);
        let expected = 1.0 - (1.0 - 0.2) * (1.0 - 0.3); // 0.44
        assert!((out - expected).abs() < 1e-6);

        // boundary
        let out2 = agg_weight(&c, 1.0_f32, 0.1_f32);
        assert!((out2 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_weight_agg_lukasiewicz_sat01() {
        let c = cfg(WeightAgg::LukasSat01, SignAgg::PreferNonNeutral, false);

        let out1 = agg_weight(&c, 0.2_f32, 0.3_f32);
        assert!((out1 - 0.5).abs() < 1e-6);

        let out2 = agg_weight(&c, 0.7_f32, 0.6_f32);
        assert!((out2 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_weight_agg_clamp01_flag() {
        let c = cfg(WeightAgg::Sum, SignAgg::PreferNonNeutral, true);

        let out1 = agg_weight(&c, -0.2_f32, 0.1_f32); // -0.1 -> clamp to 0
        assert!((out1 - 0.0).abs() < 1e-6);

        let out2 = agg_weight(&c, 0.9_f32, 0.9_f32); // 1.8 -> clamp to 1
        assert!((out2 - 1.0).abs() < 1e-6);
    }

    // --------------------------
    // Sign aggregation tests
    // --------------------------

    #[test]
    fn test_sign_agg_prefer_non_neutral() {
        let c = cfg(WeightAgg::Sum, SignAgg::PreferNonNeutral, false);

        // equal stays equal
        assert_eq!(agg_sign(&c,  1,  1, 1.0_f32, 1.0_f32),  1);
        assert_eq!(agg_sign(&c, -1, -1, 1.0_f32, 1.0_f32), -1);
        assert_eq!(agg_sign(&c,  0,  0, 1.0_f32, 1.0_f32),  0);

        // neutral yields the other
        assert_eq!(agg_sign(&c, 0,  1, 1.0_f32, 1.0_f32), 1);
        assert_eq!(agg_sign(&c, 1,  0, 1.0_f32, 1.0_f32), 1);
        assert_eq!(agg_sign(&c, 0, -1, 1.0_f32, 1.0_f32), -1);

        // conflict -> neutral
        assert_eq!(agg_sign(&c,  1, -1, 1.0_f32, 1.0_f32), 0);
        assert_eq!(agg_sign(&c, -1,  1, 1.0_f32, 1.0_f32), 0);
    }

    #[test]
    fn test_sign_agg_weighted_vote() {
        let c = cfg(WeightAgg::Sum, SignAgg::WeightedVote, false);

        // Positive wins if weighted sum > 0
        assert_eq!(agg_sign(&c, 1, -1, 10.0_f32, 1.0_f32),  1);

        // Negative wins if weighted sum < 0
        assert_eq!(agg_sign(&c, 1, -1, 1.0_f32, 10.0_f32), -1);

        // Tie -> 0
        assert_eq!(agg_sign(&c, 1, -1, 2.0_f32, 2.0_f32), 0);

        // Neutral treated as 0 in vote
        assert_eq!(agg_sign(&c, 0,  1, 10.0_f32, 1.0_f32), 1);
        assert_eq!(agg_sign(&c, 0, -1, 10.0_f32, 1.0_f32), -1);
    }

    #[test]
    fn test_sign_agg_channels3() {
        let c = cfg(WeightAgg::Sum, SignAgg::Channels3, false);

        // equal -> itself
        assert_eq!(agg_sign(&c,  1,  1, 1.0_f32, 1.0_f32),  1);
        assert_eq!(agg_sign(&c, -1, -1, 1.0_f32, 1.0_f32), -1);
        assert_eq!(agg_sign(&c,  0,  0, 1.0_f32, 1.0_f32),  0);

        // mismatch -> neutral channel
        assert_eq!(agg_sign(&c,  1,  0, 1.0_f32, 1.0_f32), 0);
        assert_eq!(agg_sign(&c,  1, -1, 1.0_f32, 1.0_f32), 0);
        assert_eq!(agg_sign(&c, -1,  0, 1.0_f32, 1.0_f32), 0);
    }

    #[test]
    fn test_agg_sign_works_for_f64_too() {
        let c = cfg(WeightAgg::Sum, SignAgg::WeightedVote, false);
        let s = agg_sign(&c, 1, -1, 0.1_f64, 0.01_f64);
        assert_eq!(s, 1);
    }
}