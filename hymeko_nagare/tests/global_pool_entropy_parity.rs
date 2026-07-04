#[path = "../examples/global_pool_entropy_parity.rs"]
#[allow(dead_code)]
mod parity;

use parity::{forward_case, parse_fixture_text, report_case};

#[test]
fn parses_plain_text_fixture_and_matches_expected_forward() {
    let text = "\
HYMEKO_GLOBAL_POOL_ENTROPY_PARITY_V1
cases 1
case tiny
samples 1
points 2
hidden 1
tensor x 4 1,2,2
1 2 3 4
endtensor
tensor embed_w 2 2,1
1
0
endtensor
tensor embed_b 1 1
0
endtensor
tensor first_w 6 3,1,2
0 0 0 0 0 0
endtensor
tensor first_b 2 2
0 0
endtensor
tensor update_w 5 5,1
1 0 0 0 0
endtensor
tensor update_b 1 1
0
endtensor
tensor head_w 6 3,1,2
1 -1 0 0 0 0
endtensor
tensor head_b 2 2
0 0
endtensor
tensor logits 2 1,2
2 -2
endtensor
tensor logits_first 2 1,2
0 0
endtensor
tensor entropy 1 1,1
1
endtensor
endcase
";
    let fixture = parse_fixture_text(text).expect("fixture should parse");
    let case = &fixture.cases[0];
    let out = forward_case(case);
    assert_eq!(out.logits, vec![2.0, -2.0]);
    assert_eq!(out.logits_first, vec![0.0, 0.0]);
    assert!((out.entropy[0] - 1.0).abs() < 1e-6);
    let report = report_case(case, 5);
    assert!(report.max_abs_logits < 1e-6);
    assert!(report.allocation_bytes_per_forward > 0);
}
