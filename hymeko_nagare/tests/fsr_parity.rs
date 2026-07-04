use std::collections::HashMap;

use hymeko_nagare::{FsrMixer, FsrRoute, LinearLayer};

const FIXTURE: &str =
    include_str!("../../docs/plans/2026-06-30-nagare-fsr-kernel-codex/fixtures/fsr_mixer_tiny.txt");

#[derive(Debug)]
struct Fixture {
    dims: Vec<usize>,
    vectors: HashMap<String, Vec<f32>>,
}

impl Fixture {
    fn parse(text: &str) -> Self {
        let mut dims = Vec::new();
        let mut vectors = HashMap::new();
        for line in text.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            let mut parts = trimmed.split_whitespace();
            let name = parts.next().unwrap();
            if name == "dims" {
                dims = parts.map(|x| x.parse::<usize>().unwrap()).collect();
            } else {
                let len = parts.next().unwrap().parse::<usize>().unwrap();
                let values: Vec<f32> = parts.map(|x| x.parse::<f32>().unwrap()).collect();
                assert_eq!(values.len(), len, "{name}");
                vectors.insert(name.to_string(), values);
            }
        }
        Self { dims, vectors }
    }

    fn get(&self, name: &str) -> &[f32] {
        self.vectors
            .get(name)
            .unwrap_or_else(|| panic!("missing {name}"))
    }
}

fn assert_close_vec(name: &str, got: &[f32], expected: &[f32], tol: f32) {
    assert_eq!(got.len(), expected.len(), "{name} length");
    let mut max_abs = 0.0f32;
    for (idx, (&a, &b)) in got.iter().zip(expected.iter()).enumerate() {
        let abs = (a - b).abs();
        max_abs = max_abs.max(abs);
        assert!(
            abs < tol,
            "{name}[{idx}] got={a} expected={b} abs={abs} max_abs={max_abs}"
        );
    }
}

#[test]
fn fsr_mixer_matches_pytorch_fixture_forward_and_backward() {
    let fixture = Fixture::parse(FIXTURE);
    assert_eq!(fixture.dims.len(), 7);
    let hidden = fixture.dims[0];
    let batch = fixture.dims[1];
    let seq_len = fixture.dims[2];
    let n_blocks = fixture.dims[3];
    let max_seq_len = fixture.dims[4];
    let k = fixture.dims[5];
    assert_eq!(hidden, 3 * n_blocks);

    let selected: Vec<u32> = fixture.get("selected").iter().map(|x| *x as u32).collect();
    let route = FsrRoute::new(&selected, fixture.get("gate"), batch, seq_len, k);
    let mixer = FsrMixer {
        offset_bivec: fixture.get("offset_bivec").to_vec(),
        offset_sign: fixture.get("offset_sign").to_vec(),
        to_v: LinearLayer {
            w: fixture.get("to_v_w").to_vec(),
            b: fixture.get("to_v_b").to_vec(),
            in_dim: hidden,
            out_dim: hidden,
        },
        out: LinearLayer {
            w: fixture.get("out_w").to_vec(),
            b: fixture.get("out_b").to_vec(),
            in_dim: hidden,
            out_dim: hidden,
        },
        n_blocks,
        max_seq_len,
    };

    let (out, cache) = mixer.forward(fixture.get("h"), route);
    assert_close_vec("out", &out, fixture.get("out"), 2e-5);

    let grad_out = vec![1.0; out.len()];
    let backward = mixer.backward(&cache, &grad_out);
    assert_close_vec("grad_h", &backward.grad_h, fixture.get("grad_h"), 2e-5);
    assert_close_vec(
        "grad_offset_bivec",
        &backward.grad_params.offset_bivec,
        fixture.get("grad_offset_bivec"),
        2e-5,
    );
    assert_close_vec(
        "grad_offset_sign",
        &backward.grad_params.offset_sign,
        fixture.get("grad_offset_sign"),
        2e-5,
    );
    assert_close_vec(
        "grad_to_v_w",
        &backward.grad_params.to_v.w,
        fixture.get("grad_to_v_w"),
        2e-5,
    );
    assert_close_vec(
        "grad_to_v_b",
        &backward.grad_params.to_v.b,
        fixture.get("grad_to_v_b"),
        2e-5,
    );
    assert_close_vec(
        "grad_out_w",
        &backward.grad_params.out.w,
        fixture.get("grad_out_w"),
        2e-5,
    );
    assert_close_vec(
        "grad_out_b",
        &backward.grad_params.out.b,
        fixture.get("grad_out_b"),
        2e-5,
    );
}
