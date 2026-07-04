//! Nagare-native holonomy toys with fused entropy feedback.

use std::{
    env,
    fs::File,
    io::{self, Write},
    path::PathBuf,
    str::FromStr,
    time::Instant,
};

use hymeko_clifford::{cayley_to_unit_quat, quat_mul, quat_rotate, Multivector, Signature};
use hymeko_nagare::{
    adam_step, fused_entropy_update_backward, fused_entropy_update_forward, linear_backward,
    linear_forward, AdamState, FusedEntropyUpdateShape, LinearLayer,
};
use rand::{rngs::StdRng, seq::SliceRandom, Rng, SeedableRng};

const FEATURE_DIM: usize = 7;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Task {
    Moons,
    Spiral,
    Xor,
}

impl Task {
    fn as_str(self) -> &'static str {
        match self {
            Self::Moons => "moons",
            Self::Spiral => "spiral",
            Self::Xor => "xor",
        }
    }
}

impl FromStr for Task {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "moons" => Ok(Self::Moons),
            "spiral" => Ok(Self::Spiral),
            "xor" => Ok(Self::Xor),
            other => Err(format!("unknown task '{other}'")),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Dataset {
    pub x: Vec<f32>,
    pub y: Vec<usize>,
    pub samples: usize,
    pub points: usize,
}

#[derive(Clone, Debug)]
pub struct Config {
    pub tasks: Vec<Task>,
    pub n_train: usize,
    pub n_test: usize,
    pub n_points: usize,
    pub hidden: usize,
    pub epochs: usize,
    pub batch_size: usize,
    pub lr: f32,
    pub seed: u64,
    pub out: Option<PathBuf>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            tasks: vec![Task::Moons, Task::Spiral, Task::Xor],
            n_train: 192,
            n_test: 96,
            n_points: 32,
            hidden: 24,
            epochs: 50,
            batch_size: 32,
            lr: 2.0e-3,
            seed: 37,
            out: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Metrics {
    pub acc: f32,
    pub loss: f32,
    pub entropy: f32,
    pub clifford_error: f32,
}

#[derive(Clone, Debug)]
pub struct Timing {
    pub median_us_per_sample: f64,
    pub mean_us_per_sample: f64,
    pub max_us_per_sample: f64,
}

#[derive(Clone, Debug)]
pub struct TaskResult {
    pub task: Task,
    pub train: Metrics,
    pub test: Metrics,
    pub timing: Timing,
    pub n_params: usize,
    pub param_bytes: usize,
    pub materialized_update_bytes: usize,
    pub fused_update_bytes: usize,
}

#[derive(Clone, Debug)]
struct DenseRelu {
    layer: LinearLayer,
    w_state: AdamState,
    b_state: AdamState,
}

impl DenseRelu {
    fn new(in_dim: usize, out_dim: usize, seed: u64) -> Self {
        let layer = LinearLayer::new(in_dim, out_dim, seed);
        Self {
            w_state: AdamState::new(layer.w.len()),
            b_state: AdamState::new(layer.b.len()),
            layer,
        }
    }

    fn forward(&self, x: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let pre = linear_forward(&self.layer, x);
        let act = pre.iter().map(|v| v.max(0.0)).collect();
        (pre, act)
    }

    fn backward(&self, x: &[f32], pre: &[f32], grad: &[f32]) -> (Vec<f32>, LinearLayer) {
        let grad_pre: Vec<f32> = grad
            .iter()
            .zip(pre.iter())
            .map(|(&g, &z)| if z > 0.0 { g } else { 0.0 })
            .collect();
        linear_backward(&self.layer, x, &grad_pre)
    }

    fn step(&mut self, grad: &LinearLayer, lr: f32) {
        adam_step(&mut self.layer.w, &grad.w, &mut self.w_state, lr);
        adam_step(&mut self.layer.b, &grad.b, &mut self.b_state, lr);
    }

    fn n_params(&self) -> usize {
        self.layer.w.len() + self.layer.b.len()
    }
}

#[derive(Clone, Debug)]
struct Dense {
    layer: LinearLayer,
    w_state: AdamState,
    b_state: AdamState,
}

impl Dense {
    fn new(in_dim: usize, out_dim: usize, seed: u64) -> Self {
        let layer = LinearLayer::new(in_dim, out_dim, seed);
        Self {
            w_state: AdamState::new(layer.w.len()),
            b_state: AdamState::new(layer.b.len()),
            layer,
        }
    }

    fn forward(&self, x: &[f32]) -> Vec<f32> {
        linear_forward(&self.layer, x)
    }

    fn backward(&self, x: &[f32], grad: &[f32]) -> (Vec<f32>, LinearLayer) {
        linear_backward(&self.layer, x, grad)
    }

    fn step(&mut self, grad: &LinearLayer, lr: f32) {
        adam_step(&mut self.layer.w, &grad.w, &mut self.w_state, lr);
        adam_step(&mut self.layer.b, &grad.b, &mut self.b_state, lr);
    }

    fn n_params(&self) -> usize {
        self.layer.w.len() + self.layer.b.len()
    }
}

#[derive(Clone, Debug)]
struct PoolCache {
    input: Vec<f32>,
    mean: Vec<f32>,
    std: Vec<f32>,
    max_index: Vec<usize>,
    batch: usize,
    points: usize,
    hidden: usize,
}

#[derive(Clone, Debug)]
struct CeOut {
    loss: f32,
    grad_logits: Vec<f32>,
    acc: f32,
    entropy: f32,
}

#[derive(Clone, Debug)]
struct HolonomySetNet {
    embed: DenseRelu,
    first: Dense,
    update: DenseRelu,
    head: Dense,
    hidden: usize,
}

impl HolonomySetNet {
    fn new(hidden: usize, seed: u64) -> Self {
        Self {
            embed: DenseRelu::new(FEATURE_DIM, hidden, seed),
            first: Dense::new(3 * hidden, 2, seed + 1),
            update: DenseRelu::new(4 * hidden + 1, hidden, seed + 2),
            head: Dense::new(3 * hidden, 2, seed + 3),
            hidden,
        }
    }

    fn train_batch(&mut self, x: &[f32], y: &[usize], batch: usize, points: usize, lr: f32) -> f32 {
        let features = quaternion_periodic_features(x, batch, points);
        let (embed_pre, h) = self.embed.forward(&features);
        let (pooled, pool_cache) = global_pool_with_cache(&h, batch, points, self.hidden);
        let logits_first = self.first.forward(&pooled);
        let first_ce = cross_entropy(&logits_first, y);
        let entropy = entropy_feature(&logits_first);
        let update_shape = FusedEntropyUpdateShape {
            batch,
            points,
            hidden: self.hidden,
        };
        let update_pre_linear =
            fused_entropy_update_forward(&self.update.layer, &h, &pooled, &entropy, update_shape);
        let h2: Vec<f32> = update_pre_linear.iter().map(|v| v.max(0.0)).collect();
        let (pooled2, pool2_cache) = global_pool_with_cache(&h2, batch, points, self.hidden);
        let logits = self.head.forward(&pooled2);
        let final_ce = cross_entropy(&logits, y);

        let (grad_pool2, grad_head) = self.head.backward(&pooled2, &final_ce.grad_logits);
        let grad_h2 = global_pool_backward(&pool2_cache, &grad_pool2);
        let grad_update_pre: Vec<f32> = grad_h2
            .iter()
            .zip(update_pre_linear.iter())
            .map(|(&g, &z)| if z > 0.0 { g } else { 0.0 })
            .collect();
        let fused_grad = fused_entropy_update_backward(
            &self.update.layer,
            &h,
            &pooled,
            &entropy,
            &grad_update_pre,
            update_shape,
        );
        let mut grad_h = fused_grad.grad_h;
        let mut grad_pool = fused_grad.grad_pooled;

        let mut grad_first_logits = first_ce.grad_logits;
        for g in &mut grad_first_logits {
            *g *= 0.25;
        }
        let (grad_pool_first, grad_first) = self.first.backward(&pooled, &grad_first_logits);
        for (dst, src) in grad_pool.iter_mut().zip(grad_pool_first.iter()) {
            *dst += src;
        }
        let grad_h_from_pool = global_pool_backward(&pool_cache, &grad_pool);
        for (dst, src) in grad_h.iter_mut().zip(grad_h_from_pool.iter()) {
            *dst += src;
        }
        let (_grad_features, grad_embed) = self.embed.backward(&features, &embed_pre, &grad_h);

        self.head.step(&grad_head, lr);
        self.update.step(&fused_grad.grad_layer, lr);
        self.first.step(&grad_first, lr);
        self.embed.step(&grad_embed, lr);
        final_ce.loss + 0.25 * first_ce.loss
    }

    fn logits(&self, x: &[f32], batch: usize, points: usize) -> Vec<f32> {
        let features = quaternion_periodic_features(x, batch, points);
        let (_, h) = self.embed.forward(&features);
        let pooled = global_pool(&h, batch, points, self.hidden);
        let logits_first = self.first.forward(&pooled);
        let entropy = entropy_feature(&logits_first);
        let update_shape = FusedEntropyUpdateShape {
            batch,
            points,
            hidden: self.hidden,
        };
        let update_pre =
            fused_entropy_update_forward(&self.update.layer, &h, &pooled, &entropy, update_shape);
        let h2: Vec<f32> = update_pre.iter().map(|v| v.max(0.0)).collect();
        self.head
            .forward(&global_pool(&h2, batch, points, self.hidden))
    }

    fn n_params(&self) -> usize {
        self.embed.n_params()
            + self.first.n_params()
            + self.update.n_params()
            + self.head.n_params()
    }
}

/// Quaternion periodic feature lift for `(x, y)` point sets.
pub fn quaternion_periodic_features(x: &[f32], batch: usize, points: usize) -> Vec<f32> {
    assert_eq!(x.len(), batch * points * 2);
    let mut out = vec![0.0; batch * points * FEATURE_DIM];
    for b in 0..batch {
        let mut hol = [1.0, 0.0, 0.0, 0.0];
        for p in 0..points {
            let src = (b * points + p) * 2;
            let px = x[src];
            let py = x[src + 1];
            let r = (px * px + py * py).sqrt();
            let angle = py.atan2(px);
            let q = cayley_to_unit_quat([0.0, 0.0, 0.5 * angle.sin()]);
            hol = quat_mul(hol, q);
            let rotated = quat_rotate(q, [px, py, r]);
            let dst = (b * points + p) * FEATURE_DIM;
            out[dst] = px;
            out[dst + 1] = py;
            out[dst + 2] = r;
            out[dst + 3] = rotated[0];
            out[dst + 4] = rotated[1];
            out[dst + 5] = hol[0];
            out[dst + 6] = hol[3];
        }
    }
    out
}

/// Clifford \(Cl(2,0)\) probability-vector squared error.
pub fn clifford_probability_error(logits: &[f32], labels: &[usize]) -> f32 {
    assert_eq!(logits.len(), labels.len() * 2);
    let sig = Signature::euclidean(2);
    let mut sum = 0.0;
    for b in 0..labels.len() {
        let (p0, p1) = softmax2(logits[2 * b], logits[2 * b + 1]);
        let mut err = Multivector::zero(2);
        err.components[1] = (p0 - f32::from(labels[b] == 0)) as f64;
        err.components[2] = (p1 - f32::from(labels[b] == 1)) as f64;
        sum += err.geo(&err, &sig).components[0] as f32;
    }
    sum / labels.len() as f32
}

fn global_pool(input: &[f32], batch: usize, points: usize, hidden: usize) -> Vec<f32> {
    global_pool_with_cache(input, batch, points, hidden).0
}

fn global_pool_with_cache(
    input: &[f32],
    batch: usize,
    points: usize,
    hidden: usize,
) -> (Vec<f32>, PoolCache) {
    assert_eq!(input.len(), batch * points * hidden);
    let mut out = vec![0.0; batch * hidden * 3];
    let mut mean = vec![0.0; batch * hidden];
    let mut std = vec![0.0; batch * hidden];
    let mut max_index = vec![0; batch * hidden];
    let inv_points = 1.0 / points as f32;
    for b in 0..batch {
        for h_idx in 0..hidden {
            let mut sum = 0.0;
            let mut max_val = f32::NEG_INFINITY;
            let mut argmax = 0;
            for p in 0..points {
                let v = input[(b * points + p) * hidden + h_idx];
                sum += v;
                if v > max_val {
                    max_val = v;
                    argmax = p;
                }
            }
            let mu = sum * inv_points;
            let mut var = 0.0;
            for p in 0..points {
                let d = input[(b * points + p) * hidden + h_idx] - mu;
                var += d * d;
            }
            let sigma = (var * inv_points + 1.0e-6).sqrt();
            mean[b * hidden + h_idx] = mu;
            std[b * hidden + h_idx] = sigma;
            max_index[b * hidden + h_idx] = argmax;
            out[b * hidden * 3 + h_idx] = mu;
            out[b * hidden * 3 + hidden + h_idx] = sigma;
            out[b * hidden * 3 + 2 * hidden + h_idx] = max_val;
        }
    }
    (
        out,
        PoolCache {
            input: input.to_vec(),
            mean,
            std,
            max_index,
            batch,
            points,
            hidden,
        },
    )
}

fn global_pool_backward(cache: &PoolCache, grad_pool: &[f32]) -> Vec<f32> {
    let mut grad = vec![0.0; cache.input.len()];
    let inv_points = 1.0 / cache.points as f32;
    for b in 0..cache.batch {
        for h_idx in 0..cache.hidden {
            let base = b * cache.hidden * 3;
            let grad_mean = grad_pool[base + h_idx];
            let grad_std = grad_pool[base + cache.hidden + h_idx];
            let grad_max = grad_pool[base + 2 * cache.hidden + h_idx];
            let mu = cache.mean[b * cache.hidden + h_idx];
            let sigma = cache.std[b * cache.hidden + h_idx].max(1.0e-6);
            let argmax = cache.max_index[b * cache.hidden + h_idx];
            for p in 0..cache.points {
                let idx = (b * cache.points + p) * cache.hidden + h_idx;
                grad[idx] += grad_mean * inv_points;
                grad[idx] += grad_std * (cache.input[idx] - mu) * inv_points / sigma;
                if p == argmax {
                    grad[idx] += grad_max;
                }
            }
        }
    }
    grad
}

fn cross_entropy(logits: &[f32], labels: &[usize]) -> CeOut {
    let batch = labels.len();
    let mut loss = 0.0;
    let mut grad = vec![0.0; logits.len()];
    let mut correct = 0;
    let mut entropy = 0.0;
    for b in 0..batch {
        let (p0, p1) = softmax2(logits[2 * b], logits[2 * b + 1]);
        let y = labels[b];
        loss += if y == 0 { -p0.ln() } else { -p1.ln() };
        correct += usize::from((p1 > p0) == (y == 1));
        entropy +=
            -(p0 * p0.max(1.0e-12).ln() + p1 * p1.max(1.0e-12).ln()) / std::f32::consts::LN_2;
        grad[2 * b] = (p0 - f32::from(y == 0)) / batch as f32;
        grad[2 * b + 1] = (p1 - f32::from(y == 1)) / batch as f32;
    }
    CeOut {
        loss: loss / batch as f32,
        grad_logits: grad,
        acc: correct as f32 / batch as f32,
        entropy: entropy / batch as f32,
    }
}

fn entropy_feature(logits: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0; logits.len() / 2];
    for b in 0..out.len() {
        let (p0, p1) = softmax2(logits[2 * b], logits[2 * b + 1]);
        out[b] = -(p0 * p0.max(1.0e-12).ln() + p1 * p1.max(1.0e-12).ln()) / std::f32::consts::LN_2;
    }
    out
}

fn softmax2(a: f32, b: f32) -> (f32, f32) {
    let m = a.max(b);
    let ea = (a - m).exp();
    let eb = (b - m).exp();
    (ea / (ea + eb), eb / (ea + eb))
}

fn train_model(
    mut model: HolonomySetNet,
    train: &Dataset,
    test: &Dataset,
    cfg: &Config,
    seed: u64,
) -> TaskResultPayload {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut indices: Vec<usize> = (0..train.samples).collect();
    for _ in 0..cfg.epochs {
        indices.shuffle(&mut rng);
        for chunk in indices.chunks(cfg.batch_size) {
            let (x, y) = gather_batch(train, chunk);
            model.train_batch(&x, &y, chunk.len(), train.points, cfg.lr);
        }
    }
    let train_metrics = evaluate(&model, train);
    let test_metrics = evaluate(&model, test);
    let timing = forward_timing(&model, test, 80);
    TaskResultPayload {
        train: train_metrics,
        test: test_metrics,
        timing,
        n_params: model.n_params(),
    }
}

#[derive(Clone, Debug)]
struct TaskResultPayload {
    train: Metrics,
    test: Metrics,
    timing: Timing,
    n_params: usize,
}

fn evaluate(model: &HolonomySetNet, data: &Dataset) -> Metrics {
    let logits = model.logits(&data.x, data.samples, data.points);
    let ce = cross_entropy(&logits, &data.y);
    Metrics {
        acc: ce.acc,
        loss: ce.loss,
        entropy: ce.entropy,
        clifford_error: clifford_probability_error(&logits, &data.y),
    }
}

fn forward_timing(model: &HolonomySetNet, data: &Dataset, repeats: usize) -> Timing {
    for _ in 0..20 {
        let _ = model.logits(&data.x, data.samples, data.points);
    }
    let mut values = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let start = Instant::now();
        let _ = model.logits(&data.x, data.samples, data.points);
        values.push(start.elapsed().as_secs_f64() * 1.0e6 / data.samples as f64);
    }
    values.sort_by(|a, b| a.total_cmp(b));
    Timing {
        median_us_per_sample: values[values.len() / 2],
        mean_us_per_sample: values.iter().sum::<f64>() / values.len() as f64,
        max_us_per_sample: values[values.len() - 1],
    }
}

pub fn make_dataset(task: Task, samples: usize, points: usize, seed: u64) -> Dataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut x = Vec::with_capacity(samples * points * 2);
    let mut y = Vec::with_capacity(samples);
    for sample in 0..samples {
        let label = sample % 2;
        y.push(label);
        match task {
            Task::Moons => sample_moons(label, points, &mut rng, &mut x),
            Task::Spiral => sample_spiral(label, points, &mut rng, &mut x),
            Task::Xor => sample_xor(label, points, &mut rng, &mut x),
        }
    }
    Dataset {
        x,
        y,
        samples,
        points,
    }
}

fn sample_moons(label: usize, points: usize, rng: &mut StdRng, out: &mut Vec<f32>) {
    for _ in 0..points {
        let theta = rng.random::<f32>() * std::f32::consts::PI;
        let (x, y) = if label == 0 {
            (theta.cos(), theta.sin())
        } else {
            (1.0 - theta.cos(), 0.45 - theta.sin())
        };
        out.push(x + normalish(rng) * 0.055);
        out.push(y + normalish(rng) * 0.055);
    }
}

fn sample_spiral(label: usize, points: usize, rng: &mut StdRng, out: &mut Vec<f32>) {
    for point in 0..points {
        let t =
            point as f32 / points as f32 * 3.5 * std::f32::consts::PI + rng.random::<f32>() * 0.15;
        let phase = if label == 0 {
            0.0
        } else {
            std::f32::consts::PI
        };
        let radius = 0.12 + 0.08 * t;
        out.push(radius * (t + phase).cos() + normalish(rng) * 0.04);
        out.push(radius * (t + phase).sin() + normalish(rng) * 0.04);
    }
}

fn sample_xor(label: usize, points: usize, rng: &mut StdRng, out: &mut Vec<f32>) {
    let centers = if label == 0 {
        [(-0.75, -0.75), (0.75, 0.75)]
    } else {
        [(-0.75, 0.75), (0.75, -0.75)]
    };
    for _ in 0..points {
        let (cx, cy) = centers[rng.random_range(0..2)];
        out.push(cx + normalish(rng) * 0.12);
        out.push(cy + normalish(rng) * 0.12);
    }
}

fn normalish(rng: &mut StdRng) -> f32 {
    let mut sum = 0.0;
    for _ in 0..6 {
        sum += rng.random::<f32>() - 0.5;
    }
    sum * 0.816_496_6
}

fn gather_batch(data: &Dataset, indices: &[usize]) -> (Vec<f32>, Vec<usize>) {
    let row = data.points * 2;
    let mut x = Vec::with_capacity(indices.len() * row);
    let mut y = Vec::with_capacity(indices.len());
    for &idx in indices {
        x.extend_from_slice(&data.x[idx * row..(idx + 1) * row]);
        y.push(data.y[idx]);
    }
    (x, y)
}

pub fn run_suite(cfg: &Config) -> Vec<TaskResult> {
    cfg.tasks
        .iter()
        .enumerate()
        .map(|(i, &task)| {
            let train = make_dataset(task, cfg.n_train, cfg.n_points, cfg.seed + i as u64 * 100);
            let test = make_dataset(
                task,
                cfg.n_test,
                cfg.n_points,
                cfg.seed + i as u64 * 100 + 1,
            );
            let model = HolonomySetNet::new(cfg.hidden, cfg.seed + i as u64 * 10);
            let payload = train_model(model, &train, &test, cfg, cfg.seed + i as u64);
            let materialized = cfg.n_test * cfg.n_points * (4 * cfg.hidden + 1) * size_of::<f32>();
            let fused = cfg.n_test * cfg.n_points * cfg.hidden * size_of::<f32>();
            TaskResult {
                task,
                train: payload.train,
                test: payload.test,
                timing: payload.timing,
                n_params: payload.n_params,
                param_bytes: payload.n_params * size_of::<f32>(),
                materialized_update_bytes: materialized,
                fused_update_bytes: fused,
            }
        })
        .collect()
}

fn write_json(path: &PathBuf, cfg: &Config, rows: &[TaskResult]) -> io::Result<()> {
    let mut file = File::create(path)?;
    writeln!(file, "{{")?;
    writeln!(file, "  \"engine\": \"hymeko_nagare\",")?;
    writeln!(
        file,
        "  \"feature_lift\": \"quaternion_periodic_holonomy\","
    )?;
    writeln!(
        file,
        "  \"error_metric\": \"clifford_probability_error_cl_2_0\","
    )?;
    writeln!(file, "  \"n_train\": {},", cfg.n_train)?;
    writeln!(file, "  \"n_test\": {},", cfg.n_test)?;
    writeln!(file, "  \"n_points\": {},", cfg.n_points)?;
    writeln!(file, "  \"hidden\": {},", cfg.hidden)?;
    writeln!(file, "  \"tasks\": [")?;
    for (idx, row) in rows.iter().enumerate() {
        let comma = if idx + 1 == rows.len() { "" } else { "," };
        writeln!(file, "    {{")?;
        writeln!(file, "      \"task\": \"{}\",", row.task.as_str())?;
        writeln!(file, "      \"test_acc\": {:.6},", row.test.acc)?;
        writeln!(file, "      \"test_loss\": {:.6},", row.test.loss)?;
        writeln!(file, "      \"test_entropy\": {:.6},", row.test.entropy)?;
        writeln!(
            file,
            "      \"test_clifford_error\": {:.6},",
            row.test.clifford_error
        )?;
        writeln!(
            file,
            "      \"median_us_per_sample\": {:.6},",
            row.timing.median_us_per_sample
        )?;
        writeln!(
            file,
            "      \"mean_us_per_sample\": {:.6},",
            row.timing.mean_us_per_sample
        )?;
        writeln!(
            file,
            "      \"max_us_per_sample\": {:.6},",
            row.timing.max_us_per_sample
        )?;
        writeln!(file, "      \"n_params\": {},", row.n_params)?;
        writeln!(file, "      \"param_bytes\": {},", row.param_bytes)?;
        writeln!(
            file,
            "      \"materialized_update_bytes\": {},",
            row.materialized_update_bytes
        )?;
        writeln!(
            file,
            "      \"fused_update_bytes\": {}",
            row.fused_update_bytes
        )?;
        writeln!(file, "    }}{comma}")?;
    }
    writeln!(file, "  ]")?;
    writeln!(file, "}}")?;
    Ok(())
}

fn parse_args() -> Result<Config, String> {
    let mut cfg = Config::default();
    let args: Vec<String> = env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        let key = &args[i];
        let value = args
            .get(i + 1)
            .ok_or_else(|| format!("missing value for {key}"))?;
        match key.as_str() {
            "--tasks" => {
                cfg.tasks = value
                    .split(',')
                    .map(Task::from_str)
                    .collect::<Result<Vec<_>, _>>()?
            }
            "--n-train" => cfg.n_train = value.parse().map_err(|_| "bad --n-train".to_string())?,
            "--n-test" => cfg.n_test = value.parse().map_err(|_| "bad --n-test".to_string())?,
            "--n-points" => {
                cfg.n_points = value.parse().map_err(|_| "bad --n-points".to_string())?
            }
            "--hidden" => cfg.hidden = value.parse().map_err(|_| "bad --hidden".to_string())?,
            "--epochs" => cfg.epochs = value.parse().map_err(|_| "bad --epochs".to_string())?,
            "--batch-size" => {
                cfg.batch_size = value.parse().map_err(|_| "bad --batch-size".to_string())?
            }
            "--lr" => cfg.lr = value.parse().map_err(|_| "bad --lr".to_string())?,
            "--seed" => cfg.seed = value.parse().map_err(|_| "bad --seed".to_string())?,
            "--out" => cfg.out = Some(PathBuf::from(value)),
            _ => return Err(format!("unknown flag {key}")),
        }
        i += 2;
    }
    Ok(cfg)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cfg = parse_args().map_err(io::Error::other)?;
    let rows = run_suite(&cfg);
    for row in &rows {
        println!(
            "{} acc={:.3} ce={:.4} clifford_error={:.4} median_us={:.2} fused_bytes={} materialized_bytes={}",
            row.task.as_str(),
            row.test.acc,
            row.test.loss,
            row.test.clifford_error,
            row.timing.median_us_per_sample,
            row.fused_update_bytes,
            row.materialized_update_bytes
        );
    }
    if let Some(path) = &cfg.out {
        write_json(path, &cfg, &rows)?;
    }
    Ok(())
}
