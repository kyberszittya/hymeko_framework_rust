#[derive(Clone, Copy, Debug)]
pub enum SignAgg {
    PreferNonNeutral, // a mostani logika
    WeightedVote,     // később
    Channels3,        // később: + / - / 0 külön csatorna
}

#[derive(Clone, Copy, Debug)]
pub enum WeightAgg {
    Sum,
    Max,
    ProbSum01,   // 1 - Π(1-w)  (feltételez [0,1])
    LukasSat01,  // min(1, a+b)
}

#[derive(Clone, Copy, Debug)]
pub struct AggCfg {
    pub sign: SignAgg,
    pub weight: WeightAgg,
    pub clamp01: bool, // ha ProbSum/Lukas, érdemes
}

#[inline(always)]
pub fn agg_sign(cfg: &AggCfg, a: i8, b: i8, wa: f32, wb: f32) -> i8 {
    match cfg.sign {
        SignAgg::PreferNonNeutral => {
            if a == b { return a; }
            if a == 0 { return b; }
            if b == 0 { return a; }
            0
        }
        SignAgg::WeightedVote => {
            let s = (a as f32) * wa + (b as f32) * wb;
            if s > 0.0 { 1 } else if s < 0.0 { -1 } else { 0 }
        }
        SignAgg::Channels3 => {
            // ezt majd nem i8-ban tároljuk, hanem 3 csatornában.
            // itt most fallback:
            if a == b { a } else { 0 }
        }
    }
}

#[inline(always)]
pub fn clamp01(x: f32) -> f32 { x.max(0.0).min(1.0) }

#[inline(always)]
pub fn agg_weight(cfg: &AggCfg, a: f32, b: f32) -> f32 {
    let mut out = match cfg.weight {
        WeightAgg::Sum => a + b,
        WeightAgg::Max => a.max(b),
        WeightAgg::ProbSum01 => 1.0 - (1.0 - a) * (1.0 - b),
        WeightAgg::LukasSat01 => (a + b).min(1.0),
    };
    if cfg.clamp01 { out = clamp01(out); }
    out
}