//! Two-block (BB / GB / coprime-BB) codes: parameters, search, circuit LER.
//!
//! ```text
//! bb_codes params <l> <m> <A> <B> [max_weight] [node_limit]
//! ```
use qsim_lab::qec::bicycle::{DistanceOpts, TwoBlockCode};
use std::time::Instant;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    match a[1].as_str() {
        "params" => {
            let l: usize = a[2].parse().unwrap();
            let m: usize = a[3].parse().unwrap();
            let c = TwoBlockCode::parse(l, m, &a[4], &a[5]);
            let mut o = DistanceOpts::default();
            if let Some(w) = a.get(6) {
                o.max_weight = w.parse().unwrap();
            }
            if let Some(w) = a.get(7) {
                o.node_limit = w.parse().unwrap();
            }
            let t = Instant::now();
            let k = c.k();
            let tk = t.elapsed().as_secs_f64();
            let t = Instant::now();
            let d = c.distance(&o);
            println!(
                "{{\"l\":{l},\"m\":{m},\"A\":\"{}\",\"B\":\"{}\",\"n\":{},\"k\":{k},\"d_lower\":{},\"d_upper\":{},\"nodes\":{},\"k_s\":{tk:.6},\"d_s\":{:.3}}}",
                a[4], a[5], c.n(), d.lower, d.upper, d.nodes, t.elapsed().as_secs_f64()
            );
        }
        x => panic!("unknown command {x}"),
    }
}
