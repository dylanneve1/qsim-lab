//! Which top-level operation first raises the logical-fault flag.
use qsim_lab::ft::backends::FrameBackend;
use qsim_lab::ft::core::Noise;
use qsim_lab::ft::logical::{Encoded, Logical, MagicMode};
use qsim_lab::ft::machine::{ideal_logical, FtConfig};
use qsim_lab::ft::shor::{run_shor15, NLOG15};
use std::collections::BTreeMap;

struct W {
    e: Encoded<FrameBackend>,
    op: usize,
    first: Option<(usize, String)>,
}
impl W {
    fn chk(&mut self, name: &str) {
        self.op += 1;
        if self.first.is_some() {
            return;
        }
        let mut bad = self.e.counts.logical_fault;
        let mut which = String::new();
        for (i, &b) in self.e.blocks.iter().enumerate() {
            let l = ideal_logical(&self.e.m.b.frame, self.e.k, b);
            if l != (false, false) {
                bad = true;
                which = format!("blk{i}:{l:?}");
            }
        }
        if bad {
            self.first = Some((self.op, format!("{name} {which} measflip={}", self.e.counts.logical_fault)));
        }
    }
}
impl Logical for W {
    fn prep(&mut self, q: usize, b: bool) { self.e.prep(q, b); self.chk(&format!("prep{q}")) }
    fn h(&mut self, q: usize) { self.e.h(q); self.chk(&format!("h{q}")) }
    fn s(&mut self, q: usize) { self.e.s(q); self.chk(&format!("s{q}")) }
    fn sdg(&mut self, q: usize) { self.e.sdg(q); self.chk(&format!("sdg{q}")) }
    fn t(&mut self, q: usize) { self.e.t(q); self.chk(&format!("t{q}")) }
    fn tdg(&mut self, q: usize) { self.e.tdg(q); self.chk(&format!("tdg{q}")) }
    fn cnot(&mut self, c: usize, t: usize) { self.e.cnot(c, t); self.chk(&format!("cnot{c}{t}")) }
    fn meas(&mut self, q: usize) -> bool { let r = self.e.meas(q); self.chk(&format!("meas{q}")); r }
    fn sdg_slot(&mut self, q: usize, a: bool) { self.e.sdg_slot(q, a); self.chk(&format!("slot{q}")) }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let k: usize = a[1].parse().unwrap();
    let p: f64 = a[2].parse().unwrap();
    let n: u64 = a[3].parse().unwrap();
    let mut hist: BTreeMap<String, u64> = BTreeMap::new();
    for s in 0..n {
        let mut w = W { e: Encoded::frame(k, NLOG15, Noise::new(p, 77 + s), FtConfig::default(), MagicMode::Model(0.0), s), op: 0, first: None };
        run_shor15(&mut w, 7, 3);
        if let Some((op, d)) = w.first {
            println!("shot {s}: op {op} {d}");
            let key = d.split_whitespace().next().unwrap().to_string();
            *hist.entry(key).or_default() += 1;
        }
    }
    println!("{hist:?}");
}
