//! Exact reference amplitudes for the CAMPS experiments (exp/camps).
//!
//! ```text
//! cargo run --release --example camps_ref -- --n 70 --d 40 --xs xs.txt [--qasm F] [--prec 32|64]
//! ```
//! `xs.txt`: one bitstring per line, character `i` = qubit `i` (q0 first).
//! Prints `index re im seconds` per line, using the chain-sweep CPU engine on
//! the circuit truncated to CZ-depth `d` (same truncation as `chain_sweep`).

use qsim_lab::circuit::Circuit;
use qsim_lab::engines::chain_sweep::{self, compile, truncate, ChainCircuit};
use std::collections::HashMap;
use std::io::Write;
use std::time::Instant;

const QASM: &str = include_str!("../research/chain-sweep/nq70_depth70_checks27_doped.qasm");

fn arg<T: std::str::FromStr>(a: &[String], k: &str, d: T) -> T {
    let key = format!("--{k}");
    a.iter()
        .position(|x| *x == key)
        .and_then(|i| a.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(d)
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let n: usize = arg(&a, "n", 70);
    let d: usize = arg(&a, "d", 70);
    let prec: usize = arg(&a, "prec", 32);
    let src = match a.iter().position(|x| x == "--qasm") {
        Some(i) => std::fs::read_to_string(&a[i + 1]).expect("read qasm"),
        None => QASM.to_string(),
    };
    let c = truncate(&Circuit::from_qasm(&src).expect("parse qasm"), n, d);
    let cc = ChainCircuit::from_circuit(&c).unwrap();
    let xs_path: String = arg(&a, "xs", "xs.txt".to_string());
    let text = std::fs::read_to_string(&xs_path).expect("read xs");
    let cfg = Default::default();
    let out = std::io::stdout();
    for (i, line) in text.lines().filter(|l| !l.trim().is_empty()).enumerate() {
        let s = line.trim().as_bytes();
        assert_eq!(s.len(), n, "bitstring length");
        let mut x = 0u128;
        for (q, &ch) in s.iter().enumerate() {
            if ch == b'1' {
                x |= 1 << q;
            }
        }
        let t = Instant::now();
        let plan = compile(&cc, x, &HashMap::new());
        let amp = if prec == 64 {
            chain_sweep::amplitude_cpu::<f64>(&plan, &cfg).unwrap()
        } else {
            chain_sweep::amplitude_cpu::<f32>(&plan, &cfg).unwrap()
        };
        let mut o = out.lock();
        writeln!(
            o,
            "{i} {:.12e} {:.12e} {:.3}",
            amp.re,
            amp.im,
            t.elapsed().as_secs_f64()
        )
        .unwrap();
        o.flush().unwrap();
    }
}
