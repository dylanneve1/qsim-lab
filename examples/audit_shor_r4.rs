//! Independent-audit helper (exp/shor-r4-audit). Emits raw data for the
//! Python cross-checks in research/data/shor_r4_audit/:
//!   dump  N a w        — gate list of the windowed controlled-U_a (one gate per line)
//!   dist  N a w        — exact outcome distribution of the sliced windowed run (f64)
//!   trace N a w seed   — one sampled sliced run: support trace, measured y, gate·branch ops
use qsim_lab::shor::{self, sliced::SlicedState, Instance, Oracle, OrderFindingState};
use qsim_lab::Gate;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let n_mod: u64 = a[1].parse().unwrap();
    let base: u64 = a[2].parse().unwrap();
    let w: usize = a[3].parse().unwrap();
    match a[0].as_str() {
        "dump" => {
            let n = shor::work_bits(n_mod);
            let lay = qsim_lab::shor::window::WindowLayout::new(n, w);
            let c = qsim_lab::shor::window::controlled_ua(&lay, base, n_mod);
            println!("nq {} n {}", c.num_qubits, n);
            for op in &c.ops {
                match op {
                    qsim_lab::circuit::Op::Gate(Gate::X(t)) => println!("X {t}"),
                    qsim_lab::circuit::Op::Gate(Gate::Cnot(c, t)) => println!("CX {c} {t}"),
                    qsim_lab::circuit::Op::Gate(Gate::Ccx(c1, c2, t)) => {
                        println!("CCX {c1} {c2} {t}")
                    }
                    other => println!("OTHER {other:?}"),
                }
            }
        }
        "dist" => {
            let inst = Instance::new(n_mod, base, Oracle::Windowed(w));
            let d = shor::semiclassical_distribution(&inst, SlicedState::<f64>::new(&inst), 0.0);
            println!("t {}", inst.t);
            for (y, p) in d.iter().enumerate() {
                if *p > 0.0 {
                    println!("{y} {p:.17e}");
                }
            }
        }
        "trace" => {
            let seed: u64 = a[4].parse().unwrap();
            let inst = Instance::new(n_mod, base, Oracle::Windowed(w));
            let mut rng = StdRng::seed_from_u64(seed);
            let s = SlicedState::<f64>::new(&inst);
            // run manually so the support trace is accessible
            let mut st = s;
            let mut y = 0u128;
            let mut gates = Vec::new();
            for i in 0..inst.t {
                let (c, _) = shor::sliced::oracle_block(&inst, inst.mults[inst.t - 1 - i]);
                gates.push(c.ops.len());
                st.round(&inst, i, y);
                let p1 = st.prob_one(0);
                let u: f64 = rand::Rng::random(&mut rng);
                let bit = u < p1;
                st.collapse(0, bit);
                st.reset_control(bit);
                if bit {
                    y |= 1 << i;
                }
            }
            println!("t {} y {} ops {}", inst.t, y, st.work_ops());
            println!("support {:?}", st.support_trace);
            println!("gates {:?}", gates);
        }
        _ => panic!("unknown mode"),
    }
}
