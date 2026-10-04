//! Gate / Toffoli counts of the windowed Shor oracle with each
//! superoptimisation (exp/superopt) switched on alone and all together,
//! summed over all `2n` controlled-U rounds of a run (the same circuits the
//! sliced engine evaluates), plus what the generic commutation-aware
//! peephole pass (`compile::peephole`) still removes afterwards.
//!
//! usage: superopt_counts N a w [peephole]
use qsim_lab::compile::peephole;
use qsim_lab::shor::{Instance, Oracle};
use qsim_lab::shor_ripple::gate_counts;
use qsim_lab::shor_superopt::{controlled_ua, Opts};
use qsim_lab::shor_window::WindowLayout;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let n_mod: u64 = a[1].parse().unwrap();
    let base: u64 = a[2].parse().unwrap();
    let w: usize = a[3].parse().unwrap();
    let peep = a.get(4).is_some_and(|s| s == "peephole");
    if a.get(4).is_some_and(|s| s == "dump") {
        // dump round-0 controlled-U (baseline and all) as text gate lists
        let inst = Instance::new(n_mod, base, Oracle::Windowed(w));
        let lay = WindowLayout::new(inst.m, w);
        let nosat = Opts { sat_rules: false, ..Opts::ALL };
        for (name, o) in [("baseline", Opts::BASELINE), ("all", Opts::ALL), ("allnosat", nosat)] {
            let c = controlled_ua(&lay, inst.mults[0], n_mod, &o);
            let mut out = format!("# N={n_mod} mult={} n={} w={w} qubits={} ancillas_from={}\n",
                inst.mults[0], inst.m, lay.num_qubits(), inst.m + 1);
            for g in c.gates() {
                use qsim_lab::gate::Gate::*;
                out += &match *g {
                    X(q) => format!("X {q}\n"),
                    Cnot(c, t) => format!("CX {c} {t}\n"),
                    Ccx(a, b, t) => format!("CCX {a} {b} {t}\n"),
                    _ => unreachable!(),
                };
            }
            let path = format!("{}_{name}.txt", a[5]);
            std::fs::write(&path, out).unwrap();
            println!("wrote {path}");
        }
        return;
    }
    if a.get(4).is_some_and(|s| s == "sweep") {
        // window sweep, baseline vs all, totals over the run
        let inst = Instance::new(n_mod, base, Oracle::Windowed(w));
        for ww in 1..=8usize {
            let lay = WindowLayout::new(inst.m, ww);
            let mut row = format!("n={} w={ww} qubits={}", inst.m, lay.num_qubits());
            let dp = Opts { window_dp: true, ..Opts::ALL };
            for (name, o) in [("baseline", Opts::BASELINE), ("all", Opts::ALL), ("all+dp", dp)] {
                let (mut g, mut t) = (0, 0);
                for &mult in &inst.mults {
                    let (gg, tt) = gate_counts(&controlled_ua(&lay, mult, n_mod, &o));
                    g += gg;
                    t += tt;
                }
                row += &format!("  {name}: total={g} ccx={t}");
            }
            println!("{row}");
        }
        return;
    }
    let inst = Instance::new(n_mod, base, Oracle::Windowed(w));
    let lay = WindowLayout::new(inst.m, w);
    let b = Opts::BASELINE;
    let variants: Vec<(&str, Opts)> = vec![
        ("baseline", b),
        ("unary", Opts { unary: true, ..b }),
        ("unary+fanout", Opts { unary: true, fanout: true, ..b }),
        ("comparator", Opts { comparator: true, ..b }),
        ("kflip", Opts { kflip: true, ..b }),
        ("direct_first", Opts { direct_first: true, ..b }),
        ("unary+keep_chain", Opts { unary: true, keep_chain: true, ..b }),
        ("peephole only", Opts { peephole: true, ..b }),
        ("all but peephole", Opts { peephole: false, ..Opts::ALL }),
        ("all but sat_rules", Opts { sat_rules: false, ..Opts::ALL }),
        ("all + window_dp", Opts { window_dp: true, ..Opts::ALL }),
        ("all", Opts::ALL),
    ];
    println!(
        "N={n_mod} a={base} n={} w={w} qubits={} rounds={}",
        inst.m,
        lay.num_qubits(),
        inst.t
    );
    let mut base_g = 0usize;
    for (name, o) in &variants {
        let (mut g, mut t, mut gp, mut tp, mut g0, mut t0) = (0, 0, 0, 0, 0, 0);
        let (mut x, mut cx) = (0usize, 0usize);
        for (k, &mult) in inst.mults.iter().enumerate() {
            let c = controlled_ua(&lay, mult, n_mod, o);
            let (gg, tt) = gate_counts(&c);
            if k == 0 {
                (g0, t0) = (gg, tt);
            }
            g += gg;
            t += tt;
            for gate in c.gates() {
                match gate {
                    qsim_lab::gate::Gate::X(_) => x += 1,
                    qsim_lab::gate::Gate::Cnot(..) => cx += 1,
                    _ => {}
                }
            }
            if peep {
                let p = peephole::optimize(&c).circuit;
                let (pg, pt) = gate_counts(&p);
                gp += pg;
                tp += pt;
            }
        }
        if *name == "baseline" {
            base_g = g;
        }
        print!(
            "{name:18} total={g:9} ccx={t:8} cnot={cx:8} x={x:7} round0={g0}/{t0} ({:+.1}% gates)",
            100.0 * (g as f64 / base_g as f64 - 1.0)
        );
        if peep {
            print!("  | after peephole: {gp} / {tp} ccx");
        }
        println!();
    }
}
