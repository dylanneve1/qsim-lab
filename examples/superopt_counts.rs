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
    if a.get(4).is_some_and(|s| s == "blocks") {
        // per-block counts for this n: lookups over all windows of the
        // round-0 multiplier, and one modular addition
        use qsim_lab::circuit::Circuit;
        use qsim_lab::shor_superopt::{add_mod_reg, lookup_unary, FanoutPlan};
        let inst = Instance::new(n_mod, base, Oracle::Windowed(w));
        let lay = WindowLayout::new(inst.m, w);
        let n = inst.m;
        let mut bse = inst.mults[0] % n_mod;
        let (mut lk, mut start) = ([(0usize, 0usize); 3], 0);
        let mut nwin = 0;
        while start < n {
            let ww = w.min(n - start);
            let table: Vec<u64> = (0..1u64 << ww)
                .map(|v| (u128::from(v) * u128::from(bse) % u128::from(n_mod)) as u64)
                .collect();
            let addr = &lay.x[start..start + ww];
            for (k, mode) in [0, 1, 2].iter().enumerate() {
                let mut c = Circuit::new(lay.num_qubits());
                match mode {
                    0 => qsim_lab::shor_window::lookup(&mut c, 0, addr, &lay.and, &lay.l, &table),
                    1 => lookup_unary(&mut c, 0, addr, &lay.and, &lay.l, &FanoutPlan::leaves(&table)),
                    _ => lookup_unary(&mut c, 0, addr, &lay.and, &lay.l, &FanoutPlan::optimal(&table, n)),
                }
                let (g, t) = gate_counts(&c);
                lk[k].0 += g;
                lk[k].1 += t;
            }
            for _ in 0..ww {
                bse = (u128::from(bse) * 2 % u128::from(n_mod)) as u64;
            }
            start += ww;
            nwin += 1;
        }
        for (k, name) in ["lookup baseline (LSB-first chain)", "lookup unary iteration", "lookup unary + optimal fanout"].iter().enumerate() {
            println!("{name:36} per lookup avg over {nwin} windows: gates={:.1} ccx={:.1}", lk[k].0 as f64 / nwin as f64, lk[k].1 as f64 / nwin as f64);
        }
        let b = Opts::BASELINE;
        for (name, o) in [
            ("modadd baseline (5 adders)", b),
            ("modadd comparator", Opts { comparator: true, ..b }),
            ("modadd comparator+kflip", Opts { comparator: true, kflip: true, ..b }),
        ] {
            let mut c = Circuit::new(lay.num_qubits());
            add_mod_reg(&mut c, &lay, n_mod, &o);
            let (g, t) = gate_counts(&c);
            let p = qsim_lab::shor_superopt::reversible_peephole(&c);
            let (gp, tp) = gate_counts(&p);
            let sp = qsim_lab::shor_superopt::sat_peephole(&p, lay.k[0]);
            let (gs, ts) = gate_counts(&qsim_lab::shor_superopt::reversible_peephole(&sp));
            println!("{name:36} gates={g} ccx={t}  | +peephole {gp}/{tp}  | +sat rules {gs}/{ts}");
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
