//! Audit (PR 4, exp/qec): hook-safe X-check CNOT order, checked WITHOUT the
//! branch's fault list. Every single fault is injected as an explicit gate
//! into `build_circuit()`, the circuit runs noiselessly on the tableau, and
//! the shot is decoded with `sc.decoder`. (a) new order: every single fault
//! is corrected at d=3 and d=5. (b) old order NW,NE,SW,SE (rebuilt by
//! swapping the 2nd/3rd CNOT of every weight-4 X check): some hook fault has
//! the SAME Z-detector pattern as a single data-X fault but a DIFFERENT
//! logical flip, so no decoder can correct both (decoder-independent).
use qsim_lab::qec::SurfaceCode;
use qsim_lab::{Circuit, Gate, Op, Tableau};
use rand::rngs::StdRng;
use rand::SeedableRng;

fn with_inserted(c: &Circuit, after: usize, gs: &[Gate]) -> Circuit {
    let mut o = Circuit::new(c.num_qubits);
    for (i, op) in c.ops.iter().enumerate() {
        o.ops.push(op.clone());
        if i == after {
            for g in gs { o.ops.push(Op::Gate(*g)); }
        }
    }
    o
}

fn shot(sc: &SurfaceCode, c: &Circuit) -> (Vec<usize>, bool) {
    let mut t = Tableau::new(c.num_qubits);
    let mut rng = StdRng::seed_from_u64(1);
    let bits = c.run(&mut t, &mut rng).unwrap();
    sc.extract_z_defects(&bits)
}

fn paulis1(q: usize) -> Vec<Vec<Gate>> { vec![vec![Gate::X(q)], vec![Gate::Y(q)], vec![Gate::Z(q)]] }

fn faults(c: &Circuit) -> Vec<(usize, Vec<Gate>)> {
    let mut v = Vec::new();
    for (i, op) in c.ops.iter().enumerate() {
        match op {
            Op::Gate(Gate::Cnot(a, b)) => {
                for pa in 0..4 { for pb in 0..4 {
                    if pa == 0 && pb == 0 { continue; }
                    let mut g = Vec::new();
                    for (p, q) in [(pa, *a), (pb, *b)] {
                        match p { 1 => g.push(Gate::X(q)), 2 => g.push(Gate::Y(q)), 3 => g.push(Gate::Z(q)), _ => {} }
                    }
                    v.push((i, g));
                }}
            }
            Op::Gate(g) => for p in paulis1(g.qubits()[0]) { v.push((i, p)); },
            Op::Reset(q) => for p in paulis1(*q) { v.push((i, p)); },
            Op::Measure(q) if i > 0 => v.push((i - 1, vec![Gate::X(*q)])), // readout flip
            _ => {}
        }
    }
    v
}

fn old_order(c: &Circuit, x_anc: &[usize]) -> Circuit {
    let mut o = c.clone();
    let mut i = 0;
    while i + 3 < o.ops.len() {
        let ctl = |op: &Op| if let Op::Gate(Gate::Cnot(a, _)) = op { Some(*a) } else { None };
        if let Some(a) = ctl(&o.ops[i]) {
            if x_anc.contains(&a) && (1..4).all(|k| ctl(&o.ops[i + k]) == Some(a)) {
                o.ops.swap(i + 1, i + 2);
                i += 4;
                continue;
            }
        }
        i += 1;
    }
    o
}

#[test]
fn every_single_fault_corrected_new_order() {
    for d in [3usize, 5] {
        let sc = SurfaceCode::new(d, d);
        let c = sc.build_circuit();
        let base = shot(&sc, &c);
        assert!(base.0.is_empty() && !base.1, "noiseless run has defects");
        let fs = faults(&c);
        let mut bad = Vec::new();
        for (i, g) in &fs {
            let (def, flip) = shot(&sc, &with_inserted(&c, *i, g));
            if flip ^ sc.decoder.decode(&def) { bad.push((*i, g.clone())); }
        }
        eprintln!("d={d}: {} single faults injected, {} uncorrected", fs.len(), bad.len());
        assert!(bad.is_empty(), "d={d} uncorrected single faults: {:?}", &bad[..bad.len().min(10)]);
    }
}

#[test]
fn old_order_hook_is_uncorrectable_new_order_is_not() {
    let d = 3;
    let sc = SurfaceCode::new(d, d);
    let x_anc: Vec<usize> = (0..sc.x_stabilizers.len()).map(|k| SurfaceCode::x_ancilla_idx(d, k)).collect();
    let new = sc.build_circuit();
    let old = old_order(&new, &x_anc);
    assert_ne!(old.ops, new.ops);
    let count = |c: &Circuit| {
        // hook faults: X (or Y) on an X ancilla right after its 2nd CNOT of a weight-4 check
        let mut twins = 0;
        let mut hooks = 0;
        for i in 1..c.ops.len().saturating_sub(2) {
            let (Op::Gate(Gate::Cnot(a, _)), Op::Gate(Gate::Cnot(a0, _)), Op::Gate(Gate::Cnot(a2, _))) = (&c.ops[i], &c.ops[i - 1], &c.ops[i + 1]) else { continue };
            if !(x_anc.contains(a) && a0 == a && a2 == a) { continue; }
            if let Some(Op::Gate(Gate::Cnot(a3, _))) = c.ops.get(i + 2) { if a3 != a { continue; } } else { continue; }
            if i >= 2 && matches!(c.ops[i - 2], Op::Gate(Gate::Cnot(x, _)) if x == *a) { continue; } // i must be the 2nd CNOT
            hooks += 1;
            let h = shot(&sc, &with_inserted(c, i, &[Gate::X(*a)]));
            // a single data X fault at the same time with identical detectors but other logical?
            for dq in 0..d * d {
                let s = shot(&sc, &with_inserted(c, i, &[Gate::X(dq)]));
                if s.0 == h.0 && s.1 != h.1 { twins += 1; break; }
            }
        }
        (hooks, twins)
    };
    let (hn, tn) = count(&new);
    let (ho, to) = count(&old);
    eprintln!("new order: {hn} hooks, {tn} indistinguishable from a data fault with other logical; old order: {ho} hooks, {to}");
    assert!(hn > 0 && tn == 0, "new order has uncorrectable hooks");
    assert!(to > 0, "old order hook claim not reproduced");
}

/// Earlier BUG 2: the DEM path must use the caller's noise (no baked p=1).
#[test]
fn dem_uses_callers_noise() {
    use qsim_lab::noise::NoiseModel;
    use qsim_lab::qec::SamplingMethod;
    let sc = SurfaceCode::new(3, 3);
    let mut rng = StdRng::seed_from_u64(3);
    let s = sc.dem_sampler(&NoiseModel::none());
    for _ in 0..2000 { let (d, f) = s.sample(&mut rng); assert!(d.is_empty() && !f); }
    let r = sc.run_experiment(&NoiseModel::none(), 2000, SamplingMethod::DetectorErrorModel, &mut rng);
    eprintln!("noiseless DEM run_experiment: {r:?}");
    let r = sc.run_experiment(&NoiseModel::none(), 2000, SamplingMethod::Tableau, &mut rng);
    eprintln!("noiseless tableau run_experiment: {r:?}");
}

/// d=5: random pairs of single faults (weight 2 < d/2 = 2.5) must be corrected.
#[test]
fn random_fault_pairs_corrected_d5() {
    use rand::Rng;
    let d = 5;
    let sc = SurfaceCode::new(d, d);
    let c = sc.build_circuit();
    let fs = faults(&c);
    let mut rng = StdRng::seed_from_u64(99);
    let mut bad = 0;
    let trials = 3000;
    for _ in 0..trials {
        let (i, g) = &fs[rng.random_range(0..fs.len())];
        let (j, h) = &fs[rng.random_range(0..fs.len())];
        let c1 = with_inserted(&c, *i, g);
        // index shift: insertion after i moves later ops by g.len()
        let j2 = if *j > *i { *j + g.len() } else { *j };
        let c2 = with_inserted(&c1, j2, h);
        let (def, flip) = shot(&sc, &c2);
        if flip ^ sc.decoder.decode(&def) { bad += 1; eprintln!("uncorrected pair: {i} {g:?} / {j} {h:?}"); }
    }
    eprintln!("d=5: {trials} random fault pairs, {bad} uncorrected");
    assert_eq!(bad, 0);
}
