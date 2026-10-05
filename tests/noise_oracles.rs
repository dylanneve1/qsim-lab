//! Differential tests of the generic noisy engine (`shor::noisy_gen`) on
//! the measurement-based oracles, against an independent gate-by-gate
//! reference on the sparse state vector:
//!
//! * the reference applies every resolved op as a real gate (`Z`, `CZ`,
//!   `SWAP` included), every fault as a real Pauli gate, and every X-basis
//!   measurement as `H`, a projective measurement onto the projection
//!   outcome (recorded outcome XOR readout fault) with its true probability
//!   `P`, multiplying the path weight by `2P`, and a reset (`X` if the
//!   projection was 1, another `X` for a reset fault);
//! * the engine's exact weighted distribution of the recorded integer
//!   (whole control-measurement tree) must equal the reference's to 1e-12,
//!   for the MBU oracles (N = 15, 21, 33), the windowed-opt oracle, and
//!   random op streams that make branches collide (measured qubits that are
//!   not a function of the rest), where the weights differ from 1.

use qsim_lab::shor::noisy::{Fault, NoiseKind, Pauli, Site};
use qsim_lab::shor::noisy_gen::{self, GenCircuit, NOp, ResetMode, Resolved, Round, K192};
use qsim_lab::shor::{Instance, Oracle};
use qsim_lab::{Gate, SparseState};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn pauli_gate(p: Pauli, q: usize) -> Gate {
    match p {
        Pauli::X => Gate::X(q),
        Pauli::Y => Gate::Y(q),
        Pauli::Z => Gate::Z(q),
    }
}

/// One round on the sparse reference. Returns `(P(1), readout flipped,
/// weight factor)` or `None` if some projection had probability 0.
fn ref_round(
    s: &mut SparseState,
    res: &Resolved,
    i: usize,
    y_low: u128,
    fs: &[Fault],
) -> Option<(f64, bool, f64)> {
    let at = |site: Site| fs.iter().find(|f| f.site == site).map(|f| f.pauli);
    let mut wf = 1.0;
    if at(Site::Prep).is_some() {
        s.apply_gate(&Gate::X(0)).unwrap();
    }
    s.apply_gate(&Gate::H(0)).unwrap();
    if let Some(p) = at(Site::H1) {
        s.apply_gate(&pauli_gate(p, 0)).unwrap();
    }
    for (gi, op) in res.rounds[i].ops.iter().enumerate() {
        let slot_fault = |slot: u8| {
            at(Site::Gate {
                gate: gi as u32,
                slot,
            })
        };
        match *op {
            NOp::G(g) => {
                s.apply_gate(&g).unwrap();
                for (slot, q) in g.qubits().into_iter().enumerate() {
                    if let Some(p) = slot_fault(slot as u8) {
                        s.apply_gate(&pauli_gate(p, q)).unwrap();
                    }
                }
            }
            NOp::Ctrl(..) => unreachable!("windowed rounds use ref_distribution_windowed"),
            NOp::ResetZ(g) => {
                let grp = &res.rounds[i].groups[g as usize];
                // joint value of the group: the most probable one (ties: the
                // smallest), weight P(value), as noisy_gen::argmax_choice
                let mut mass: std::collections::BTreeMap<u64, f64> = Default::default();
                for (k, a) in s.iter() {
                    let v = grp.iter().fold(0u64, |acc, &q| acc | (k & (1u64 << q)));
                    *mass.entry(v).or_insert(0.0) += a.norm_sqr();
                }
                let tot: f64 = mass.values().sum();
                let mx = mass.values().copied().fold(0.0, f64::max);
                let (&v, &pv) = mass.iter().find(|(_, &x)| x >= mx - 1e-9 * tot).unwrap();
                wf *= pv / tot;
                for &q in grp {
                    let b = (v >> q) & 1 == 1;
                    s.collapse(q as usize, b);
                    if b {
                        s.apply_gate(&Gate::X(q as usize)).unwrap();
                    }
                }
                for (slot, &q) in grp.iter().enumerate() {
                    if slot_fault(slot as u8).is_some() {
                        s.apply_gate(&Gate::X(q as usize)).unwrap();
                    }
                }
            }
            NOp::MeasX(q, m) => {
                let q = q as usize;
                let proj = m ^ slot_fault(0).is_some();
                s.apply_gate(&Gate::H(q)).unwrap();
                let p1 = s.prob_one(q);
                let p = if proj { p1 } else { 1.0 - p1 };
                if p < 1e-14 {
                    return None;
                }
                s.collapse(q, proj);
                wf *= 2.0 * p;
                if proj {
                    s.apply_gate(&Gate::X(q)).unwrap();
                }
                if slot_fault(1).is_some() {
                    s.apply_gate(&Gate::X(q)).unwrap();
                }
            }
        }
    }
    if y_low != 0 {
        s.apply_gate(&Gate::Phase(0, Instance::correction(i, y_low)))
            .unwrap();
    }
    if let Some(p) = at(Site::Phase) {
        s.apply_gate(&pauli_gate(p, 0)).unwrap();
    }
    s.apply_gate(&Gate::H(0)).unwrap();
    if let Some(p) = at(Site::H2) {
        s.apply_gate(&pauli_gate(p, 0)).unwrap();
    }
    Some((s.prob_one(0), at(Site::Meas).is_some(), wf))
}

fn ref_distribution(gc: &GenCircuit, res: &Resolved, faults: &[Fault]) -> Vec<f64> {
    let t = gc.inst.t;
    let mut out = vec![0.0; 1 << t];
    #[allow(clippy::too_many_arguments)]
    fn walk(
        s: SparseState,
        gc: &GenCircuit,
        res: &Resolved,
        faults: &[Fault],
        i: usize,
        y: u128,
        p: f64,
        out: &mut [f64],
    ) {
        if i == gc.inst.t {
            out[y as usize] += p;
            return;
        }
        let fs: Vec<Fault> = faults
            .iter()
            .filter(|f| f.round as usize == i)
            .copied()
            .collect();
        let mut s = s;
        let Some((p1, flip, wf)) = ref_round(&mut s, res, i, y, &fs) else {
            return;
        };
        for bit in [false, true] {
            let pb = if bit { p1 } else { 1.0 - p1 };
            if pb <= 1e-14 {
                continue;
            }
            let mut c = s.clone();
            c.collapse(0, bit);
            if bit {
                c.apply_gate(&Gate::X(0)).unwrap();
            }
            let rec = u128::from(bit ^ flip) << i;
            walk(c, gc, res, faults, i + 1, y | rec, p * pb * wf, out);
        }
    }
    let mut s = SparseState::new(gc.nq);
    s.apply_gate(&Gate::X(1)).unwrap();
    walk(s, gc, res, faults, 0, 0, 1.0, &mut out);
    out
}

fn assert_close(a: &[f64], b: &[f64], what: &str) {
    let worst = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max);
    assert!(worst < 1e-12, "{what}: max |diff| = {worst:e}");
}

/// Faults at every site kind of a round, at measurements in particular.
fn special_patterns(res: &Resolved, rng: &mut StdRng) -> Vec<Vec<Fault>> {
    let t = res.rounds.len();
    let mut pats = Vec::new();
    for &i in &[0usize, t / 2, t - 1] {
        for site in [Site::Prep, Site::Meas] {
            pats.push(vec![Fault {
                round: i as u32,
                site,
                pauli: Pauli::X,
            }]);
        }
        for site in [Site::H1, Site::Phase, Site::H2] {
            for pauli in [Pauli::X, Pauli::Y, Pauli::Z] {
                pats.push(vec![Fault {
                    round: i as u32,
                    site,
                    pauli,
                }]);
            }
        }
        let r = &res.rounds[i];
        let meas: Vec<usize> = (0..r.ops.len())
            .filter(|&g| matches!(r.ops[g], NOp::MeasX(..)))
            .collect();
        if meas.is_empty() {
            continue;
        }
        for _ in 0..2 {
            let g = meas[rng.random_range(0..meas.len())];
            // readout flip, reset flip, and every Pauli on the measured
            // qubit right before the measurement (after the previous op on it)
            for slot in [0u8, 1] {
                pats.push(vec![Fault {
                    round: i as u32,
                    site: Site::Gate {
                        gate: g as u32,
                        slot,
                    },
                    pauli: Pauli::X,
                }]);
            }
            let NOp::MeasX(q, _) = r.ops[g] else {
                unreachable!()
            };
            if let Some(prev) = (0..g).rev().find(|&h| match r.ops[h] {
                NOp::G(gg) => gg.qubits().contains(&(q as usize)),
                NOp::MeasX(qq, _) => qq == q,
                NOp::ResetZ(_) | NOp::Ctrl(..) => false,
            }) {
                if let NOp::G(gg) = r.ops[prev] {
                    let slot = gg.qubits().iter().position(|&x| x == q as usize).unwrap();
                    for pauli in [Pauli::X, Pauli::Y, Pauli::Z] {
                        pats.push(vec![Fault {
                            round: i as u32,
                            site: Site::Gate {
                                gate: prev as u32,
                                slot: slot as u8,
                            },
                            pauli,
                        }]);
                    }
                }
            }
        }
    }
    pats
}

#[test]
fn mbu_oracles_fixed_patterns_match_sparse_reference() {
    let mut rng = StdRng::seed_from_u64(2024);
    let mut checked = 0;
    // (N, a, oracle, resolved streams, random patterns per stream, special patterns)
    for (n_mod, a, oracle, streams, nrand, special) in [
        (15u64, 7u64, Oracle::WindowedMbu(2), 2, 8, true),
        (15, 2, Oracle::WindowedMbuLookup(2), 2, 8, true),
        (21, 2, Oracle::WindowedMbu(2), 1, 4, false),
        (21, 5, Oracle::WindowedMbuLookup(3), 1, 4, false),
        (15, 7, Oracle::WindowedOpt(2), 1, 6, true),
    ] {
        let inst = Instance::new(n_mod, a, oracle);
        for kind in [
            NoiseKind::Depolarizing,
            NoiseKind::BitFlip,
            NoiseKind::PhaseFlip,
        ] {
            let gc = GenCircuit::new(&inst, kind);
            for _ in 0..streams {
                let res = gc.resolve_rng(&mut rng);
                let mut pats = if special && kind == NoiseKind::Depolarizing {
                    special_patterns(&res, &mut rng)
                } else {
                    Vec::new()
                };
                for k in [0usize, 1, 1, 1, 2, 2, 3, 4].into_iter().take(nrand) {
                    pats.push(res.sample_k(k, &mut rng));
                }
                for f in &pats {
                    let e = noisy_gen::trajectory_distribution::<u128>(&gc, &res, f);
                    let e2 = noisy_gen::trajectory_distribution::<K192>(&gc, &res, f);
                    let r = ref_distribution(&gc, &res, f);
                    assert_close(&e, &r, &format!("N={n_mod} {oracle:?} {kind:?} {f:?}"));
                    assert_close(&e, &e2, "u128 vs K192");
                    checked += 1;
                }
            }
        }
    }
    eprintln!("{checked} fixed fault patterns agree");
}

#[test]
fn random_streams_with_collisions_match_sparse_reference() {
    // random X/CNOT/CCX/Z/CZ/SWAP/MeasX programs on 8 qubits (control 0,
    // work 1..=4, ancillas 5..=7); measured qubits are often not functions
    // of the rest, so branches collide and the weights differ from 1
    let inst = Instance::new(15, 7, Oracle::Windowed(1));
    let nq = 8;
    let mut rng = StdRng::seed_from_u64(99);
    let mut nontrivial = 0;
    for trial in 0..40 {
        let kind = [
            NoiseKind::Depolarizing,
            NoiseKind::BitFlip,
            NoiseKind::PhaseFlip,
        ][trial % 3];
        let rounds: Vec<Round> = (0..inst.t)
            .map(|_| {
                let mut ops = Vec::new();
                for _ in 0..12 {
                    let q = |rng: &mut StdRng| rng.random_range(0..nq);
                    let op = match rng.random_range(0..9) {
                        0 => NOp::G(Gate::X(q(&mut rng))),
                        1 | 2 => {
                            let (a, b) = (q(&mut rng), q(&mut rng));
                            if a == b {
                                NOp::G(Gate::X(a))
                            } else {
                                NOp::G(Gate::Cnot(a, b))
                            }
                        }
                        3 => {
                            let (a, b, c) = (q(&mut rng), q(&mut rng), q(&mut rng));
                            if a == b || b == c || a == c {
                                NOp::G(Gate::Z(a))
                            } else {
                                NOp::G(Gate::Ccx(a, b, c))
                            }
                        }
                        4 => NOp::G(Gate::Z(q(&mut rng))),
                        5 => {
                            let (a, b) = (q(&mut rng), q(&mut rng));
                            if a == b {
                                NOp::G(Gate::Z(a))
                            } else if rng.random() {
                                NOp::G(Gate::Cz(a, b))
                            } else {
                                NOp::G(Gate::Swap(a, b))
                            }
                        }
                        _ => NOp::MeasX(rng.random_range(1..nq) as u32, rng.random()),
                    };
                    ops.push(op);
                }
                let n = ops.len();
                Round::new(ops, vec![0; n], kind)
            })
            .collect();
        let gc = GenCircuit::with_rounds(&inst, kind, nq, rounds);
        let res = gc.resolve(&mut || false);
        for k in [0usize, 1, 2] {
            let f = res.sample_k(k, &mut rng);
            let e = noisy_gen::trajectory_distribution::<u128>(&gc, &res, &f);
            let r = ref_distribution(&gc, &res, &f);
            assert_close(&e, &r, &format!("random stream {trial} k={k}"));
            let tot: f64 = e.iter().sum();
            if (tot - 1.0).abs() > 1e-6 {
                nontrivial += 1;
            }
        }
    }
    assert!(
        nontrivial > 10,
        "too few streams with collisions ({nontrivial})"
    );
    eprintln!("{nontrivial} of 120 patterns had total weight != 1");
}

#[test]
fn reset_variants_match_sparse_reference() {
    // the design variants (Z-basis resets of should-be-clean ancillas after
    // every window / every round): noiseless they change nothing; with
    // faults (reset flips included) the engine equals the reference
    let mut rng = StdRng::seed_from_u64(77);
    let mut checked = 0;
    let mut dirty_resets = 0;
    for (n_mod, a, oracle) in [
        (15u64, 7u64, Oracle::WindowedOpt(2)),
        (15, 2, Oracle::WindowedMbuLookup(2)),
        (15, 7, Oracle::WindowedMbu(2)),
        (21, 2, Oracle::WindowedOpt(2)),
    ] {
        let inst = Instance::new(n_mod, a, oracle);
        for mode in [ResetMode::Round, ResetMode::Window] {
            for kind in [NoiseKind::Depolarizing, NoiseKind::BitFlip] {
                let base = GenCircuit::new(&inst, kind);
                let gc = GenCircuit::with_resets(&inst, kind, mode);
                let mut r1 = StdRng::seed_from_u64(5);
                let mut r2 = StdRng::seed_from_u64(5);
                let res0 = base.resolve_rng(&mut r1);
                let res = gc.resolve_rng(&mut r2);
                let d0 = noisy_gen::trajectory_distribution::<u128>(&base, &res0, &[]);
                let d1 = noisy_gen::trajectory_distribution::<u128>(&gc, &res, &[]);
                assert_close(&d0, &d1, "noiseless with resets");
                let mut pats: Vec<Vec<Fault>> = Vec::new();
                for k in [1usize, 1, 1, 2, 2, 3] {
                    pats.push(res.sample_k(k, &mut rng));
                }
                // faults on the reset slots themselves
                for (i, rd) in res.rounds.iter().enumerate().take(3) {
                    if let Some(g) = rd.ops.iter().position(|o| matches!(o, NOp::ResetZ(_))) {
                        pats.push(vec![Fault {
                            round: i as u32,
                            site: Site::Gate {
                                gate: g as u32,
                                slot: 0,
                            },
                            pauli: Pauli::X,
                        }]);
                    }
                }
                for f in &pats {
                    let e = noisy_gen::trajectory_distribution::<u128>(&gc, &res, f);
                    let r = ref_distribution(&gc, &res, f);
                    assert_close(
                        &e,
                        &r,
                        &format!("N={n_mod} {oracle:?} {mode:?} {kind:?} {f:?}"),
                    );
                    let tot: f64 = e.iter().sum();
                    if (tot - 1.0).abs() > 1e-9 {
                        dirty_resets += 1;
                    }
                    checked += 1;
                }
            }
        }
    }
    assert!(
        dirty_resets > 5,
        "too few patterns with a dirty reset ({dirty_resets})"
    );
    eprintln!("{checked} reset-variant patterns agree ({dirty_resets} with a non-trivial reset)");
}

/// Reference walk of a windowed exponentiation (exponent qubits `res.eq`,
/// `we` per window, highest power measured first).
fn ref_distribution_windowed(gc: &GenCircuit, res: &Resolved, faults: &[Fault]) -> Vec<f64> {
    let t = gc.inst.t;
    let mut out = vec![0.0; 1 << t];
    let at = |k: usize, gi: usize, slot: u8| {
        faults
            .iter()
            .find(|f| {
                f.round as usize == k
                    && f.site
                        == Site::Gate {
                            gate: gi as u32,
                            slot,
                        }
            })
            .map(|f| f.pauli)
    };
    #[allow(clippy::too_many_arguments)]
    fn bits(
        s: SparseState,
        gc: &GenCircuit,
        res: &Resolved,
        at: &dyn Fn(usize, usize, u8) -> Option<Pauli>,
        k: usize,
        jj: usize,
        y: u128,
        p: f64,
        out: &mut [f64],
    ) {
        let we = gc.we;
        if jj == we {
            window(s, gc, res, at, k + 1, y, p, out);
            return;
        }
        let r = &res.rounds[k];
        let i = k * we + jj;
        let j = (we - 1 - jj) as u8;
        let q = res.eq[j as usize] as usize;
        let find = |kind: u8| {
            let gi = r.ops.iter().position(|o| *o == NOp::Ctrl(kind, j)).unwrap();
            at(k, gi, 0)
        };
        let mut s = s;
        let y_low = y & ((1u128 << i) - 1);
        if y_low != 0 {
            s.apply_gate(&Gate::Phase(q, Instance::correction(i, y_low)))
                .unwrap();
        }
        if let Some(pp) = find(2) {
            s.apply_gate(&pauli_gate(pp, q)).unwrap();
        }
        s.apply_gate(&Gate::H(q)).unwrap();
        if let Some(pp) = find(3) {
            s.apply_gate(&pauli_gate(pp, q)).unwrap();
        }
        let flip = find(4).is_some();
        let p1 = s.prob_one(q);
        for bit in [false, true] {
            let pb = if bit { p1 } else { 1.0 - p1 };
            if pb <= 1e-14 {
                continue;
            }
            let mut c = s.clone();
            c.collapse(q, bit);
            if bit {
                c.apply_gate(&Gate::X(q)).unwrap();
            }
            let rec = u128::from(bit ^ flip) << i;
            bits(c, gc, res, at, k, jj + 1, y | rec, p * pb, out);
        }
    }
    #[allow(clippy::too_many_arguments)]
    fn window(
        s: SparseState,
        gc: &GenCircuit,
        res: &Resolved,
        at: &dyn Fn(usize, usize, u8) -> Option<Pauli>,
        k: usize,
        y: u128,
        p: f64,
        out: &mut [f64],
    ) {
        if k == gc.inst.t / gc.we {
            out[y as usize] += p;
            return;
        }
        let mut s = s;
        let r = &res.rounds[k];
        let mut wf = 1.0;
        for (gi, op) in r.ops.iter().enumerate() {
            let slot_fault = |slot: u8| at(k, gi, slot);
            match *op {
                NOp::Ctrl(0, j) => {
                    let q = res.eq[j as usize] as usize;
                    if slot_fault(0).is_some() {
                        s.apply_gate(&Gate::X(q)).unwrap();
                    }
                    s.apply_gate(&Gate::H(q)).unwrap();
                }
                NOp::Ctrl(1, j) => {
                    if let Some(pp) = slot_fault(0) {
                        s.apply_gate(&pauli_gate(pp, res.eq[j as usize] as usize))
                            .unwrap();
                    }
                }
                NOp::Ctrl(..) => {}
                NOp::G(g) => {
                    s.apply_gate(&g).unwrap();
                    for (slot, q) in g.qubits().into_iter().enumerate() {
                        if let Some(pp) = slot_fault(slot as u8) {
                            s.apply_gate(&pauli_gate(pp, q)).unwrap();
                        }
                    }
                }
                NOp::MeasX(q, m) => {
                    let q = q as usize;
                    let proj = m ^ slot_fault(0).is_some();
                    s.apply_gate(&Gate::H(q)).unwrap();
                    let p1 = s.prob_one(q);
                    let pr = if proj { p1 } else { 1.0 - p1 };
                    if pr < 1e-14 {
                        return;
                    }
                    s.collapse(q, proj);
                    wf *= 2.0 * pr;
                    if proj {
                        s.apply_gate(&Gate::X(q)).unwrap();
                    }
                    if slot_fault(1).is_some() {
                        s.apply_gate(&Gate::X(q)).unwrap();
                    }
                }
                NOp::ResetZ(_) => unreachable!(),
            }
        }
        bits(s, gc, res, at, k, 0, y, p * wf, out);
    }
    let mut s = SparseState::new(gc.nq);
    s.apply_gate(&Gate::X(1)).unwrap();
    window(s, gc, res, &at, 0, 0, 1.0, &mut out);
    out
}

#[test]
fn windowed_exponentiation_matches_textbook_and_sparse_reference() {
    let mut rng = StdRng::seed_from_u64(31337);
    let mut checked = 0;
    for (n_mod, a, we, wm) in [(15u64, 7u64, 2usize, 2usize), (21, 2, 2, 2), (15, 2, 4, 2)] {
        for kind in [
            NoiseKind::Depolarizing,
            NoiseKind::BitFlip,
            NoiseKind::PhaseFlip,
        ] {
            let gc = GenCircuit::new_ge(n_mod, a, we, wm, kind);
            let res = gc.resolve_rng(&mut rng);
            // noiseless: the textbook distribution of order finding
            let d0 = noisy_gen::trajectory_distribution::<u128>(&gc, &res, &[]);
            let tb = qsim_lab::shor::full_qft_distribution(n_mod, a);
            assert_close(&d0, &tb, &format!("noiseless GE N={n_mod} we={we}"));
            let mut pats: Vec<Vec<Fault>> = Vec::new();
            for k in [1usize, 1, 1, 1, 2, 2, 3] {
                pats.push(res.sample_k(k, &mut rng));
            }
            // every control site of the exponent qubits in window 0 and 1
            for k in 0..2usize {
                let r = &res.rounds[k];
                for (gi, op) in r.ops.iter().enumerate() {
                    if let NOp::Ctrl(kind_, _) = op {
                        for pauli in [Pauli::X, Pauli::Y, Pauli::Z] {
                            if matches!(kind_, 0 | 4) && pauli != Pauli::X {
                                continue;
                            }
                            if kind == NoiseKind::PhaseFlip && matches!(kind_, 0 | 4) {
                                continue;
                            }
                            pats.push(vec![Fault {
                                round: k as u32,
                                site: Site::Gate {
                                    gate: gi as u32,
                                    slot: 0,
                                },
                                pauli,
                            }]);
                        }
                    }
                }
            }
            for f in &pats {
                let e = noisy_gen::trajectory_distribution::<u128>(&gc, &res, f);
                let r = ref_distribution_windowed(&gc, &res, f);
                assert_close(&e, &r, &format!("GE N={n_mod} we={we} {kind:?} {f:?}"));
                checked += 1;
            }
        }
    }
    eprintln!("{checked} windowed-exponentiation patterns agree");
}
