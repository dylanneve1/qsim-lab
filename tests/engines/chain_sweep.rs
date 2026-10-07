//! Chain-sweep amplitudes against the dense state vector: random brickwork
//! CZ circuits, the IBM doped-Clifford circuit truncated to small sizes, and
//! bond slicing (slice sum = full amplitude, each slice = a state vector of
//! the circuit with that CZ replaced by `P_k ⊗ Z^k`).

use num_complex::Complex64;
use qsim_lab::circuit::Circuit;
use qsim_lab::engines::chain_sweep::{self, compile, sliced_ops, truncate, ChainCircuit};
use qsim_lab::engines::statevector::StateVector;
use qsim_lab::gate::Gate;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;

const QASM: &str = include_str!("../../research/chain-sweep/nq70_depth70_checks27_doped.qasm");

fn brickwork(n: usize, d: usize, rng: &mut StdRng) -> Circuit {
    let mut c = Circuit::new(n);
    for layer in 0..d {
        for q in 0..n {
            for _ in 0..rng.random_range(0..3) {
                c.gate(match rng.random_range(0..7) {
                    0 => Gate::H(q),
                    1 => Gate::S(q),
                    2 => Gate::Sx(q),
                    3 => Gate::Sxdg(q),
                    4 => Gate::T(q),
                    5 => Gate::Rz(q, rng.random_range(-3.0..3.0)),
                    _ => Gate::Ry(q, rng.random_range(-3.0..3.0)),
                });
            }
        }
        let mut a = layer % 2;
        while a + 1 < n {
            // skip a few CZs so the layers are irregular
            if rng.random_bool(0.85) {
                c.gate(Gate::Cz(a, a + 1));
            }
            a += 2;
        }
    }
    c
}

fn sv(c: &Circuit) -> StateVector<f64> {
    let mut s = StateVector::<f64>::new(c.num_qubits);
    s.apply_circuit(c).unwrap();
    s
}

fn close(a: Complex64, b: Complex64, scale: f64, tol: f64) -> bool {
    (a - b).norm() <= tol * scale
}

#[test]
fn random_brickwork_all_amplitudes() {
    let mut rng = StdRng::seed_from_u64(7);
    for (n, d) in [(2, 3), (3, 5), (5, 8), (6, 11), (7, 12), (8, 9)] {
        let c = brickwork(n, d, &mut rng);
        let s = sv(&c);
        let cc = ChainCircuit::from_circuit(&c).unwrap();
        let scale = (1.0 / (1u64 << n) as f64).sqrt();
        for x in 0..1u128 << n {
            let a = chain_sweep::amplitude(&cc, x).unwrap();
            let e = s.amplitude(x as usize);
            assert!(close(a, e, scale, 1e-10), "n={n} d={d} x={x}: {a} vs {e}");
        }
    }
}

#[test]
fn register_width_is_half_the_depth() {
    let c = Circuit::from_qasm(QASM).unwrap();
    for d in [10, 21, 40] {
        let t = truncate(&c, 30, d);
        let cc = ChainCircuit::from_circuit(&t).unwrap();
        let plan = compile(&cc, 0, &HashMap::new());
        assert!(
            plan.width <= d.div_ceil(2) + 1,
            "d={d} width={}",
            plan.width
        );
    }
}

#[test]
fn doped_circuit_truncated_matches_state_vector() {
    let c = Circuit::from_qasm(QASM).unwrap();
    let mut rng = StdRng::seed_from_u64(3);
    for (n, d) in [(12, 14), (14, 24)] {
        let t = truncate(&c, n, d);
        let s = sv(&t);
        let cc = ChainCircuit::from_circuit(&t).unwrap();
        let scale = (1.0 / (1u64 << n) as f64).sqrt();
        for _ in 0..6 {
            let x = rng.random_range(0..1u128 << n);
            let a = chain_sweep::amplitude(&cc, x).unwrap();
            let e = s.amplitude(x as usize);
            assert!(close(a, e, scale, 1e-10), "n={n} d={d} x={x}: {a} vs {e}");
        }
    }
}

#[test]
fn slices_sum_to_amplitude_and_match_state_vector() {
    let c = Circuit::from_qasm(QASM).unwrap();
    let (n, d) = (10, 12);
    let t = truncate(&c, n, d);
    let cc = ChainCircuit::from_circuit(&t).unwrap();
    // three bonds on the middle edge plus one elsewhere
    let mut bonds: Vec<usize> = cc.edge_bonds(n / 2 - 1).into_iter().take(3).collect();
    bonds.push(cc.edge_bonds(1)[0]);
    let x = 0b1011001110u128;
    let full = chain_sweep::amplitude(&cc, x).unwrap();
    let parts =
        chain_sweep::slice_amplitudes_cpu::<f64>(&cc, x, &bonds, &Default::default()).unwrap();
    let sum: Complex64 = parts.iter().sum();
    let scale = (1.0 / (1u64 << n) as f64).sqrt();
    assert!(close(sum, full, scale, 1e-10), "{sum} vs {full}");
    for (s, part) in parts.iter().enumerate() {
        let mut st = StateVector::<f64>::new(n);
        for (q, m, cz) in sliced_ops(&t, &bonds, s) {
            match cz {
                Some(r) => st.apply_gate(&Gate::Cz(q, r)).unwrap(),
                None => st.apply_1q_matrix(q, &m),
            }
        }
        let e = st.amplitude(x as usize);
        assert!(close(*part, e, scale, 1e-10), "slice {s}: {part} vs {e}");
    }
}

#[test]
fn f32_agrees_with_f64() {
    let c = Circuit::from_qasm(QASM).unwrap();
    let t = truncate(&c, 16, 20);
    let cc = ChainCircuit::from_circuit(&t).unwrap();
    let plan = compile(&cc, 12345, &HashMap::new());
    let a64 = chain_sweep::amplitude_cpu::<f64>(&plan, &Default::default()).unwrap();
    let a32 = chain_sweep::amplitude_cpu::<f32>(&plan, &Default::default()).unwrap();
    assert!((a64 - a32).norm() <= 1e-5 / 256.0, "{a64} vs {a32}");
}

#[test]
fn meet_in_the_middle_matches_sweep_and_slices() {
    let c = Circuit::from_qasm(QASM).unwrap();
    let (n, d) = (11, 13);
    let t = truncate(&c, n, d);
    let cc = ChainCircuit::from_circuit(&t).unwrap();
    let scale = (1.0 / (1u64 << n) as f64).sqrt();
    for (x, e) in [
        (0b10110011101u128, 5),
        (0b00111000101, 2),
        (0b11111000000, 8),
    ] {
        let full = chain_sweep::amplitude(&cc, x).unwrap();
        let ct = chain_sweep::cut_tensors_cpu::<f64>(&cc, x, e, &Default::default()).unwrap();
        assert_eq!(ct.bonds, cc.edge_bonds(e));
        let a = ct.amplitude();
        assert!(close(a, full, scale, 1e-10), "e={e}: {a} vs {full}");
        let pos = [0, 2, 3];
        let sums = ct.slice_sums(&pos);
        let bonds: Vec<usize> = pos.iter().map(|&p| ct.bonds[p]).collect();
        let parts =
            chain_sweep::slice_amplitudes_cpu::<f64>(&cc, x, &bonds, &Default::default()).unwrap();
        for (s, (p, q)) in sums.iter().zip(&parts).enumerate() {
            assert!(close(*p, *q, scale, 1e-10), "slice {s}: {p} vs {q}");
        }
    }
}

/// Emulated reduced-precision storage: f32 "storage" reproduces the f32
/// sweep, 16-bit storage stays close to the exact amplitude, and unscaled
/// fp16 underflows where block-scaled fp16 does not.
#[test]
fn lowprec_storage_fidelity() {
    use qsim_lab::engines::chain_lowprec::{passes, run_lowprec, Granularity, LowPrec};
    let c = Circuit::from_qasm(QASM).unwrap();
    let c = truncate(&c, 70, 20);
    let cc = ChainCircuit::from_circuit(&c).unwrap();
    let cfg = Default::default();
    let mut rng = StdRng::seed_from_u64(7);
    let fmts = ["f32", "bf16", "fp16:b256", "fp16", "e4m3:b64"];
    let mut acc = vec![(Complex64::new(0.0, 0.0), 0.0, 0.0); fmts.len()];
    for _ in 0..6 {
        let x: u128 = (0..70).fold(0, |a, i| a | ((rng.random_bool(0.5) as u128) << i));
        let plan = compile(&cc, x, &HashMap::new());
        let e = chain_sweep::amplitude_cpu::<f64>(&plan, &cfg).unwrap();
        let ps = passes(&plan, &cfg, Granularity::Every(16));
        for (j, f) in fmts.iter().enumerate() {
            let r = run_lowprec(&plan, &ps, &LowPrec::parse(f).unwrap(), 0).unwrap();
            if *f == "f32" {
                assert!((r.amp - e).norm() <= 1e-4 * e.norm().max(1e-30));
            }
            acc[j].0 += e.conj() * r.amp;
            acc[j].1 += e.norm_sqr();
            acc[j].2 += r.amp.norm_sqr();
        }
    }
    let fid: Vec<f64> = acc
        .iter()
        .map(|(o, a, b)| {
            if *b > 0.0 {
                o.norm_sqr() / (a * b)
            } else {
                0.0
            }
        })
        .collect();
    assert!(fid[0] > 1.0 - 1e-9, "{fid:?}");
    assert!(fid[1] > 0.999, "{fid:?}");
    assert!(fid[2] > 0.9999, "{fid:?}");
    assert!(fid[3] < 0.5, "unscaled fp16 should underflow: {fid:?}");
    assert!(fid[4] > 0.8, "{fid:?}");
}

/// Packed storage reproduces the emulated low-precision sweep bit for bit:
/// (a) against `run_lowprec` itself, one full-register pass per worldline;
/// (b) against the emulator run on the same gathered stages, with stage
/// blocks smaller than the register (real gather / scatter of packed runs).
#[test]
fn packed_storage_is_bit_exact_with_emulation() {
    use qsim_lab::engines::blocked::{BlockConfig, Stage};
    use qsim_lab::engines::chain_lowprec::{passes, run_lowprec, Granularity, LowPrec};
    use qsim_lab::engines::chain_packed::{packed_stages, run_emulated_stages, run_packed};
    let full = Circuit::from_qasm(QASM).unwrap();
    let fmts = [
        "int4:b16:h",
        "int5:b64",
        "int5:b16:h",
        "int6:b64",
        "int8:b64",
    ];
    let mut rng = StdRng::seed_from_u64(11);

    // (a) small register (one block per pass, as in run_lowprec)
    let c = truncate(&full, 70, 20);
    let cc = ChainCircuit::from_circuit(&c).unwrap();
    let cfg = BlockConfig {
        fuse_1q: false,
        ..BlockConfig::default()
    };
    for _ in 0..3 {
        let x: u128 = (0..70).fold(0, |a, i| a | ((rng.random_bool(0.5) as u128) << i));
        let plan = compile(&cc, x, &HashMap::new());
        assert!(plan.width <= cfg.small_n);
        let ps = passes(&plan, &cfg, Granularity::Qubit);
        let stages: Vec<Stage> = ps
            .iter()
            .map(|p| Stage {
                inner: (0..plan.width).collect(),
                ops: p.clone(),
            })
            .collect();
        for f in fmts {
            let lp = LowPrec::parse(f).unwrap();
            let e = run_lowprec(&plan, &ps, &lp, 0).unwrap().amp;
            let p = run_packed(&plan, &stages, &lp, &cfg).unwrap();
            assert_eq!(p.amp, e, "{f}");
            assert_eq!((p.underflow, p.overflow), (0, 0));
        }
    }

    // (b) gathered stages on a wider register (last 28 layers)
    let c = chain_sweep::truncate_window(&full, 70, 42, 70);
    let cc = ChainCircuit::from_circuit(&c).unwrap();
    let mut acc = vec![(Complex64::new(0.0, 0.0), 0.0, 0.0); fmts.len()];
    let mut accn = acc.clone();
    for _ in 0..4 {
        let x: u128 = (0..70).fold(0, |a, i| a | ((rng.random_bool(0.5) as u128) << i));
        let plan = compile(&cc, x, &HashMap::new());
        assert!(plan.width >= 13, "width {}", plan.width);
        let ex = chain_sweep::amplitude_cpu::<f64>(&plan, &Default::default()).unwrap();
        let stages = packed_stages(&plan.ops, plan.width, plan.width - 4, 3);
        assert!(stages.len() > plan.width);
        assert!(stages
            .iter()
            .any(|s| s.inner != (0..s.inner.len()).collect::<Vec<_>>()));
        // direct: the stage runs as one block; nested: re-planned inside the
        // gather buffer with 2^6-amplitude cache blocks
        let nested = BlockConfig {
            block_bytes: 8 << 6,
            ..cfg.clone()
        };
        for (j, f) in fmts.iter().enumerate() {
            let lp = LowPrec::parse(f).unwrap();
            let e = run_emulated_stages(&plan, &stages, &lp, &cfg).unwrap();
            let p = run_packed(&plan, &stages, &lp, &cfg).unwrap();
            assert_eq!(p.amp, e, "{f}");
            let en = run_emulated_stages(&plan, &stages, &lp, &nested).unwrap();
            let pn = run_packed(&plan, &stages, &lp, &nested).unwrap();
            assert_eq!(pn.amp, en, "{f} nested");
            // the two compute paths differ only by f32 rounding, which a
            // requantization can turn into one quantization step: compare
            // their fidelities against the exact amplitude instead
            acc[j].0 += ex.conj() * p.amp;
            acc[j].1 += ex.norm_sqr();
            acc[j].2 += p.amp.norm_sqr();
            accn[j].0 += ex.conj() * pn.amp;
            accn[j].1 += ex.norm_sqr();
            accn[j].2 += pn.amp.norm_sqr();
        }
    }
    for acc in [acc, accn] {
        let fid: Vec<f64> = acc.iter().map(|(o, a, b)| o.norm_sqr() / (a * b)).collect();
        // int8 is nearly exact; int4 loses the most
        assert!(fid[4] > 0.99, "{fid:?}");
        assert!(fid[0] < fid[4], "{fid:?}");
    }
}

// ----- tail-open batches (chain_tail) -------------------------------------

use qsim_lab::engines::chain_tail::{run_tail, tail_amplitudes, CpuExact, CpuPacked, TailPlan};

/// Little-endian `x` with the suffix of `x` and tail completion `j`
/// (`j` = bits q0..q(m-1), q0 the most significant).
fn with_tail(x: u128, m: usize, j: usize) -> u128 {
    let mut y = x & !((1u128 << m) - 1);
    for i in 0..m {
        y |= (((j >> (m - 1 - i)) & 1) as u128) << i;
    }
    y
}

fn rand_bits(rng: &mut StdRng, n: usize) -> u128 {
    (0..n).fold(0, |a, i| a | ((rng.random_bool(0.5) as u128) << i))
}

fn tail_f64(cc: &ChainCircuit, x: u128, m: usize) -> Vec<Complex64> {
    let tp = TailPlan::new(cc, x, m);
    let mut be = CpuExact::<f64>::new(Default::default());
    run_tail(&mut be, &tp, &mut |_, _| {}).unwrap().0
}

/// Every tail size m = 1..=8 against the dense state vector (random
/// brickwork and the truncated IBM circuit), to f64 round-off.
#[test]
fn tail_batch_matches_statevector() {
    let mut rng = StdRng::seed_from_u64(21);
    let full = Circuit::from_qasm(QASM).unwrap();
    let mut cases = vec![truncate(&full, 14, 24), truncate(&full, 12, 40)];
    for _ in 0..2 {
        cases.push(brickwork(13, 18, &mut rng));
    }
    for c in &cases {
        let n = c.num_qubits;
        let mut sv = StateVector::<f64>::new(n);
        sv.apply_circuit(c).unwrap();
        let cc = ChainCircuit::from_circuit(c).unwrap();
        let rms = (1.0 / (1u64 << n) as f64).sqrt();
        for m in 1..=8 {
            let x = rand_bits(&mut rng, n);
            let got = tail_f64(&cc, x, m);
            assert_eq!(got.len(), 1 << m);
            for (j, a) in got.iter().enumerate() {
                let e = sv.amplitude(with_tail(x, m, j) as usize);
                assert!(
                    (a - e).norm() < 1e-10 * rms,
                    "n={n} m={m} j={j}: {a} vs {e}"
                );
            }
        }
    }
}

/// n = 70: the tail batch against the plain chain sweep's amplitude of each
/// completion (exact f64 engine), and the f32 register to f32 round-off.
#[test]
fn tail_batch_matches_chain_sweep_n70() {
    let full = Circuit::from_qasm(QASM).unwrap();
    let mut rng = StdRng::seed_from_u64(5);
    for (d, m, every) in [(16, 8, 1), (24, 6, 1), (30, 8, 37)] {
        let c = truncate(&full, 70, d);
        let cc = ChainCircuit::from_circuit(&c).unwrap();
        let x = rand_bits(&mut rng, 70);
        let got = tail_f64(&cc, x, m);
        let tp = TailPlan::new(&cc, x, m);
        let mut be32 = CpuExact::<f32>::new(Default::default());
        let got32 = run_tail(&mut be32, &tp, &mut |_, _| {}).unwrap().0;
        let norm: f64 = got.iter().map(|a| a.norm_sqr()).sum::<f64>().sqrt();
        let mut max_rel32 = 0f64;
        for j in (0..1usize << m).step_by(every) {
            let e = chain_sweep::amplitude(&cc, with_tail(x, m, j)).unwrap();
            assert!(
                (got[j] - e).norm() < 1e-9 * norm,
                "d={d} m={m} j={j}: {} vs {e}",
                got[j]
            );
        }
        for (a, b) in got.iter().zip(&got32) {
            max_rel32 = max_rel32.max((a - b).norm() / norm);
        }
        assert!(max_rel32 < 1e-5, "d={d} f32 rel err {max_rel32}");
    }
}

/// Packed store: random reads (`get`) equal the block decoder bit for bit,
/// the tail batch over the packed store equals the batch over its decoded
/// copy, and the batch fidelity is sane per format.
#[test]
fn tail_batch_packed_is_consistent() {
    use qsim_lab::engines::chain_lowprec::LowPrec;
    let full = Circuit::from_qasm(QASM).unwrap();
    let c = chain_sweep::truncate_window(&full, 70, 40, 70);
    let cc = ChainCircuit::from_circuit(&c).unwrap();
    let mut rng = StdRng::seed_from_u64(9);
    let x = rand_bits(&mut rng, 70);
    let tp = TailPlan::new(&cc, x, 6);
    let exact = tail_f64(&cc, x, 6);
    let w = tp.sweep.width;
    for f in [
        "int4:b16:h",
        "int5:b16:h",
        "int5:b64",
        "int6:b64",
        "int3:b16",
        "int7:b64",
        "int8:b64",
    ] {
        let lp = LowPrec::parse(f).unwrap();
        let mut be = CpuPacked::new(lp, Default::default(), w.saturating_sub(3), 2);
        let (got, st, _) = run_tail(&mut be, &tp, &mut |_, _| {}).unwrap();
        assert!(st.passes > 1);
        let s = be.store().unwrap();
        let blk = s.decode_all();
        let rnd: Vec<_> = (0..1usize << w).map(|i| s.get(i)).collect();
        assert_eq!(rnd, blk, "{f}: get vs block decode");
        let copy: Vec<Complex64> = blk
            .iter()
            .map(|a| Complex64::new(a.re as f64, a.im as f64))
            .collect();
        assert_eq!(tail_amplitudes(&tp, &|i| copy[i]), got, "{f}");
        let ov: Complex64 = exact.iter().zip(&got).map(|(e, l)| e.conj() * l).sum();
        let ne: f64 = exact.iter().map(|a| a.norm_sqr()).sum();
        let nl: f64 = got.iter().map(|a| a.norm_sqr()).sum();
        let fid = ov.norm_sqr() / (ne * nl);
        // loose sanity bounds at R passes (rates r = -ln F / R from LOWPREC.md)
        let r = match &f[..4] {
            "int3" => 0.06,
            "int4" => 0.02,
            "int5" => 0.005,
            "int6" => 0.0015,
            "int7" => 5e-4,
            _ => 1e-4,
        };
        eprintln!("{f}: R={} F={fid:.6}", st.passes);
        assert!(
            fid > (-r * st.passes as f64).exp(),
            "{f}: R={} F = {fid}",
            st.passes
        );
    }
}

/// The GEMM tail pass equals the depth-first one (every split point h).
#[test]
fn tail_gemm_matches_dfs() {
    use qsim_lab::engines::chain_tail::{tail_amplitudes_dfs, tail_amplitudes_gemm, SweepBackend};
    let full = Circuit::from_qasm(QASM).unwrap();
    let mut rng = StdRng::seed_from_u64(77);
    for (d, m) in [(24, 8), (30, 6), (20, 1), (26, 3)] {
        let c = truncate(&full, 70, d);
        let cc = ChainCircuit::from_circuit(&c).unwrap();
        let x = rand_bits(&mut rng, 70);
        let tp = TailPlan::new(&cc, x, m);
        assert_eq!(
            tp.slots,
            (0..tp.num_bonds()).collect::<Vec<_>>(),
            "d={d} m={m}"
        );
        let mut be = CpuExact::<f64>::new(Default::default());
        be.sweep(&tp.sweep, &mut |_, _| {}).unwrap();
        let reg = |i: usize| be.amp(i);
        let r = tail_amplitudes_dfs(&tp, &reg);
        let norm: f64 = r.iter().map(|a| a.norm_sqr()).sum::<f64>().sqrt();
        for h in [0, 1, tp.num_bonds() / 2, tp.num_bonds()] {
            let g = tail_amplitudes_gemm(&tp, &reg, Some(h));
            let err = r
                .iter()
                .zip(&g)
                .map(|(a, b)| (a - b).norm())
                .fold(0.0, f64::max);
            assert!(err < 1e-12 * norm, "d={d} m={m} h={h}: {err:e}");
        }
    }
}
