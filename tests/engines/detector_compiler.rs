//! The backward-sweep detector compiler must produce *exactly* what the old
//! route (forward symbolic frame -> parity rows -> relative to reference ->
//! pruning) produces: the same variable groups in the same order and the
//! same rows per variable. Then `FastSampler::from_columns` (fast table
//! builder) must equal `FastSampler::new` (original builder), hit tables
//! included, and the `.stim` front-end (REPEAT blocks kept) must equal the
//! unrolled-circuit front-end. All comparisons are `==` on the data, on
//! every circuit family in the test suite plus random circuits.

use qsim_lab::engines::stabilizer::detector_compiler::{compile_circuit, compile_stim, Columns};
use qsim_lab::engines::stabilizer::fast_sampler::FastSampler;
use qsim_lab::engines::stabilizer::symphase::SymPhaseSampler;
use qsim_lab::io::stim::{parse_stim, parse_stim_circuit, parse_stim_reference, to_stim};
use qsim_lab::qec::bb_circuit::{memory as bb_memory, IBM_SCHEDULE};
use qsim_lab::qec::bicycle::TwoBlockCode;
use qsim_lab::qec::color::{ColorCode, ColorNoise, KF_SCHEDULE};
use qsim_lab::qec::schedules::{Schedule, ScheduledSurfaceCode};
use qsim_lab::{Circuit, Gate, NoiseModel, Op, SimError, SurfaceCode};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::f64::consts::{FRAC_PI_2, PI};

/// The old route.
fn old_columns(
    c: &Circuit,
    noise: &NoiseModel,
    dets: &[Vec<usize>],
    obs: &[Vec<usize>],
) -> Result<(Columns, FastSampler), SimError> {
    let sets: Vec<Vec<usize>> = dets.iter().chain(obs).cloned().collect();
    let s = SymPhaseSampler::new(c, noise)?
        .with_parities(&sets)
        .relative_to_reference();
    Ok((Columns::from_symphase(&s), FastSampler::new(&s)))
}

/// New route == old route, columns and samplers (with hit tables).
fn assert_same(c: &Circuit, noise: &NoiseModel, dets: &[Vec<usize>], obs: &[Vec<usize>], what: &str) {
    let (oc, of) = old_columns(c, noise, dets, obs).unwrap();
    let nc = compile_circuit(c, noise, dets, obs).unwrap();
    assert_eq!(nc, oc, "{what}: columns differ");
    let nf = FastSampler::from_columns(nc, true);
    assert!(nf == of, "{what}: samplers differ");
}

fn random_circuit(n: usize, len: usize, feedback: bool, rng: &mut StdRng) -> (Circuit, usize) {
    let mut c = Circuit::new(n);
    let mut meas = 0usize;
    for _ in 0..len {
        let a = rng.random_range(0..n);
        let b = (a + rng.random_range(1..n.max(2))) % n;
        let p = [0.001, 0.01, 0.1, 0.25, 0.4][rng.random_range(0..5)];
        let two = n > 1;
        match rng.random_range(0..26) {
            0..=2 => {
                c.gate(Gate::H(a));
            }
            3 => {
                c.gate(Gate::S(a));
            }
            4 => {
                c.gate(Gate::Sdg(a));
            }
            5 => {
                c.gate(Gate::Phase(a, FRAC_PI_2 * rng.random_range(0..4) as f64));
            }
            6 => {
                c.gate(Gate::Sx(a));
            }
            7 => {
                c.gate(Gate::Sxdg(a));
            }
            8 => {
                c.gate(Gate::I(a));
            }
            9 => {
                c.gate([Gate::X(a), Gate::Y(a), Gate::Z(a)][rng.random_range(0..3)]);
            }
            10..=12 if two => {
                c.gate(Gate::Cnot(a, b));
            }
            13 if two => {
                c.gate(Gate::Cz(a, b));
            }
            14 if two => {
                c.gate(Gate::Swap(a, b));
            }
            15 if two => {
                c.gate([Gate::ISwap(a, b), Gate::ISwapdg(a, b)][rng.random_range(0..2)]);
            }
            16 if two => {
                c.gate(Gate::CPhase(a, b, PI * rng.random_range(0..3) as f64));
            }
            17 | 18 => {
                c.measure(a);
                meas += 1;
            }
            19 => {
                c.reset(a);
            }
            20 => {
                c.x_flip(a, p);
            }
            21 => {
                c.y_flip(a, p);
            }
            22 => {
                c.z_flip(a, p);
            }
            23 => {
                c.depolarize_1q(a, p);
            }
            24 if two => {
                c.depolarize_2q(a, b, p);
            }
            25 if feedback && meas > 0 => {
                let m = rng.random_range(0..meas);
                let g = [Gate::X(a), Gate::Y(a), Gate::Z(a)][rng.random_range(0..3)];
                c.classic_controlled(g, m, rng.random_bool(0.5));
            }
            _ => {
                c.gate(Gate::H(a));
            }
        }
    }
    for q in 0..n {
        c.measure(q);
        meas += 1;
    }
    (c, meas)
}

fn random_sets(meas: usize, count: usize, rng: &mut StdRng) -> Vec<Vec<usize>> {
    (0..count)
        .map(|_| {
            let k = rng.random_range(0..5);
            // duplicates allowed (they cancel)
            (0..k).map(|_| rng.random_range(0..meas)).collect()
        })
        .collect()
}

#[test]
fn columns_equal_old_route_on_random_circuits() {
    let mut rng = StdRng::seed_from_u64(1);
    for i in 0..1500 {
        let n = rng.random_range(1..9);
        let len = rng.random_range(1..80);
        let noise = match i % 4 {
            0 => NoiseModel::none(),
            1 => NoiseModel::none().with_meas(0.01).with_reset(0.02),
            2 => NoiseModel::gate_depolarizing(0.002, 0.003),
            _ => NoiseModel::circuit_level(0.001, 0.3).with_reset(0.5),
        };
        // feedback is only valid without 1-qubit gate noise
        let feedback = noise.p_1q == 0.0;
        let (c, meas) = random_circuit(n, len, feedback, &mut rng);
        let nd = rng.random_range(0..12);
        let dets = random_sets(meas, nd, &mut rng);
        let no = rng.random_range(0..3);
        let obs = random_sets(meas, no, &mut rng);
        assert_same(&c, &noise, &dets, &obs, &format!("random circuit {i}"));
    }
}

#[test]
fn equal_on_surface_codes() {
    for d in [3, 5, 7] {
        let sc = SurfaceCode::new(d, d);
        let c = sc.build_circuit();
        let dets = sc.detector_records();
        let obs = vec![sc.observable_records()];
        for noise in [
            NoiseModel::circuit_level(0.001, 0.001),
            NoiseModel::circuit_level(0.003, 0.002).with_reset(0.004),
            NoiseModel::uniform(0.3),
        ] {
            assert_same(&c, &noise, &dets, &obs, &format!("surface d={d} {noise:?}"));
        }
    }
    for sched in [Schedule::standard(), Schedule::standard_interleaved()] {
        let sc = ScheduledSurfaceCode::new(5, 4, sched);
        let noise = NoiseModel::circuit_level(0.002, 0.001);
        assert_same(
            &sc.build_circuit(),
            &noise,
            &sc.detector_records(),
            &[sc.observable_records()],
            "scheduled surface",
        );
    }
}

#[test]
fn equal_on_colour_codes() {
    for d in [3, 5] {
        let cc = ColorCode::new(d);
        let s = cc.uniform_schedule(KF_SCHEDULE);
        for (noise, xb) in [
            (ColorNoise::Cnot(0.002), false),
            (ColorNoise::Uniform(0.001), false),
            (ColorNoise::Uniform(0.003), true),
        ] {
            let m = cc.memory_basis(&s, d, noise, xb);
            assert_same(
                &m.circuit,
                &m.noise,
                &m.detectors,
                &m.observables,
                &format!("colour d={d} {noise:?} x={xb}"),
            );
        }
    }
}

#[test]
fn equal_on_bivariate_bicycle_code() {
    // [[72, 12, 6]]
    let c = TwoBlockCode::parse(6, 6, "x^3+y+y^2", "y^3+x+x^2");
    for (rounds, xb) in [(1, false), (3, true)] {
        let m = bb_memory(&c, &IBM_SCHEDULE, rounds, 0.001, xb);
        assert_same(
            &m.circuit,
            &m.noise,
            &m.detectors,
            &m.observables,
            &format!("bb72 rounds={rounds} x={xb}"),
        );
    }
}

/// Stim circuits: Stim's own generated circuits (tests/data) and our
/// exported families. The fast parser equals the reference parser, and the
/// REPEAT-aware `.stim` front-end equals the unrolled-circuit front-end.
#[test]
fn stim_front_end_equals_circuit_front_end() {
    let mut texts: Vec<(String, String)> = Vec::new();
    for f in [
        "stim_rotated_memory_z_d3_p0.003.stim",
        "stim_rotated_memory_x_d5_p0.002.stim",
        "stim_unrotated_memory_z_d3_p0.001.stim",
        "stim_repetition_memory_d5_p0.01.stim",
        "stim_color_memory_xyz_d5_p0.003_decomposed.stim",
        "stim_subset_features.stim",
    ] {
        let path = format!("{}/tests/data/{f}", env!("CARGO_MANIFEST_DIR"));
        texts.push((f.to_string(), std::fs::read_to_string(path).unwrap()));
    }
    for d in [3, 5] {
        let sc = SurfaceCode::new(d, d);
        let t = to_stim(
            &sc.build_circuit(),
            &NoiseModel::circuit_level(0.002, 0.003),
            &sc.detector_records(),
            &[sc.observable_records()],
        )
        .unwrap();
        texts.push((format!("export surface d={d}"), t));
    }
    let cc = ColorCode::new(5);
    let m = cc.memory_basis(&cc.uniform_schedule(KF_SCHEDULE), 3, ColorNoise::Uniform(0.002), false);
    texts.push((
        "export colour d=5".into(),
        to_stim(&m.circuit, &m.noise, &m.detectors, &m.observables).unwrap(),
    ));
    for (name, text) in &texts {
        let prog = parse_stim(text).unwrap();
        assert_eq!(prog, parse_stim_reference(text).unwrap(), "{name}: parsers differ");
        let sc = parse_stim_circuit(text).unwrap();
        assert_eq!(sc.num_measurements(), prog.circuit.ops.iter().filter(|o| matches!(o, Op::Measure(_))).count());
        assert_eq!(sc.num_detectors(), prog.detectors.len());
        assert_eq!(sc.num_observables(), prog.observables.len());
        let a = compile_stim(&sc);
        let b = compile_circuit(&prog.circuit, &prog.noise, &prog.detectors, &prog.observables).unwrap();
        assert_eq!(a, b, "{name}: .stim front-end differs");
        let (oc, of) = old_columns(&prog.circuit, &prog.noise, &prog.detectors, &prog.observables).unwrap();
        assert_eq!(a, oc, "{name}: differs from the old route");
        assert!(FastSampler::from_columns(a, true) == of, "{name}: samplers differ");
    }
}

#[test]
fn errors_match_the_old_route() {
    let mut c = Circuit::new(2);
    c.h(0).gate(Gate::T(1)).measure(0);
    assert!(matches!(
        compile_circuit(&c, &NoiseModel::none(), &[vec![0]], &[]),
        Err(SimError::Unsupported { .. })
    ));
    assert!(SymPhaseSampler::new(&c, &NoiseModel::none()).is_err());
    let mut c = Circuit::new(2);
    c.h(0).measure(0);
    assert!(matches!(
        compile_circuit(&c, &NoiseModel::none(), &[vec![1]], &[]),
        Err(SimError::ClassicalBitOutOfRange { bit: 1, available: 1 })
    ));
    // feedback needs an earlier measurement, a Pauli, and no 1-qubit gate noise
    let mut c = Circuit::new(2);
    c.measure(0);
    c.classic_controlled(Gate::H(1), 0, true);
    c.measure(1);
    assert!(compile_circuit(&c, &NoiseModel::none(), &[vec![1]], &[]).is_err());
    assert!(SymPhaseSampler::new(&c, &NoiseModel::none()).is_err());
    let mut c = Circuit::new(2);
    c.measure(0);
    c.classic_controlled(Gate::X(1), 0, true);
    c.measure(1);
    let noisy = NoiseModel::gate_depolarizing(0.01, 0.0);
    assert!(compile_circuit(&c, &noisy, &[vec![1]], &[]).is_err());
    assert!(SymPhaseSampler::new(&c, &noisy).is_err());
    assert!(compile_circuit(&c, &NoiseModel::none(), &[vec![0, 1]], &[]).is_ok());
    let mut c = Circuit::new(2);
    c.ops.push(Op::Measure(5));
    assert!(matches!(
        compile_circuit(&c, &NoiseModel::none(), &[], &[]),
        Err(SimError::QubitOutOfRange { qubit: 5, .. })
    ));
}

/// Random `.stim` programs in the supported subset (nested REPEAT blocks,
/// every instruction and alias): the fast parser equals the reference
/// parser, and the `.stim` front-end equals the circuit front-end.
#[test]
fn random_stim_programs() {
    let mut rng = StdRng::seed_from_u64(7);
    for i in 0..400 {
        let n = rng.random_range(1..6);
        let mut text = String::new();
        let mut meas = 0usize;
        let p = [0.0, 0.001, 0.02, 0.3][rng.random_range(0..4)];
        gen_block(&mut text, n, &mut meas, p, 0, &mut rng);
        // a final layer so detectors have something to read (same readout
        // flip as the body: the unrolled program has one p_meas)
        text.push_str(&if p > 0.0 { format!("M({p})") } else { "M".to_string() });
        for q in 0..n {
            text.push_str(&format!(" {q}"));
        }
        text.push('\n');
        meas += n;
        for _ in 0..rng.random_range(0..4) {
            let k = rng.random_range(1..=meas.min(6));
            text.push_str(&format!("DETECTOR rec[-{k}] rec[-1]\n"));
        }
        text.push_str(&format!("OBSERVABLE_INCLUDE({}) rec[-1]\n", rng.random_range(0..2)));
        let prog = parse_stim(&text).unwrap_or_else(|e| panic!("{e}\n{text}"));
        assert_eq!(prog, parse_stim_reference(&text).unwrap(), "program {i}:\n{text}");
        let a = compile_stim(&parse_stim_circuit(&text).unwrap());
        let b = compile_circuit(&prog.circuit, &prog.noise, &prog.detectors, &prog.observables).unwrap();
        assert_eq!(a, b, "program {i}:\n{text}");
    }
}

fn gen_block(text: &mut String, n: usize, meas: &mut usize, p: f64, depth: usize, rng: &mut StdRng) {
    let lines = rng.random_range(1..10);
    for _ in 0..lines {
        let q = rng.random_range(0..n);
        let r = (q + rng.random_range(1..n.max(2))) % n;
        let pp = rng.random_range(0.0..0.2f64);
        match rng.random_range(0..20) {
            0 => text.push_str(&format!("H {q}\n")),
            1 => text.push_str(&format!("{} {q}\n", ["S", "S_DAG", "SQRT_Z", "SQRT_Z_DAG", "H_XZ", "X", "Y", "Z", "I"][rng.random_range(0..9)])),
            2 | 3 if n > 1 => text.push_str(&format!("{} {q} {r}\n", ["CX", "CNOT", "ZCX", "CZ", "ZCZ", "SWAP"][rng.random_range(0..6)])),
            4 => text.push_str(&format!("{} {q}\n", ["R", "RZ", "RX"][rng.random_range(0..3)])),
            5 | 6 => {
                let name = ["M", "MZ", "MX", "MR", "MRZ", "MRX"][rng.random_range(0..6)];
                if p > 0.0 {
                    text.push_str(&format!("{name}({p}) {q}\n"));
                } else {
                    text.push_str(&format!("{name} {q}\n"));
                }
                *meas += 1;
            }
            7 => text.push_str(&format!("{}({pp}) {q}\n", ["X_ERROR", "Y_ERROR", "Z_ERROR", "DEPOLARIZE1"][rng.random_range(0..4)])),
            8 if n > 1 => text.push_str(&format!("DEPOLARIZE2({pp}) {q} {r}\n")),
            9 if *meas > 0 => {
                let k = rng.random_range(1..=(*meas).min(4));
                text.push_str(&format!("DETECTOR(1, 2) rec[-{k}]\n"));
            }
            10 if *meas > 0 => {
                let k = rng.random_range(1..=(*meas).min(4));
                text.push_str(&format!("OBSERVABLE_INCLUDE({}) rec[-{k}]\n", rng.random_range(0..3)));
            }
            11 => text.push_str("TICK\n"),
            12 if depth < 2 => {
                let count = rng.random_range(1..4);
                text.push_str(&format!("REPEAT {count} {{\n"));
                let before = *meas;
                gen_block(text, n, meas, p, depth + 1, rng);
                let body = *meas - before;
                *meas = before + body * count;
                text.push_str("}\n");
            }
            13 => text.push_str(&format!("# comment\nQUBIT_COORDS(0, 1) {q}\n")),
            _ => text.push_str(&format!("CX {q} {r}\n").replace(&format!("CX {q} {q}"), &format!("H {q}"))),
        }
    }
}

/// The sampler-x driver: output bit-identical for every thread count (per
/// batch random streams) and with or without hit tables (both paths consume
/// the stream identically, so a hit XORs the same rows either way); a
/// different seed gives a different output.
#[test]
fn write_ptb64_identical_across_threads_and_table_modes() {
    let sc = SurfaceCode::new(5, 5);
    let surface = to_stim(
        &sc.build_circuit(),
        &NoiseModel::circuit_level(0.004, 0.003),
        &sc.detector_records(),
        &[sc.observable_records()],
    )
    .unwrap();
    let path = format!(
        "{}/tests/data/stim_color_memory_xyz_d5_p0.003_decomposed.stim",
        env!("CARGO_MANIFEST_DIR")
    );
    let colour = std::fs::read_to_string(path).unwrap();
    for (text, shots) in [(&surface, 100_017usize), (&colour, 40_000), (&surface, 63)] {
        let cols = compile_stim(&parse_stim_circuit(text).unwrap());
        let ft = FastSampler::from_columns(cols.clone(), true);
        let fc = FastSampler::from_columns(cols, false);
        let mut want = Vec::new();
        ft.write_ptb64(shots, 5, 1, &mut want).unwrap();
        assert_eq!(want.len(), shots.div_ceil(64) * ft.rows() * 8);
        for threads in [2, 3, 8] {
            for slab in [1, 50_000, 4 << 20] {
                let mut o = Vec::new();
                ft.write_ptb64_with(shots, 5, threads, FastSampler::batch_words(shots), slab, &mut o)
                    .unwrap();
                assert!(o == want, "threads {threads} slab {slab}");
            }
        }
        for threads in [1, 4] {
            let mut o = Vec::new();
            fc.write_ptb64_with(shots, 5, threads, FastSampler::batch_words(shots), 1, &mut o)
                .unwrap();
            assert!(o == want, "no tables, threads {threads}");
        }
        // the AVX-512 gather/scatter kernel (where the CPU has it)
        let mut fs = ft.clone();
        if fs.set_simd(true) {
            for threads in [1, 3] {
                let mut o = Vec::new();
                fs.write_ptb64_with(shots, 5, threads, FastSampler::batch_words(shots), 1, &mut o)
                    .unwrap();
                assert!(o == want, "avx512, threads {threads}");
            }
        }
        let mut o = Vec::new();
        ft.write_ptb64(shots, 6, 1, &mut o).unwrap();
        assert!(o != want);
    }
}
