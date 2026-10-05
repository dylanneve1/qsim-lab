//! Known algorithm outcomes on several backends.

use qsim_lab::algorithms;
use qsim_lab::circuit::Simulator;
use qsim_lab::{Mps, StateVectorF32, StateVectorF64, Tableau};
use rand::rngs::StdRng;
use rand::SeedableRng;

fn bits_to_u64(bits: &[bool]) -> u64 {
    bits.iter()
        .enumerate()
        .fold(0, |a, (i, &b)| a | (u64::from(b) << i))
}

#[test]
fn bernstein_vazirani_on_every_backend() {
    let mut rng = StdRng::seed_from_u64(1);
    for (n, secret) in [
        (4, 0b1011u64),
        (10, 0b11_0110_1001),
        (30, 0x2BAD_F00D & ((1 << 30) - 1)),
    ] {
        let c = algorithms::bernstein_vazirani(n, secret);
        let mut backends: Vec<Box<dyn Simulator>> =
            vec![Box::new(Tableau::new(n + 1)), Box::new(Mps::new(n + 1, 8))];
        if n <= 20 {
            backends.push(Box::new(StateVectorF64::new(n + 1)));
        }
        for b in backends.iter_mut() {
            let out = c.run(b.as_mut(), &mut rng).unwrap();
            assert_eq!(bits_to_u64(&out), secret, "{} n={n}", b.name());
        }
    }
}

#[test]
fn grover_finds_marked_item() {
    let mut rng = StdRng::seed_from_u64(2);
    for n in [3, 6, 10] {
        let marked = (0b10_1001_1011 >> (10 - n)) as usize;
        let (_, p) = algorithms::grover::<f64, _>(n, marked, &mut rng);
        let floor = if n == 3 { 0.9 } else { 0.99 };
        assert!(p > floor, "n={n}: p={p}");
    }
    let (found, p) = algorithms::grover::<f32, _>(12, 1234, &mut rng);
    assert!(p > 0.99);
    assert_eq!(found, 1234);
}

#[test]
fn ghz_measurements_are_correlated() {
    let mut rng = StdRng::seed_from_u64(3);
    let mut c = algorithms::ghz(20);
    c.measure_all();
    for _ in 0..5 {
        let mut s = StateVectorF32::new(20);
        let bits = c.run(&mut s, &mut rng).unwrap();
        assert!(bits.iter().all(|&b| b == bits[0]));
    }
}

#[test]
fn shor_factors_15_and_21() {
    let mut rng = StdRng::seed_from_u64(4);
    let (f, _) = algorithms::shor_factor(15, &mut rng);
    assert_eq!(f, Some((3, 5)));
    let (f, _) = algorithms::shor_factor(21, &mut rng);
    assert_eq!(f, Some((3, 7)));
}

#[test]
fn qft_then_inverse_is_identity_on_mps_and_sv() {
    let n = 6;
    let mut c = algorithms::qft(n);
    let inv = c.inverse();
    c.append(&inv);
    let mut s = StateVectorF64::basis_state(n, 0b101101);
    s.apply_circuit(&c).unwrap();
    assert!((s.amplitude(0b101101).norm() - 1.0).abs() < 1e-10);
    let mut m = Mps::new(n, 64);
    let mut rng = StdRng::seed_from_u64(5);
    c.run(&mut m, &mut rng).unwrap();
    assert!((m.amplitude(0).norm() - 1.0).abs() < 1e-10);
}
