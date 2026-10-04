//! Semiclassical order finding for N = 15 (one recycled control, t rounds)
//! written against [`Logical`], so the same circuit runs unencoded, encoded
//! (frame) or encoded (dense validation).
//!
//! Every multiplier a^(2^j) mod 15 is ±2^k, i.e. a cyclic rotation of the
//! 4-bit register (2^4 ≡ 1), followed by a bitwise NOT when the sign is −
//! (15 − x = NOT x). This is a genuine modular multiplication on all inputs
//! 1…14 (not input-specialised); controlled, a rotation is a chain of
//! controlled swaps (Fredkin = CNOT·Toffoli·CNOT, 7 T each) and the NOT is
//! four CNOTs from the control. For a = 7, t = 3: 5 Fredkins = 35 T gates.

use super::logical::Logical;

/// Logical qubit layout: 0 = control, 1..=4 = work bits x0 (LSB)..x3.
pub const NLOG15: usize = 5;

/// m mod 15 as (left-rotation k, negate). Panics for non-units.
pub fn mult_program(m: u64) -> (usize, bool) {
    let m = m % 15;
    for k in 0..4 {
        if m == (1u64 << k) % 15 {
            return (k, false);
        }
        if m == 15 - (1u64 << k) % 15 {
            return (k, true);
        }
    }
    panic!("{m} is not ±2^k mod 15");
}

/// Swaps realising a left rotation by k of (x0..x3).
pub fn rotation_swaps(k: usize) -> Vec<(usize, usize)> {
    match k % 4 {
        0 => vec![],
        1 => vec![(2, 3), (1, 2), (0, 1)],
        2 => vec![(0, 2), (1, 3)],
        _ => vec![(0, 1), (1, 2), (2, 3)],
    }
}

/// Controlled multiplication by m mod 15 on the work register.
pub fn controlled_mult<L: Logical>(l: &mut L, m: u64) {
    let (k, neg) = mult_program(m);
    for (i, j) in rotation_swaps(k) {
        l.cswap(0, 1 + i, 1 + j);
    }
    if neg {
        for i in 0..4 {
            l.cnot(0, 1 + i);
        }
    }
}

/// One run of semiclassical order finding; returns y (bit j = round j).
/// Round j applies controlled-U^(2^(t−1−j)), the phase correction
/// −2π Σ_{i<j} b_i / 2^(j−i+1) (only S† and T† occur for t ≤ 3), H, measure.
pub fn run_shor15<L: Logical>(l: &mut L, a: u64, t: usize) -> u64 {
    assert!(t <= 3, "phase corrections finer than T need synthesis");
    l.prep(1, true);
    for i in 1..4 {
        l.prep(1 + i, false);
    }
    let mut y = 0u64;
    for j in 0..t {
        l.prep(0, false);
        l.h(0);
        let e = 1u64 << (t - 1 - j);
        let mut m = 1u64;
        for _ in 0..e {
            m = m * a % 15;
        }
        if m != 1 {
            controlled_mult(l, m);
        }
        for i in 0..j {
            let bit = (y >> i) & 1 == 1;
            match j - i + 1 {
                // S† correction: a fixed slot (S† or noisy identity)
                2 => l.sdg_slot(0, bit),
                // T† correction: only when the bit is set (never in an
                // error-free run of an r = 4 instance, whose b0 is 0)
                3 => {
                    if bit {
                        l.tdg(0)
                    }
                }
                _ => unreachable!(),
            }
        }
        l.h(0);
        if l.meas(0) {
            y |= 1 << j;
        }
    }
    y
}

/// Exact noiseless distribution of y for N = 15 with r = ord(a): uniform on
/// the multiples of 2^t / r.
pub fn ideal_distribution(a: u64, t: usize) -> Vec<f64> {
    let mut r = 1;
    let mut v = a % 15;
    while v != 1 {
        v = v * a % 15;
        r += 1;
    }
    let mut d = vec![0.0; 1 << t];
    let step = (1usize << t) / r;
    for s in 0..r {
        d[s * step] = 1.0 / r as f64;
    }
    d
}

#[cfg(test)]
mod tests {
    use super::super::core::Noise;
    use super::super::logical::{LSv, Unencoded};
    use super::*;

    #[test]
    fn multipliers_are_modular_multiplication() {
        for m in [2u64, 4, 7, 8, 11, 13, 14] {
            for x in 1..15u64 {
                let mut u = Unencoded::new(NLOG15, Noise::new(0.0, 1), false, 3);
                u.prep(0, true);
                for i in 0..4 {
                    u.prep(1 + i, (x >> i) & 1 == 1);
                }
                controlled_mult(&mut u, m);
                let mut got = 0;
                for i in 0..4 {
                    if u.sv.prob1(1 + i) > 0.5 {
                        got |= 1 << i;
                    }
                }
                assert_eq!(got, m * x % 15, "m={m} x={x}");
            }
        }
    }

    #[test]
    fn toffoli_decomposition_is_exact() {
        // compare decomposed vs native CCX on a random-ish entangled input
        let mut a = Unencoded::new(3, Noise::new(0.0, 1), false, 1);
        let mut b = Unencoded::new(3, Noise::new(0.0, 1), true, 1);
        for u in [&mut a, &mut b] {
            u.sv = LSv::new(3);
            u.sv.h(0);
            u.sv.t(0);
            u.sv.h(1);
            u.sv.cnot(0, 2);
            u.sv.h(2);
            u.sv.s(2);
        }
        a.ccx(0, 1, 2);
        b.ccx(0, 1, 2);
        let mut ov = num_complex::Complex64::new(0.0, 0.0);
        for i in 0..8 {
            ov += a.sv.a[i].conj() * b.sv.a[i];
        }
        assert!((ov.norm() - 1.0).abs() < 1e-12);
        // global phase too
        assert!((ov.re - 1.0).abs() < 1e-12, "{ov}");
    }

    #[test]
    fn noiseless_shor15_matches_ideal() {
        for a in [7u64, 2, 11] {
            let ideal = ideal_distribution(a, 3);
            let mut hist = [0u32; 8];
            let n = 4000;
            for s in 0..n {
                let mut u = Unencoded::new(NLOG15, Noise::new(0.0, 1), false, s);
                hist[run_shor15(&mut u, a, 3) as usize] += 1;
            }
            for y in 0..8 {
                let f = hist[y] as f64 / n as f64;
                assert!((f - ideal[y]).abs() < 0.03, "a={a} y={y} {f} vs {}", ideal[y]);
                if ideal[y] == 0.0 {
                    assert_eq!(hist[y], 0);
                }
            }
        }
    }
}
