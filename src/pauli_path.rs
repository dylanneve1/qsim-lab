//! Clifford+T simulation by Pauli-path summation (Heisenberg picture).
//!
//! To get an expectation value `<0| U† O U |0>` for a Pauli observable `O`,
//! push `O` backwards through the circuit: `O -> G† O G` for each gate, last
//! gate first. A Clifford gate maps a Pauli string to a single Pauli string
//! (with a sign), exactly as in the stabilizer tableau, so a Clifford circuit
//! keeps `O` a single term. A T gate (or any Z rotation) instead maps
//!
//! ```text
//!   X -> cos θ X + sin θ Y,   Y -> cos θ Y - sin θ X,   I, Z unchanged
//! ```
//!
//! so every term with an X or Y on that qubit splits in two. After `t` such
//! gates there are up to `2^t` terms: the cost is polynomial in the number of
//! qubits but exponential in the number of non-Clifford gates. Terms with
//! identical Pauli strings are merged after each branching gate, which is
//! exact and sometimes keeps the sum much smaller than `2^t`.
//!
//! At the end, `<0...0| P |0...0>` is the sign of `P` if `P` contains only
//! I and Z, and 0 otherwise.
//!
//! This is the dual of a "sum over stabilizer states" simulator (which
//! splits the *state* at each T gate instead of the observable); both cost
//! `O(2^t poly(n))` and are exact.

use crate::circuit::{check_gate, Circuit, Op, SimError};
use crate::gate::{is_multiple_of_half_pi, Gate};
use rayon::prelude::*;
use std::f64::consts::FRAC_PI_4;

pub use crate::pauli_frame::FrameOptions;

/// Default cap on the number of Pauli terms kept at once.
pub const DEFAULT_MAX_TERMS: usize = 1 << 22;

/// A real linear combination of Hermitian Pauli strings on `n` qubits.
///
/// Each term is stored as `2w` words (`x` bits then `z` bits, `w = ceil(n/64)`)
/// with `(x, z) = (1, 1)` meaning Y, as in the CHP tableau.
#[derive(Clone, Debug, PartialEq)]
pub struct PauliSum {
    pub(crate) n: usize,
    pub(crate) w: usize,
    pub(crate) keys: Vec<u64>,
    pub(crate) coefs: Vec<f64>,
}

/// Statistics of a propagation run.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PathStats {
    /// Largest number of terms alive at any point.
    pub peak_terms: usize,
    /// Number of terms at the end.
    pub final_terms: usize,
    /// Non-Clifford gates encountered.
    pub non_clifford_gates: usize,
    /// Rotations actually propagated (after any exact rotation merging).
    pub rotations: usize,
    /// Sum over propagated rotations of the number of terms alive when the
    /// rotation was applied: the per-term work of the run.
    pub term_visits: u64,
    /// Terms discarded by the frame engine's x-span pruning (0 otherwise).
    pub pruned_terms: u64,
}

impl PauliSum {
    /// Parses a single Pauli string such as `"XIZY"` (qubit 0 first) with
    /// coefficient 1.
    pub fn from_str_single(s: &str) -> Self {
        let n = s.chars().count();
        let mut p = Self::identity(n);
        for (q, ch) in s.chars().enumerate() {
            let (x, z) = match ch {
                'I' | '_' => (false, false),
                'X' => (true, false),
                'Y' => (true, true),
                'Z' => (false, true),
                _ => panic!("invalid Pauli character {ch:?}"),
            };
            p.set(0, q, x, z);
        }
        p
    }

    /// The identity operator on `n` qubits.
    pub fn identity(n: usize) -> Self {
        let w = n.div_ceil(64).max(1);
        PauliSum {
            n,
            w,
            keys: vec![0; 2 * w],
            coefs: vec![1.0],
        }
    }

    /// `Z` on each of `qubits` (their product).
    pub fn z_product(n: usize, qubits: &[usize]) -> Self {
        let mut p = Self::identity(n);
        for &q in qubits {
            assert!(q < n);
            p.set(0, q, false, true);
        }
        p
    }

    fn set(&mut self, term: usize, q: usize, x: bool, z: bool) {
        let s = 2 * self.w;
        let k = &mut self.keys[term * s..(term + 1) * s];
        let (wi, b) = (q / 64, 1u64 << (q % 64));
        k[wi] = if x { k[wi] | b } else { k[wi] & !b };
        k[self.w + wi] = if z {
            k[self.w + wi] | b
        } else {
            k[self.w + wi] & !b
        };
    }

    pub fn num_terms(&self) -> usize {
        self.coefs.len()
    }

    /// `<0...0| self |0...0>`.
    pub fn expectation_zero_state(&self) -> f64 {
        let s = 2 * self.w;
        self.keys
            .par_chunks(s)
            .zip(self.coefs.par_iter())
            .filter(|(k, _)| k[..self.w].iter().all(|&x| x == 0))
            .map(|(_, c)| *c)
            .sum()
    }

    /// Conjugates by a Clifford circuit `C`: `P -> C P C†` (Schrödinger
    /// picture, gates in time order). Errors on non-Clifford gates.
    pub fn conjugate_by_clifford(&mut self, c: &Circuit) -> Result<(), SimError> {
        for g in c.gates() {
            check_gate(g, self.n)?;
            if !g.is_clifford() {
                return Err(SimError::Unsupported {
                    backend: "pauli-path (Clifford conjugation)",
                    gate: *g,
                });
            }
            self.conjugate_clifford(g);
        }
        Ok(())
    }

    /// Conjugates every term by a Clifford gate: `P -> G P G†`.
    fn conjugate_clifford(&mut self, g: &Gate) {
        let w = self.w;
        let g = *g;
        self.keys
            .par_chunks_mut(2 * w)
            .zip(self.coefs.par_iter_mut())
            .for_each(|(k, c)| {
                if conj_string(k, w, &g) == 1 {
                    *c = -*c;
                }
            });
    }

    /// Conjugates by `U = diag(1, e^{iθ})` on qubit `a`: `P -> U P U†`.
    /// Terms with X or Y on `a` split in two; equal strings are merged.
    fn conjugate_phase(&mut self, a: usize, theta: f64, max_terms: usize) -> Result<(), SimError> {
        let (sn, cs) = theta.sin_cos();
        let w = self.w;
        let s = 2 * w;
        let (wi, b) = (a / 64, 1u64 << (a % 64));
        let mut keys = Vec::with_capacity(self.keys.len() * 2);
        let mut coefs = Vec::with_capacity(self.coefs.len() * 2);
        for (k, &c) in self.keys.chunks(s).zip(&self.coefs) {
            if k[wi] & b == 0 {
                keys.extend_from_slice(k);
                coefs.push(c);
                continue;
            }
            let is_y = k[w + wi] & b != 0;
            keys.extend_from_slice(k);
            coefs.push(c * cs);
            keys.extend_from_slice(k);
            let last = keys.len() - s;
            keys[last + w + wi] ^= b; // X <-> Y
            coefs.push(if is_y { -c * sn } else { c * sn });
        }
        if coefs.len() > max_terms {
            return Err(SimError::TooManyTerms {
                terms: coefs.len(),
                limit: max_terms,
            });
        }
        // Merge equal strings: sort term indices by key, then sum runs.
        let mut idx: Vec<usize> = (0..coefs.len()).collect();
        idx.par_sort_unstable_by(|&i, &j| keys[i * s..(i + 1) * s].cmp(&keys[j * s..(j + 1) * s]));
        self.keys.clear();
        self.coefs.clear();
        let mut r = 0;
        while r < idx.len() {
            let key = &keys[idx[r] * s..(idx[r] + 1) * s];
            let mut c = 0.0;
            while r < idx.len() && &keys[idx[r] * s..(idx[r] + 1) * s] == key {
                c += coefs[idx[r]];
                r += 1;
            }
            if c.abs() > 1e-14 {
                self.keys.extend_from_slice(key);
                self.coefs.push(c);
            }
        }
        Ok(())
    }
}

/// Conjugates one Hermitian Pauli string (`2w` words, x then z) by a
/// Clifford gate, `P -> G P G†`, in place. Returns 1 if the sign flips.
pub(crate) fn conj_string(k: &mut [u64], w: usize, g: &Gate) -> u64 {
    let bit = |k: &[u64], q: usize| (k[q / 64] >> (q % 64)) & 1;
    let flip = |k: &mut [u64], q: usize| k[q / 64] ^= 1 << (q % 64);
    match *g {
        Gate::H(a) => {
            let (x, z) = (bit(k, a), bit(k, w * 64 + a));
            if x != z {
                flip(k, a);
                flip(k, w * 64 + a);
            }
            x & z
        }
        Gate::S(a) | Gate::Sdg(a) => {
            let (x, z) = (bit(k, a), bit(k, w * 64 + a));
            if x == 1 {
                flip(k, w * 64 + a);
            }
            if matches!(g, Gate::S(_)) {
                x & z
            } else {
                x & (z ^ 1)
            }
        }
        Gate::X(a) => bit(k, w * 64 + a),
        Gate::Z(a) => bit(k, a),
        Gate::Y(a) => bit(k, a) ^ bit(k, w * 64 + a),
        Gate::Cnot(ct, t) => {
            let (xc, zc) = (bit(k, ct), bit(k, w * 64 + ct));
            let (xt, zt) = (bit(k, t), bit(k, w * 64 + t));
            if xc == 1 {
                flip(k, t);
            }
            if zt == 1 {
                flip(k, w * 64 + ct);
            }
            xc & zt & (xt ^ zc ^ 1)
        }
        Gate::Cz(a, b) => {
            let (xa, za) = (bit(k, a), bit(k, w * 64 + a));
            let (xb, zb) = (bit(k, b), bit(k, w * 64 + b));
            if xb == 1 {
                flip(k, w * 64 + a);
            }
            if xa == 1 {
                flip(k, w * 64 + b);
            }
            xa & xb & (za ^ zb)
        }
        Gate::Swap(a, b) => {
            for off in [0, w * 64] {
                if bit(k, off + a) != bit(k, off + b) {
                    flip(k, off + a);
                    flip(k, off + b);
                }
            }
            0
        }
        _ => unreachable!("not a Clifford gate: {g:?}"),
    }
}

/// Exact expectation value `<0| U† O U |0>` of a Pauli observable for a
/// circuit `U` of arbitrary gates. Uses the rotation-frame engine with
/// exact pruning ([`FrameOptions::default`]); see [`crate::pauli_frame`].
/// Returns the value and path statistics.
pub fn expectation(
    circuit: &Circuit,
    observable: &PauliSum,
    max_terms: usize,
) -> Result<(f64, PathStats), SimError> {
    expectation_with(
        circuit,
        observable,
        &FrameOptions {
            max_terms,
            ..FrameOptions::default()
        },
    )
}

/// [`expectation`] with explicit engine options. Falls back to the legacy
/// engine for registers wider than 512 qubits.
pub fn expectation_with(
    circuit: &Circuit,
    observable: &PauliSum,
    opt: &FrameOptions,
) -> Result<(f64, PathStats), SimError> {
    assert_eq!(observable.n, circuit.num_qubits);
    match crate::pauli_frame::expectation(circuit, observable, opt) {
        Some(r) => r,
        None => expectation_legacy(circuit, observable, opt.max_terms),
    }
}

/// The original engine: pushes the observable through every gate, one pass
/// over all terms per gate, and merges by sorting after every split.
/// Kept as the reference for cross-checks and benchmarks.
pub fn expectation_legacy(
    circuit: &Circuit,
    observable: &PauliSum,
    max_terms: usize,
) -> Result<(f64, PathStats), SimError> {
    assert_eq!(observable.n, circuit.num_qubits);
    let mut gates = Vec::new();
    for op in &circuit.ops {
        match op {
            Op::Gate(g) => {
                check_gate(g, circuit.num_qubits)?;
                gates.extend(g.decompose_to_clifford_rz());
            }
            _ => panic!("pauli_path::expectation: circuit must be unitary"),
        }
    }
    let mut o = observable.clone();
    let mut stats = PathStats {
        peak_terms: o.num_terms(),
        ..Default::default()
    };
    // O -> G† O G for the last gate first; G† O G = (G†) O (G†)†.
    for g in gates.iter().rev().map(|g| g.inverse()) {
        if g.is_clifford() {
            o.conjugate_clifford(&g);
            continue;
        }
        let (a, theta) = match g {
            Gate::T(a) => (a, FRAC_PI_4),
            Gate::Tdg(a) => (a, -FRAC_PI_4),
            Gate::Phase(a, t) | Gate::Rz(a, t) => (a, t),
            _ => unreachable!("decomposition produced {g:?}"),
        };
        if is_multiple_of_half_pi(theta) {
            let k = (theta / std::f64::consts::FRAC_PI_2)
                .round()
                .rem_euclid(4.0) as usize;
            for _ in 0..k {
                o.conjugate_clifford(&Gate::S(a));
            }
            continue;
        }
        stats.non_clifford_gates += 1;
        stats.rotations += 1;
        stats.term_visits += o.num_terms() as u64;
        o.conjugate_phase(a, theta, max_terms)?;
        stats.peak_terms = stats.peak_terms.max(o.num_terms());
    }
    stats.final_terms = o.num_terms();
    Ok((o.expectation_zero_state(), stats))
}

/// `<Z_q>` after the circuit.
pub fn expectation_z(circuit: &Circuit, q: usize) -> Result<f64, SimError> {
    let obs = PauliSum::z_product(circuit.num_qubits, &[q]);
    expectation(circuit, &obs, DEFAULT_MAX_TERMS).map(|(v, _)| v)
}

/// Exact joint distribution of measuring `qubits` after the circuit, via
/// the `2^k` Z-product expectation values:
/// `P(b) = 2^-k sum_S (-1)^{b·S} <Z_S>`. Index bit `i` = outcome of
/// `qubits[i]`.
pub fn marginal_distribution(circuit: &Circuit, qubits: &[usize]) -> Result<Vec<f64>, SimError> {
    let k = qubits.len();
    assert!(k <= 16, "marginal over too many qubits");
    let mut ev = vec![0.0; 1 << k];
    for (mask, e) in ev.iter_mut().enumerate() {
        let sel: Vec<usize> = (0..k)
            .filter(|i| mask >> i & 1 == 1)
            .map(|i| qubits[i])
            .collect();
        let obs = PauliSum::z_product(circuit.num_qubits, &sel);
        *e = expectation(circuit, &obs, DEFAULT_MAX_TERMS)?.0;
    }
    Ok((0..1usize << k)
        .map(|b| {
            ev.iter()
                .enumerate()
                .map(|(mask, e)| {
                    if (b & mask).count_ones() % 2 == 1 {
                        -e
                    } else {
                        *e
                    }
                })
                .sum::<f64>()
                / (1 << k) as f64
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_t_on_plus_state() {
        // T H|0>: <X> = cos(π/4), <Y> = sin(π/4), <Z> = 0
        let mut c = Circuit::new(1);
        c.h(0).t(0);
        let h = std::f64::consts::FRAC_1_SQRT_2;
        let x = expectation(&c, &PauliSum::from_str_single("X"), 100)
            .unwrap()
            .0;
        let y = expectation(&c, &PauliSum::from_str_single("Y"), 100)
            .unwrap()
            .0;
        let z = expectation(&c, &PauliSum::from_str_single("Z"), 100)
            .unwrap()
            .0;
        assert!((x - h).abs() < 1e-12 && (y - h).abs() < 1e-12 && z.abs() < 1e-12);
    }

    #[test]
    fn clifford_circuit_stays_one_term() {
        let mut c = Circuit::new(3);
        c.h(0).cnot(0, 1).s(1).cz(1, 2).h(2);
        let obs = PauliSum::z_product(3, &[2]);
        let (_, st) = expectation_legacy(&c, &obs, 10).unwrap();
        assert_eq!(st.peak_terms, 1);
        assert_eq!(st.non_clifford_gates, 0);
        // The frame engine may drop the single term outright: with no
        // rotations, any string with an X or Y has value 0.
        let (v, st) = expectation(&c, &obs, 10).unwrap();
        assert!(st.peak_terms <= 1);
        assert_eq!(st.non_clifford_gates, 0);
        assert_eq!(v, expectation_legacy(&c, &obs, 10).unwrap().0);
    }

    #[test]
    fn clifford_conjugation_of_observable() {
        let mut c = Circuit::new(2);
        c.h(0).cnot(0, 1);
        let mut p = PauliSum::from_str_single("ZI");
        p.conjugate_by_clifford(&c).unwrap();
        assert_eq!(p, PauliSum::from_str_single("XX"));
        c.t(1);
        assert!(p.conjugate_by_clifford(&c).is_err());
    }

    #[test]
    fn term_budget_is_enforced() {
        let n = 8;
        let mut c = Circuit::new(n);
        for q in 0..n {
            c.h(q).t(q).h(q).t(q);
        }
        let obs = PauliSum::from_str_single("XXXXXXXX");
        assert!(matches!(
            expectation_legacy(&c, &obs, 4),
            Err(SimError::TooManyTerms { .. })
        ));
        let noprune = FrameOptions {
            max_terms: 4,
            prune: false,
            ..FrameOptions::default()
        };
        assert!(matches!(
            expectation_with(&c, &obs, &noprune),
            Err(SimError::TooManyTerms { .. })
        ));
    }
}
