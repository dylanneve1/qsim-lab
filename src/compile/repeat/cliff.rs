//! Exact fast paths for Clifford blocks.
//!
//! A Clifford `U` acts on Paulis by conjugation, `P -> U P U†`. Its action
//! on the `2n` generators `X_j`, `Z_j` (with signs) is the *symplectic
//! tableau* of `U`: [`CliffordMap`]. Maps compose, so `U^r` costs
//! `O(n^3 log r / 64)` by repeated squaring instead of `r·|B|` gate
//! applications. A map determines `U` up to a global phase, which is all a
//! sampling or expectation request can see (not an amplitude request).
//!
//! * [`CliffordMap::from_gates`], [`CliffordMap::then`], [`CliffordMap::pow`]
//! * [`CliffordMap::synthesize`]: a gate list with that map (`O(n^2)` gates)
//! * [`power_gates`]: `B^r` as a short gate list
//! * [`canonical_stabilizers`]: row-reduced stabilizer group, for exact
//!   state comparison
//! * [`sample_program`]: shot sampling of Clifford programs with measurement
//!   rounds. Once a repeated block leaves the state unchanged and all its
//!   measurements were deterministic, every further copy gives the same
//!   outcomes and the copies are skipped.

use super::{Node, Program};
use crate::circuit::Op;
use crate::engines::stabilizer::Tableau;
use crate::gate::{Gate, Mat2, Mat4};
use num_complex::Complex64;
use rand::Rng;
use std::mem::{discriminant, Discriminant};

/// `i^k X^x Z^z` over `n` qubits (bit `q` of the vectors is qubit `q`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pauli {
    /// X part, packed in 64-bit words.
    pub x: Vec<u64>,
    /// Z part, packed in 64-bit words.
    pub z: Vec<u64>,
    /// Phase exponent: the operator carries the factor `i^k`, `k` in `0..4`.
    pub k: u8,
}

impl Pauli {
    /// The identity with `words` 64-bit words per part.
    pub fn identity(words: usize) -> Self {
        Pauli {
            x: vec![0; words],
            z: vec![0; words],
            k: 0,
        }
    }

    /// `self * o`, using `X^a Z^b X^c Z^d = (-1)^{b·c} X^{a+c} Z^{b+d}`.
    pub fn mul_assign(&mut self, o: &Pauli) {
        let mut s = 0u32;
        for i in 0..self.x.len() {
            s += (self.z[i] & o.x[i]).count_ones();
            self.x[i] ^= o.x[i];
            self.z[i] ^= o.z[i];
        }
        self.k = ((self.k as u32 + o.k as u32 + 2 * (s & 1)) & 3) as u8;
    }

    fn bit(v: &[u64], i: usize) -> bool {
        v[i / 64] >> (i % 64) & 1 == 1
    }

    /// True if the Pauli is `±` a Hermitian string (even `k - |x∧z|`).
    pub fn is_hermitian(&self) -> bool {
        let pop: u32 = self
            .x
            .iter()
            .zip(&self.z)
            .map(|(a, b)| (a & b).count_ones())
            .sum();
        (self.k as u32 + 4 - (pop & 3)).is_multiple_of(2)
    }
}

/// Conjugation table of a Clifford gate on 1 or 2 qubits: local Pauli index
/// `xa | za<<1 | xb<<2 | zb<<3` -> `(new index, sign flip)`.
#[derive(Clone, Debug)]
struct GateTable {
    arity: usize,
    map: Vec<(u8, bool)>,
}

fn c(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

fn pauli2(idx: u8) -> Mat2 {
    let (x, z) = (idx & 1, idx >> 1 & 1);
    match (x, z) {
        (0, 0) => [[c(1., 0.), c(0., 0.)], [c(0., 0.), c(1., 0.)]],
        (1, 0) => [[c(0., 0.), c(1., 0.)], [c(1., 0.), c(0., 0.)]],
        (0, 1) => [[c(1., 0.), c(0., 0.)], [c(0., 0.), c(-1., 0.)]],
        _ => [[c(0., 0.), c(0., -1.)], [c(0., 1.), c(0., 0.)]],
    }
}

fn kron(a: &Mat2, b: &Mat2) -> Mat4 {
    let mut m = [[c(0., 0.); 4]; 4];
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..2 {
                for l in 0..2 {
                    m[2 * i + k][2 * j + l] = a[i][j] * b[k][l];
                }
            }
        }
    }
    m
}

fn matmul<const N: usize>(a: &[[Complex64; N]; N], b: &[[Complex64; N]; N]) -> [[Complex64; N]; N] {
    let mut m = [[c(0., 0.); N]; N];
    for i in 0..N {
        for j in 0..N {
            for k in 0..N {
                m[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    m
}

fn dagger<const N: usize>(a: &[[Complex64; N]; N]) -> [[Complex64; N]; N] {
    let mut m = [[c(0., 0.); N]; N];
    for i in 0..N {
        for j in 0..N {
            m[i][j] = a[j][i].conj();
        }
    }
    m
}

fn close<const N: usize>(a: &[[Complex64; N]; N], b: &[[Complex64; N]; N], sign: f64) -> bool {
    (0..N).all(|i| (0..N).all(|j| (a[i][j] - b[i][j] * sign).norm() < 1e-9))
}

impl GateTable {
    /// Derived numerically from the gate's matrix, so no hand-written rules
    /// can disagree with the state vector.
    fn of(g: &Gate) -> Option<GateTable> {
        if !g.is_clifford() {
            return None;
        }
        if let Some(u) = g.matrix_1q() {
            let ud = dagger(&u);
            let mut map = Vec::new();
            for p in 0..4u8 {
                let img = matmul(&matmul(&u, &pauli2(p)), &ud);
                let mut found = None;
                for q in 0..4u8 {
                    for (flip, s) in [(false, 1.0), (true, -1.0)] {
                        if close(&img, &pauli2(q), s) {
                            found = Some((q, flip));
                        }
                    }
                }
                map.push(found?);
            }
            Some(GateTable { arity: 1, map })
        } else if let Some(u) = g.matrix_2q() {
            let ud = dagger(&u);
            let pm = |p: u8| kron(&pauli2(p & 3), &pauli2(p >> 2));
            let mut map = Vec::new();
            for p in 0..16u8 {
                let img = matmul(&matmul(&u, &pm(p)), &ud);
                let mut found = None;
                for q in 0..16u8 {
                    for (flip, s) in [(false, 1.0), (true, -1.0)] {
                        if close(&img, &pm(q), s) {
                            found = Some((q, flip));
                        }
                    }
                }
                map.push(found?);
            }
            Some(GateTable { arity: 2, map })
        } else {
            None
        }
    }
}

/// The action of a Clifford `U` on Paulis: images of `X_j` (rows `0..n`)
/// and `Z_j` (rows `n..2n`) under `P -> U P U†`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CliffordMap {
    n: usize,
    w: usize,
    rows: Vec<Pauli>,
}

/// Caches conjugation tables per gate kind.
#[derive(Default)]
pub struct TableCache {
    tabs: Vec<(Discriminant<Gate>, Option<GateTable>)>,
}

impl TableCache {
    fn get(&mut self, g: &Gate) -> Option<&GateTable> {
        let d = discriminant(g);
        if let Some(i) = self.tabs.iter().position(|(k, _)| *k == d) {
            return self.tabs[i].1.as_ref();
        }
        self.tabs.push((d, GateTable::of(g)));
        self.tabs.last().unwrap().1.as_ref()
    }
}

impl CliffordMap {
    /// The identity map on `n` qubits.
    pub fn identity(n: usize) -> Self {
        let w = n.div_ceil(64).max(1);
        let mut rows = Vec::with_capacity(2 * n);
        for j in 0..n {
            let mut p = Pauli::identity(w);
            p.x[j / 64] |= 1 << (j % 64);
            rows.push(p);
        }
        for j in 0..n {
            let mut p = Pauli::identity(w);
            p.z[j / 64] |= 1 << (j % 64);
            rows.push(p);
        }
        CliffordMap { n, w, rows }
    }

    /// Number of qubits.
    pub fn num_qubits(&self) -> usize {
        self.n
    }

    /// Row `i` as `(sign, text)`, e.g. `"-XZI"` (qubit 0 first); for tests.
    pub fn row_string(&self, i: usize) -> String {
        let r = &self.rows[i];
        let pop: u32 =
            r.x.iter()
                .zip(&r.z)
                .map(|(a, b)| (a & b).count_ones())
                .sum();
        let s = ((r.k as u32 + 4 - (pop & 3)) & 3) / 2;
        let mut out = String::from(if s == 1 { "-" } else { "+" });
        for q in 0..self.n {
            out.push(match (Pauli::bit(&r.x, q), Pauli::bit(&r.z, q)) {
                (false, false) => 'I',
                (true, false) => 'X',
                (true, true) => 'Y',
                (false, true) => 'Z',
            });
        }
        out
    }

    /// `g` after the current map: `M <- M_g ∘ M`. Returns false for a
    /// non-Clifford gate.
    pub fn apply_gate(&mut self, g: &Gate, cache: &mut TableCache) -> bool {
        let Some(tab) = cache.get(g) else {
            return false;
        };
        let qs = g.qubits();
        let (a, b) = (qs[0], qs.get(1).copied().unwrap_or(0));
        let two = tab.arity == 2;
        for r in &mut self.rows {
            let bit = |v: &[u64], q: usize| (v[q / 64] >> (q % 64)) & 1;
            let (xa, za) = (bit(&r.x, a), bit(&r.z, a));
            let (xb, zb) = if two {
                (bit(&r.x, b), bit(&r.z, b))
            } else {
                (0, 0)
            };
            let idx = (xa | za << 1 | xb << 2 | zb << 3) as usize;
            let (new, flip) = tab.map[idx];
            let new = new as u64;
            let old_pop = (xa & za) + (xb & zb);
            let (nxa, nza, nxb, nzb) = (new & 1, new >> 1 & 1, new >> 2 & 1, new >> 3 & 1);
            let new_pop = (nxa & nza) + (nxb & nzb);
            let set = |v: &mut [u64], q: usize, val: u64| {
                v[q / 64] = (v[q / 64] & !(1 << (q % 64))) | (val << (q % 64));
            };
            set(&mut r.x, a, nxa);
            set(&mut r.z, a, nza);
            if two {
                set(&mut r.x, b, nxb);
                set(&mut r.z, b, nzb);
            }
            r.k = ((r.k as i64 + 2 * flip as i64 + new_pop as i64 - old_pop as i64).rem_euclid(4))
                as u8;
        }
        true
    }

    /// The map of a Clifford gate list on `n` qubits, or `None` if some gate
    /// is not Clifford.
    pub fn from_gates(n: usize, gates: &[Gate]) -> Option<Self> {
        let mut m = CliffordMap::identity(n);
        let mut cache = TableCache::default();
        for g in gates {
            if !m.apply_gate(g, &mut cache) {
                return None;
            }
        }
        Some(m)
    }

    /// `other ∘ self`: first `self`, then `other`.
    pub fn then(&self, other: &CliffordMap) -> CliffordMap {
        assert_eq!(self.n, other.n);
        let rows = self
            .rows
            .iter()
            .map(|r| {
                let mut acc = Pauli::identity(self.w);
                for j in 0..self.n {
                    if Pauli::bit(&r.x, j) {
                        acc.mul_assign(&other.rows[j]);
                    }
                }
                for j in 0..self.n {
                    if Pauli::bit(&r.z, j) {
                        acc.mul_assign(&other.rows[self.n + j]);
                    }
                }
                acc.k = ((acc.k as u32 + r.k as u32) & 3) as u8;
                acc
            })
            .collect();
        CliffordMap {
            n: self.n,
            w: self.w,
            rows,
        }
    }

    /// `M^e` by repeated squaring.
    pub fn pow(&self, mut e: u64) -> CliffordMap {
        let mut result = CliffordMap::identity(self.n);
        let mut base = self.clone();
        while e > 0 {
            if e & 1 == 1 {
                result = result.then(&base);
            }
            e >>= 1;
            if e > 0 {
                base = base.then(&base);
            }
        }
        result
    }

    /// Image of a stabilizer state's generators: applies the map to Pauli
    /// rows (given in generic form).
    pub fn apply_to(&self, p: &Pauli) -> Pauli {
        let mut acc = Pauli::identity(self.w);
        for j in 0..self.n {
            if Pauli::bit(&p.x, j) {
                acc.mul_assign(&self.rows[j]);
            }
        }
        for j in 0..self.n {
            if Pauli::bit(&p.z, j) {
                acc.mul_assign(&self.rows[self.n + j]);
            }
        }
        acc.k = ((acc.k as u32 + p.k as u32) & 3) as u8;
        acc
    }

    fn sym(&self, row: usize, q: usize) -> (bool, bool) {
        (
            Pauli::bit(&self.rows[row].x, q),
            Pauli::bit(&self.rows[row].z, q),
        )
    }

    fn sign(&self, row: usize) -> bool {
        let r = &self.rows[row];
        let pop: u32 =
            r.x.iter()
                .zip(&r.z)
                .map(|(a, b)| (a & b).count_ones())
                .sum();
        ((r.k as u32 + 4 - (pop & 3)) & 3) / 2 == 1
    }

    /// A gate list `G` (time order) whose map equals `self`. Gates are
    /// `H, S, Sx, X, Z, CNOT, SWAP`. Equal up to a global phase.
    ///
    /// Column-by-column reduction: bring `M(X_j)` to `X_j` and `M(Z_j)` to
    /// `Z_j` with gates on qubits `>= j` (the earlier rows are already the
    /// identity, so the later images avoid qubits `< j`), then fix the signs.
    pub fn synthesize(&self) -> Vec<Gate> {
        let n = self.n;
        let mut m = self.clone();
        let mut cache = TableCache::default();
        let mut seq: Vec<Gate> = Vec::new();
        let mut put = |m: &mut CliffordMap, g: Gate, seq: &mut Vec<Gate>| {
            let ok = m.apply_gate(&g, &mut cache);
            debug_assert!(ok);
            seq.push(g);
        };
        for j in 0..n {
            // X_j image -> X-type everywhere
            for q in j..n {
                match m.sym(j, q) {
                    (false, true) => put(&mut m, Gate::H(q), &mut seq),
                    (true, true) => put(&mut m, Gate::S(q), &mut seq),
                    _ => {}
                }
            }
            let support: Vec<usize> = (j..n).filter(|&q| m.sym(j, q).0).collect();
            let t = if support.contains(&j) { j } else { support[0] };
            for &q in &support {
                if q != t {
                    put(&mut m, Gate::Cnot(t, q), &mut seq);
                }
            }
            if t != j {
                put(&mut m, Gate::Swap(t, j), &mut seq);
            }
            // Z_j image -> Z_j or Y_j, then Z_j
            let zr = n + j;
            for q in j + 1..n {
                match m.sym(zr, q) {
                    (true, false) => put(&mut m, Gate::H(q), &mut seq),
                    (true, true) => {
                        put(&mut m, Gate::S(q), &mut seq);
                        put(&mut m, Gate::H(q), &mut seq);
                    }
                    _ => {}
                }
            }
            for q in j + 1..n {
                if m.sym(zr, q).1 {
                    put(&mut m, Gate::Cnot(q, j), &mut seq);
                }
            }
            if m.sym(zr, j) == (true, true) {
                put(&mut m, Gate::Sx(j), &mut seq);
            }
            if m.sign(j) {
                put(&mut m, Gate::Z(j), &mut seq);
            }
            if m.sign(zr) {
                put(&mut m, Gate::X(j), &mut seq);
            }
        }
        debug_assert_eq!(m, CliffordMap::identity(n));
        // g_m … g_1 ∘ M = I  =>  U = g_1† … g_m†: time order g_m†, …, g_1†.
        seq.iter().rev().map(|g| g.inverse()).collect()
    }
}

/// The qubits a gate list touches (sorted) and the gates re-indexed onto
/// `0..k`.
pub fn compact(gates: &[Gate]) -> (Vec<usize>, Vec<Gate>) {
    let mut qs: Vec<usize> = gates.iter().flat_map(|g| g.qubits()).collect();
    qs.sort_unstable();
    qs.dedup();
    let idx = |q: usize| qs.binary_search(&q).unwrap();
    let out = gates.iter().map(|g| remap_gate(g, &idx)).collect();
    (qs, out)
}

/// The gate with every qubit `q` replaced by `f(q)`.
pub fn remap_gate(g: &Gate, f: &dyn Fn(usize) -> usize) -> Gate {
    use Gate::*;
    match *g {
        I(q) => I(f(q)),
        H(q) => H(f(q)),
        X(q) => X(f(q)),
        Y(q) => Y(f(q)),
        Z(q) => Z(f(q)),
        S(q) => S(f(q)),
        Sdg(q) => Sdg(f(q)),
        T(q) => T(f(q)),
        Tdg(q) => Tdg(f(q)),
        Sx(q) => Sx(f(q)),
        Sxdg(q) => Sxdg(f(q)),
        Rx(q, a) => Rx(f(q), a),
        Ry(q, a) => Ry(f(q), a),
        Rz(q, a) => Rz(f(q), a),
        Phase(q, a) => Phase(f(q), a),
        U(q, a, b, c) => U(f(q), a, b, c),
        Cnot(a, b) => Cnot(f(a), f(b)),
        Cz(a, b) => Cz(f(a), f(b)),
        Swap(a, b) => Swap(f(a), f(b)),
        ISwap(a, b) => ISwap(f(a), f(b)),
        ISwapdg(a, b) => ISwapdg(f(a), f(b)),
        CPhase(a, b, t) => CPhase(f(a), f(b), t),
        Ccx(a, b, t) => Ccx(f(a), f(b), f(t)),
    }
}

/// `B^r` for a Clifford gate list `B` as a gate list equal up to a global
/// phase, via the symplectic map. `None` if some gate is not Clifford.
pub fn power_gates(body: &[Gate], r: u64) -> Option<Vec<Gate>> {
    let (qs, local) = compact(body);
    let k = qs.len();
    let m = CliffordMap::from_gates(k, &local)?;
    let g = m.pow(r).synthesize();
    Some(g.iter().map(|g| remap_gate(g, &|q| qs[q])).collect())
}

// ------------------------------------------------------ stabilizer states

/// Parses `Tableau::stabilizers()` rows into Paulis.
pub fn stabilizer_rows(tab: &mut Tableau) -> Vec<Pauli> {
    let n = tab.num_qubits();
    let w = n.div_ceil(64).max(1);
    tab.stabilizers()
        .into_iter()
        .map(|s| {
            let mut p = Pauli::identity(w);
            let mut chars = s.chars();
            let neg = chars.next() == Some('-');
            let mut pop = 0u32;
            for (q, ch) in chars.enumerate() {
                let (x, z) = match ch {
                    'X' => (1, 0),
                    'Y' => (1, 1),
                    'Z' => (0, 1),
                    _ => (0, 0),
                };
                p.x[q / 64] |= x << (q % 64);
                p.z[q / 64] |= z << (q % 64);
                pop += (x & z) as u32;
            }
            p.k = ((pop + 2 * neg as u32) & 3) as u8;
            p
        })
        .collect()
}

/// Reduced row echelon form of a stabilizer group (signs tracked), unique
/// for a given group: equal forms <=> equal states.
pub fn canonical_stabilizers(mut rows: Vec<Pauli>, n: usize) -> Vec<Pauli> {
    let mut rank = 0;
    for col in 0..2 * n {
        let (v, q) = if col < n { (0, col) } else { (1, col - n) };
        let has = |p: &Pauli| {
            let vec = if v == 0 { &p.x } else { &p.z };
            Pauli::bit(vec, q)
        };
        let Some(piv) = (rank..rows.len()).find(|&i| has(&rows[i])) else {
            continue;
        };
        rows.swap(rank, piv);
        let pr = rows[rank].clone();
        for (i, row) in rows.iter_mut().enumerate() {
            if i != rank && has(row) {
                row.mul_assign(&pr);
            }
        }
        rank += 1;
    }
    rows
}

/// Canonical form of a tableau's state.
pub fn canonical_state(tab: &mut Tableau) -> Vec<Pauli> {
    let n = tab.num_qubits();
    canonical_stabilizers(stabilizer_rows(tab), n)
}

// ------------------------------------------------------------- sampling

/// What [`sample_program`] did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SampleStats {
    /// Repeat copies executed op by op.
    pub copies_run: usize,
    /// Copies skipped because the block reached a deterministic steady state.
    pub copies_skipped: usize,
    /// Copies skipped by the symplectic power of a unitary Clifford block.
    pub copies_powered: usize,
}

/// True if `p` only has Clifford gates, measurements and resets.
pub fn is_clifford_program(p: &Program) -> bool {
    fn ok(nodes: &[Node]) -> bool {
        nodes.iter().all(|n| match n {
            Node::Ops(o) => o.iter().all(|op| match op {
                Op::Gate(g) => g.is_clifford(),
                Op::Measure(_) | Op::Reset(_) => true,
                _ => false,
            }),
            Node::Repeat { body, .. } => ok(body),
            Node::Param { .. } => false,
        })
    }
    ok(&p.nodes)
}

fn body_gates(nodes: &[Node]) -> Option<Vec<Gate>> {
    let mut out = Vec::new();
    for n in nodes {
        match n {
            Node::Ops(o) => {
                for op in o {
                    match op {
                        Op::Gate(g) => out.push(*g),
                        _ => return None,
                    }
                }
            }
            Node::Repeat { body, reps } => {
                let g = body_gates(body)?;
                if g.len().saturating_mul(*reps) > 1 << 22 {
                    return None;
                }
                for _ in 0..*reps {
                    out.extend_from_slice(&g);
                }
            }
            Node::Param { .. } => return None,
        }
    }
    Some(out)
}

const PROBE_COPIES: usize = 4;

struct Sampler<'a, R: Rng> {
    tab: Tableau,
    rng: &'a mut R,
    out: Vec<bool>,
    random: bool,
    stats: SampleStats,
    steady: bool,
    power: bool,
}

impl<R: Rng> Sampler<'_, R> {
    fn run(&mut self, nodes: &[Node]) {
        for node in nodes {
            match node {
                Node::Ops(ops) => {
                    for op in ops {
                        self.op(op);
                    }
                }
                Node::Repeat { body, reps } => self.repeat(body, *reps),
                Node::Param { .. } => unreachable!("checked by is_clifford_program"),
            }
        }
    }

    fn op(&mut self, op: &Op) {
        match op {
            Op::Gate(g) => self.tab.apply_gate(g).expect("clifford gate"),
            Op::Measure(q) => {
                let (b, rnd) = self.tab.measure_with(*q, None, self.rng);
                self.random |= rnd;
                self.out.push(b);
            }
            Op::Reset(q) => {
                let (b, rnd) = self.tab.measure_with(*q, None, self.rng);
                self.random |= rnd;
                if b {
                    self.tab.x(*q);
                }
            }
            _ => unreachable!("checked by is_clifford_program"),
        }
    }

    fn repeat(&mut self, body: &[Node], reps: usize) {
        // Unitary Clifford block: symplectic power, if cheaper.
        if self.power {
            if let Some(g) = body_gates(body) {
                let cost_plain = g.len().saturating_mul(reps);
                let k = super::cliff::compact(&g).0.len();
                if cost_plain > 25 * (k * k + 8) {
                    if let Some(p) = power_gates(&g, reps as u64) {
                        for gate in &p {
                            self.tab.apply_gate(gate).expect("clifford");
                        }
                        // The synthesised gates fix the state up to a global
                        // phase, which a tableau does not carry.
                        self.stats.copies_powered += reps;
                        return;
                    }
                }
            }
        }
        let mut done = 0;
        while done < reps {
            let probing = self.steady && done < PROBE_COPIES;
            let before = if probing {
                Some(self.tab.clone())
            } else {
                None
            };
            let start = self.out.len();
            let rnd_before = std::mem::replace(&mut self.random, false);
            self.run(body);
            done += 1;
            self.stats.copies_run += 1;
            let was_random = self.random;
            self.random |= rnd_before;
            if let Some(mut before) = before {
                if !was_random && done < reps {
                    let a = canonical_state(&mut before);
                    let b = canonical_state(&mut self.tab);
                    if a == b {
                        let block: Vec<bool> = self.out[start..].to_vec();
                        let left = reps - done;
                        self.out.reserve(block.len() * left);
                        for _ in 0..left {
                            self.out.extend_from_slice(&block);
                        }
                        self.stats.copies_skipped += left;
                        return;
                    }
                }
            }
        }
    }
}

/// One shot of a Clifford program (see [`is_clifford_program`]) on a
/// tableau, drawing from `rng` exactly like `Circuit::run` on a `Tableau`
/// would for the expanded circuit when every skipped round is
/// deterministic (which is the only case that is skipped). Returns `None` if
/// the program is not a Clifford program.
pub fn sample_program<R: Rng>(
    p: &Program,
    steady_state: bool,
    power: bool,
    rng: &mut R,
) -> Option<(Vec<bool>, SampleStats)> {
    if !is_clifford_program(p) {
        return None;
    }
    let mut s = Sampler {
        tab: Tableau::new(p.num_qubits),
        rng,
        out: Vec::new(),
        random: false,
        stats: SampleStats::default(),
        steady: steady_state,
        power,
    };
    s.run(&p.nodes);
    Some((s.out, s.stats))
}
