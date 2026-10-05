//! Pauli-frame front end: a Clifford+T circuit as a sequence of `π/8`
//! Pauli rotations followed by a Clifford (`U = C · Π_j e^{-iπ/8 · k_j P_j}`
//! up to a global phase), with every rotation axis `P_j` written in the
//! input frame. Rotations whose axes commute can be reordered, merged
//! (equal axes) and grouped; a group of pairwise commuting axes is
//! diagonalised by a Clifford and becomes one phase polynomial, which is
//! where TODD applies. This generalises the slot model of the parent
//! module, which only groups terms between the circuit's own Hadamards.

use super::gf2::Bits;
use super::PGate;

/// A Hermitian Pauli string `(-1)^sign · i^{|x∧z|} X^x Z^z` (so `x = z = 1`
/// on a qubit means `Y`).
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Pauli {
    /// X part.
    pub x: Bits,
    /// Z part.
    pub z: Bits,
    /// Sign bit.
    pub sign: bool,
}

impl Pauli {
    /// `Z_q` on `n` qubits.
    pub fn z(n: usize, q: usize) -> Self {
        Pauli {
            x: Bits::zeros(n),
            z: Bits::unit(n, q),
            sign: false,
        }
    }

    /// `X_q` on `n` qubits.
    pub fn x(n: usize, q: usize) -> Self {
        Pauli {
            x: Bits::unit(n, q),
            z: Bits::zeros(n),
            sign: false,
        }
    }

    /// True if the two strings commute.
    pub fn commutes(&self, o: &Pauli) -> bool {
        !(self.x.dot(&o.z) ^ self.z.dot(&o.x))
    }

    /// True if the string has no X/Y factor.
    pub fn is_diagonal(&self) -> bool {
        self.x.is_zero()
    }

    /// The product `self · o` of two commuting Hermitian strings (again
    /// Hermitian).
    pub fn mul_commuting(&self, o: &Pauli) -> Pauli {
        debug_assert!(self.commutes(o));
        // X^a Z^b X^c Z^d = (-1)^{b·c} X^{a+c} Z^{b+d}; with the i^{|x∧z|}
        // convention the phase of the product is i^{e} with
        // e = |a∧b| + |c∧d| + 2(b·c) - |(a+c)∧(b+d)|  (mod 4), which is even
        // for commuting strings.
        let ab = self.x.and(&self.z).count_ones() as i64;
        let cd = o.x.and(&o.z).count_ones() as i64;
        let bc = self.z.and(&o.x).count_ones() as i64;
        let mut x = self.x.clone();
        x.xor_with(&o.x);
        let mut z = self.z.clone();
        z.xor_with(&o.z);
        let xz = x.and(&z).count_ones() as i64;
        let e = (ab + cd + 2 * bc - xz).rem_euclid(4);
        debug_assert!(e % 2 == 0, "commuting Hermitian strings multiply to a Hermitian string");
        Pauli {
            x,
            z,
            sign: self.sign ^ o.sign ^ (e == 2),
        }
    }
}

/// The images `C† Z_q C` and `C† X_q C` of the accumulated Clifford `C`
/// (the inverse tableau), updated gate by gate.
#[derive(Clone, Debug)]
pub struct Frame {
    /// `C† Z_q C`.
    pub zs: Vec<Pauli>,
    /// `C† X_q C`.
    pub xs: Vec<Pauli>,
}

impl Frame {
    /// The identity frame on `n` qubits.
    pub fn new(n: usize) -> Self {
        Frame {
            zs: (0..n).map(|q| Pauli::z(n, q)).collect(),
            xs: (0..n).map(|q| Pauli::x(n, q)).collect(),
        }
    }

    /// Appends a Clifford gate `G` (`C ← G·C`): the new images are
    /// `C† (G† P G) C`.
    pub fn apply(&mut self, g: &PGate) {
        match *g {
            PGate::H(q) => std::mem::swap(&mut self.zs[q], &mut self.xs[q]),
            PGate::X(q) => {
                // X† Z X = -Z
                self.zs[q].sign = !self.zs[q].sign;
            }
            PGate::Phase(q, k) => {
                debug_assert!(k % 2 == 0, "only Clifford phases move the frame");
                for _ in 0..(k / 2) % 4 {
                    // S† X S = -Y = -(i X Z) ; Y as Hermitian string: X Z with
                    // i-convention, so image(X) <- -(image(X)·image(Z)) in
                    // the Hermitian product sense with an extra sign.
                    let y = mul_anticommuting(&self.xs[q], &self.zs[q]);
                    let mut y = y;
                    y.sign = !y.sign;
                    self.xs[q] = y;
                }
            }
            PGate::Cnot(c, t) => {
                // CNOT† Z_t CNOT = Z_c Z_t ; CNOT† X_c CNOT = X_c X_t
                let zt = self.zs[c].mul_commuting(&self.zs[t]);
                let xc = self.xs[c].mul_commuting(&self.xs[t]);
                self.zs[t] = zt;
                self.xs[c] = xc;
            }
            PGate::Cz(a, b) => {
                // CZ† X_a CZ = X_a Z_b
                let xa = self.xs[a].mul_commuting(&self.zs[b]);
                let xb = self.xs[b].mul_commuting(&self.zs[a]);
                self.xs[a] = xa;
                self.xs[b] = xb;
            }
            PGate::Swap(a, b) => {
                self.zs.swap(a, b);
                self.xs.swap(a, b);
            }
            PGate::Ccz(..) => panic!("CCZ is not Clifford"),
        }
    }

    /// The input-frame axis of `Z` on the parity of `wires` at the
    /// current point (`C† Z_{w1} ⋯ Z_{wk} C`).
    pub fn z_axis(&self, wires: &[usize]) -> Pauli {
        let mut p = self.zs[wires[0]].clone();
        for &w in &wires[1..] {
            p = p.mul_commuting(&self.zs[w]);
        }
        p
    }
}

/// `i · a · b` for two anticommuting Hermitian strings, which is Hermitian
/// (used for `Y = i X Z`).
fn mul_anticommuting(a: &Pauli, b: &Pauli) -> Pauli {
    debug_assert!(!a.commutes(b));
    let ab = a.x.and(&a.z).count_ones() as i64;
    let cd = b.x.and(&b.z).count_ones() as i64;
    let bc = a.z.and(&b.x).count_ones() as i64;
    let mut x = a.x.clone();
    x.xor_with(&b.x);
    let mut z = a.z.clone();
    z.xor_with(&b.z);
    let xz = x.and(&z).count_ones() as i64;
    // a·b = i^{ab + cd + 2bc - xz} X^x Z^z (times signs); times i:
    let e = (ab + cd + 2 * bc - xz + 1).rem_euclid(4);
    debug_assert!(e % 2 == 0);
    Pauli {
        x,
        z,
        sign: a.sign ^ b.sign ^ (e == 2),
    }
}


/// `ω^{k·(I - P)/2}`: phase `ω^k` on the `-1` eigenspace of `P` (for
/// `P = Z_q` this is `diag(1, ω^k)` on qubit `q`). Odd `k` costs one T.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rotation {
    /// Axis, with `sign == false` (normalised).
    pub axis: Pauli,
    /// Exponent mod 8.
    pub k: u8,
}

/// Forward conjugation `P ← G P G†` for the Clifford gates used in
/// diagonalisation (Aaronson–Gottesman update rules).
pub fn conjugate(p: &mut Pauli, g: &DGate) {
    match *g {
        DGate::H(q) => {
            let (x, z) = (p.x.get(q), p.z.get(q));
            p.sign ^= x && z;
            p.x.set(q, z);
            p.z.set(q, x);
        }
        DGate::S(q) => {
            let (x, z) = (p.x.get(q), p.z.get(q));
            p.sign ^= x && z;
            p.z.set(q, z ^ x);
        }
        DGate::Sdg(q) => {
            // S† = S·S·S
            for _ in 0..3 {
                conjugate(p, &DGate::S(q));
            }
        }
        DGate::Cnot(a, b) => {
            let (xa, za, xb, zb) = (p.x.get(a), p.z.get(a), p.x.get(b), p.z.get(b));
            p.sign ^= xa && zb && !(xb ^ za);
            p.x.set(b, xb ^ xa);
            p.z.set(a, za ^ zb);
        }
    }
}

/// Gates used by the diagonalising Cliffords.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DGate {
    /// Hadamard.
    H(usize),
    /// `S`.
    S(usize),
    /// `S†`.
    Sdg(usize),
    /// `Cnot(control, target)`.
    Cnot(usize, usize),
}

/// A Clifford `D` (as gates, applied in order) with `D P D†` diagonal for
/// every `P` in `axes` (which must pairwise commute), and the conjugated
/// axes.
pub fn diagonalize(axes: &[Pauli]) -> (Vec<DGate>, Vec<Pauli>) {
    let mut rows: Vec<Pauli> = axes.to_vec();
    let mut gates = Vec::new();
    let apply = |rows: &mut Vec<Pauli>, gates: &mut Vec<DGate>, g: DGate| {
        for r in rows.iter_mut() {
            conjugate(r, &g);
        }
        gates.push(g);
    };
    for i in 0..rows.len() {
        let Some(q) = rows[i].x.first_one() else {
            continue;
        };
        // clear the other X bits of row i: CNOT(q, t) maps x_t ^= x_q
        let others: Vec<usize> = rows[i].x.ones().filter(|&t| t != q).collect();
        for t in others {
            apply(&mut rows, &mut gates, DGate::Cnot(q, t));
        }
        // Y -> X on q
        if rows[i].z.get(q) {
            apply(&mut rows, &mut gates, DGate::Sdg(q));
            if rows[i].z.get(q) {
                // S† maps Y to -X or X depending on convention; one more
                // S·S fixes either way (never reached with the rules above)
                apply(&mut rows, &mut gates, DGate::S(q));
                apply(&mut rows, &mut gates, DGate::S(q));
            }
        }
        // clear Z bits on other qubits t: CZ(q,t) = H(t) CNOT(q,t) H(t)
        let zs: Vec<usize> = rows[i].z.ones().filter(|&t| t != q).collect();
        for t in zs {
            apply(&mut rows, &mut gates, DGate::H(t));
            apply(&mut rows, &mut gates, DGate::Cnot(q, t));
            apply(&mut rows, &mut gates, DGate::H(t));
        }
        apply(&mut rows, &mut gates, DGate::H(q));
        debug_assert!(rows[i].x.is_zero());
    }
    debug_assert!(rows.iter().all(|r| r.x.is_zero()));
    (gates, rows)
}

/// The rotation list of a circuit: `U = ω^global · C_final · Π_j R_j`
/// with each `R_j = ω^{k_j (I - P_j)/2}` in the input frame, `R_1`
/// applied first, and `C_final` the product of the circuit's Clifford
/// gates (which is the circuit with its odd phase parts removed).
pub fn rotations(c: &super::PhaseCircuit) -> (Vec<Rotation>, u8) {
    let n = c.num_qubits;
    let mut frame = Frame::new(n);
    let mut out = Vec::new();
    let mut global = c.global;
    let push = |axis: Pauli, k: u8, out: &mut Vec<Rotation>, global: &mut u8| {
        let k = k % 8;
        if k == 0 {
            return;
        }
        let mut axis = axis;
        let k = if axis.sign {
            // ω^{k(I+P)/2} = ω^k · ω^{-k(I-P)/2}
            axis.sign = false;
            *global = (*global + k) % 8;
            (8 - k) % 8
        } else {
            k
        };
        out.push(Rotation { axis, k });
    };
    for g in &c.gates {
        match *g {
            PGate::Phase(q, k) => {
                if k % 2 == 1 {
                    push(frame.z_axis(&[q]), 1, &mut out, &mut global);
                }
                if (k - k % 2) % 8 != 0 {
                    frame.apply(&PGate::Phase(q, k - k % 2));
                }
            }
            PGate::Ccz(a, b, t) => {
                for (set, k) in [
                    (&[a][..], 1u8),
                    (&[b][..], 1),
                    (&[t][..], 1),
                    (&[a, b][..], 7),
                    (&[a, t][..], 7),
                    (&[b, t][..], 7),
                    (&[a, b, t][..], 1),
                ] {
                    push(frame.z_axis(set), k, &mut out, &mut global);
                }
            }
            other => frame.apply(&other),
        }
    }
    (out, global)
}


/// Zhang–Chen merging: each rotation moves back past rotations whose
/// axes commute with it and merges into an equal axis; rotations that
/// become the identity are dropped. Repeated until nothing changes.
pub fn merge(mut rots: Vec<Rotation>) -> Vec<Rotation> {
    loop {
        let before = rots.len();
        let mut kept: Vec<Rotation> = Vec::with_capacity(rots.len());
        for r in rots {
            let mut merged = false;
            for i in (0..kept.len()).rev() {
                if kept[i].axis == r.axis {
                    kept[i].k = (kept[i].k + r.k) % 8;
                    merged = true;
                    break;
                }
                if !kept[i].axis.commutes(&r.axis) {
                    break;
                }
            }
            if !merged {
                kept.push(r);
            }
        }
        kept.retain(|r| r.k != 0);
        let done = kept.len() == before;
        rots = kept;
        if done {
            return rots;
        }
    }
}

/// `R† Q R` for the Clifford rotation `R = ω^{k(I-P)/2}` (`k` even) and
/// a Hermitian string `Q`: unchanged if they commute, else `-Q` (`k = 4`)
/// or `∓ i Q P` (`k = 2`, `6`).
pub fn conjugate_by_rotation(q: &Pauli, p: &Pauli, k: u8) -> Pauli {
    if q.commutes(p) || k % 8 == 0 {
        return q.clone();
    }
    match k % 8 {
        4 => {
            let mut r = q.clone();
            r.sign = !r.sign;
            r
        }
        2 | 6 => {
            // i·Q·P is Hermitian for anticommuting Q, P; k = 2 gives -iQP
            let mut r = mul_anticommuting(q, p);
            if k % 8 == 2 {
                r.sign = !r.sign;
            }
            r
        }
        _ => panic!("conjugate_by_rotation needs an even exponent"),
    }
}

/// Merging with Clifford absorption: after [`merge`], every rotation with
/// an even exponent (a Clifford) is moved to the end of the sequence,
/// conjugating the rotations after it, and merging is repeated. Returns
/// the odd-and-even rotations left and the absorbed Clifford rotations in
/// the order they must be applied after them (`U = C · A_r ⋯ A_1 · Π R`,
/// i.e. the returned list is in time order).
pub fn merge_absorb(rots: Vec<Rotation>, global: &mut u8) -> (Vec<Rotation>, Vec<Rotation>) {
    let mut rots = merge(rots);
    let mut absorbed: Vec<Rotation> = Vec::new();
    while let Some(i) = rots.iter().position(|r| r.k % 2 == 0) {
        let r = rots.remove(i);
        for later in rots.iter_mut().skip(i) {
            let mut q = conjugate_by_rotation(&later.axis, &r.axis, r.k);
            if q.sign {
                q.sign = false;
                *global = (*global + later.k) % 8;
                later.k = (8 - later.k) % 8;
            }
            later.axis = q;
        }
        absorbed.push(r);
        rots = merge(rots);
    }
    // Π L = A_1 · Π L' and later absorptions act inside: the time order
    // after the rotations is A_last, ..., A_1.
    absorbed.reverse();
    (rots, absorbed)
}

/// Greedy commuting groups: each rotation joins the earliest group it can
/// be moved to (it commutes with every member of that group and of all
/// later groups), else opens a new group.
pub fn group(rots: &[Rotation]) -> Vec<Vec<usize>> {
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for (j, r) in rots.iter().enumerate() {
        let mut target = None;
        for g in (0..groups.len()).rev() {
            if groups[g].iter().all(|&i| rots[i].axis.commutes(&r.axis)) {
                target = Some(g);
            } else {
                break;
            }
        }
        match target {
            Some(g) => groups[g].push(j),
            None => groups.push(vec![j]),
        }
    }
    groups
}

/// Local search over the group assignment: a rotation may move to any
/// group strictly after every earlier anticommuting rotation's group and
/// strictly before every later one's (so the order of every
/// anticommuting pair is kept); a move is kept when deterministic TODD on
/// the two groups involved lowers their total T-count.
pub fn regroup(
    n: usize,
    rots: &[Rotation],
    groups: &mut Vec<Vec<usize>>,
    passes: usize,
    seconds: f64,
    seed: u64,
) {
    use rand::rngs::StdRng;
    use rand::seq::SliceRandom;
    use rand::SeedableRng;
    use rayon::prelude::*;
    let m = rots.len();
    if m == 0 || passes == 0 {
        return;
    }
    let start = std::time::Instant::now();
    let anti: Vec<Vec<usize>> = (0..m)
        .into_par_iter()
        .map(|j| {
            (0..m)
                .filter(|&l| l != j && !rots[l].axis.commutes(&rots[j].axis))
                .collect()
        })
        .collect();
    let mut gof = vec![0usize; m];
    for (g, ids) in groups.iter().enumerate() {
        for &i in ids {
            gof[i] = g;
        }
    }
    let cost_of = |ids: &[usize]| -> usize {
        if ids.is_empty() {
            return 0;
        }
        let axes: Vec<Pauli> = ids.iter().map(|&i| rots[i].axis.clone()).collect();
        let (_, rows) = diagonalize(&axes);
        let mut cols = Vec::new();
        for (row, &i) in rows.iter().zip(ids) {
            let k = if row.sign { (8 - rots[i].k) % 8 } else { rots[i].k };
            if k % 2 == 1 {
                cols.push(row.z.clone());
            }
        }
        super::tensor::todd(
            cols,
            n,
            &super::tensor::ToddParams::default(),
            &mut StdRng::seed_from_u64(0),
        )
        .len()
    };
    let mut cost: Vec<usize> = groups.par_iter().map(|g| cost_of(g)).collect();
    let mut rng = StdRng::seed_from_u64(seed ^ 0x6A09_E667_F3BC_C908);
    let odd: Vec<usize> = (0..m).filter(|&j| rots[j].k % 2 == 1).collect();
    for _ in 0..passes {
        let mut improved = false;
        let mut order = odd.clone();
        order.shuffle(&mut rng);
        for &j in &order {
            if start.elapsed().as_secs_f64() > seconds {
                break;
            }
            let lo = anti[j]
                .iter()
                .filter(|&&l| l < j)
                .map(|&l| gof[l] as i64)
                .max()
                .unwrap_or(-1);
            let hi = anti[j]
                .iter()
                .filter(|&&l| l > j)
                .map(|&l| gof[l] as i64)
                .min()
                .unwrap_or(groups.len() as i64);
            let from = gof[j];
            let cands: Vec<usize> = ((lo + 1)..hi)
                .map(|g| g as usize)
                .filter(|&g| g != from)
                .collect();
            if cands.is_empty() {
                continue;
            }
            let mut a = groups[from].clone();
            a.retain(|&x| x != j);
            let ca = cost_of(&a);
            let best = cands
                .par_iter()
                .map(|&g| {
                    let mut b = groups[g].clone();
                    b.push(j);
                    b.sort_unstable();
                    let cb = cost_of(&b);
                    (cb as i64 - cost[g] as i64, g, cb)
                })
                .min();
            if let Some((db, g, cb)) = best {
                if db + ca as i64 - (cost[from] as i64) < 0 {
                    groups[from] = a;
                    groups[g].push(j);
                    groups[g].sort_unstable();
                    cost[from] = ca;
                    cost[g] = cb;
                    gof[j] = g;
                    improved = true;
                }
            }
        }
        if !improved || start.elapsed().as_secs_f64() > seconds {
            break;
        }
    }
    // drop empty groups (order of the rest is unchanged)
    groups.retain(|g| !g.is_empty());
}

/// Total deterministic-TODD T-count of a grouping.
pub fn groups_cost(n: usize, rots: &[Rotation], groups: &[Vec<usize>]) -> usize {
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use rayon::prelude::*;
    groups
        .par_iter()
        .map(|ids| {
            let axes: Vec<Pauli> = ids.iter().map(|&i| rots[i].axis.clone()).collect();
            let (_, rows) = diagonalize(&axes);
            let cols: Vec<Bits> = rows
                .iter()
                .zip(ids)
                .filter(|(row, &i)| {
                    let k = if row.sign { (8 - rots[i].k) % 8 } else { rots[i].k };
                    k % 2 == 1
                })
                .map(|(row, _)| row.z.clone())
                .collect();
            super::tensor::todd(
                cols,
                n,
                &super::tensor::ToddParams::default(),
                &mut StdRng::seed_from_u64(0),
            )
            .len()
        })
        .sum()
}

/// What [`optimize_pauli`] did.
#[derive(Clone, Debug, Default)]
pub struct PauliReport {
    /// T-count of the input.
    pub t_input: usize,
    /// Odd rotations after merging.
    pub t_merged: usize,
    /// T-count of the output.
    pub t_output: usize,
    /// Number of commuting groups.
    pub groups: usize,
    /// Global phase (units of π/4): `input = ω^global_phase · output`.
    pub global_phase: u8,
}

fn push_dgate(c: &mut crate::Circuit, g: DGate, inverse: bool) {
    match (g, inverse) {
        (DGate::H(q), _) => {
            c.h(q);
        }
        (DGate::S(q), false) | (DGate::Sdg(q), true) => {
            c.gate(crate::Gate::S(q));
        }
        (DGate::Sdg(q), false) | (DGate::S(q), true) => {
            c.gate(crate::Gate::Sdg(q));
        }
        (DGate::Cnot(a, b), _) => {
            c.cnot(a, b);
        }
    }
}

/// T-count optimisation in the Pauli frame: [`rotations`], [`merge`],
/// [`group`], then per group a diagonalising Clifford, TODD on the
/// diagonal phase polynomial, and re-synthesis
/// `D · (CNOT/T network) · D†`; the circuit's Clifford skeleton follows
/// at the end. The output is exactly equal to the input up to the
/// reported global phase (checked semantically by the callers; the
/// Hadamard structure changes, so the path-sum identity does not apply).
pub fn optimize_pauli(c: &super::PhaseCircuit, opts: &super::ToddOptions) -> (crate::Circuit, PauliReport) {
    use rayon::prelude::*;
    let n = c.num_qubits;
    let (rots, mut global) = rotations(c);
    let (rots, absorbed) = if opts.absorb_cliffords {
        merge_absorb(rots, &mut global)
    } else {
        (merge(rots), Vec::new())
    };
    let mut groups = group(&rots);
    if opts.todd {
        // the same greedy run backwards ("as late as possible"); keep the
        // assignment with the lower deterministic-TODD total
        let rev: Vec<Rotation> = rots.iter().rev().cloned().collect();
        let mut late: Vec<Vec<usize>> = group(&rev)
            .into_iter()
            .rev()
            .map(|g| {
                let mut g: Vec<usize> = g.into_iter().map(|i| rots.len() - 1 - i).collect();
                g.sort_unstable();
                g
            })
            .collect();
        late.retain(|g| !g.is_empty());
        if groups_cost(n, &rots, &late) < groups_cost(n, &rots, &groups) {
            groups = late;
        }
    }
    if opts.todd && opts.reassign_passes > 0 {
        regroup(n, &rots, &mut groups, opts.reassign_passes, opts.reassign_seconds, opts.seed);
    }
    let mut report = PauliReport {
        t_input: c.t_count(),
        t_merged: rots.iter().filter(|r| r.k % 2 == 1).count(),
        groups: groups.len(),
        ..Default::default()
    };
    let blocks: Vec<(crate::Circuit, u8, usize)> = groups
        .par_iter()
        .enumerate()
        .map(|(gi, ids)| {
            let axes: Vec<Pauli> = ids.iter().map(|&i| rots[i].axis.clone()).collect();
            let (dg, rows) = diagonalize(&axes);
            let mut gph = 0u8;
            let mut cols = Vec::new();
            let mut extra: Vec<(Bits, u8)> = Vec::new();
            for (row, &i) in rows.iter().zip(ids) {
                let k0 = rots[i].k;
                let k = if row.sign {
                    gph = (gph + k0) % 8;
                    (8 - k0) % 8
                } else {
                    k0
                };
                if k % 2 == 1 {
                    if k != 1 {
                        extra.push((row.z.clone(), (k + 7) % 8));
                    }
                    cols.push(row.z.clone());
                } else if k != 0 {
                    extra.push((row.z.clone(), k));
                }
            }
            let new = if opts.todd {
                super::best_todd(
                    &cols,
                    n,
                    opts.restarts,
                    opts.lns_rounds,
                    opts.seed ^ (gi as u64).wrapping_mul(0x2545_F491_4F6C_DD1D),
                )
            } else {
                super::tensor::clean(cols.clone())
            };
            let (terms, t_out) = super::group_terms(&cols, new, extra, n);
            let mut blk = crate::Circuit::new(n);
            for &g in &dg {
                push_dgate(&mut blk, g, false);
            }
            for (co, k) in &terms {
                let wires: Vec<usize> = co.ones().collect();
                super::push_parity_phase(&mut blk, &wires, *k);
            }
            for &g in dg.iter().rev() {
                push_dgate(&mut blk, g, true);
            }
            (blk, gph, t_out)
        })
        .collect();
    let mut out = crate::Circuit::new(n);
    for (blk, gph, _) in &blocks {
        out.append(blk);
        global = (global + gph) % 8;
    }
    for a in &absorbed {
        let (dg, rows) = diagonalize(std::slice::from_ref(&a.axis));
        let k = if rows[0].sign {
            global = (global + a.k) % 8;
            (8 - a.k) % 8
        } else {
            a.k
        };
        for &g in &dg {
            push_dgate(&mut out, g, false);
        }
        let wires: Vec<usize> = rows[0].z.ones().collect();
        super::push_parity_phase(&mut out, &wires, k);
        for &g in dg.iter().rev() {
            push_dgate(&mut out, g, true);
        }
    }
    for g in &c.gates {
        match *g {
            PGate::H(q) => {
                out.h(q);
            }
            PGate::X(q) => {
                out.x(q);
            }
            PGate::Cnot(a, b) => {
                out.cnot(a, b);
            }
            PGate::Swap(a, b) => {
                out.swap(a, b);
            }
            PGate::Cz(a, b) => {
                out.cz(a, b);
            }
            PGate::Phase(q, k) => super::push_phase(&mut out, q, k - k % 2),
            PGate::Ccz(..) => {}
        }
    }
    report.t_output = out
        .gates()
        .filter(|g| matches!(g, crate::Gate::T(_) | crate::Gate::Tdg(_)))
        .count();
    report.global_phase = global;
    (out, report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_images_match_textbook_conjugations() {
        let n = 2;
        let mut f = Frame::new(n);
        f.apply(&PGate::H(0));
        assert_eq!(f.zs[0], Pauli::x(n, 0));
        let mut f = Frame::new(n);
        f.apply(&PGate::Cnot(0, 1));
        let z0z1 = Pauli::z(n, 0).mul_commuting(&Pauli::z(n, 1));
        assert_eq!(f.zs[1], z0z1);
        // S then S = Z: S†S† X S S = Z X Z = -X
        let mut f = Frame::new(1);
        f.apply(&PGate::Phase(0, 4));
        let mut minus_x = Pauli::x(1, 0);
        minus_x.sign = true;
        assert_eq!(f.xs[0], minus_x);
        // S† X S = -Y
        let mut f = Frame::new(1);
        f.apply(&PGate::Phase(0, 2));
        assert_eq!(f.xs[0].x, Bits::unit(1, 0));
        assert_eq!(f.xs[0].z, Bits::unit(1, 0));
        assert!(f.xs[0].sign);
    }
}
