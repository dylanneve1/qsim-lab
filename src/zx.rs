use crate::{Circuit, Gate, Op};
use std::collections::{HashMap, HashSet};
use std::f64::consts::{FRAC_PI_2, PI};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ZxStats {
    pub gates_before: usize,
    pub gates_after: usize,
    pub t_before: usize,
    pub t_after: usize,
    pub phase_fusions: usize,
    pub identities_removed: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ZxResult {
    pub circuit: Circuit,
    pub stats: ZxStats,
}

#[derive(Clone, Debug)]
struct Vertex {
    phase: f64,
    is_boundary: bool,
}

#[derive(Clone, Debug, Default)]
struct Graph {
    vertices: Vec<Option<Vertex>>,
    edges: Vec<HashSet<usize>>,
    tracker: Vec<Option<usize>>, // If some, this vertex is a phase gadget pendant for the given circuit gate
    shifts: HashMap<usize, f64>,
}

impl Graph {
    fn new() -> Self {
        Self::default()
    }

    fn add_vertex(&mut self, phase: f64, is_boundary: bool) -> usize {
        let id = self.vertices.len();
        self.vertices.push(Some(Vertex { phase, is_boundary }));
        self.edges.push(HashSet::new());
        self.tracker.push(None);
        id
    }

    fn add_tracker(&mut self, orig_idx: usize) -> usize {
        let id = self.add_vertex(0.0, false);
        self.tracker[id] = Some(orig_idx);
        id
    }

    fn add_edge_reg(&mut self, u: usize, v: usize) {
        self.edges[u].insert(v);
        self.edges[v].insert(u);
    }

    fn toggle_edge(&mut self, u: usize, v: usize) {
        if u == v {
            self.add_phase(u, PI);
        } else {
            if self.edges[u].contains(&v) {
                self.edges[u].remove(&v);
                self.edges[v].remove(&u);
            } else {
                self.edges[u].insert(v);
                self.edges[v].insert(u);
            }
        }
    }

    fn add_phase(&mut self, u: usize, p: f64) {
        if let Some(idx) = self.tracker[u] {
            *self.shifts.entry(idx).or_insert(0.0) += p;
        } else {
            self.vertices[u].as_mut().unwrap().phase += p;
        }
    }

    fn remove_vertex(&mut self, v: usize) {
        let neighbors: Vec<usize> = self.edges[v].iter().copied().collect();
        for n in neighbors {
            self.edges[n].remove(&v);
        }
        self.edges[v].clear();
        self.vertices[v] = None;
    }

    fn has_no_boundary_neighbors(&self, v: usize) -> bool {
        for &n in &self.edges[v] {
            if self.vertices[n].as_ref().unwrap().is_boundary {
                return false;
            }
        }
        true
    }
}

fn apply_h(g: &mut Graph, cur: &mut [usize], q: usize) {
    let new_v = g.add_vertex(0.0, false);
    g.toggle_edge(cur[q], new_v);
    cur[q] = new_v;
}

fn apply_cz(g: &mut Graph, cur: &mut [usize], c: usize, t: usize) {
    g.toggle_edge(cur[c], cur[t]);
}

fn apply_cnot(g: &mut Graph, cur: &mut [usize], c: usize, t: usize) {
    apply_h(g, cur, t);
    apply_cz(g, cur, c, t);
    apply_h(g, cur, t);
}

fn apply_phase_untracked(g: &mut Graph, cur: &mut [usize], q: usize, phase: f64) {
    g.add_phase(cur[q], phase);
}

fn apply_phase_tracked(g: &mut Graph, cur: &mut [usize], q: usize, idx: usize) {
    apply_h(g, cur, q);
    let root = g.add_vertex(0.0, false);
    g.toggle_edge(cur[q], root);
    cur[q] = root;
    let pendant = g.add_tracker(idx);
    g.toggle_edge(root, pendant);
    apply_h(g, cur, q);
}

fn is_multiple_of_pi(p: f64) -> bool {
    let k = p / PI;
    (k - k.round()).abs() < 1e-10
}

fn is_half_pi(p: f64) -> bool {
    let k = p / FRAC_PI_2;
    (k - k.round()).abs() < 1e-10 && (k.round() as i64) % 2 != 0
}

fn normalize_phase(p: f64) -> f64 {
    let mut a = p % (2.0 * PI);
    if a < -PI + 1e-10 {
        a += 2.0 * PI;
    } else if a > PI - 1e-10 {
        a -= 2.0 * PI;
    }
    a
}

fn clifford_simp(g: &mut Graph) {
    loop {
        let mut changed = false;

        // Identity2
        let mut id2_cand = None;
        for i in 0..g.vertices.len() {
            if g.vertices[i].is_none() || g.tracker[i].is_some() || g.vertices[i].as_ref().unwrap().is_boundary {
                continue;
            }
            if g.has_no_boundary_neighbors(i) && is_multiple_of_pi(g.vertices[i].as_ref().unwrap().phase) && g.edges[i].len() == 2 {
                id2_cand = Some(i);
                break;
            }
        }
        if let Some(v) = id2_cand {
            let p = g.vertices[v].as_ref().unwrap().phase;
            let ns: Vec<usize> = g.edges[v].iter().copied().collect();
            let (n1, n2) = (ns[0], ns[1]);
            g.remove_vertex(v);
            g.toggle_edge(n1, n2);
            g.add_phase(n1, p); // Phase is added to one of the neighbors
            changed = true;
            continue;
        }

        // LC
        let mut lc_cand = None;
        for i in 0..g.vertices.len() {
            if g.vertices[i].is_none() || g.tracker[i].is_some() || g.vertices[i].as_ref().unwrap().is_boundary {
                continue;
            }
            if g.has_no_boundary_neighbors(i) && is_half_pi(g.vertices[i].as_ref().unwrap().phase) {
                lc_cand = Some(i);
                break;
            }
        }
        if let Some(v) = lc_cand {
            let p = g.vertices[v].as_ref().unwrap().phase;
            let ns: Vec<usize> = g.edges[v].iter().copied().collect();
            for i in 0..ns.len() {
                for j in (i + 1)..ns.len() {
                    g.toggle_edge(ns[i], ns[j]);
                }
            }
            for &n in &ns {
                let sign = if p > 0.0 { -1.0 } else { 1.0 };
                g.add_phase(n, sign * FRAC_PI_2);
            }
            g.remove_vertex(v);
            changed = true;
            continue;
        }

        // Pivot
        let mut pivot_cand = None;
        'outer: for u in 0..g.vertices.len() {
            if g.vertices[u].is_none() || g.tracker[u].is_some() || g.vertices[u].as_ref().unwrap().is_boundary {
                continue;
            }
            if !g.has_no_boundary_neighbors(u) || !is_multiple_of_pi(g.vertices[u].as_ref().unwrap().phase) {
                continue;
            }
            for &v in &g.edges[u] {
                if g.tracker[v].is_some() || g.vertices[v].as_ref().unwrap().is_boundary {
                    continue;
                }
                if g.has_no_boundary_neighbors(v) && is_multiple_of_pi(g.vertices[v].as_ref().unwrap().phase) {
                    pivot_cand = Some((u, v));
                    break 'outer;
                }
            }
        }
        if let Some((u, v)) = pivot_cand {
            let pu = g.vertices[u].as_ref().unwrap().phase;
            let pv = g.vertices[v].as_ref().unwrap().phase;
            let nu: Vec<usize> = g.edges[u].iter().copied().filter(|&n| n != v).collect();
            let nv: Vec<usize> = g.edges[v].iter().copied().filter(|&n| n != u).collect();
            for &n_u in &nu {
                for &n_v in &nv {
                    g.toggle_edge(n_u, n_v);
                }
            }
            for &n_u in &nu {
                g.add_phase(n_u, pv);
            }
            for &n_v in &nv {
                g.add_phase(n_v, pu);
            }
            g.remove_vertex(u);
            g.remove_vertex(v);
            changed = true;
            continue;
        }

        // Phase Gadget Fusion
        let mut gadgets: HashMap<Vec<usize>, usize> = HashMap::new();
        let mut to_fuse = None;
        for i in 0..g.vertices.len() {
            if g.vertices[i].is_none() || g.tracker[i].is_some() || g.vertices[i].as_ref().unwrap().is_boundary {
                continue;
            }
            
            // Push pi to tracker if needed
            let p = normalize_phase(g.vertices[i].as_ref().unwrap().phase);
            if p.abs() > 1e-10 {
                if (p.abs() - PI).abs() < 1e-10 {
                    let mut moved = false;
                    let ns: Vec<usize> = g.edges[i].iter().copied().collect();
                    for &n in &ns {
                        if g.tracker[n].is_some() {
                            g.add_phase(n, PI);
                            moved = true;
                        }
                    }
                    if moved {
                        g.vertices[i].as_mut().unwrap().phase = 0.0;
                        changed = true;
                    }
                }
                // Only phase 0 roots can be used for fusion
                continue;
            }

            let mut non_trackers = Vec::new();
            let mut trackers = Vec::new();
            for &n in &g.edges[i] {
                if g.tracker[n].is_some() {
                    trackers.push(n);
                } else {
                    non_trackers.push(n);
                }
            }
            if trackers.is_empty() {
                continue;
            }
            non_trackers.sort_unstable();

            if let Some(&existing_root) = gadgets.get(&non_trackers) {
                to_fuse = Some((i, existing_root, trackers));
                break;
            } else {
                gadgets.insert(non_trackers, i);
            }
        }
        if let Some((i, existing_root, trackers)) = to_fuse {
            for t in trackers {
                g.edges[i].remove(&t);
                g.edges[t].remove(&i);
                g.edges[existing_root].insert(t);
                g.edges[t].insert(existing_root);
            }
            g.remove_vertex(i);
            changed = true;
            continue;
        }

        if !changed {
            break;
        }
    }
}

pub trait ZxSimplify {
    fn zx_simplify(&self) -> ZxResult;
}

impl ZxSimplify for Circuit {
    fn zx_simplify(&self) -> ZxResult {
        simplify(self)
    }
}

pub fn simplify(circuit: &Circuit) -> ZxResult {
    let mut stats = ZxStats {
        gates_before: circuit.num_gates(),
        t_before: circuit.t_count(),
        ..ZxStats::default()
    };
    
    // Pre-decompose all compound gates (CCX, CPhase, Rx, Ry) into
    // Clifford+T primitives so every T/Tdg becomes individually trackable.
    let mut flat_ops: Vec<Op> = Vec::new();
    for op in &circuit.ops {
        match op {
            Op::Gate(g) => {
                let decomposed = g.decompose_to_clifford_rz();
                if decomposed.len() == 1 && decomposed[0] == *g {
                    flat_ops.push(Op::Gate(*g));
                } else {
                    for dg in decomposed {
                        flat_ops.push(Op::Gate(dg));
                    }
                }
            }
            Op::Measure(q) => flat_ops.push(Op::Measure(*q)),
        }
    }

    let mut out_circuit = Circuit::new(circuit.num_qubits);
    let mut current_segment: Vec<(usize, Gate)> = Vec::new();

    for (idx, op) in flat_ops.iter().enumerate() {
        match op {
            Op::Gate(g) => {
                current_segment.push((idx, *g));
            }
            Op::Measure(q) => {
                simplify_segment(&mut out_circuit, &current_segment, &mut stats);
                current_segment.clear();
                out_circuit.measure(*q);
            }
        }
    }
    simplify_segment(&mut out_circuit, &current_segment, &mut stats);

    stats.gates_after = out_circuit.num_gates();
    stats.t_after = out_circuit.t_count();
    ZxResult { circuit: out_circuit, stats }
}

fn phase_angle(g: &Gate) -> Option<f64> {
    use Gate::*;
    match g {
        Z(_) => Some(PI),
        S(_) => Some(FRAC_PI_2),
        Sdg(_) => Some(-FRAC_PI_2),
        T(_) => Some(PI / 4.0),
        Tdg(_) => Some(-PI / 4.0),
        Phase(_, p) | Rz(_, p) => Some(*p),
        _ => None,
    }
}

fn is_tracked(g: &Gate) -> bool {
    match g {
        Gate::T(_) | Gate::Tdg(_) => true,
        Gate::Phase(_, p) | Gate::Rz(_, p) => {
            // Only track non-Clifford angles (not multiples of π/2)
            let k = *p / FRAC_PI_2;
            (k - k.round()).abs() > 1e-10
        }
        _ => false,
    }
}

fn canonical_angle(q: usize, mut angle: f64) -> Vec<Gate> {
    angle = angle.rem_euclid(2.0 * PI);
    if angle > 2.0 * PI - 1e-10 {
        angle = 0.0;
    }
    let k = (angle / (PI / 4.0)).round();
    if (angle - k * (PI / 4.0)).abs() > 1e-10 {
        return vec![Gate::Phase(q, angle)];
    }
    match (k as i64).rem_euclid(8) {
        0 => vec![],
        1 => vec![Gate::T(q)],
        2 => vec![Gate::S(q)],
        3 => vec![Gate::S(q), Gate::T(q)],
        4 => vec![Gate::Z(q)],
        5 => vec![Gate::Z(q), Gate::T(q)],
        6 => vec![Gate::Z(q), Gate::S(q)],
        7 => vec![Gate::Z(q), Gate::Tdg(q)],
        _ => unreachable!(),
    }
}

fn simplify_segment(out_circuit: &mut Circuit, segment: &[(usize, Gate)], stats: &mut ZxStats) {
    if segment.is_empty() {
        return;
    }
    let mut g = Graph::new();
    let num_qubits = out_circuit.num_qubits;
    let mut cur = vec![0; num_qubits];
    
    for q in 0..num_qubits {
        let in_v = g.add_vertex(0.0, true);
        let sp_v = g.add_vertex(0.0, false);
        g.add_edge_reg(in_v, sp_v);
        cur[q] = sp_v;
    }

    let mut orig_phases = HashMap::new();
    let mut active_trackers = Vec::new();

    for (i, &(orig_idx, gate)) in segment.iter().enumerate() {
        if is_tracked(&gate) {
            let p = phase_angle(&gate).unwrap();
            orig_phases.insert(i, (gate.qubits()[0], p));
            apply_phase_tracked(&mut g, &mut cur, gate.qubits()[0], i);
            active_trackers.push(i);
        } else {
            for dg in gate.decompose_to_clifford_rz() {
                if let Some(p) = phase_angle(&dg) {
                    apply_phase_untracked(&mut g, &mut cur, dg.qubits()[0], p);
                } else {
                    match dg {
                        Gate::H(q) => apply_h(&mut g, &mut cur, q),
                        Gate::Cz(c, t) => apply_cz(&mut g, &mut cur, c, t),
                        Gate::Cnot(c, t) => apply_cnot(&mut g, &mut cur, c, t),
                        Gate::Swap(a, b) => {
                            apply_cnot(&mut g, &mut cur, a, b);
                            apply_cnot(&mut g, &mut cur, b, a);
                            apply_cnot(&mut g, &mut cur, a, b);
                        }
                        Gate::X(q) => {
                            apply_h(&mut g, &mut cur, q);
                            apply_phase_untracked(&mut g, &mut cur, q, PI);
                            apply_h(&mut g, &mut cur, q);
                        }
                        Gate::Y(q) => {
                            apply_h(&mut g, &mut cur, q);
                            apply_phase_untracked(&mut g, &mut cur, q, PI);
                            apply_h(&mut g, &mut cur, q);
                            apply_phase_untracked(&mut g, &mut cur, q, PI);
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    for q in 0..num_qubits {
        let out_v = g.add_vertex(0.0, true);
        g.add_edge_reg(cur[q], out_v);
    }

    clifford_simp(&mut g);

    let mut tracker_roots: HashMap<usize, Vec<usize>> = HashMap::new();
    for v in 0..g.vertices.len() {
        if let Some(idx) = g.tracker[v] {
            if g.vertices[v].is_some() {
                let ns: Vec<usize> = g.edges[v].iter().copied().collect();
                if ns.len() == 1 {
                    tracker_roots.entry(ns[0]).or_default().push(idx);
                } else {
                    // Pendants should have degree 1 because we never pivot on them.
                    // If they have degree > 1, we just don't fuse them.
                    tracker_roots.entry(v).or_default().push(idx); // Just put it somewhere unique
                }
            }
        }
    }

    let mut new_angles = HashMap::new();
    for (_, trackers) in tracker_roots {
        if trackers.is_empty() {
            continue;
        }
        // Sum the effective phase of each tracker: original_phase + shift
        let mut total_angle = 0.0;
        for &t in &trackers {
            let shift = g.shifts.get(&t).copied().unwrap_or(0.0);
            total_angle += orig_phases[&t].1 + shift;
        }
        // Assign the combined angle to the first tracker, zero the rest
        let root_idx = trackers[0];
        new_angles.insert(root_idx, total_angle);
        for &t in &trackers[1..] {
            new_angles.insert(t, 0.0);
            stats.phase_fusions += 1;
        }
    }

    for (i, &(_orig_idx, gate)) in segment.iter().enumerate() {
        if is_tracked(&gate) {
            let q = orig_phases[&i].0;
            let angle = new_angles.get(&i).copied().unwrap_or(orig_phases[&i].1);
            for g in canonical_angle(q, angle) {
                out_circuit.gate(g);
            }
        } else {
            out_circuit.gate(gate);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::StateVectorF64;
    use rand::{rngs::StdRng, SeedableRng};

    fn fidelity(a: &Circuit, b: &Circuit) -> f64 {
        let mut x = StateVectorF64::new(a.num_qubits);
        let mut y = StateVectorF64::new(b.num_qubits);
        for g in a.gates() {
            x.apply_gate(g).unwrap();
        }
        for g in b.gates() {
            y.apply_gate(g).unwrap();
        }
        x.fidelity(&y)
    }

    #[test]
    fn fuses_phases_across_cnot_controls_and_swap() {
        let mut c = Circuit::new(3);
        c.t(0).cnot(0, 1).tdg(0).t(2).swap(1, 2).tdg(1);
        let r = simplify(&c);
        assert_eq!(r.stats.t_before, 4);
        assert_eq!(r.stats.t_after, 0);
        assert!(fidelity(&c, &r.circuit) > 1.0 - 1e-12);
    }

    #[test]
    fn random_circuits_are_exact_up_to_global_phase() {
        let mut rng = StdRng::seed_from_u64(9);
        for n in 1..=12 {
            for _ in 0..4 {
                let c = Circuit::random_clifford_t(n, 12, 0.35, &mut rng);
                let r = simplify(&c);
                assert!(fidelity(&c, &r.circuit) > 1.0 - 1e-12, "n={n}");
            }
        }
    }
}
