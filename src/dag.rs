//! A dependency-graph (DAG) intermediate representation of a [`Circuit`].
//!
//! Every operation of the circuit becomes a *node*. Nodes are joined by
//! three kinds of edges:
//!
//! * **wire edges**: for every qubit, the previous op on that qubit points
//!   to the next one. A node has one incoming and one outgoing wire slot per
//!   qubit it touches (`prev`/`next`), so walking a qubit's history is O(1)
//!   per step.
//! * **classical edges**: a `Measure` node points to every
//!   `ClassicControlled` node that reads its outcome.
//! * **record edges**: the measurements form a chain in outcome-record
//!   order. `ClassicControlled` ops address outcomes by their position in
//!   the record, so keeping the measurements in their original relative
//!   order keeps every `meas_index` meaningful without renumbering. Record
//!   edges are bookkeeping: layering and light cones ignore them.
//!
//! Any topological order of the DAG (all three edge kinds) is a valid
//! circuit with exactly the same semantics as the original: two ops that
//! are not ordered by an edge act on disjoint qubits and are not linked
//! classically, so they commute as channels, branch by branch.
//! [`Dag::to_circuit`] emits the order that is closest to the original
//! program order, so `Dag::from_circuit(c).to_circuit() == c` exactly.
//!
//! On top of the graph this module offers:
//!
//! * queries: per-wire neighbours, front layer, ASAP/ALAP layers, reverse
//!   reachability ([`Dag::ancestors`]), qubit components, and an exact
//!   commutation test for two ops ([`ops_commute`]);
//! * rewrites: [`Dag::remove`], [`Dag::replace`], [`Dag::merge`] and
//!   [`Dag::slide`], each of which keeps the graph consistent;
//! * passes: [`light_cone`], [`light_cone_marginal`], [`components`] and
//!   a commutation-aware peephole optimiser ([`peephole`]) that walks wires
//!   directly instead of re-deriving dependencies from a flat op list.

use crate::circuit::{Circuit, Op, SimError};
use crate::gate::Gate;
use num_complex::Complex64;
use std::cell::RefCell;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI, TAU};
use std::fmt;
use std::hash::{BuildHasherDefault, Hasher};

/// Index of a node. Ids are stable: removing a node never renumbers the
/// others, and new nodes get fresh ids.
pub type NodeId = u32;

/// "No node" marker in wire, classical and record links.
pub const NONE: NodeId = u32::MAX;

/// Errors from building or rewriting a [`Dag`].
#[derive(Clone, Debug, PartialEq)]
pub enum DagError {
    /// The circuit would also be rejected by [`Circuit::run`].
    Sim(SimError),
    /// A two-qubit noise channel names the same qubit twice.
    RepeatedNoiseQubit(usize),
    /// The node id does not refer to a live node.
    DeadNode(NodeId),
    /// A measurement still has live classically controlled readers.
    MeasurementHasReaders(NodeId),
    /// The two nodes are not adjacent on every wire they share.
    NotAdjacent(NodeId, NodeId),
    /// The two nodes do not commute, so one cannot slide past the other.
    DoNotCommute(NodeId, NodeId),
    /// A rewrite may only use unitary gates on the qubits of the node it
    /// replaces (the replacement must not change classical structure).
    BadReplacement(&'static str),
}

impl fmt::Display for DagError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DagError::Sim(e) => write!(f, "{e}"),
            DagError::RepeatedNoiseQubit(q) => {
                write!(f, "two-qubit noise channel uses qubit {q} twice")
            }
            DagError::DeadNode(n) => write!(f, "node {n} is not live"),
            DagError::MeasurementHasReaders(n) => {
                write!(f, "measurement node {n} still has classical readers")
            }
            DagError::NotAdjacent(a, b) => {
                write!(
                    f,
                    "nodes {a} and {b} are not adjacent on their shared wires"
                )
            }
            DagError::DoNotCommute(a, b) => write!(f, "nodes {a} and {b} do not commute"),
            DagError::BadReplacement(why) => write!(f, "invalid replacement: {why}"),
        }
    }
}

impl std::error::Error for DagError {}

impl From<SimError> for DagError {
    fn from(e: SimError) -> Self {
        DagError::Sim(e)
    }
}

// ---------------------------------------------------------------------------
// Op helpers
// ---------------------------------------------------------------------------

/// The qubits of a gate without allocating: `(qubits, count)`.
#[inline]
pub fn gate_qubits(g: &Gate) -> ([usize; 3], usize) {
    use Gate::*;
    match *g {
        I(q) | H(q) | X(q) | Y(q) | Z(q) | S(q) | Sdg(q) | T(q) | Tdg(q) | Sx(q) | Sxdg(q) => {
            ([q, 0, 0], 1)
        }
        Rx(q, _) | Ry(q, _) | Rz(q, _) | Phase(q, _) | U(q, ..) => ([q, 0, 0], 1),
        Cnot(a, b) | Cz(a, b) | Swap(a, b) | ISwap(a, b) | ISwapdg(a, b) | CPhase(a, b, _) => {
            ([a, b, 0], 2)
        }
        Ccx(a, b, t) => ([a, b, t], 3),
    }
}

/// The qubits an op touches: `(qubits, count)`. A classically controlled
/// gate touches its gate's qubits; the bit it reads is a classical edge.
#[inline]
pub fn op_qubits(op: &Op) -> ([usize; 3], usize) {
    match *op {
        Op::Gate(ref g) | Op::ClassicControlled { gate: ref g, .. } => gate_qubits(g),
        Op::Measure(q)
        | Op::Reset(q)
        | Op::XFlip(q, _)
        | Op::YFlip(q, _)
        | Op::ZFlip(q, _)
        | Op::Depolarize1q(q, _) => ([q, 0, 0], 1),
        Op::Depolarize2q(a, b, _) => ([a, b, 0], 2),
    }
}

// ---------------------------------------------------------------------------
// The graph
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct Node {
    op: Op,
    /// Qubits in the op's argument order; slot `i` of `prev`/`next` belongs
    /// to `qs[i]`.
    qs: [u32; 3],
    nq: u8,
    alive: bool,
    prev: [NodeId; 3],
    next: [NodeId; 3],
    /// For a `ClassicControlled` node: the `Measure` node it reads.
    cpred: NodeId,
    /// For a `Measure` node: the live `ClassicControlled` nodes reading it.
    readers: Vec<NodeId>,
    /// Record chain (measurements only).
    rprev: NodeId,
    rnext: NodeId,
    /// Tie-break for [`Dag::to_circuit`]: program position of the op this
    /// node came from (shifted left 20 bits) plus a sub-position for
    /// replacement sequences.
    key: u64,
}

impl Node {
    #[inline]
    fn slot(&self, q: usize) -> Option<usize> {
        (0..self.nq as usize).find(|&i| self.qs[i] as usize == q)
    }
    #[inline]
    fn qubits(&self) -> impl Iterator<Item = usize> + '_ {
        self.qs[..self.nq as usize].iter().map(|&q| q as usize)
    }
}

/// The dependency graph of a circuit. See the [module docs](self).
#[derive(Clone, Debug)]
pub struct Dag {
    num_qubits: usize,
    nodes: Vec<Node>,
    /// First / last live node on each qubit wire.
    head: Vec<NodeId>,
    tail: Vec<NodeId>,
    /// First / last measurement in record order.
    rhead: NodeId,
    rtail: NodeId,
    live: usize,
    /// Program position for the next appended op (key tie-break).
    next_pos: u64,
    /// Every edge goes from a smaller key to a larger one, so sorting live
    /// nodes by key is a topological order (and is exactly the order the
    /// min-key Kahn sort would produce).
    keys_topological: bool,
    /// Node ids increase with keys (no out-of-order insertions), so the
    /// key order is just id order.
    ids_in_key_order: bool,
}

const KEY_SHIFT: u32 = 20;

impl Dag {
    /// An empty DAG on `num_qubits` wires.
    pub fn new(num_qubits: usize) -> Self {
        Dag {
            num_qubits,
            nodes: Vec::new(),
            head: vec![NONE; num_qubits],
            tail: vec![NONE; num_qubits],
            rhead: NONE,
            rtail: NONE,
            live: 0,
            next_pos: 0,
            keys_topological: true,
            ids_in_key_order: true,
        }
    }

    /// Builds the DAG of a circuit. Fails on anything [`Circuit::run`] would
    /// reject at run time: qubits out of range, a qubit used twice by one op,
    /// or a classically controlled op reading a measurement that has not
    /// happened yet.
    pub fn from_circuit(c: &Circuit) -> Result<Dag, DagError> {
        let mut d = Dag::new(c.num_qubits);
        d.nodes.reserve(c.ops.len());
        // Measurement node of every record position so far.
        let mut meas_nodes: Vec<NodeId> = Vec::new();
        for (pos, op) in c.ops.iter().enumerate() {
            let cpred = match *op {
                Op::ClassicControlled { meas_index, .. } => {
                    if meas_index >= meas_nodes.len() {
                        return Err(SimError::ClassicalBitOutOfRange {
                            bit: meas_index,
                            available: meas_nodes.len(),
                        }
                        .into());
                    }
                    meas_nodes[meas_index]
                }
                _ => NONE,
            };
            let id = d.push_back(*op, (pos as u64) << KEY_SHIFT, cpred)?;
            if matches!(op, Op::Measure(_)) {
                meas_nodes.push(id);
            }
        }
        Ok(d)
    }

    /// Appends an op at the end of every wire it touches. `cpred` is the
    /// measurement node a `ClassicControlled` op reads (ignored otherwise).
    fn push_back(&mut self, op: Op, key: u64, cpred: NodeId) -> Result<NodeId, DagError> {
        let (qs, k) = op_qubits(&op);
        for i in 0..k {
            if qs[i] >= self.num_qubits {
                return Err(SimError::QubitOutOfRange {
                    qubit: qs[i],
                    num_qubits: self.num_qubits,
                }
                .into());
            }
            if qs[..i].contains(&qs[i]) {
                return Err(match op {
                    Op::Gate(g) | Op::ClassicControlled { gate: g, .. } => {
                        DagError::Sim(SimError::RepeatedQubit(g))
                    }
                    _ => DagError::RepeatedNoiseQubit(qs[i]),
                });
            }
        }
        let id = self.nodes.len() as NodeId;
        assert!(id != NONE, "too many nodes");
        let mut node = Node {
            op,
            qs: [qs[0] as u32, qs[1] as u32, qs[2] as u32],
            nq: k as u8,
            alive: true,
            prev: [NONE; 3],
            next: [NONE; 3],
            cpred: NONE,
            readers: Vec::new(),
            rprev: NONE,
            rnext: NONE,
            key,
        };
        for (i, &q) in qs[..k].iter().enumerate() {
            let t = self.tail[q];
            node.prev[i] = t;
            if t == NONE {
                self.head[q] = id;
            } else {
                let tn = &mut self.nodes[t as usize];
                let s = tn.slot(q).expect("wire link");
                tn.next[s] = id;
            }
            self.tail[q] = id;
        }
        match op {
            Op::ClassicControlled { .. } => {
                node.cpred = cpred;
                self.nodes[cpred as usize].readers.push(id);
            }
            Op::Measure(_) => {
                node.rprev = self.rtail;
                if self.rtail == NONE {
                    self.rhead = id;
                } else {
                    self.nodes[self.rtail as usize].rnext = id;
                }
                self.rtail = id;
            }
            _ => {}
        }
        if let Some(last) = self.nodes.last() {
            if last.key >= key {
                self.ids_in_key_order = false;
                self.keys_topological = false;
            }
        }
        self.nodes.push(node);
        self.live += 1;
        self.next_pos = self.next_pos.max((key >> KEY_SHIFT) + 1);
        Ok(id)
    }

    /// Appends an op to the end of the circuit. A `ClassicControlled` op's
    /// `meas_index` is interpreted against the current record.
    pub fn push(&mut self, op: Op) -> Result<NodeId, DagError> {
        let cpred = match op {
            Op::ClassicControlled { meas_index, .. } => {
                let ms = self.measurements();
                if meas_index >= ms.len() {
                    return Err(SimError::ClassicalBitOutOfRange {
                        bit: meas_index,
                        available: ms.len(),
                    }
                    .into());
                }
                ms[meas_index]
            }
            _ => NONE,
        };
        self.push_back(op, self.next_pos << KEY_SHIFT, cpred)
    }

    /// Number of qubit wires.
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Number of live nodes.
    pub fn len(&self) -> usize {
        self.live
    }

    pub fn is_empty(&self) -> bool {
        self.live == 0
    }

    /// Upper bound (exclusive) on node ids, for sizing per-node tables.
    pub fn capacity(&self) -> usize {
        self.nodes.len()
    }

    /// True if `id` is a live node.
    pub fn is_live(&self, id: NodeId) -> bool {
        (id as usize) < self.nodes.len() && self.nodes[id as usize].alive
    }

    /// Ids of all live nodes, ascending.
    pub fn node_ids(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| n.alive)
            .map(|(i, _)| i as NodeId)
    }

    #[inline]
    fn n(&self, id: NodeId) -> &Node {
        &self.nodes[id as usize]
    }

    fn check_live(&self, id: NodeId) -> Result<(), DagError> {
        if self.is_live(id) {
            Ok(())
        } else {
            Err(DagError::DeadNode(id))
        }
    }

    /// The op of a node. For a `ClassicControlled` node `meas_index` is the
    /// value from construction; use [`Dag::to_circuit`] for current indices.
    pub fn op(&self, id: NodeId) -> &Op {
        &self.n(id).op
    }

    /// The qubits a node touches, in op argument order.
    pub fn qubits(&self, id: NodeId) -> &[u32] {
        let n = self.n(id);
        &n.qs[..n.nq as usize]
    }

    /// First live node on wire `q` (`None` for an idle wire).
    pub fn first_on(&self, q: usize) -> Option<NodeId> {
        Some(self.head[q]).filter(|&x| x != NONE)
    }

    /// Last live node on wire `q`.
    pub fn last_on(&self, q: usize) -> Option<NodeId> {
        Some(self.tail[q]).filter(|&x| x != NONE)
    }

    /// The previous node on wire `q` (`id` must touch `q`).
    pub fn wire_prev(&self, id: NodeId, q: usize) -> Option<NodeId> {
        let n = self.n(id);
        let s = n.slot(q).expect("node does not touch this wire");
        Some(n.prev[s]).filter(|&x| x != NONE)
    }

    /// The next node on wire `q` (`id` must touch `q`).
    pub fn wire_next(&self, id: NodeId, q: usize) -> Option<NodeId> {
        let n = self.n(id);
        let s = n.slot(q).expect("node does not touch this wire");
        Some(n.next[s]).filter(|&x| x != NONE)
    }

    /// For a `ClassicControlled` node, the `Measure` node it reads.
    pub fn classical_source(&self, id: NodeId) -> Option<NodeId> {
        Some(self.n(id).cpred).filter(|&x| x != NONE)
    }

    /// For a `Measure` node, the `ClassicControlled` nodes reading it.
    pub fn classical_readers(&self, id: NodeId) -> &[NodeId] {
        &self.n(id).readers
    }

    /// Distinct predecessors over wire and classical edges (not record
    /// edges), in slot order.
    pub fn predecessors(&self, id: NodeId) -> Vec<NodeId> {
        let n = self.n(id);
        let mut out: Vec<NodeId> = Vec::with_capacity(4);
        for &p in &n.prev[..n.nq as usize] {
            if p != NONE && !out.contains(&p) {
                out.push(p);
            }
        }
        if n.cpred != NONE && !out.contains(&n.cpred) {
            out.push(n.cpred);
        }
        out
    }

    /// Distinct successors over wire and classical edges (not record edges).
    pub fn successors(&self, id: NodeId) -> Vec<NodeId> {
        let n = self.n(id);
        let mut out: Vec<NodeId> = Vec::with_capacity(4);
        for &s in n.next[..n.nq as usize].iter().chain(&n.readers) {
            if s != NONE && !out.contains(&s) {
                out.push(s);
            }
        }
        out
    }

    /// Live measurement nodes in record order.
    pub fn measurements(&self) -> Vec<NodeId> {
        let mut out = Vec::new();
        let mut m = self.rhead;
        while m != NONE {
            out.push(m);
            m = self.n(m).rnext;
        }
        out
    }

    /// Nodes with no wire or classical predecessor: the ops that can run
    /// first. Ascending by id.
    pub fn front_layer(&self) -> Vec<NodeId> {
        self.node_ids()
            .filter(|&id| {
                let n = self.n(id);
                n.cpred == NONE && n.prev[..n.nq as usize].iter().all(|&p| p == NONE)
            })
            .collect()
    }

    // -----------------------------------------------------------------------
    // Ordering
    // -----------------------------------------------------------------------

    /// A topological order over all three edge kinds, preferring the lowest
    /// key (original program position) among ready nodes. On a freshly
    /// built DAG this is exactly program order.
    pub fn topo_order(&self) -> Vec<NodeId> {
        if self.keys_topological {
            let mut ids: Vec<NodeId> = self.node_ids().collect();
            if !self.ids_in_key_order {
                ids.sort_unstable_by_key(|&id| (self.n(id).key, id));
            }
            return ids;
        }
        self.topo_order_with(|id| self.n(id).key)
    }

    /// A topological order over all three edge kinds, popping the ready
    /// node with the smallest `priority` (ties by id). Every priority gives
    /// a valid circuit with the same semantics; tests use random ones.
    pub fn topo_order_with(&self, priority: impl Fn(NodeId) -> u64) -> Vec<NodeId> {
        let cap = self.nodes.len();
        let mut indeg = vec![0u32; cap];
        let mut heap: BinaryHeap<Reverse<(u64, NodeId)>> = BinaryHeap::new();
        for id in self.node_ids() {
            let n = self.n(id);
            let mut d = n.prev[..n.nq as usize]
                .iter()
                .filter(|&&p| p != NONE)
                .count() as u32;
            d += (n.cpred != NONE) as u32 + (n.rprev != NONE) as u32;
            indeg[id as usize] = d;
            if d == 0 {
                heap.push(Reverse((priority(id), id)));
            }
        }
        let mut out = Vec::with_capacity(self.live);
        while let Some(Reverse((_, id))) = heap.pop() {
            out.push(id);
            let n = self.n(id);
            let succ = n.next[..n.nq as usize]
                .iter()
                .chain(&n.readers)
                .chain(std::iter::once(&n.rnext));
            for &s in succ {
                if s == NONE {
                    continue;
                }
                let d = &mut indeg[s as usize];
                *d -= 1;
                if *d == 0 {
                    heap.push(Reverse((priority(s), s)));
                }
            }
        }
        debug_assert_eq!(out.len(), self.live, "DAG has a cycle");
        out
    }

    /// The circuit in [`Dag::topo_order`]. Measurement record positions are
    /// recomputed, so `meas_index` stays correct after measurements are
    /// removed.
    pub fn to_circuit(&self) -> Circuit {
        self.circuit_in_order(&self.topo_order())
    }

    /// Emits the given nodes (a topological order of the live nodes, or of
    /// an ancestor-closed subset) as a circuit, renumbering measurement
    /// indices by their position among the emitted measurements.
    pub fn circuit_in_order(&self, order: &[NodeId]) -> Circuit {
        let mut bit = vec![usize::MAX; self.nodes.len()];
        let mut nbits = 0usize;
        let mut ops = Vec::with_capacity(order.len());
        for &id in order {
            let n = self.n(id);
            let op = match n.op {
                Op::Measure(_) => {
                    bit[id as usize] = nbits;
                    nbits += 1;
                    n.op
                }
                Op::ClassicControlled {
                    gate, target_value, ..
                } => {
                    let b = bit[n.cpred as usize];
                    assert!(b != usize::MAX, "classical source emitted after reader");
                    Op::ClassicControlled {
                        gate,
                        meas_index: b,
                        target_value,
                    }
                }
                op => op,
            };
            ops.push(op);
        }
        Circuit {
            num_qubits: self.num_qubits,
            ops,
        }
    }

    /// ASAP layers over wire and classical edges: layer 0 is the front
    /// layer and every node sits one layer after its latest predecessor.
    /// Record edges are ignored, so executing layer by layer may measure in
    /// a different order than the record; [`Dag::record_position`] gives
    /// each measurement's bit.
    pub fn asap_layers(&self) -> Vec<Vec<NodeId>> {
        let order = self.topo_order();
        let mut layer = vec![0usize; self.nodes.len()];
        let mut depth = 0;
        for &id in &order {
            let l = self
                .predecessors(id)
                .iter()
                .map(|&p| layer[p as usize] + 1)
                .max()
                .unwrap_or(0);
            layer[id as usize] = l;
            depth = depth.max(l + 1);
        }
        let mut out = vec![Vec::new(); depth];
        for &id in &order {
            out[layer[id as usize]].push(id);
        }
        out
    }

    /// ALAP layers: every node as late as possible with the same depth as
    /// [`Dag::asap_layers`].
    pub fn alap_layers(&self) -> Vec<Vec<NodeId>> {
        let order = self.topo_order();
        let mut height = vec![0usize; self.nodes.len()];
        let mut depth = 0;
        for &id in order.iter().rev() {
            let h = self
                .successors(id)
                .iter()
                .map(|&s| height[s as usize] + 1)
                .max()
                .unwrap_or(0);
            height[id as usize] = h;
            depth = depth.max(h + 1);
        }
        let mut out = vec![Vec::new(); depth];
        for &id in &order {
            out[depth - 1 - height[id as usize]].push(id);
        }
        out
    }

    /// Position of a measurement node in the outcome record.
    pub fn record_position(&self, id: NodeId) -> Option<usize> {
        if !matches!(self.n(id).op, Op::Measure(_)) {
            return None;
        }
        let mut k = 0;
        let mut m = self.n(id).rprev;
        while m != NONE {
            k += 1;
            m = self.n(m).rprev;
        }
        Some(k)
    }

    // -----------------------------------------------------------------------
    // Reachability and components
    // -----------------------------------------------------------------------

    /// Reverse reachability: `seeds` and every node they depend on through
    /// wire edges and classical edges (record edges are not followed).
    /// Returns a per-node membership table indexed by id.
    pub fn ancestors(&self, seeds: &[NodeId]) -> Vec<bool> {
        let mut seen = vec![false; self.nodes.len()];
        let mut stack: Vec<NodeId> = Vec::new();
        for &s in seeds {
            if self.is_live(s) && !seen[s as usize] {
                seen[s as usize] = true;
                stack.push(s);
            }
        }
        while let Some(id) = stack.pop() {
            let n = self.n(id);
            for &p in n.prev[..n.nq as usize]
                .iter()
                .chain(std::iter::once(&n.cpred))
            {
                if p != NONE && !seen[p as usize] {
                    seen[p as usize] = true;
                    stack.push(p);
                }
            }
        }
        seen
    }

    /// Forward reachability through wire and classical edges.
    pub fn descendants(&self, seeds: &[NodeId]) -> Vec<bool> {
        let mut seen = vec![false; self.nodes.len()];
        let mut stack: Vec<NodeId> = Vec::new();
        for &s in seeds {
            if self.is_live(s) && !seen[s as usize] {
                seen[s as usize] = true;
                stack.push(s);
            }
        }
        while let Some(id) = stack.pop() {
            let n = self.n(id);
            for &s in n.next[..n.nq as usize].iter().chain(&n.readers) {
                if s != NONE && !seen[s as usize] {
                    seen[s as usize] = true;
                    stack.push(s);
                }
            }
        }
        seen
    }

    /// Connected components of the qubits: two qubits are joined if a node
    /// touches both, or if a classically controlled node on one reads a
    /// measurement of the other. Sorted by smallest qubit, each ascending;
    /// idle qubits are singletons.
    pub fn qubit_components(&self) -> Vec<Vec<usize>> {
        let n = self.num_qubits;
        let mut parent: Vec<usize> = (0..n).collect();
        fn find(p: &mut [usize], mut x: usize) -> usize {
            while p[x] != x {
                p[x] = p[p[x]];
                x = p[x];
            }
            x
        }
        fn union(p: &mut [usize], a: usize, b: usize) {
            let (a, b) = (find(p, a), find(p, b));
            if a != b {
                p[a.max(b)] = a.min(b);
            }
        }
        for id in self.node_ids() {
            let node = self.n(id);
            let q0 = node.qs[0] as usize;
            for &q in &node.qs[1..node.nq as usize] {
                union(&mut parent, q0, q as usize);
            }
            if node.cpred != NONE {
                union(&mut parent, q0, self.n(node.cpred).qs[0] as usize);
            }
        }
        let mut by_root: Vec<Vec<usize>> = vec![Vec::new(); n];
        for q in 0..n {
            let r = find(&mut parent, q);
            by_root[r].push(q);
        }
        by_root.into_iter().filter(|v| !v.is_empty()).collect()
    }

    // -----------------------------------------------------------------------
    // Commutation
    // -----------------------------------------------------------------------

    /// True if the two nodes may be exchanged when adjacent: they are not
    /// linked by a classical or record edge and their ops commute exactly,
    /// branch by branch (see [`ops_commute`]).
    pub fn commutes(&self, a: NodeId, b: NodeId) -> bool {
        let (na, nb) = (self.n(a), self.n(b));
        if na.cpred == b || nb.cpred == a {
            return false;
        }
        if matches!(na.op, Op::Measure(_)) && matches!(nb.op, Op::Measure(_)) {
            // Record order.
            return false;
        }
        ops_commute(&na.op, &nb.op)
    }

    /// True if `b` directly follows `a` on every wire they share (and they
    /// share at least one).
    pub fn adjacent(&self, a: NodeId, b: NodeId) -> bool {
        let na = self.n(a);
        let mut shared = false;
        for i in 0..na.nq as usize {
            let q = na.qs[i] as usize;
            if self.n(b).slot(q).is_some() {
                shared = true;
                if na.next[i] != b {
                    return false;
                }
            }
        }
        shared
    }

    // -----------------------------------------------------------------------
    // Rewrites
    // -----------------------------------------------------------------------

    /// Removes a node, splicing its wires (and the record chain for a
    /// measurement). A measurement with live classical readers cannot be
    /// removed.
    pub fn remove(&mut self, id: NodeId) -> Result<(), DagError> {
        self.check_live(id)?;
        if !self.n(id).readers.is_empty() {
            return Err(DagError::MeasurementHasReaders(id));
        }
        let node = self.nodes[id as usize].clone();
        for i in 0..node.nq as usize {
            let q = node.qs[i] as usize;
            let (p, nx) = (node.prev[i], node.next[i]);
            self.set_next(p, q, nx);
            self.set_prev(nx, q, p);
        }
        if node.cpred != NONE {
            self.nodes[node.cpred as usize].readers.retain(|&r| r != id);
        }
        if matches!(node.op, Op::Measure(_)) {
            if node.rprev == NONE {
                self.rhead = node.rnext;
            } else {
                self.nodes[node.rprev as usize].rnext = node.rnext;
            }
            if node.rnext == NONE {
                self.rtail = node.rprev;
            } else {
                self.nodes[node.rnext as usize].rprev = node.rprev;
            }
        }
        let n = &mut self.nodes[id as usize];
        n.alive = false;
        n.readers = Vec::new();
        self.live -= 1;
        Ok(())
    }

    /// Sets the outgoing link on wire `q` of node `p` (or the wire head).
    fn set_next(&mut self, p: NodeId, q: usize, to: NodeId) {
        if p == NONE {
            self.head[q] = to;
        } else {
            let n = &mut self.nodes[p as usize];
            let s = n.slot(q).expect("wire link");
            n.next[s] = to;
        }
    }

    /// Sets the incoming link on wire `q` of node `x` (or the wire tail).
    fn set_prev(&mut self, x: NodeId, q: usize, to: NodeId) {
        if x == NONE {
            self.tail[q] = to;
        } else {
            let n = &mut self.nodes[x as usize];
            let s = n.slot(q).expect("wire link");
            n.prev[s] = to;
        }
    }

    /// Replaces a unitary gate node by a sequence of unitary gates acting
    /// on a subset of its qubits (an empty sequence removes it). Returns the
    /// new node ids in order.
    pub fn replace(&mut self, id: NodeId, gates: &[Gate]) -> Result<Vec<NodeId>, DagError> {
        self.check_live(id)?;
        let old = self.nodes[id as usize].clone();
        if !matches!(old.op, Op::Gate(_)) {
            return Err(DagError::BadReplacement("only gate nodes can be replaced"));
        }
        for g in gates {
            let (qs, k) = gate_qubits(g);
            for (i, &q) in qs[..k].iter().enumerate() {
                if old.slot(q).is_none() {
                    return Err(DagError::BadReplacement(
                        "replacement touches a qubit the node does not",
                    ));
                }
                if qs[..i].contains(&q) {
                    return Err(SimError::RepeatedQubit(*g).into());
                }
            }
        }
        // The new nodes take keys old.key, old.key+1, ...; they stay
        // topological if they remain below every wire successor's key.
        let top = old.key + gates.len().saturating_sub(1) as u64;
        if old.next[..old.nq as usize]
            .iter()
            .any(|&x| x != NONE && self.n(x).key <= top)
        {
            self.keys_topological = false;
        }
        if !gates.is_empty() {
            self.ids_in_key_order = false;
        }
        // Unlink the old node but remember its wire neighbours.
        let mut cur_prev = old.prev;
        self.nodes[id as usize].alive = false;
        self.live -= 1;
        let mut ids = Vec::with_capacity(gates.len());
        for (j, g) in gates.iter().enumerate() {
            let (qs, k) = gate_qubits(g);
            let nid = self.nodes.len() as NodeId;
            let mut node = Node {
                op: Op::Gate(*g),
                qs: [qs[0] as u32, qs[1] as u32, qs[2] as u32],
                nq: k as u8,
                alive: true,
                prev: [NONE; 3],
                next: [NONE; 3],
                cpred: NONE,
                readers: Vec::new(),
                rprev: NONE,
                rnext: NONE,
                key: old.key + j as u64,
            };
            for (i, &q) in qs[..k].iter().enumerate() {
                let s = old.slot(q).expect("checked");
                node.prev[i] = cur_prev[s];
                cur_prev[s] = nid;
            }
            self.nodes.push(node);
            // Link the predecessors to the new node.
            for &q in &qs[..k] {
                let p = self.nodes[nid as usize].prev[self.nodes[nid as usize].slot(q).unwrap()];
                self.set_next(p, q, nid);
            }
            self.live += 1;
            ids.push(nid);
        }
        // Close every wire: last node in the sequence (or the old prev) to
        // the old next.
        let k = old.nq as usize;
        for ((&q, &last), &nx) in old.qs[..k].iter().zip(&cur_prev[..k]).zip(&old.next[..k]) {
            let q = q as usize;
            self.set_next(last, q, nx);
            self.set_prev(nx, q, last);
        }
        Ok(ids)
    }

    /// Changes the op of a gate node to another gate on the same set of
    /// qubits (argument order may differ).
    pub fn set_gate(&mut self, id: NodeId, g: Gate) -> Result<(), DagError> {
        self.check_live(id)?;
        let node = &self.nodes[id as usize];
        if !matches!(node.op, Op::Gate(_)) {
            return Err(DagError::BadReplacement("only gate nodes can be changed"));
        }
        let (qs, k) = gate_qubits(&g);
        if k != node.nq as usize || qs[..k].iter().any(|&q| node.slot(q).is_none()) {
            return Err(DagError::BadReplacement(
                "set_gate needs the same set of qubits",
            ));
        }
        let mut prev = [NONE; 3];
        let mut next = [NONE; 3];
        for (i, &q) in qs[..k].iter().enumerate() {
            let s = node.slot(q).unwrap();
            prev[i] = node.prev[s];
            next[i] = node.next[s];
        }
        let n = &mut self.nodes[id as usize];
        n.qs = [qs[0] as u32, qs[1] as u32, qs[2] as u32];
        n.prev = prev;
        n.next = next;
        n.op = Op::Gate(g);
        Ok(())
    }

    /// Merges two gate nodes on the same qubits, with `b` directly after
    /// `a` on every wire, into `merged` placed at `a` (or removes both if
    /// `merged` is `None`). The caller guarantees `merged = b·a`.
    pub fn merge(&mut self, a: NodeId, b: NodeId, merged: Option<Gate>) -> Result<(), DagError> {
        self.check_live(a)?;
        self.check_live(b)?;
        let (na, nb) = (self.n(a), self.n(b));
        if !matches!(na.op, Op::Gate(_)) || !matches!(nb.op, Op::Gate(_)) {
            return Err(DagError::BadReplacement("merge needs two gate nodes"));
        }
        if na.nq != nb.nq || !self.adjacent(a, b) {
            return Err(DagError::NotAdjacent(a, b));
        }
        self.remove(b)?;
        match merged {
            Some(g) => self.set_gate(a, g),
            None => self.remove(a),
        }
    }

    /// True if [`Dag::slide`] would succeed: both live, adjacent on their
    /// shared wires, commuting, and not otherwise dependent.
    pub fn can_slide(&self, a: NodeId, b: NodeId) -> bool {
        self.is_live(a)
            && self.is_live(b)
            && self.adjacent(a, b)
            && self.commutes(a, b)
            && !self.has_indirect_path(a, b)
    }

    /// Exchanges two adjacent nodes that commute: `a` directly before `b`
    /// on every shared wire becomes `b` directly before `a`. Fails if `b`
    /// also depends on `a` through another path (the exchange would create
    /// a cycle); that check is a DFS from `a`, so `slide` is O(reachable).
    pub fn slide(&mut self, a: NodeId, b: NodeId) -> Result<(), DagError> {
        self.check_live(a)?;
        self.check_live(b)?;
        if !self.adjacent(a, b) {
            return Err(DagError::NotAdjacent(a, b));
        }
        if !self.commutes(a, b) {
            return Err(DagError::DoNotCommute(a, b));
        }
        if self.has_indirect_path(a, b) {
            // a -> x -> b through some other wire or classical edge:
            // exchanging a and b would create a cycle.
            return Err(DagError::NotAdjacent(a, b));
        }
        let qa: Vec<usize> = self.n(a).qubits().collect();
        for q in qa {
            let sb = match self.n(b).slot(q) {
                Some(s) => s,
                None => continue,
            };
            let sa = self.n(a).slot(q).unwrap();
            let p = self.n(a).prev[sa];
            let nx = self.n(b).next[sb];
            // p -> b -> a -> nx
            self.set_next(p, q, b);
            self.nodes[b as usize].prev[sb] = p;
            self.nodes[b as usize].next[sb] = a;
            self.nodes[a as usize].prev[sa] = b;
            self.nodes[a as usize].next[sa] = nx;
            self.set_prev(nx, q, a);
        }
        let (ka, kb) = (self.n(a).key, self.n(b).key);
        if ka < kb {
            self.nodes[a as usize].key = kb;
            self.nodes[b as usize].key = ka;
            // Other neighbours of a and b may now sit on the wrong side.
            self.keys_topological = false;
            self.ids_in_key_order = false;
        }
        Ok(())
    }

    /// True if `b` is reachable from `a` by a path other than the direct
    /// wire edges `a -> b` (through any edge kind). Costs a DFS over the
    /// nodes reachable from `a`.
    fn has_indirect_path(&self, a: NodeId, b: NodeId) -> bool {
        let mut seen = vec![false; self.nodes.len()];
        let mut stack: Vec<NodeId> = Vec::new();
        let push_succ = |id: NodeId, stack: &mut Vec<NodeId>, skip_b: bool| {
            let n = self.n(id);
            for &s in n.next[..n.nq as usize]
                .iter()
                .chain(&n.readers)
                .chain(std::iter::once(&n.rnext))
            {
                if s != NONE && !(skip_b && s == b) {
                    stack.push(s);
                }
            }
        };
        push_succ(a, &mut stack, true);
        while let Some(x) = stack.pop() {
            if x == b {
                return true;
            }
            if std::mem::replace(&mut seen[x as usize], true) {
                continue;
            }
            push_succ(x, &mut stack, false);
        }
        false
    }

    /// Checks every internal link (for tests): wires are consistent doubly
    /// linked lists over live nodes, classical and record links match, and
    /// the graph is acyclic.
    pub fn check_invariants(&self) -> Result<(), String> {
        for q in 0..self.num_qubits {
            let mut prev = NONE;
            let mut cur = self.head[q];
            let mut steps = 0;
            while cur != NONE {
                let n = self.n(cur);
                if !n.alive {
                    return Err(format!("dead node {cur} on wire {q}"));
                }
                let s = n
                    .slot(q)
                    .ok_or(format!("node {cur} on wire {q} lacks slot"))?;
                if n.prev[s] != prev {
                    return Err(format!("bad prev link at node {cur} wire {q}"));
                }
                prev = cur;
                cur = n.next[s];
                steps += 1;
                if steps > self.nodes.len() {
                    return Err(format!("cycle on wire {q}"));
                }
            }
            if self.tail[q] != prev {
                return Err(format!("bad tail on wire {q}"));
            }
        }
        for id in self.node_ids() {
            let n = self.n(id);
            for i in 0..n.nq as usize {
                for &x in &[n.prev[i], n.next[i]] {
                    if x != NONE && !self.n(x).alive {
                        return Err(format!("node {id} links dead node {x}"));
                    }
                }
            }
            if n.cpred != NONE && !self.n(n.cpred).readers.contains(&id) {
                return Err(format!("node {id} missing from its source's readers"));
            }
        }
        if self.topo_order().len() != self.live {
            return Err("cycle".into());
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Exact commutation
// ---------------------------------------------------------------------------

/// How an op acts on one of its qubits. If on qubit `q` every tensor
/// factor of the op lies in `span{I, P}` for one Pauli `P`, the op is
/// `P`-type on `q`. Two ops commute if on every shared qubit they have the
/// same (non-`Other`) type, because then every pair of tensor terms
/// commutes factor by factor. `Any` is the identity, which commutes with
/// everything.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Axis {
    Any,
    Z,
    X,
    Y,
    Other,
}

/// The [`Axis`] of a gate on qubit `q` (which it must touch).
pub fn gate_axis(g: &Gate, q: usize) -> Axis {
    use Gate::*;
    match *g {
        I(_) => Axis::Any,
        Z(_) | S(_) | Sdg(_) | T(_) | Tdg(_) | Phase(..) | Rz(..) | Cz(..) | CPhase(..) => Axis::Z,
        X(_) | Rx(..) | Sx(_) | Sxdg(_) => Axis::X,
        Y(_) | Ry(..) => Axis::Y,
        Cnot(c, _) => {
            if q == c {
                Axis::Z
            } else {
                Axis::X
            }
        }
        Ccx(_, _, t) => {
            if q == t {
                Axis::X
            } else {
                Axis::Z
            }
        }
        H(_) | U(..) | Swap(..) | ISwap(..) | ISwapdg(..) => Axis::Other,
    }
}

/// The [`Axis`] of an op on qubit `q`.
///
/// * A computational-basis measurement is `Z`-type: its projectors are
///   diagonal, so a diagonal gate commutes with it outcome by outcome.
/// * `XFlip`/`YFlip`/`ZFlip` apply `I` or one Pauli in every branch, so
///   they are `X`/`Y`/`Z`-type.
/// * A classically controlled gate applies the gate or nothing in every
///   branch, so it has its gate's type (its classical edge is handled
///   separately).
/// * Resets and depolarizing channels are opaque.
pub fn op_axis(op: &Op, q: usize) -> Axis {
    match op {
        Op::Gate(g) | Op::ClassicControlled { gate: g, .. } => gate_axis(g, q),
        Op::Measure(_) | Op::ZFlip(..) => Axis::Z,
        Op::XFlip(..) => Axis::X,
        Op::YFlip(..) => Axis::Y,
        Op::Reset(_) | Op::Depolarize1q(..) | Op::Depolarize2q(..) => Axis::Other,
    }
}

/// Gates whose 4x4 matrix is symmetric under exchanging the two qubits and
/// that commute with each other on the same pair: SWAP, CZ, CPhase, iSWAP
/// and iSWAP† are all `SWAP^a · D` with `D` diagonal and symmetric.
fn symmetric_pair(g: &Gate) -> bool {
    matches!(
        g,
        Gate::Swap(..) | Gate::Cz(..) | Gate::CPhase(..) | Gate::ISwap(..) | Gate::ISwapdg(..)
    )
}

fn same_qubit_set(a: &Gate, b: &Gate) -> bool {
    let (aq, an) = gate_qubits(a);
    let (bq, bn) = gate_qubits(b);
    an == bn && aq[..an].iter().all(|q| bq[..bn].contains(q))
}

/// True for gates without a continuous parameter (their matrices have
/// entries in `Z[1/√2, e^{iπ/4}]`, so a floating-point commutator is
/// either zero or far from it).
fn parameter_free(g: &Gate) -> bool {
    use Gate::*;
    !matches!(g, Rx(..) | Ry(..) | Rz(..) | Phase(..) | U(..) | CPhase(..))
}

/// Exact commutation test for two ops, branch by branch:
///
/// 1. ops on disjoint qubits commute;
/// 2. axis rule (see [`Axis`]) on every shared qubit;
/// 3. symmetric two-qubit gates on the same pair;
/// 4. identical unitary gates;
/// 5. for parameter-free unitary gates spanning at most 3 qubits, an
///    explicit matrix commutator (memoised; entries are exact algebraic
///    numbers, so the 1e-9 threshold cannot misclassify).
///
/// Anything else is reported as not commuting (sound, not complete).
/// Classical structure (a reader and its measurement) is checked by
/// [`Dag::commutes`], not here.
pub fn ops_commute(a: &Op, b: &Op) -> bool {
    let (aq, an) = op_qubits(a);
    let (bq, bn) = op_qubits(b);
    let mut shared = false;
    let mut axis_ok = true;
    for &q in &aq[..an] {
        if bq[..bn].contains(&q) {
            shared = true;
            let (x, y) = (op_axis(a, q), op_axis(b, q));
            let ok = x == Axis::Any || y == Axis::Any || (x == y && x != Axis::Other);
            axis_ok &= ok;
        }
    }
    if !shared || axis_ok {
        return true;
    }
    let (Op::Gate(ga), Op::Gate(gb)) = (a, b) else {
        return false;
    };
    if an == 2 && bn == 2 && symmetric_pair(ga) && symmetric_pair(gb) && same_qubit_set(ga, gb) {
        return true;
    }
    if ga == gb {
        return true;
    }
    // Two single-qubit named gates commute iff they share an eigenbasis
    // (Z: diagonal, X: X/Sx, Y: Y, or H's), which the axis rule and the
    // equality test above already decide; only multi-qubit pairs need the
    // matrix check.
    if (an > 1 || bn > 1) && parameter_free(ga) && parameter_free(gb) {
        return matrix_commute_cached(ga, gb);
    }
    false
}

thread_local! {
    /// Memo of matrix commutation checks, keyed by [`shape_key`].
    static COMMUTE_CACHE: RefCell<HashMap<u32, bool, BuildHasherDefault<MulHasher>>> =
        RefCell::new(HashMap::default());
}

/// Multiplicative hasher for small integer keys (the std SipHash costs
/// more than the rest of a commutation check).
#[derive(Default)]
struct MulHasher(u64);

impl Hasher for MulHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0.rotate_left(5) ^ b as u64).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
        }
    }
    fn write_u32(&mut self, x: u32) {
        self.0 = (x as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    }
}

/// Index of a parameter-free gate kind (`None` for parametrised gates).
fn kind_index(g: &Gate) -> Option<u32> {
    use Gate::*;
    Some(match g {
        I(_) => 0,
        H(_) => 1,
        X(_) => 2,
        Y(_) => 3,
        Z(_) => 4,
        S(_) => 5,
        Sdg(_) => 6,
        T(_) => 7,
        Tdg(_) => 8,
        Sx(_) => 9,
        Sxdg(_) => 10,
        Cnot(..) => 11,
        Cz(..) => 12,
        Swap(..) => 13,
        ISwap(..) => 14,
        ISwapdg(..) => 15,
        Ccx(..) => 16,
        Rx(..) | Ry(..) | Rz(..) | Phase(..) | U(..) | CPhase(..) => return None,
    })
}

/// A key identifying a pair of parameter-free gates up to a relabelling
/// of their qubits (in order of first appearance), plus the relabelling.
/// `None` if a gate is parametrised or the pair spans more than 3 qubits.
fn shape_key(a: &Gate, b: &Gate) -> Option<(u32, [usize; 3], usize)> {
    let (ka, kb) = (kind_index(a)?, kind_index(b)?);
    let (aq, an) = gate_qubits(a);
    let (bq, bn) = gate_qubits(b);
    let mut map = [usize::MAX; 3];
    let mut k = 0;
    let mut key = ka | kb << 5;
    let mut shift = 10;
    for &q in aq[..an].iter().chain(&bq[..bn]) {
        let idx = match map[..k].iter().position(|&m| m == q) {
            Some(i) => i,
            None => {
                if k == 3 {
                    return None;
                }
                map[k] = q;
                k += 1;
                k - 1
            }
        };
        key |= (idx as u32) << shift;
        shift += 2;
    }
    Some((key, map, k))
}

/// Applies a qubit relabelling to a gate.
pub fn map_gate(g: &Gate, f: impl Fn(usize) -> usize) -> Gate {
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
        Rx(q, t) => Rx(f(q), t),
        Ry(q, t) => Ry(f(q), t),
        Rz(q, t) => Rz(f(q), t),
        Phase(q, t) => Phase(f(q), t),
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

fn matrix_commute_cached(a: &Gate, b: &Gate) -> bool {
    let Some((key, map, k)) = shape_key(a, b) else {
        return false;
    };
    if let Some(v) = COMMUTE_CACHE.with(|c| c.borrow().get(&key).copied()) {
        return v;
    }
    let f = |q: usize| map[..k].iter().position(|&m| m == q).unwrap();
    let (ra, rb) = (map_gate(a, f), map_gate(b, f));
    let ma = dense(&ra, k);
    let mb = dense(&rb, k);
    let ab = matmul(&ma, &mb, 1 << k);
    let ba = matmul(&mb, &ma, 1 << k);
    let v = ab.iter().zip(&ba).all(|(x, y)| (x - y).norm() < 1e-9);
    COMMUTE_CACHE.with(|c| c.borrow_mut().insert(key, v));
    v
}

/// Dense `2^k x 2^k` matrix of a gate on qubits `0..k` (qubit `q` = bit
/// `q` of the index), built column by column from the state-vector kernel.
pub fn dense(g: &Gate, k: usize) -> Vec<Complex64> {
    let dim = 1usize << k;
    let mut m = vec![Complex64::new(0.0, 0.0); dim * dim];
    for col in 0..dim {
        let mut sv = crate::statevector::StateVectorF64::basis_state(k, col);
        sv.apply_gate(g).expect("valid gate");
        for (row, a) in sv.amplitudes().iter().enumerate() {
            m[row * dim + col] = *a;
        }
    }
    m
}

fn matmul(a: &[Complex64], b: &[Complex64], d: usize) -> Vec<Complex64> {
    let mut out = vec![Complex64::new(0.0, 0.0); d * d];
    for i in 0..d {
        for k in 0..d {
            let x = a[i * d + k];
            if x.norm_sqr() == 0.0 {
                continue;
            }
            for j in 0..d {
                out[i * d + j] += x * b[k * d + j];
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Passes
// ---------------------------------------------------------------------------

/// Light cone of every measurement and of the final state of `outputs`,
/// computed as reverse reachability on the DAG (wire + classical edges).
/// Same contract as the flat-list light cone of the compile module: every
/// measurement is kept, dropped ops are trace-preserving channels on
/// qubits nothing kept later touches, so the joint distribution of the
/// whole record and the reduced state of `outputs` are unchanged.
pub fn light_cone(c: &Circuit, outputs: &[usize]) -> Result<Circuit, DagError> {
    let d = Dag::from_circuit(c)?;
    let mut seeds = d.measurements();
    for &q in outputs {
        seeds.extend(d.last_on(q));
    }
    Ok(d.restrict_to(&d.ancestors(&seeds)))
}

/// Light cone of a *subset* of the measurements (record positions in
/// `keep_bits`, any order) and of the final state of `outputs`. Unlike
/// [`light_cone`], measurements outside the cone are dropped too, so the
/// result reproduces the *marginal* distribution of the selected bits.
///
/// Returns the circuit and, for each bit of its record, the bit of the
/// original record it reproduces (ascending). A selected measurement's
/// classical dependencies are followed, so a kept `ClassicControlled` op
/// always keeps the measurement it reads.
pub fn light_cone_marginal(
    c: &Circuit,
    keep_bits: &[usize],
    outputs: &[usize],
) -> Result<(Circuit, Vec<usize>), DagError> {
    let d = Dag::from_circuit(c)?;
    let ms = d.measurements();
    let mut seeds = Vec::new();
    for &b in keep_bits {
        let m = *ms
            .get(b)
            .ok_or(DagError::Sim(SimError::ClassicalBitOutOfRange {
                bit: b,
                available: ms.len(),
            }))?;
        seeds.push(m);
    }
    for &q in outputs {
        seeds.extend(d.last_on(q));
    }
    let keep = d.ancestors(&seeds);
    let bits = ms
        .iter()
        .enumerate()
        .filter(|(_, &m)| keep[m as usize])
        .map(|(i, _)| i)
        .collect();
    Ok((d.restrict_to(&keep), bits))
}

impl Dag {
    /// The circuit of the nodes marked in `keep` (which must be closed
    /// under predecessors), in topological order.
    pub fn restrict_to(&self, keep: &[bool]) -> Circuit {
        let order: Vec<NodeId> = self
            .topo_order()
            .into_iter()
            .filter(|&id| keep[id as usize])
            .collect();
        self.circuit_in_order(&order)
    }
}

/// Connected components of the qubit-interaction graph (see
/// [`Dag::qubit_components`]).
pub fn components(c: &Circuit) -> Result<Vec<Vec<usize>>, DagError> {
    Ok(Dag::from_circuit(c)?.qubit_components())
}

// ---------------------------------------------------------------------------
// Commutation-aware peephole
// ---------------------------------------------------------------------------

/// Angles within this distance of a special value snap to it. Snapping a
/// rotation by `ε` changes amplitudes by at most `ε / 2`.
pub const ANGLE_EPS: f64 = 1e-13;

/// An optimised circuit: `U_original = e^{i global_phase} U_circuit` (the
/// same type the flat-list passes in [`crate::compile`] return).
pub use crate::compile::Optimized;

/// Options for [`peephole`].
#[derive(Clone, Copy, Debug)]
pub struct PeepholeOptions {
    /// Maximum number of nodes a gate may slide past (per wire) while
    /// looking for a merge partner.
    pub window: usize,
    /// Repeat whole passes until nothing changes (a cancellation can open
    /// a path for a gate that was blocked before).
    pub fixpoint: bool,
    /// If false, a gate only merges with its direct wire neighbour (the
    /// adjacent-only baseline, for A/B measurements).
    pub commute: bool,
    /// After a rewrite, how many earlier nodes per wire are re-queued for
    /// another partner search (they may have been blocked by the removed
    /// gates). 0 relies on whole extra passes alone.
    pub look_back: usize,
}

impl Default for PeepholeOptions {
    fn default() -> Self {
        PeepholeOptions {
            window: 1024,
            fixpoint: true,
            commute: true,
            look_back: 4,
        }
    }
}

/// Statistics of a peephole run.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PeepholeStats {
    pub passes: usize,
    pub merges: usize,
    pub cancellations: usize,
    pub removed_identities: usize,
    /// Partner searches started, and nodes walked past in total.
    pub searches: usize,
    pub walk_steps: usize,
}

/// Commutation-aware peephole on a circuit (see [`peephole`]).
pub fn optimize(c: &Circuit) -> Result<Optimized, DagError> {
    optimize_with(c, PeepholeOptions::default()).map(|(o, _)| o)
}

/// Commutation-aware peephole with options and statistics.
pub fn optimize_with(
    c: &Circuit,
    opts: PeepholeOptions,
) -> Result<(Optimized, PeepholeStats), DagError> {
    let mut d = Dag::from_circuit(c)?;
    let (phase, stats) = peephole(&mut d, opts);
    Ok((
        Optimized {
            circuit: d.to_circuit(),
            global_phase: phase,
        },
        stats,
    ))
}

/// Rotation families: every gate in a family is `e^{iφ} R(θ)` for one
/// generator. `Z`: `R = Phase(θ)` (`θ` in units of π/4 plus a continuous
/// part), `X`: `R = Rx(θ)`, `Y`: `R = Ry(θ)` (units of π/2), `CP`:
/// `R = CPhase(θ)` on a pair (units of π/4).
#[derive(Clone, Copy, Debug, PartialEq)]
enum Family {
    Z,
    X,
    Y,
    Cp,
}

/// `gate = e^{i phase} R_family(units·quantum + cont)`.
#[derive(Clone, Copy, Debug)]
struct Rot {
    fam: Family,
    units: i64,
    cont: f64,
    phase: f64,
    /// Emit `Rz` rather than `Phase` for a continuous Z rotation.
    rz_style: bool,
}

fn rot_of(g: &Gate) -> Option<Rot> {
    use Gate::*;
    let r = |fam, units, cont, phase| Rot {
        fam,
        units,
        cont,
        phase,
        rz_style: false,
    };
    Some(match *g {
        I(_) => r(Family::Z, 0, 0.0, 0.0),
        Z(_) => r(Family::Z, 4, 0.0, 0.0),
        S(_) => r(Family::Z, 2, 0.0, 0.0),
        Sdg(_) => r(Family::Z, -2, 0.0, 0.0),
        T(_) => r(Family::Z, 1, 0.0, 0.0),
        Tdg(_) => r(Family::Z, -1, 0.0, 0.0),
        Phase(_, t) => r(Family::Z, 0, t, 0.0),
        // Rz(θ) = e^{-iθ/2} Phase(θ)
        Rz(_, t) => Rot {
            rz_style: true,
            ..r(Family::Z, 0, t, -t / 2.0)
        },
        // X = i Rx(π), Sx = e^{iπ/4} Rx(π/2), Sx† = e^{-iπ/4} Rx(-π/2)
        X(_) => r(Family::X, 2, 0.0, FRAC_PI_2),
        Sx(_) => r(Family::X, 1, 0.0, FRAC_PI_4),
        Sxdg(_) => r(Family::X, -1, 0.0, -FRAC_PI_4),
        Rx(_, t) => r(Family::X, 0, t, 0.0),
        // Y = i Ry(π)
        Y(_) => r(Family::Y, 2, 0.0, FRAC_PI_2),
        Ry(_, t) => r(Family::Y, 0, t, 0.0),
        // CZ = CPhase(π)
        Cz(..) => r(Family::Cp, 4, 0.0, 0.0),
        CPhase(_, _, t) => r(Family::Cp, 0, t, 0.0),
        _ => return None,
    })
}

fn quantum(f: Family) -> f64 {
    match f {
        Family::Z | Family::Cp => FRAC_PI_4,
        Family::X | Family::Y => FRAC_PI_2,
    }
}

/// Moves continuous angles that are (within [`ANGLE_EPS`]) a multiple of
/// the family quantum into `units`, and reduces `units` to the family
/// period. `Rx`/`Ry` have period 4π with `R(2π) = -I`.
fn normalize(mut r: Rot) -> Rot {
    let qn = quantum(r.fam);
    if r.cont != 0.0 {
        let k = (r.cont / qn).round();
        if (r.cont - k * qn).abs() < ANGLE_EPS && k.abs() < 1e15 {
            // R(cont) -> R(k·qn) changes nothing up to ε/2. For Rz the phase
            // stays the -cont/2 that was recorded (also within ε/2).
            r.units += k as i64;
            r.cont = 0.0;
        }
    }
    match r.fam {
        Family::Z | Family::Cp => {
            r.units = r.units.rem_euclid(8);
        }
        Family::X | Family::Y => {
            // units in [-1, 2] (quantum π/2), each 4π/... shift by 2π adds π.
            r.units = r.units.rem_euclid(8);
            if r.units >= 4 {
                r.units -= 4;
                r.phase += PI;
            }
            if r.units == 3 {
                r.units = -1;
                r.phase += PI;
            }
        }
    }
    r
}

/// The gate for a normalised rotation on `qs` and the extra global phase
/// (`R = e^{i extra} gate`); `None` means the identity.
fn emit(r: Rot, qs: &[usize]) -> (Option<Gate>, f64) {
    use Gate::*;
    let q = qs[0];
    let qn = quantum(r.fam);
    if r.cont != 0.0 {
        let theta = r.units as f64 * qn + r.cont;
        return match r.fam {
            Family::Z => {
                if r.rz_style {
                    // Phase(θ) = e^{iθ/2} Rz(θ)
                    (Some(Rz(q, theta)), theta / 2.0)
                } else {
                    (Some(Phase(q, theta)), 0.0)
                }
            }
            Family::X => (Some(Rx(q, theta)), 0.0),
            Family::Y => (Some(Ry(q, theta)), 0.0),
            Family::Cp => (Some(CPhase(qs[0], qs[1], theta)), 0.0),
        };
    }
    match r.fam {
        Family::Z => (
            match r.units {
                0 => None,
                1 => Some(T(q)),
                2 => Some(S(q)),
                4 => Some(Z(q)),
                6 => Some(Sdg(q)),
                7 => Some(Tdg(q)),
                u => Some(Phase(q, u as f64 * FRAC_PI_4)),
            },
            0.0,
        ),
        Family::X => match r.units {
            0 => (None, 0.0),
            1 => (Some(Sx(q)), -FRAC_PI_4),
            2 => (Some(X(q)), -FRAC_PI_2),
            -1 => (Some(Sxdg(q)), FRAC_PI_4),
            _ => unreachable!(),
        },
        Family::Y => match r.units {
            0 => (None, 0.0),
            2 => (Some(Y(q)), -FRAC_PI_2),
            u => (Some(Ry(q, u as f64 * FRAC_PI_2)), 0.0),
        },
        Family::Cp => (
            match r.units {
                0 => None,
                4 => Some(Cz(qs[0], qs[1])),
                u => Some(CPhase(qs[0], qs[1], u as f64 * FRAC_PI_4)),
            },
            0.0,
        ),
    }
}

/// Canonical form of a single gate: named gate for special angles, `None`
/// for the identity. Returns the global phase introduced.
fn canonical(g: &Gate) -> (Option<Gate>, f64) {
    let Some(r) = rot_of(g) else {
        return (Some(*g), 0.0);
    };
    let (qs, k) = gate_qubits(g);
    let r = normalize(r);
    let (out, extra) = emit(r, &qs[..k]);
    // Keep the original gate if canonicalisation would only rename it.
    match out {
        Some(o) if gate_cost_eq(&o, g) => (Some(*g), 0.0),
        _ => (out, r.phase + extra),
    }
}

/// True if two gates are the same up to an exactly representable rename
/// we do not want to apply on its own (e.g. `CPhase(a,b,θ)` vs
/// `CPhase(b,a,θ)`, or a continuous rotation that stays continuous).
fn gate_cost_eq(a: &Gate, b: &Gate) -> bool {
    use Gate::*;
    matches!(
        (a, b),
        (Rz(..), Rz(..))
            | (Phase(..), Phase(..))
            | (Rx(..), Rx(..))
            | (Ry(..), Ry(..))
            | (CPhase(..), CPhase(..))
    ) || a == b
}

/// Family of a gate for [`combinable`]: rotation families and the
/// self-inverse kinds.
#[inline]
fn merge_class(g: &Gate) -> u8 {
    use Gate::*;
    match g {
        I(_) | Z(_) | S(_) | Sdg(_) | T(_) | Tdg(_) | Phase(..) | Rz(..) => 1,
        X(_) | Sx(_) | Sxdg(_) | Rx(..) => 2,
        Y(_) | Ry(..) => 3,
        Cz(..) | CPhase(..) => 4,
        H(_) => 5,
        Swap(..) => 6,
        Cnot(..) => 7,
        Ccx(..) => 8,
        ISwap(..) | ISwapdg(..) => 9,
        U(..) => 10,
    }
}

/// Cheap test: `combine(first, second).is_some()`.
#[inline]
fn combinable(first: &Gate, second: &Gate) -> bool {
    use Gate::*;
    let c = merge_class(first);
    if c != merge_class(second) || !same_qubit_set(first, second) {
        return false;
    }
    match (*first, *second) {
        (Cnot(c1, t1), Cnot(c2, t2)) => c1 == c2 && t1 == t2,
        (Ccx(_, _, t1), Ccx(_, _, t2)) => t1 == t2,
        (ISwap(..), ISwapdg(..)) | (ISwapdg(..), ISwap(..)) => true,
        (ISwap(..), ISwap(..)) | (ISwapdg(..), ISwapdg(..)) => false,
        (U(..), U(..)) => *second == first.inverse(),
        _ => true,
    }
}

/// If `second · first` (first applied first) is a single gate or the
/// identity, returns it with the global phase introduced.
fn combine(first: &Gate, second: &Gate) -> Option<(Option<Gate>, f64)> {
    use Gate::*;
    if !same_qubit_set(first, second) {
        return None;
    }
    match (*first, *second) {
        (H(_), H(_)) | (Swap(..), Swap(..)) => return Some((None, 0.0)),
        (Cnot(c1, t1), Cnot(c2, t2)) if c1 == c2 && t1 == t2 => return Some((None, 0.0)),
        // Same set of three qubits and same target: same controls.
        (Ccx(_, _, t1), Ccx(_, _, t2)) if t1 == t2 => return Some((None, 0.0)),
        (ISwap(..), ISwapdg(..)) | (ISwapdg(..), ISwap(..)) => return Some((None, 0.0)),
        // U(θ, φ, λ)† = U(-θ, -λ, -φ) exactly.
        (U(..), U(..)) if *second == first.inverse() => return Some((None, 0.0)),
        _ => {}
    }
    let (ra, rb) = (rot_of(first)?, rot_of(second)?);
    if ra.fam != rb.fam {
        return None;
    }
    let sum = Rot {
        fam: ra.fam,
        units: ra.units + rb.units,
        cont: ra.cont + rb.cont,
        phase: ra.phase + rb.phase,
        rz_style: ra.rz_style && rb.rz_style,
    };
    let r = normalize(sum);
    let (qs, k) = gate_qubits(first);
    let (out, extra) = emit(r, &qs[..k]);
    Some((out, r.phase + extra))
}

/// Commutation-aware cancellation and merging on a DAG.
///
/// Gates are visited in topological order. Each gate walks forward along
/// its wires past nodes it commutes with ([`ops_commute`]; a gate never has
/// a classical edge, so wire neighbours are all it must pass) and stops at
/// the first node it does not commute with. If on every wire it reaches
/// the same gate of its own family on the same qubits, the two are merged
/// at the later position (`H…H -> I`, `T…T -> S`, `Rz(a)…Rz(b) ->
/// Rz(a+b)`, `CZ…CPhase(θ) -> CPhase(θ+π)`, `CNOT…CNOT -> I`, …). This is
/// exact because the first gate commutes with everything between them, so
/// it can be slid next to the second one.
///
/// Every rewrite's global phase is accumulated in the return value:
/// `U_before = e^{i phase} U_after`. Only `Op::Gate` nodes are rewritten;
/// measurements, resets, noise and classically controlled gates stay as
/// they are (gates may slide past them when they commute).
pub fn peephole(d: &mut Dag, opts: PeepholeOptions) -> (f64, PeepholeStats) {
    let mut phase = 0.0;
    let mut stats = PeepholeStats::default();
    // Canonicalise single gates (identity rotations vanish).
    for id in d.node_ids().collect::<Vec<_>>() {
        if let Op::Gate(g) = *d.op(id) {
            let (out, ph) = canonical(&g);
            phase += ph;
            match out {
                None => {
                    d.remove(id).expect("live gate");
                    stats.removed_identities += 1;
                }
                Some(o) if o != g => d.set_gate(id, o).expect("same qubits"),
                Some(_) => {}
            }
        }
    }
    // Worklist: gates in topological order; after a rewrite, the nodes just
    // before it on its wires are re-examined at once (their forward walk may
    // have been blocked by the removed gates), so nested patterns such as
    // `A B C C† B† A†` collapse in one sweep. Whole passes repeat until
    // nothing changes (when `fixpoint` is set) to catch anything the bounded
    // look-back missed.
    let mut queued = vec![false; d.capacity()];
    loop {
        stats.passes += 1;
        let mut changed = false;
        let mut work: Vec<NodeId> = d.topo_order();
        work.reverse(); // pop() yields topological order
        for &id in &work {
            queued[id as usize] = true;
        }
        while let Some(id) = work.pop() {
            queued[id as usize] = false;
            if !d.is_live(id) {
                continue;
            }
            let Op::Gate(g) = *d.op(id) else { continue };
            stats.searches += 1;
            let Some((m, mg)) = find_partner(d, id, &g, opts, &mut stats.walk_steps) else {
                continue;
            };
            let (out, ph) = combine(&g, &mg).expect("partner is combinable");
            phase += ph;
            let mut revisit = wire_predecessors(d, id, opts.look_back);
            d.remove(id).expect("live");
            match out {
                None => {
                    revisit.extend(wire_predecessors(d, m, opts.look_back));
                    d.remove(m).expect("live");
                    stats.cancellations += 1;
                }
                Some(o) => {
                    d.set_gate(m, o).expect("same qubits");
                    revisit.push(m);
                    stats.merges += 1;
                }
            }
            if queued.len() < d.capacity() {
                queued.resize(d.capacity(), false);
            }
            // Re-examine later nodes after earlier ones: push in reverse
            // key order so the earliest is popped first.
            revisit.sort_by_key(|&x| Reverse(d.n(x).key));
            for x in revisit {
                if d.is_live(x) && !std::mem::replace(&mut queued[x as usize], true) {
                    work.push(x);
                }
            }
            changed = true;
        }
        if !changed || !opts.fixpoint {
            break;
        }
    }
    (phase.rem_euclid(TAU), stats)
}

/// Up to `k` live nodes before `id` on each of its wires.
fn wire_predecessors(d: &Dag, id: NodeId, k: usize) -> Vec<NodeId> {
    let n = d.n(id);
    let mut out = Vec::new();
    for i in 0..n.nq as usize {
        let q = n.qs[i] as usize;
        let mut cur = n.prev[i];
        let mut steps = 0;
        while cur != NONE && steps < k {
            out.push(cur);
            let c = d.n(cur);
            cur = c.prev[c.slot(q).expect("wire")];
            steps += 1;
        }
    }
    out
}

/// Walks forward from gate node `id` along each of its wires, past nodes
/// it commutes with, to the first node it can combine with. Returns that
/// node if it is the same on every wire.
fn find_partner(
    d: &Dag,
    id: NodeId,
    g: &Gate,
    opts: PeepholeOptions,
    walked: &mut usize,
) -> Option<(NodeId, Gate)> {
    let gop = Op::Gate(*g);
    let node = d.n(id);
    let mut found: Option<NodeId> = None;
    for i in 0..node.nq as usize {
        let q = node.qs[i] as usize;
        let mut cur = node.next[i];
        let mut steps = 0;
        let hit = loop {
            if cur == NONE {
                return None;
            }
            let c = d.n(cur);
            if let Op::Gate(cg) = c.op {
                if combinable(g, &cg) {
                    break cur;
                }
            }
            if !opts.commute || steps >= opts.window || !ops_commute(&gop, &c.op) {
                return None;
            }
            steps += 1;
            *walked += 1;
            cur = c.next[c.slot(q).expect("wire")];
        };
        match found {
            None => found = Some(hit),
            Some(f) if f == hit => {}
            Some(_) => return None,
        }
    }
    let m = found?;
    let Op::Gate(mg) = d.n(m).op else {
        return None;
    };
    Some((m, mg))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::statevector::StateVectorF64;

    fn sv(c: &Circuit) -> StateVectorF64 {
        let mut s = StateVectorF64::new(c.num_qubits);
        s.apply_circuit(c).unwrap();
        s
    }

    #[test]
    fn round_trip_is_identity() {
        let mut c = Circuit::new(3);
        c.h(0)
            .cnot(0, 1)
            .measure(1)
            .t(2)
            .c_if(0, Gate::X(2))
            .measure(2);
        c.depolarize_2q(0, 2, 0.1).reset(1);
        let d = Dag::from_circuit(&c).unwrap();
        d.check_invariants().unwrap();
        assert_eq!(d.to_circuit(), c);
        assert_eq!(d.len(), c.ops.len());
    }

    #[test]
    fn wires_and_classical_edges() {
        let mut c = Circuit::new(2);
        c.h(0).measure(0).c_if(0, Gate::X(1)).cnot(0, 1);
        let d = Dag::from_circuit(&c).unwrap();
        assert_eq!(d.first_on(0), Some(0));
        assert_eq!(d.wire_next(0, 0), Some(1));
        assert_eq!(d.classical_source(2), Some(1));
        assert_eq!(d.classical_readers(1), &[2]);
        assert_eq!(d.predecessors(3), vec![1, 2]);
        assert_eq!(d.front_layer(), vec![0]);
        assert_eq!(d.asap_layers(), vec![vec![0], vec![1], vec![2], vec![3]]);
    }

    #[test]
    fn layers() {
        let mut c = Circuit::new(3);
        c.h(0).h(1).cnot(0, 1).h(2);
        let d = Dag::from_circuit(&c).unwrap();
        assert_eq!(d.asap_layers(), vec![vec![0, 1, 3], vec![2]]);
        assert_eq!(d.alap_layers(), vec![vec![0, 1], vec![2, 3]]);
    }

    #[test]
    fn slide_and_merge() {
        let mut c = Circuit::new(2);
        c.t(0).cnot(0, 1).t(0);
        let mut d = Dag::from_circuit(&c).unwrap();
        d.slide(0, 1).unwrap();
        d.check_invariants().unwrap();
        assert!(d.adjacent(0, 2));
        d.merge(0, 2, Some(Gate::S(0))).unwrap();
        d.check_invariants().unwrap();
        let out = d.to_circuit();
        assert_eq!(
            out.ops,
            vec![Op::Gate(Gate::Cnot(0, 1)), Op::Gate(Gate::S(0))]
        );
        // S(0) commutes with the CNOT control: slide it back and forth.
        d.slide(1, 0).unwrap();
        d.slide(0, 1).unwrap();
        d.check_invariants().unwrap();
        // H does not commute with a CNOT control; non-adjacent nodes cannot slide.
        let mut c = Circuit::new(3);
        c.h(0).cnot(0, 1).x(2);
        let mut d = Dag::from_circuit(&c).unwrap();
        assert_eq!(d.slide(0, 1), Err(DagError::DoNotCommute(0, 1)));
        assert_eq!(d.slide(0, 2), Err(DagError::NotAdjacent(0, 2)));
    }

    #[test]
    fn replace_keeps_wires() {
        let mut c = Circuit::new(2);
        c.h(0).swap(0, 1).h(1);
        let mut d = Dag::from_circuit(&c).unwrap();
        let ids = d
            .replace(1, &[Gate::Cnot(0, 1), Gate::Cnot(1, 0), Gate::Cnot(0, 1)])
            .unwrap();
        assert_eq!(ids.len(), 3);
        d.check_invariants().unwrap();
        let out = d.to_circuit();
        let (a, b) = (sv(&c), sv(&out));
        assert!(a.fidelity(&b) > 1.0 - 1e-12);
        d.replace(ids[1], &[]).unwrap();
        d.check_invariants().unwrap();
        assert_eq!(d.to_circuit().ops.len(), 4);
    }

    /// The key-sorted fast path of `topo_order` must agree with the
    /// min-key Kahn sort whenever it is used.
    #[test]
    fn fast_topo_order_matches_kahn() {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(5);
        for trial in 0..200 {
            let c = Circuit::random_clifford_t(4, 6, 0.3, &mut rng);
            let mut d = Dag::from_circuit(&c).unwrap();
            for _ in 0..10 {
                let ids: Vec<_> = d.node_ids().collect();
                if ids.is_empty() {
                    break;
                }
                let a = ids[rng.random_range(0..ids.len())];
                match rng.random_range(0..4) {
                    0 => {
                        d.remove(a).unwrap();
                    }
                    1 => {
                        if let Op::Gate(g) = *d.op(a) {
                            let qs = gate_qubits(&g).0;
                            d.replace(a, &[Gate::H(qs[0]), g, Gate::H(qs[0])]).unwrap();
                        }
                    }
                    2 if trial % 3 == 0 => {
                        for b in d.successors(a) {
                            if d.can_slide(a, b) {
                                d.slide(a, b).unwrap();
                                break;
                            }
                        }
                    }
                    _ => {}
                }
                let kahn = d.topo_order_with(|id| d.n(id).key);
                assert_eq!(d.topo_order(), kahn, "trial {trial}");
                d.check_invariants().unwrap();
            }
        }
    }

    #[test]
    fn combinable_agrees_with_combine() {
        use Gate::*;
        let mut gates = Vec::new();
        for q in 0..2 {
            gates.extend([
                I(q),
                H(q),
                X(q),
                Y(q),
                Z(q),
                S(q),
                Sdg(q),
                T(q),
                Tdg(q),
                Sx(q),
            ]);
            gates.extend([Sxdg(q), Rx(q, 0.3), Ry(q, -0.2), Rz(q, 1.1), Phase(q, 0.4)]);
            gates.extend([U(q, 0.1, 0.2, 0.3), U(q, -0.1, -0.3, -0.2)]);
        }
        for (a, b) in [(0, 1), (1, 0)] {
            gates.extend([Cnot(a, b), Cz(a, b), Swap(a, b), ISwap(a, b), ISwapdg(a, b)]);
            gates.push(CPhase(a, b, 0.7));
        }
        gates.extend([Ccx(0, 1, 2), Ccx(1, 0, 2), Ccx(0, 2, 1)]);
        for a in &gates {
            for b in &gates {
                assert_eq!(combinable(a, b), combine(a, b).is_some(), "{a:?} {b:?}");
            }
        }
    }

    #[test]
    fn commutation_rules() {
        use Gate::*;
        let g = |g| Op::Gate(g);
        assert!(ops_commute(&g(T(0)), &g(Cnot(0, 1))));
        assert!(!ops_commute(&g(T(1)), &g(Cnot(0, 1))));
        assert!(ops_commute(&g(Rx(1, 0.3)), &g(Cnot(0, 1))));
        assert!(ops_commute(&g(Swap(0, 1)), &g(Cz(1, 0))));
        assert!(ops_commute(&g(ISwap(0, 1)), &g(Swap(0, 1))));
        assert!(ops_commute(&g(Cnot(0, 2)), &g(Ccx(0, 1, 2))));
        assert!(!ops_commute(&g(H(0)), &g(X(0))));
        assert!(ops_commute(&Op::Measure(0), &g(Cz(0, 1))));
        assert!(!ops_commute(&Op::Measure(0), &g(H(0))));
        assert!(!ops_commute(&Op::Reset(0), &g(Z(0))));
        assert!(ops_commute(&Op::ZFlip(0, 0.2), &g(T(0))));
        // matrix check: CNOT(0,1) and CNOT(0,1) identical; X(1) and CNOT(0,1) via axis
        assert!(ops_commute(&g(X(1)), &g(Cnot(0, 1))));
        // matrix check finds Swap(0,1) commutes with Y(0)Y(1)? not a gate; check H⊗H-type:
        // CZ and Swap handled; Cnot(0,1) and Cnot(1,0) do not commute.
        assert!(!ops_commute(&g(Cnot(0, 1)), &g(Cnot(1, 0))));
    }

    #[test]
    fn peephole_cancels_across_commuting_gates() {
        let mut c = Circuit::new(2);
        c.t(0).cnot(0, 1).tdg(0).h(1).cnot(0, 1).h(1);
        c.rz(1, 0.3).cz(0, 1).rz(1, 0.4);
        let o = optimize(&c).unwrap();
        // T..T† cancel across the CNOT control; Rz merges across the CZ.
        assert_eq!(o.circuit.num_gates(), 6);
        let a = sv(&c);
        let b = sv(&o.circuit);
        let ph = Complex64::from_polar(1.0, o.global_phase);
        for (x, y) in a.amplitudes().iter().zip(b.amplitudes()) {
            assert!((x - ph * y).norm() < 1e-12);
        }
    }
}
