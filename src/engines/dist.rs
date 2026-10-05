//! Distributed state vector over two machines (MPI-style, global-qubit swaps).
//!
//! The `2^n` amplitudes are split into `2^G` *ranks* of `2^L` amplitudes
//! (`L = local_bits`, `G = n - L`). Physical qubits `0..L` index inside a
//! rank, physical qubits `L..n` (the *global* qubits) select the rank. Each
//! rank is owned by one of two nodes; the ownership table is arbitrary, so
//! a symmetric split (`G = 1`, one rank per node) and an asymmetric one (e.g.
//! `G = 3`, the laptop owns 7 ranks and the server 1) use the same code.
//!
//! * **Local gates** (every non-diagonal qubit of the gate is local) run on
//!   each owned rank independently with the cache-blocked kernels of
//!   [`crate::engines::blocked`]. A global qubit may appear in a *diagonal or control*
//!   role ([`needs_local`]): inside rank `r` it has the fixed value
//!   `bit(r, p - L)`, so the gate is *specialised* per rank (a `CPhase` with
//!   one global qubit becomes a `Phase` or nothing, a `Cnot` with a global
//!   control becomes `X` or nothing, a diagonal gate on a global qubit
//!   becomes a scalar). No data moves for these.
//! * **Global-qubit swaps**: to act non-diagonally on a global qubit, swap it
//!   with a local one. For every rank pair `(r, r | 1<<g)` the half of `r`
//!   with local bit `l = 1` is exchanged with the half of the partner with
//!   `l = 0`. Pairs on the same node swap in memory; pairs split across the
//!   nodes stream their halves over the [`Link`] in both directions at once
//!   (half a rank each way per cross pair).
//! * **Swap gates** that touch a global qubit are folded into the
//!   logical->physical map (no data moves), and the initial layout is free:
//!   the starting basis state is a product state, so any qubit relabelling of
//!   it is the same basis state with a permuted index. The planner puts the
//!   qubits whose first non-diagonal use is latest on the global slots.
//!
//! Both nodes run the same deterministic plan (SPMD, no coordinator); the
//! [`handshake`] compares a fingerprint of the plan and layout before any
//! amplitude moves. Transport is a trait ([`Link`]) with a TCP implementation
//! ([`TcpLink`], also used over loopback and SSH tunnels) and an in-process
//! channel implementation ([`chan_pair`]) for tests.
//!
//! Exactness: gate kernels are the in-RAM blocked kernels; swaps and
//! specialisation are exact (index moves and multiplication by the same
//! diagonal entries the kernels would use).

use crate::circuit::{check_gate, Circuit, Op, SimError};
use crate::engines::blocked::{BlockConfig, BlockedChunkExecutor};
use crate::engines::ooc::remap_gate;
use crate::engines::ooc_window::permute_bits;
use crate::engines::statevector::Real;
use crate::gate::Gate;
use num_complex::{Complex, Complex64};
use rayon::prelude::*;
use std::io::{self, Read, Write};
use std::net::{TcpListener, TcpStream, ToSocketAddrs};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Transport
// ---------------------------------------------------------------------------

/// A bidirectional byte stream to the other node.
///
/// `split` hands out independent send and receive halves so a full-duplex
/// exchange can write from one thread while reading on another.
pub trait Link: Send {
    /// The send and receive halves.
    fn split(&mut self) -> (&mut (dyn Write + Send), &mut (dyn Read + Send));

    /// Sends all of `buf` and flushes.
    fn send_all(&mut self, buf: &[u8]) -> io::Result<()> {
        let (w, _) = self.split();
        w.write_all(buf)?;
        w.flush()
    }

    /// Receives exactly `buf.len()` bytes.
    fn recv_exact(&mut self, buf: &mut [u8]) -> io::Result<()> {
        let (_, r) = self.split();
        r.read_exact(buf)
    }
}

/// [`Link`] over a TCP connection (loopback, LAN, or an SSH port forward).
pub struct TcpLink {
    tx: TcpStream,
    rx: TcpStream,
}

impl TcpLink {
    /// Wraps a connected stream (sets `TCP_NODELAY`).
    pub fn from_stream(s: TcpStream) -> io::Result<Self> {
        s.set_nodelay(true)?;
        let rx = s.try_clone()?;
        Ok(TcpLink { tx: s, rx })
    }

    /// Accepts one connection on `listener`.
    pub fn accept(listener: &TcpListener) -> io::Result<Self> {
        let (s, _) = listener.accept()?;
        Self::from_stream(s)
    }

    /// Connects to `addr`, retrying until `timeout` (the peer may not be
    /// listening yet).
    pub fn connect<A: ToSocketAddrs + Clone>(addr: A, timeout: Duration) -> io::Result<Self> {
        let t0 = Instant::now();
        loop {
            match TcpStream::connect(addr.clone()) {
                Ok(s) => return Self::from_stream(s),
                Err(e) if t0.elapsed() >= timeout => return Err(e),
                Err(_) => std::thread::sleep(Duration::from_millis(200)),
            }
        }
    }
}

impl Link for TcpLink {
    fn split(&mut self) -> (&mut (dyn Write + Send), &mut (dyn Read + Send)) {
        (&mut self.tx, &mut self.rx)
    }
}

/// [`Link`] over any pair of byte streams: a child process's stdout/stdin
/// (the parent side of `ssh host cmd`), or this process's own stdin/stdout
/// (the remote side). This is how the two machines talk when the only path
/// between them is an SSH session without port forwarding.
pub struct PipeLink<R, W> {
    r: R,
    w: W,
}

impl<R: Read + Send, W: Write + Send> PipeLink<R, W> {
    /// Reads from `r`, writes to `w`.
    pub fn new(r: R, w: W) -> Self {
        PipeLink { r, w }
    }
}

impl<R: Read + Send, W: Write + Send> Link for PipeLink<R, W> {
    fn split(&mut self) -> (&mut (dyn Write + Send), &mut (dyn Read + Send)) {
        (&mut self.w, &mut self.r)
    }
}

struct ChanWriter(SyncSender<Vec<u8>>);

impl Write for ChanWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0
            .send(buf.to_vec())
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "peer dropped"))?;
        Ok(buf.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

struct ChanReader {
    rx: Receiver<Vec<u8>>,
    buf: Vec<u8>,
    pos: usize,
}

impl Read for ChanReader {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        if out.is_empty() {
            return Ok(0);
        }
        while self.pos == self.buf.len() {
            match self.rx.recv() {
                Ok(v) => {
                    self.buf = v;
                    self.pos = 0;
                }
                Err(_) => return Ok(0),
            }
        }
        let k = out.len().min(self.buf.len() - self.pos);
        out[..k].copy_from_slice(&self.buf[self.pos..self.pos + k]);
        self.pos += k;
        Ok(k)
    }
}

/// In-process [`Link`] (bounded channels); see [`chan_pair`].
pub struct ChanLink {
    tx: ChanWriter,
    rx: ChanReader,
}

impl Link for ChanLink {
    fn split(&mut self) -> (&mut (dyn Write + Send), &mut (dyn Read + Send)) {
        (&mut self.tx, &mut self.rx)
    }
}

/// Two connected in-process links (for tests and single-process runs).
pub fn chan_pair() -> (ChanLink, ChanLink) {
    let (ta, rb) = sync_channel(16);
    let (tb, ra) = sync_channel(16);
    let mk = |tx, rx| ChanLink {
        tx: ChanWriter(tx),
        rx: ChanReader {
            rx,
            buf: Vec::new(),
            pos: 0,
        },
    };
    (mk(ta, ra), mk(tb, rb))
}

// ---------------------------------------------------------------------------
// Gate roles and per-rank specialisation
// ---------------------------------------------------------------------------

/// Qubits of `g` that must be local for the distributed executor: the
/// non-diagonal ("target") qubits. Diagonal gates (`Z`, `S`, `T`, `Rz`,
/// `Phase`, `Cz`, `CPhase`, ...) need none; `Cnot`/`Ccx` need only the target.
pub fn needs_local(g: &Gate) -> Vec<usize> {
    if g.diagonal_1q().is_some() {
        return Vec::new();
    }
    match *g {
        Gate::Cz(..) | Gate::CPhase(..) => Vec::new(),
        Gate::Cnot(_, t) | Gate::Ccx(_, _, t) => vec![t],
        _ => g.qubits(),
    }
}

/// Specialises a gate in physical coordinates for rank `r` of a layout with
/// `l` local qubits: global qubits (`>= l`) take the rank's bit value.
/// Returns the residual local gate (if any) and a scalar factor for the rank.
///
/// Panics (debug) if a global qubit appears in a non-diagonal role.
pub fn specialize(g: &Gate, l: usize, r: usize) -> (Option<Gate>, Complex64) {
    use Gate::*;
    let one = Complex64::new(1.0, 0.0);
    let glob = |q: usize| q >= l;
    let bit = |q: usize| (r >> (q - l)) & 1 == 1;
    if let Some((d0, d1)) = g.diagonal_1q() {
        let q = g.qubits()[0];
        if glob(q) {
            return (None, if bit(q) { d1 } else { d0 });
        }
        return (Some(*g), one);
    }
    match *g {
        Cz(a, b) | CPhase(a, b, _) => {
            let ph = match *g {
                CPhase(_, _, t) => Complex64::from_polar(1.0, t),
                _ => Complex64::new(-1.0, 0.0),
            };
            match (glob(a), glob(b)) {
                (false, false) => (Some(*g), one),
                (true, true) => (None, if bit(a) && bit(b) { ph } else { one }),
                _ => {
                    let (gq, lq) = if glob(a) { (a, b) } else { (b, a) };
                    if !bit(gq) {
                        (None, one)
                    } else {
                        let res = match *g {
                            CPhase(_, _, t) => Phase(lq, t),
                            _ => Z(lq),
                        };
                        (Some(res), one)
                    }
                }
            }
        }
        Cnot(c, t) if glob(c) => {
            debug_assert!(!glob(t));
            (if bit(c) { Some(X(t)) } else { None }, one)
        }
        Ccx(a, b, t) if glob(a) || glob(b) => {
            debug_assert!(!glob(t));
            let res = match (glob(a), glob(b)) {
                (true, true) => (bit(a) && bit(b)).then_some(X(t)),
                (true, false) => bit(a).then_some(Cnot(b, t)),
                _ => bit(b).then_some(Cnot(a, t)),
            };
            (res, one)
        }
        _ => {
            debug_assert!(
                g.qubits().iter().all(|&q| !glob(q)),
                "gate {g:?} has a global qubit in a non-diagonal role (l={l})"
            );
            (Some(*g), one)
        }
    }
}

// ---------------------------------------------------------------------------
// Planner
// ---------------------------------------------------------------------------

/// One step of a distributed plan.
#[derive(Clone, Debug, PartialEq)]
pub enum DistStep {
    /// Gates in physical coordinates; global qubits (`>= local_bits`) appear
    /// only in diagonal / control roles and are specialised per rank.
    Local(Vec<Gate>),
    /// Exchange physical local qubit `local` with physical global qubit `global`.
    Swap { local: usize, global: usize },
    /// Physical swaps of local slots that *relocate* logical qubits (layout
    /// change, used to restore canonical order): applied as `Swap` gates and
    /// the logical->physical map follows them.
    Relabel(Vec<(usize, usize)>),
    /// A folded `Swap` gate: the logical qubits on physical slots `a` and `b`
    /// exchange labels; no amplitude moves.
    Rename { a: usize, b: usize },
}

/// Planner knobs.
#[derive(Clone, Debug)]
pub struct PlanOptions {
    /// Choose the initial qubit layout freely (valid because execution starts
    /// from a basis state): qubits whose first non-diagonal use comes last go
    /// to the global slots.
    pub free_initial_layout: bool,
    /// Fold `Swap` gates that touch a global qubit into the layout.
    pub fold_swaps: bool,
    /// End in canonical order (`v2p = identity`); costs extra swaps. Without
    /// it the final layout is recorded in the plan and [`DistState::gather`]
    /// / [`DistState::logical_index`] account for it.
    pub restore_order: bool,
}

impl Default for PlanOptions {
    fn default() -> Self {
        PlanOptions {
            free_initial_layout: true,
            fold_swaps: true,
            restore_order: false,
        }
    }
}

/// A distributed execution plan (identical on both nodes).
#[derive(Clone, Debug, PartialEq)]
pub struct DistPlan {
    /// Total qubits.
    pub n: usize,
    /// Local qubits per rank (`L`).
    pub local_bits: usize,
    /// Logical -> physical map the state must start in.
    pub initial_v2p: Vec<usize>,
    /// The steps.
    pub steps: Vec<DistStep>,
    /// Logical -> physical map after the plan.
    pub final_v2p: Vec<usize>,
    /// Global-qubit swap steps.
    pub swaps: usize,
    /// `Swap` gates folded into the layout.
    pub folded_swaps: usize,
    /// Local gate runs.
    pub runs: usize,
}

/// Plans a unitary circuit.
pub fn plan_circuit(
    circuit: &Circuit,
    local_bits: usize,
    opts: &PlanOptions,
) -> Result<DistPlan, SimError> {
    let mut gates = Vec::with_capacity(circuit.ops.len());
    for op in &circuit.ops {
        match op {
            Op::Gate(g) => gates.push(*g),
            _ => {
                return Err(SimError::NotSupported {
                    what: "distributed simulation supports unitary circuits only",
                })
            }
        }
    }
    plan_gates(&gates, circuit.num_qubits, local_bits, opts)
}

/// Plans a unitary gate list on `n` qubits with `local_bits` local qubits.
pub fn plan_gates(
    gates: &[Gate],
    n: usize,
    local_bits: usize,
    opts: &PlanOptions,
) -> Result<DistPlan, SimError> {
    let l = local_bits;
    if l > n || (l < 3 && l < n) {
        return Err(SimError::NotSupported {
            what: "distributed plan needs 3 <= local_bits <= n",
        });
    }
    for g in gates {
        check_gate(g, n)?;
    }
    let m = gates.len();
    let is_fold_candidate = |g: &Gate| opts.fold_swaps && matches!(g, Gate::Swap(..));

    // Wire dependencies.
    let mut last: Vec<Option<usize>> = vec![None; n];
    let mut succ: Vec<Vec<usize>> = vec![Vec::new(); m];
    let mut indeg = vec![0u32; m];
    for (i, g) in gates.iter().enumerate() {
        for q in g.qubits() {
            if let Some(p) = last[q] {
                if succ[p].last() != Some(&i) {
                    succ[p].push(i);
                    indeg[i] += 1;
                }
            }
            last[q] = Some(i);
        }
    }
    // Non-diagonal uses per logical qubit (Belady eviction + initial layout).
    let mut tuse: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (i, g) in gates.iter().enumerate() {
        if is_fold_candidate(g) {
            continue;
        }
        for q in needs_local(g) {
            tuse[q].push(i);
        }
    }
    let mut tptr = vec![0usize; n];
    let mut done = vec![false; m];

    // Initial layout.
    let mut v2p: Vec<usize> = (0..n).collect();
    if opts.free_initial_layout && l < n {
        let first = |q: usize| tuse[q].first().copied().unwrap_or(usize::MAX);
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| first(b).cmp(&first(a)).then(b.cmp(&a)));
        let mut globals: Vec<usize> = order[..n - l].to_vec();
        globals.sort_unstable();
        let locals: Vec<usize> = (0..n).filter(|q| !globals.contains(q)).collect();
        for (p, &q) in locals.iter().enumerate() {
            v2p[q] = p;
        }
        for (j, &q) in globals.iter().enumerate() {
            v2p[q] = l + j;
        }
    }
    let initial_v2p = v2p.clone();
    let mut p2v = vec![0usize; n];
    for (v, &p) in v2p.iter().enumerate() {
        p2v[p] = v;
    }

    let mut ready: Vec<usize> = (0..m).filter(|&i| indeg[i] == 0).collect();
    let mut steps: Vec<DistStep> = Vec::new();
    let mut run: Vec<Gate> = Vec::new();
    let (mut swaps, mut folded, mut runs) = (0usize, 0usize, 0usize);

    let flush = |steps: &mut Vec<DistStep>, run: &mut Vec<Gate>, runs: &mut usize| {
        if !run.is_empty() {
            steps.push(DistStep::Local(std::mem::take(run)));
            *runs += 1;
        }
    };

    loop {
        // Drain everything executable in the current layout.
        loop {
            let mut progressed = false;
            let mut i = 0;
            while i < ready.len() {
                let gi = ready[i];
                let g = &gates[gi];
                let fold = match *g {
                    Gate::Swap(a, b) => opts.fold_swaps && (v2p[a] >= l || v2p[b] >= l),
                    _ => false,
                };
                if fold {
                    if let Gate::Swap(a, b) = *g {
                        let (pa, pb) = (v2p[a], v2p[b]);
                        v2p[a] = pb;
                        v2p[b] = pa;
                        p2v[pa] = b;
                        p2v[pb] = a;
                        // Labels only: the pending run is in physical
                        // coordinates, so it need not be flushed.
                        steps.push(DistStep::Rename { a: pa, b: pb });
                    }
                    folded += 1;
                } else if needs_local(g).iter().all(|&q| v2p[q] < l) {
                    run.push(remap_gate(g, &v2p));
                } else {
                    i += 1;
                    continue;
                }
                done[gi] = true;
                ready.swap_remove(i);
                progressed = true;
                for &s in &succ[gi] {
                    indeg[s] -= 1;
                    if indeg[s] == 0 {
                        ready.push(s);
                    }
                }
            }
            if !progressed {
                break;
            }
        }
        if ready.is_empty() {
            break;
        }
        flush(&mut steps, &mut run, &mut runs);

        // Stuck: bring in the global qubits of the best ready gate.
        let mut next_use = vec![usize::MAX; n];
        for q in 0..n {
            while tptr[q] < tuse[q].len() && done[tuse[q][tptr[q]]] {
                tptr[q] += 1;
            }
            if let Some(&i) = tuse[q].get(tptr[q]) {
                next_use[q] = i;
            }
        }
        let mut best: Option<(f64, Vec<usize>, Vec<usize>)> = None;
        for &gi in &ready {
            let nl = needs_local(&gates[gi]);
            let needed: Vec<usize> = nl.iter().copied().filter(|&q| v2p[q] >= l).collect();
            if needed.is_empty() {
                continue;
            }
            let mut pool: Vec<usize> = (0..l).map(|p| p2v[p]).filter(|q| !nl.contains(q)).collect();
            pool.sort_by(|&a, &b| next_use[b].cmp(&next_use[a]).then(a.cmp(&b)));
            if pool.len() < needed.len() {
                continue;
            }
            let evict = pool[..needed.len()].to_vec();
            let mut hyp = v2p.clone();
            for (&qi, &qo) in needed.iter().zip(&evict) {
                hyp.swap(qi, qo);
            }
            let hits = ready
                .iter()
                .filter(|&&o| needs_local(&gates[o]).iter().all(|&q| hyp[q] < l))
                .count();
            let score = (hits as f64 + 1.0) / needed.len() as f64;
            if best.as_ref().map_or(true, |b| score > b.0) {
                best = Some((score, needed, evict));
            }
        }
        let Some((_, needed, evict)) = best else {
            return Err(SimError::NotSupported {
                what: "distributed planner: no gate can be made local",
            });
        };
        for (&qi, &qo) in needed.iter().zip(&evict) {
            let (pl, pg) = (v2p[qo], v2p[qi]);
            steps.push(DistStep::Swap {
                local: pl,
                global: pg,
            });
            swaps += 1;
            v2p[qi] = pl;
            v2p[qo] = pg;
            p2v[pl] = qi;
            p2v[pg] = qo;
        }
    }
    flush(&mut steps, &mut run, &mut runs);

    if opts.restore_order {
        fn do_swap(
            steps: &mut Vec<DistStep>,
            v2p: &mut [usize],
            p2v: &mut [usize],
            swaps: &mut usize,
            pl: usize,
            pg: usize,
        ) {
            steps.push(DistStep::Swap {
                local: pl,
                global: pg,
            });
            *swaps += 1;
            let (vl, vg) = (p2v[pl], p2v[pg]);
            p2v[pl] = vg;
            p2v[pg] = vl;
            v2p[vg] = pl;
            v2p[vl] = pg;
        }
        for g in l..n {
            if p2v[g] == g {
                continue;
            }
            let q = v2p[g];
            if q < l {
                do_swap(&mut steps, &mut v2p, &mut p2v, &mut swaps, q, g);
            } else {
                do_swap(&mut steps, &mut v2p, &mut p2v, &mut swaps, 0, q);
                do_swap(&mut steps, &mut v2p, &mut p2v, &mut swaps, 0, g);
            }
        }
        let mut local = Vec::new();
        for i in 0..l {
            while p2v[i] != i {
                let j = p2v[i];
                local.push((i, j));
                let (vi, vj) = (p2v[i], p2v[j]);
                p2v[i] = vj;
                p2v[j] = vi;
                v2p[vj] = i;
                v2p[vi] = j;
            }
        }
        if !local.is_empty() {
            steps.push(DistStep::Relabel(local));
        }
        debug_assert!((0..n).all(|i| v2p[i] == i));
    }

    Ok(DistPlan {
        n,
        local_bits: l,
        initial_v2p,
        steps,
        final_v2p: v2p,
        swaps,
        folded_swaps: folded,
        runs,
    })
}

/// FNV-1a fingerprint of a plan, the rank ownership table and the element
/// size: both nodes must agree on it before exchanging amplitudes.
pub fn fingerprint(plan: &DistPlan, owner: &[u8], elem_bytes: usize) -> u64 {
    let s = format!("{plan:?}|{owner:?}|{elem_bytes}");
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

const MAGIC: u32 = 0x5153_4456; // "QSDV"
const VERSION: u32 = 1;

/// Verifies that the peer is the other node and runs the same plan.
pub fn handshake(link: &mut dyn Link, node: u8, fp: u64) -> io::Result<()> {
    let mut msg = [0u8; 24];
    msg[0..4].copy_from_slice(&MAGIC.to_le_bytes());
    msg[4..8].copy_from_slice(&VERSION.to_le_bytes());
    msg[8] = node;
    msg[16..24].copy_from_slice(&fp.to_le_bytes());
    link.send_all(&msg)?;
    let mut got = [0u8; 24];
    link.recv_exact(&mut got)?;
    let bad = |what: &str| io::Error::new(io::ErrorKind::InvalidData, what.to_string());
    if got[0..4] != MAGIC.to_le_bytes() || got[4..8] != VERSION.to_le_bytes() {
        return Err(bad("handshake: bad magic/version"));
    }
    if got[8] != 1 - node {
        return Err(bad("handshake: both ends claim the same node id"));
    }
    if got[16..24] != fp.to_le_bytes() {
        return Err(bad("handshake: plan/layout fingerprint mismatch"));
    }
    Ok(())
}

/// Exchanges one byte each way (a barrier between the two nodes).
pub fn barrier(link: &mut dyn Link) -> io::Result<()> {
    link.send_all(&[0xB5])?;
    let mut b = [0u8; 1];
    link.recv_exact(&mut b)?;
    if b[0] != 0xB5 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "barrier: bad byte",
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Executor
// ---------------------------------------------------------------------------

/// Executor knobs.
#[derive(Clone, Debug)]
pub struct DistConfig {
    /// Kernel configuration for local gate runs.
    pub block: BlockConfig,
    /// Amplitudes per network message during a cross-node exchange.
    pub msg_amps: usize,
}

impl Default for DistConfig {
    fn default() -> Self {
        DistConfig {
            block: BlockConfig::default(),
            msg_amps: 1 << 19,
        }
    }
}

/// Execution counters (per node).
#[derive(Clone, Debug, Default)]
pub struct DistStats {
    /// Local gate runs executed.
    pub runs: usize,
    /// Global-qubit swap steps.
    pub swaps: usize,
    /// Rank pairs exchanged over the link (summed over swap steps).
    pub cross_pairs: usize,
    /// Rank pairs swapped in memory (both ranks on this node).
    pub local_pairs: usize,
    /// Payload bytes sent / received over the link.
    pub bytes_sent: u64,
    /// Payload bytes received.
    pub bytes_recv: u64,
    /// Time in local gate runs.
    pub compute: Duration,
    /// Time in cross-node exchanges.
    pub exchange: Duration,
    /// Time in same-node rank-pair swaps.
    pub local_swap: Duration,
}

#[derive(Clone, Copy)]
struct SendPtr<T>(*mut T);
// SAFETY: used only for disjoint per-block regions, synchronised by an atomic counter.
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}

fn as_bytes<T: Real>(s: &[Complex<T>]) -> &[u8] {
    // SAFETY: Complex<f32/f64> is two plain floats, no padding; u8 has alignment 1.
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s)) }
}

fn as_bytes_mut<T: Real>(s: &mut [Complex<T>]) -> &mut [u8] {
    // SAFETY: as above; every byte pattern is a valid float.
    unsafe { std::slice::from_raw_parts_mut(s.as_mut_ptr() as *mut u8, std::mem::size_of_val(s)) }
}

/// Physical index (within a rank) of element `t` of the half with local bit `lq == bit`.
#[inline]
fn half_index(t: usize, lq: usize, bit: usize) -> usize {
    ((t >> lq) << (lq + 1)) | (bit << lq) | (t & ((1usize << lq) - 1))
}

/// Copies half-elements `t0..t0+out.len()` into `out`.
///
/// # Safety
/// `base` must point to a rank of `2^(lq+1+..)` elements covering the range,
/// and no other thread may write that range concurrently.
unsafe fn pack_half<T: Copy>(base: *const T, lq: usize, bit: usize, t0: usize, out: &mut [T]) {
    let run = 1usize << lq;
    if run >= out.len() {
        std::ptr::copy_nonoverlapping(
            base.add(half_index(t0, lq, bit)),
            out.as_mut_ptr(),
            out.len(),
        );
    } else {
        for (k, o) in out.chunks_mut(run).enumerate() {
            let p = half_index(t0 + k * run, lq, bit);
            std::ptr::copy_nonoverlapping(base.add(p), o.as_mut_ptr(), run);
        }
    }
}

/// Inverse of [`pack_half`].
///
/// # Safety
/// As for [`pack_half`], with exclusive access to the range.
unsafe fn unpack_half<T: Copy>(base: *mut T, lq: usize, bit: usize, t0: usize, src: &[T]) {
    let run = 1usize << lq;
    if run >= src.len() {
        std::ptr::copy_nonoverlapping(src.as_ptr(), base.add(half_index(t0, lq, bit)), src.len());
    } else {
        for (k, s) in src.chunks(run).enumerate() {
            let p = half_index(t0 + k * run, lq, bit);
            std::ptr::copy_nonoverlapping(s.as_ptr(), base.add(p), run);
        }
    }
}

/// Swaps the `lq = 1` half of `a` with the `lq = 0` half of `b` (same node).
fn swap_halves<T: Send>(a: &mut [T], b: &mut [T], lq: usize) {
    let run = 1usize << lq;
    let chunk = (2 * run).max(1 << 14).min(a.len());
    a.par_chunks_mut(chunk)
        .zip(b.par_chunks_mut(chunk))
        .for_each(|(ca, cb)| {
            for (pa, pb) in ca.chunks_mut(2 * run).zip(cb.chunks_mut(2 * run)) {
                pa[run..].swap_with_slice(&mut pb[..run]);
            }
        });
}

/// Full-duplex streaming exchange: sends the `lq == bit` half of `data` and
/// overwrites it with the peer's half, `blk` amplitudes per message. Block
/// `j` is overwritten only after it has been sent, so one block of staging
/// per direction suffices.
fn exchange_half<T: Real>(
    link: &mut dyn Link,
    data: &mut [Complex<T>],
    lq: usize,
    bit: usize,
    msg_amps: usize,
) -> io::Result<()> {
    let half = data.len() / 2;
    let blk = msg_amps.next_power_of_two().min(half).max(1);
    let nblk = half / blk;
    let ptr = SendPtr(data.as_mut_ptr());
    let sent = AtomicUsize::new(0);
    let failed = AtomicBool::new(false);
    let (w, r) = link.split();
    std::thread::scope(|s| {
        let (sent, failed) = (&sent, &failed);
        let sender = s.spawn(move || -> io::Result<()> {
            let p = ptr;
            let mut buf = vec![Complex::<T>::new(T::zero(), T::zero()); blk];
            let res = (|| {
                for j in 0..nblk {
                    // SAFETY: block j is not written by the receiver until `sent > j`.
                    unsafe { pack_half(p.0 as *const Complex<T>, lq, bit, j * blk, &mut buf) };
                    w.write_all(as_bytes(&buf))?;
                    sent.store(j + 1, Ordering::Release);
                }
                w.flush()
            })();
            if res.is_err() {
                failed.store(true, Ordering::Release);
            }
            res
        });
        let mut rbuf = vec![Complex::<T>::new(T::zero(), T::zero()); blk];
        let mut res = Ok(());
        'outer: for j in 0..nblk {
            if let Err(e) = r.read_exact(as_bytes_mut(&mut rbuf)) {
                res = Err(e);
                break;
            }
            while sent.load(Ordering::Acquire) <= j {
                if failed.load(Ordering::Acquire) {
                    break 'outer;
                }
                std::thread::sleep(Duration::from_micros(50));
            }
            // SAFETY: block j has been sent; the sender only touches blocks > j now.
            unsafe { unpack_half(ptr.0, lq, bit, j * blk, &rbuf) };
        }
        let sres = sender.join().expect("sender thread panicked");
        res.and(sres)
    })
}

/// This node's part of a distributed state vector.
pub struct DistState<T: Real = f64> {
    n: usize,
    l: usize,
    node: u8,
    owner: Vec<u8>,
    ranks: Vec<Option<Vec<Complex<T>>>>,
    v2p: Vec<usize>,
    cfg: DistConfig,
    stats: DistStats,
}

impl<T: Real> DistState<T> {
    /// Basis state `|x>` (logical index) in layout `v2p`, ranks owned per
    /// `owner[r] == node` (`owner.len() == 2^(n - l)`).
    pub fn new_basis(
        n: usize,
        l: usize,
        node: u8,
        owner: Vec<u8>,
        v2p: &[usize],
        x: usize,
        cfg: DistConfig,
    ) -> Self {
        assert!(l <= n && n < usize::BITS as usize);
        assert_eq!(
            owner.len(),
            1 << (n - l),
            "owner table must have 2^(n-l) entries"
        );
        assert!(node <= 1 && owner.iter().all(|&o| o <= 1));
        assert_eq!(v2p.len(), n);
        let zero = Complex::new(T::zero(), T::zero());
        let ranks: Vec<_> = owner
            .iter()
            .map(|&o| (o == node).then(|| vec![zero; 1 << l]))
            .collect();
        let mut st = DistState {
            n,
            l,
            node,
            owner,
            ranks,
            v2p: v2p.to_vec(),
            cfg,
            stats: DistStats::default(),
        };
        let mut phys = 0usize;
        for (v, &p) in v2p.iter().enumerate() {
            phys |= ((x >> v) & 1) << p;
        }
        let (r, off) = (phys >> l, phys & ((1 << l) - 1));
        if let Some(d) = &mut st.ranks[r] {
            d[off] = Complex::new(T::one(), T::zero());
        }
        st
    }

    /// Ground state in the plan's initial layout.
    pub fn for_plan(plan: &DistPlan, node: u8, owner: Vec<u8>, cfg: DistConfig) -> Self {
        Self::new_basis(
            plan.n,
            plan.local_bits,
            node,
            owner,
            &plan.initial_v2p,
            0,
            cfg,
        )
    }

    /// Number of qubits.
    pub fn num_qubits(&self) -> usize {
        self.n
    }
    /// Current logical -> physical map.
    pub fn layout(&self) -> &[usize] {
        &self.v2p
    }
    /// Counters.
    pub fn stats(&self) -> &DistStats {
        &self.stats
    }
    /// Ranks owned by this node with their data.
    pub fn owned(&self) -> impl Iterator<Item = (usize, &[Complex<T>])> {
        self.ranks
            .iter()
            .enumerate()
            .filter_map(|(r, d)| d.as_deref().map(|d| (r, d)))
    }
    /// Bytes of amplitude storage on this node.
    pub fn local_bytes(&self) -> usize {
        self.owned().count() * (1 << self.l) * std::mem::size_of::<Complex<T>>()
    }

    /// Logical basis index of amplitude `off` of rank `r` in the current layout.
    pub fn logical_index(&self, r: usize, off: usize) -> usize {
        let phys = (r << self.l) | off;
        let mut x = 0;
        for (v, &p) in self.v2p.iter().enumerate() {
            x |= ((phys >> p) & 1) << v;
        }
        x
    }

    /// Runs a plan (the state must be in `plan.initial_v2p`).
    pub fn run(&mut self, plan: &DistPlan, link: &mut dyn Link) -> io::Result<()> {
        if plan.n != self.n || plan.local_bits != self.l || plan.initial_v2p != self.v2p {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "plan does not match the state's shape or layout",
            ));
        }
        for step in &plan.steps {
            match step {
                DistStep::Local(gates) => self.apply_local(gates),
                DistStep::Swap { local, global } => self.swap_qubits(*local, *global, link)?,
                DistStep::Relabel(pairs) => self.relabel(pairs),
                DistStep::Rename { a, b } => self.rename(*a, *b),
            }
        }
        debug_assert_eq!(self.v2p, plan.final_v2p);
        Ok(())
    }

    /// Applies gates in physical coordinates to every owned rank (global
    /// qubits only in diagonal / control roles).
    pub fn apply_local(&mut self, gates: &[Gate]) {
        let t0 = Instant::now();
        let l = self.l;
        let mut cache: Vec<(Vec<Gate>, Option<BlockedChunkExecutor<T>>)> = Vec::new();
        for (r, slot) in self.ranks.iter_mut().enumerate() {
            let Some(data) = slot else { continue };
            let mut spec = Vec::with_capacity(gates.len());
            let mut scalar = Complex64::new(1.0, 0.0);
            for g in gates {
                let (rg, s) = specialize(g, l, r);
                if let Some(rg) = rg {
                    spec.push(rg);
                }
                scalar *= s;
            }
            let idx = match cache.iter().position(|(k, _)| *k == spec) {
                Some(i) => i,
                None => {
                    let ex = if spec.is_empty() {
                        None
                    } else {
                        Some(
                            BlockedChunkExecutor::<T>::new(&spec, l, &self.cfg.block)
                                .expect("specialised gates are local"),
                        )
                    };
                    cache.push((spec, ex));
                    cache.len() - 1
                }
            };
            if let Some(ex) = &cache[idx].1 {
                ex.apply_to_chunk(data);
            }
            if (scalar - Complex64::new(1.0, 0.0)).norm() > 0.0 {
                let s = Complex::new(T::from_f64(scalar.re), T::from_f64(scalar.im));
                data.par_iter_mut().for_each(|a| *a = *a * s);
            }
        }
        self.stats.runs += 1;
        self.stats.compute += t0.elapsed();
    }

    /// Exchanges the logical labels of physical slots `a` and `b` (applies a
    /// logical `Swap` gate without moving data).
    pub fn rename(&mut self, a: usize, b: usize) {
        let va = self.v2p.iter().position(|&p| p == a).unwrap();
        let vb = self.v2p.iter().position(|&p| p == b).unwrap();
        self.v2p[va] = b;
        self.v2p[vb] = a;
    }

    /// Swaps local physical slots pairwise and moves the logical qubits with
    /// the data (a layout change; the logical state is unchanged).
    pub fn relabel(&mut self, pairs: &[(usize, usize)]) {
        let gates: Vec<Gate> = pairs.iter().map(|&(a, b)| Gate::Swap(a, b)).collect();
        self.apply_local(&gates);
        for &(a, b) in pairs {
            let va = self.v2p.iter().position(|&p| p == a).unwrap();
            let vb = self.v2p.iter().position(|&p| p == b).unwrap();
            self.v2p[va] = b;
            self.v2p[vb] = a;
        }
    }

    /// Exchanges physical local qubit `lq` with physical global qubit `gq`.
    pub fn swap_qubits(&mut self, lq: usize, gq: usize, link: &mut dyn Link) -> io::Result<()> {
        assert!(lq < self.l && gq >= self.l && gq < self.n);
        let gb = gq - self.l;
        let me = self.node;
        let esz = std::mem::size_of::<Complex<T>>() as u64;
        let half_bytes = (1u64 << (self.l - 1)) * esz;
        for r0 in 0..self.ranks.len() {
            if r0 >> gb & 1 == 1 {
                continue;
            }
            let r1 = r0 | (1 << gb);
            match (self.owner[r0] == me, self.owner[r1] == me) {
                (true, true) => {
                    let t = Instant::now();
                    let (lo, hi) = self.ranks.split_at_mut(r1);
                    let a = lo[r0].as_mut().unwrap();
                    let b = hi[0].as_mut().unwrap();
                    swap_halves(a, b, lq);
                    self.stats.local_pairs += 1;
                    self.stats.local_swap += t.elapsed();
                }
                (mine0, mine1) if mine0 || mine1 => {
                    let t = Instant::now();
                    let (r, bit) = if mine0 { (r0, 1) } else { (r1, 0) };
                    let mut hdr = [0u8; 16];
                    hdr[0..4].copy_from_slice(&(lq as u32).to_le_bytes());
                    hdr[4..8].copy_from_slice(&(gq as u32).to_le_bytes());
                    hdr[8..16].copy_from_slice(&(r0 as u64).to_le_bytes());
                    link.send_all(&hdr)?;
                    let mut got = [0u8; 16];
                    link.recv_exact(&mut got)?;
                    if got != hdr {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "swap: peer is at a different step",
                        ));
                    }
                    let data = self.ranks[r].as_mut().unwrap();
                    exchange_half(link, data, lq, bit, self.cfg.msg_amps)?;
                    self.stats.cross_pairs += 1;
                    self.stats.bytes_sent += half_bytes;
                    self.stats.bytes_recv += half_bytes;
                    self.stats.exchange += t.elapsed();
                }
                _ => {}
            }
        }
        let vl = self.v2p.iter().position(|&p| p == lq).unwrap();
        let vg = self.v2p.iter().position(|&p| p == gq).unwrap();
        self.v2p[vl] = gq;
        self.v2p[vg] = lq;
        self.stats.swaps += 1;
        Ok(())
    }

    /// Sum of `|amp|^2` over both nodes.
    pub fn norm_sqr(&self, link: &mut dyn Link) -> io::Result<f64> {
        let mine: f64 = self
            .owned()
            .map(|(_, d)| d.par_iter().map(|a| a.norm_sqr().to_f64()).sum::<f64>())
            .sum();
        link.send_all(&mine.to_le_bytes())?;
        let mut b = [0u8; 8];
        link.recv_exact(&mut b)?;
        Ok(mine + f64::from_le_bytes(b))
    }

    /// Gathers the full state on node 0 in canonical (logical) order; node 1
    /// returns `None`. Only for states that fit in node 0's RAM.
    pub fn gather(&self, link: &mut dyn Link) -> io::Result<Option<Vec<Complex<T>>>> {
        let rs = 1usize << self.l;
        if self.node == 1 {
            for (_, d) in self.owned() {
                link.send_all(as_bytes(d))?;
            }
            return Ok(None);
        }
        let zero = Complex::new(T::zero(), T::zero());
        let mut phys = vec![zero; 1 << self.n];
        for (r, slot) in self.ranks.iter().enumerate() {
            let dst = &mut phys[r * rs..(r + 1) * rs];
            match slot {
                Some(d) => dst.copy_from_slice(d),
                None => link.recv_exact(as_bytes_mut(dst))?,
            }
        }
        let mut out = vec![zero; 1 << self.n];
        permute_bits(&phys, &mut out, &self.v2p);
        Ok(Some(out))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_index_partitions_rank() {
        for lq in 0..5 {
            let mut seen = [false; 32];
            for bit in 0..2 {
                for t in 0..16 {
                    let p = half_index(t, lq, bit);
                    assert_eq!((p >> lq) & 1, bit);
                    assert!(!seen[p]);
                    seen[p] = true;
                }
            }
        }
    }

    #[test]
    fn chan_link_roundtrip() {
        let (mut a, mut b) = chan_pair();
        let h = std::thread::spawn(move || {
            let mut buf = [0u8; 5];
            b.recv_exact(&mut buf).unwrap();
            b.send_all(&buf).unwrap();
        });
        a.send_all(b"hello").unwrap();
        let mut got = [0u8; 5];
        a.recv_exact(&mut got).unwrap();
        assert_eq!(&got, b"hello");
        h.join().unwrap();
    }
}
