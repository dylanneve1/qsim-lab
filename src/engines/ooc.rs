//! Out-of-core (disk-backed) state vector simulation for large qubit counts.
//!
//! When the state vector exceeds physical RAM (e.g. 26+ qubits in f32, 25+ in f64),
//! the in-RAM blocked executor cannot allocate the required $2^n$ amplitudes.
//! This module removes that memory ceiling by storing the state vector in a backing
//! file, partitioned into chunks of $2^c$ amplitudes (where $c$ is the number of local
//! qubits, typically chosen such that one or two chunks fit comfortably in RAM).
//!
//! # Execution Architecture
//!
//! 1. **Local vs Global Qubits**:
//!    The $n$ qubits are partitioned into $c$ *local* physical qubits ($0 \le p < c$)
//!    and $n - c$ *global* physical qubits ($c \le p < n$).
//!    An amplitude index $i \in [0, 2^n - 1]$ has local bits $i \& ((1 \ll c) - 1)$
//!    (offset within a chunk) and global bits $i \gg c$ (chunk index $k$).
//!
//! 2. **Local Gate Runs (Streaming Passes)**:
//!    Any sequence of gates acting strictly on physical qubits $< c$ acts entirely
//!    within each chunk independently. The file is streamed sequentially chunk by chunk:
//!    each chunk is read into an in-RAM buffer, transformed using cache-blocked kernels
//!    ([`crate::engines::blocked::BlockedChunkExecutor`]), and written back to disk.
//!    One full sequential streaming pass over the file applies the entire run of local gates.
//!
//! 3. **Global-Local Qubit Swap Passes**:
//!    To execute a gate on a global qubit $g \ge c$, that qubit must be brought into
//!    the local set by swapping it with a local qubit $l < c$. A qubit swap exchanges
//!    bit $l$ and bit $g$ across all amplitudes. For every chunk pair $(C_0, C_1)$ where
//!    $C_1 = C_0 \mid (1 \ll (g - c))$, elements with bit $l = 1$ in $C_0$ are swapped
//!    with elements with bit $l = 0$ in $C_1$. This is executed as vectorized slice swaps
//!    between two in-RAM chunk buffers, touching each chunk exactly once (1 file pass).
//!
//! 4. **Greedy Swap Scheduler with Lookahead (DAG-Driven)**:
//!    Gates are scheduled from the dependency DAG ([`crate::dag::Dag`]). The scheduler
//!    greedily executes all ready gates that touch only current local qubits, extending
//!    local stages as far as possible. When no ready gate can proceed locally, the
//!    scheduler looks ahead at the DAG's front layer and uses Belady's MIN algorithm
//!    (optimal cache replacement) to evict the local qubit whose next use is farthest
//!    in the future, minimizing total swap passes over the file.
//!
//! 5. **Exactness Guarantee**:
//!    Swap passes are exact memory bit-permutations with zero arithmetic error. Local
//!    gate runs use the exact same cache-blocked unitary kernels as the in-RAM executor.
//!    Final permutations are restored to canonical identity order, yielding bit-identical
//!    or floating-point identical amplitudes ($\le 10^{-6}$ for f32, $\le 10^{-12}$ for f64).

use crate::circuit::{check_gate, Circuit, Op, SimError};
use crate::dag::{Dag, NodeId};
use crate::engines::blocked::{BlockConfig, BlockedChunkExecutor};
use crate::engines::ooc_window::{
    permute_bits, schedule_window_gates, WindowOptions, WindowPass, WindowPlan,
};
use crate::engines::statevector::Real;
use crate::gate::Gate;
use num_complex::Complex;
use num_traits::Zero;
use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::io;
#[cfg(unix)]
use std::os::unix::fs::FileExt;
#[cfg(windows)]
use win_file_ext::FileExt;

/// Positional file I/O on Windows with the Unix `FileExt` names (so the
/// crate builds there; the out-of-core engine was only tuned on Unix).
#[cfg(windows)]
mod win_file_ext {
    use std::fs::File;
    use std::io;
    use std::os::windows::fs::FileExt as WinExt;

    pub trait FileExt {
        fn read_exact_at(&self, buf: &mut [u8], offset: u64) -> io::Result<()>;
        fn write_all_at(&self, buf: &[u8], offset: u64) -> io::Result<()>;
    }

    impl FileExt for File {
        fn read_exact_at(&self, mut buf: &mut [u8], mut offset: u64) -> io::Result<()> {
            while !buf.is_empty() {
                match self.seek_read(buf, offset) {
                    Ok(0) => return Err(io::ErrorKind::UnexpectedEof.into()),
                    Ok(n) => {
                        buf = &mut buf[n..];
                        offset += n as u64;
                    }
                    Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                    Err(e) => return Err(e),
                }
            }
            Ok(())
        }
        fn write_all_at(&self, mut buf: &[u8], mut offset: u64) -> io::Result<()> {
            while !buf.is_empty() {
                match self.seek_write(buf, offset) {
                    Ok(0) => return Err(io::ErrorKind::WriteZero.into()),
                    Ok(n) => {
                        buf = &buf[n..];
                        offset += n as u64;
                    }
                    Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                    Err(e) => return Err(e),
                }
            }
            Ok(())
        }
    }
}
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::sync_channel;
use std::time::{Duration, Instant};

/// Casts a slice of `Complex<T>` to a byte slice for zero-copy file I/O.
#[inline]
fn as_u8_slice<T: Real>(slice: &[Complex<T>]) -> &[u8] {
    // SAFETY: Complex<T> (where T is f32 or f64) consists purely of two IEEE 754 floats
    // with standard layout and no padding. Casting &[Complex<T>] to &[u8] of length
    // len * size_of::<Complex<T>>() is safe because u8 has an alignment of 1 (<= align_of::<T>())
    // and every byte sequence of IEEE floats is valid memory.
    let len = std::mem::size_of_val(slice);
    unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, len) }
}

/// Casts a mutable slice of `Complex<T>` to a mutable byte slice for zero-copy file I/O.
#[inline]
fn as_u8_slice_mut<T: Real>(slice: &mut [Complex<T>]) -> &mut [u8] {
    // SAFETY: Same invariant as as_u8_slice; mutable borrow guarantees exclusive access.
    let len = std::mem::size_of_val(slice);
    unsafe { std::slice::from_raw_parts_mut(slice.as_mut_ptr() as *mut u8, len) }
}

/// Remaps a gate from virtual qubit IDs to physical qubit IDs.
pub fn remap_gate(g: &Gate, v2p: &[usize]) -> Gate {
    match *g {
        Gate::I(q) => Gate::I(v2p[q]),
        Gate::X(q) => Gate::X(v2p[q]),
        Gate::Y(q) => Gate::Y(v2p[q]),
        Gate::Z(q) => Gate::Z(v2p[q]),
        Gate::H(q) => Gate::H(v2p[q]),
        Gate::S(q) => Gate::S(v2p[q]),
        Gate::Sdg(q) => Gate::Sdg(v2p[q]),
        Gate::T(q) => Gate::T(v2p[q]),
        Gate::Tdg(q) => Gate::Tdg(v2p[q]),
        Gate::Sx(q) => Gate::Sx(v2p[q]),
        Gate::Sxdg(q) => Gate::Sxdg(v2p[q]),
        Gate::Rx(q, th) => Gate::Rx(v2p[q], th),
        Gate::Ry(q, th) => Gate::Ry(v2p[q], th),
        Gate::Rz(q, th) => Gate::Rz(v2p[q], th),
        Gate::Phase(q, th) => Gate::Phase(v2p[q], th),
        Gate::U(q, th, ph, la) => Gate::U(v2p[q], th, ph, la),
        Gate::Cnot(c, t) => Gate::Cnot(v2p[c], v2p[t]),
        Gate::Cz(c, t) => Gate::Cz(v2p[c], v2p[t]),
        Gate::Swap(a, b) => Gate::Swap(v2p[a], v2p[b]),
        Gate::ISwap(a, b) => Gate::ISwap(v2p[a], v2p[b]),
        Gate::ISwapdg(a, b) => Gate::ISwapdg(v2p[a], v2p[b]),
        Gate::CPhase(a, b, th) => Gate::CPhase(v2p[a], v2p[b], th),
        Gate::Ccx(a, b, t) => Gate::Ccx(v2p[a], v2p[b], v2p[t]),
    }
}

/// One step in an out-of-core execution plan.
#[derive(Clone, Debug, PartialEq)]
pub enum OocStep {
    /// A run of local gates acting purely on physical qubits `< chunk_bits`.
    /// Streamed sequentially chunk by chunk over the state vector file.
    LocalRun(Vec<Gate>),
    /// A global<->local qubit swap exchanging physical qubit `local < chunk_bits`
    /// and `global >= chunk_bits`.
    Swap {
        /// Physical qubit below `chunk_bits` (inside a chunk).
        local: usize,
        /// Physical qubit at or above `chunk_bits` (selects the chunk).
        global: usize,
    },
}

/// Execution plan produced by the lookahead swap scheduler.
#[derive(Clone, Debug, PartialEq)]
pub struct OocPlan {
    /// Ordered steps of local runs and swap passes.
    pub steps: Vec<OocStep>,
    /// Number of local run passes.
    pub local_passes: usize,
    /// Number of swap passes.
    pub swap_passes: usize,
}

/// Performance and I/O metrics recorded during out-of-core execution.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct OocStats {
    /// Total passes over the entire state vector file.
    pub file_passes: usize,
    /// Number of local-qubit execution passes over chunks.
    pub local_passes: usize,
    /// Number of global<->local qubit swap passes.
    pub swap_passes: usize,
    /// Total bytes read from disk.
    pub bytes_read: u64,
    /// Total bytes written to disk.
    pub bytes_written: u64,
    /// Total wall-clock time spent in simulation.
    pub wall_time: Duration,
    /// Wall-clock time spent in file I/O (reads + writes).
    pub io_time: Duration,
    /// Wall-clock time spent in CPU compute (cache-blocked kernels + RAM swaps).
    pub compute_time: Duration,
    /// Part of `compute_time` spent permuting buffer bits (layout changes).
    pub perm_time: Duration,
    /// Time the compute thread spent blocked waiting for a read to finish or a
    /// free buffer (windowed scheduler with I/O overlap): the *exposed* I/O.
    pub stall_time: Duration,
    /// Total number of gates executed.
    pub num_gates: usize,
}

/// Configuration knobs for the out-of-core executor.
#[derive(Clone, Debug)]
pub struct OocConfig {
    /// Number of local qubits per chunk (chunk size = `2^chunk_bits` amplitudes).
    pub chunk_bits: usize,
    /// Directory for backing files. Defaults to `std::env::temp_dir()`.
    pub scratch_dir: Option<PathBuf>,
    /// Block configuration for the in-RAM blocked chunk executor.
    pub block_config: BlockConfig,
    /// Whether to restore canonical qubit order `(v2p[i] == i)` upon completion.
    pub restore_order: bool,
    /// Which scheduler plans the passes.
    pub scheduler: OocScheduler,
    /// Windowed scheduler: number of high qubits `k` gathered per pass (a pass
    /// holds `2^(chunk_bits + k)` amplitudes per buffer). Ignored by
    /// [`OocScheduler::Swap`].
    pub group_bits: usize,
    /// Windowed scheduler: overlap file reads/writes with compute using a
    /// reader and a writer thread (3 group buffers instead of 1).
    pub overlap_io: bool,
}

/// Pass scheduler selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OocScheduler {
    /// One pass per local run and one per global<->local swap
    /// ([`schedule_ooc`], the original scheduler).
    Swap,
    /// Windowed passes with fused layout changes
    /// ([`crate::engines::ooc_window::schedule_window`]).
    Window,
}

impl Default for OocConfig {
    fn default() -> Self {
        OocConfig {
            chunk_bits: 20, // 2^20 amplitudes = 8 MiB in f32, 16 MiB in f64
            scratch_dir: None,
            block_config: BlockConfig::default(),
            restore_order: true,
            scheduler: OocScheduler::Window,
            group_bits: 3,
            overlap_io: true,
        }
    }
}

/// Schedules a circuit into out-of-core steps using a DAG-driven greedy lookahead scheduler.
///
/// Uses Belady's MIN algorithm to evict the local qubit whose next appearance in the DAG
/// is farthest in the future, minimizing total swap passes over the backing file.
pub fn schedule_ooc(
    circuit: &Circuit,
    num_qubits: usize,
    chunk_bits: usize,
) -> Result<OocPlan, SimError> {
    if chunk_bits >= num_qubits {
        // All qubits fit in a single chunk: purely local execution, 0 swaps.
        let mut gates = Vec::with_capacity(circuit.ops.len());
        for op in &circuit.ops {
            match op {
                Op::Gate(g) => {
                    check_gate(g, num_qubits)?;
                    gates.push(*g);
                }
                _ => {
                    return Err(SimError::NotSupported {
                        what: "out-of-core simulation currently supports unitary circuits",
                    });
                }
            }
        }
        let steps = if gates.is_empty() {
            Vec::new()
        } else {
            vec![OocStep::LocalRun(gates)]
        };
        return Ok(OocPlan {
            local_passes: if steps.is_empty() { 0 } else { 1 },
            swap_passes: 0,
            steps,
        });
    }

    let dag = Dag::from_circuit(circuit).map_err(|e| match e {
        crate::dag::DagError::Sim(sim_err) => sim_err,
        _ => SimError::NotSupported {
            what: "circuit contains non-unitary operations incompatible with DAG",
        },
    })?;

    let mut v2p: Vec<usize> = (0..num_qubits).collect();
    let mut p2v: Vec<usize> = (0..num_qubits).collect();
    let mut local_set: HashSet<usize> = (0..chunk_bits).collect();

    // Node in-degrees (predecessor count) and live successors
    let node_count = dag.topo_order().len();
    let mut indeg = vec![
        0u32;
        dag.front_layer()
            .iter()
            .map(|&x| x as usize)
            .max()
            .unwrap_or(0)
            + node_count
            + 10
    ];
    let mut all_nodes = Vec::with_capacity(node_count);

    // Compute topological order and in-degrees
    let topo = dag.topo_order();
    let max_id = topo.iter().copied().max().unwrap_or(0) as usize;
    if indeg.len() <= max_id {
        indeg.resize(max_id + 1, 0);
    }

    for &id in &topo {
        all_nodes.push(id);
        let preds = dag.predecessors(id);
        indeg[id as usize] = preds.len() as u32;
    }

    // Ready queue initialized with nodes having in-degree 0
    let mut ready: Vec<NodeId> = dag.front_layer();

    let mut steps: Vec<OocStep> = Vec::new();
    let mut current_local_run: Vec<Gate> = Vec::new();
    let mut local_passes = 0;
    let mut swap_passes = 0;

    let flush_local_run =
        |steps: &mut Vec<OocStep>, current: &mut Vec<Gate>, passes: &mut usize| {
            if !current.is_empty() {
                steps.push(OocStep::LocalRun(std::mem::take(current)));
                *passes += 1;
            }
        };

    while !ready.is_empty() {
        // Phase 1: Greedily drain all ready gates whose qubits are all in local_set
        let mut progress = true;
        while progress {
            progress = false;
            let mut i = 0;
            while i < ready.len() {
                let id = ready[i];
                let qs = dag.qubits(id);
                let can_run = qs.iter().all(|&q| local_set.contains(&(q as usize)));
                if can_run {
                    ready.swap_remove(i);
                    progress = true;

                    // Lower / remap gate to physical qubits
                    if let Op::Gate(ref g) = *dag.op(id) {
                        current_local_run.push(remap_gate(g, &v2p));
                    }

                    // Decrement successors' in-degrees
                    for s in dag.successors(id) {
                        let d = &mut indeg[s as usize];
                        *d -= 1;
                        if *d == 0 {
                            ready.push(s);
                        }
                    }
                } else {
                    i += 1;
                }
            }
        }

        if ready.is_empty() {
            break;
        }

        // Phase 2: All remaining ready nodes require global qubits.
        // Lookahead with Belady's MIN algorithm to select the best node to enable.
        flush_local_run(&mut steps, &mut current_local_run, &mut local_passes);

        // Precompute next use of each qubit among remaining unfinished nodes
        let mut next_use: Vec<usize> = vec![usize::MAX; num_qubits];
        for (step_idx, &node_id) in topo.iter().enumerate() {
            if indeg[node_id as usize] > 0 || ready.contains(&node_id) {
                for &q in dag.qubits(node_id) {
                    let q = q as usize;
                    if next_use[q] == usize::MAX {
                        next_use[q] = step_idx;
                    }
                }
            }
        }

        // Evaluate candidate nodes in ready queue
        struct CandidateChoice {
            needed: Vec<usize>,
            evicted: Vec<usize>,
            score: f64,
        }

        let mut best_candidate: Option<CandidateChoice> = None;

        for &cand_id in &ready {
            let qs = dag.qubits(cand_id);
            let needed: Vec<usize> = qs
                .iter()
                .map(|&q| q as usize)
                .filter(|q| !local_set.contains(q))
                .collect();
            let k = needed.len();
            if k == 0 || k > chunk_bits {
                continue;
            }

            // Find best k qubits to evict from local_set \ qs using Belady's MIN
            let mut eviction_pool: Vec<usize> = local_set
                .iter()
                .copied()
                .filter(|q| !qs.iter().any(|&gq| gq as usize == *q))
                .collect();

            // Sort descending by next_use (furthest in future or never used)
            eviction_pool.sort_unstable_by(|&a, &b| next_use[b].cmp(&next_use[a]).then(a.cmp(&b)));

            if eviction_pool.len() < k {
                continue;
            }

            let evicted: Vec<usize> = eviction_pool[..k].to_vec();

            // Estimate benefit: how many ready gates can be satisfied by this new local set?
            let mut candidate_local = local_set.clone();
            for &e in &evicted {
                candidate_local.remove(&e);
            }
            for &n in &needed {
                candidate_local.insert(n);
            }

            let mut potential_hits = 0;
            for &other_id in &ready {
                if dag
                    .qubits(other_id)
                    .iter()
                    .all(|&q| candidate_local.contains(&(q as usize)))
                {
                    potential_hits += 1;
                }
            }

            let score = (potential_hits as f64 + 1.0) / (k as f64);

            if best_candidate.as_ref().is_none_or(|c| score > c.score) {
                best_candidate = Some(CandidateChoice {
                    needed,
                    evicted,
                    score,
                });
            }
        }

        let Some(choice) = best_candidate else {
            return Err(SimError::NotSupported {
                what: "out-of-core chunk too small for a gate (need chunk_bits >= gate arity)",
            });
        };

        // Schedule the chosen swaps
        for (&q_in, &q_out) in choice.needed.iter().zip(choice.evicted.iter()) {
            let p_local = v2p[q_out];
            let p_global = v2p[q_in];
            debug_assert!(p_local < chunk_bits);
            debug_assert!(p_global >= chunk_bits);

            steps.push(OocStep::Swap {
                local: p_local,
                global: p_global,
            });
            swap_passes += 1;

            // Update mappings
            v2p[q_in] = p_local;
            v2p[q_out] = p_global;
            p2v[p_local] = q_in;
            p2v[p_global] = q_out;

            local_set.remove(&q_out);
            local_set.insert(q_in);
        }
    }

    // Flush any trailing local gates
    flush_local_run(&mut steps, &mut current_local_run, &mut local_passes);

    // Phase 3: Canonical qubit restoration
    // Restore v2p[i] == i for all qubits so the state vector on disk is in canonical basis order.
    //
    // Global slots are fixed one at a time. For global slot `g` holding virtual qubit
    // `v != g`, virtual qubit `g` sits at physical slot `q = v2p[g]`:
    //  * `q` local: one swap(q, g) puts `g` home (and parks `v` in local slot `q`);
    //  * `q` global: shuttle through local slot 0: swap(0, q) then swap(0, g).
    // Earlier (already correct) global slots are never touched, since they hold their own
    // virtual qubit. Afterwards every local slot holds a local virtual qubit, and the
    // remaining permutation is done inside the chunk by in-RAM swap gates.
    let mut local_swaps = Vec::new();
    let mut do_swap = |steps: &mut Vec<OocStep>,
                       v2p: &mut Vec<usize>,
                       p2v: &mut Vec<usize>,
                       l: usize,
                       g: usize| {
        steps.push(OocStep::Swap {
            local: l,
            global: g,
        });
        swap_passes += 1;
        let vl = p2v[l];
        let vg = p2v[g];
        p2v[l] = vg;
        p2v[g] = vl;
        v2p[vg] = l;
        v2p[vl] = g;
    };
    for g in chunk_bits..num_qubits {
        if p2v[g] == g {
            continue;
        }
        let q = v2p[g];
        if q < chunk_bits {
            do_swap(&mut steps, &mut v2p, &mut p2v, q, g);
        } else {
            do_swap(&mut steps, &mut v2p, &mut p2v, 0, q);
            do_swap(&mut steps, &mut v2p, &mut p2v, 0, g);
        }
    }

    // Finally, all global slots are restored: only local slots 0..chunk_bits can be permuted.
    for i in 0..chunk_bits {
        while p2v[i] != i {
            let j = p2v[i];
            debug_assert!(j < chunk_bits);
            local_swaps.push((i, j));
            let val_i = p2v[i];
            let val_j = p2v[j];
            p2v[i] = val_j;
            p2v[j] = val_i;
            v2p[val_j] = i;
            v2p[val_i] = j;
        }
    }

    if !local_swaps.is_empty() {
        let swap_gates: Vec<Gate> = local_swaps
            .into_iter()
            .map(|(a, b)| Gate::Swap(a, b))
            .collect();

        // If the last step is a LocalRun, fuse into it; otherwise add a new LocalRun
        if let Some(OocStep::LocalRun(ref mut gates)) = steps.last_mut() {
            gates.extend(swap_gates);
        } else {
            steps.push(OocStep::LocalRun(swap_gates));
            local_passes += 1;
        }
    }

    debug_assert!(
        (0..num_qubits).all(|i| v2p[i] == i),
        "qubit restoration must restore identity mapping"
    );

    Ok(OocPlan {
        steps,
        local_passes,
        swap_passes,
    })
}

/// An out-of-core state vector backed by a disk file.
pub struct OocStateVector<T: Real = f64> {
    n: usize,
    chunk_bits: usize,
    file: File,
    path: PathBuf,
    is_temp: bool,
    v2p: Vec<usize>,
    p2v: Vec<usize>,
    stats: OocStats,
    cfg: OocConfig,
    _marker: std::marker::PhantomData<T>,
}

impl<T: Real> OocStateVector<T> {
    /// Creates a new out-of-core state vector backed by a file at `path`.
    ///
    /// Initializes the state vector to the standard ground state $|0\dots0\rangle$.
    pub fn new(n: usize, chunk_bits: usize, path: PathBuf, cfg: OocConfig) -> io::Result<Self> {
        assert!(
            chunk_bits <= n,
            "chunk_bits ({chunk_bits}) cannot exceed n ({n})"
        );
        assert!(
            chunk_bits >= 2 || n < 2,
            "chunk_bits must be >= 2 for 2-qubit gates"
        );

        let elem_size = std::mem::size_of::<Complex<T>>();
        let total_amps = 1usize << n;
        let total_bytes = (total_amps as u64) * (elem_size as u64);

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)?;

        // Pre-allocate / set length on disk
        file.set_len(total_bytes)?;

        // Initialize state |0...0>: amplitude at index 0 is 1.0, all other amplitudes are 0.0.
        // On Unix, ftruncate / set_len fills extended space with zeros, which corresponds exactly
        // to IEEE float 0.0 + 0.0i. We explicitly write the first chunk with amplitude 0 = 1.0.
        let chunk_amps = 1usize << chunk_bits;
        let mut first_chunk = vec![Complex::<T>::zero(); chunk_amps];
        first_chunk[0] = Complex::new(T::one(), T::zero());
        file.write_all_at(as_u8_slice(&first_chunk), 0)?;

        let v2p = (0..n).collect();
        let p2v = (0..n).collect();

        Ok(OocStateVector {
            n,
            chunk_bits,
            file,
            path,
            is_temp: false,
            v2p,
            p2v,
            stats: OocStats::default(),
            cfg,
            _marker: std::marker::PhantomData,
        })
    }

    /// Creates a temporary out-of-core state vector in `cfg.scratch_dir` or system temp.
    /// The backing file is automatically removed when dropped.
    pub fn temp(n: usize, chunk_bits: usize, cfg: OocConfig) -> io::Result<Self> {
        let dir = cfg.scratch_dir.clone().unwrap_or_else(std::env::temp_dir);
        let id = rand::random::<u64>();
        let file_name = format!("qsim_ooc_state_{}_{}_{}.dat", n, chunk_bits, id);
        let path = dir.join(file_name);

        let mut sv = Self::new(n, chunk_bits, path, cfg)?;
        sv.is_temp = true;
        Ok(sv)
    }

    /// Returns the total number of qubits $n$.
    #[inline]
    pub fn num_qubits(&self) -> usize {
        self.n
    }

    /// Returns the number of local qubits $c$ per chunk.
    #[inline]
    pub fn chunk_bits(&self) -> usize {
        self.chunk_bits
    }

    /// Returns the chunk size in amplitudes ($2^c$).
    #[inline]
    pub fn chunk_size(&self) -> usize {
        1usize << self.chunk_bits
    }

    /// Returns the total number of chunks ($2^{n-c}$).
    #[inline]
    pub fn num_chunks(&self) -> usize {
        1usize << (self.n - self.chunk_bits)
    }

    /// Total size of the state vector file in bytes.
    #[inline]
    pub fn total_bytes(&self) -> u64 {
        ((1usize << self.n) as u64) * (std::mem::size_of::<Complex<T>>() as u64)
    }

    /// Path to the backing file on disk.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Accumulated metrics for file passes, bytes moved, and elapsed times.
    pub fn stats(&self) -> &OocStats {
        &self.stats
    }

    /// Resets execution statistics.
    pub fn reset_stats(&mut self) {
        self.stats = OocStats::default();
    }

    /// Reads a chunk from disk into `buf`.
    pub fn read_chunk(&mut self, chunk_idx: usize, buf: &mut [Complex<T>]) -> io::Result<()> {
        let chunk_amps = self.chunk_size();
        assert_eq!(buf.len(), chunk_amps);
        let elem_size = std::mem::size_of::<Complex<T>>() as u64;
        let offset = (chunk_idx as u64) * (chunk_amps as u64) * elem_size;

        let t0 = Instant::now();
        self.file.read_exact_at(as_u8_slice_mut(buf), offset)?;
        let elapsed = t0.elapsed();

        self.stats.io_time += elapsed;
        self.stats.bytes_read += (chunk_amps as u64) * elem_size;
        Ok(())
    }

    /// Writes a chunk from `buf` to disk.
    pub fn write_chunk(&mut self, chunk_idx: usize, buf: &[Complex<T>]) -> io::Result<()> {
        let chunk_amps = self.chunk_size();
        assert_eq!(buf.len(), chunk_amps);
        let elem_size = std::mem::size_of::<Complex<T>>() as u64;
        let offset = (chunk_idx as u64) * (chunk_amps as u64) * elem_size;

        let t0 = Instant::now();
        self.file.write_all_at(as_u8_slice(buf), offset)?;
        let elapsed = t0.elapsed();

        self.stats.io_time += elapsed;
        self.stats.bytes_written += (chunk_amps as u64) * elem_size;
        Ok(())
    }

    /// Reads the entire state vector into memory in canonical qubit order
    /// (use only when the state fits in RAM).
    pub fn read_amplitudes(&self) -> io::Result<Vec<Complex<T>>> {
        let total_amps = 1usize << self.n;
        let mut raw = vec![Complex::<T>::zero(); total_amps];
        self.file.read_exact_at(as_u8_slice_mut(&mut raw), 0)?;
        if self.v2p.iter().enumerate().all(|(q, &p)| p == q) {
            return Ok(raw);
        }
        // Canonical index i has bit q = logical qubit q, stored at file bit v2p[q].
        let mut out = vec![Complex::<T>::zero(); total_amps];
        for (i, o) in out.iter_mut().enumerate() {
            let mut j = 0usize;
            for (q, &p) in self.v2p.iter().enumerate() {
                j |= ((i >> q) & 1) << p;
            }
            *o = raw[j];
        }
        Ok(out)
    }

    /// Logical -> physical qubit map of the file (identity unless
    /// `restore_order` was disabled).
    pub fn layout(&self) -> &[usize] {
        &self.v2p
    }

    /// Computes the total Euclidean norm ($\sqrt{\sum |a_i|^2}$) streaming chunk by chunk.
    pub fn state_norm(&mut self) -> io::Result<f64> {
        let chunk_amps = self.chunk_size();
        let num_chunks = self.num_chunks();
        let mut buf = vec![Complex::<T>::zero(); chunk_amps];
        let mut sum = 0.0f64;

        for k in 0..num_chunks {
            self.read_chunk(k, &mut buf)?;
            for z in &buf {
                sum += z.norm_sqr().to_f64();
            }
        }
        Ok(sum.sqrt())
    }

    /// Executes a global<->local qubit swap between physical qubits `local < c` and `global >= c`.
    ///
    /// Exchanging physical qubit $l$ and global qubit $g$ swaps slice runs of length $2^l$
    /// between chunk pairs $(C_0, C_1)$ where $C_1 = C_0 \mid 2^{g - c}$.
    /// Exactly 1 file pass, touching each chunk once with sequential streaming.
    pub fn swap_qubits(&mut self, local: usize, global: usize) -> io::Result<()> {
        let c = self.chunk_bits;
        assert!(local < c, "local qubit must be < chunk_bits ({c})");
        assert!(
            global >= c && global < self.n,
            "global qubit must be >= chunk_bits and < n"
        );

        let g_bit = global - c;
        let num_chunks = self.num_chunks();
        let chunk_amps = self.chunk_size();
        let elem_size = std::mem::size_of::<Complex<T>>() as u64;
        let chunk_bytes = (chunk_amps as u64) * elem_size;

        let mut buf0 = vec![Complex::<T>::zero(); chunk_amps];
        let mut buf1 = vec![Complex::<T>::zero(); chunk_amps];

        let step = 1usize << (local + 1);
        let block = 1usize << local;

        for k in 0..num_chunks {
            if (k >> g_bit) & 1 == 0 {
                let k0 = k;
                let k1 = k | (1 << g_bit);

                let offset0 = (k0 as u64) * chunk_bytes;
                let offset1 = (k1 as u64) * chunk_bytes;

                // Read pair
                let t_io = Instant::now();
                self.file
                    .read_exact_at(as_u8_slice_mut(&mut buf0), offset0)?;
                self.file
                    .read_exact_at(as_u8_slice_mut(&mut buf1), offset1)?;
                self.stats.io_time += t_io.elapsed();

                // Compute / swap in RAM
                let t_comp = Instant::now();
                for base in (0..chunk_amps).step_by(step) {
                    buf0[base + block..base + 2 * block]
                        .swap_with_slice(&mut buf1[base..base + block]);
                }
                self.stats.compute_time += t_comp.elapsed();

                // Write pair
                let t_io = Instant::now();
                self.file.write_all_at(as_u8_slice(&buf0), offset0)?;
                self.file.write_all_at(as_u8_slice(&buf1), offset1)?;
                self.stats.io_time += t_io.elapsed();
            }
        }

        let total_file_bytes = (num_chunks as u64) * chunk_bytes;
        self.stats.bytes_read += total_file_bytes;
        self.stats.bytes_written += total_file_bytes;
        self.stats.swap_passes += 1;
        self.stats.file_passes += 1;

        // Update virtual-to-physical tracking
        let v_local = self.p2v[local];
        let v_global = self.p2v[global];
        self.p2v[local] = v_global;
        self.p2v[global] = v_local;
        self.v2p[v_global] = local;
        self.v2p[v_local] = global;

        Ok(())
    }

    /// Applies a run of gates strictly on physical qubits `< chunk_bits`.
    ///
    /// Pre-compiles the cache-blocked kernels once, then streams through all chunks.
    /// Exactly 1 sequential streaming pass over the file.
    pub fn apply_local_gates(&mut self, gates: &[Gate]) -> io::Result<()> {
        if gates.is_empty() {
            return Ok(());
        }

        let c = self.chunk_bits;
        let executor = BlockedChunkExecutor::<T>::new(gates, c, &self.cfg.block_config)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e.to_string()))?;

        let num_chunks = self.num_chunks();
        let chunk_amps = self.chunk_size();
        let elem_size = std::mem::size_of::<Complex<T>>() as u64;
        let chunk_bytes = (chunk_amps as u64) * elem_size;

        let mut buf = vec![Complex::<T>::zero(); chunk_amps];

        for k in 0..num_chunks {
            let offset = (k as u64) * chunk_bytes;

            let t_io = Instant::now();
            self.file.read_exact_at(as_u8_slice_mut(&mut buf), offset)?;
            self.stats.io_time += t_io.elapsed();

            let t_comp = Instant::now();
            executor.apply_to_chunk(&mut buf);
            self.stats.compute_time += t_comp.elapsed();

            let t_io = Instant::now();
            self.file.write_all_at(as_u8_slice(&buf), offset)?;
            self.stats.io_time += t_io.elapsed();
        }

        let total_file_bytes = (num_chunks as u64) * chunk_bytes;
        self.stats.bytes_read += total_file_bytes;
        self.stats.bytes_written += total_file_bytes;
        self.stats.local_passes += 1;
        self.stats.file_passes += 1;
        self.stats.num_gates += gates.len();

        Ok(())
    }

    /// Simulates a sequence of gates using out-of-core streaming and DAG lookahead swap scheduling.
    pub fn simulate_gates(&mut self, gates: &[Gate]) -> io::Result<OocStats> {
        let mut c = Circuit::new(self.n);
        for g in gates {
            c.gate(*g);
        }
        self.simulate_circuit(&c)
    }

    /// Simulates a quantum circuit out of core with the scheduler selected in
    /// [`OocConfig::scheduler`].
    pub fn simulate_circuit(&mut self, circuit: &Circuit) -> io::Result<OocStats> {
        let t_total = Instant::now();

        match self.cfg.scheduler {
            OocScheduler::Swap => {
                // Plan the execution steps using the DAG-driven lookahead scheduler
                let plan = schedule_ooc(circuit, self.n, self.chunk_bits)
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e.to_string()))?;

                for step in plan.steps {
                    match step {
                        OocStep::LocalRun(gates) => {
                            self.apply_local_gates(&gates)?;
                        }
                        OocStep::Swap { local, global } => {
                            self.swap_qubits(local, global)?;
                        }
                    }
                }
                // The plan ends in canonical order. (Its final in-chunk `Swap`
                // gates permute bits without going through `swap_qubits`, so
                // reset the bookkeeping explicitly.)
                self.v2p = (0..self.n).collect();
                self.p2v = (0..self.n).collect();
            }
            OocScheduler::Window => {
                let mut gates = Vec::with_capacity(circuit.ops.len());
                for op in &circuit.ops {
                    match op {
                        Op::Gate(g) => gates.push(*g),
                        _ => {
                            return Err(io::Error::new(
                                io::ErrorKind::InvalidInput,
                                "out-of-core simulation currently supports unitary circuits",
                            ))
                        }
                    }
                }
                let opts = WindowOptions {
                    extra_bits: self.cfg.group_bits,
                    restore_order: self.cfg.restore_order,
                    ..WindowOptions::default()
                };
                let plan = schedule_window_gates(&gates, self.n, self.chunk_bits, &self.v2p, &opts)
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e.to_string()))?;
                self.run_window_plan(&plan)?;
            }
        }

        self.stats.wall_time += t_total.elapsed();
        Ok(self.stats.clone())
    }

    /// Executes a precomputed [`WindowPlan`].
    pub fn run_window_plan(&mut self, plan: &WindowPlan) -> io::Result<()> {
        for pass in &plan.passes {
            self.run_window_pass(pass)?;
        }
        // `run_window_pass` does not touch the layout; the plan's final map is
        // installed once all passes have run.
        self.v2p = plan.final_v2p.clone();
        self.p2v = {
            let mut p2v = vec![0; self.n];
            for (v, &p) in self.v2p.iter().enumerate() {
                p2v[p] = v;
            }
            p2v
        };
        Ok(())
    }

    /// One windowed pass: gather each group of `2^k` chunks that differ in the
    /// pass's high bits into a `2^(c+k)` buffer, apply the pass's gates, apply
    /// its bit permutation, and write the group back. Reads and writes the
    /// whole file exactly once.
    pub fn run_window_pass(&mut self, pass: &WindowPass) -> io::Result<()> {
        let c = self.chunk_bits;
        let n = self.n;
        let k = pass.high.len();
        let m = c + k;
        let chunk_amps = 1usize << c;
        let group_amps = 1usize << m;
        let elem = std::mem::size_of::<Complex<T>>() as u64;
        let chunk_bytes = (chunk_amps as u64) * elem;
        for &p in &pass.high {
            assert!(p >= c && p < n, "window qubit {p} out of range");
        }

        // Chunk-index bit positions: window bits and the rest.
        let hbits: Vec<usize> = pass.high.iter().map(|&p| p - c).collect();
        let obits: Vec<usize> = (0..n - c).filter(|b| !hbits.contains(b)).collect();
        let outer = 1usize << obits.len();
        let deposit = |x: usize, bits: &[usize]| {
            let mut v = 0usize;
            for (j, &b) in bits.iter().enumerate() {
                v |= ((x >> j) & 1) << b;
            }
            v
        };
        let h_offsets: Vec<usize> = (0..1usize << k).map(|h| deposit(h, &hbits)).collect();

        let executor = if pass.gates.is_empty() {
            None
        } else {
            Some(
                BlockedChunkExecutor::<T>::new(&pass.gates, m, &self.cfg.block_config)
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e.to_string()))?,
            )
        };

        let file = &self.file;
        let io_nanos = AtomicU64::new(0);
        let read_group = |o: usize, buf: &mut [Complex<T>]| -> io::Result<()> {
            let t0 = Instant::now();
            let base = deposit(o, &obits);
            for (h, &off) in h_offsets.iter().enumerate() {
                let chunk = base | off;
                file.read_exact_at(
                    as_u8_slice_mut(&mut buf[h * chunk_amps..(h + 1) * chunk_amps]),
                    (chunk as u64) * chunk_bytes,
                )?;
            }
            io_nanos.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
            Ok(())
        };
        let write_group = |o: usize, buf: &[Complex<T>]| -> io::Result<()> {
            let t0 = Instant::now();
            let base = deposit(o, &obits);
            for (h, &off) in h_offsets.iter().enumerate() {
                let chunk = base | off;
                file.write_all_at(
                    as_u8_slice(&buf[h * chunk_amps..(h + 1) * chunk_amps]),
                    (chunk as u64) * chunk_bytes,
                )?;
            }
            io_nanos.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
            Ok(())
        };

        let mut compute = Duration::ZERO;
        let mut perm_dur = Duration::ZERO;
        let mut stall = Duration::ZERO;
        let mut scratch: Vec<Complex<T>> = if pass.perm.is_some() {
            vec![Complex::<T>::zero(); group_amps]
        } else {
            Vec::new()
        };
        let mut process = |buf: &mut Vec<Complex<T>>, compute: &mut Duration| {
            let t0 = Instant::now();
            if let Some(ex) = &executor {
                ex.apply_to_chunk(buf);
            }
            if let Some(perm) = &pass.perm {
                let t1 = Instant::now();
                permute_bits(buf, &mut scratch, perm);
                std::mem::swap(buf, &mut scratch);
                perm_dur += t1.elapsed();
            }
            *compute += t0.elapsed();
        };

        let t_pass = Instant::now();
        if !self.cfg.overlap_io || outer == 1 {
            let mut buf = vec![Complex::<T>::zero(); group_amps];
            for o in 0..outer {
                read_group(o, &mut buf)?;
                process(&mut buf, &mut compute);
                write_group(o, &buf)?;
            }
        } else {
            let nbuf = 3usize.min(outer + 1);
            let (free_tx, free_rx) = sync_channel::<Vec<Complex<T>>>(nbuf);
            let (full_tx, full_rx) = sync_channel::<(usize, Vec<Complex<T>>)>(nbuf);
            let (done_tx, done_rx) = sync_channel::<(usize, Vec<Complex<T>>)>(nbuf);
            for _ in 0..nbuf {
                free_tx
                    .send(vec![Complex::<T>::zero(); group_amps])
                    .expect("channel has capacity");
            }
            let (rg, wg) = (&read_group, &write_group);
            let res: io::Result<()> = std::thread::scope(|sc| {
                let reader = sc.spawn(move || -> io::Result<()> {
                    for o in 0..outer {
                        let Ok(mut buf) = free_rx.recv() else {
                            return Ok(());
                        };
                        rg(o, &mut buf)?;
                        if full_tx.send((o, buf)).is_err() {
                            return Ok(());
                        }
                    }
                    Ok(())
                });
                let writer = sc.spawn(move || -> io::Result<()> {
                    for _ in 0..outer {
                        let Ok((o, buf)) = done_rx.recv() else {
                            return Ok(());
                        };
                        wg(o, &buf)?;
                        // The reader may already be done (and have dropped its
                        // end of the pool): the buffer is then simply dropped.
                        let _ = free_tx.send(buf);
                    }
                    Ok(())
                });
                for _ in 0..outer {
                    let t0 = Instant::now();
                    let Ok((o, mut buf)) = full_rx.recv() else {
                        break;
                    };
                    stall += t0.elapsed();
                    process(&mut buf, &mut compute);
                    if done_tx.send((o, buf)).is_err() {
                        break;
                    }
                }
                drop(done_tx);
                // Unblock the reader if the writer died early.
                drop(full_rx);
                let r = reader.join().expect("reader thread panicked");
                let w = writer.join().expect("writer thread panicked");
                r.and(w)
            });
            res?;
        }
        let _ = t_pass;

        let total = (1u64 << n) * elem;
        self.stats.bytes_read += total;
        self.stats.bytes_written += total;
        self.stats.file_passes += 1;
        if !pass.gates.is_empty() {
            self.stats.local_passes += 1;
            self.stats.num_gates += pass.gates.len();
        }
        if pass.perm.is_some() {
            self.stats.swap_passes += 1;
        }
        self.stats.io_time += Duration::from_nanos(io_nanos.load(Ordering::Relaxed));
        self.stats.compute_time += compute;
        self.stats.perm_time += perm_dur;
        self.stats.stall_time += stall;

        // Update the layout from the pass's permutation.
        if let Some(perm) = &pass.perm {
            let bit_pos: Vec<usize> = (0..c).chain(pass.high.iter().copied()).collect();
            let old: Vec<usize> = bit_pos.iter().map(|&p| self.p2v[p]).collect();
            for (b, &ob) in perm.iter().enumerate() {
                let l = old[ob];
                self.v2p[l] = bit_pos[b];
                self.p2v[bit_pos[b]] = l;
            }
        }
        Ok(())
    }
}

impl<T: Real> Drop for OocStateVector<T> {
    fn drop(&mut self) {
        if self.is_temp {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engines::blocked::BlockConfig;
    use crate::engines::statevector::StateVector;

    #[test]
    fn ooc_ground_state_init() {
        let mut ooc = OocStateVector::<f32>::temp(4, 2, OocConfig::default()).unwrap();
        assert_eq!(ooc.num_qubits(), 4);
        assert_eq!(ooc.chunk_bits(), 2);
        assert_eq!(ooc.num_chunks(), 4);

        let amps = ooc.read_amplitudes().unwrap();
        assert_eq!(amps.len(), 16);
        assert_eq!(amps[0], Complex::new(1.0, 0.0));
        for a in &amps[1..] {
            assert_eq!(*a, Complex::zero());
        }

        let norm = ooc.state_norm().unwrap();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn ooc_swap_pass_correctness() {
        // 4 qubits: c=2 local qubits (0, 1), global qubits (2, 3).
        // Apply an X gate to qubit 0 (local), then swap qubit 0 (local) and qubit 2 (global).
        let mut ooc = OocStateVector::<f64>::temp(4, 2, OocConfig::default()).unwrap();
        ooc.apply_local_gates(&[Gate::X(0)]).unwrap();

        // State is now |0001> (amplitude 1 at index 1)
        let amps = ooc.read_amplitudes().unwrap();
        assert_eq!(amps[1], Complex::new(1.0, 0.0));
        assert_eq!(amps[0], Complex::zero());

        // Swap physical qubit 0 and physical qubit 2
        ooc.swap_qubits(0, 2).unwrap();

        // The swap moves data between physical positions but is tracked in the
        // logical->physical map, so the canonical (logical) state is unchanged
        // while the layout records the relocation.
        assert_eq!(ooc.layout()[0], 2);
        assert_eq!(ooc.layout()[2], 0);
        let amps_after = ooc.read_amplitudes().unwrap();
        assert_eq!(amps_after[1], Complex::new(1.0, 0.0));
        assert_eq!(amps_after[4], Complex::zero());
    }

    #[test]
    fn ooc_matches_in_ram_blocked() {
        let n = 6;
        let c = 3; // 8 chunks of 8 amplitudes
                   // A circuit with gates on local, global, and cross-boundary qubits
        let gates = vec![
            Gate::H(0),
            Gate::Cnot(0, 1),
            Gate::Cnot(1, 4),
            Gate::H(5),
            Gate::Cz(4, 5),
            Gate::T(3),
            Gate::Cnot(3, 0),
            Gate::Rx(2, 0.7),
            Gate::Swap(1, 5),
        ];

        // 1. In-RAM blocked simulation
        let mut sv = StateVector::<f64>::new(n);
        sv.apply_gates_blocked(&gates, &BlockConfig::default())
            .unwrap();

        // 2. Out-of-core simulation
        let mut ooc = OocStateVector::<f64>::temp(n, c, OocConfig::default()).unwrap();
        ooc.simulate_gates(&gates).unwrap();

        let ooc_amps = ooc.read_amplitudes().unwrap();
        let sv_amps = sv.amplitudes();

        assert_eq!(ooc_amps.len(), sv_amps.len());
        for i in 0..ooc_amps.len() {
            let diff = (ooc_amps[i] - sv_amps[i]).norm();
            assert!(
                diff <= 1e-12,
                "amplitude mismatch at index {i}: ooc={:?}, sv={:?}, diff={diff}",
                ooc_amps[i],
                sv_amps[i]
            );
        }
    }
}
