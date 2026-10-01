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
//!    ([`crate::blocked::BlockedChunkExecutor`]), and written back to disk.
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

use crate::blocked::{BlockConfig, BlockedChunkExecutor};
use crate::circuit::{check_gate, Circuit, Op, SimError};
use crate::dag::{Dag, NodeId};
use crate::gate::Gate;
use crate::statevector::Real;
use num_complex::Complex;
use num_traits::Zero;
use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::io;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
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
    Swap { local: usize, global: usize },
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
}

impl Default for OocConfig {
    fn default() -> Self {
        OocConfig {
            chunk_bits: 20, // 2^20 amplitudes = 8 MiB in f32, 16 MiB in f64
            scratch_dir: None,
            block_config: BlockConfig::default(),
            restore_order: true,
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
            eviction_pool.sort_unstable_by(|&a, &b| next_use[b].cmp(&next_use[a]));

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

            if best_candidate.as_ref().map_or(true, |c| score > c.score) {
                best_candidate = Some(CandidateChoice {
                    needed,
                    evicted,
                    score,
                });
            }
        }

        let choice = best_candidate.expect("at least one candidate node must be viable");

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
    let mut local_swaps = Vec::new();

    // First restore any global qubits that ended up in local slots
    loop {
        let mut fixed_any = false;
        for v in chunk_bits..num_qubits {
            let p = v2p[v];
            if p < chunk_bits {
                // v is global, but sitting in local slot p. Swap p with home slot v.
                let target_global = v;
                steps.push(OocStep::Swap {
                    local: p,
                    global: target_global,
                });
                swap_passes += 1;

                let displaced_v = p2v[target_global];
                p2v[target_global] = v;
                p2v[p] = displaced_v;
                v2p[v] = target_global;
                v2p[displaced_v] = p;

                fixed_any = true;
                break;
            }
        }
        if !fixed_any {
            break;
        }
    }

    // Next, fix any displaced global slots that hold another global qubit
    for g in chunk_bits..num_qubits {
        if p2v[g] != g {
            // Pick local slot 0 as temporary shuttle
            let l = 0;
            steps.push(OocStep::Swap {
                local: l,
                global: g,
            });
            swap_passes += 1;

            let displaced = p2v[g];
            let loc_v = p2v[l];
            p2v[g] = loc_v;
            p2v[l] = displaced;
            v2p[loc_v] = g;
            v2p[displaced] = l;

            // Now displaced is in local slot l; swap it to its home
            let home = displaced;
            if home >= chunk_bits && home != g {
                steps.push(OocStep::Swap {
                    local: l,
                    global: home,
                });
                swap_passes += 1;

                let other = p2v[home];
                p2v[home] = displaced;
                p2v[l] = other;
                v2p[displaced] = home;
                v2p[other] = l;
            }
        }
    }

    // Finally, all global slots are restored: only local slots 0..chunk_bits can be permuted.
    for i in 0..chunk_bits {
        while p2v[i] != i {
            let j = p2v[i];
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

    /// Reads the entire state vector into memory (use only when state fits in RAM).
    pub fn read_amplitudes(&self) -> io::Result<Vec<Complex<T>>> {
        let total_amps = 1usize << self.n;
        let mut amps = vec![Complex::<T>::zero(); total_amps];
        self.file.read_exact_at(as_u8_slice_mut(&mut amps), 0)?;
        Ok(amps)
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

    /// Simulates a quantum circuit using out-of-core streaming and DAG lookahead swap scheduling.
    pub fn simulate_circuit(&mut self, circuit: &Circuit) -> io::Result<OocStats> {
        let t_total = Instant::now();

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

        self.stats.wall_time = t_total.elapsed();
        Ok(self.stats.clone())
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
    use crate::blocked::BlockConfig;
    use crate::statevector::StateVector;

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

        // Index 1 has bit 0=1, bit 2=0. After swap, bit 0=0, bit 2=1 -> index 4 (|0100>)
        let amps_after = ooc.read_amplitudes().unwrap();
        assert_eq!(amps_after[4], Complex::new(1.0, 0.0));
        assert_eq!(amps_after[1], Complex::zero());
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
