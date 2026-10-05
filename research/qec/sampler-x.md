# Sampler-x: Stim-beating detector sampling at every shot count

Branch `exp/sampler-x` (base `main` 6b21728), 5 October 2026. Machine: Intel Xeon Gold 6548Y+
(Emerald Rapids), Hyper-V VM, 16 vCPU = 8 cores × 2 HT, L1d 48 KB / L2 2 MB per core, L3 60 MB,
AVX-512 (F, VL, VPOPCNTDQ, ...), shared with six other agents and other users.
Code: `src/engines/stabilizer/detector_compiler.rs` and `frame_sampler.rs` (new),
`src/engines/stabilizer/fast_sampler.rs`, `src/io/stim.rs` (new parser), `src/engines/stabilizer/symphase.rs`
(bug fix), `examples/stim_compare.rs` (`sample-x`, `bench-x`, `check-x`, `dem-support-x`). Tests:
`tests/engines/detector_compiler.rs` and `frame_sampler.rs` (new), `tests/engines/symphase.rs`. Data and
scripts: `research/data/sampler-x/`.

**Question.** `FastSampler` ([fast-sampler.md](fast-sampler.md)) samples 8–18× faster than AVX2 Stim
once compiled, but its compile step (55–80 ms at d = 15) made it lose to Stim below about 3·10⁴ shots.
Can a qsim-lab process beat the best Stim build end to end at *every* shot count, 10² to 10⁷, and how
fast does it go with threads?

**Answer.** _(filled in from the measurements below)_

---

## 1. The floor on this machine

**Contenders** (single thread, the same `.stim` file for everyone: Stim's
`surface_code:rotated_memory_z`, rounds = d, all four noise knobs = p, written by `make_circuits.py`;
detection events with the observables appended, ptb64, to `/dev/null`):
- **Stim native**: Stim `main` at 131793e (version string 1.17.dev0), built from source with CMake
  Release and its default `-march=native -O3`. Stim has **no AVX-512 path**: `simd_word.h` picks
  `bitword<256>` (AVX2) whenever `__AVX2__` is defined, and the source has no AVX-512 bitword or
  intrinsic. GCC 11 auto-vectorises a few loops with zmm registers (1,143 zmm vs 2,278 ymm instructions
  in the binary), but its simulation words are 256 bits. `stim detect --out_format ptb64
  --append_observables`.
- **Stim pip**: 1.16.0, the latest release on PyPI (upgraded from 1.15.0 in the shared venv; logged).
  On this CPU the wheel loads `stim._stim_sse2` (128-bit words; the wheel ships no AVX2 module).
  `compile_detector_sampler().sample_write(shots, os.devnull, "ptb64", append_observables=True)` in
  process; Python start-up excluded, so it only enters the sampling-only comparison.
- **ours (old)**: `stim_compare sample-fast` at `main` 6b21728 (the FastSampler pipeline of
  [fast-sampler.md](fast-sampler.md)).
- **ours (new)**: `stim_compare sample-x` (this branch).

Whole-process times come from `wtime` (`posix_spawn` + `wait4`, `CLOCK_MONOTONIC`; its own floor, a
`/bin/true` spawn, is 0.27 ms). Sampling-only throughput follows [fast-sampler-audit.md](fast-sampler-audit.md):
Stim native = wall time at N shots minus at 64 shots (start-up, parse and compile removed, which favours
Stim); pip Stim = in-process `sample_write`; ours = the internal timer of `bench-x` (compile excluded) and
also wall(N) − wall(64). Every timing block holds `/dev/shm/qsim/bench.lock`, interleaves the
contenders (order rotated per repetition), reports the minimum of ≥ 3 repetitions, and records `uptime`
and `free -g` before and after.

### 1.1 The old comparison, reproduced (heavily loaded)

`fast-sampler.md` §3.1 with its own script (`research/data/fast-sampler/timing.py`, unchanged) against the
`stim_compare` built at `main` 6b21728, Stim pip 1.16.0 and native Stim; same shot counts as the original
table; `old_repro.jsonl`. The 1-minute load was **19–36** throughout (other agents' builds and
multi-threaded jobs), so these are loaded numbers; the ratios are interleaved minima of 3.

| d | p | shots | Stim pip | Stim native (net) | old FastSampler | old / native | end-to-end old / Stim | old compile | load |
|---|---|---|---|---|---|---|---|---|---|
| 3 | 0.001 | 4,000,000 | 29.65 | 29.04 | 462.8 | **15.9×** | 12.51× | 0.4 ms | 36 |
| 7 | 0.001 | 1,000,000 | 3.12 | 3.65 | 34.3 | **9.4×** | 8.24× | 3.5 ms | 35 |
| 11 | 0.001 | 256,000 | 0.82 | 0.91 | 8.4 | **9.1×** | 5.48× | 18.6 ms | 33 |
| 15 | 0.001 | 128,000 | 0.32 | 0.36 | 2.9 | **8.0×** | 2.86× | 93.5 ms | 33 |
| 3 | 0.003 | 4,000,000 | 19.94 | 21.26 | 192.0 | **9.0×** | 8.02× | 0.3 ms | 33 |
| 7 | 0.003 | 1,000,000 | 1.86 | 2.18 | 14.0 | **6.4×** | 6.06× | 3.4 ms | 32 |
| 11 | 0.003 | 256,000 | 0.48 | 0.57 | 3.4 | **6.0×** | 4.74× | 19.1 ms | 28 |
| 15 | 0.003 | 128,000 | 0.19 | 0.22 | 1.3 | **5.9×** | 3.19× | 80.3 ms | 19 |

(Mshot/s. "net" = wall time at N shots minus at 64 shots.) Stim's DEM sampler, timed in the same runs, is
not the stronger baseline: native `sample_dem` (net of a 64-shot run, DEM given for free) reaches 30.9 /
1.48 / 0.33 / 0.12 Mshot/s at d = 3 / 7 / 11 / 15 (p = 0.1 %), against 29.0 / 3.65 / 0.91 / 0.36 for
`stim detect`; only at d = 3 is it marginally ahead. The per-shot lead of the old FastSampler
reproduces in kind (6–16× against native Stim, 8–18× on the idle EPYC), lower at p = 0.3 %, where the
sampler does 3× more random memory work per shot and shares the cores' caches with the load. Its
compile was 80–94 ms at d = 15 here (load 19–33), against 2.1–2.6 ms for Stim's whole 64-shot process.

## 2. Killing the compile bottleneck

### 2.1 Where the old compile went

The old pipeline (`sample-fast`) was: `parse_stim` (unrolls every `REPEAT` by re-parsing each body line
on every iteration) → `SymPhaseSampler::new` (a full CHP tableau run for the noiseless reference sample,
then a *forward* symbolic Pauli frame: per qubit two bit lines of `#variables` bits, every gate XORs whole
lines) → `with_parities` (detector rows as XORs of raw measurement rows, which grow linearly with the round
for data-qubit faults) → prune → `FastSampler::new` (CSC transpose, then per (group, Pauli) entry:
concatenate, sort, cancel pairs). Its cost grows like `gates × variables / 64`.

**Table 2.1 — parse + compile, Stim's `rotated_memory_z` (p = 0.1 %), milliseconds.** Old: `bench-fast`
at `main` (11:33, load 6–9). New: `bench-x`, warm = min of 5 repetitions in one process, cold = the first
repetition (fresh memory, as in a one-shot process); load 22–27 (`devlog_first_build.txt`).

| d | detectors | noise groups | old parse | old compile | new parse (warm) | new compile (warm / cold) | tables (warm) |
|---|---|---|---|---|---|---|---|
| 3 | 24 | 189 | 0.08 | 0.19 | 0.03 | 0.015 | 0.03 |
| 7 | 336 | 2,625 | 0.41 | 1.89 | 0.11 | 0.15 | 0.40 |
| 15 | 3,360 | 26,505 | 3.33 | 51.8 | 0.49 | 1.42 / 2.7–3.1 | 3.99 |
| 25 | 15,600 | 123,675 | 14.0 | 1,146 | 0.83 | 4.80 / 12.5–12.9 | 11.2 |

The compile is 37× (d = 15) to 240× (d = 25) faster warm, and Stim's own `detector_error_model()` (the
comparable precomputation, pip, with error merging) takes 68 ms at d = 15 and 210 ms at d = 25 here.

### 2.2 The backward detector compiler (`detector_compiler.rs`)

Detection events are frame flips: a detector's value is the XOR of the measurement flips it reads,
relative to the noiseless reference (Stim's convention). The reference sample itself is never needed, so
the compiler does no tableau simulation at all. It sweeps the circuit **backwards** (as Stim's error
analyzer does), keeping for every qubit the set of detector/observable rows an X or a Z flip at the current
point would toggle:

| op (backwards) | update |
|---|---|
| `DETECTOR` / `OBSERVABLE_INCLUDE` | add the row to the pending set of each measurement it reads |
| Z measurement `m` of `q` | record the coin (`zs[q]`) and the readout flip (`pending[m]`); `xs[q] ^= pending[m]` |
| reset of `q` | record the reset flip (`xs[q]`) and the coin (`zs[q]`); clear both |
| `H` | swap `xs[q]`, `zs[q]` |
| `S`, `S_DAG` | `xs[q] ^= zs[q]` |
| `CX c t` | `xs[c] ^= xs[t]`, `zs[t] ^= zs[c]` |
| `CZ a b` | `xs[a] ^= zs[b]`, `xs[b] ^= zs[a]` |
| X / Y / Z flip, `DEPOLARIZE1/2` | record the columns `xs`, `xs ^ zs`, `zs` (1, 2 or 4 variables) |
| classically controlled Pauli on `q` reading `m` | `pending[m] ^= sens(P on q)` |

Each noise channel's columns are read off directly; the recording (in reverse) is flipped into program
order at the end, giving *exactly* the variable groups, order and columns of the old route after pruning
(§4). Implementation details that matter for speed:
- sets are sorted `u32` lists in one **append-only arena**: an update appends the merged set and repoints
  a `(offset, len)` descriptor, so an empty destination just shares its source's storage and recording a
  column is a descriptor push, not a copy;
- the symmetric-difference merge is branch-free apart from its exit test (min of the heads written, the
  output cursor advances only when the heads differ);
- no allocation per op, no `Vec` from `Gate::qubits()`;
- the `.stim` front end walks the parsed program backwards *without unrolling it*: the new parser
  (`parse_stim_circuit`) parses every line once and keeps `REPEAT` blocks as blocks; the walker
  (`for_each_op_rev`) unrolls them on the fly and resolves `rec[-k]` against a running measurement count.

Hit tables (the padded (group, Pauli) → rows entries of the FastSampler) are built by a new builder: per
group, the sorted union of its 1, 2 or 4 columns with an incidence nibble per row (branch-free merges);
a row belongs to Pauli `t`'s entry iff `INPAT[incidence]` has bit `t` (odd overlap). It produces the
same tables as the original builder (`==`, §4).

**Why not bit-packed sensitivity sets with AVX-512?** A dense representation keeps, per qubit, two
bit sets over all rows (R = 3,361 at d = 15, 15,601 at d = 25, i.e. 53 and 244 words). A CX then costs two
R/64-word XORs (cheap with AVX-512: 7 and 31 zmm XORs), but every noise channel has to *read out* its 1–4
columns, i.e. scan up to 4·R/64 words and extract the set bits: about 26k channels × 4 × 53 words ≈ 5.6M
word reads at d = 15 and 100k × 4 × 244 ≈ 98M at d = 25, against a few hundred thousand list entries
for the sparse sets (detector sets are local in time: a fault toggles 1–4 detectors). VPOPCNTDQ would
only speed up the size counts, which the sparse lists have for free. Without a windowing scheme the dense
route loses by an order of magnitude at d = 25, so it was not built; the measured sparse compile is in
table 2.1.

**Tables only when they pay off.** Building the tables costs about one write per (group, Pauli) entry
(220k entries at d = 15); sampling through the columns costs a few ns more per hit. `sample-x` builds
them only when the run is expected to draw at least one hit per entry (`tables_pay_off`). The column path
uses the *same* blocked slot generator as the table path, so with or without tables the output is
bit-identical for a given seed (§4); only the speed differs.

### 2.3 Short runs: no compile at all (`frame_sampler.rs`)

The first build of the backward compiler (`devlog_first_build.txt`, load 23) beat Stim at d ≤ 11 for
every shot count tried but **lost at 128 shots for d ≥ 15** (0.58× at d = 15, 0.49× at d = 25). Two
reasons, both measured:
- in a fresh process the compile is 2–3× slower than in a warm one (d = 25: 12.5 ms cold vs 4.8 ms warm):
  the arena, the recorded columns and the CSC are ~30 MB of first-touch memory, and page faults in this
  VM are expensive;
- even warm, a sparse set merge per gate costs more than pushing a two-word (128-shot) Pauli frame
  through the gate, which is all Stim does for a short run.

So `sample-x` does not compile short runs at all: up to `FRAMES_UP_TO` shots it runs a Pauli-frame
simulation straight off the parsed program (`FrameSampler`): word-parallel frames, coins after every
measurement and reset, detectors as XORs of recorded flips, the `REPEAT` structure walked without
unrolling, and the measurement record kept as a ring of `max_lookback` slots so it stays in cache.
Noise uses FastSampler's exact hit model per target: the number of hits over the batch's `64 W` shots
is one lookup in a precomputed Poisson inverse-CDF table, each hit one random word (shot from the top
bits, uniform Pauli), so no logarithm is evaluated per fault; channels at or past full mixing are
sampled literally. It is exact (the standard frame method plus the hit model; §4 has its tests). Above
the threshold the compiled FastSampler takes over.

## 3. Sampling: tables or columns, batch size, threads, AVX-512

The sampling loop is FastSampler's (Poisson hits from per-(kind, p) streams, blocked slot generation,
padded `u16` hit tables; [fast-sampler.md](fast-sampler.md) §1). Changes on this branch:
- **no-table column path on the blocked stream**: a hit on (group, Pauli) XORs the Pauli's variable
  columns (rows in two columns cancel) instead of one table entry; same slot stream, so bit-identical
  output. Used when the tables would not pay for themselves (§2.2);
- **`u16`-only tables**: the `u32` copy is no longer kept when every row fits in 16 bits (half the build
  traffic); the ablation paths rebuild it on demand;
- **zeroed batches and zero-copy output**: detection-event batches start from `fill(0)` instead of
  copying an all-zero reference, and a batch's 64-shot blocks go to the writer as one `writev` of views
  into the sample buffer (no byte copy on little-endian targets);
- **threads with per-batch streams**: batch `b` draws from `batch_rng(seed, b)`, a jump of `b·2³²`
  words into one wyrand sequence, so the output is the same for any thread count; slabs of consecutive
  batches (about 4 MB of output each) are sampled in rounds of `T` slabs on a rayon pool and written in
  order;
- **an AVX-512 hit kernel** (`set_simd`, `tables = simd`): per hit, the 4 `u16` rows are widened
  (`vpmovzxwq`), the 4 output words gathered (`vpgatherqq`), XOR-ed and scattered back (`vpscatterqq`,
  AVX-512F/VL); padding entries all point at the sink word, so duplicate scatter lanes are harmless.

### 3.1 The AVX-512 hit kernel loses (negative result)

Per hit the scalar kernel does 4 independent load-XOR-stores to addresses in the L1/L2-resident batch
buffer; the AVX-512 kernel replaces them by one 4-lane gather and one 4-lane scatter. From the first
build's profile (`devlog_first_build.txt`, bench-x, 1,024 shots, warm, load 22–27), sampling time per
batch: d = 7: 22 vs 25 µs, d = 15: **233 vs 337 µs** (the AVX-512 kernel is 1.45× slower), d = 25:
1.20 vs 1.30 ms; at d = 3 both take 2 µs. Gather/scatter on this core cost more than the four scalar
accesses they replace, and the scatter serialises the stores. The kernel stays in the code
(`set_simd`, `sample-x … simd`) for the record, off by default. Bit-identity with the scalar kernel is
tested.

### 3.2 Batch width on this cache hierarchy

_(wsweep.jsonl: W = 4 … 64 words per batch, d = 3, 15, 25; table below)_

### 3.3 Threads

_(threads.jsonl: 1, 2, 4, 8 threads on distinct physical cores, 16 threads on all logical CPUs; table
below)_

### 3.4 Sampling only, single thread, against Stim

_(throughput.jsonl; table below)_

## 4. Exactness

Nothing in the sampler's distribution changed: the new compiler produces the same columns as the old
one, and the sampling paths added here (no-table columns, threads, AVX-512) are bit-identical to the
table path for a given seed. All checks below are in `cargo test` unless marked as a script.

**Exact equality with the old compiler** (`tests/engines/detector_compiler.rs`; `==` on the data, not a
statistical test):
- `Columns` (variable groups, order, distributions, rows per variable) of `compile_circuit` equal those
  of the old route (`SymPhaseSampler::new` → `with_parities` → `relative_to_reference` → transpose) on
  1,500 random noisy Clifford circuits (1–8 qubits, up to 80 ops, every gate the tableau accepts
  including `I`, `Sx`, `Sxdg`, `ISwap`, `ISwapdg`, `CPhase(kπ)`, `Phase(kπ/2)`; mid-circuit
  measurement and reset; X/Y/Z flips, `DEPOLARIZE1/2` at p up to 0.4 so dense groups appear;
  classically controlled Paulis; four noise models incl. readout and reset noise; random detectors and
  observables with repeated records);
- and on every QEC circuit family in the test suite: `SurfaceCode` d = 3, 5, 7 under three noise
  models (incl. p = 0.3, dense path), `ScheduledSurfaceCode` (two schedules), the colour code (d = 3, 5,
  Kishony–Fowler schedule, CNOT and uniform noise, Z and X basis) and the [[72, 12, 6]] bivariate
  bicycle code (1 and 3 rounds, both bases);
- the resulting samplers are equal too (`FastSampler::from_columns(.., true) == FastSampler::new(..)`):
  **the new table builder's hit tables equal the old builder's entry for entry**;
- the `.stim` front end (`compile_stim` on the unparsed `REPEAT` structure) equals the unrolled-circuit
  front end and the old route on Stim's own generated circuits (rotated memory Z d = 3 and X d = 5,
  unrotated Z d = 3, repetition d = 5, colour `memory_xyz` d = 5 decomposed, which has detectors of
  noiseless parity 1), on a hand-written file with every instruction and alias of the subset, nested
  `REPEAT`s and observables between detectors, and on 400 random programs with nested `REPEAT` blocks;
- the new parser equals the old one (`parse_stim == parse_stim_reference`) on all of those.
- `check-x` (script) asserts old == new on the timing circuits, d = 3 … 25 _(see §4 table)_.

**Bit-identical sampling paths** (`write_ptb64_identical_across_threads_and_table_modes`): for
surface (d = 5) and colour (d = 5) circuits and shot counts 63, 4·10⁴ and 100,017, the ptb64 bytes are
identical for 1, 2, 3, 8 threads, for slab sizes from one batch to 4 MB, with and without hit tables,
and with the AVX-512 kernel; a different seed changes them.

**Frame sampler** (`tests/engines/frame_sampler.rs`): the full joint histogram of 60 random programs
(every instruction of the subset, `REPEAT`, p up to 0.3, readout flips; 2¹⁸ shots each, batch widths
1–16 words, scalar and AVX-512 builds alternating) passes a χ² test against the exact distribution
enumerated from `compile_stim`'s columns (Wilson–Hilferty z < 4.5, Bonferroni); a program with one
channel at +25 % is rejected at z > 10; on Stim's d = 5 rotated memory-X circuit every detector rate and
DEM-correlated pair rate agrees with the FastSampler at 2²⁰ shots (1 % family-wise); the AVX-512 build
is bit-identical.

**Two-sample tests against Stim at 10⁶ shots** (`equivalence_x.py`, `negative_control_x.py`,
`equivalence_other_x.py`; the qec-r4 / fast-sampler test battery unchanged: T0 DEM support, T1
marginals, T2 DEM-correlated pairs, T3 events-per-shot mean and variance, T4 joint 4-detector
histograms for the other circuits; Bonferroni at 1 % family-wise per cell) on the new paths: _(table
4.1)_

**Regression found on the way.** `SymPhaseSampler::new` panicked (`unreachable!`) on `I`, `Sx`, `Sxdg`,
`ISwap` and `ISwapdg`, gates the tableau accepts: the symbolic frame had no rule for them. Fixed, with
`exact_distribution_with_every_tableau_clifford` (exact enumeration against the branching tableau on
150 random circuits with those gates and gate noise), which fails on the old code.

## 5. End to end at every shot count

`e2e.py` via `run_grid.py`: for every (d, p) one bench-lock hold; within it, every shot count
(10², 10³, …, 10⁷, rounded up to a multiple of 64 because Stim refuses ptb64 otherwise: 128, 1,024,
10,048, 100,032, 10⁶, 10⁷) runs the contenders interleaved (order rotated per repetition), minimum of
3; whole process from `posix_spawn` to `wait4`. Contenders: native `stim detect` and `stim_compare
sample-x` with its default policy (frames up to 2,048 shots, compiled above, tables when they pay off),
plus the old pipeline (`sample-fast` at `main`) for reference.

## 6. Caveats and negative results

- **Shared machine.** Every timing was taken under the swarm's bench lock, which stops other *timing*
  runs, not other users: the 1-minute load is recorded per block and was often far above the 16 vCPUs
  from other agents' builds and multi-threaded jobs. Rows taken at load > 16 are marked; they inflate
  both sides' wall times (interleaved, min of ≥ 3), so the ratios are less affected than the absolute
  numbers, but they are not idle-machine numbers.
- **Detection events only.** The backward compiler needs no reference sample, so it compiles detector
  and observable sampling; raw measurement sampling (`sample` without detectors) still goes through
  `SymPhaseSampler` (which needs the reference).
- **Same subset of `.stim`** as before (module docs of `src/io/stim.rs`): `PAULI_CHANNEL_*`,
  `E`/`ELSE_CORRELATED_ERROR`, `HERALDED_*`, `MPP`, `MY`, feedback `CX rec[-1] q` and other gates are
  rejected with an error, never approximated. The new parser also rejects two inputs the old one
  silently accepted: `REPEAT 0` and a measurement flip probability outside [0, 1].
- **What "exact" means on both sides.** Our samplers are exact up to f64 rounding of the hit rates and
  Poisson tables (relative ~1e-16) and the 2⁻⁶⁴ grid of the random words. Stim's frame sampler converts
  every noise probability to `float` before its geometric skipping (`RareErrorIterator`), i.e. it samples
  `X_ERROR(0.001)` at p = 0.0010000000475 (relative 5·10⁻⁸). Neither is visible at 10⁶ shots.
- **wyrand streams.** Batch `b` uses words `b·2³² …` of one wyrand sequence (period 2⁶⁴), so streams
  of different batches never overlap as long as a batch draws fewer than 2³² words (the largest batch
  here draws about 4·10⁵) and there are fewer than 2³² batches (4·10¹² shots).

## 7. Reproduction

Everything below is in `research/data/sampler-x/` (scripts take their inputs as arguments):

```sh
export CARGO_TARGET_DIR=...            # one per worktree
cargo build --release --example stim_compare
B=$CARGO_TARGET_DIR/release/examples/stim_compare
gcc -O2 -o wtime research/data/sampler-x/wtime.c        # process timer
git clone --depth 1 https://github.com/quantumlib/Stim && cmake -S Stim -B Stim/build \
    -DCMAKE_BUILD_TYPE=Release && cmake --build Stim/build --target stim -j 4   # -march=native -O3
S=Stim/build/out/stim
python3 research/data/sampler-x/make_circuits.py circ          # Stim's rotated_memory_z, d = 3 ... 25
# exactness (also in cargo test)
for f in circ/*.stim; do $B check-x $f; done
# whole-process timing grid (one bench-lock hold per cell, waits for load < 16)
python3 research/data/sampler-x/run_grid.py e2e $B $S ./wtime circ e2e.jsonl
# sampling-only throughput, single thread
python3 research/data/sampler-x/run_grid.py throughput $B $S ./wtime circ throughput.jsonl 3,5,7,11,15 \
    0.001,0.003 auto 3
# batch width, AVX-512 kernel, threads
python3 research/data/sampler-x/wsweep.py $B circ/B_d15_p0.001.stim 15 0.001 200000 3 4,8,16,32,64
python3 research/data/sampler-x/threads.py $B circ/B_d3_p0.001.stim 3 0.001 1e9 3 1,2,4,8,16
# 10^6-shot two-sample equivalence with Stim (all cells of §4), negative controls, DEM support
research/data/sampler-x/run_equivalence.sh $B /tmp/eq circ
# auto-policy calibration: frames vs compiled without / with tables, whole process
python3 research/data/sampler-x/e2e.py $B $S ./wtime circ/B_d15_p0.001.stim 15 0.001 3 \
    128,512,2048,8192,32768,131072 x_frames,x_notables,x_tables
```
`SAMPLER_X_FRAMES_UP_TO` and `SAMPLER_X_KAPPA` override `sample-x`'s auto thresholds (for calibration).
Tables for this notebook: `python3 research/data/sampler-x/make_tables.py`.
