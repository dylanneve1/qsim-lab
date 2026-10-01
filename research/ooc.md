# Out-of-core state vector: fewer passes over disk

Branch `exp/ooc-window` (on top of `exp/ooc`). Code: `src/ooc.rs` (engine, old
"swap" scheduler), `src/ooc_window.rs` (new windowed scheduler), planner
`examples/ooc_plan.rs`, bench `examples/ooc_bench.rs`, tests `tests/ooc.rs`.

## Hypothesis
Cost of the OOC engine is the number of full passes over the file. The old
scheduler pays one pass per global<->local qubit swap plus one per local run
(QFT n=28, c=22: 55 passes = 24 local + 31 swap). Instead: gather `2^k` chunks
that differ in `k` high qubits into one RAM buffer (`2^(c+k)` amps), so *any*
gate on the `c+k` window qubits runs in the same pass, and apply the qubit
relocation as an in-buffer bit permutation before write-back (a "swap" costs
no extra I/O). `Swap` gates are folded into the logical->physical map. The
planner is greedy with lookahead (drain executable gates, add the high qubits
that unlock most gates, Belady rule for the post-pass layout, several scoring
alphas tried, fewest passes kept).

## Pass counts (planner only, no disk; `cargo run --release --example ooc_plan`)
Chunk `c=22` (4M amps), window `k=4` (`c+k=26` bits in RAM buffer). "old" =
swap scheduler step count. Full grid (c=18/20/22, k=2..6, n up to 32) in
`research/data/ooc/plan_passes.txt`.

| workload | n | old (swap sched, c=22) | new c=20,k=4 (buffer 2^24) | new c=22,k=4 (buffer 2^26) |
|---|---|---|---|---|
| qft | 24 | 14 | 1(1/0) | 1(1/0) |
| qft | 26 | 34 | 6(5/6) | 1(1/0) |
| qft | 28 | 55 | 7(6/6) | 6(5/6) |
| qft | 30 | 78 | 9(7/8) | 7(6/6) |
| brick | 24 | 7 | 1(1/0) | 1(1/0) |
| brick | 26 | 13 | 3(2/3) | 1(1/0) |
| brick | 28 | 19 | 3(2/2) | 3(2/3) |
| brick | 30 | 25 | 5(3/5) | 3(2/2) |
| brick16 | 24 | 7 | 1(1/0) | 1(1/0) |
| brick16 | 26 | 13 | 3(2/3) | 1(1/0) |
| brick16 | 28 | 19 | 3(2/2) | 3(2/3) |
| brick16 | 30 | 25 | 5(3/5) | 3(2/2) |

Notation `P(g/p)` in the data file: P passes, g with gates, p with a layout
permutation (they overlap). QFT n=28, the headline case: **55 -> 7** passes
(c=20,k=4, buffer 128 MiB at f32) or **55 -> 6** (c=22,k=4, buffer 512 MiB at
f32; too big once I/O overlap double-buffers under the 1.2 GB cap). Planning
takes a few ms. Odd n and c=18 are in `plan_passes.txt` / `ooc_plan` output.

## Exactness
`tests/ooc.rs` (all passing): deterministic cases at n=16..22 with chunk sizes
c=3..9 (many swaps), all-gate-type random circuits (1q, 2q incl. iSWAP, CPhase,
Swap, Toffoli), window sweep k=3,4,5,12, overlap on/off, restore_order on/off,
degenerate shapes (n==c, n==c+1), and a proptest (24 cases, n 14..20, c 4..9,
k 3..5, random gates) checking BOTH schedulers against the in-RAM blocked
executor. Tolerances: <=1e-12 max |dAmp| (f64) and <=1e-5 (f32), stricter than
the 1e-10 / 1e-4 requested. The n=26..28 timing runs only check the final norm
(no in-RAM reference fits); exactness at scale rests on the small-n tests.

## Timings (shared 4-vCPU VM, scratch in /tmp/ooc-scratch)
f32; old = swap scheduler c=22; new = window c=20,k=4, overlapped I/O.
Interleaved old/new, 3 rounds, each run under `bench.sh`; min wall of 3 shown
(all runs in `research/data/ooc/interleaved.csv`; 1-min load at each start in
`interleaved.load`: 8.6 at the very first run, then 3-5, mostly ~3.3).
Script: `research/data/ooc/run_interleaved.sh`.

| workload | n | old passes | old s | new passes | new s | speedup |
|---|---|---|---|---|---|---|
| qft | 26 | 34 | 11.3 | 6 | 3.5 | 3.2x |
| qft | 27 | 42 | 23.6 | 6 | 6.7 | 3.5x |
| qft | 28 | 55 | 62.9 | 7 | 11.4 | 5.5x |
| brick (4 layers) | 26 | 13 | 5.4 | 3 | 2.2 | 2.5x |
| brick | 27 | 16 | 12.0 | 3 | 3.9 | 3.1x |
| brick | 28 | 19 | 27.1 | 3 | 7.5 | 3.6x |

Speedup (5.5x) is below the QFT-28 pass ratio (7.9x) because the new passes do
more compute per byte (in-buffer permutation, larger buffers). Earlier
single-run numbers in `scaling.csv` (taken at load ~7) are superseded by this
table.

## Verdict / unfinished
* Positive: 2.5-5.5x wall clock, 3-8x fewer passes, no accuracy loss.
* The greedy planner is not optimal; beam/DP could cut more passes
  (QFT-30 c=20,k=4 needs 9). Not attempted.
* Only c=20,k=4 was timed interleaved; other (c,k) are single runs in
  `scaling.csv`.
* Shared-VM timings; page-cache effects on the 2 GiB file are not controlled.
