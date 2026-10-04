# Scaling Shor's algorithm: lab notebook (topic `shor`, branch `exp/shor`)

Machine: shared VPS, 4 vCPU AMD EPYC-Rome (AVX2), 7.7 GB RAM, rustc 1.93.1,
release build (portable, no `target-cpu=native`). Every timing below went
through `qsim-swarm/bench.sh` (flock), but the box was shared with other
agents' compiles and a transcription job; the 1-minute load at start ranged
from 1.8 to 11. Treat times as ±30 %. Peak RSS is the process `VmHWM`. Raw
outputs are in `research/data/shor/`.

## Summary

| path | qubits | largest N factored here | bits | time | peak RSS | runs |
|---|---|---|---|---|---|---|
| (a) old: 2n counting + n work, full inverse QFT, permutation oracle, dense f64 | 3n | 253 = 11 × 23 | 8 | 16.3 s | 533 MiB | 5 |
| (b) semiclassical (1 recycled control), permutation oracle, dense f64, gate-by-gate | n+1 | 16 744 463 = 4091 × 4093 | 24 | 13.9 s | 773 MiB | 1 |
| (b') same circuit, fused rounds, dense f64 | n+1 | 16 744 463 = 4091 × 4093 | 24 | 7.4 s | 517 MiB | 1 |
| (b') fused rounds, dense f32 | n+1 | 33 489 353 = 5783 × 5791 | 25 | 44.7 s | 517 MiB | 3 |
| (b') fused rounds, dense f32 | n+1 | 66 994 189 = 8179 × 8191 | 26 | 96.9 s | 1029 MiB | 3 |
| (c) semiclassical, permutation oracle, sparse, fused rounds; generic N ≈ 1e6 | n+1 = 21 | 1 005 973 = 997 × 1009 | 20 | 0.024 s | 10 MiB | 1 |
| (c) same, special N with λ(N) ≈ 2√N (see caveat) | n+1 = 44 | 8 795 579 227 471 = 2 097 091 × 4 194 181 | 43 | 12.9 s | 213 MiB | 1 |
| (d) semiclassical, **gate-level Beauregard oracle**, dense f64, blocked executor | 2n+3 = 23 | 1003 = 17 × 59 | 10 | 63.4 s | 139 MiB | 1 (a = 2 given) |
| (e) semiclassical, **gate-level ripple-carry oracle** (Cuccaro, X/CNOT/CCX), sparse u64 keys, block per key (exp/shor-ripple; VPS) | 3n+4 = 64 | 1 005 973 = 997 × 1009 | 20 | 10.8 s | 14 MiB | 1 |
| (e) same circuit, same N, **Mac M1 Pro** (A/B baseline for round 4) | 64 | 1 005 973 | 20 | 2.47 s | 15 MiB | 1 (min of 3) |
| (f) **round 4**: same ripple circuit, **bit-sliced branch tracking** (`--sliced`), f64, Mac | 64 | 1 005 973 | 20 | 0.078 s | 12 MiB | 1 (min of 3) |
| (g) round 4: **windowed table-lookup oracle** (w = 4, X/CNOT/CCX), sliced, f64, Mac | 4n+8 = 88 | 1 005 973 | 20 | 0.050 s | 9 MiB | 1 (min of 3) |
| (g) same, generic semiprime, random base (seed 1), Mac | 120 | **221 643 407 = 14 207 × 15 601** | 28 | 16.1 s | 2.82 GB | 1 |
| (g) same, generic, random base (seed 2), Mac | 124 | 282 304 153 = 12 391 × 22 783 | 29 | 44.2 s | 1.86 GB | 1 (seed 1: 3 runs, right order, a^(r/2) = −1) |
| (g) same, **f32**, generic, random base (seed 2), Mac — **this repo's gate-level record** | 4n+8 = 132 | **1 537 596 787 = 29 287 × 52 501** | **31** | **134.4 s** | 4.28 GB | 1 (seed 1: 1 run, right order, a^(r/2) = −1) |
| (g) same, special N = p(2p−1) (row (c)'s N; same caveat), Mac | 180 | 8 795 579 227 471 | 43 | 2.54 s | 78 MB | 1 |
| (g) same, special N = p(2p−1) (caveat as row (c)), Mac | 216 | 3 384 163 410 217 561 = 41 134 921 × 82 269 841 | 52 | 60.5 s | 2.40 GB | 1 |

Runs use `qsim run shor --modulus N --semiclassical [--fused] [--sparse] [--f32]
[--oracle beauregard|ripple|windowed] [--sliced] [--window w] [--base a] --seed 1`; without `--base` the bases are
drawn at random (like the old path) until a factor is found.

What this does and does not show — be clear about it:

* (b)/(c) still use a **permutation oracle**: controlled `U_a` is a classical
  lookup table applied to basis states. That is why the work register stays
  sparse and why a 43-bit N runs on a laptop-class VM. It demonstrates the
  qubit-recycling trick and exact simulation of the *measurement statistics*,
  not a quantum circuit you could compile for hardware.
* With the permutation oracle and the sparse state, the cost is
  `O(t · r)` hash operations, `r` = order of `a`. That is the *classical*
  difficulty of order finding. The 43-bit row uses `N = p(2p − 1)` so that
  `λ(N) = 2(p − 1) ≈ √(2N)`; such N are classically trivial (solve
  `2p² − p = N`), and Pollard-style methods would also be fast. For generic
  semiprimes `λ(N) ≈ N/2`, and the sparse state is no smaller than the dense
  one (row "generic m=22" below). The sparse path is a demonstration that the
  simulation cost tracks `r`, not a quantum speed-up.
* (d) is the genuine quantum circuit: Draper/Beauregard QFT adders, modular
  adder, controlled multiplier and controlled SWAP, built only from H, X,
  CNOT, CCX, Phase and CPhase. Its cost is a dense `2^(2n+3)` state × `O(n^3)`
  gates per round × `2n` rounds. 10 bits is where it stops on this box
  (estimated ~5 min/run at n = 11, not run, to keep the shared bench lock
  under ~2 min).

### N ≈ 1e6 (requested target)

All with the **permutation oracle**, semiclassical QFT, `--seed 1`, random bases,
each factored on the first run (raw: `data/shor/n1e6.txt`):

| N | a | order r | factors | sparse fused | sparse gate-by-gate | dense fused f64 | dense gate-by-gate f64 |
|---|---|---|---|---|---|---|---|
| 1 005 973 | 980 062 | 41 832 | 997 × 1009 | 0.024 s, 10 MiB, peak 20 916 amps | 0.046 s, 9 MiB | 0.56 s, 37 MiB | 0.91 s, 53 MiB |
| 1 003 883 | 978 026 | 9 108 | 991 × 1013 | 0.013 s, 6 MiB, peak 4 554 amps | 0.017 s, 6 MiB | 0.50 s, 37 MiB | 0.90 s, 53 MiB |
| 1 001 677 | 975 877 | 499 838 | 983 × 1019 | 0.81 s, 73 MiB, peak 249 919 amps | 3.9 s, 117 MiB | 0.45 s, 37 MiB | 0.92 s, 53 MiB |

21 simulated qubits (control + 20 work). Every path measured the same 40-bit
integer for the same seed (an end-to-end cross-check at this size). The last
row shows the crossover: when `r ≈ N/2` the sparse map holds as many entries
as the dense vector and is slower.

**Why the gate-level oracle is not used at 1e6.** Beauregard's circuit needs
2n + 3 = 43 qubits at n = 20: dense is 2^43 × 16 B = 128 TiB. The sparse state
does not help either: the accumulator `b` lives in Fourier space, so every
`φADD` puts it in a uniform-magnitude superposition over all 2^(n+1) values
(measured: peak nnz = the full 2^13 / 2^15 at N = 21 / 35, sparse 18–50×
slower than dense; `data/shor/beauregard_sparse_vs_dense.txt`). A reversible
ripple-carry multiplier (Cuccaro or VBE adders: X / CNOT / Toffoli only) keeps
every register a classical basis state given the control and work values, so
the sparse state would stay at ≤ 2r entries and a *gate-level* 1e6 run would
cost `O(r · gates)`. That is the next step; it is not implemented here.

## Levers

### 1. Semiclassical QFT with one recycled control qubit

`src/shor.rs`. For measured bit `i = 0 .. 2n−1` (least significant first):
H on the control, controlled `U^(2^(2n−1−i))`, `Phase(−2π · y_low / 2^(i+1))`
where `y_low` is the integer of bits already measured, H, measure, recycle.
This is the Griffiths–Niu semiclassical inverse QFT; by the deferred
measurement principle the distribution of the measured integer equals the
textbook circuit's.

`Op::ClassicControlled` conditions on a single bit, but none is needed for an
extra op: the correction factorises, `−2π y_low / 2^(i+1) = Σ_l y_l · (−π /
2^(i−l))`, so `semiclassical_circuit` emits one `c_if(l, Phase(0, −π/2^(i−l)))`
per earlier bit, and recycles the control with `c_if(i−1, X(0))` (an
`Op::Reset` would re-measure, which is equivalent but consumes randomness and
would desynchronise seeded comparisons). That circuit (gate-level oracle)
runs through the stock `Circuit::run` on any `Simulator`.

Exactness (tests/shor_scale.rs):
* `semiclassical_equals_full_qft_distribution`: the exact distribution of the
  measured integer, obtained by walking the whole measurement tree
  (`semiclassical_distribution`, no pruning), equals the marginal of the old
  3n-qubit state (`algorithms::shor_full_state`) to < 1e-12 for N = 15
  (7 bases), 21 (4), 33 (2), 35 (2).
* `semiclassical_circuit_runs_through_classic_control`: the plain-Circuit
  version run through `Circuit::run` measures the same bits as the driver for
  the same seed (N = 15, 21).

Speed: the old 3n path needs 24 qubits for N = 253 (≈3.3 s and 533 MiB per
run); the semiclassical path needs 9 qubits (0.015 s, 5 MiB per run; its
random bases differ because the RNG is consumed differently). The qubit count drops from 3n to n+1, so dense simulation goes from
8-bit to 24-bit N in f64 (26-bit in f32).

### 1b. Fused rounds (`src/shor/fused.rs`)

Hypothesis: between rounds the control is |0⟩, so the state is `|0⟩⊗ψ`; one
round maps it to `|0⟩(ψ + e^{iφ}Uψ)/2 + |1⟩(ψ − e^{iφ}Uψ)/2`. Storing only ψ
plus a buffer for Uψ halves memory and replaces ~7 passes over 2^(n+1)
amplitudes with 2 passes over 2^n: one gather (computing `Uψ` and
`‖ψ − e^{iφ}Uψ‖²` together, with the modular index advanced incrementally —
no division per amplitude), one combine. Same linear algebra, different
bookkeeping; the simulated circuit is unchanged.

Exactness: `fused_rounds_match_gate_path` (exact distributions vs the
gate-by-gate n+1-qubit state, < 1e-12 f64, < 1e-5 f32; N = 15, 21, 35, 39)
and `fused_runs_measure_the_same_bits` (identical measured integers for
identical seeds up to N = 1 022 117).

| N (m) | base | dense gate-by-gate | dense fused f64 | dense fused f32 | sparse gate-by-gate | sparse fused |
|---|---|---|---|---|---|---|
| 4 186 067 (22), generic, r = 1 045 494 | 3 | 3.00 s / 197 MiB | 1.39 s / 133 MiB | 1.10 s / 69 MiB | 9.27 s / 179 MiB | 1.19 s / 141 MiB |
| 16 744 463 (24) | 3 | 13.3 s / 773 MiB | 9.7 s → 6.8 s / 517 MiB (before / after fusing P(1) into the gather; the 9.7 s run had load 9.8, so part of this is noise) | — | — | — |
| 137 404 090 531 (37), r = 262 110 | 3 | — | — | — | 9.5 s / 117 MiB | 5.6 s → 1.13 s / 45 MiB (serial → parallel lookups) |

(min of 3 where three runs were taken; `data/shor/generic_m22_ab.txt`,
`ab_fused_1022117.txt`.) Verdict: keep. Dense fused ≈ 2×, sparse fused
≈ 8× over the gate-by-gate path, both bit-identical in outcome.

Negative/neutral: the dense gather is a strided random access (`z·inv mod N`)
and is memory-latency bound; at m = 24 it is most of the 6.8 s.

### 2. Sparse state (`src/sparse.rs`)

`SparseState`: hash map basis index → amplitude (`u64` keys, ≤ 64 qubits,
multiplicative hasher), implements `Simulator`, applies every `Gate` with the
same per-amplitude arithmetic as the dense kernels, drops only amplitudes that
are exactly 0.0 (no thresholds). Permutation gates (X, CNOT, CCX, SWAP) only
re-key the entries that move.

Exactness: `sparse_equals_dense_on_random_circuits` (60 random circuits,
2–10 qubits, all gate types, max |Δamp| < 1e-12),
`sparse_measurement_matches_dense` (same seed → same outcomes),
`sparse_equals_dense_semiclassical` (exact Shor distributions, N = 15, 21, 35,
39), `sparse_and_dense_runs_agree`.

Verdict: wins by orders of magnitude when the order r ≪ 2^n (N ≈ 1e6 rows),
loses when r ≈ N/2 (generic m = 22: 1.19 s fused sparse vs 1.10 s f32 dense;
gate-by-gate sparse 3× slower than dense), and is useless for the Beauregard
oracle (Fourier-space accumulator is dense).

### 3. Gate-level oracle (`src/shor_arith.rs`)

Beauregard 2003, 2n+3 qubits: control (0), x (1..=n), b (n+1 qubits, Fourier
space, QFT without the bit-reversal SWAPs, so qubit j carries `2πb/2^(j+1)`),
ancilla. `φADD(a)` = one phase per qubit; doubly-controlled phases are
decomposed exactly into CPhase/CNOT with the two CNOTs shared across the
whole register (3L CPhase + 2 CNOT per doubly-controlled add);
`φADD(a)MOD(N)` = 5 adders + 4 QFTs + ancilla uncompute; `CMULT`; controlled
`U_a = CMULT(a) · CSWAP · CMULT(a⁻¹)⁻¹`. About 5 000 gates per controlled
`U_a` at n = 8.

Exactness:
* `beauregard_controlled_ua_is_the_permutation`: for N = 15, 21, 35, three
  bases each, every input `|c⟩|x⟩|0⟩|0⟩` (c ∈ {0,1}, x < N) maps to
  `|c⟩|a^c x mod N⟩|0⟩|0⟩` with amplitude 1 (phase included) to 1e-12, no
  leakage (> 1e-24) anywhere else — i.e. the same unitary as the permutation
  oracle on the subspace that matters, ancillas returned clean.
* `beauregard_semiclassical_distribution_matches_permutation`: full Shor
  outcome distributions equal (< 1e-12) for N = 15 (a = 7, 2, 11), 21 (a = 2).
* `beauregard_blocked_matches_unblocked`: the cache-blocked executor
  (`src/blocked.rs`, used for each round's gate list) agrees with
  `apply_gate` per gate.

Blocked vs per-gate, N = 143, a = 2, 19 qubits, min of 3: **8.6 s vs 11.0 s**
(1.3×; `data/shor/ab_blocked_143.txt`). Gate-level runs: 143 (19 q) 8.6 s;
493 = 17 × 29 (21 q) 12.6 s; 1003 = 17 × 59 (23 q) 63.4 s; bases given with
`--base` (chosen so the order is even and gives a factor; the quantum part is
unchanged).

## Classical post-processing

`shor::postprocess` (new paths only): the old continued-fraction rule plus
the standard fix for `gcd(s, r) > 1` (try `k·q` for each convergent
denominator q, k ≤ 256, then strip small factors that keep `a^r = 1`).
Purely classical; it changes how many runs are needed, not the quantum
distribution. Before it, m = 25 f32 needed 19 runs / 200 s
(`m25_fused_f32_factor.txt`); after, 3 runs / 44.7 s. The old path still uses
`algorithms::shor_postprocess` unchanged. Tested by
`postprocess_recovers_order_when_s_and_r_share_a_factor`.

## Rejected / not done

* In-place cycle-following permutation for the dense fused path (would halve
  memory again): sequential random walk, estimated > 5× slower; not tried.
* Sparse + Beauregard: measured, negative (above).
* Ripple-carry (Cuccaro / VBE) Toffoli oracle for a sparse gate-level path:
  done in exp/shor-ripple, then bit-sliced and windowed in round 4 (below).
* Gate-level n = 11 (25 qubits): ~5 min/run estimated, not run (bench-lock
  etiquette).

## Commands

```
B=/mnt/HC_Volume_106989832/dylan/qsim-swarm/bench.sh; Q=./target/release/qsim
$B $Q run shor --modulus 253 --seed 1                                   # (a)
$B $Q run shor --modulus 16744463 --semiclassical --seed 1              # (b)
$B $Q run shor --modulus 16744463 --semiclassical --fused --seed 1      # (b')
$B $Q run shor --modulus 66994189 --semiclassical --fused --f32 --seed 1 --tries 3
$B $Q run shor --modulus 1005973 --semiclassical --fused --sparse --seed 1   # (c)
$B $Q run shor --modulus 8795579227471 --semiclassical --fused --sparse --seed 1
$B $Q run shor --modulus 1003 --semiclassical --oracle beauregard --base 2   # (d)
$B $Q run shor --modulus 143 --semiclassical --oracle beauregard --base 2 --no-blocked
cargo test --release --test shor_scale
```

## Gate-level ripple-carry oracle: the actual circuit at N ≈ 10^6 (exp/shor-ripple)

The agent hit its time limit, so the parent recorded these results from its
bench run. Raw output: `research/data/shor/ripple_benchmarks.txt`; script:
`bench_ripple.sh`.

`src/shor_ripple.rs` builds the controlled modular multiplier only from X,
CNOT and Toffoli gates: a Cuccaro ripple-carry adder, then a modular adder,
then the controlled modular multiplier, with ancillas uncomputed. Those gates
map basis states to basis states, so the exact sparse state never holds more
than 2r nonzero amplitudes. The circuit is simulated **gate by gate** (or by
evaluating each reversible block per sparse branch). Both modes give the same
measured integer.

| N | factors | qubits | total gates | Toffoli | peak nonzero amps | time | peak RSS |
|---|---|---|---|---|---|---|---|
| 143 | 11 × 13 | 28 | 74,540 | 25,846 | 40 | 0.026 s | 5 MiB |
| 1003 | 17 × 59 | 34 | 149,430 | 52,560 | 928 | 0.066 s (gate-by-gate 0.195 s) | 6 MiB |
| 1,003,883 | 991 × 1013 | 64 | 1,149,685 | 416,746 | 18,216 | 4.3 s | 10 MiB |
| **1,005,973** | **997 × 1009** | **64** | **1,148,440** | **415,498** | **83,664** | **10.8 s** | **14 MiB** |

The gate-level Beauregard path was limited to 10-bit N because its
Fourier-space adder makes the state dense. The ripple-carry circuit keeps it
sparse, so **the full gate-level Shor circuit for a 20-bit semiprime (64
qubits, 1.15 million gates) runs exactly in about 11 s and 14 MB**. For the
same seed, the measured value, the order (r = 41,832 for 1,005,973) and the
base match the permutation-oracle path.

The caveat still applies: the simulation cost tracks the order r, which is
exactly why Shor is easy to simulate in this structured way and hard in
general. This is a demonstration of exact gate-level simulation at scale, not
a classical speedup for factoring.

## Round 4 (exp/shor-r4): the gate-level circuit to 31-bit generic N, and the law that sets its cost

Machines: **Mac** = Dylan's MacBook Pro, Apple M1 Pro (8 cores, NEON), 16 GB,
release build, every timing under `/tmp/qsim-mac-bench.lock` (another agent
was benchmarking between my cells, so the 1-minute load average printed in
the log, 5–13, includes the tail of its runs and my own 8 threads); peak RSS
from `/usr/bin/time -l`. **VPS** = development only (load 5–19 from other
agents during this round, so no VPS timing is claimed below). Raw logs:
`research/data/shor_r4/` (`mac_runs.log` has every Mac run and the scripts).

### What changed

1. **Bit-sliced branch tracking** (`src/shor/sliced.rs`, `--sliced`). The
   controlled-`U` circuit of the ripple oracle (and of the new windowed
   oracle) contains only X, CNOT and CCX, so it maps each computational
   basis state to one basis state; the exact state of the whole circuit is a
   list of (basis state, amplitude) branches. Every gate is applied to every
   branch, but 64·L branches at a time: a batch is stored as one `[u64; L]`
   word per qubit and a gate is `w[t] ^= w[c1] & w[c2]` (an all-ones word
   stands in for missing controls). There is no qubit limit (the old sparse
   state used `u64` keys, so 3n+4 ≤ 64, n ≤ 20). Per round, the circuit is
   evaluated on `|1⟩|x⟩|0…⟩` **and** `|0⟩|x⟩|0…⟩` for every stored `x`
   (the control-0 half is half the work; it is the real circuit, so it is
   simulated, not assumed), and every output is **checked**: control
   unchanged, every ancilla qubit 0, control-0 branch returns `x`, distinct
   inputs give distinct outputs. If a check failed the run would panic (the
   bookkeeping drops the ancilla qubits between rounds, which is only exact
   if they are clean). The control-qubit algebra (H, phase correction, H,
   measure) is the same linear algebra as `fused.rs`: `Uψ` comes from the
   gate-evaluated outputs via a sort-merge join on the output keys.
   Engineering on top: 64×64 block bit transposes in/out of the slices
   (eval −35 % on the VPS), AVX2 runtime dispatch (~5–10 %, noisy),
   `L = 16` (Mac: 1.00 s vs 1.16 s for L = 8, 1.05 s for L = 32 at
   N = 4 297 567), the join on `(Ux, ψ_x)` tuples written in place by the
   evaluator (peak RSS at 28 bits 3.71 → 2.82 GB), and the final collapse
   skipped (after the last P(1) the measured integer is complete).
2. **Windowed oracle** (`src/shor_window.rs`, `--oracle windowed --window w`;
   Gidney 2019, arXiv:1905.07682), also only X/CNOT/CCX: per window of `w`
   bits of `x`, a table lookup (QROM: AND-chain over control + address bits,
   CNOT fan-out of `T[v] = v·a·2^(kw) mod N` into a lookup register, chain
   prefixes shared between consecutive addresses) feeds ONE quantum–quantum
   modular addition (same 5-adder VBE/Beauregard structure, Cuccaro adders),
   then the lookup is undone. `4n + 4 + w` qubits. Gate counts (whole
   circuit, `gate_counts.txt`): w = 4 is optimal for n ≤ 56, with
   **2.7–3.3× fewer gates and 2.1–2.8× fewer Toffolis than the ripple
   oracle** (n = 28: 1.01 M vs 2.89 M gates; n = 62: 9.4 M vs 31.0 M).
3. CLI: `--sliced`, `--oracle windowed`, `--window`, `--f32` (amplitudes);
   output adds `gate_branch_ops` (gate applications × branches, counted).
   `QSIM_SLICE_PROFILE=1` prints a time breakdown, `QSIM_SLICE_TRACE=1` the
   per-round P(1) and support.

### Exactness (differential tests, all green: `cargo test --release --test shor_scale`, lib tests `shor`)

* `sliced_eval_matches_per_key_eval` (unit): the bit-sliced evaluator equals
  the existing per-key evaluator on every input `x < N`, both control values,
  N = 15, 21, 35, 143.
* `sliced_ripple_distribution_matches_gate_by_gate_and_permutation`: exact
  outcome distributions (whole measurement tree) of sliced-ripple = the
  gate-by-gate `SparseState` run of the same circuit (every gate through
  `apply_gate`, N = 15, 21) = the permutation oracle (N = 15, 21, 33, 35;
  up to 7 bases), max |Δp| < 1e-12.
* `windowed_distribution_matches_permutation`: windowed oracle, w = 1..4,
  both sliced and gate-by-gate on `SparseState`, vs the permutation oracle,
  < 1e-12 (N = 15, 21, 35).
* `windowed_controlled_ua_exhaustive_small` / `lookup_exhaustive` (unit):
  every input of controlled-U (N = 15…63, w = 1..4, 5 bases, both control
  values) maps to `a^c x mod N` with all ancillas clean.
* `sliced_runs_measure_the_same_bits`: same seed → same measured integer for
  sliced ripple, sliced windowed, the old sparse ripple path (≤ 1003) and the
  fused permutation path, up to N = 1 005 973.
* `sliced_beyond_64_qubits_matches_permutation`: 24-bit N (76 and 104
  qubits, beyond the old u64-key limit), same measured integers as the
  permutation path.
* `sliced_f32_close_to_f64`: distributions within 1e-5.

### Results (Mac; min of 3 where 3 runs are listed)

**Same N as the old record, same circuit** (N = 1 005 973, a = 980 062,
seed 1, interleaved ×3; identical measured integer 475 634 978 396 in all 9
runs):

| path | gates | time | peak RSS |
|---|---|---|---|
| (e) ripple, sparse `u64` keys, block evaluated per key (exp/shor-ripple) | 1 148 438 | 2.468 s | 15 MB |
| (f) ripple, **sliced** | 1 148 438 | 0.078 s (**32×**) | 12 MB |
| (g) **windowed w=4, sliced** | 501 948 | 0.050 s (**49×**) | 9 MB |

**Scaling with random bases** (`--oracle windowed --window 4 --sliced`, no
`--base`; N are the first balanced semiprime per bit size from a seeded
generator (`research/data/shor_r4/gen_instances.py`, `random.seed(1)`), not
selected on their orders):

| N | bits | qubits | gates / run | a (random) | r | ν₂(r) | time / run | peak RSS | outcome |
|---|---|---|---|---|---|---|---|---|---|
| 10 161 323 | 24 | 104 | 0.82 M | 9 899 614 | 2 538 720 | 5 | 0.28 s (audit re-run; 0.95 s at load 35 on an earlier commit) | 105 MB (was 133 MB before the final collapse was skipped) | factored, run 1 |
| 18 942 389 | 25 | 108 | 0.96 M | 18 454 521 | 157 776 | 4 | 0.27 s | 21 MB | factored, run 1 |
| 43 584 217 | 26 | 112 | 1.04 M | 42 001 717 | 21 785 498 | 1 | 18.5 s | 0.96 GB | factored, run 2 (seed 2) |
| 82 337 219 | 27 | 116 | 1.14 M | 80 216 601 | 1 371 968 | 6 | 0.20 s | 76 MB | factored, run 1 |
| 221 643 407 | 28 | 120 | 1.26 M | 215 934 921 | 110 806 800 = λ | 4 | 16.1 s (f32 15.2 s) | 2.82 GB (f32 2.02 GB) | factored, run 1 |
| 282 304 153 | 29 | 124 | 1.46 M | 83 936 318 | 47 044 830 = λ | 1 | 44.2 s | 1.86 GB | factored (seed 2, run 1; seed 1: 3 runs found r, a^(r/2) = −1) |
| **1 537 596 787** | **31** | **132** | **1.70 M** | 457 167 243 | **256 252 500 = λ** | 2 | **134.4 s (f32)** | **4.28 GB** | **factored** (seed 2, run 1; seed 1: r = 42 708 750 found, a^(r/2) = −1) |

Re-confirmed after rebasing onto main 643b6bd (branch head 7ccb497, same
machine, one lock): the 31-bit record run took **133.4 s**, same base, same
measured integer 2 059 039 373 337 077 151, 4.28 GB; the 1 005 973 A/B gave
2.492 / 0.076 / 0.049 s (min of 3).

Every run's order is the true multiplicative order (checked classically
afterwards); the failures are the classical part of Shor (odd r or
a^(r/2) ≡ −1). For 43 584 217 and 282 304 153 both p − 1 and q − 1 have
2-adic valuation 1, so P(success per base) = 1/2 exactly.

Special N (same caveat as row (c): N = p(2p − 1) has λ = 2(p − 1) ≈ √(2N)
and is classically trivial; included to show cost tracks r, not N):
43-bit 8 795 579 227 471 (row (c)'s N) in 2.54 s / 180 qubits / 4.35 M
gates; 48-bit 248 376 613 912 741 in 24.8 s / 200 qubits; **52-bit
3 384 163 410 217 561 = 41 134 921 × 82 269 841 in 60.5 s / 216 qubits /
7.41 M gates / 2.40 GB**, random base each time, factored on the first run.

### Follow-up: superoptimised oracle (exp/superopt)

`--oracle windowed-opt` (`src/shor_superopt.rs`, `research/superopt.md`) is
the same layout and arithmetic with cheaper, proved-correct blocks:
unary-iteration lookups (the round-4 lookup recomputed its whole AND chain
for every address), optimal fan-out, a comparator-based modular adder, and
SAT-derived window rewrites. On the 31-bit record run: 1.70 M → 1.04 M
gates, 528 k → 261 k Toffolis, and 135.7 s → 97.6 s on the Mac (min of 3),
with the same measured integer.

### Follow-up: measurement-based uncomputation (exp/mbu-shor)

`--oracle windowed-mbu-lookup` / `windowed-mbu` (`src/shor_mbu.rs`,
`research/mbu-shor.md`). The sliced engine gains a per-branch sign word, so
X-basis measurements of deterministic ancillas with Z/CZ fix-ups run
exactly; the engine asserts that every branch has sign +1 after each
block. Temporary-AND lookups, a measurement-based unlookup and a measured
modular-adder flag give 31-bit 261 k → 218 k Toffolis at the same 132
qubits, and 98.0 → 89.3 s on the Mac (min of 3, same measured integer).
Gidney adders on top take Toffolis to 119 k (−54 %) at 162 qubits. That
variant has +20 % more engine steps, so it runs 112 s. With the
lookup-only oracle, the first seeded 32-bit generic N, 3 631 204 201 =
58 907 × 61 643, was factored at gate level: 136 qubits, peak support
1.30·10⁸ exactly as predicted, 6.3 GB, 498 s on 2 threads. The base was
chosen by seed so that its order fits in RAM; that choice used λ(N). See
§5b there.

### The law: reachable support of the gate-level circuit, and the cost it implies (new)

Before round `i` (bits `0..i−1` measured, `U^(2^(t−1)) … U^(2^(t−i))`
applied), the work register can only hold `a^(m·2^(t−i))` for `m < 2^i`,
so its support has at most

    B_i = min(2^i, r / gcd(r, 2^(t−i)))

elements; between rounds every ancilla is clean (checked), so this is the
support of the **whole** (4n+8)-qubit state up to the factor 2 of the
control, and inside a round permutation gates cannot change the number of
branches. Measured on the real circuit (`examples/shor_support.rs`,
windowed w = 4, random semiprimes 8–24 bits, random bases, 1–2 seeds):

* **|S_i| = B_i in 5 076 of 5 088 rounds** (8–20 bits: 4 356/4 368;
  21–24 bits: 720/720). It is never larger. The 12 exceptions are exact
  destructive interference for particular measured prefixes when 2^i
  barely exceeds the period (e.g. N = 143, r = 20, prefix y = 4 at i = 3:
  the pairs `m, m+5` cancel, support 2 instead of 5; that prefix has
  probability 1/8).
* The counted work equals `W = Σ_i 2·B_i·G_i` exactly in 148/156 runs
  (8–20 bits, exact per-round gate counts G_i; ratio ≥ 0.944 in the
  others, the cancellation cases). For the Mac runs above, using the
  average Ḡ = total gates / t instead of G_i (`cost_law.py`,
  `cost_law_records.txt`), prediction and counter agree within 0.3 %
  (31-bit: predicted 1.340e14, counted 1.341e14; 52-bit: 6.864e13 vs
  6.860e13); the largest Ḡ-induced gap seen was 1.4 % (23-bit dev run, whose
  support equalled B_i in all 46 rounds).

With `t = 2n`, `ν = ν₂(r)` and `r_odd = r / 2^ν` (and 2^n > N > r, so
`gcd(r, 2^(t−i)) = 2^ν` except in the last ν rounds) this sums to

    Σ_i B_i ≈ r_odd · (2n − log₂ r) + r,    W ≈ 2·Ḡ·Σ_i B_i,    Ḡ = Θ(n²) gates/round

(audit correction, exp/shor-r4-audit: this line first read `+ 2r`. The rounds
`i ≤ t − ν` contribute `r_odd · (t − ν − log₂ r_odd) + O(r_odd)`, the last ν
rounds `r_odd·(2 + 4 + … + 2^(ν−1)) = r − 2·r_odd`, so the tail is `+ r`. Against
the exact `Σ B_i` of all 12 records in `cost_law_records.txt` the `+ r` form is
within 0.3 % (the 0.3 % W match quoted above uses the exact `Σ B_i`, not this
closed form); the `+ 2r` form was 6–66 % high.)

and the peak support is `max(r_odd, r/2)`. Consequences, all visible in the
table: the cost is **linear in the order r, quadratic in n through the gate
count, and divided by 2^ν₂(r) in its leading term** — the 27-bit N runs
in 0.2 s because its random base has r = 2^6 · 21 437, while the 26-bit N
needs 18.5 s with r = 2 · 10 892 749. The final ν rounds cost about
`2·Ḡ·(r − r_odd)` (≈ 2·Ḡ·r for large ν; audit correction, first written `4·Ḡ·r`).
Memory: the stored state is 16 / 24 B per support element (f32 / f64), but
during a round ψ and the `(Ux, ψ_x)` join array coexist, so measured peak RSS is
≈ 33 B (f32, 31-bit: 4.28 GB / 1.28·10^8) and ≈ 51 B (f64, 28-bit: 2.82 GB /
5.5·10^7) per element of the peak support (audit correction, first written
~16–24 B).

**What is exponential in what.** For a generic semiprime and a random base,
r ≈ λ(N)/small ≈ N/c, so the work is Θ(N · n² · n/2^ν) — exponential in the
bit length, exactly as classical order finding by brute force. The
simulation is exact and simulates every gate of a circuit you could compile
for hardware (X/CNOT/Toffoli, 4n+8 qubits), but its cost is set by the
classical difficulty of the instance; it is not a factoring speed-up, and
nothing here contradicts the expected hardness of simulating Shor in
general. The structured 43/48/52-bit rows only show that r, not N, is what
matters. The memory wall on the 16 GB Mac is a peak support of ≈ 2·10^8
(r ≈ 4·10^8); the next seeded generic N (30 bits: λ = 4.3·10^8; 32 bits:
λ = 1.8·10^9, ν₂ = 1) would need ≈ 7 GB / 3.5 min and ≈ 29 GB respectively
— not run.

**Audit note (exp/shor-r4-audit) on the word "record".** It means the largest
N this repo has factored by simulating every gate of an X/CNOT/Toffoli Shor
circuit. It is not a record for simulated Shor in general: e.g. Willsch et al.
2023 (arXiv:2308.05047) factored the 39-bit 549 755 813 701 = 712 321 × 771 781
by simulating Shor's algorithm on a GPU supercomputer (a different circuit
construction, so not a like-for-like comparison), and no simulation of this
kind is a classical factoring speed-up (see above).

### f32 vs f64 (lever 3)

Exact distributions, sliced windowed, f32 vs f64 amplitudes, N = 15 … 247
(66 (N, a) pairs, `precision_f32_f64.tsv`): total-variation distance ≤
1.5e-7; the textbook success probability (r appears as a continued-fraction
convergent of y/2^t) changes by ≤ 8.7e-8. On full runs: N = 4 297 567,
max |ΔP(1)| over 46 rounds = 8.8e-8, same measured integer; N = 221 643 407
(28 bits) same measured integer, 15.2 s vs 16.1 s, peak RSS 2.02 vs 2.82 GB
(−28 %). Verdict: f32 is safe for factor recovery; use it for the largest
runs (the 31-bit record is f32).

### Rejected / neutral this round

* **Phase folding (exp/phasepoly, audited SOUND by qsim-audit-merge,
  9ab6c29)**: not applicable to the record path. The ripple/windowed
  circuits contain no phase gates; folding only reduces T-count after
  Toffolis are lowered to Clifford+T (auditor: controlled-U n = 2, T 742 →
  428), and that lowering inserts H gates, which would break the
  basis-state branch structure that makes this simulator work. T-count is a
  hardware-cost metric; for that the windowed oracle's 2.1–2.8× Toffoli
  reduction is the relevant number here. Not pursued for the dense
  Beauregard path either (it stays at 2n+3 qubits dense, 10 bits).
* Skipping the control-0 half of each round would halve the work but stops
  simulating half the circuit; not done (`skip_ctrl0` exists only as an
  off-by-default knob).
* Presized collapse buffers: within noise. Lane width 8/16/32: ≤ 15 %.
  AVX2 vs SSE2 on the VPS: ~5–10 % (VPS loaded, not claimed).
* Time breakdown at 31 bits (f32): gate evaluation 81 % (55 s control-1 +
  53 s control-0), sort 12 %, collapse 6 %, P(1) 1 %. Remaining headroom is
  in the evaluator (≈ 1.2·10^12 gate·branch/s on 8 M1 cores, i.e. ≈ 0.4 ns
  per gate per 64-branch word per core) and in r itself.

### Commands (round 4)

```
cargo build --release
Q=./target/release/qsim
# record: 31-bit generic semiprime, random base, gate-level windowed circuit (Mac, f32, 134 s, 4.3 GB)
$Q run shor --modulus 1537596787 --semiclassical --oracle windowed --window 4 --sliced --f32 --seed 2 --tries 1
# 28-bit / 29-bit generic
$Q run shor --modulus 221643407 --semiclassical --oracle windowed --window 4 --sliced --seed 1 --tries 1
$Q run shor --modulus 282304153 --semiclassical --oracle windowed --window 4 --sliced --seed 2 --tries 1
# 52-bit special N (caveat: classically trivial), 216 qubits
$Q run shor --modulus 3384163410217561 --semiclassical --oracle windowed --window 4 --sliced --seed 1 --tries 1
# A/B vs exp/shor-ripple on the old record
$Q run shor --modulus 1005973 --semiclassical --oracle ripple --sparse --base 980062 --seed 1
$Q run shor --modulus 1005973 --semiclassical --oracle ripple --sliced --base 980062 --seed 1
$Q run shor --modulus 1005973 --semiclassical --oracle windowed --sliced --base 980062 --seed 1
# support law / cost law, precision
cargo run --release --example shor_support -- 8 20 6 2
cargo run --release --example shor_precision
python3 research/data/shor_r4/cost_law.py 1537596787 256252500 1704645
cargo test --release --test shor_scale
```

## Noise (exp/shor-noise)

How much circuit-level Pauli noise the round-4 gate-level circuit tolerates,
measured by exact noisy trajectories at 10–24 bits: see
[`shor-noise.md`](shor-noise.md).
