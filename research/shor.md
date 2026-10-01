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

Runs use `qsim run shor --modulus N --semiclassical [--fused] [--sparse] [--f32]
[--oracle beauregard] [--base a] --seed 1`; without `--base` the bases are
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
  the right next step, not implemented.
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
