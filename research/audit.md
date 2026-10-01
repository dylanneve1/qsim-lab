# Audit notebook (exp/audit)

Independent verification of the speed-swarm's claims. Machine: shared 4 vCPU
AMD EPYC-Rome VM (AVX2), 7.7 GB RAM, ~8 other agents compiling concurrently;
load average 12–19 during the swarm. All timings go through
`qsim-swarm/bench.sh` (flock) and are reported as **interleaved old-vs-new
ratios, min-of-5**, because absolute times are dominated by box load.

## 1. Differential fuzz harness — `tests/differential_fuzz.rs`

Design: a deliberately naive dense reference simulator (`RefSv`) lives in
the test file with its *own* gate matrices and an out-of-place
`O(2^n)`-per-gate loop. Nothing is shared with `src/` except the `Gate`
enum, so any rewritten kernel is checked against code it cannot have
co-evolved with.

Generators are edge-biased: qubit 0 / top qubit chosen 50% of the time,
pairs biased to (0, n−1), adjacent, (0,1), (n−2,n−1), both orders; Toffoli
triples (0, n/2, n−1), consecutive, random, all orderings; angles drawn from
{0, π/4, π/2, π, 3π/2, 2π, −π/2, −π} ± {0, 1e−15, 1e−9, 1e−6} or uniform in
[−4π, 4π]; n ∈ {1..11, 13} plus 16 and 18 qubits (above the rayon
threshold `PAR_MIN_LEN = 2^14`).

| test | checks | tolerance |
|---|---|---|
| `sv_f64_matches_reference` | `apply_circuit` and `Circuit::run` amplitudes | ≤ 1e−12 |
| `sv_f32_matches_reference` | f32 amplitudes, depth < 60 | ≤ 1e−5 |
| `sv_large_registers_match_reference` | 16/18 qubits, parallel kernels, f64+f32 | 1e−12 / 1e−5 |
| `sv_measure_reset_match_reference` | mid-circuit measure/reset, collapsed state, repeat measurement | outcome prob > 0, Δ < 1e−10 |
| `sv_sampling_distribution` | `sample()` chi-square vs reference | ~6σ bound |
| `tableau_probabilities_exact` | every outcome probability **bit-exact** dyadic; stabilizer signs vs ⟨P⟩ | exact |
| `tableau_measure_reset_match_reference` | `peek` determinism, measurement, reset, repeat | exact |
| `tableau_sampling_distribution` | `sample()` support + chi-square | exact support, 6σ |
| `pauli_path_matches_reference` | random Pauli ⟨P⟩, ≤ 8 non-Clifford gates incl. arbitrary Rz/Rx/Ry/CPhase; 3-qubit marginals | ≤ 1e−10 |
| `mps_exact_matches_reference` | MPS with untruncating bond, amplitudes incl. phase | ≤ 1e−9 |
| `memory_caps_reject_oversized_registers` | caps return `Err` | — |

Scale with `QSIM_FUZZ_ITERS=<k>` (default 1); change seeds with
`QSIM_FUZZ_SEED`. Running against another branch:

```
git worktree add wt/<topic> origin/exp/<topic>
cp tests/differential_fuzz.rs wt/<topic>/tests/
cd wt/<topic> && cargo test --release --test differential_fuzz
```

### Previous auditor's leftovers
`examples/adversarial_fuzz.rs` + `tests/adversarial_fuzz.rs` (uncommitted)
did **not compile** against the current API (`StateVector::measure`/`reset`
do not exist; it is `measure_qubit`/`reset_qubit`), included an example with
`fn main` into a test via `#[path]` (dead-code under clippy), and its memory
test allocated 512 MiB registers on a shared box. Its ideas (n=1, extreme
angles, (0,n−1) pairs, repeated measurement, stabilizer sign check, memory
caps) are all subsumed by the new harness; the files were dropped.

