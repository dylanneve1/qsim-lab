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

Sensitivity check (mutation test): perturbing the state-vector CPhase angle
by 1e−9 rad (`from_polar(1.0, th + 1e-9)`) is caught immediately by
`sv_f64_matches_reference` (Δ = 4.0e−10 at n = 2). On main @ 86e5d67 all 11
tests pass in 8 s (release).

Per-branch adapters live in `audit-adapters/` (copied into `tests/` /
`examples/` of a worktree of the branch under audit, together with
`tests/audit_common/`).

### Previous auditor's leftovers
`examples/adversarial_fuzz.rs` + `tests/adversarial_fuzz.rs` (uncommitted)
did **not compile** against the current API (`StateVector::measure`/`reset`
do not exist; it is `measure_qubit`/`reset_qubit`), included an example with
`fn main` into a test via `#[path]` (dead-code under clippy), and its memory
test allocated 512 MiB registers on a shared box. Its ideas (n=1, extreme
angles, (0,n−1) pairs, repeated measurement, stabilizer sign check, memory
caps) are all subsumed by the new harness; the files were dropped.

## 2. QEC on main (86e5d67) — surface-code fast detector sampler

`tests/qec_dem_audit.rs` (`#[ignore]`d because it fails on main; run with
`cargo test --release --test qec_dem_audit -- --ignored --nocapture`)
samples the d=3, rounds=3 memory experiment both ways — full noisy tableau
(`build_circuit` + `run_noisy`) and the `error_mechanisms` list that
`run_experiment_fast` samples — at `NoiseModel::circuit_level(0.005, 0.005)`,
20 000 shots each, and compares per-detector firing rates and logical rates
with two-proportion z-tests (Bonferroni, |z| > 4.9 ⇒ disagree).

Result (raw output `research/data/audit/qec_dem_main_86e5d67.txt`):

| quantity | tableau (circuit) | fast sampler | z |
|---|---|---|---|
| bulk detector firing rate (dets 5,6,9,10,13,14) | 0.058–0.061 | 0.023–0.032 | +14 … +17.7 |
| boundary detector rate (dets 4,7,8,11,12,15) | 0.030–0.038 | 0.015–0.020 | +8.5 … +11.6 |
| round-0 detectors (0–3) | 0.011–0.020 | 0.014–0.025 | −0.8 … −4.2 |
| raw logical flip | 0.0767 | 0.0573 | +7.8 |
| **decoded logical error rate** | **0.0250** | **0.0043** | **+17.3** |

**Verdict: BUG.** The fast sampler under-reports the decoded logical error
rate by **~5.9×** at p = 0.5% (12/16 detectors disagree beyond 4.9σ). Causes,
from reading `src/qec/surface.rs`:

1. `build_error_mechanisms` is a hand-written phenomenological model (one
   X flip per data qubit per round at rate `p_1q`, one measurement flip per
   check at rate `p_1q`). It ignores `p_2q` (the dominant source: 4 CNOTs per
   check, each depolarising), CNOT propagation of ancilla errors to data
   (hook errors → weight-2 data errors / diagonal edges), Y errors, reset
   errors and `p_meas`; every mechanism uses `p_1q` regardless of type.
2. `run_experiment` silently switches from tableau to the fast sampler when
   `shots > 300` or `d > 5`, so the *same* experiment returns a ~6× lower
   logical error rate just by asking for more shots. Any threshold plot made
   with `run_experiment` mixes the two models.
3. `tests/surface.rs::fast_sampling_matches_tableau_sampling` only asserts
   both rates are < 0.15, so it cannot detect either problem (0.0043 and
   0.025 both pass).
4. The decoding graph has the same phenomenological structure (no diagonal
   hook edges, uniform weights), so even the tableau path decodes with a
   mismatched model; this inflates tableau logical error rates but is not a
   correctness bug of the simulator.

What a fix must pass: the same test with mechanisms *derived from the
circuit* (propagate every single fault of `run_noisy`'s noise channels
through the Clifford circuit to detectors/observable, merge identical
symptoms with p ← p₁(1−p₂)+p₂(1−p₁)), and agreement at d=3 within the
Bonferroni bound for every detector and both logical rates.

## 3. exp/sv @ 973f40b — cache-blocked executor (`apply_circuit_blocked`)

No PR yet; audited pre-emptively.

**Accuracy.** `audit-adapters/sv_blocked.rs`: blocked executor vs the
independent reference, n ∈ {1..11, 13} with 5 configs per circuit (default +
4 random adversarial: `block_bytes` ∈ {8 B … 256 KiB}, `slots` 0–8, fusion
on/off, `small_n` 0–3 forcing the multi-chunk path at tiny n), plus 16/18
qubits (default and 4 KiB/2-slot configs) and 500-gate single-qubit fusion
runs. `QSIM_FUZZ_ITERS=10`: **pass**, worst |Δamp| f64 = 1.0e−15, f32 =
3.7e−7. Their own `tests/blocked.rs` compares against the in-crate
gate-by-gate path only; this adds an independent oracle and edge-biased
gates (Ccx in all orderings, Swap, CPhase with angles at 0/π/2π ± ε).

**Merge hazard.** exp/sv is based on e86e91b; main's `Op` enum gained
Reset/ClassicControlled/noise variants, so the exhaustive
`match op { Op::Gate, Op::Measure }` in `apply_circuit_blocked` will not
compile after merging main. Told the sv agent.

**Speed** (`audit-adapters/bench_sv_blocked.rs`: base and blocked runs
*interleaved* in one process, 5 reps each, min reported; default
`BlockConfig`; portable release build; through `bench.sh`; raw:
`research/data/audit/sv_973f40b_bench.txt`):

| workload | n | prec | base min (s) | blocked min (s) | speedup | load |
|---|---|---|---|---|---|---|
| qft | 22 | f32 | 0.442 | 0.073 | 6.05× | 11.4 |
| brick (20 layers) | 22 | f32 | 3.735 | 0.775 | 4.82× | 11.4→10.5 |
| qft | 22 | f64 | 0.736 | 0.101 | 7.29× | 10.5 |
| qft | 24 | f32 | 1.724 | 0.214 | 8.05× | 10.3 |
| brick (20 layers) | 24 | f32 | 13.92 | 2.80 | 4.97× | 10.0→8.3 |

Base-run spread is large (e.g. qft-24 base 1.72–3.93 s) because of box
load; blocked runs are tighter. Even taking the *worst* blocked rep against
the *best* base rep the ratio stays ≥ 3.5×. No headline claim to compare
against yet (EXPERIMENTS-sv.md has only the profile).

## 4. PR #1 (improvements-and-optimizations @ ca89cdc) — gates, optimize, QASM

Adapter `audit-adapters/pr1_gates_qasm.rs`; reference matrices for the new
gates written from textbook/Qiskit definitions in the adapter.
`QSIM_FUZZ_ITERS=5`. Verdict posted on the PR: **BUG** (QASM I/O only).

| check | result |
|---|---|
| I/Sx/Sxdg/U/ISwap/ISwapdg on SV f64/f32, exact MPS | pass (≤1e-12 / 1e-5 / 1e-9) |
| new Cliffords on tableau (exact probabilities, stabilizer signs) | pass |
| new gates in Pauli-path ⟨P⟩ | pass (≤1e-10) |
| `optimize()` preserves state | pass **up to global phase**; drops Rx/Ry/Rz(2πk) = −I (278/1000 circuits differ by a sign) |
| QASM round trip of unitary circuits | pass (≤1e-12) |
| `parse_param` precedence | **BUG**: `pi/2*3`→π/6, `1/2/4`→2; `pi-1`, `pi/2+pi/4` rejected |
| `to_qasm` with `c_if` / noise ops | **BUG**: silently dropped |
| `creg` size with repeated measurement | **BUG**: `creg c[n]` but writes `c[k]` for k ≥ n |
| `reset_all` | pass |

Side finding (pre-existing on main, not PR #1): `Mps` default relative
cutoff `1e-14` on s²/Σs² truncates singular values ~1e-7 relative, so
"exact" MPS (bond ≥ Schmidt rank) differs from the state vector by up to
~3e-8 in amplitude on 5-qubit circuits with iSWAP/U gates. With
`set_cutoff(0.0)` it is exact to 1e-15. The audit's exact-MPS tests now set
cutoff 0; the MPS agent's accuracy claims should state which cutoff they use.

## 5. exp/sv HEAD ea41235 (+ eb03e29) and the merge with main

**Merge with main.** Textually clean, but does not compile:
`apply_circuit_blocked` matches `Op` exhaustively with only `Gate`/`Measure`
(E0004: Reset, ClassicControlled, XFlip, … not covered). One-line fix: a
`_ =>` panic arm. With it, **973f40b + main passes every test in the tree**
(lib 34, algorithms, blocked, cross_check, identities, noise, properties,
repetition, surface) plus `differential_fuzz` and the blocked adapter at
`QSIM_FUZZ_ITERS=5`. Verdict for 973f40b: **REPRODUCED, safe to merge**
(with the fix).

**BUG at HEAD ea41235** (regression introduced by eb03e29 "real-rotation/
phase split of fused 1q gates"). `apply_circuit_blocked` with
`BlockConfig::default()` returns wrong amplitudes; greedy delta-debugging
(`audit-adapters/sv_blocked_repro_ea41235.rs`) shrinks the fuzz failure
(seed 2698035465) to 10 gates on 5 qubits:

```
[Y(0), Cnot(0,4), Rx(0, π/4), Z(0), H(0), X(0), S(0), Z(0), T(0), H(0)]
max |Δamp| = 0.261 vs reference
```

It fails only with `fuse_1q && split_phases` (either `schedule_diag`); it
passes without the leading `Y(0)` or without the CNOT, and with
`split_phases = false`. So the phase factor of a split fused run is applied
on the wrong side of a non-diagonal op on the same qubit. The branch's own
`tests/blocked.rs` (random brickwork / universal circuits vs gate-by-gate)
did not catch it. Verdict for ea41235: **BUG — do not merge**. (The sv agent
had already exited, so the repro was handed to the parent.)

## 6. exp/qec @ 100c2dc — circuit-derived detector error model (WIP)

Adapter `audit-adapters/qec_dem_exp_qec.rs` (same statistics as §2, using
`SurfaceCode::with_noise` and `ErrorMechanism::probability`; channel mix via
`QSIM_DEM_NOISE`). d=3, rounds=3, p=0.005, 20 000 shots/side unless noted.

| noise | detectors beyond 4.9σ | raw logical (tab / DEM) | decoded (tab / DEM) | verdict |
|---|---|---|---|---|
| circuit_level (all channels) | 1 (worst 5.5σ, 15/16 z>0) | 0.0767 / 0.0746 | 0.0250 / 0.0221 | FAIL |
| no reset errors | 0 | 0.0783 / 0.0757 | 0.0218 / 0.0236 | pass |
| reset only (p=0.02) | **12 (worst 28.6σ)** | 0 / 0 | 0 / 0 | **FAIL** |
| readout only (0.02) | 0 | 0.0597 / 0.0607 | 0.0093 / 0.0083 | pass |
| 2q depolarising only (0.01) | 0 | 0.1173 / 0.1119 | 0.0475 / 0.0446 | pass |
| 1q depolarising only (0.02) | 0 | 0 / 0 | 0 / 0 | pass |
| **all channels, with audit fix, 80k shots** | **0 (max |z| 2.4)** | 0.07306 / 0.07340 | 0.02371 / 0.02390 (z=−0.2) | **pass** |

**BUG 1 (root cause of the mismatch).** `build_circuit_derived_dem`
propagates an injected fault through `ops[op_idx..]`, i.e. starting *with*
the op that produced it. A reset error (X after reset) is therefore cleared
by the very Reset it follows, so reset noise is missing from the DEM
entirely. Gate errors are also conjugated by their own gate; that happens to
be harmless for uniform depolarising channels (Clifford-invariant) but is
wrong for any biased channel. One-line fix verified
(`research/data/audit/qec_100c2dc_reset_fix.patch`): start at `op_idx + 1`
except for measurements (readout flip modelled as X just before). Raw 80k
output: `research/data/audit/qec_100c2dc_fixed_80k.txt`.

**BUG 2.** `SurfaceCode::new(d, r)` builds the DEM with
`NoiseModel::uniform(1.0)` and `run_experiment_fast(&noise, …)` ignores its
noise argument (`_noise`), so `SurfaceCode::new(3,3).run_experiment_fast(
&NoiseModel::none(), 500, …)` reports 262/500 logical errors. Two tests in
the branch's own `tests/surface.rs` fail because of it.

**Other.** `run_experiment` still switches tableau→DEM at shots > 300 (now
harmless once BUG 1 is fixed, since both agree, but surprising); three
unused-variable warnings will fail `clippy -D warnings`.

Verdict: **BUG — not mergeable as is; mergeable after the one-line
propagation fix + making the fast path use the caller's noise model.**
Findings and fix sent to the QEC agent.

## 7. exp/compiler @ 49facf8 — compile module (peephole, light cone, components, suffix, Clifford prefix, plans)

**Merge.** Based on e86e91b; merged with main (4d151f4) it fails to compile
at 10 exhaustive `match op` sites in `src/compile/{analysis,peephole,plan}.rs`
(new `Op` variants). Needs a real port (non-gate ops as barriers / fallback),
not a one-liner. Audited on its own base.

**Accuracy** — `audit-adapters/compiler_plans.rs`, `QSIM_FUZZ_ITERS=5`, all pass:

| target | check | worst |
|---|---|---|
| `optimize` (peephole) | e^{iφ}·U_opt vs U on cancellation-rich circuits (inverse pairs, commuting Rz in between, rotations summing to 2πk) | 3.9e-14 |
| `compile_unitary` + `statevector::<f64>` | amplitudes incl. global phase; random pass subsets; universal / Clifford / Clifford+T / disconnected+idle / monomial-suffix families; n = 1…12 | 2.8e-14 |
| `factored().amplitude(x)` | every x | ≤1e-12 |
| `statevector::<f32>` | | ≤1e-5 |
| `expectation_z_product` | random qubit subsets incl. 0 and n−1 | ≤1e-10 |
| `compile_sampling` terminal | `exact_distribution()` vs reference; chi-square of `sample()` f64 (4000 shots) and f32 (2000); measure_all / random subset in random order / duplicate measures | ≤1e-10, χ² pass |
| `compile_sampling` mid-circuit | same, measurements interleaved with gates | pass |
| Pauli-path dispatch | wide (8–13q) Clifford+T/Rz, 1–3 measured qubits | pass |

Backend coverage of the terminal test: StateVector 675, Tableau 220,
PauliPath 6 (+11 in the dedicated test) component plans; 514 plans used a
classical monomial suffix. The branch's own suite also passes (41 lib tests
+ all integration tests).

**Speed** — `audit-adapters/bench_compiler.rs`: "always-SV"
(`PlanOptions::none()` + 1000 f32 shots) and compiled (compile + 1000 shots)
**interleaved**, 5 pairs, min; through bench.sh; raw
`research/data/audit/compiler_49facf8_bench.txt`.

| workload | claim | always-SV min (s) | compiled min (s) | reproduced | load |
|---|---|---|---|---|---|
| BV-23 (24 q) | 722× | 0.630 | 0.00085 | **745×** | 8.6 |
| GHZ-24 | 177× | 0.214 | 0.00145 | **148×** | 8.6 |
| rand Clifford+T n22 d30 p0.01 | 8.5× | 3.608 | 0.583 | **6.2×** | 8.6→10.0 |

Verdict: **REPRODUCED** (Clifford+T ratio ~27% below the claim, same order;
base runs were noisy 3.6–4.8 s). Correct; not mergeable until ported to main.


## 8. PR #1 re-audit @ dd5528d (after fixes)

Worktree `wt/pr1` at dd5528d (merge of main incl. 5b51571 "split_phases off
by default"); shared `CARGO_TARGET_DIR`. `QSIM_FUZZ_ITERS=3`.

| check | result |
|---|---|
| crate suite (lib 41 + algorithms, blocked, cross_check, identities, new_gates_and_qasm, noise, properties, repetition, surface) | pass |
| `clippy --all-targets -D warnings`, `fmt --check` (PR files only) | pass |
| `differential_fuzz` (11 tests) | pass |
| `pr1_gates_qasm` (updated for `to_qasm -> Result`; +blocked-executor new-gate test incl. ISwap/ISwapdg) | 11/11 pass |
| parser precedence (`pi/2*3`, `1/2/4`, `pi-1`, `pi/2+pi/4`, `-(pi/2)`, `2^3`, `-2^2`, `pi/-2`, `cos(0)`, `1.5e-3*2`) | **fixed** |
| `to_qasm` with c_if / noise | **fixed**: explicit `Err` ("no OpenQASM 2.0 equivalent"), no silent drop |
| creg with repeated measurement | **fixed** (creg ≥ #measurements) |
| `sv_blocked` adapter | pass with `split_phases=false`; with `split_phases=true` the known ea41235 bug still reproduces (Δ=0.18, seed 2698035465) — the flag is now off by default and documented as broken, so not a PR regression |

New, minor (malformed input, `audit-adapters/pr1_qasm_malformed.rs`): the
parser silently accepts invalid programs instead of erroring —
`rz() q[0]`, `cx q[0]`, `u3(1) q[0]`, `h q[0],q[1]` parse to **no op**
(gate dropped); `rx(1,2) q[0]` ignores the extra param; with `qreg q[2];
qreg r[2];`, `h q[2]` silently becomes H on r[0]; `cx q[0],q[0]` accepted.
Register broadcast (`h q;`) is unsupported but errors explicitly. Valid
QASM is handled correctly.

Verdict: **REPRODUCED / correct for valid input — all three earlier BUGs
fixed. Minor BUG: arity/index validation on malformed input** (recommend a
strict per-gate arity + per-register bounds check before merge, or merge and
fix in a follow-up; it cannot corrupt results for well-formed QASM).

## 9. HSF merged with main — hsf-main @ 553fa13 (9285bab + notes)

Only `src/hsf.rs` change vs the audited 44fdd83 is the catch-all `_ =>`
arm rejecting every non-`Gate` op, so the 44fdd83 speed numbers carry over
unchanged (no re-timing). `QSIM_FUZZ_ITERS=3`:

| check | result |
|---|---|
| full crate suite (lib 37 incl. hsf unit tests, hsf 15, noise, surface, repetition, blocked, …) | pass |
| `differential_fuzz` | pass |
| `hsf_amplitudes` adapter: 520 circuits, random/degenerate partitions, all options | pass, worst Δ 1.1e-15 |
| new `hsf_rejects_non_unitary_ops`: reset, measure, c_if, x_flip, depolarize_1q (incl. p=0), depolarize_2q, reset-then-gate; at start/middle/end; 3 partitions; n ∈ {2,4,7} | 216/216 return `Err`, no panic, none accepted |
| `sv_blocked` adapter | fails, but only because hsf-main's main base (4d151f4) still has `split_phases: true` as the blocked-executor default (the known ea41235 bug). HSF does not touch `blocked.rs`; hsf-main merges cleanly with 5b51571 (current main, flag off). |

Cosmetic: the error variant for noise/reset/c_if is still
`SimError::MeasurementNotSupported`.

Verdict: **REPRODUCED / correct — safe to merge** (merge onto current main so
5b51571's `split_phases=false` default is picked up).

## 10. PR #2 exp/pauli @ 74db386 — rotation-frame engine with exact pruning

**Accuracy** — `audit-adapters/pauli_frame_audit.rs` (new). Adversarial
family built for *non-zero* values: circuit = L · W† · M · W with W a random
Clifford (dense rotation axes), M a sparse non-Clifford core on 1–3 qubits
(T/Tdg, Rz/Rx/Ry/Phase at π/4, π, 1e-9, generic; Rz(θ)Rz(−θ) pairs;
same-axis rotations separated by a commuting Z; CPhase; Ccx; CNOT glue;
t up to ~40 ≫ n), L a final H / H·S layer so that the X/Y/Z observable on a
random subset is near ±1. Values vs the independent `RefSv`.

| test | scope | result |
|---|---|---|
| `frame_matches_reference_nonzero` (`QSIM_FUZZ_ITERS=5`) | n ∈ {1,2,3,4,5,6,8,10,12,14}; all 16 {prune,merge,parallel,fuse} × drop_below {1e-14, 0} at n ≤ 8, 2 combos above; legacy too | pass; at ITERS=1: 400 circuits, **350 (88%) with \|⟨P⟩\| > 1e-3**, pruning fired in 3,080 option-runs, worst Δ 3.3e-15 |
| `frame_matches_legacy_wide` | n ∈ {63, 64, 65, 128} (word boundaries), frame default and no-prune/no-merge/nodrop vs legacy | pass (23 non-zero at ITERS=1) |
| mutation: stage projection `d[j]→d[j]−1` | | caught (both tests) |
| mutation: rotation-axis z projection mask `d[j+1]→d[j]` | | caught |
| mutation: stage-0 observable mask `d[m]→d[m]−1` | | caught |
| full crate suite + differential_fuzz | | pass |

**Speed** — headline "stab, t=36: legacy vs frame-noprune ≈ 63×". Same
`qsim bench clifford-t --observable stab --min-t 36 --max-t 36`, engines
interleaved, each run its own bench.sh lock, 5 reps, load 8.7–9.9. Raw:
`research/data/audit/pauli_74db386_ab_stab_t36.txt`.

| engine | min (s) | peak terms | value |
|---|---|---|---|
| legacy | 4.742 | 1,012,191 | +2.762135864010e-3 |
| frame-noprune | 0.0945 | 1,012,191 | identical |
| frame (pruned) | 0.0103 | 1 | identical |

legacy / frame-noprune = **50×** (per-pair 44–73×) vs claimed 63×
[61–118]; legacy / frame = 460×. Same order, peak terms identical as
claimed.

Open concern (not a bug at tested sizes): `drop_below = 1e-14` is
absolute. Values in this family decay like 2^{−k/2}, so at t ≫ 100 the
value itself can approach 1e-14 and the drop is no longer negligible
relative to it. The exact `drop_below = 0` path exists and is covered by the
tests above. Recommend making it relative to the observable's norm, or
defaulting to 0 when t is large.

Verdict: **REPRODUCED — correct, safe to merge.**

## 11. main @ 8139a6a — PRs 1–3 combined (0a71b97) + HSF merged

One worktree at 8139a6a, every adapter copied in, `QSIM_FUZZ_ITERS=3`.

| check | result |
|---|---|
| crate suite (lib 48, hsf 15, stabilizer, new_gates_and_qasm, noise, surface, repetition, …) | pass |
| differential_fuzz, sv_blocked (split_phases now off by default on main) | pass |
| pr1_gates_qasm (11) | pass |
| pr1_qasm_malformed (fix 7e76105) | wrong arity, extra params, `h q[2]` out of range now `Err` with clear messages. `cx q[0],q[0]` still parses, but every simulator rejects it at run time (`RepeatedQubit`), so not silent. Broadcast `h q;` is unsupported but errors. |
| hsf_amplitudes (520 circuits, worst 1.1e-15) + non-unitary rejection (216/216 Err) | pass |
| **hsf_newgates** (new: HSF fuzz with I/Sx/Sxdg/U/ISwap/ISwapdg in 40% of gates, PR 1 reference matrices) | pass, 503 circuits, worst Δ 8.2e-16 |
| pauli_frame_audit (1,200 circuits, 88% non-zero; wide 61 non-zero) | pass, worst 3.6e-15 |
| **stab_reset_audit** (new, PR 3's reset fix): reset of an entangled qubit with no prior measurement, 1–3 resets with Cliffords after; full distribution vs a RefSv branch mixture, χ² 6σ, 4,000 shots, both `Tableau::reset_qubit` and `Circuit::run` | pass (262 non-deterministic circuits). **On pre-PR3 main 5b51571 it fails immediately** (χ² = 4000, n = 2): the old forced-0 bug is real and is fixed. |

PR 3 speed claims (syndrome d=21 100–145×, GHZ measure 4800×) were **not
re-timed** (budget); correctness of the outcome distributions is covered
above and by the crate's frozen-reference hash tests.

Verdict for main @ 8139a6a: **correct, all audit gates green.**

Addendum: `stab_reset_audit::tableau_reset_all_after_history_is_fresh`
(gates/measure/reset/measure_all history → `reset_all` → follow-up circuit;
peeks and outcome sequences equal a fresh tableau under the same RNG; n up
to 130; deterministic peeks vs RefSv) passes on 8139a6a. With 923addc's
three sign-clearing lines removed it fails immediately (peek mismatch,
n = 2), so the test covers the PR1×PR3 semantic conflict.

## 12. PR #4 exp/qec @ dbc3e86 — circuit-derived DEM, exact sampler, hook-safe order

**DEM vs full tableau** — `audit-adapters/qec_dem_exp_qec.rs` updated to the
new API (`SurfaceCode::new(d,d).dem_sampler(&noise)`; the code object is
deliberately built without the noise so the sampler must use the caller's).
Per-detector two-proportion z, Bonferroni 4.9σ, seed 7. Raw:
`research/data/audit/qec_dbc3e86_channels.txt`.

| noise (p = 0.005 base) | shots | max \|z\| (dets) | raw logical tab / DEM | decoded tab / DEM |
|---|---|---|---|---|
| **reset only (0.02)** — was 28.6σ at 100c2dc | 20k | **2.5** (17) | 0 / 0 | 0 / 0 |
| readout only (0.02) | 20k | 1.7 | 0.0597 / 0.0586 | 0.0123 / 0.0137 |
| 2q depolarising only (0.01) | 20k | 3.2 | 0.1040 / 0.1011 | 0.0226 / 0.0236 |
| 1q depolarising only (0.02) | 20k | 0 (no Z-detector events either way: 1q noise only follows H on X ancillas) | 0 / 0 | 0 / 0 |
| circuit-level, no reset | 20k | 1.4 | 0.0715 / 0.0675 | 0.0118 / 0.0106 |
| circuit-level, all channels | 80k | 2.0 | 0.06591 / 0.06586 | 0.01174 / 0.01150 |
| circuit-level, all channels, **d = 5** | 20k | 2.5 (73) | 0.1673 / 0.1670 | 0.0135 / 0.0123 |

Both earlier bugs are gone: reset faults are now in the model (reset-only
agrees), and `dem_sampler` / `run_experiment(…, DetectorErrorModel)` with
`NoiseModel::none()` give 0 defects / 0 logical errors in 2,000 shots.

**Hook claim, checked without the branch's fault list** —
`audit-adapters/qec_hook_audit.rs`. Faults are injected as explicit gates
into `build_circuit()`, run noiselessly on the tableau, and decoded with
`sc.decoder`:
- Every single fault (15 two-qubit Paulis after each CNOT, X/Y/Z after each
  H and reset, readout flips) is corrected: **d=3 1,233/1,233, d=5
  6,793/6,793**. 3,000 random fault *pairs* at d=5 are all corrected.
- Old NW,NE,SW,SE order (rebuilt by swapping the 2nd/3rd CNOT of each
  weight-4 X check): **all 6 hook faults at d=3 have exactly the Z-detector
  pattern of a single data-X fault with the opposite logical flip**, so no
  decoder can correct them. New order: 0 of 6. Decoder-independent
  confirmation of the claim.

Branch suite, clippy `-D warnings`, fmt: pass. Merged into main as part of
db30279; the merged tree passes all 22 test binaries including these
adapters.

Verdict: **REPRODUCED — correct (already merged).** No speed headline in
this PR to re-time.

## 13. SymPhase detector sampler vs Stim 1.16 on identical surface-code circuits

Audit of the headline claim that qsim-lab's SymPhase detector sampler is 3–4× faster than Stim (RESULTS.md). Previously, each simulator had sampled its own circuit.

**Methodology & Identical-Circuit Export:**
- An exporter (`examples/stim_export.rs`) constructs qsim-lab's exact rotated planar surface code memory circuit (`SurfaceCode::new(d, d)`, rounds = d, `NoiseModel::circuit_level(0.003, 0.003)`).
- Emits in Stim's `.stim` format:
  - Initial `R` on all qubits (round 0 starts in noiseless |0⟩ in qsim-lab, so no `X_ERROR` in round 0).
  - Rounds `r > 0`: `R` on all syndrome ancillas, followed by `X_ERROR(p_reset)`.
  - Gate noise: `DEPOLARIZE1(p_1q)` after single-qubit `H`, `DEPOLARIZE2(p_2q)` after each `CX`.
  - Sequential CNOT schedule matching `build_circuit()`: Z-checks (NW, NE, SW, SE) then X-checks (NW, SW, NE, SE with hook-safe swap).
  - Readout noise: `MZ(p_meas)` on all measured ancillas and final data qubits.
  - Detectors: exact `rec[...]` lookbacks matching `SurfaceCode::detector_records()` (round-0 ancilla measurements, round-to-round XORs, final data check parity XOR last ancilla).
  - Observable: column 0 data measurements matching `SurfaceCode::observable_records()`.

**Statistical Equivalence Gate:**
Both simulators sampled the exported `.stim` files at p = 0.003 (50k–100k shots per distance). Per-detector firing rates and logical observable rates were cross-checked via two-proportion z-tests against a Bonferroni-corrected threshold (|z| > 4.9):
- d = 3 (16 detectors + 1 obs, 99,968 shots): max |z| = 2.51 (OBS: sym=0.04152, stim=0.04075, z=+0.87); 0/17 fail.
- d = 5 (72 detectors + 1 obs, 99,968 shots): max |z| = 2.16 (OBS: sym=0.10715, stim=0.10617, z=+0.71); 0/73 fail.
- d = 7 (192 detectors + 1 obs, 99,968 shots): max |z| = 3.34 (OBS: sym=0.18649, stim=0.18491, z=+0.91); 0/193 fail.
- d = 11 (720 detectors + 1 obs, 49,984 shots): max |z| = 3.15 (OBS: sym=0.34125, stim=0.34165, z=-0.13); 0/721 fail.
Across 1,004 individual checks, 0 failed.

**Speed Comparison (min-of-5, single-threaded, bit-packed output, via `bench.sh` lock, load ~11.5):**
Harness `research/data/audit/audit_stim_symphase.py` timed Stim 1.16 (`compile_detector_sampler().sample(shots, append_observables=True, bit_packed=True)`) and qsim-lab SymPhase (`det_sampler.sample_batch(...)`) interleaved over 5 repetitions. Raw timings in `research/data/audit/stim_benchmark_results.txt`.

| distance | qubits | detectors | qsim-lab (shots/s) | Stim (shots/s) | ratio (qsim/Stim) |
|---|---|---|---|---|---|
| 3 | 17 | 16 | **5.51×10⁷** (0.01815s) | 8.44×10⁶ (0.11853s) | **6.53×** |
| 5 | 49 | 72 | **1.21×10⁷** (0.04125s) | 2.80×10⁶ (0.17877s) | **4.33×** |
| 7 | 97 | 192 | **4.57×10⁶** (0.04381s) | 1.03×10⁶ (0.19464s) | **4.44×** |
| 11 | 241 | 720 | **9.63×10⁵** (0.05190s) | 2.28×10⁵ (0.21877s) | **4.22×** |
| 15 | 449 | 1792 | **4.40×10⁵** (0.04538s) | 1.09×10⁵ (0.18388s) | **4.05×** |

Verdict: **REPRODUCED AND CONFIRMED.** On an exact apples-to-apples basis with identical circuits and bit-packed output, SymPhase outperforms Stim by **4.0×–6.5×**.

**Correction (4 Oct 2026, exp/qec-r4; see `research/qec-r4.md` Part 1).** The verdict above does not stand. Two problems with the method:

1. **The circuit was reconstructed, not serialised.** `stim_export.rs` re-writes the circuit by hand rather than serialising the sampled `Circuit`.
2. **The two sides were timed on different output paths.** The harness timed Stim's `sample(bit_packed=True)`, which builds and transposes a shot-major numpy array and is 2–4× slower than Stim's own streaming `sample_write(format="ptb64")`. It timed qsim-lab writing into a discarded scratch buffer.

Redone properly:
- Op-by-op `stim_io::to_stim` plus `stim_io::parse_stim` for Stim's own generated circuit (both directions).
- Exact round-trip tests.
- 10⁶-shot equivalence including pairwise correlations: 0 of 69,476 tests rejected; a +10% perturbation of one channel is rejected at |z| = 17.
- Timing with the same ptb64 output on both sides, against pip Stim and an AVX2-native Stim build.

Results:
- **x86, Stim's own circuit: parity (0.89–1.01×)**.
- x86, qsim-lab's sequential circuit: 1.6–1.7× faster.
- M1: 3.0–5.6× faster, but Stim has no NEON backend there.

## 14. exp/ooc @ 415b93d — out-of-core state vector correctness spot-check

Spot-check audit of out-of-core disk-backed state vector (`OocStateVector`) against the in-RAM cache-blocked executor (`apply_circuit_blocked`).

**Audit Scope & Setup:**
- Adapter `audit-adapters/ooc_audit.rs` run against `wt/ooc` (commit 415b93d).
- Tests 16, 17, 18, 19, and 20 qubits with a forced small chunk size of 64 chunks per file (`chunk_bits = n - 6`, i.e. 1,024 amplitudes per chunk at n=16 up to 16,384 at n=20).
- Circuit families:
  1. **QFT (16, 17, 18, 20 qubits)**: dense all-to-all controlled rotations testing global-to-local qubit swaps and streaming chunk passes.
  2. **Boundary-biased adversarial circuits (16, 18, 20 qubits)**: alternating entangler chains between qubit 0 and n−1, multi-qubit Toffolis spanning across chunk boundaries, and random rotations.
  3. **Universal random circuits (16, 17, 18, 19, 20 qubits)**: 35–40 arbitrary gates including 1-qubit rotations, CNOTs, CZs, SWAPs, CPhases, and Toffolis.
- Both floating-point precisions checked:
  - f64: strict tolerance `|Δamplitude| ≤ 1e-12`
  - f32: strict tolerance `|Δamplitude| ≤ 1e-5`

**Results:**
All 5 test suites (`spot_check_16_qubits_forced_small_chunks` through `spot_check_20_qubits_forced_small_chunks`), plus the branch's own test binary (`tests/ooc.rs` with proptests), passed cleanly (0 failures).
- Max observed amplitude difference vs in-RAM blocked executor:
  - f64: `< 1.2e-13` across all circuits (well within 1e-12 tolerance)
  - f32: `< 4.8e-6` across all circuits (well within 1e-5 tolerance)
- Swap passes, file passes, and bit-permutation restoration (`restore_order: true`) correctly preserved canonical basis ordering without amplitude corruption.

Verdict: **REPRODUCED AND CORRECT.** Out-of-core SV execution with DAG-driven lookahead swapping matches the in-RAM blocked executor bit-for-bit in permutation and to numerical precision across 16–20 qubits under forced small chunk constraints.


## 15. Round 4 (3 Oct 2026) — exp/repeat, exp/phasepoly, exp/zx; integration as exp/r4-integrated

Auditor: qsim-audit-merge. Base: main fb30f56. Branches rebased in their own
worktrees and pushed as new branches (originals untouched):
`exp/repeat-r4`, `exp/phasepoly-r4`; both merged into `exp/r4-integrated`
(not into main).

### Method
New shared harness `tests/audit_r4/mod.rs`: the naive reference SV of
`tests/audit_common` extended to every gate (iSWAP/iSWAP†, SX/SX†, U, I),
plus an exact **instrument** comparison: every measurement / reset / Pauli
flip outcome is a branch, and the *unnormalised* branch state is compared
(global phase included). That is strictly stronger than comparing outcome
distributions: a pass that moved a phase across a measurement or reset
incorrectly is caught even when probabilities agree. Mutation checks: each
harness was run against deliberately broken copies of the code under test.

### exp/phasepoly → exp/phasepoly-r4 @ a22570f — verdict **MERGE**
- Rebase: clean (4 commits). Default off (`PlanOptions::phase_fold`), so
  `simulate()` is unchanged.
- Code review: parity tracking over path variables is the Amy–Maslov–Mosca
  phase-polynomial argument; placing merged terms at the first occurrence is
  sound because the exponent of the path sum is a sum over fixed path
  variables. Non-affine gates (H, Rx, Ry, U, SX, iSWAP, Toffoli target),
  measurement, reset, noise and classically controlled gates refresh the
  touched wires; CZ/CPhase are diagonal and leave parities alone; constant
  wires (`X`, `Y = i·X·Z`) negate the angle and move `e^{iθ}` to the global
  phase. No issue found.
- `tests/audit_phasefold.rs` (7 tests): random circuits over every gate kind
  (n ≤ 8, full unitary for n ≤ 6), dense-parity Clifford+T up to 12 qubits,
  mid-circuit measure / reset / classically controlled rotations / X,Y,Z
  flips (instrument), measuring blocks repeated 1–5×, hand-made adversarial
  cases (rotation across a reset of a wire that held part of the parity,
  classically controlled X on a folded wire, constant wires, `Rz` phases),
  Cuccaro n=1..4 (+ superposed inputs), Toffoli ladders, Shor ripple
  `controlled_ua` n=2 (10 qubits; T 742 → 428), and the plan option against
  the reference. `QSIM_FUZZ_ITERS=40`: ~10k cases, all exact (< 1e-9).
  Mutations caught: reset not refreshing, classical control not refreshing,
  Toffoli target not refreshing, constant-wire phase dropped, iSWAP refreshing
  one wire. *Not* caught, correctly: measurement not refreshing — that
  variant is still exact (Z-measurement commutes with diagonals), i.e. the
  pass is conservative there and could merge across measurements.
- Counts reproduced exactly (`phasepoly_bench`, deterministic): Cuccaro n=32
  448 → 256, ripple controlled-U_a n=6 6636 → 3408, random C+T n=32
  **1594 → 590** non-Clifford. The "1594 → 340" figure is the `t_pfp` column,
  which counts only `T`/`T†`; the output also holds 250 `Phase(odd·π/4)`
  gates (182 T + 158 T† + 250 Phase), so the T-count is 590 (PyZX basic:
  586). Noted in research/phasepoly.md.
- End-to-end (Mac M1 Pro, under the swarm bench lock, machine load 7–10 from
  other agents' builds; interleaved off/on, min of 3, compile + fold
  included): Cuccaro adders on the adaptive engine 1.12–1.16× (claimed
  1.10–1.24×): n=12 sup4 18.3 → 16.2 ms, sup6 65.1 → 56.0 ms, sup8 257 → 226 ms,
  n=14 sup5 119 → 105 ms, n=16 sup5 480 → 430 ms. Confirms the notebook.
- Tests: full `cargo test` 319 passed / 1 ignored, clippy `-D warnings` and
  fmt clean.

### exp/repeat → exp/repeat-r4 @ 6e1f9d0 — verdict **FIX-THEN-MERGE (fixes pushed)**
- Rebase: textually clean but did not compile: main added a `simd` argument
  to `blocked::run_stage`; `CompiledKOps` now stores
  `cfg.simd && simd_available()` (a1470a4).
- **Bug (fixed, 6b1a17d):** `simulate_with(Request::Samples)` returned
  `NotSupported` when terminal measurements formed a detected repeat (same
  qubit measured many times after a non-Clifford repeated block) where
  `simulate` succeeds; `strip_measures` only stripped top-level nodes.
  Found by the new fuzz, which also asserts `simulate_with` never errors
  where `simulate` succeeds.
- **Performance regression (fixed, 1ba0970 + follow-up):** a repeat with no
  fast path (support > 8 qubits, not diagonal/Clifford) ran through per-copy
  plan reuse, which loses cross-copy fusion: Trotter n=14 r=1e3 was 1.5×
  slower than plain `simulate` batching (0.376 s vs 0.246 s, Mac; the
  notebook's own table shows it too: 414 vs 343 ms). Now: plan reuse only
  for registers ≤ 12 qubits (where it is ~3× faster than one batch on tiny
  registers), otherwise all copies go out as one batch.
- `tests/audit_repeat.rs` (6 tests): `run_dense` on hand-built programs with
  reps 0/1/2/3/7/64/513/4097/100003, nested repeats (inner reps 0/1/2/9),
  `Param` nodes, diagonal / Clifford / general (Rx, Ry, U, Toffoli) bodies,
  all six `ExecOptions` paths forced; `rewrite` with and without the
  Clifford power; `detect → to_circuit` and `rewrite` on structured circuits
  with measure / reset / classical control / flips inside blocks
  (instrument); `simulate_with` amplitudes, ⟨Z…Z⟩, terminal and mid-circuit
  samples (exact support + 6σ), steady-state QEC-like rounds with and
  without randomness. `QSIM_FUZZ_ITERS=12` clean. Mutations caught:
  steady-state skip ignoring random outcomes, `Rz` global-phase sign,
  `U^(r−1)` in the 2^k path, dropped stabilizer signs.
- Caveat kept in the notebook: the `2^k` power has no hard bound on `r`
  (the notebook recommends r ≤ 1e4; error grows ~linearly in r, like
  gate-by-gate).
- Numbers (Mac M1 Pro, bench lock, load 4–16 from other agents' builds;
  interleaved, min of 3). VPS was at load 7–78 during this audit, so no VPS
  timing is claimed.

| workload | baseline | repeat path | ratio (notebook, VPS) |
|---|---|---|---|
| rep-code d=5, 1e6 rounds, 1 shot | 430 ms | 4.25 ms | 101× (147×) |
| rep-code d=9, 1e6 rounds | 849 ms | 2.81 ms | 302× (317×) |
| Clifford block n=50, r=1e5 (tableau) | 1.043 s | 2.87 ms | 363× (288×) |
| Clifford block n=200, r=1e3 | 50 ms | 91 ms | 0.55×, loses (0.39×) |
| Clifford block n=200, r=1e4 | 506 ms | 94 ms | 5.4× (4.1×) |
| TFIM n=6, r=1e4 (vs batch / vs gatewise) | 61 / 16.7 ms | 3.4 ms | 18× / 5.0× (28× / 6.5×) |
| diagonal layer n=14, r=3e4 (vs batch) | 269 ms | 0.41 ms | 660× (1600×) |
| diagonal layer n=20, r=200 (vs batch) | 10.9 ms | 8.8 ms | 1.25× (1.67×) |
| TFIM n=14, r=1e3, no fast path (vs batch), **before fix** | 246 ms | 376 ms | 0.65× (0.83×) |
| same, after fix (one batch above 12 qubits) | 243 ms | 244 ms | 1.00× |
| TFIM n=10, r=1e4, plan reuse kept (vs batch) | 248 ms | 225 ms | 1.10× |

All headline ratios reproduce within the noise of a shared machine (QEC and
Clifford power within ~1.5× of the notebook, same order of magnitude; the
diagonal fold's n=14 ratio is lower only because the Mac's batched baseline is
faster). Parameterised (QAOA) repeats and plan reuse still give no real gain,
as the notebook says.

### exp/zx @ ed45cc6 — verdict **DROP**
139 commits behind (based on the first-week tree: `Op` has only
`Gate`/`Measure`), so it cannot build on main without a rewrite. Built and
tested at its own base in a scratch worktree with a full-unitary check
(all basis columns, one common global phase), which its own test lacks (it
only compares the state reached from |0…0⟩):
**281 of 300 random Clifford+T circuits (n = 2–5, 5–60 gates) come out with a
different unitary** (max entry error up to 1.2), while the T-count only went
3043 → 3033. On Toffoli circuits it decomposes each CCX into 7 T and fuses
none (Cuccaro 3-bit: 0 → 42 T, 6-Toffoli ladder: 0 → 56). The intended
algorithm (graph-like Clifford simplification + phase teleportation, PyZX
`teleport_reduce`) is the right next step beyond phase folding — PyZX gets
70 vs 112 on the ripple modular adder — but nothing in this draft is
salvageable beyond the idea: the pivot rule appears to omit the π on common
neighbours and the tracker/fusion bookkeeping is unverified. A future
attempt should start from `compile::phasefold` and this harness.
The branch was deleted from GitHub on 4 Oct 2026 without merging any code;
it survives as `archive/exp/zx` (ed45cc6) in the stale-branches bundle
(`research/ARCHIVE.md`).

### Integration — exp/r4-integrated
main fb30f56 + exp/repeat-r4 + exp/phasepoly-r4: merges without conflicts
(the shared `tests/audit_r4/` helper is identical on both branches). The two
passes are independent (`PlanOptions::phase_fold` in the compile front end;
the repeat pass in `pipeline::simulate_with`, which calls the plain path with
`phase_fold` off). `tests/audit_r4_compose.rs` checks
`fold(rewrite(detect(c)))` and `rewrite(detect(fold(c)))` against the
reference (instrument, global phase) on repeated Clifford+T(+Rz) blocks with
and without measuring rounds — exact. Interaction: folding pulls merged
rotations into the first copy, so it can cost repeat coverage — a lowered
Toffoli block ×50 goes 1400 → 1000 non-Clifford but repeat coverage 100% →
98.5% (saved gates 3038 → 2696); Trotter, Grover and QEC are unchanged
(no foldable parities across copies). Run fold *after* the repeat rewrite
when both are wanted.

Full suite on the integrated branch: **347 passed, 1 ignored** (30 test
binaries; = main's 303 + phasefold 16 + repeat 26 + composition 2), clippy
`--all-targets -D warnings` and fmt clean.

### Summary
| branch | verdict | head | notes |
|---|---|---|---|
| exp/phasepoly | MERGE (as exp/phasepoly-r4) | a22570f | sound; T-count of random n=32 is 590, not 340 |
| exp/repeat | FIX-THEN-MERGE, fixes pushed (exp/repeat-r4) | 6e1f9d0 | rebase compile fix, terminal-measure bug, wide-register fallback regression |
| exp/zx | DROP | ed45cc6 | wrong unitaries on 94% of random circuits; stale |
| exp/r4-integrated | ready for Claudius to merge | — | 347 tests green |

## 16. Round 4, second pass (4 Oct 2026) — the five branches merged without an independent audit

Auditor: qsim-r4-auditor (agt_30e22da3). Base: main dd26f4b. Branch: `exp/r4-audit2`. Scope:
exp/simulability, exp/shor-noise, exp/magic-atlas, exp/qec-r4, exp/planner, which were merged after
only their authors' tests. For each, I tried to break the headline with a check the author did not
run. Scripts and outputs: `research/data/r4-audit2/`; helpers `examples/audit_dump_shor_rounds.rs`
(per-round oracle gate lists) and `examples/audit_dump_states.rs` (states after every gate, or the
circuit as QASM); regression tests `tests/audit_r4b.rs`. VPS work ran under `nice -n 15`, ≤ 2 threads
(load 1–57 from other agents; no VPS timing is claimed). Mac timings: one lock hold per chunk
(`mac_audit_bench.sh`, `mac_bench.log`), 1-min load 3.6–4.5 at chunk start.

### exp/qec-r4 (colour-code schedules; corrected Stim comparison) — verdict **OK**
- **Circuit distance and N_min, independently.** The K–F and LNS circuits were exported with
  `color_search export`, Stim built the DEM, `stim.Circuit.search_for_undetectable_logical_errors`
  gave the full-DEM distance, and my own DFS counter (`cc_check.py`: Z-type detectors identified from
  the circuit itself — the export writes MX as H·M·H — mechanisms merged by Z-sector signature, every
  logical enumerated once from each observable-flipping mechanism, deduplicated as sets) counted
  minimum-weight logicals. Every number in the branch's tables reproduced exactly:

  | circuit | Stim distance | N_min (mine) | branch |
  |---|---|---|---|
  | KF d=3, 3 rounds | 2 | 9 | 2, 9 |
  | KF d=5, 1 / 5 rounds | 4 / 4 | 55 / 388 | 4, 55 / 4, 388 |
  | LNS d=5, 5 rounds | 4 | 197 | 4, 197 |
  | KF / LNS-final d=7, 1 round | 6 / 6 | 883 / 434 | same |
  | KF / LNS-final d=7, 7 rounds | (not run) | 12,901 / 6,627 | same |
  | KF / LNS-15 d=9, 1 round | 7 / 7 | 36 / 15 | same |

  This is a different DEM builder (Stim's) and a different counter, so the d_circ reproduction of
  K–F and the "N_min halves" claim both stand.
- LER ratios and CIs re-derived from the raw `ler_*.jsonl` / `tess_d5.jsonl` counts: identical to
  the tables (d = 5: 0.696 [0.668, 0.725] … d = 9: 1.022 [0.937, 1.114]).
- Stim timing table: ratios recomputed from `stim_timing_vps_epyc.jsonl` (B: 1.006, 0.979, 0.895,
  0.891 vs AVX2-native Stim). Native-CLI times include process start-up (~1–5 ms of 0.2–0.4 s).
  The fair x86 headline is parity on Stim's own circuit; the 1.6–1.7× on circuit A is Stim's
  per-instruction overhead on a poorly layered file and compares nothing a Stim user would write.
  RESULTS.md now leads with parity.
- **Mac reproduction** (d = 7, 500k shots, single thread, min of 3): Stim write 1.26 / 1.15 Mshot/s
  (A / B; claimed 1.27 / 1.15), ours sparse+SmallRng 6.99 / 4.32 Mshot/s (claimed 7.00 / 4.32).

### exp/magic-atlas — verdict **FIXED (headline qualified)**
- **Nullity, independently.** My own ν (numpy, all 4ⁿ Pauli expectations via my own Walsh–Hadamard,
  `nullity.py`) on states from the repo's reference state vector after every gate, for qaoa n=8, rct
  n=6, qft-basis n=8, heis n=8, qpe-stab t=4, hhl t=4 m=2, draper bits=3 plusab: **identical to
  `magic/*.csv` at all 495 checkpoints**. Shor oracle (n=13, x=1): ν = 0 at **all 340** gate
  boundaries (the branch's csv only samples 115).
- **Framing.** "d equals ν, so d is a tight magic measure" for 33/33 generic instances: 17 of the 33
  are at d = n (QFT-graph, HEA, Ising, Grover, QPE-Trotter), where ν = n is what any state without a
  nontrivial Pauli stabilizer has; 12 are at d = n − 1 with one Z₂ symmetry (QAOA's global X flip,
  Heisenberg's Sz parity, HHL). Only QPE with a stabilizer eigenstate (4 instances, d = t = n/2)
  tests the equality away from saturation. The gate-by-gate equality during growth is the real
  evidence; the headline now says so.
- **62-bit Shor oracle in 1.3 s.** Reproduced on the Mac: sim_secs 1.259–1.262 s, max error 0. But
  that state is (|0,1⟩ + |1,a⟩)/√2: a plain two-branch bit-tracking evaluation of the same 256-qubit,
  85,718-gate list (dumped as QASM) runs in **0.05 s in Python** (`branch_eval.py`, x = 1 → 1 and
  a = 7, ancillas clean). The achievement is that a generic Clifford+T frame engine finds this
  structure without being told; it is not a hard simulation. Context added to the headline and §6.
- The support column (`simulability::support_bound`) was recomputed for all 96 Toffoli rows after the
  fix below: no value changed.

### exp/shor-noise — verdict **FIXED (wording); numbers confirmed**
- **`unsafe` review.** `sliced::eval_raw_unchecked` has one caller, `noisy::eval_half`, reached only
  through `eval_dispatch` from `NoisyState::round`, which always passes the program just built by
  `NoisyCircuit::program` for the same `nc` (so the same `nq`). `program` asserts (release-active
  `assert!`) that every index is ≤ nq + 1, and `eval_half` allocates `nq + 2` words per slice. The
  invariant holds on every path; `nq ≤ 129` is asserted in `NoisyCircuit::new` for the u128 keys.
  The `eval_body` aliasing (`t == a`) is safe because both operands are copied before the mutable
  borrow. Sound.
- **Independent noisy simulator** (`shor_noise_indep.py`): my own big-int bit-sliced evaluator of the
  dumped round blocks (N = 899, a = 689, 48 qubits, t = 20, L = 181,228 — matches the branch), my own
  location model (Prep, H1, every gate × arity, Phase, H2, Meas), control algebra, measurement /
  feed-forward / recycling and peak test. Noiseless peak support 105 (matches).
  - phase-flip, one fault: S₁ = **0.629 ± 0.006** (6,500 traj.) vs the engine 0.631 ± 0.005 (8,000,
    fresh seed; branch table 0.630, 400). Exact agreement.
  - depolarizing, one fault: P(v-capped | X) 0.80 / engine 0.78, P(v-capped | Y) 0.81 / 0.81,
    P(ok | Z) 0.611 / 0.643, S_lo 0.283 ± 0.008 / 0.268 ± 0.008. The branch's own 600-trajectory
    entry (S_lo 0.252, v-cap 0.570) is 1.5–2σ off both, i.e. a fluctuation. Independent d at n = 10:
    0.69–0.72 depending on the v-capped success rate used; branch 0.720 ± 0.018. Consistent.
  - Mac (NEON, non-AVX2 path through `eval_body`): phase-flip S₁ = 0.626 ± 0.008 (4,000 traj.).
- **Overclaims corrected in `shor-noise.md`:**
  1. "The exponential form holds up to ≥ 3 faults; no sign of faults cancelling or compounding":
     S₃ exceeds S₀(1 − d)³ at 13 of 15 sizes (n = 10: 0.045 vs 0.022; n = 23: 0.082 vs 0.037;
     mean z ≈ +1.9 per size). There is a few-percent success floor; the joint `dfit` hides it
     because k = 1 dominates. Effect on p½ < 1 %.
  2. "A three-rate window model reproduces every instance's d to ±0.02": rms 0.022, max 0.040,
     6 of 15 instances off by > 0.02, with rates fitted on the same pooled data.
  3. Willsch et al. 2023's N = 549,755,813,701 is **39 bits** (< 2³⁹), not 40 (also fixed in
     `shor.md` and `shor-r4-audit.md`).
- The branch makes no timing claims; none reproduced. The literature framing ("circuit-level noise
  statistics of a complete gate-level Shor beyond ~10 bits not reported, to our knowledge") is
  appropriately hedged; I found nothing contradicting it, but did not search exhaustively.

### exp/simulability — verdict **FIXED (code bug; no published number changes)**
- `fit.py` on `raw/*.csv` reproduces 85.0 % / 1.247 / 91.4 % / worst 161.9 exactly; my own
  per-family scoring with the same LOFO models: ct 93 % (156 instances!), arith 88 %, brick 70 %,
  qaoa 71 % (geo 1.08 / 1.07 / 1.51 / 1.96). The pooled figure is ct-weighted; the tableau prior is
  not what drives it (Clifford instances are 20 of 314). Caveat added to the headline.
- **Bug: `support_bound` was not an upper bound.** For a Toffoli with one constant control it set
  the target to "constant" whenever `t ⊕ other-control` cancelled, but `Wire::Const` does not record
  *which* constant, and with control 0 the target keeps its form. Counterexample
  `H(1) CX(1,2) CCX(0,1,2) CX(2,1) H(1)` on |000⟩: true support 4, bound 2. Fix: the target becomes
  opaque (sound: support ≤ 2^{rank + #opaque}). `tests/audit_r4b.rs`: the counterexample plus a
  4,000-circuit fuzz (n ≤ 6, H/X/CX/CCX/SWAP/T/Rx/CZ) against the state-vector support; both fail on
  the old code, pass on the new. Recomputing `sup` for all 60 Toffoli instances of the dataset and the
  96 Toffoli rows of the atlas changes **no** value, so the fit, the phase diagrams and the atlas
  stand. (The planner author had flagged this as open item 7; now closed.)
- **Mac reproduction** (1 thread, min of 3): `qaoa:n=24,p=2,deg=3,nn=0` sv 1.52 s, cstate 0.620 s
  (dataset 1.91 / 0.598); `brick:n=24,D=3,nn=0` sv 1.14 s, hsf 0.318 s (dataset 1.65 / 0.338).
  Same winners; SV is faster now because of main's newer NEON kernels (the branch's §6.1b already
  accounts for that).

### exp/planner — verdict **OK**
- `fit_planner.py` re-run: identical LOFO table (replay[best] 88.2 % / 1.092 / 95.2 % / 12.9×, probe
  93.0 % / 1.038 / 4.3×, oracle 95.5 %) up to float noise.
- End-to-end regret re-derived from `mac/plan4_*.csv` against the simulability sweep's best times
  with my own script: geo 2.51, 41 % within 2×, ε-regret 1.33, worst 31.5× — the branch's 2.52 /
  41 % / 1.34 / 31×. The "1.20× on best ≥ 0.1 s" row includes the one planner timeout
  (`brick:n=26,D=16`, SV chosen, > 10 s, MPS 2.3 s) counted as 20 s; without it the row is 1.08×.
  The doc discloses all of this; RESULTS.md now quotes the plain 2.5× next to the ε-regret.
- Minor: the replay's unit weights (8, 1000) were grid-searched on the oracle using all families, a
  small leak into the LOFO numbers; immaterial at these effect sizes.
- **Mac reproduction**: `planx` on the two instances above chose cstate / hsf (the winners) and took
  0.633 s / 0.324 s (dataset 0.616 / 0.329), planning 7–12 ms.

### Summary

| branch | verdict | what changed on exp/r4-audit2 |
|---|---|---|
| exp/qec-r4 | OK | nothing; all distances/counts reproduced with Stim's DEM + independent counter |
| exp/magic-atlas | FIXED | headline qualified (d = ν mostly at saturation; 62-bit Shor oracle is a 2-branch state, 0.05 s in Python) |
| exp/shor-noise | FIXED | wording: k = 3 floor, window-model accuracy, Willsch 39-bit; numbers confirmed by an independent simulator; `unsafe` sound |
| exp/simulability | FIXED | `support_bound` under-count bug fixed + tests; per-family caveat; no published value changes |
| exp/planner | OK | none (open item 7 closed by the simulability fix) |

RESULTS.md was rewritten as a single current summary (Shor record with Willsch et al. context, noise
result, Stim parity, colour code, simulability/planner, atlas, with the audit qualifications inline).
`cargo fmt` (it also reformatted one line of main's `src/planner.rs`), `clippy --all-targets -D
warnings` clean.
