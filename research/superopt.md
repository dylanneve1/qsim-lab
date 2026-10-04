# Superoptimising the gate-level Shor oracle (exp/superopt)

Branch `exp/superopt`, based on main `ecb83ea`. Code: `src/shor_superopt.rs`
(the optimised blocks, `Oracle::WindowedOpt`, CLI `--oracle windowed-opt`),
`src/shor_superopt_rules.txt` (SAT-derived rewrite rules),
`tools/superopt/` (SAT synthesiser `synth.py`, block certificates
`blocks.py`, SAT peephole `peep.py`), `examples/superopt_counts.rs`
(counts, ablations, window sweep, per-block counts, circuit dumps), tests in
`src/shor_superopt.rs`, `tests/superopt.rs`, `tests/theory_shor_opt.rs`. Raw
data: `research/data/superopt/`.

**Machines.** All builds, SAT runs and timings ran on the Mac (Apple M1 Pro,
8 cores, 16 GB). Timings were taken under `/tmp/qsim-mac-bench.lock` with
baseline/opt interleaved. The Mac was shared: 1-min load average 17–35
during every timing cell, from other agents' unlocked campaigns. SAT and
peephole searches ran without the lock (≤ 4 workers) and were SIGSTOPped
whenever anyone held it (`tools/superopt/runq.sh`). The VPS was used for git
only.

## TL;DR

* The windowed oracle from round 4 had one large inefficiency. Its table
  lookup shares AND-chain prefixes LSB-first, and in counting order
  consecutive addresses always differ in bit 0, so it never actually shares
  anything. It costs `≈ 2w` Toffolis per address (120 per `w = 4` lookup,
  where unary iteration needs 30).
* Six provably correct block changes, a generic peephole pass, and a
  SAT-derived rewrite-rule pass cut the controlled-U circuit sharply. For the
  31-bit record run (N = 1 537 596 787, a = 457 167 243, all 62 rounds):
  **1 704 428 → 1 039 807 gates (−39.0 %), 528 178 → 256 006 Toffolis
  (−51.5 %)**. Qubits are unchanged (132), and so is the measured integer.
  20/24/28-bit: −46 % / −44 % / −42 % gates and −62 % / −58 % / −55 %
  Toffolis.
* End to end on the Mac, the 31-bit record goes **135.7 s → 97.6 s (min of
  3, −28 %)**. The gate evaluation drops 36 %. Sort, P(1) and collapse are
  unchanged, so the 28-bit run goes 16.2 s → 13.1 s. Small instances
  (24-bit, 0.3 s) were *slower* with global passes because of circuit-build
  time. See "build cost" below.
* SAT (CaDiCaL via PySAT) gives exact optimality certificates for small
  blocks (comparators, adders, lookups; tables below). It also gives a
  constant-aware peephole superoptimiser over the real circuit: windows of
  ≤ 5 wires and ≤ 10 gates, 33 distinct rewrite rules. These rules
  converged after 2 iterations and remove a further 1.6 % of gates and
  2.4 % of Toffolis.
* **Known vs new, honestly.** Every big win re-implements published
  constructions that our code didn't use: unary iteration (Babbush et al.
  2018), comparator-based flag uncomputation (standard since Cuccaro 2004 /
  Häner et al. 2016), and the trivial first-window copy. The generic
  peephole pass already recovers most of the comparator saving by itself.
  None of this beats published constructions. Gidney–Ekerå 2019 and Gidney
  2025 avoid modular reduction entirely (coset / approximate residue
  arithmetic) and use measurement-based uncomputation, which halves
  Toffolis again. Those are outside this exact X/CNOT/CCX simulation model.
  Two things may be new as stated, though both are small: the exact
  tree-XOR fan-out DP for lookups (CNOT-only, −20 % of lookup gates), and
  the constant-aware SAT window rules on a real Shor circuit.

## 1. What changed (all behind `shor_superopt::Opts`, `Oracle::WindowedOpt(w)`)

Same layout as `shor_window` (4n + 4 + w qubits). Every option preserves
the exact permutation. All of them are checked exhaustively in unit tests
and differentially through the sliced engine (section 4).

| option | what | proof idea |
|---|---|---|
| `unary` | lookup by unary iteration over the address tree, MSB first: per internal node `CCX(p, bit, f)`, right subtree, `CNOT(p, f)` (switch to left sibling `p∧¬bit`), left subtree, `CNOT(p, f)`, `CCX(p, bit, f)`. `2(2^w − 1)` CCX, no X gates. | each node flag equals `ctrl ∧ prefix` while its subtree runs; induction on depth; exhaustive test w ≤ 5 |
| `fanout` | output bit *i* must be XORed with the indicator of the set S_i of addresses whose entry has bit *i* set. Every tree node's flag is live during its subtree, and it is the XOR of its leaves. Write S_i as the XOR of the fewest tree nodes: `cost(node, flip) = min(cost(L,flip)+cost(R,flip), 1+cost(L,¬flip)+cost(R,¬flip))`. | DP exact for this laminar family (each node used or not; children then need S or its complement); brute-force-checked against all node subsets for every 1-bit table, w ≤ 3 |
| `comparator` | modular adder `b ← b+L mod N`: the flag uncompute `sub L; read sign; add L` (4n CCX) becomes `t ^= [b < L]` (half a Cuccaro subtraction, CNOT out, half undone: 2n CCX) + `X(t)` | the top carry of the half subtraction is the borrow; t = [no reduction] = ¬[b_new < L]; exhaustive for all b, L < N, N ≤ 21 |
| `kflip` | `K: N → t·N` by `X(t) CNOT(t,K_j) X(t)` instead of unload + controlled reload | trivial |
| `direct_first` | accumulator is 0 before window 0: look up straight into `b` (no modadd, no unlookup) | 0 + T[v] = T[v] < N |
| `keep_chain` | the lookup's trailing gates on ctrl/address/AND ancillas commute with the modadd (disjoint qubits) and cancel against the unlookup's head | `A B M B⁻¹ A⁻¹ = A M A⁻¹` when `[B, M] = 0` |
| `peephole` | the existing commutation-aware peephole (`compile::peephole`), output checked to be X/CNOT/CCX only (a global phase on a permutation matrix must be 1) | existing tested pass |
| `sat_rules` | 33 SAT-derived window rewrite rules (≤ 5 wires, ≤ 10 gates, with known-constant wires), CCX-non-increasing, **re-verified exhaustively when loaded** | rule table verified on load; constant propagation is sound |
| `block_passes` | run `peephole` + `sat_rules` once on the modadd block instead of on the whole circuit (build-cost fix) | as above |
| `window_dp` | exact gate-count DP over window sizes (≤ w) per multiplier | any split is a correct multiplier |

## 2. Per-block before → after (n = 31, N = 1 537 596 787, w = 4)

From `superopt_counts N a 4 blocks` (lookup counts averaged over the 8
windows of the round-0 multiplier; the last window has 3 bits):

| block | gates | CCX |
|---|---|---|
| lookup, baseline (LSB-first chain) | 349.8 | 110.2 |
| lookup, unary iteration | 267.5 | 28.0 |
| lookup, unary + optimal fan-out | **210.8** | **28.0** |
| modadd, baseline (5 Cuccaro adders) | 1019 | 310 |
| modadd, baseline + generic peephole | 807 | 248 |
| modadd, comparator | 830 | 248 |
| modadd, comparator + kflip | 812 | 248 |
| modadd, comparator + kflip + peephole + SAT rules | **764** | **247** |

The generic peephole alone turns the baseline's `sub L … add L` pair into
the comparator: the backward half of the subtraction and the forward half
of the addition are inverse pairs that cancel. The auditor's estimated
"comparator saves ~12 %" is real (−11.0 % gates on its own), but it was
already available from the existing compiler pass. Per controlled-U round,
the CCX budget goes from 8519 to 4128. Of the 4128, 82 % sit in the 14
modular adders (8n each), the rest in 30 lookups (28–30 each) and the 31
controlled swaps.

### Ablation (31-bit record, all 62 rounds; `research/data/superopt/counts_31.txt`)

| variant | gates | Δ gates | CCX | Δ CCX |
|---|---|---|---|---|
| baseline (`--oracle windowed`) | 1 704 428 | — | 528 178 | — |
| unary | 1 541 244 | −9.6 % | 364 994 | −30.9 % |
| unary + fanout | 1 427 640 | −16.2 % | 364 994 | −30.9 % |
| comparator | 1 516 940 | −11.0 % | 466 674 | −11.6 % |
| kflip | 1 686 572 | −1.0 % | 528 178 | 0 |
| direct_first | 1 531 761 | −10.1 % | 474 858 | −10.1 % |
| unary + keep_chain | 1 527 852 | −10.4 % | 357 306 | −32.4 % |
| generic peephole only | 1 492 016 | −12.5 % | 464 690 | −12.0 % |
| all but peephole/SAT | 1 064 487 | −37.5 % | 256 006 | −51.5 % |
| all but SAT rules | 1 056 597 | −38.0 % | 262 322 | −50.3 % |
| **all (global passes)** | **1 039 807** | **−39.0 %** | **256 006** | **−51.5 %** |
| all + window DP | 1 036 327 | −39.2 % | 256 163 | −51.5 % |

BLOCKPASS_ROW

Window sweep with everything on (`… sweep`): w = 3 / 4 / 5 / 6 give
1.21 M / 1.06 M / 1.20 M / 1.50 M gates (baseline: 1.80 / 1.70 / 2.14 /
3.04 M). With the lookup fixed, the baseline's steep Toffoli growth with w
(the 2w-per-address chain) disappears. 262 k–291 k CCX for w = 4–6, but w = 4
is still optimal for gates at n = 31. The exact window-size DP (windows of
mixed size ≤ w per multiplier) gains only 0.3 %, so it is off by default.

## 3. SAT: exact synthesis and optimality certificates

`tools/superopt/synth.py` encodes "exists a circuit of exactly k gates over
{X, CNOT, CCX}". Per step there is a one-hot target, two ordered control
selectors with "none", per-row wire values and `v' = v ⊕ (T ∧ c1 ∧ c2)`.
Specification rows can leave outputs unconstrained. Clean ancillas appear
only in rows with 0 inputs and are required to return to 0. k is iterated
upward: the first SAT k is the minimum, and each smaller k was refuted
(UNSAT) by CaDiCaL 1.9.5. Every decoded circuit is re-simulated against the
spec. Optional CCX caps (sequential-counter cardinality) with NOP padding
give Toffoli-minimal circuits.

SATTABLE

## 4. Verification

* Unit (`cargo test --release --lib shor_superopt`, 9 tests):
  `lookup_unary_exhaustive` (w ≤ 5, both fan-out modes, every address and
  control, two initial outputs); `compare_lt_exhaustive` (n ≤ 5, all a, b,
  t); `fanout_plan_optimal_bruteforce_small` (DP = minimum over all node
  subsets, every 1-bit table, w ≤ 3); `fanout_plan_reproduces_table…`
  (random tables w ≤ 6); `add_mod_reg_exhaustive` (every option, all b, L
  < N for N ≤ 21); `controlled_ua_exhaustive_small_all_opts` (N = 15…63,
  w = 1…4, 12 option sets incl. ALL, global passes and window DP: every x <
  N, both controls, all ancillas clean); `baseline_opts_reproduce_shor_window_exactly`
  (gate-for-gate); `sat_rules_parse_and_verify` (every rule exhaustively).
* Differential (`tests/superopt.rs`): exact outcome distributions of the
  whole measurement tree, sliced engine and gate-by-gate sparse state vs the
  permutation oracle, < 1e-12 (N = 15, 21, 35; w = 1…4); same measured
  integers as the permutation and unoptimised windowed paths up to N =
  1 005 973; beyond 64 qubits (24-bit N, w = 3, 4, 5).
* `tests/theory_shor_opt.rs` re-runs **all 12 theorem checks** of
  `tests/theory_shor.rs` (support law on whole gate-level measurement
  trees, the work-counter identity `W = Σ 2 B_i G_i`, the noise-window
  theorems T3, …) on `Oracle::WindowedOpt(4)`. All pass. `shor_scale`
  regression tests also pass.
* Every Mac timing run reports the same order, factor and measured integer
  for baseline and optimised (31-bit: 2 059 039 373 337 077 151).

## 5. End to end on the Mac (bench lock, interleaved, `research/data/superopt/bench2.log`)

TIMINGTABLE

## 6. Literature: known vs new

LITERATURE

## 7. What did not work / limits

LIMITS
