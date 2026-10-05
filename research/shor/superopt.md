# Superoptimising the gate-level Shor oracle (exp/superopt)

Branch `exp/superopt`, based on main `ecb83ea`. Code: `src/shor/superopt.rs`
(the optimised blocks, `Oracle::WindowedOpt`, CLI `--oracle windowed-opt`),
`src/shor/superopt_rules.txt` (SAT-derived rewrite rules),
`tools/superopt/` (SAT synthesiser `synth.py`, block certificates
`blocks.py`, SAT peephole `peep.py`), `examples/superopt_counts.rs`
(counts, ablations, window sweep, per-block counts, circuit dumps), tests in
`src/shor/superopt.rs`, `tests/superopt.rs`, `tests/theory_shor_opt.rs`. Raw
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
  **1 704 428 → 1 039 303 gates (−39.0 %), 528 178 → 261 454 Toffolis
  (−50.5 %)** with the default `Opts::ALL`. Running the passes on the
  whole circuit gives 256 006 Toffolis (−51.5 %). Qubits are unchanged
  (132), and so is the measured integer. 20/24/28-bit: −45.9 % / −43.4 % /
  −41.7 % gates and −60.0 % / −56.5 % / −53.6 % Toffolis.
* End to end on the Mac (interleaved, under the bench lock, min of 3), the
  31-bit record goes **137.8 s → 97.5 s (−29 %)** and the 28-bit run
  **16.3 s → 12.0 s (−27 %)**. The 24-bit run is unchanged at 0.29 s: its
  gate evaluation is 35 % faster, but at that size build and sort
  dominate. The gate evaluation itself drops 36 %. Sort, P(1) and collapse
  don't depend on the circuit and are unchanged.
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
the CCX budget goes from 8519 to 4217 (default; 4128 with global passes).
Of the 4217, 82 % sit in the 14
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
| all but peephole (SAT rules on) | 1 064 487 | −37.5 % | 256 006 | −51.5 % |
| all but SAT rules | 1 056 597 | −38.0 % | 262 322 | −50.3 % |
| **all (global passes)** | **1 039 807** | **−39.0 %** | **256 006** | **−51.5 %** |
| all + window DP | 1 036 327 | −39.2 % | 256 163 | −51.5 % |

**Default (`Opts::ALL`) = block passes.** Running the peephole and the SAT
rules over the whole 17 k-gate circuit of every round costs 0.5–1.2 s of
build per run (2 passes each, 62 rounds), which made small instances
slower end to end. `block_passes` runs both passes once per multiplier on
the modular-adder block, which is identical in every window. It gives the
same gate count (31-bit: **1 039 303** gates, 261 454 CCX, −39.0 % /
−50.5 %; build 94 ms vs 572 ms). It loses about 2 % of the Toffoli saving:
junction rewrites between a lookup and the adder. The global variant stays
available as `block_passes: false`.

| instance | baseline gates / CCX | ALL (block passes) | Δ | ALL, global passes | Δ |
|---|---|---|---|---|---|
| 20-bit N = 1 005 973 (88 q) | 501 808 / 176 800 | 271 380 / 70 720 | −45.9 % / −60.0 % | 269 848 / 68 058 | −46.2 % / −61.5 % |
| 24-bit N = 10 161 323 (104 q) | 819 104 / 277 632 | 463 922 / 120 672 | −43.4 % / −56.5 % | 462 867 / 116 872 | −43.5 % / −57.9 % |
| 28-bit N = 221 643 407 (120 q) | 1 264 308 / 409 248 | 736 962 / 189 728 | −41.7 % / −53.6 % | 739 283 / 185 014 | −41.5 % / −54.8 % |
| 31-bit N = 1 537 596 787 (132 q) | 1 704 428 / 528 178 | 1 039 303 / 261 454 | −39.0 % / −50.5 % | 1 039 807 / 256 006 | −39.0 % / −51.5 % |

(Bases as in the round-4 record runs; counts are whole-run sums over the
2n controlled-U rounds, `research/data/superopt/counts_final.txt`.) The
relative saving shrinks with n because the lookup share of the circuit
falls as the n-proportional adders grow.

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

Certificates (`research/data/superopt/sat*.jsonl`; "refuted" = every
smaller gate count proved UNSAT):

| block (spec) | wires | SAT optimum | refuted | our construction |
|---|---|---|---|---|
| comparator n = 1, `t ^= [b < a]` | 3 | **2 gates** (1 CCX): `CNOT(a0,t) CCX(a0,b0,t)` | k ≤ 1 | 7 gates / 2 CCX (Cuccaro half-sub; the SAT rules shorten it in the circuit) |
| comparator n = 2, no ancilla | 5 | **8 gates** (6 CCX) | k ≤ 7 | 13 gates / 4 CCX + clean `c0` |
| comparator n = 2, clean `c0` allowed | 6 | **8 gates** (5 CCX), does not use `c0` | k ≤ 7 | 13 / 4 |
| adder n = 2 (`b += a`, b has a carry bit), no ancilla | 5 | **7 gates** (5 CCX) | k ≤ 6 | Cuccaro: 13 gates / 4 CCX + `c0` |
| adder n = 2, 1 clean ancilla | 6 | **7 gates** (5 CCX) | k ≤ 6 | 13 / 4 |
| modadd N = 3 (b, L < 3, 2 bits), no ancilla | 4 | **6 gates** (4 CCX) | k ≤ 5 | general construction at n = 2: 62 gates / 16 CCX (47 / 15 after peephole + SAT rules) |
| modadd N = 3, 1 clean ancilla | 5 | **6 gates** (4 CCX) | k ≤ 5 | — |
| controlled lookup w = 2, T = [0,5,3,7] (3 bits), 1 or 2 clean ancillas | 7–8 | **6 gates** (all CCX; uses an output bit as a temporary control and toggles it back) | k ≤ 5 | unary + optimal fan-out: 13 gates / 6 CCX |
| same lookup, minimum CCX at ≤ 10 gates | 8 | ≤ 4 CCX found; the ≤ 3 CCX query hit the 50-min timeout (open) | — | 6 CCX |
| linear table T = [0,5,3,6] (T[3] = T[1] ⊕ T[2]) | 7 | 4 gates, ≤ 2 CCX | k ≤ 3 | — |
| comparator n = 3, no ancilla | 7 | not finished: k ≤ 8 refuted (k = 8 took 456 s; k = 9 stopped after 13 min), so **optimum ≥ 9** | k ≤ 8 | 19 gates / 6 CCX |
| comparator n = 2, minimum CCX (≤ 13 gates, NOP padding) | 6 | not resolved: the first query (≤ 4 CCX) ran > 12 min, killed | | 4 |

What these say. At these tiny widths our general constructions are 1.6–2×
longer than optimal in gates. They are often *better* in Toffolis: the
gate-optimal circuits trade CNOTs for CCX (5–6 CCX vs 4). The optimal
small circuits do not extend into a better n-bit family that we could
find. The 7-gate n = 2 adder writes carries straight into the next sum
bit; its general form is the no-ancilla adders of Takahashi–Tani–Kunihiro,
which cost more Toffolis. The constant-specific modadd (6 gates for N = 3)
is a truth-table optimum and doesn't extend to general N. So the
certificates mostly certify "small-n optima are constant-specific". The
transferable output of the SAT work is the **window rules** below.

**SAT peephole on the real circuit** (`tools/superopt/peep.py`, section 1
`sat_rules`). It slides over the round-0 controlled-U of each record
instance. From every gate it takes the longest run of consecutive gates on
≤ 5 wires (≤ 10 gates), plus the wires whose value is known at that point
(constant propagation from "ancillas start at 0"). It asks SAT for the
shortest equivalent on every input consistent with those constants.
Results: 220–226 distinct windows per circuit, 32–35 of them improvable,
37 distinct rules over the four instances. 33 of these do not increase CCX
and are kept. A second iteration on the rewritten circuits found only 2–3
more gates per circuit, so this window shape has converged. The applied
rewrites are as follows.
`MAJ; CNOT(carry, out); UMA → CCX; CNOT; CCX; CNOT`. This is the Cuccaro
top-bit simplification (7 → 4 gates), at every adder's carry-out. The same
idea gives the comparator's top (7 → 4–5). Several constant-aware rules
use `c0 = 0`, `t = 1` or a 0 ancilla to drop a CCX entirely (e.g. `6 → 3`
gates, 2 → 1 CCX). The rest are 1-gate CNOT/CCX reorderings at block
junctions. Net effect: −1.6 % gates, −2.4 % CCX on top of everything
else. A bigger window shape (≤ 6 wires, ≤ 12 gates) on the already
rewritten 20-bit circuit found 4 more improvable windows (6→5, 10→8, 9→7,
8→7 gates; all constant-aware) in the first ~12 min. That search was
stopped. These rules are not in the table, and their effect would be a
fraction of a percent.

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

`qsim run shor --semiclassical --sliced --window 4 --oracle {windowed |
windowed-opt} --modulus N --seed S --tries 1` (31-bit: `--f32 --seed 2`).
One lock per 31-bit run, 40 s gaps between locks. Final binary is
`Opts::ALL` (block passes), `research/data/superopt/bench3.log`. Load
averages were 17–35 throughout, from other agents' unlocked work, so
absolute times are 0–15 % above the quiet-machine record (133.4 s). The
interleaved ratio is the number to use.

| N (bits, qubits) | oracle | gates / run | Toffolis / run | time, 3 runs | eval (gate work) | min | Δ |
|---|---|---|---|---|---|---|---|
| 10 161 323 (24, 104) | windowed | 819 272 | 277 632 | 0.893 / 0.287 / 0.312 s | 0.17 s | 0.287 s | |
| | windowed-opt | 464 090 | 120 672 | 0.507 / 0.287 / 0.299 s | 0.11 s | 0.287 s | ±0 (eval −35 %) |
| 221 643 407 (28, 120) | windowed | 1 264 500 | 409 248 | 16.73 / 16.30 / 16.55 s | 12.25 s | 16.30 s | |
| | windowed-opt | 737 154 | 189 728 | 11.98 / 12.03 / 13.09 s | 7.81 s | 11.98 s | **−26.5 %** |
| 1 537 596 787 (31, 132) | windowed | 1 704 645 | 528 178 | 153.2 / 148.5 / 137.8 s | 112.4 s | 137.8 s | |
| | windowed-opt | 1 039 520 | 261 454 | 105.5 / 100.5 / 97.5 s | 72.0 s | **97.5 s** | **−29.2 %** |

Every pair measured the same integer (24-bit 150 071 647 041 326, 28-bit
19 301 499 017 721 646, 31-bit 2 059 039 373 337 077 151), found the same
order, and factored N. `gate_branch_ops` for the 31-bit run went
1.341e14 → 8.18e13 (−39 %). Peak RSS is unchanged (4.28 GB): memory is set
by the support, not the circuit. An earlier session with global passes
(`bench2.log`) gave 135.7 → 97.6 s at 31 bits and 16.21 → 13.12 s at 28
bits, but 0.29 → 1.41 s at 24 bits (build cost). That regression is what
`block_passes` fixed.

## 6. Literature: known vs new

| our block | literature | verdict |
|---|---|---|
| unary-iteration lookup, `2(2^w−1)` CCX reversible | Babbush et al. 2018 (arXiv:1805.03662, Fig. 7): `L − 1` Toffolis with temporary-AND (measurement) uncompute, `2(L − 1)` reversible; Gidney 2019 windowed arithmetic (arXiv:1905.07682) uses it | **known**. Our baseline simply didn't implement it: the LSB-first prefix sharing never shares in counting order |
| lookup with fewer ancillas | Khattar & Gidney 2024 (arXiv:2407.17966): unary iteration with conditionally clean ancillae, 2.5N Toffolis with log* n ancillas | theirs uses fewer qubits and more Toffolis; ours (w ancillas, 2N) is the standard trade-off |
| unlookup | Gidney 2019: measurement-based unlookup, O(√L) Toffolis | **better than ours**, but needs X-basis measurement and phase fix-ups. The exact sliced engine tracks X/CNOT/CCX permutations only, so it is out of scope (it would roughly halve the lookup Toffolis again) |
| tree-XOR optimal fan-out DP | — | not found in the papers we checked (Babbush 2018, Gidney 2019, Khattar–Gidney 2024); elementary; CNOT-only (−20 % of lookup gates, no Toffoli change). Possibly folklore in production QROM code |
| comparator-based modadd, 8n Toffolis (3 adders + 1 comparator) | Cuccaro et al. 2004 (quant-ph/0410184) comparator; VBE/Beauregard structures; Häner–Roetteler–Svore 2017 (arXiv:1611.07995) | **known**; the existing generic peephole already derives it from the baseline's `sub L; add L` |
| Toffoli cost of addition | Gidney 2017 (arXiv:1709.06648): n Toffolis per adder with temporary AND | **better than ours** (≈ half) in a measurement-capable model; our 2n is the reversible Cuccaro count |
| comparators | Vandaele 2026 (arXiv:2603.12917): Θ(n) gates, Θ(log n) depth, minimal qubits | ours is a plain ripple (2n CCX, linear depth); not better |
| the `K` register (n qubits holding N) | Gidney 2025, classical–quantum adder with 3 clean ancillae and 4n Toffolis (arXiv:2507.23079) | would save n qubits at +2n Toffolis per constant addition; not adopted (qubit count is not the simulator's cost driver) |
| modular reduction at all | Gidney–Ekerå 2019 (arXiv:1905.09749): coset representation; Gidney 2025 (arXiv:2505.15917): approximate residue arithmetic (Chevignard–Fouque–Schrottenloher 2024) | **better than ours** and the real state of the art. They remove the comparison/reduction adders entirely, at the price of a bounded approximation. This exact-arithmetic study deliberately did not adopt them |
| MAJ;CNOT;UMA top-bit (7 → 4), constant-aware window rewrites | Cuccaro 2004 top-bit simplification; Takahashi et al. (2009) ancilla-free adders; T-count/T-depth ripple-adder optimisations (arXiv:2401.17921) | the top-bit rule is **known**. The constant-aware rule table is a mechanical rediscovery plus a few junction-specific rewrites; nothing structurally new |
| SAT exact synthesis of reversible circuits | Große, Wille, Dueck, Drechsler 2009 (SAT-based exact Toffoli synthesis); Golubitsky & Maslov 2012 (all optimal 4-bit NCT circuits) | **known technique**. Our certificates (≤ 8 wires, partially specified with clean-ancilla don't-cares) are small and confirm the expected picture |

**Bottom line.** Against our own implementation: −39 % gates, −51 %
Toffolis, −28 % end-to-end time at 31 bits, with no new qubits. Against
the published state of the art: nothing here beats Gidney–Ekerå-style
constructions. Those use measurement-based uncomputation (lookups and
adders at about half our Toffoli count) and avoid exact modular reduction.
The optimised oracle is now a faithful exact reversible version of the
standard windowed multiplier, which is what the baseline was meant to be.

## 7. What did not work / limits

* **Toffolis are still 8n per modular adder.** This dominates: 82 % of the
  CCX per round. Every exact reversible alternative we considered needs
  ≥ 8n. Folding `−N` into the lookup table (look up `T − N`, one (n+1)-bit
  adder, a conditional `+N`, an extra lookup that converts `T − N` back to
  `T`, then the comparator) would give `6n + O(2^w)` CCX. It saves ~12 % of
  modadd Toffolis but no gates, so it was not implemented.
* **Window rules are consecutive-gate windows.** The SAT peephole does not
  commute gates past each other. Commutation-aware windows (the peephole
  pass's DAG) could find more; the consecutive shape converged after two
  iterations.
* **SAT scaling.** The gate-count search proves n = 2 blocks in seconds.
  The n = 3 comparator reached k = 7 UNSAT in 49 s, and later k took much
  longer (k = 8 UNSAT took 456 s; k = 9 was stopped unresolved after 13 min). Toffoli-minimisation with NOP padding is weak: the
  n = 2 comparator's "≤ 4 CCX" query did not finish in 12 min. A dedicated
  encoding (symmetry breaking for commuting gates, a CCX-count objective
  via assumptions) would be the next step.
* **Build cost.** With passes on the whole circuit (`block_passes: false`),
  the build costs 0.5–1.2 s per run and made the 24-bit run 4× slower
  (0.29 → 1.4 s). The block-pass default fixes this (24-bit: 0.31 → 0.30
  s). `run_semiclassical` builds every round's circuit twice: once for the
  gate count and once in the engine.
* The window-size DP gains 0.3 %; w = 4 remains optimal at n ≤ 31.
* Time saving < gate saving. Only the gate evaluation (≈ 80 % of the
  31-bit run) scales with gates; sort, P(1) merge and collapse are
  unchanged. The 28-bit run is evaluation-light (sort/collapse are ~25 %),
  so it saves less.
