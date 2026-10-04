# Audit of exp/shor-r4 (head 66acd8b): gate-level Shor to 31-bit generic N

Independent audit (qsim-shor-r4-audit, 3 Oct 2026). Verdict: **FIX-THEN-MERGE**
— the record and the simulation are sound; four quantitative sentences in
`research/shor.md` were wrong and are corrected on this branch.

## 1. Is it really the circuit?

Code read (`src/shor/sliced.rs`, `src/shor_window.rs`, `src/shor.rs`):

* Each round builds the controlled-`U_{a^(2^k)}` circuit with
  `shor_window::controlled_ua` and compiles it with `SlicedProgram::compile`,
  which **rejects** every gate except X / CNOT / CCX / SWAP (test
  `compile_rejects_non_permutation_gates`). `prog.eval` loops over every
  compiled op for every 64·L-branch slice; the output work-register values
  (`uk`) are produced only by that evaluation. No `a^x mod N` or `a·x mod N` of
  the quantum register is computed anywhere in the sliced path. The only
  classical arithmetic is circuit compilation: the per-round multiplier
  `a^(2^k) mod N` and the lookup tables `v·a^(2^k)·2^(jw) mod N` — that is how
  every compiled Shor circuit (Beauregard, Gidney) is built.
* The safety checks are `assert!` (active in release) and do fire: new tests
  in `tests/shor_r4_audit.rs` remove one lookup-uncompute Toffoli
  (→ "ancillas did not return to 0"), append `X(ctrl)` (→ "control qubit
  changed"), and append an unconditional `X(x0)` (→ control-0 "not the
  identity"). Injectivity of the outputs is also asserted after the sort.
* Independent cross-check of the oracle (`research/data/shor_r4_audit/perm_check.py`):
  a gate interpreter written from scratch (numpy bit ops) runs the gate list
  dumped by `examples/audit_shor_r4.rs`. N = 15, four bases, w = 1..4
  (21–24 qubits, 16 circuits): **all 2^nq basis inputs**, the map is a
  bijection and on the valid domain gives `x → a·x mod N` (control 1) / `x`
  (control 0) with every ancilla clean (`perm_check_all.out`; the N = 21
  all-input case at 25 qubits hit my 50-min timeout and was dropped).
  N = 21…253 (5–8 bits), 110 (N, a, w) circuits, w = 1..5: valid domain,
  same result (`perm_check_valid.out`).
  At full size (`big_sample_check.py`): the 31-bit record's circuits
  (132 qubits, ≈ 27.4 k gates per round, for a, a², a^(2^61)) and the
  52-bit special N's (216 qubits, ≈ 71 k gates) on 4006 random valid inputs
  each: correct products, ancillas clean. 27 431 × 62 rounds ≈ the reported
  `total_gates` 1 704 645.

## 2. Is the simulation exact?

* Non-classical gates act only on the recycled control: the round is
  `H(0)`, the permutation block, `Phase(0, φ)`, `H(0)`, measure. The state
  before each round is `|0⟩|ψ⟩|0…0⟩` (ancillas checked clean), so storing the
  work-register support `{x: ψ_x}` is exact, and `P(1) = ‖ψ − e^{iφ}Uψ‖²/4`,
  collapse `(ψ ∓ e^{iφ}Uψ)/(2√p)` is the exact control algebra (checked by
  hand against the code; the sign convention `ph = −ph` for outcome 1 is right).
* Measured-outcome distributions, whole measurement tree, sliced windowed
  f64 (`dist_check.py`):
  - vs the textbook phase-estimation distribution computed here from scratch
    (`P(y) = Σ_z |2^−t Σ_{x: a^x = z} e^{2πixy/2^t}|²`): 48 (N, a, w) cases,
    N = 15…119, max |Δp| ≤ 1.5e-13 (direct exponential sums; the script now
    uses an FFT, ≤ 1.6e-16), and
    8-bit N = 143, 187, 221, 247 (t = 16): max |Δp| ≤ 1.6e-16
    (`dist_check_A.out`, `dist_check_8bit.out`);
  - vs a dense state-vector simulation, written here, of the **whole
    (4n+4+w)-qubit register** (H/phase on qubit 0, the controlled-U as the
    permutation obtained from the gate list on every basis index, measure and
    reset; N = 15, 21–23 qubits): max |Δp| = 1.7e-15 (`dist_check_B.out`);
  - in-repo dense reference: new test
    `sliced_windowed_equals_dense_textbook_distribution` (vs
    `full_qft_distribution`, the 3n-qubit dense textbook circuit; N = 15, 21,
    33, 35, 39, w = 1..4) passes at 1e-12. The branch's own `shor_scale`
    suite passes.

## 3. Reproduction on the Mac (M1 Pro, under the bench lock; `mac_repro.log`)

| N | bits | claimed | audit re-run |
|---|---|---|---|
| 10 161 323 | 24 | 0.95 s / 133 MB (load 35, earlier commit) | 0.293 / 0.277 s, 105 MB, same a / measured / r |
| 221 643 407 | 28 | 16.1 s / 2.82 GB (f32 15.2 s / 2.02 GB) | 16.04 s / 2.82 GB (f32 14.99 s / 2.02 GB), same measured integer, factor 15 601 |
| 1 537 596 787 | 31 | 134.4 s / 4.28 GB (f32) | **134.09 s / 4.28 GB**, same a = 457 167 243, same measured 2 059 039 373 337 077 151, r = 256 252 500, factor 52 501 |

Load 1.4–5.9 (1-min) during the runs. Exact command from `research/shor.md`.
The 24-bit row was stale (measured before the final collapse was skipped,
at load 35) and is updated.

## 4. Support law on fresh instances (`support_check.py`, seed 2026)

15 new (N, a) with 12–22-bit N, random bases, deliberately **odd r**
(`a = b^(2^ν₂(λ))`) and **high ν₂(r)** (p ≡ q ≡ 1 mod 2^(h−4); r = 256 = 2^8,
r = 4224 = 2^7·33): `|S_i| = B_i` in **508 / 508 rounds**, never above; the
work counter equals `Σ 2·|S_i|·G_i` exactly in all 15; peak support equals
`max(r_odd, r/2)` in all 15.

## Corrections made to research/shor.md

1. Closed form: `Σ_i B_i ≈ r_odd·(2n − log₂ r) + 2r` → **`+ r`**. Derivation: the
   last ν rounds contribute `r_odd(1 + 2 + … + 2^(ν−1)) ≈ r`, not 2r. Against the
   exact Σ B_i of all 12 records in `cost_law_records.txt`, `+ r` is within
   0.3 %; `+ 2r` was 6–66 % high. (The quoted 0.3 % agreement of W with the
   counter used the exact Σ B_i, so that claim stands; the closed form did not
   match to 0.3 % as stated.)
2. "The final ν rounds cost about 4·Ḡ·r" → `2·Ḡ·(r − r_odd)`.
3. "memory ~16–24 B per element of the peak support" → measured peak RSS
   ≈ 33 B (f32) / 51 B (f64) per peak-support element (ψ and the `(Ux, ψ_x)`
   join coexist); 16/24 B is the stored state alone.
4. "gate-level record" → "this repo's gate-level record", plus a note that
   simulated Shor has been run at larger N elsewhere (Willsch et al. 2023,
   39-bit N, 549 755 813 701, on a GPU supercomputer, different construction).

## Framing

The round-4 text is honest: it says the cost is linear in r, that r ≈ N/c
for generic N and random bases (exponential in the bit length), that this is
exact simulation of a compilable X/CNOT/Toffoli circuit and not a factoring
speed-up, and it labels the 43–52-bit p(2p−1) rows as classically trivial.
The branch does not touch README.md / RESULTS.md; if the parent adds the
31-bit row there, keep the "cost ∝ r, exponential in bits for generic N"
sentence next to it and the word "this repo's" before "record".

## Files

`examples/audit_shor_r4.rs` (dump / dist / trace helper), `tests/shor_r4_audit.rs`,
`research/data/shor_r4_audit/` (scripts + outputs).
