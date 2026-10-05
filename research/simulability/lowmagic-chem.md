# Low-magic quantum chemistry beyond state-vector size: where it exists, and why it does not help

Branch `exp/lowmagic-chem`. Author: qsim-lowmagic-chem agent (round 5, 5 Oct 2026). Base: main b569597.
Code: `src/chem.rs` (FCIDUMP, Pauli-rotation programs, span-filtered Jordan–Wigner Hamiltonian, compressed-state
energies, Rotosolve, register Lanczos), `src/engines/adaptive.rs` (`CompressedState::register_terms`),
`examples/lowmagic_chem.rs` (`profile`, `energy`, `check`), `tests/lowmagic_chem.rs` (8 tests; fixtures in
`tests/data/lowmagic-chem/`). Python: `research/data/lowmagic-chem/{chem.py, dmrg.py, run.py, crosscheck.py, tables.py, headline.py}`.
Data: `research/data/lowmagic-chem/` (`refs/` classical references, `runs/*.jsonl` every engine run,
`tables.md` all tables). Machine: Mac M1 Pro, 1 thread per job, ≤ 2 jobs, 1-min load 11–24 from other agents —
all timings are indicative only.

## Headline

**No chemistry or lattice workload we tried stays low-magic in a way that yields a chemically useful number beyond
state-vector size.** The structure is simple and exact:

1. **For molecular circuits the active dimension d is the tapered qubit count.** For any circuit of fermionic
   rotations on a determinant (Trotter steps of the molecular Hamiltonian, UCCSD, UCCD, k-UpCCGSD, QPE on the
   system register) d equals n − k, where k is the number of independent Z₂ symmetries (N_α and N_β parity, plus the
   abelian point-group bits) — exactly the number of qubits removable by symmetry tapering (Bravyi et al. 2017,
   Setia et al. 2020). Measured on 14 molecules × 10 circuit families, n = 4–24, and from the GF(2) rank at
   n = 24–100: n − d ∈ {2, …, 5} always. d is identical in Jordan–Wigner and Bravyi–Kitaev (a theorem: both map
   occupation vectors linearly), independent of the Trotter step dt (0.01, 0.1, 0.5 give the same d, W_d and
   saturation point), and reached within the first 1–4 % of the first Trotter step for n ≥ 12 (JW; 7–11 % in BK,
   whose term order differs). Pair (seniority-zero)
   ansätze have d = n/2 − 1 (− point-group bits). The branching rank overflows (> 1024 terms) for every Trotter,
   UCCSD, UpCCGSD and QPE circuit with n ≥ 8.
2. **The only way to keep d ≪ n is to truncate the ansatz.** d is then a design parameter: we select CCSD
   excitations by |t| until the GF(2) span of their x-vectors reaches D dimensions, then add every other excitation
   whose x-vector is already in the span for free ("span-closed" selection; K ≫ D, e.g. 537 generators in a
   24-dimensional register for H₅₀). The compressed register is then a determinant space of 2^D slots
   (HF ⊕ span) of which only `nnz` are physical (number- and spin-conserving): 106 of 256 at D = 8, 14,688 of 65,536
   at D = 16.
3. **Those registers are far too small to hold chemistry beyond 30–40 qubits.** We bound every circuit in a register
   by the lowest eigenvalue of H projected onto that register (Lanczos, `register_ground`; no circuit with the same
   rotation span can go lower). At 32–100 qubits the register optimum at D ≤ 20 (and the D = 24 circuit energy)
   recovers **4–72 % of the correlation energy** — H₅₀/STO-3G (100 qubits) 11 %, H₃₀ 27 %, H₂₀ 44 %,
   N₂/cc-pVDZ (52 qubits) 49 %, H₂O/cc-pVDZ 52 %, 4×4 H sheet 69 % — i.e. errors of 78–740 mEh at equilibrium,
   while CCSD is within 4–26 mEh, CCSD(T) within 3 mEh, and DMRG (our reference) is essentially exact. The largest
   exact simulation is a **100-qubit, 537-generator (4,296 Pauli rotations) UCC state of H₅₀ in a 24-qubit register:
   one exact energy in 229 s, error +723 mEh** (CCSD: +26 mEh in seconds). The largest *best* result is the 4×4 sheet
   at D = 24 (32 qubits, 1,141 generators, 18 min): +78 mEh, still 5× CCSD's error.
4. **Where CCSD fails (stretched bonds) the variational exact simulation beats CCSD only at sizes where FCI or DMRG
   is trivial.** Stretched H₆ (12 qubits): CCSD −126 mEh (overshoots), span-closed UCC with d = 8 and Rotosolve
   +12 mEh. Stretched N₂/STO-3G (16 qubits): CCSD −102 mEh, UCC (d = 12) +14 mEh. At 32–100 qubits the stretched
   systems are out of reach: 4×4 sheet at 1.8 Å, register optimum +742 mEh (CCSD unconverged, CCSD(T) +297 mEh);
   N₂/cc-pVDZ at 2.0 Å (52 qubits), +179 mEh vs CCSD +70 / CCSD(T) −33 mEh — and DMRG (M = 300–500) solves all of
   them to sub-mEh in 1–15 minutes.
5. **Exactness was checked independently at 32–100 qubits.** A separate sparse-determinant UCC simulator with a
   Slater–Condon energy (`crosscheck.py`, shares only the integrals) reproduces the compressed-state energies of
   H₂₀, 4×4 H, N₂/cc-pVDZ, H₂O/cc-pVDZ and H₅₀ (d = 12) to ≤ 4e-11 Eh with identical determinant counts — and runs in
   0.1–1.2 s, which is the honest classical comparator for this kind of state.

So: low magic in chemistry exists only as "few rotations" (truncated ansätze, d = number of independent excitation
patterns) or as symmetry (tapering, k ≤ 5 qubits); neither gives a chemically meaningful number beyond SV size, and
for every system we ran the classical method (CCSD/CCSD(T) at equilibrium, DMRG everywhere) is better and cheaper.
This is a negative result with a precise reason (§2) and a quantitative ceiling (§4).

---------------------------------------------------------------------------------------------------

## 1. Setup

**Systems** (PySCF 2.14 RHF; frozen core for N₂): H₂ (0.74, 2.5 Å), LiH, BeH₂, H₂O, N₂ (1.098, 2.0 Å) in STO-3G;
linear H₄, H₆, H₈, H₁₀ (1.0 Å; H₄/H₆/H₁₀ also stretched, 2.0/1.8 Å); 2D H sheets 3×4, 4×4, 4×5 (1.0/1.8 Å);
H₂₀, H₃₀, H₅₀ chains (1.0 Å; H₅₀ also 1.8 Å); N₂/cc-pVDZ (eq. and 2.0 Å) and H₂O/cc-pVDZ. Qubits n = 2 × active
orbitals = 4 … 100.

**References.** HF, MP2, CCSD, CCSD(T) (PySCF); FCI where the determinant space is ≤ 3·10⁷ (n ≤ 24); DMRG
(block2 SU(2), M = 300–500, Löwdin AOs in chain/snake order for hydrogen systems, Fiedler-ordered canonical MOs
otherwise) for the rest. DMRG accuracy check: stretched 3×4 sheet, M = 300: 6 µEh above FCI.

**Circuits** (OpenFermion JW/BK, written as Pauli-rotation programs, `chem.py`):
- first-order Trotter of the full qubit Hamiltonian from HF, 2 steps, dt ∈ {0.01, 0.1, 0.5};
- UCCSD and UCCD with the CCSD amplitudes as angles; 1- and 2-UpCCGSD (generalized pair doubles + singles);
  pUCCD (all pair doubles);
- selected UCC `selK`: the K largest |t| CCSD excitations;
- span-closed UCC `spanD` (above): K = 14 … 1,818 generators with d ≤ D;
- textbook QPE (t = 3, 5 counting qubits, controlled first-order Trotter step, inverse QFT, all as Pauli rotations +
  H) on HF;
- lattice models: 1D Hubbard (L = 6, 10; U/t = 0, 4, t = 0) and J₁–J₂ Heisenberg chains (L = 12, J₂ = 0 and the
  Majumdar–Ghosh point 0.5) from the exact dimer (stabilizer) ground state.

**Engines.** The atlas profile (`magic_atlas::profile`: d, f, W_d, saturation point) on the lowered circuit; the
branching-rank engine (`stab_rank`, cap 1024); exact energies on the compressed state (`CompressedState`), with
the Jordan–Wigner Hamiltonian generated in Rust directly from the FCIDUMP integrals and **restricted to the
fermionic monomials whose x-vector lies in the circuit's GF(2) span** (exact for programs whose Clifford part only
flips/phases qubits, e.g. a determinant reference: other terms have zero expectation). For H₅₀ this keeps 18,836
of 12,257,500 monomials and builds in 1.2 s. Angles are optimised by Rotosolve (each excitation enters the energy
as a degree-2 trigonometric polynomial, fixed exactly by 5 evaluations).

**Correctness.** `check` (33 runs, n ≤ 20): compressed-state energy vs state vector |ΔE| ≤ 9e-11, infidelity
≤ 1e-12, filtered = unfiltered Hamiltonian; HF energies equal PySCF RHF to 1e-10 (Slater–Condon and Pauli);
Rotosolve on H₂ reaches FCI to 1e-10; the H₄ selected-UCC energy equals an independent OpenFermion evaluation
(expm of the fermionic generators, OpenFermion's own JW Hamiltonian) to 1e-12, and OpenFermion's sector ground state
equals PySCF FCI. All in `tests/lowmagic_chem.rs` (8 tests) or `runs/check.jsonl`.

## 2. Structure: d is the tapered qubit count

**Lemma 1 (encoding invariance).** JW, BK and parity encodings map a determinant with occupation vector o to the
basis state |B o⟩ for an invertible GF(2) matrix B, and every fermionic operator that changes occupations by v
to Pauli strings with x-part B v. A circuit of fermionic rotations on a determinant therefore has rotation x-vectors
B·{v_j}, and d = rank{v_j} in every such encoding. *Measured:* JW and BK give identical d for all 56 (molecule,
circuit) pairs; only the position where d saturates moves (BK: later, because the term order differs).

**Lemma 2 (symmetry bound).** If every generator conserves N_α, N_β and the irreducible representation of an
abelian point group, every v_j is orthogonal (mod 2) to the α mask, the β mask and the Z₂ character masks, so
d ≤ n − k, k = their GF(2) rank. This is exactly the set of Z₂ symmetries used for qubit tapering. Equality holds
as soon as the excitation x-vectors span the orthogonal complement — true for the Hamiltonian itself, every Trotter
step, UCCSD, and generalized ansätze (which break point-group symmetry: d = n − 2).

| circuit family | measured d | n − d |
|---|---|---|
| Trotter (any dt, JW or BK), UCCSD | n − 2 (no symmetry) … n − 5 (BeH₂, D₂h) | 2–5 |
| UCCD | n − k − 1 (no singles: one more parity) | 3–5 |
| k-UpCCGSD (k = 1, 2) | n − 2 | 2 |
| pUCCD (pairs) | n/2 − 1 (− point group) | ≈ n/2 |
| QPE, t counting qubits | t + d(H) | 2–4 |
| Hubbard, t ≠ 0 (U = 0 or 4) | n − 2 | 2 |
| Hubbard, t = 0 | 0 | n |
| Heisenberg J₁–J₂ from the dimer state | n − 2 | 2 |

(Full table: `tables.md` §"Structure map"; the GF(2)-rank table covers n = 24–100, e.g. H₅₀: d(UCCSD) = 97,
d(pUCCD) = 49.)

(Equality needs the MO basis to exhibit the symmetry: N₂ in STO-3G has D₂h, k = 5, but the canonical degenerate π
orbitals come out mixed, so only k = 4 is visible to d; with symmetry-adapted orbitals tapering and d agree.)

So the compressed state is, for chemistry, automatic qubit tapering: 2–4 qubits, never the 10–60 that "beyond SV"
needs.

**Angle blindness and saturation.** d, W_d and the saturation point are identical for dt = 0.01, 0.1 and 0.5. d
saturates after 21 of the 665 rotations of the first step (BeH₂, JW, 3 %), 83 of 7,150 (H₁₀, 1.2 %), 29 of 1,176
(N₂, 2.5 %). The magic atlas (§6.4) already noted that Trotter step size is invisible to d except at exact Clifford angles;
for molecules there are no Clifford points (the coefficients are incommensurate). Exact frame simulation of a
molecular Trotter circuit therefore costs `2^{n−k}` per rotation from the first percent of the first step on,
regardless of dt — there is no time-step/error trade-off to exploit. (The *state* magic of e^{−iHt}|HF⟩ is
O(t²) small at small t, but exploiting that is approximate simulation — stabilizer extent / Pauli propagation — not
the exact engines studied here.)

**QPE.** With an HF reference, d = t + d(H): H₂ 4 = 3 + 1, LiH 11 = 3 + 8, H₆ 12 = 3 + 9 (t = 3). The atlas's
d = t case needs a stabilizer state that is an eigenstate of *every* Trotter factor; for molecules this happens
only in trivial limits (H₂ at R → ∞: the singlet-pair state is stabilised by every Pauli term of the minimal-basis
Hamiltonian; at R = 2.5 Å the ground state is already non-stabilizer). The Majumdar–Ghosh dimer product is an exact
stabilizer eigenstate of the J₁–J₂ chain, but not of the individual bond terms, so a Trotterised evolution from it
has d = n − 2 (measured, L = 12).

**Lattice special points.** Hubbard at t = 0 (atomic limit) is diagonal: d = 0 for any U, trivially. Hubbard at U = 0
(free fermions) has d = n − 2 — maximal for the frame, although matchgate/Gaussian simulation makes it trivially
easy: d (and nullity) do not see free-fermion structure, just as they do not see the product structure of QFT|x⟩
(atlas §6.2).

**Branching rank.** Overflows the 1024-term cap for every Trotter, UCC and QPE circuit with n ≥ 8 (H₂: r = 3–4,
H₄ pUCCD: 109). Determinants are stabilizer states, so the rank is bounded by the number of determinants reached,
which for these circuits is the full symmetry sector; the term-merging heuristics of theory-rank.md find no
low-rank structure.

## 3. Truncated ansätze: d as a design parameter

For a UCC product on HF, d = rank{x(excitation)}: each single/double adds at most one dimension, and an excitation
whose x-vector is a sum of chosen ones adds none (spin-flipped partners, same-spin combinations, chains of
excitations sharing orbitals). Consequences, measured:

- `selK` (top-K amplitudes): d ≈ K/2 (H₅₀: d = 13 at K = 28; N₂/cc-pVDZ: 11 at K = 24–28).
- `spanD` (span-closed): K grows fast at fixed D. H₅₀: D = 8 → 22 generators, 12 → 48, 16 → 107, 20 → 248, 24 → 537;
  3×4 stretched sheet: 1,818 at D = 24.
- The register is wasteful for number-conserving states: `nnz/2^d` = 106/256, 1,132/4,096, 14,688/65,536,
  187,353/1,048,576, 2,565,141/16,777,216 (H chains, D = 8 … 24): 15–40 % of the register is physical, because d only
  knows the two parities, not N (N₂/cc-pVDZ at D = 20: 3 %).

This is the regime the brief asked about (Clifford + few rotations / HF + few excitations). Exact simulation cost
is ~2^d per rotation and the energy evaluation is fast with the span-filtered Hamiltonian (H₅₀, d = 16, 107
generators: 0.2 s per energy; d = 20, 248 generators: 6 s; d = 24, 537 generators: 229 s; 1 thread on a loaded
laptop).

## 4. Energies: what these registers can hold

Errors in mEh against FCI (n ≤ 24) or DMRG (larger); `E_reg` is the best any circuit in the same register can do.
Selected rows (all rows: `tables.md` §"Energies"):

Beyond state-vector size (D = register size; `E_opt` after Rotosolve of the 20 largest amplitudes, `E_reg` the
register optimum, `—` not run; † CCSD not converged; * CCSD(T) reference, DMRG not completed — see caveats):

| system | n | ref | CCSD | CCSD(T) | D=8 E_reg | D=16 E_opt / E_reg | D=20 E_opt / E_reg | D=24 E (CCSD angles) | % corr, best register |
|---|---|---|---|---|---|---|---|---|---|
| h4x4 | 32 | DMRG 500 | +15 | +3 | +212 | +149 / +146 | +133 / +128 | +78 | 69 % |
| h4x4_s | 32 | DMRG 500 | +415† | +297 | +928 | +832 / +815 | +806 / +742 | — | 27 % |
| h20 | 40 | DMRG 300 | +8 | +1 | +276 | +230 / +228 | +204 / +200 | +185 | 44 % |
| h4x5_s | 40 | DMRG 500 | +418† | +164 | +1226 | +1146 / +1096 | +1087 / +891 | — | 31 % |
| h2o_dz | 46 | DMRG 400 | +4 | +1 | +175 | +127 / +127 | +102 / +102 | — | 52 % |
| n2_dz | 52 | DMRG 400 | +12 | +1 | +244 | +179 / +178 | +164 / +163 | — | 49 % |
| n2_dz_s | 52 | DMRG 400 | +70 | -33 | +236 | +217 / +201 | +209 / +179 | — | 72 % |
| h30 | 60 | DMRG 300 | +16 | +3 | +454 | +414 / +411 | +391 / +386 | +363 | 27 % |
| h50 | 100 | CCSD(T)* | +26 | +0 | +794 | +763 / +759 | +746 / +740 | +723 | 11 % |
| h50_s | 100 | CCSD(T)* | +211† | +0 | +4098 | +4138 / +4054 | +4171 / +3958 | +4142 | 4 % |

At and below SV size (FCI reference; all rows in `tables.md`):

| system | n | CCSD | span-closed UCC: D → (d, K, nnz) | E_opt | E_reg |
|---|---|---|---|---|---|
| LiH | 12 | +0.0 | 8 → (8, 34, 69) | +0.0 | — |
| H₂O | 14 | +0.1 | 8 → (8, 30, 65) | +0.2 | +0.1 |
| H₈ | 16 | +1.1 | 8 → (8, 22, 106) / 12 → (12, 168, 1252) | +48.1 / +1.8 | +48.0 / +1.7 |
| N₂ | 16 | +3.9 | 8 → (8, 26, 84) / 12 → (12 = n − 4, 85, 780) | +26.1 / +2.2 | +25.7 / 0.0 |
| H₁₀ | 20 | +2.0 | 12 → (12, 75, 1132) / 16 → (16, 417, 15912) | +56.3 / +3.0 | +55.8 / +2.9 |
| H₆ 2.0 Å | 12 | −125.7 | 8 → (8, 51, 104) | +12.4 | — |
| N₂ 2.0 Å | 16 | −101.8 | 8 → (8, 47, 104) / 12 → (12, 85, 738) | +27.8 / +13.6 | +18.1 / 0.0 |


Reading the table:

- **Small molecules (n ≤ 16, equilibrium):** a span-closed UCC with d = n − k is UCCSD and is as good as CCSD
  (LiH +0.01, H₂O +0.10, BeH₂ +0.38 mEh) — but this *is* state-vector size. At d = 8 below the symmetry bound the
  errors are 0.2–27 mEh (H₂O, BeH₂ fine; N₂ 26 mEh; H₈ 48 mEh).
- **Beyond SV size (n = 32–100):** the register optimum at D = 16 recovers 7 % (H₅₀), 16 % (H₃₀), 31 % (H₂₀),
  41 % (4×4 sheet), 41–44 % (H₂O, N₂ cc-pVDZ) of the correlation energy; each +4 in D adds 2–9 points (D = 24: H₅₀
  11 %, H₂₀ 44 %, 4×4 69 %). CCSD errors are 4–26 mEh at equilibrium, CCSD(T) ≤ 3 mEh. The fraction falls with
  system size at fixed D because correlation energy is extensive and 2^D determinants are not.
- **Stretched (strong correlation):** CCSD diverges or fails to converge (3×4: −307 mEh at the last iterate,
  4×4 / 4×5 / H₅₀ at 1.8 Å: not converged). The variational UCC beats CCSD for n ≤ 16 (H₆: +12 vs −126 mEh; N₂: +14 vs
  −102 mEh), where FCI costs milliseconds. Beyond SV size the register optimum is 0.18–0.9 Ha above DMRG
  (H₅₀ at 1.8 Å: ~4 Ha above the CCSD(T) number, itself unreliable there).
- **Rotosolve vs CCSD angles:** at equilibrium, CCSD amplitudes are already within 0.1–1 mEh of the optimum in
  the same register; stretched, they are useless (H₆: +388 → +12 mEh after optimisation).

The scaling argument behind the numbers: correlation energy is a sum of O(o²v²) pair contributions; capturing a
fixed fraction needs the number of independent excitation patterns, i.e. d, to grow with the system. Every
excitation pattern beyond the first ~25 costs a doubling of the register. Compressed exact simulation is therefore
a selected-CI method in a GF(2)-closed determinant space, and a worse one than the classical selected-CI and
sparse-UCC solvers (which store only the nnz physical determinants and choose them adaptively).

## 5. Is anything useful computed beyond SV size?

The largest exact simulations we ran (all with 1 thread, exact to the 1e-10 level by the same code path validated
in §1):

| simulation | qubits | generators / rotations | d | time per energy | error vs best reference | CCSD error |
|---|---|---|---|---|---|---|
| H₅₀ chain, span24 (STO-3G, 1.0 Å) | 100 | 537 / 4,296 | 24 | 229 s | +723 mEh (vs CCSD(T)) | +26 mEh |
| H₃₀ chain, span24 | 60 | 537 / 4,296 | 24 | 282 s | +363 mEh (DMRG) | +16 mEh |
| 4×4 H sheet, span24 (1.0 Å) | 32 | 1,141 / 9,128 | 24 | 1,107 s | +78 mEh (DMRG) | +15 mEh |
| H₂₀ chain, span24 | 40 | 300 / 2,400 | 24 | 136 s | +185 mEh (DMRG) | +8 mEh |
| N₂/cc-pVDZ 2.0 Å, span20 + Lanczos | 52 | 185 / 1,480 | 20 | 15 s | +179 mEh (register optimum, DMRG) | +70 mEh |

None of these is chemically meaningful (chemical accuracy 1.6 mEh); each is worse than CCSD at equilibrium by one to
two orders of magnitude, and DMRG gives the exact answer for every system in this study. Variants that would make
the frame engines win — a product of non-interacting fragments (f = largest fragment, any n) or Hubbard at t = 0 —
are exactly the cases classical chemistry already treats as trivial (fragment methods, the atomic limit).

## 6. Known vs new

**Known.**
- Symmetry tapering of qubits: Bravyi, Gambetta, Mezzacapo, Temme, arXiv:1701.08213; point-group tapering: Setia,
  Chen, Rice, Mezzacapo, Pistoia, Whitfield, JCTC 16, 6091 (2020). Our d = n − k is this count.
- Classical simulation of UCC far beyond SV size by sparse wavefunctions / factorised UCC: Chen, Cheng, Freericks,
  JCTC 17, 841 (2021); Mullinax, Anastasiou, Larson, Barnes, Mayhall, Tubman et al., arXiv:2301.05726 (up to 64
  qubits, tens of thousands of parameters); Misiewicz et al., "Beyond MP2 initialization for UCC", Quantum 8, 1538
  (2024); sparse VQE simulation, arXiv:2404.10047. These store only physical determinants and are the right
  classical tool; our compressed register is a strictly larger space for the same state.
- Clifford/stabilizer starting points for VQE: CAFQA, Ravi et al., arXiv:2202.12924 (best Clifford state; recovers
  HF-level or better energies, mostly useful at stretched geometries); Clifford-augmented MPS (CAMPS,
  arXiv:2412.17209). HF is a stabilizer state; the stabilizer reference only shifts the starting point.
- Magic of molecular ground states and along bond stretching ("Are molecules magical?", arXiv:2504.06673): ground
  states are generically magic; magic peaks at intermediate bond lengths.
- CCSD breakdown for stretched bonds and 2D/1D hydrogen systems; DMRG as the method of choice for H chains (Motta
  et al., PRX 7, 031059 (2017)).

**New here (as far as I found).**
1. The identification d = n − (number of Z₂ symmetries) for every molecular circuit family on a determinant, with the
   proof via encoding linearity (JW = BK) and the measurement across 14 molecules and 10 families, including the
   saturation point (first 1–4 % of the first Trotter step, JW, n ≥ 12) and the dt-independence.
2. Span-closed excitation selection: at fixed register size D, all excitations in the GF(2) span are free; the
   Rust Hamiltonian builder that restricts the Jordan–Wigner Hamiltonian to span monomials (12 M → 19 k for H₅₀).
3. The register-optimum bound (Lanczos on H projected onto the 2^d frame register): a cheap certificate of what
   *any* circuit with a given rotation span can achieve, used to show the beyond-SV ceiling (2–42 % of correlation at
   D ≤ 20, n = 32–100) independently of optimiser quality.
4. The negative answer itself, with numbers: no low-magic chemistry regime found that is both exactly simulable
   beyond SV size and chemically useful; the frame engines reduce to tapering (§2) or to a poor selected CI (§4).

## 7. Caveats

- Ansatz choice: we selected excitations by CCSD |t|; ADAPT-style energy-gradient selection would give a better
  register at the same D. The register optimum E_reg makes this mostly moot: it is the best any circuit in the same
  span can reach, and it is still far from useful beyond SV size. A *different* span (other excitations) could do
  better, but the extensivity argument (§4) applies to any fixed D.
- Rotosolve was limited (1–4 sweeps, top 20–30 parameters for large spans); `E_opt` is an upper bound on the
  ansatz optimum, `E_reg` a lower bound.
- Stretched-geometry CCSD amplitudes come from unconverged CCSD (marked †); they only serve as a selection order.
- We did not try CAFQA-style Clifford references (non-determinant stabilizer states). They change the reference,
  not the structure: d still counts independent rotation patterns, and the extensivity argument is unchanged.
- QPE with a true stabilizer eigenstate (atlas: d = t) was not found for any non-trivial molecule; we did not
  attempt non-Trotter (exact) controlled evolutions.
- References: DMRG for H₅₀ was stopped before convergence (≈ 4.5 min per sweep at M = 200 on the shared laptop), so
  the H₅₀ rows use CCSD(T); for H₂₀ and H₃₀ CCSD(T) is within 1.2 and 2.8 mEh of DMRG, irrelevant next to the
  700-mEh errors. For H₅₀ at 1.8 Å CCSD did not converge and no reliable reference exists here; its row only shows
  that the register optimum barely moves below HF.
- Timings are from a heavily shared laptop (1-min load 11–24), single runs, not benchmarks.
- FCIDUMPs and programs are not committed (H₅₀: 40 MB); `chem.py prep NAME DIR` regenerates them deterministically.

## 8. Reproduce

```
python3 -m venv ~/qsim-chem-venv && ~/qsim-chem-venv/bin/pip install pyscf openfermion block2
cargo build --release --example lowmagic_chem
B=target/release/examples/lowmagic_chem; D=~/lmc-data; cd research/data/lowmagic-chem
for m in $(python chem.py list); do python chem.py prep $m $D; done; python chem.py lattice $D
python dmrg.py h50 $D 300                                   # DMRG references (and others)
WORKERS=2 python run.py $B $D $D/out/check.jsonl 'check:*.uccsd.jw.prog' 'check:*.sel8.jw.prog'
WORKERS=2 python run.py $B $D $D/out/profile.jsonl 'profile:*.prog:1024'
LANCZOS=40 WORKERS=2 python run.py $B $D $D/out/energy.jsonl 'energy:*.span16.jw.prog:1:20'
$B energy $D/h50.fcidump $D/h50.span24.jw.prog 0             # 100 qubits, 537 generators, d = 24
tools/datafiles.py unpack                                    # (repo root) restores runs/profile.jsonl
python tables.py . > tables.md && python headline.py .       # in research/data/lowmagic-chem
cargo test --release --test lowmagic_chem
```
