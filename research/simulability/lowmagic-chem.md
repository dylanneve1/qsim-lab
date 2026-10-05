# Low-magic quantum chemistry beyond state-vector size: where it exists, and why it does not help

Branch `exp/lowmagic-chem`. Author: qsim-lowmagic-chem agent (round 5, 5 Oct 2026). Base: main b569597.
Code: `src/chem.rs` (FCIDUMP, Pauli-rotation programs, span-filtered Jordan–Wigner Hamiltonian, compressed-state
energies, Rotosolve, register Lanczos), `src/adaptive.rs` (`CompressedState::register_terms`),
`examples/lowmagic_chem.rs` (`profile`, `energy`, `check`), `tests/lowmagic_chem.rs` (8 tests; fixtures in
`tests/data/lowmagic-chem/`). Python: `research/data/lowmagic-chem/{chem.py, dmrg.py, run.py, tables.py}`.
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
   n = 24–100: n − d ∈ {2, 3, 4} always. d is identical in Jordan–Wigner and Bravyi–Kitaev (a theorem: both map
   occupation vectors linearly), independent of the Trotter step dt (0.01, 0.1, 0.5 give the same d, W_d and
   saturation point), and reached within the first 1–4 % of the first Trotter step. Pair (seniority-zero)
   ansätze have d = n/2 − 1 (− point-group bits). The branching rank overflows (> 1024 terms) for every circuit
   with n ≥ 8.
2. **The only way to keep d ≪ n is to truncate the ansatz.** d is then a design parameter: we select CCSD
   excitations by |t| until the GF(2) span of their x-vectors reaches D dimensions, then add every other excitation
   whose x-vector is already in the span for free ("span-closed" selection; K ≫ D, e.g. 537 generators in a
   24-dimensional register for H₅₀). The compressed register is then a determinant space of 2^D slots
   (HF ⊕ span) of which only `nnz` are physical (number- and spin-conserving): 106 of 256 at D = 8, 14,688 of 65,536
   at D = 16.
3. **Those registers are far too small to hold chemistry beyond 30–40 qubits.** We bound every circuit in a register
   by the lowest eigenvalue of H projected onto that register (Lanczos, `register_ground`; no circuit with the same
   rotation span can go lower). At 40–100 qubits, the register optimum at D = 16–20 recovers **2–42 % of the
   correlation energy** (H₅₀/STO-3G, 100 qubits: 7 % at D = 16; N₂/cc-pVDZ, 52 qubits: <!--N2D16-->; 4×4 H sheet,
   32 qubits: 42 %), while CCSD recovers 97–99 % at equilibrium in seconds and DMRG is near-exact. The largest exact
   simulation run is a **100-qubit, 537-generator UCC state of H₅₀ in a 24-qubit register** (<!--H50D24-->), exact
   to ~1e-12 and chemically useless (error <!--H50D24ERR--> mEh).
4. **Where CCSD fails (stretched bonds) the variational exact simulation wins over CCSD only at sizes where FCI
   or DMRG is trivial.** Stretched H₆ (12 qubits): CCSD −126 mEh (overshoots), span-closed UCC with d = 8 and
   Rotosolve +12 mEh. Stretched N₂/STO-3G (16 qubits): CCSD −102 mEh, UCC (d = 12) +14 mEh. At 32–100 qubits the
   stretched sheets/chains need d ≈ n − k to get within 100 mEh, i.e. state-vector size again, and DMRG (M = 300–500)
   solves them to µEh in minutes.

So: low magic in chemistry exists only as "few rotations" (truncated ansätze, d = number of independent excitation
patterns) or as symmetry (tapering, k ≤ 4 qubits); neither gives a chemically meaningful number beyond SV size, and
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

**Correctness.** `check` (30 programs, n ≤ 20): compressed-state energy vs state vector |ΔE| ≤ 9e-11, infidelity
≤ 1e-12, filtered = unfiltered Hamiltonian; HF energies equal PySCF RHF to 1e-10 (Slater–Condon and Pauli);
Rotosolve on H₂ reaches FCI to 1e-10; the H₄ selected-UCC energy equals an independent OpenFermion evaluation
(expm of the fermionic generators, OpenFermion's own JW Hamiltonian) to 1e-12, and OpenFermion's sector ground state
equals PySCF FCI. All in `tests/lowmagic_chem.rs` (8 tests) or `runs/check.jsonl`.

## 2. Structure: d is the tapered qubit count

**Lemma 1 (encoding invariance).** JW, BK and parity encodings map a determinant with occupation vector o to the
basis state |B o⟩ for an invertible GF(2) matrix B, and every fermionic operator that changes occupations by v
to Pauli strings with x-part B v. A circuit of fermionic rotations on a determinant therefore has rotation x-vectors
B·{v_j}, and d = rank{v_j} in every such encoding. *Measured:* JW and BK give identical d for all 92 (molecule,
circuit) pairs; only the position where d saturates moves (BK: later, because the term order differs).

**Lemma 2 (symmetry bound).** If every generator conserves N_α, N_β and the irreducible representation of an
abelian point group, every v_j is orthogonal (mod 2) to the α mask, the β mask and the Z₂ character masks, so
d ≤ n − k, k = their GF(2) rank. This is exactly the set of Z₂ symmetries used for qubit tapering. Equality holds
as soon as the excitation x-vectors span the orthogonal complement — true for the Hamiltonian itself, every Trotter
step, UCCSD, and generalized ansätze (which break point-group symmetry: d = n − 2).

| circuit family | measured d | n − d |
|---|---|---|
| Trotter (any dt, JW or BK), UCCSD | n − 2 (no symmetry) … n − 4 (D₂h-like) | 2–4 |
| UCCD | n − k − 1 (no singles: one more parity) | 3–5 |
| k-UpCCGSD (k = 1, 2) | n − 2 | 2 |
| pUCCD (pairs) | n/2 − 1 (− point group) | ≈ n/2 |
| QPE, t counting qubits | t + d(H) | 2–4 |
| Hubbard, t ≠ 0 (U = 0 or 4) | n − 2 | 2 |
| Hubbard, t = 0 | 0 | n |
| Heisenberg J₁–J₂ from the dimer state | n − 2 | 2 |

(Full table: `tables.md` §"Structure map"; the GF(2)-rank table covers n = 24–100, e.g. H₅₀: d(UCCSD) = 97,
d(pUCCD) = 49.)

So the compressed state is, for chemistry, automatic qubit tapering: 2–4 qubits, never the 10–60 that "beyond SV"
needs.

**Angle blindness and saturation.** d, W_d and the saturation point are identical for dt = 0.01, 0.1 and 0.5. d
saturates after 21 of 1,330 rotations (BeH₂, JW, 1.6 % of the first step), 83 of 7,150 (H₁₀, 1.2 %), 29 of 2,352
(N₂). The magic atlas (§6.4) already noted that Trotter step size is invisible to d except at exact Clifford angles;
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

- `selK` (top-K amplitudes): d ≈ K/2 (H₅₀: d = 13 at K = 28; N₂/cc-pVDZ: 11).
- `spanD` (span-closed): K grows fast at fixed D. H₅₀: D = 8 → 22 generators, 12 → 48, 16 → 107, 20 → 248, 24 → 537;
  3×4 stretched sheet: 1,818 at D = 24.
- The register is wasteful for number-conserving states: `nnz/2^d` = 106/256, 1,132/4,096, 14,688/65,536 (H chains,
  D = 8, 12, 16): about 25 % of the register is physical, because d only knows the two parities, not N.

This is the regime the brief asked about (Clifford + few rotations / HF + few excitations). Exact simulation cost
is ~2^d per rotation and the energy evaluation is fast with the span-filtered Hamiltonian (H₅₀, d = 16, 107
generators: 0.2 s per energy; d = 20: <!--T20--> s; d = 24: <!--T24--> s).

## 4. Energies: what these registers can hold

Errors in mEh against FCI (n ≤ 24) or DMRG (larger); `E_reg` is the best any circuit in the same register can do.
Selected rows (all rows: `tables.md` §"Energies"):

<!--ENERGY_TABLE-->

Reading the table:

- **Small molecules (n ≤ 16, equilibrium):** a span-closed UCC with d = n − k is UCCSD and is as good as CCSD
  (LiH +0.01, H₂O +0.10, BeH₂ +0.38 mEh) — but this *is* state-vector size. At d = 8 below the symmetry bound the
  errors are 0.2–27 mEh (H₂O, BeH₂ fine; N₂ 26 mEh; H₈ 48 mEh).
- **Beyond SV size (n = 32–100):** the register optimum at D = 16 recovers 7 % (H₅₀), 17 % (H₃₀), 31 % (H₂₀),
  42 % (4×4 sheet) of the correlation energy; D = 20 adds a few percent. CCSD errors are 8–26 mEh at equilibrium,
  DMRG ~0. The fraction falls with system size at fixed D because correlation energy is extensive and 2^D
  determinants are not.
- **Stretched (strong correlation):** CCSD diverges or fails to converge (3×4: −307 mEh at the last iterate,
  4×4 / 4×5 / H₅₀ at 1.8 Å: not converged). The variational UCC beats CCSD for n ≤ 16 (H₆: +12 vs −126 mEh; N₂: +14 vs
  −102 mEh), where FCI costs milliseconds. Beyond SV size the register optimum is 0.5–4 Ha above the DMRG energy.
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
<!--LARGEST-->

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
   saturation point (first 1–4 % of the first Trotter step) and the dt-independence.
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
python tables.py RESULTS > tables.md                         # RESULTS = refs/ + runs/*.jsonl
cargo test --release --test lowmagic_chem
```
