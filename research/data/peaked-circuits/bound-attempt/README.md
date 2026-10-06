# Rigorous lower bound on |<s|C|0>|^2 for HQAP P11 / P12: attempt and (negative) result

Research record (negative result) for the full-circuit bound; scripts expect the venv and QASM paths of `../reproduce.sh`.
Every number below comes from a script in this directory. Its log is named next to it.

## Target
C = P · M · R. The model C' = P · Pi · R, where Pi is the wire permutation L from `solve_peaked_v2.Core`, and the final cores add
R-gate 59 (P11) and R-gates 60, 64 (P12) to R. Then p_true >= (sqrt(p_model) - eps)^2 with eps >= ||M - e^{iθ}Pi||.
A non-vacuous bound needs eps < sqrt(0.3029) = 0.550 (P11) and eps < sqrt(0.2366) = 0.486 (P12).

## Method: rigorous mirror-zipper (`zipper.py`, `zipper2.py`)
- Cut a segment G = W·V. The identity G = W_rem · (Pi·E) · V_rem is kept exact throughout. Gates are absorbed into the dense
  operator E (≤ 9–11 qubits) from both sides.
- A qubit is "peeled" when E ≈ τ·(A ⊗ u): τ is a transposition (possibly trivial), and A ⊗ u comes from an operator-Schmidt factorisation
  refined by alternating polar steps. The step pays δ = min_θ ||E − e^{iθ}τ(A⊗u)||. This is exact, because F†E is unitary and
  δ = 2 sin(half-width of the smallest eigenphase arc / 2). Then u is pushed into a neighbouring gate exactly and Pi is updated.
- Triangle inequality: ||G − Pi_final·(1q gates)|| ≤ Σδ. The leftover 1q gates are exact and can be folded into the core.
- Validated on synthetic V ▷ SWAPs ▷ σV†σ circuits (`test_zipper.py`, `log_test_zipper_v2.txt`): cost 9e-16 when exact. With 1e-3
  gate noise, cost 0.039 and the permutation is recovered.

## Structure found (all from scripts)
1. Nesting (`p12conn.py`, `pairs.py`, `split_inner.py`): both circuits are [U] ▷ [inner block A] ▷ [U_b†] ▷ [inner block B] ▷ [U_a†].
   - P11: sec1 = U (318 gates); sec3 + sec5 mirror it (147 + 147 connectivity matches).
   - P12: the section splitter merged U into block A. sec1 = 455 "before-hull" gates (U) + a 593-gate hull + 33 after-hull gates.
     sec2 and sec4 mirror U under f_A and f_A·f_B (197/226 and 199/224 matches).
2. Inner hull = union of the causal diamonds of the anchor-paired units (`split_inner.py`, `log_split_inner.txt`). Hull sizes:
   P11 458 / 460, P12 593 / 610. Every hull spans all 98 wires.
3. Mirror centres are SWAP·CZ, not SWAP (`clus3.py`, `log_clus3.txt`). The 6-gate centre cluster on wires {8,10,72,75} factorises
   EXACTLY across {8,10}|{72,75} (operator-Schmidt rank 1). Each factor is in the SWAP·CZ class (Makhlin invariants (0,−1)).
   The third CZ of each transposition pair (w, f(w)) often sits outside the hull, at the block edge (`tcz.py`, `log_tcz.txt`):
   - P11 A: 28 of 49 pairs "open"; P11 B: 32 open.
   - P12 A: 21 open; P12 B: 14 open.
4. Outer U is a nearly-diagonal CZ network (`segstats.py`, `log_segstats.txt`). About 94% of the 1-qubit segments between CZs in the
   outer sections are within ~0.005 (median, spectral norm) of an exactly diagonal or anti-diagonal unitary. NONE is exact.
   Summed deviations per section are 0.9–4.7. Inside the hulls, about 9% of segments are exact (these are the anchors).
5. Only 15 (P11) / 24 (P12) same-pair adjacent unit chains exist in the whole circuit (`chains.py`, `log_chains.txt`).
   Of these, 11 / 9 are near-local, with δ = 1e-4 … 1.3e-2.

## Rigorous results
| piece | P11 | P12 | script / log |
|---|---|---|---|
| inner hull alone → σ_f | does NOT close: saturates at k=10 within ~10 absorptions; every forced peel costs 0.7654 = dist(CZ, nearest product unitary) | same | `run_full.py`, `full_p11_A.log`, `f2_p11_A.log` |
| move boundary transposition-CZ into the hull edge (exact CZ commutation + paid 1q-segment commutation) | 24 of 28 CZs for 0.131 (A) and 30 of 32 for 0.152 (B); outliers cost 0.46–3.1 | 18 of 21 for 0.096, 13 of 14 for 0.076 | `transport.py`, `log_transport.txt` |
| move them all the way to the centre partner (needed for the hull to close) | ≈1.54 each (crosses an H-like segment), ≈45 per block | ≈1.55 each | `transport2.py`, `log_transport2.txt` |
| outer mirror around block A, conditional on hulls = permutations, first 200 / 140 central gates | lossy Σδ = 0.207 (168 peels, median 3e-5, mean 1.2e-3) + 4 × 0.765 residual CZs (the 4 outlier boundary CZs) | Σδ = 0.084 + 4 residuals (0.77, 0.70, 0.63, 0.50) | `run_full2.py`, `f2_p11_O_200.log`, `f2_p12_O_140.log`, `result2_*.json` |
| earlier full outer run (v1 zipper, section-based outer, blocks assumed = perms) | total 13.65: 0.74 lossy + 12.9 from 15 forced peels | – | `outer_p11_k8_t05.log` |

**Rigorous eps: none below the threshold. The bound is vacuous for both circuits.** No complete factorisation of the middle with a
finite total below ~10 was obtained. The best complete accounting (v1, P11) gives 13.65, and that one still *assumes* the inner
hulls equal their permutations, which the hull zippers show is false locally.

Why the bound is vacuous:
- (a) Structural: each open transposition pair leaves an iSWAP-class (SWAP·CZ) residual at the hull centre. Its compensating CZ sits
  ~9 segments away, behind an H-like segment, so no local region closes and commuting it there costs ~1.5.
- (b) Fundamental for operator-norm telescoping: even where the mirror does zip, the lossy residual is about 1.0e-3 (P11) /
  6e-4 (P12) per gate. Over 1715 / 2178 middle gates that extrapolates to eps ≈ 1.77 / 1.30, which is 3–4× the threshold, even with
  perfect structure (`estimate.py`, `log_estimate.txt`). This matches the paper's "deliberately lossy" sweeping. A linear,
  worst-case sum cannot exploit cancellation between independent patch errors.

Tightenings tried (all rigorous):
- Optimal global phase via the eigenphase arc instead of the trace phase, plus refinement: synthetic-test total 0.0433 → 0.0387.
- Peeling SWAP-type (r≠q) factors.
- Larger blocks (kmax 8 → 10–11).
- Anchor-diamond regions (`regions.py`): every innermost region costs ~4. Regions are not closed because of (a).
- Hull/outer split with transported boundary CZs (`run_full2.py`).
- State-dependent bound: rejected. Telescoping from the inside out puts deep states (V_outer R|0>) at the inner pieces, so their
  reduced density matrices aren't computable.

## NON-rigorous estimates (labelled; NOT bounds) — `estimate.py`, `log_estimate.txt`
These extrapolate the certified central-outer peel costs to the whole middle, assuming errors add in quadrature (incoherent):
- P11: op-norm RSS 0.110 → p ≈≳ 0.19. Average-case (normalised Frobenius) 0.077 → p ≈ 0.22.
- P12: op-norm RSS 0.063 → p ≈≳ 0.18. Average-case 0.053 → p ≈ 0.19.

These rest on the outer-mirror error rate. They ignore any extra loss inside the hulls and assume no coherent build-up.

## Most promising next step
Treat the middle symbolically, not with dense local blocks:
1. Snap each near-diagonal or anti-diagonal segment to its exact form and pay the snap distance once.
2. Rewrite the swap-transformation (CZ … H … SWAP·CZ) with an exact CZ-network/phase-polynomial (ZX) reduction, which handles the
   non-local CZ cancellation of point (a).

Step 1 alone already sums to ~1–5 per section (`log_segstats.txt`), so the result would still be vacuous as a worst-case bound. A
useful certificate therefore needs a state-dependent or statistical argument on top: for example, estimate the echo
Re<0|R†Pi†M R|0> by Pauli-path or light-cone Monte Carlo, after the symbolic reduction has removed the exact part.
