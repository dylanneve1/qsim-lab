# P6 rings work (agent p6-rings, 2026-10-06 night)
Answers only in private/. Python: /tmp/pk/research/data/peaked-circuits/.venv/bin/python, jobs via run_capped.sh (2.5 GB, nice).

## Labels
- P6_rings.json order is [A(20), C(22), B(20)]! parse.py reorders to rings=[A,B,C] with B = twin ring of A.
- Twin map (P6_twin.json): pos_B(twin(a)) = (13 - pos_A(a)) mod 20, i.e. B reversed and shifted; C fixed.

## Twin extent (twin_profile in this file's scripts)
- First u3 on a and twin(a): theta and lambda identical for 15/20 pairs (phi often differs; phi is an Rz after Ry so it only
  changes a diagonal phase on |0>-state... it does matter for the state but commutes with following CZ).
- CZ pair co-occurrence (AA pair vs its twin BB pair within +-1 CZ layer): 43/53 in CZ layers 0-12, then 18-29 of ~45 per 13
  layers = control level (control = twin map rotated by 1 or 5 positions: 13-29/45). So pair-level twinning is only in the first ~13 CZ layers.
- Generic (2-3 CZ) block invariant matches to twin pair: ~40% of A/B blocks in block layers 0-59, ~0 afterwards.
- No twinning at the end (reverse ASAP layering: twin == control).

## Ring-aware MPO order (brief item 1)
- relabel.py writes P6 with wires reordered (ring / fold / twinfold / interleave / cab). Gate distance: orig mean 20.6 (9% <=2),
  fold 12.2 (50% <=2); mean cut 1180 -> 696 gates.
- scan_mpou.py (greedy unswap grower, stop at bond 64, 120 s per centre) on P6_fold.qasm: absorbed 47/69/51/49/37/36/56/55/48 units
  at centre fractions 0.1..0.9 (logs/scan_fold.log). Same as default order (23-71). Ring ordering does NOT help the middle-out grower.

## Burst / gap structure (NEW, segment.py)
- CZ ASAP layering shows 17 BURSTS of 2-5 CZ layers with ~20 ring-nearest-neighbour CZs each, separated by 16 GAPS of ~8-9 layers
  with ~14 non-ring (cross-ring or long-range) CZs per layer. Period 13 CZ layers. Epochs: 0,2,..,32 = bursts, 1,3,..,31 = gaps.
  Class counts: burst epochs N 1315, X 66, L 20; gap epochs N 369, X 1237, L 487 (ASAP fuzz at the edges).
- Resynthesised units (Weyl class, weylcls.py): non-ring units are almost all CNOT class (X: 720 cnot, 54 local, 36 generic;
  L: 322 cnot, 19 local, 13 generic). Generic 2q gates sit on ring edges (352). Within gaps the 1q gates are mostly diagonal / X-type / Clifford.
- locscan.py (Heisenberg localisation of X_q,Z_q through 4-layer windows of resyn units): oscillates with period ~11 unit layers;
  highly local (median 1.0) in the middle of every gap, delocalised at bursts.
- gapact.py (exact gate-level Pauli propagation through one gap epoch): gap 1 acts as a single-qubit unitary on ~33/62 wires
  (X and Z images single-site with weight 1.0); the rest are residual ring-edge couplings (ASAP fuzz) plus a few non-ring pairs
  (A19-B13, C11-C5). gapres.py lists non-ring residual pairs per gap: ~5-25 per gap (logs/gapres.txt), incl. second-order ones.
  => gaps are mostly masked identities (CZ gadgets that cancel), the "real" circuit lives on ring edges in the bursts.
- crossw.py (logs/crossw.txt): Heisenberg images per epoch; weight leaving the ring. Bursts contain 1-9 cross-ring CZ-like couplings
  (X_q -> X_q Z_w with weight 1.0), gaps 2-17 "cross weight", mostly the continuation of burst-edge gadgets.
- gapperm.py (Hungarian on image supports): gaps carry NO cross-ring permutation; only partial ring-edge swaps (ring gates fuzzed in).
  So no swap transformations visible at gap level; gaps = near-identity masks + few cross-ring entanglers.
- crossmirror.py: cross-ring CZ pairs shared between gap epochs ~ chance; no mirror centre in the cross-pair pattern.
- Per-wire op strings: cross pairs come in even counts (2+2, 4) inside a gap (gadgets), with 'g' (generic) 1q gates between:
  trained masks, not algebraic cancellations.
- blocksim.py (block MPS, one block per ring, exact ring-internal ops, SVD on inter-ring bonds): chi hits 8x8 within the first
  ~150 ops (gap 1 gadgets) -> transient cross-ring entanglement too large for gate-by-gate block simulation in 2.5 GB.

## Effective circuit (gaps compressed) — effcirc.py, allimgs.py, clthr.py
- Exact gate-level Heisenberg images (eps 1e-7) of X_q, Z_q through every gap epoch (private/imgs_e*.pkl). Clusters = wires coupled
  with image weight > 1e-3; per cluster the unitary is reconstructed from the images (joint +1 eigenvector of the Z images, X images
  generate the other columns, polar projection). Kept weight >= 0.999, reconstruction fidelity >= 0.9995 for every cluster.
- Gap clusters have <= 10 wires (mostly 1-4); cross-ring clusters per gap: 1,2,4,5,3,2,8,5,4,3,5,6,2,2,3,4 (logs/effcirc.txt).
- private/eff_all.pkl: 4933 ops (bursts as raw u3/cz, gaps as cluster unitaries). blocksim2.py (validated exactly vs dense
  statevector on a 10-qubit toy, test_bs2.py) on epochs 0-6: chi(A|BC)=10 after ~3 bursts -> real cross-ring entanglement
  grows ~1-2 ebits per burst; block MPS with A-B-C blocks runs out of memory (A-C ops pass through B).
- ttn.py: star tree tensor network (one leaf per ring = 2^20/2^20/2^22 x a_r isometry, core K(a_A,a_B,a_C)); cross-ring ops via
  operator-Schmidt MPO over rings, Gram trick (expanded leaf never materialised), legs truncated by core SVD. Exact test vs dense
  statevector on a 10-qubit toy: overlap 1-1e-14 (test_ttn.py). run_ttn.py drives it on private/eff_all.pkl.
  First epochs: K bonds (2,2,1) -> (8,4,2) at epoch 3 -> (3,3,2) at epoch 4: cross-ring entanglement is transient and SHRINKS again.

## KEY: re-segmentation epochs3 (segment.py)
- epochs3(): non-ring CZs at the edge of a burst (last CZ on both wires in the burst, or first on both) are moved into the
  adjacent gap (86 moved, causal order intact). After this, bursts contain ONLY ring-edge CZs (1315), all 1810 cross/long-range
  CZs are in gaps.
- Recomputed gap images (private/imgs3_e*.pkl, SEG=3). Gaps 1,3,5: NO cross-ring cluster at all (effcirc SEG=3) ->
  cross-ring CZs are pure masking gadgets that cancel; the rings decouple.
- SEG=3 all gaps (logs/effcirc3.txt): only 2 cross-ring clusters in the whole circuit (gap 7: 9 B wires + C19; gap 9: B2-C19;
  B2 and C19 are idle in burst epoch 8, so this is a gadget straddling an empty burst). private/eff3_all.pkl = 4752 ops.
- TTN on eff3_all (tol 1e-8, maxa 64, logs/ttn3.log): core bonds (1,1,1) through epoch 7, then (1,2,2) to the end; zero truncation.
  ttn_peak.py e3: ring marginal maxima A 0.0831, B 0.2304, C 0.5626; P(candidate)=0.010772, max single-flip P=2.25e-5 (ratio 478).
  Candidate in private/peak_e3.txt. Twin check: 11/20 bits of B agree with twin(A) -> no twin symmetry in the peak.
- epochs4 (segment.py): additionally moves 335 ring CZs from gap edges into bursts (gaps keep 34 N CZs). SEG=4 images
  (private/imgs4_*) -> gaps are almost purely LOCAL (clusters of size 1-2; cross only gap 7 {B1,B2,B3,B4,C19} and gap 9 {B2,C19}).
  private/eff4_all.pkl 5985 ops. TTN (logs/ttn4.log): bonds (1,2,2), zero truncation. ttn_peak.py e4: SAME bitstring as e3
  (cmp identical), P=0.010772, marginal maxima 0.08309/0.23044/0.56261, max single flip 2.25e-5.
- THR 1e-2 instead of 1e-3 gives an identical cluster partition (couplings are either strong or < 1e-3).
- Per-ring marginal top-4: A [0.0831 3.1e-4 2.9e-4 ...], B [0.2304 1.1e-3 8e-4], C [0.5626 1.2e-3 1.1e-3]: unique peaks.
- gapcert.py: EXACT TN contraction of Tr(W_eff^dag W_raw)/2^62 per gap (raw gates vs reconstructed clusters, SEG=4):
  gap1 0.99965, gap3 0.99987, gap5 0.99993, gap7 0.99996, gap9 0.99997 (rest running, logs/gapcert4.log).
- gapcert (all 17 gap epochs incl. final 1q layer): F = 0.99965, 0.99987, 0.99993, 0.99996, 0.99997, 0.99999 ... product 0.99931.
  The effective circuit (raw bursts + reconstructed gaps, exact causal composition) reproduces the raw unitary to trace
  fidelity ~0.9993 -> the TTN peak is the peak of the raw circuit.

## RESULT
- P6 = ring circuits A (20q), B (20q), C (22q) with masked-identity gaps; only one B-C coupling survives (bond 2).
- Peak (private/peak_e3.txt == private/peak_e4.txt): P = 0.01077 (product of ring peaks 0.0831 x 0.2304 x 0.5626),
  max single-flip P = 2.25e-5, unique per ring by a factor ~200-450 over the runner-up.
- ringsim.py (plain per-ring statevector, independent of ttn.py) on eff4 ring A: argmax P=0.08309, same ring-A bits as TTN.
- tn_amp.py: raw single-amplitude TN contraction width 135 (log10 flops 43.7) -> infeasible; certification relies on gapcert.
