# Generic peaked-circuit solver — running notes (2026-10-06)

Workspace: /tmp/peaked-generic. Python: /tmp/pk/research/data/peaked-circuits/.venv/bin/python (quimb 1.11.2, cotengra 0.8.2).
Originals /tmp/peaked-amp/solve_peaked{,_v2}.py copied here untouched (solve_peaked.py, solve_peaked_v2.py).

## 1. Generic front end (gparse.py) — DONE
- QASM2/3 tokenizer (qreg / qubit[n] declarations, multiple registers), 1q gates u,u3,U,u2,u1,p,rz,rx,ry,x,y,z,h,s,sdg,t,tdg,sx,sxdg,id;
  2q gates cz, cx/cnot, cy, swap (3 cx), rzz/cp/crz (any diagonal gate whose interaction phase is pi is CZ-class; others rejected).
- Unit = (a, b, Pa, Pb): full 1q segment on each wire since its previous 2q gate, then CZ. Trailing 1q gates kept in `tail[q]`.
  Diagonal gates: local part split half before / half after the CZ (so a gate and its inverse give mirror segments).
- test_gparse.py: 12 synthetic 6-qubit circuits (every gate type, QASM2+3) and 3 random <=11-qubit sub-circuits of each
  real file: state fidelity |<ref|parsed>| = 1 to 1e-14. Non-CZ-class rzz(0.3) is rejected. Log: logs_test_gparse.log.
  All 5 circuits (+ q2 copies) parse: 1917 / 4020 / 5072 / 1999 / 2457 CZ units.

## 2. Structure stage on all five circuits (struct_probe.py, approx_probe.py, layer_probe.py)
- Exact-inverse anchors (unique SU(2) segment fingerprint whose inverse is also unique), with and without
  quotienting by S-powers on both sides (to allow rzz(+-pi/2) compiled differently):
  P11 419, P12 569 (same as before, centres in two clusters) ; **P9 0, heavy_hex_4020 0, heavy_hex_5072 0**.
- Nearest-inverse distance 1-max|<q_i, q_j^-1>| for P9/heavy-hex is distributed like random points on S^3
  (bulk 1e-3..1e-2) -> no approximate segment mirror either. P11 has 930 segments at <1e-12.
  => In P9/heavy-hex every segment was re-trained (sweeping), so ANY segment-fingerprint method is dead there.
  This is consistent with the paper (2510.25838 sec 3.1/5, Fig 10: angle sweeping hides angle correlations).
- So sections / anchors / maps all break on those three for the same root cause: zero anchors. (sections()
  itself returns something for any circuit but block_maps finds no block with >=50 anchors -> v2 raises.)
- heavy_hex: 54 distinct pairs (hardware edges), layered brickwork (17/18/19 period-3 layer sizes); fixed-wire
  pair-sequence mirror score peaks at layer 109.5 (4020: 1343/1908 shared edges) and ~141 (5072) but brickwork
  periodicity makes this weak evidence.  P9: all-to-all, 1069 distinct pairs; layer mirror score weak (wires relabelled).
- Kremer-Dupuis (2604.21908) solved P9 by contracting both halves into a middle MPO + greedy "unswapping"
  (single datacenter GPU, 4059 s). Their key point: cancellation is only visible at the operator level, not per gate.
- pitfall: pgrep/pkill -f with the pattern in my own command line kills my shell; use pid files.

## 3a. Operator-level mirror detection (the anchor-free route)
- mirror.py + pauli.py: Heisenberg images of the 2n generators X_q, Z_q under a window W grown around a cut,
  as sparse Pauli sums. Discriminates mirrors (heavy_hex_4020: layer-synchronous window [79,141) keeps mean
  image weight ~1.2 = single-qubit, no relabelling; control cut at layer 60 absorbs only 48 gates), BUT the
  lossy sweeps leave a noise floor of small Pauli terms whose count doubles per layer -> blows up at ~[76,144)
  with eps=1e-2, worse with eps=5e-2 (truncation errors break later cancellation). Pauli truncation is the wrong
  compression for "permutation + small lossy residual". (logs_layergrow_hh4020*.log)
- tno.py: own TNO with one in/one out leg per tensor and swap-aware SVD split (exact on random 4q circuits;
  collapses a mirror with hidden SWAP network to bond 1 and recovers the permutation: test_tno.py). On heavy-hex
  its local SVD without canonical gauge lets bonds grow on the loopy lattice (max bond 36 at [102,118)).
- tnoq.py: same middle-out growth but with quimb tensor_network_ag_compress (canonised local compression, the
  building block also used by Kremer-Dupuis' TNO notebooks). heavy_hex_4020 from layer 110: the WHOLE circuit
  [0,222) compresses to a 49-tensor operator with max bond 2-4 until the last ~4 layers (bond 8 at the end),
  8.8 s, 257 MB. So on heavy-hex the "middle" is everything except ~2 layers at each end.

## 4a. heavy-hex results (solve_generic.py = gparse + centre scan + tnoq middle-out + exact final contraction)
- Centre scan (coarse grid D/24, then +-step/2; score = gates absorbed before W exceeds 2e4 elements; ties -> smaller
  max W size): 4020 -> coarse 108:4020 vs 176..186 elsewhere; fine picks 112 (106..112 all absorb everything).
  5072 -> coarse 143:5072 vs ~180; fine picks 140.  Very sharp, no thresholds.
- 4020: whole circuit absorbed, W max bond 8, final network width 13.6, log2 flops 20.6; peak p = 0.6949
  (0.6990 from centre 110), same peak both centres. 91.8 s wall, 212 MB peak RSS. The circuit is advertised as
  "70% peaked" (tracker issue #27 title) -> consistent. No public reference bitstring found.
- 5072: whole circuit absorbed, W max bond 8, width 12.3, log2 flops 19.2; p = 0.6973; 91.4 s, 207 MB.
  POST HOC: identical to the reference (tracker #105 / KD notebook / top count of hardware run #93) when the
  reference is read in Qiskit order (qubit 0 rightmost); Hamming 0/49.
- Wall time is dominated by the centre scan (67-74 s); growth 4-10 s; the 49 marginals ~10 s.

## 4b. P9 attempts (operator route) — not solved so far
- Fixed-site arbitrary-geometry TNO (tnoq) from the scanned centre 51: window only [39,63), 422/1917 gates, 1.3M elems.
  Centre scan signal weak (max 298 gates absorbed at c=51 vs ~200 elsewhere, budget 2e4).
- Swap-aware TNO (tnos = tno.py split with both leg assignments + quimb canonical compression): exact on tests,
  reproduces heavy-hex 4020 (moved 0, 3.4 s). P9 from c=50/51: bond graph densifies (nbonds 47 -> 177 by [41,59)),
  elements x2.7 per layer pair (75840 at [41,59), 25.7 s and rising) -> not converging within the RAM budget.
- Gate-level greedy (absorb the frontier gate whose local split has minimal rank) is WORSE than layer-synchronous
  even on heavy-hex (stalls at [79,148)): non-canonical local ranks mislead the greedy.
- Pauli-image probe confirms a relabelling mirror centred near layer 50 (c=50: 230 gates absorbed, 29 wires moved;
  other cuts 39-93 gates). Kremer-Dupuis report the same: the first ~300 unitaries around the midpoint are the hard
  phase (native permutation concentrated near the centre); they needed MPO + greedy unswapping on a datacenter GPU (4059 s);
  a CPU port took 734 s on a laptop CPU (tracker #153).
- P11 with tnos from block-A centre (layer 30): RSS reached 3.1 GB after 19 CPU-min -> killed (over budget).

## 3b. Anchor route v3 (solve_anchor.py) — heuristics replaced, P11/P12 reproduced
- blocks: anchors sorted by mirror centre (ASAP-layer midpoint); a new block starts exactly when an anchor
  contradicts the current involution (blocks.py). P11 -> 2 blocks (211 @ layer 29.5, 208 @ 65.0), P12 -> 2 blocks
  (291 @ 50.5, 278 @ 104.0); every block covers all 98 wires, is a fixed-point-free involution, and equals the old
  v2 maps exactly (dbg_p11.py: L, both maps, R0, P0 identical). Replaces ">=50 anchors", "two blocks", centre-section grouping.
- sections: serial greedy layers; boundaries = size jumps whose log ratio is in the upper Otsu class (sections2.py).
  P11 identical to the 9/3 rule; P12 gets one extra split inside section 1 (60..490 | 491..496 | 497..1140),
  R and P identical. Only R = first and P = last section are used; the frontier is the plain DAG frontier.
- single-qubit attachment: gparse.parse_nearest attaches each 1q gate to the nearest 2q gate in FILE order
  (ties -> next). For P11/P12 this reproduces the strict 5-line units bit-for-bit (dbg_units.py). The first
  attempt (fold all 1q gates into the next unit) gave core p = 0 on P11: the merged boundary segment
  post_R*pre_U lost post_R into the dropped middle. Pre/post attachment matters at the R/middle and middle/P seams.
- P11: initial p 0.1279 -> +R59 -> 0.3029, stop. 278.7 s, 284 MB. P12: 0.0052 -> +R60 0.0542 -> +R64 0.2366, stop.
  335.7 s, 305 MB. Identical to the earlier max-p runs (notes file), peaks identical to the P11/P12 references (post hoc).
- adaptive side selection (tnos.grow_adaptive) on P9 from c=50: same blow-up (76k elems at [44,62), 180 bonds, 86 s/step, RSS 1.08 GB) -> killed.
- P9 centre structure (p9_exact.py, p9_centre.py): 593 non-trivial EXACT inverse segment pairs exist but all are
  repeated fixed 1q segments (e.g. quaternion (0,0,.7419,.6705) x21, (0,0,.7071,.7071) x12, identity x13), all in
  ASAP layers 46-55: an explicit swap-type network at the mirror centre (pairs reused 3x, interleaved, e.g.
  (5,13) at layers 46/48/50 with (5,19) at 47/49). Non-unique -> no anchors. As a band, [46,55) is NOT a clean
  permutation (Pauli images mean weight 1.9-2.6, eps 1e-2; exact images blow up), because ordinary U/U^dag gates are
  interleaved with the network. This is the "native permutation concentrated near the centre" that makes the
  first ~300 absorbed unitaries hard for Kremer-Dupuis too.
- Verdict: P9 NOT solved here within 2.5 GB / CPU. Needs a proper MPO+unswapping (or a network-aware unswap of the
  central swap band) - not built in this session.
- tno.unswap_pass (arbitrary-geometry unswapping: exchange output/input legs of a bonded pair if that lowers the SVD
  rank; exact on tests, turns two interleaved 3-CZ SWAPs from bond 4 into bond 1 with the right map) DOES crack the
  central band: tnos.grow(..., unswap=True) from c=50 reaches [45,55) with W = 696 elements, 32 bonds, 52 of 56 wires
  relabelled (logs_p9_unswap.log). Outside the band W grows again (x1.5-3 per layer: 28k elements at [41,59)) and the
  unswap passes become the cost (541 s for that step, 1.08 GB) -> killed.
- With adaptive side choice + one unswap pass on bond>=4 pairs (logs_p9_unswap_ad.log): band passed at [45,55)
  (~900 elems), then the before side grows x1.3-1.6 per layer: 53k elements at [37,57) after 406 s, RSS ~1 GB -> stopped.
  So P9 remains unsolved here; the remaining obstacle is outside the band (no further cancellation with this
  alignment), not the band itself.

## 5. Scaling (task 5)
- P11/P12 final cores (scaling.py): amplitude width 12.0 / 8.0, log2 flops 17.9 / 15.4; worst marginal (light cone
  83 / 78 gates) width 11, log2 flops 17.0 / 17.1. Runtime 279 / 336 s is evaluation count x per-call overhead, not flops.
- heavy-hex operator route: growth ~0.025 s per layer pair after warm-up (logs_tnoq_hh4020.log: 5.9 s -> 8.8 s over
  110 steps); scan 67-74 s; final width 12.3-15.2, log2 flops 19.2-22.3; peak RSS ~210 MB.
- Parent's runs of solve_generic.py (results/parent_p4.json, parent_p7.json): P7 45q/1275 width 16.0, log2 flops 22.6,
  40 s; P4 48q ring/5096 W bond 14, width 12.0, log2 flops 20.0, 96 s, norm 0.867.
- P8: exact final width 32.5 / log2 flops 41.6 -> approximate (bond-capped) candidate + uncapped amplitude check.
- P9: exponential regrowth of W outside the central swap band (x1.3-1.6 / layer).
- Mixed-cutoff attempt (run_p9.py 50 1e-3 1e-2 16 45 55; logs_p9_mixed.log): cutoff 1e-3 through the band, then 1e-2
  with bond cap 16 + unswap + adaptive sides. Identical trajectory to 1e-3: max bond stays 4 (the cap never binds),
  but the NUMBER of bonds grows (53 -> 119 over [44,56) -> [38,57)), elements x1.3-1.8 per layer (31k at [38,57),
  275 s, RSS 934 MB), after side never chosen past layer 57. The growth is connectivity (many independent bond-4
  pairs), not bond size, so a looser cutoff/cap does not help; the region outside the band does not cancel in this
  layer alignment. Stopped; P9 unsolved by this pipeline.
