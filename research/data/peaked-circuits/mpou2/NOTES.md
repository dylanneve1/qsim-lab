# mpou-v2: own MPO + unswapping peaked-circuit solver (2026-10-06/07)

Code in /tmp/mpou-v2 (Mac copy /tmp/peaked-mac/mpou2). Portal answer strings only in private/.
Python: VPS /tmp/pk/research/data/peaked-circuits/.venv/bin/python (runcap.sh: 2.5 GB cap, nice); Mac ~/peaked-venv/bin/python (launch.sh).

## Components (all ours; borrowed ideas cited)
- mpou2.py: 1D MPO W = C_right . C_left, site tensor (Dl, up, dn, Dr), mixed canonical. Each site holds the output leg of
  wire su[s] and the input leg of wire sd[s]. Exact relabelling moves: 'both' (exchange two sites = layout change),
  'up' / 'dn' (exchange one leg kind = pairing change). Midpoint MPO + unswapping idea: Kremer-Dupuis arXiv 2604.21908.
  Every two-site update is done on a QR-reduced core (each site QR'd over the legs it keeps; SVD of Ra.G.Rb only;
  same singular values as the full two-site matrix; idea from our /tmp/mlx-port/tno_fast.py). complex128.
  Truncation 'rel' (s > eps s_max, as quimb) or 'sum2' (discarded weight <= eps). Tests: test_mpou2.py (operator error 2e-14,
  marginals 3e-15, relabelled mirror with swap network collapses to bond 1 with greedy or matching unswap).
- Routing: 'both' exchanges with a meeting point chosen by a carried-entanglement proxy (default); options: lookahead,
  Sabre-style swap selection (idea: Li-Ding-Xie 2019 / Qiskit SabreSwap), 'side' routing (KD-like, one leg kind).
- Unswap: greedy per bond (probe up/dn/both on the reduced core with svdvals; accept lower rank, ties broken by purity);
  hot bonds (>= hot*maxbond) after growth, full sweeps over all bonds when elements > tau.
- match2.py: matching unswap, score S[w,v] from Tr(W P_w W^+ Q_v) (stat 'sum' as validated in /tmp/unswap-match/proto2.py;
  new stat 'smax' = largest singular value^2 of the 3x3 Pauli transfer block, blind to local unitaries and CZ-like residues),
  Hungarian assignment, bubble sort of output legs. Skipped when the median score < 0.6 (on P9 the 'sum' median is 1/3
  because residues are CZ-like, and forcing the assignment blew W up).
- centre.py: dense exact centre block: band blocks split into small connected groups, each group's unitary contracted
  densely, pairing by dense Pauli score + Hungarian, best site order (exhaustive for m <= 6), successive SVD.
- tno_centre.py: centre block via our swap-aware arbitrary-geometry TNO (tno.py from /tmp/peaked-generic + tno_fast.py),
  layer pairs by distance from m, canonical compression + unswap passes, then TNO -> MPO2: DFS order of the bond graph,
  crossing bonds carried through sites as identities, exact, then SVD compression sweep.
- endgame.py: exact single-qubit marginals of psi = Pg . W . Rg |0> by quimb/cotengra contraction (Pg pruned to the
  backward light cone of the measured qubit), amplitudes for peak and flips. Used when the shallow outer layers would blow
  up a 1D operator (P5).
- run2.py driver: blocks = consecutive same-pair CZ units merged (P5 902, P9 1885, P6 2593); orders 'asap' (ASAP layer
  of the first unit, centre m), 'file', 'cutlayers'; phase 1 = centre band (dense / TNO / tight growth), phase 2 =
  outward growth. Scheduling 'dist' (layer distance), 'foot', and 'pair' (mirror-partner first: after absorbing a block on
  one side, register the wire pair that sits on the same two sites on the other leg kind; an available frontier block on
  the other side with that wire pair is absorbed next). W -> W|0> as soon as the input side is complete; optional
  state_left (outer input blocks applied to |0> as an MPS). Checkpoints (pickle) every ckpt_every s, RSS guard, max_time.

## P5 (validation, 44q, 1892 CZ)
- Dense centre band CZ layers [36,48) = 11 four-qubit groups; pairing scores 0.96-0.99; middle-cut non-product weight
  0.1-1% per group (6% total) -> centre truncated to bond 1 (eps_centre 1e-2 sum2).
- Growth with rel 1e-3: bond grows 40 -> 130 -> 256 over layer pairs 32/50 .. 28/54 (the residue is patch-wise, not
  gate-wise: after layers 34/48 W = 22 disjoint bond-3/4 pairs). With rel 1e-2: W stays at bond <= 4 through all of U/V
  (elements ~200), exactly like the earlier quimb TNO solver at cutoff 1e-2.
- Absorbing R (input layers 0,2) or P (output layers >= 80) into the 1D operator blows up (bond 157-256); P.W.R as a
  network is shallow: endgame (end_left 4, end_right 80) gives the marginals exactly.
- Result (logs/p5j.log, logs/p5j.json, VPS, 22 s incl. 10 s endgame): blind peak = graded answer (private/p5_result.json),
  p = 0.0109, max single flip 0.00123 (0 higher), min|Z| 0.362, truncation 0.025.
- rel 3e-3 (dist or pair scheduling): bond grows to ~50 by step 120 (slow on the loaded VPS); not needed.

## P9 (56q, 1917 rzz)
- File order is not layer order (95-gate windows span 15-20 ASAP layers); no sequence mirror in file order.
- Repeated fixed 1q segments: 68 units with both segments repeated (+29 with one), ASAP layers 46-53 = the swap network;
  it connects 54 wires (pair multiplicity 1-2), almost convex (2 violators). Grown alone in file order it reaches bond 16
  after 30 units (partial network = flat-spectrum entangler).
- Pure MPO growth (asap m=49/49.5, file, cutlayers; both/side/Sabre/lookahead routing; full unswap above tau) blows up at
  60-200 blocks with exact flat spectra (bonds 2^k, truncation ~1e-6 even at rel 1e-2): smax diagnostics show the pairing
  is already right (50-56/56); the residue is CZ-like (uncancelled gates).
- TNO centre band [45,55) (m=49): TNO 736-824 elements, 52 wires relabelled; -> MPO 932 elements, bond 6.
  With 'dist' scheduling phase 2 still blows up at ~120 blocks (layers 41/57; again pairing right, CZ-like residues).
- With 'pair' scheduling (mirror partner first) the whole circuit is absorbed with bond <= 8 and the final W|0> has bond 1:
  3 s per run on the Mac. Same peak at rel 1e-2, 3e-3, 1e-3, 3e-4 (private/p9_pair_e*.json on the Mac, private/p9_candidate.json):
  p = 0.0999-0.1000 (design ~10%), all 56 single flips lower (max 0.0042), min|Z| 0.919. 811 pair-scheduled blocks.
- Without the TNO centre band, 'pair' scheduling also blows up (step ~80, 390k elements): the exact centre block is essential.

- Robustness (Mac, private/p9_var_*.json): same peak with m=49.5, band [44,56), band [46,54), TNO cutoff 1e-4.
- The peak also equals that of the separate KD-style runs on the Mac (parent's confirmation; a file
  shepherd-reported.txt written into this directory by another agent was moved to private/ unread until after our result).

## P6 (62q, 3494 CZ) - no own peak
- MPO middle-out does not apply: 'pair' centre scan (ASAP m = 20..190, 40 s budget each) finds local optima at m ~ 30, 70,
  108-110, 160 (~65% of absorbed blocks paired) but W blows up after +-7 layers even with a TNO centre band (logs scan6*).
- Structure (own analysis, p6layers.py): bursts of ring-internal CZ layers (0 cross units) every ~13 ASAP layers, gaps with
  ~10 cross-ring units per layer; three rings (20/20/22) recovered from the most frequent pairs (ringsim.py).
- ringsim.py (ours): exact per-ring state blocks (2^20/2^20/2^22) in a 3-block chain with small Schmidt ranks; ring gates
  applied exactly; cross-ring gates via operator-Schmidt terms + Gram recompression; MODE=tno compresses each gap first as an
  operator with the swap-aware TNO and applies its bond-graph clusters (dense up to 12 wires, single-ring by contraction,
  multi-ring split along the cluster's own cross bonds, wires moved across rings carried as dim-2 cross bonds); SEGW=1 =
  wire-aware gap segmentation (latest burst behind / earliest burst ahead per unit in the DAG). Exact on 11-qubit 3-ring toys
  with and without cross-ring SWAPs (ring11*.qasm).
- P6 result: gate-by-gate ring blocks blow up (Schmidt ranks 16x16 within the first 30 cross gates, even at tol 1e-2).
  TNO-gap mode gets through the first 3-4 gaps with ranks <= 4 but then meets multi-ring clusters (14-22 wires, up to
  3 rings) whose cross bonds multiply to 2^8-2^9 (contraction width 31 = 2^31 elements) at every TNO cutoff tried
  (1e-8 .. 5e-2) and every segmentation tried (layer slabs with XB 1/3/5, burst-edge moves, slab extension, wire-aware).
  Runs were killed by memory (width 31) on the Mac. No independent P6 peak; p6-rings' candidate was not read.
