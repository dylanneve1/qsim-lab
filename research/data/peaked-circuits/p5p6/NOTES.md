# P5 / P6 notes (2026-10-06)

Code: /tmp/peaked-p5p6 (copies/imports of /tmp/peaked-generic, which is not modified). Answers only in private/.
Python: /tmp/pk/research/data/peaked-circuits/.venv/bin/python, capped with run_capped.sh (2.5 GB, nice).

## P5 construction (cstruct.py, blocks2.py, mirror_pairs.py, cycles.py, invmatch.py, perm2.py, middle.py, netcheck.py)
- 44 qubits, 1892 CZ. Every 2q gate is a 2-CZ block CZ.(u3(t,+-pi,0) x u3(t',0,-pi)).CZ, so the Weyl coordinates are (c1,c2,0).
  After merging consecutive same-pair CZs: 902 blocks (870 x 2CZ, 22 x 4, 8 x 6, 2 x 8). Block-ASAP depth 43.
  Every layer is a perfect matching (22 blocks) except layers 18-26 (14-19 blocks).
- The cycle structure of consecutive matching pairs is mirror symmetric around layer 20.5 (layer l <-> 41-l).
- Makhlin invariants (G1, G2) of blocks: ~38% of blocks have a partner within 1e-4 (median 3e-5), all at layer sum 40-41.
  The wire map from partner pairs is ONE global permutation pi (bijective, consistent over layers 2-17); pi consists of 4-cycles and 2-cycles.
  Forward check: blocks of layers 2..17 at (a,b) have a partner at (pi a, pi b) in layer 41-l (+-1) for 342 of ~350 blocks.
- Unmatched blocks: layers 0,1 (44 = R), 40,41,42 (66 = P), and ~94 blocks in layers 16-23 = the swap network.
  The swap network lives on 4-qubit groups (union of matchings of layers 18/19 = eleven 4-cycles). netcheck.py: the dense operator
  of the unmatched blocks of each group equals P_pi on that group up to ~1e-2 operator entanglement (8 of 10 groups;
  the 8-qubit group and {16,19,24,30} do not fit, probably because a few blocks are mis-assigned or interleave with matched ones).
  The network gates are perturbed CNOT/iSWAP/identity/SWAP class gates.
- So P5 = R (2 layers) |> U (layers 2-17) |> N (swap network = pi, layers 18-23) |> pi U^dag pi^-1 (layers ~21-39) |> P (3 layers),
  i.e. the T[R] |> T[U] |> U^dag |> P construction with a central swap network.
- The generic middle-out (solve_generic) fails because in-leg a and out-leg pi(a) live on different sites.
  Relabelling the second half by pi^-1 with explicit SWAPs at the cut (relabel.py, tested exactly on 6 qubits) did NOT fix the
  layer-synchronous growth (462 gates, bond 56). Pair-driven growth (pairgrow.py) also blows up (bond 54, 2.2M elements after 498 units).
  So the intermediate operators are not close to a permutation (consistent with re-trained/swept gates); only global cancellation.
- Route taken: core = R |> seam 1q |> pi |> P (core.py), the peak of the shallow trained part.
- Correction: the netcheck mismatches came from spurious matches between special-class (CNOT/iSWAP-class) network gates;
  with them excluded (match2) all 9 groups of the unmatched network equal P_pi up to ~1e-2 operator entanglement.
  Heisenberg images (heis.py, window [18,23], exact Pauli propagation) of Z/X on network-only output wires are 96-99% on the single qubit pi^-1(q).
- Core R(layers 0-1) |> pi |> P(layers 40-42), with or without the seam 1q gates: p(peak) ~1e-9 -> NOT peaked. The middle is not Pi.
- segcheck.py: the 1q segments between consecutive matched U gates are NOT the inverses (nor T, conj) of the V-side segments
  (median distance 1.5). Paper (2510.25838 sec 3.1): "angle sweeping" = patches re-trained to mimic the original patch (lossy);
  so cancellation only holds patch-wise, not gate-wise. That is why middle-out growth gets bond ~50 (defects of half-absorbed patches).
- KD port (Kremer-Dupuis midpoint MPO + greedy unswapping, Apache-2.0; numpy port taken from another agent's ~/peaked-p9 on the Mac)
  copied to kd/ with a generic driver run_kd.py (QASM arg, measure_all if no clbits, RSS guard from env RSS_GB).
  qiskit 2.5.2 / numba / qiskit-quimb / pyzx installed into /tmp/peaked-p5p6/pylib (PYTHONPATH), venv numpy 2.2.6 kept.
  Running on P5 on the VPS: private/kd_p5_a.log (cutoff 1e-3, seed 123, centre 0.5).

## P6 construction (so far)
- 62 qubits, 3494 CZ -> 2593 blocks (1865 single-CZ, 555 two-CZ, 173 three-CZ); depth 182 block layers.
- 62 "heavy" pairs (13-25 occurrences each) form three disjoint rings: A (20q), B (20q), C (22q) (P6_rings.json).
  1190 blocks are ring-nearest-neighbour; 998 are cross-ring; 405 are long-range within a ring.
- Rings A and B are twins: the map A->B (P6_twin.json; B = A reversed) maps the first gates/u3s exactly (same theta, lambda);
  449/537 AA blocks have a twin pair present in BB. Twinning does not hold gate-for-gate later in the circuit.
- No mirror at the 2q-invariant level (generic blocks have exact partners only at layer distance 0-3 = twins).
- ~1/3 of the u3 are within 1e-4 of theta in {0, pi/2, pi}: long-range gates sit in CZ-H-CZ (CNOT/SWAP-like) Clifford
  patterns = swap transformations / masking. czsimp.py (exact Clifford-monomial CZ cancellation, tested exactly on random
  5-qubit circuits) removes CZ pairs: 3494 -> 3080 (tol 1e-6) / 2672 (1e-4) / 2456 (1e-2) CZ. Long-range blocks 1403 -> 689.
- pyzx full_reduce (|0> plugged, Clifford snapping 1e-4): 10092 vertices, 13026 edges, 9840 non-Clifford phases; not obviously useful.

## What the public CPU MPO+unswap solver (alexgalda-m/peaked-mpo-solver, built on Kremer-Dupuis) does differently (read for ideas only; not run on portal problems)
- Geometry: a 1D MPO over physical sites. Both halves of the circuit (split at the gate-count midpoint, center_ratio 0.5) are routed to
  linear connectivity with Qiskit Sabre (90 trials for the first route, 50 for reroutes), so long-range gates become SWAP chains.
  Ours: an arbitrary-geometry TNO, one site per wire, with no routing; permutations become long bonds.
- Growth: absorb one routed layer at a time from the left OR the right, choosing whichever gives the smaller tensor footprint
  (ours: layer-synchronous from both sides).
- Unswapping (the key trick): when total elements exceed the unswap threshold (5e5) or max_bond is hit, probe even/odd layers of adjacent SWAPs
  applied to the MPO's input legs, output legs, or both ("both, left, right"), compressed. Keep every swap whose local bond
  dimension decreases, update the left/right permutations, iterate (max_its 20), then re-route the remaining layers with Sabre
  around the new permutation. This discovers the hidden wire permutation (swap transformations) on the fly.
- Compression: SVD cutoff 6e-4 (KD used 2e-3), max_bond 512 (KD 8192), complex128 on CPU (complex64 is an option on GPU).
  Probe SWAPs are decomposed to CX (raw SWAP probes misalign logical legs vs sites).
- Engineering: cached SWAP-layer MPOs, reuse of the full parity probe, less copying; a fail-fast stall guard
  (entanglement_blowup vs swap_thrash).
- End: MPO -> MPS by applying the remaining outer layers to |0>, then 1000 samples, peak = most frequent sample.

## Own MPO + unswapping grower (mpou/) — ours; idea of leg-unswapping from Kremer-Dupuis / Galda
- mpou.py: 1D MPO (mixed canonical, site tensor (Dl,up,dn,Dr)); routing = SWAPs on up/dn legs = pure relabelling
  (pu/pd/su/sd maps), so the represented operator stays exact; greedy local unswap per bond (try SWAP on up/dn/both,
  keep the smallest truncated bond, svdvals dry run, optional hot-bond restriction). Layer-balanced middle-out growth.
  test_mpou.py: 7-qubit random circuits, operator error 2e-14, marginal error 5e-15; relabelled mirror stays at bond 1.
- match.py: global matching unswap (statistic S[w,v] = 1/3 sum_{P,Q} |Tr(W P_w W^+ Q_v)|^2 / Tr(WW^+)^2, proposed and
  validated densely by the parent agent): environments contracted as (D,D') matrices (O(D^3) per step, 3n^2 steps),
  Hungarian assignment, then upper legs bubble-sorted to the assignment. test_match.py (n=8, 16 random gates, 17-swap
  network, noisy inverse): permutation recovered exactly at 1q noise eps=0, 0.05, 0.15 (median score 1.0 / 0.90 / 0.42);
  noiseless case collapses to bond 1. In run_mpou.py: triggered when max bond >= MATCH_BOND, reverted if it does not help.
- P6 centre scan with the greedy grower (stop at bond 64): 23-71 units absorbed at every centre fraction 0.1-0.9 (no preferred centre).
- Heisenberg windows (uheis.py) around the parent's nested twin pairs (units 316-331, 317-329, 319-328 of the resynthesised
  P6 units): images stay inside ring A or inside ring B, so there is no A->B transfer there; the twin pairs look like copies (same gate on A and
  later on B), not inverse patches. (Same invariants cannot distinguish G from G^dag for these gates.)
- Runs of the own grower on the Mac (load 8-70 from other agents' jobs):
  * P5 validation (P5 already solved elsewhere): greedy unswap + layer-balanced from CZ layer 42: absorbed 270/1892 units
    with bond 64-256 and truncation 1e-5, then stalled (bond 256-512, sweeps too slow). With matching unswap: at ~260 units
    W is almost exactly Pi.local (assignment scores median 0.98, bond 16), but it blows up again at ~270-280 units. The cause:
    in CZ-unit ASAP layering the central swap network (block layers 18-23 = CZ layers 36-48) is interleaved with U's
    last layer (CZ layer 35) and V's first layers (CZ layers 47-48). Growth then absorbs U gates before the network is closed.
    Routing with SWAPs on both legs (ROUTE=both) made the first 270 units 16x faster (10 s vs 162 s).
    Footprint side choice (SCHED=foot) and a centre at CZ layer 36 did not get past this point before the runs were stopped.
  * P6 (centre 0.5): 50/3494 units, bond 256, match scores median 0.33 (no dominant permutation). Stalled.
- Verdict: the tool is correct (exact tests) and finds permutations, but on this hardware the large-bond phase is too slow,
  and the unit ordering around a swap network needs an exact centre block (as in the P5 solver ~/peaked-p5/solve_p5.py).
