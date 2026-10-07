# P6 "Titan Pinnacle" (62q): ring decomposition

P6 alternates 17 **bursts** (2–5 layers of CZs on ring edges only) with 16 **gaps** (8–9 layers of cross-ring and
long-range CZ gadgets that act as masked identities) on three qubit rings: A (20), B (20, a reversed twin of A in the
first ~5 bursts) and C (22).

Method (every step is in `NOTES.md` with its script):
1. `segment.py`: burst/gap detection. `epochs3()` moves the 86 cross CZs at burst edges into the adjacent gap;
   `epochs4()` also moves 335 gap-edge ring CZs into bursts. Causal order is preserved.
2. `gapact.py`, `allimgs.py`: exact gate-level Heisenberg images of X_q and Z_q through each gap.
3. `effcirc.py`: each gap factors into small clusters (≤10 wires), and only 2 cross-ring clusters remain in the whole
   circuit. Cluster unitaries are rebuilt with fidelity ≥ 0.9995.
4. `ttn.py`: a star tree tensor network with one exact state-vector leaf per ring (2^20/2^20/2^22). The core bond is at
   most 2, with zero truncation. `test_ttn.py` checks it is exact against a dense simulation.
5. `gapcert.py`: exact tensor-network fidelity of the raw gates against each rebuilt gap. Every gap scores ≥ 0.99965,
   and the product over all 17 gap epochs is 0.99931.
6. `ttn_peak.py`: the peak, plus single-flip amplitudes. Both segmentations give the identical string, P ≈ 0.0108,
   with the best single flip at 2.25e-5 (478× lower).

The answer string is deliberately not stored here; see `../PORTAL.md` for the salted commitment.
