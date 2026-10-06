# BlueQubit Peak Portal: methods, evidence, results

Our run at BlueQubit's [Peak Portal](https://app.bluequbit.io/hackathons) (12 peaked circuits, 2570 points).
Every answer comes blind from our own tooling: no target string is read at runtime, and wrong-answer overlap feedback from the
portal is never used as an oracle. Each answer is submitted only once the evidence below is in hand.

**Answer strings for problems that are still live are not published here.** Publishing them would hand the answers to every
other participant. Each one is instead **committed** as `sha256(salt ‖ answer)`, with a random 256-bit salt kept privately. That
timestamps the answer in git now, and anyone can verify it once salt and answer are revealed after the portal closes.
P11/P12 were already public (`reproduce.sh` checks them against the Helios-1 hardware peaks).

| P | qubits / 2q gates | method | evidence | score | commitment (sha256 of salt‖answer) |
|---|---|---|---|---|---|
| 1 | 4 / 0 | exact by hand: X then RY(0.8π) is a product state | exact | 10/10 | `8dab9cc77f1b3f4731777735ce15a61623a216f2430d1a418479af176b1cc9d3` |
| 2 | 28 / 210 CZ | exact state vector over all 2^28 amplitudes (`tools/sv.py`, complex64) | global argmax p = 0.349, runner-up 0.0045 (78×); both re-checked with complex128 TN amplitudes (quimb): 0.348765 / 0.004476 | 20/20 | `990cdd9e7ebf0479a2107b338a121c3ee58b5da8d8cfd7969b55003aa0f0b057` |
| 3 | 44 ring / 178 CZ | exact TN light-cone marginals + exact amplitude (`tools/direct.py`), certificate (`tools/cert.py`) | **certified**: p = 0.1127; the exact joint marginal over the 16 ambiguous wires has only the peak's configuration ≥ p (0.219, next 0.044); all other wires' minority < p (max 0.11199) | 50/50 | `d5258a532a38094c6182ec829bb5675d34dfba2fb80396d91b615c0f5cf34f9b` |
| 10 | 49 heavy-hex / 4020 CZ | middle-out TN-operator compression from the mirror centre (quimb `tensor_network_ag_compress`, cutoff 1e-3), then exact contraction | same peak from centre layers 110 and 112; p = 0.695–0.699 (the circuit is advertised as ~70% peaked); min \|⟨Z⟩\| 0.837. The unchanged pipeline on the tracker's heavy_hex_49x5072 matched its published reference on all 49 bits | 100/100 | `c11aae36d370d5a0081c21c82e701c7c825129cb766838fee587112b75de8b1f` |
| 11 | 98 / 1999 CZ | structural de-obfuscation, `solve_peaked_v2.py` | matches Helios-1 hardware peak; see `V2_MAXP.md` | 500/500 | public (`reproduce.sh`) |
| 12 | 98 / 2457 CZ | structural de-obfuscation, `solve_peaked_v2.py` | matches Helios-1 hardware peak; see `V2_MAXP.md` | 500/500 | public (`reproduce.sh`) |

Total so far: **1180 / 2570**. Open: P4 (48q ring, 5096 CZ), P5 (44q all-to-all, 1892 CZ), P6 (62q all-to-all, 3494 CZ),
P7 (45q heavy-hex, 1275 CZ), P8 (40q grid, 888 iSWAP), P9 (56q HQAP, 1917 rzz).

## What we learned about the families
- **Small or shallow (P2, P3):** exact simulation is enough. The new tool is a *marginal certificate*: a string S with probability
  p is the unique peak if every other wire's minority marginal is below p and the exact joint marginal over the remaining
  "ambiguous" wires reaches p only at S's configuration.
- **HQAP P11/P12:** the obfuscated identity leaves exact inverse single-qubit segments ("anchors"), so a structural rewrite
  works. P9 and the heavy-hex circuits have **zero** anchors (every segment was re-swept), so segment fingerprints are useless there.
- **Heavy-hex (P10, tracker 5072):** cancellation is visible only at the operator level. Growing a compressed operator outward
  from the mirror centre (located by a parameter-free scan for where the operator stays compressible) absorbs the whole circuit
  at bond dimension 8, in about 90 s and 210 MB.
- **Full-circuit certificate for P11/P12:** a worst-case bound is fundamentally vacuous; see `bound-attempt/README.md`.
