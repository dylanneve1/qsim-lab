# Profile output (2026-10-06)
```
qubits 70; gates {'h': 350, 'sx': 5061, 'sxdg': 161, 'cz': 2415, 's': 2547, 'rz': 468}
CZ: 2415 total, distance histogram {1: 2415}, distinct edges 69, CZ-depth 70
CZ per layer: min 34 max 35
T gates: 468; per CZ-layer min/mean/max 1/6.9/62 over 68 layers; per qubit min/max 2/30
first T at CZ-layer 2, last at 70
T by chain decile: [29, 24, 21, 32, 128, 65, 36, 59, 38, 36]
T by depth decile: [15, 23, 16, 13, 28, 19, 17, 28, 60, 249]

[undoped (T removed)] entanglement (bits) across chain cut k|n-k, k=1..69:
  max 32 at k=33; MPS bond dim needed 2^32
  random half-bipartitions (200): min 32 max 35

[T->S Clifford proxy] entanglement (bits) across chain cut k|n-k, k=1..69:
  max 32 at k=33; MPS bond dim needed 2^32
  random half-bipartitions (200): min 32 max 35
T count 468
weight pushed to END: min 1 median 7 max 61; hist(<=4,<=10,<=20,>20) 186,92,44,146
X-weight at END: min 1 median 5 max 45; hist(<=4,<=10,<=20,>20) 223,78,39,128
weight pushed to START: min 1 median 51 max 61; hist(<=4,<=10,<=20,>20) 6,13,23,426
X-weight at START (0 = trivial phase on |0>): min 0 median 34 max 49; hist(<=4,<=10,<=20,>20) 9,24,26,409
min(end,start) weight: hist(<=4,<=10,<=20,>20) 192 105 67 104
T trivially absorbed at start (no X part): 1
```
