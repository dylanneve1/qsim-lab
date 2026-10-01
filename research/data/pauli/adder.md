[bench.sh] load at start: 11.78
## Cuccaro adder, Pauli-path summation (legacy)
| bits | qubits | T gates | observable | Pauli terms (peak) | value | time (s) |
|---|---|---|---|---|---|---|
| 2 | 6 | 28 | Z_cout | 18 | +2.500000000000e-1 | 0.0002 |
| 2 | 6 | 28 | Z_b0 Z_b_top | 8 | -0.000000000000e0 | 0.0004 |
| 4 | 10 | 56 | Z_cout | 78 | +6.250000000000e-2 | 0.0005 |
| 4 | 10 | 56 | Z_b0 Z_b_top | 128 | -0.000000000000e0 | 0.0010 |
| 8 | 18 | 112 | Z_cout | 1278 | +3.906250000000e-3 | 0.0084 |
| 8 | 18 | 112 | Z_b0 Z_b_top | 32768 | -0.000000000000e0 | 0.0608 |
| 16 | 34 | 224 | Z_cout | 327678 | +1.525878906250e-5 | 0.3002 |
| 16 | 34 | 224 | Z_b0 Z_b_top | aborted: 8388608 Pauli terms exceeds the limit of 4194304 | - | 1.8193 |
[bench.sh] load at start: 11.96
## Cuccaro adder, Pauli-path summation (frame-noprune-nomerge)
| bits | qubits | T gates | observable | Pauli terms (peak) | value | time (s) |
|---|---|---|---|---|---|---|
| 2 | 6 | 28 | Z_cout | 18 | +2.500000000000e-1 | 0.0036 |
| 2 | 6 | 28 | Z_b0 Z_b_top | 8 | -0.000000000000e0 | 0.0008 |
| 4 | 10 | 56 | Z_cout | 78 | +6.250000000000e-2 | 0.0019 |
| 4 | 10 | 56 | Z_b0 Z_b_top | 128 | -0.000000000000e0 | 0.0018 |
| 8 | 18 | 112 | Z_cout | 1278 | +3.906250000000e-3 | 0.0083 |
| 8 | 18 | 112 | Z_b0 Z_b_top | 32768 | -0.000000000000e0 | 0.0196 |
| 16 | 34 | 224 | Z_cout | 327678 | +1.525878906250e-5 | 0.0918 |
| 16 | 34 | 224 | Z_b0 Z_b_top | aborted: 8388608 Pauli terms exceeds the limit of 4194304 | - | 0.8661 |
[bench.sh] load at start: 11.96
## Cuccaro adder, Pauli-path summation (frame-noprune)
| bits | qubits | T gates | observable | Pauli terms (peak) | value | time (s) |
|---|---|---|---|---|---|---|
| 2 | 6 | 28 | Z_cout | 18 | +2.500000000000e-1 | 0.0013 |
| 2 | 6 | 28 | Z_b0 Z_b_top | 8 | -0.000000000000e0 | 0.0010 |
| 4 | 10 | 56 | Z_cout | 78 | +6.250000000000e-2 | 0.0022 |
| 4 | 10 | 56 | Z_b0 Z_b_top | 128 | -0.000000000000e0 | 0.0021 |
| 8 | 18 | 112 | Z_cout | 1278 | +3.906250000000e-3 | 0.0028 |
| 8 | 18 | 112 | Z_b0 Z_b_top | 32768 | -0.000000000000e0 | 0.0190 |
| 16 | 34 | 224 | Z_cout | 327678 | +1.525878906250e-5 | 0.0912 |
| 16 | 34 | 224 | Z_b0 Z_b_top | aborted: 8388608 Pauli terms exceeds the limit of 4194304 | - | 0.7763 |
[bench.sh] load at start: 11.96
## Cuccaro adder, Pauli-path summation (frame)
| bits | qubits | T gates | observable | Pauli terms (peak) | value | time (s) |
|---|---|---|---|---|---|---|
| 2 | 6 | 28 | Z_cout | 4 | +2.500000000000e-1 | 0.0022 |
| 2 | 6 | 28 | Z_b0 Z_b_top | 8 | -0.000000000000e0 | 0.0016 |
| 4 | 10 | 56 | Z_cout | 4 | +6.250000000000e-2 | 0.0044 |
| 4 | 10 | 56 | Z_b0 Z_b_top | 128 | -0.000000000000e0 | 0.0023 |
| 8 | 18 | 112 | Z_cout | 4 | +3.906250000000e-3 | 0.0026 |
| 8 | 18 | 112 | Z_b0 Z_b_top | 32768 | -0.000000000000e0 | 0.0106 |
| 16 | 34 | 224 | Z_cout | 4 | +1.525878906250e-5 | 0.0050 |
| 16 | 34 | 224 | Z_b0 Z_b_top | aborted: 8388608 Pauli terms exceeds the limit of 4194304 | - | 0.8083 |
