[bench.sh] load at start: 14.01
| circuit | method | value | peak terms | switch | dense qubits | dense memory | time (s) |
|---|---|---|---|---|---|---|---|
| adder n=50 t=4 | frame | -0.000000000000e0 | 128 | - | - | - | 0.0018 |
| adder n=50 t=4 | auto | -1.185723729455e-17 | 4 | k=42 (4 terms) | 10 | 0 MiB (frame 0.000 s + dense 0.000 s) | 0.0002 |
| adder n=50 t=4 | dense | -7.086747735616e-18 | 1 | k=44 (1 terms) | 10 | 0 MiB (frame 0.000 s + dense 0.000 s) | 0.0001 |
| adder n=50 t=8 | frame | -0.000000000000e0 | 32768 | - | - | - | 0.0170 |
| adder n=50 t=8 | auto | -1.085948013535e-18 | 4 | k=86 (4 terms) | 18 | 4 MiB (frame 0.000 s + dense 0.059 s) | 0.0597 |
| adder n=50 t=8 | dense | -6.505213034913e-19 | 1 | k=88 (1 terms) | 18 | 4 MiB (frame 0.000 s + dense 0.044 s) | 0.0440 |
| adder n=50 t=12 | frame | aborted: 4194304 Pauli terms exceeds the limit of 4000000 | - | - | - | - | 0.4028 |
