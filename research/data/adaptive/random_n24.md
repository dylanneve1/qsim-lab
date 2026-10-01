[bench.sh] load at start: 8.93
| circuit | method | value | peak terms | switch | dense qubits | dense memory | time (s) |
|---|---|---|---|---|---|---|---|
| random n=24 t=16 | frame | +3.535533905933e-1 | 1 | - | - | - | 0.0080 |
| random n=24 t=16 | dense | +3.535533905933e-1 | 1 | k=13 (1 terms) | 13 | 0 MiB (frame 0.002 s + dense 0.000 s) | 0.0018 |
| random n=24 t=16 | auto | +3.535533905933e-1 | 1 | k=4 (1 terms) | 4 | 0 MiB (frame 0.002 s + dense 0.000 s) | 0.0019 |
| random n=24 t=24 | frame | -0.000000000000e0 | 1 | - | - | - | 0.0086 |
| random n=24 t=24 | dense | +2.857874402017e-17 | 1 | k=21 (1 terms) | 20 | 16 MiB (frame 0.002 s + dense 0.020 s) | 0.0228 |
| random n=24 t=24 | auto | -0.000000000000e0 | 1 | - | - | - (frame 0.002 s + dense 0.000 s) | 0.0019 |
| random n=24 t=32 | frame | -0.000000000000e0 | 8 | - | - | - | 0.0026 |
| random n=24 t=32 | dense | -3.299951373136e-18 | 1 | k=28 (1 terms) | 23 | 128 MiB (frame 0.002 s + dense 0.209 s) | 0.2188 |
| random n=24 t=32 | auto | -0.000000000000e0 | 8 | - | - | - (frame 0.006 s + dense 0.000 s) | 0.0057 |
| random n=24 t=40 | frame | -0.000000000000e0 | 108 | - | - | - | 0.0035 |
| random n=24 t=40 | dense | -6.513029775379e-18 | 1 | k=36 (1 terms) | 24 | 256 MiB (frame 0.002 s + dense 0.684 s) | 0.7076 |
| random n=24 t=40 | auto | -0.000000000000e0 | 108 | - | - | - (frame 0.010 s + dense 0.000 s) | 0.0105 |
| random n=24 t=48 | frame | -1.220703125000e-4 | 1428 | - | - | - | 0.0102 |
| random n=24 t=48 | dense | -1.220703125000e-4 | 1 | k=44 (1 terms) | 24 | 256 MiB (frame 0.003 s + dense 1.103 s) | 1.1232 |
| random n=24 t=48 | auto | -1.220703124999e-4 | 1 | k=44 (1 terms) | 24 | 256 MiB (frame 0.003 s + dense 0.994 s) | 1.0116 |
| random n=24 t=56 | frame | -2.866637659363e-5 | 3135 | - | - | - | 0.0070 |
| random n=24 t=56 | dense | -2.866637659369e-5 | 1 | k=52 (1 terms) | 24 | 256 MiB (frame 0.004 s + dense 1.384 s) | 1.4086 |
| random n=24 t=56 | auto | -2.866637659368e-5 | 1 | k=52 (1 terms) | 24 | 256 MiB (frame 0.003 s + dense 1.350 s) | 1.3689 |
| random n=24 t=64 | frame | -5.082015779726e-4 | 262585 | - | - | - | 0.1109 |
| random n=24 t=64 | dense | -5.082015779726e-4 | 1 | k=60 (1 terms) | 24 | 256 MiB (frame 0.004 s + dense 1.823 s) | 1.8467 |
| random n=24 t=64 | auto | -5.082015779726e-4 | 1 | k=60 (1 terms) | 24 | 256 MiB (frame 0.005 s + dense 1.836 s) | 1.8566 |
