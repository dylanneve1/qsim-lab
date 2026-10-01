[bench.sh] load at start: 14.78
| circuit | method | active qubits | memory | state (s) | sampler setup (s) | shots (s) | shots/s |
|---|---|---|---|---|---|---|---|
| random n=22 t=8 | compressed | 7 | 0 MiB | 0.0006 | 0.0000 (rand 14 / dense 7 / det 1, 8 passes) | 0.0221 | 4.517e6 |
| random n=22 t=8 | sv | 22 | 64 MiB | 5.3669 | - | 0.0390 | 2.563e6 |
| random n=22 t=16 | compressed | 15 | 0 MiB | 0.0047 | 0.0058 (rand 7 / dense 15 / det 0, 16 passes) | 0.0269 | 3.719e6 |
| random n=22 t=16 | sv | 22 | 64 MiB | 9.1827 | - | 0.0274 | 3.643e6 |
[bench.sh] load at start: 16.62
| random n=22 t=16 | sv | +1.250000000000e-1 | - | - | - | 64 MiB | 9.9821 |
| random n=22 t=16 | frame | +1.250000000000e-1 | 1 | - | - | - | 0.0204 |
| random n=22 t=16 | auto | +1.250000000000e-1 | 1 | k=5 (1 terms) | 4 | 0 MiB (frame 0.002 s + dense 0.000 s) | 0.0024 |
| random n=22 t=16 | dense | +1.250000000000e-1 | 1 | k=16 (1 terms) | 15 | 0 MiB (frame 0.001 s + dense 0.005 s) | 0.0068 |
| random n=22 t=40 | sv | -6.199251861263e-4 | - | - | - | 64 MiB | 24.1954 |
| random n=22 t=40 | frame | -6.199251861262e-4 | 1305 | - | - | - | 0.0059 |
| random n=22 t=40 | auto | -6.199251861262e-4 | 1305 | k=5 (14 terms) | 5 | 0 MiB (frame 0.005 s + dense 0.000 s) | 0.0052 |
| random n=22 t=40 | dense | -6.199251861263e-4 | 1 | k=40 (1 terms) | 22 | 64 MiB (frame 0.003 s + dense 0.258 s) | 0.2653 |
