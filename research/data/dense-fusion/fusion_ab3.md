## 2026-10-04T22:59:53Z threads=6 load={ 18.01 23.18 21.81 } args: su4 24 f32 3 20 off:dense=0 k2:dense=2 off2:dense=0 k3:dense=3
threads 6 (rayon); cycles column = wall x clock x threads / amplitude-updates of the fused op list
| workload | n | prec | config | min wall s | core-cyc/amp-op | max dAmp vs first | all wall s | tiling |
|---|---|---|---|---|---|---|---|---|
| su4 | 24 | f32 | off | 2.5503 | 1.707 | 0.0e0 | 3.631 2.550 2.562 | passes 2070 |
| su4 | 24 | f32 | k2 | 1.1885 | 0.795 | 2.3e-9 | 1.189 1.245 1.320 | passes 243 dense2 230 dense3 0 |
| su4 | 24 | f32 | off2 | 2.5700 | 1.720 | 0.0e0 | 2.759 2.570 2.678 | passes 2070 |
| su4 | 24 | f32 | k3 | 1.2459 | 0.834 | 2.3e-9 | 1.497 1.246 1.334 | passes 243 dense2 230 dense3 0 |
## end 2026-10-04T23:00:18Z load={ 17.79 22.73 21.69 }
## 2026-10-04T23:01:23Z threads=6 load={ 20.33 22.35 21.61 } args: brick 24 f32 3 20 off:dense=0 k2:dense=2 off2:dense=0 k3:dense=3
threads 6 (rayon); cycles column = wall x clock x threads / amplitude-updates of the fused op list
| workload | n | prec | config | min wall s | core-cyc/amp-op | max dAmp vs first | all wall s | tiling |
|---|---|---|---|---|---|---|---|---|
| brick | 24 | f32 | off | 1.0582 | 2.117 | 0.0e0 | 1.735 1.058 1.064 | passes 692 |
| brick | 24 | f32 | k2 | 0.9998 | 2.000 | 0.0e0 | 1.352 1.051 1.000 | passes 692 dense2 0 dense3 0 |
| brick | 24 | f32 | off2 | 1.0673 | 2.135 | 0.0e0 | 1.349 1.077 1.067 | passes 692 |
| brick | 24 | f32 | k3 | 0.9965 | 1.994 | 0.0e0 | 0.996 1.025 1.096 | passes 692 dense2 0 dense3 0 |
## end 2026-10-04T23:01:37Z load={ 19.24 22.02 21.50 }
## 2026-10-04T23:02:42Z threads=6 load={ 17.75 21.04 21.17 } args: adder 24 f32 3 20 off:dense=0 k2:dense=2 off2:dense=0 k3:dense=3
threads 6 (rayon); cycles column = wall x clock x threads / amplitude-updates of the fused op list
| workload | n | prec | config | min wall s | core-cyc/amp-op | max dAmp vs first | all wall s | tiling |
|---|---|---|---|---|---|---|---|---|
| adder | 24 | f32 | off | 1.5636 | 1.805 | 0.0e0 | 2.134 1.580 1.564 | passes 1780 |
| adder | 24 | f32 | k2 | 1.5337 | 1.771 | 6.5e-8 | 1.553 1.534 1.563 | passes 1780 dense2 0 dense3 0 |
| adder | 24 | f32 | off2 | 1.5174 | 1.752 | 0.0e0 | 1.517 1.579 1.575 | passes 1780 |
| adder | 24 | f32 | k3 | 1.5478 | 1.787 | 6.0e-8 | 1.571 1.548 1.564 | passes 1780 dense2 0 dense3 0 |
## end 2026-10-04T23:03:02Z load={ 16.85 20.63 21.02 }
## 2026-10-04T23:04:07Z threads=6 load={ 14.59 19.26 20.48 } args: qft 24 f32 3 1 off:dense=0 k2:dense=2 off2:dense=0 k3:dense=3
threads 6 (rayon); cycles column = wall x clock x threads / amplitude-updates of the fused op list
| workload | n | prec | config | min wall s | core-cyc/amp-op | max dAmp vs first | all wall s | tiling |
|---|---|---|---|---|---|---|---|---|
| qft | 24 | f32 | off | 0.1218 | 1.420 | 0.0e0 | 0.203 0.136 0.122 | passes 59 |
| qft | 24 | f32 | k2 | 0.1218 | 1.421 | 0.0e0 | 0.160 0.135 0.122 | passes 59 dense2 0 dense3 0 |
| qft | 24 | f32 | off2 | 0.1239 | 1.445 | 0.0e0 | 0.124 0.139 0.138 | passes 59 |
| qft | 24 | f32 | k3 | 0.1239 | 1.445 | 0.0e0 | 0.136 0.140 0.124 | passes 59 dense2 0 dense3 0 |
## end 2026-10-04T23:04:09Z load={ 14.59 19.26 20.48 }
## 2026-10-04T23:05:14Z threads=6 load={ 13.41 18.08 19.95 } args: ghz 24 f32 3 1 off:dense=0 k2:dense=2 off2:dense=0 k3:dense=3
threads 6 (rayon); cycles column = wall x clock x threads / amplitude-updates of the fused op list
| workload | n | prec | config | min wall s | core-cyc/amp-op | max dAmp vs first | all wall s | tiling |
|---|---|---|---|---|---|---|---|---|
| ghz | 24 | f32 | off | 0.0521 | 4.811 | 0.0e0 | 0.084 0.052 0.056 | passes 24 |
| ghz | 24 | f32 | k2 | 0.0531 | 4.901 | 0.0e0 | 0.099 0.053 0.053 | passes 24 dense2 0 dense3 0 |
| ghz | 24 | f32 | off2 | 0.0436 | 4.030 | 0.0e0 | 0.060 0.058 0.044 | passes 24 |
| ghz | 24 | f32 | k3 | 0.0420 | 3.875 | 0.0e0 | 0.043 0.052 0.042 | passes 24 dense2 0 dense3 0 |
## end 2026-10-04T23:05:15Z load={ 13.30 17.98 19.90 }
## 2026-10-04T23:06:20Z threads=6 load={ 13.02 16.87 19.33 } args: su4 24 f64 3 20 off:dense=0 k2:dense=2 off2:dense=0 k3:dense=3
threads 6 (rayon); cycles column = wall x clock x threads / amplitude-updates of the fused op list
| workload | n | prec | config | min wall s | core-cyc/amp-op | max dAmp vs first | all wall s | tiling |
|---|---|---|---|---|---|---|---|---|
| su4 | 24 | f64 | off | 3.4297 | 2.295 | 0.0e0 | 4.366 3.430 3.435 | passes 2070 |
| su4 | 24 | f64 | k2 | 2.2793 | 1.525 | 5.6e-18 | 2.420 2.279 2.310 | passes 243 dense2 230 dense3 0 |
| su4 | 24 | f64 | off2 | 3.4278 | 2.294 | 0.0e0 | 3.428 3.442 3.442 | passes 2070 |
| su4 | 24 | f64 | k3 | 2.1722 | 1.454 | 5.6e-18 | 2.298 2.276 2.172 | passes 243 dense2 230 dense3 0 |
## end 2026-10-04T23:06:55Z load={ 14.88 16.96 19.27 }
DONE
