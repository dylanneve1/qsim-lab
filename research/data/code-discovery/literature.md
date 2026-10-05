# Published BB / GB / coprime-BB / multivariate / 2BGA codes (n <= ~300) — literature table

Compiled 2026-10-05 by sub-agent qldpc-lit-tables from arXiv HTML versions (tables parsed directly; polynomials copied verbatim modulo LaTeX->ASCII).

## Conventions
* BB (Bravyi et al.): group Z_l x Z_m, x = S_l (x) I_m (cyclic shift on Z_l), y = I_l (x) S_m. H_X=[A|B], H_Z=[B^T|A^T]. n=2lm. In IBM code: `x[i]=kron(np.roll(I_l,i,axis=1),I_m)`, `y[i]=kron(I_l,np.roll(I_m,i,axis=1))`.
* Coprime-BB (Wang&Mueller): gcd(l,m)=1, pi = xy generates Z_lm, so these are *GB codes on cyclic Z_{lm}*.
* Multivariate (Voss+): third variable z = xy.
* Generalized toric (Liang+ 2503.03827): f=1+x+x^a y^b, g=1+y+x^c y^d on twisted torus Z^2/<a1,a2> (abelian group of order n/2; = 2BGA/BB on that group). These tables claim to be *optimal* (exhaustive) weight-6 codes of that form for each n<=400 (one representative per n, plus some ties). This is the single most important "do not rediscover" list.
* GB (1D): cyclic group Z_l, A=a(x), B=b(x).
* exact? column: Y = exact (ILP/MILP or paper says exact); P = paper-reported, method not stated/assumed exact; Q = QDistRnd randomized (upper bound, usually tight); ~ = Postema&Kokkelmans "accurate up to ±2"; <= = upper bound only; range = lower–upper.
* Check weight 6 unless noted (column w).

## IBM syndrome circuit (sbravyi/BivariateBicycleCodes, decoder_setup.py, fetched raw)
```
sX = ['idle', 1, 4, 3, 5, 0, 2]
sZ = [3, 5, 0, 1, 2, 4, 'idle']
# neighbour labelling (direction index -> data qubit):
# X check i: 0:L via A1[i,:], 1:L via A2[i,:], 2:L via A3[i,:], 3:R via B1[i,:], 4:R via B2[i,:], 5:R via B3[i,:]
# Z check i: 0:L via B1[:,i], 1:L via B2[:,i], 2:L via B3[:,i], 3:R via A1[:,i], 4:R via A2[:,i], 5:R via A3[:,i]
#   (i.e. Z check uses B^T on left, A^T on right; "nonzero(B1[:,i])" = column i)
# A1=x^a1, A2=y^a2, A3=y^a3 ; B1=y^b1, B2=x^b2, B3=x^b3   (A = x^a1+y^a2+y^a3, B = y^b1+x^b2+x^b3)
# x[i] = kron(np.roll(I_ell,i,axis=1), I_m); y[i] = kron(I_ell, np.roll(I_m,i,axis=1))
```
Round structure (one syndrome cycle, 8 layers incl. prep/meas):
* round 0: PrepX on X-checks; CNOT data(nbs[Z,sZ[0]=3]) -> Zcheck (control=data, target=Zcheck); other data idle.
* rounds 1..5: CNOT Xcheck -> data(nbs[X,sX[t]]) (control=Xcheck) AND CNOT data(nbs[Z,sZ[t]]) -> Zcheck.
* round 6: MeasZ on Z-checks; CNOT Xcheck -> data(nbs[X,sX[6]=2]); remaining data idle.
* round 7: all data idle; MeasX on X-checks; PrepZ on Z-checks.
So per round t=0..6: X-check direction sX[t], Z-check direction sZ[t]:
| t | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| X check | idle | A2 (L,dir1) | B2 (R,dir4) | B1 (R,dir3) | B3 (R,dir5) | A1 (L,dir0) | A3 (L,dir2) |
| Z check | A1^T (R,dir3) | A3^T (R,dir5) | B1^T (L,dir0) | B2^T (L,dir1) | B3^T (L,dir2) | A2^T (R,dir4) | idle |
Noise in reference sims: depolarizing p for init/idle/CNOT/meas (error_rate=0.003 in file default), num_cycles=12. Paper (2308.07915 Table 1): d_circ ≤6/≤8/≤8/≤10/≤18 for 72/90/108/144/288; pL per round at p=1e-3: 7e-5, 5e-6, 3e-6, 2e-7, 2e-12; pseudo-thresholds 0.48/0.53/0.58/0.65/0.69%.
Note: this schedule is *not* distance-preserving (gross code d_circ ≤10 < 12). Voss+ 2406.19151 Table 3 gives an alternative depth-7 schedule for their weight-5 [[30,4,5]] (rounds: A1/A3^T, B2/B1^T, A2/A2^T, B1/B2^T, A3/A1^T).
Tour de gross (2506.03094) Table 4: gross d_circ ≤10, two-gross d_circ ≤18 (idle); idle logical error per cycle at p=1e-3: gross 10^-8.8, two-gross 10^-20.1 (Relay-BP, per 8-timestep cycle, whole block). Tour de gross presents gross/two-gross as a_→=−b_↑=3, a_↑=b_→=−1 on l=12,m=6 / l=12,m=12 (toric+2 long-range edges form, equivalent to x^3+y+y^2 / y^3+x+x^2 up to relabelling).

## Per-source tables

### BCGMRY 2308.07915 T3

| n | k | d | exact? | w | group | A | B | kd²/n | notes |
|---|---|---|---|---|---|---|---|---|---|
| 72 | 12 | 6 | Y | 6 | Z6xZ6 | `x^3+y+y^2` | `y^3+x+x^2` | 6.00 | d_circ<=6; pL(1e-3)=7e-5/round; p0=0.48% |
| 90 | 8 | 10 | Y | 6 | Z15xZ3 | `x^9+y+y^2` | `1+x^2+x^7` | 8.89 | d_circ<=8; pL(1e-3)=5e-6; p0=0.53% |
| 108 | 8 | 10 | Y | 6 | Z9xZ6 | `x^3+y+y^2` | `y^3+x+x^2` | 7.41 | d_circ<=8; pL(1e-3)=3e-6; p0=0.58% |
| 144 | 12 | 12 | Y | 6 | Z12xZ6 | `x^3+y+y^2` | `y^3+x+x^2` | 12.00 | gross; d_circ<=10; pL(1e-3)=2e-7; p0=0.65% |
| 288 | 12 | 18 | Y | 6 | Z12xZ12 | `x^3+y^2+y^7` | `y^3+x+x^2` | 13.50 | two-gross; d_circ<=18; pL(1e-3)=2e-12; p0=0.69% |
| 360 | 12 | 24 | <= | 6 | Z30xZ6 | `x^9+y+y^2` | `y^3+x^25+x^26` | 19.20 | upper bound (Symons 2511.13560 lists it as 24 bold w/o <=; LLM paper 2606.02418 still says MILP incumbent <=24) |
| 756 | 16 | 34 | <= | 6 | Z21xZ18 | `x^3+y^10+y^17` | `y^5+x^3+x^19` | 24.47 | n>300 |

### IBM GitHub decoder_setup.py

| n | k | d | exact? | w | group | A | B | kd²/n | notes |
|---|---|---|---|---|---|---|---|---|---|
| 784 | 24 | 24 | ? | 6 | Z28xZ14 | `x^26+y^6+y^8` | `y^7+x^9+x^20` | 17.63 | commented example in sbravyi/BivariateBicycleCodes; n>300 |

### Wang&Mueller 2408.10001 T1 (Alg1)

| n | k | d | exact? | w | group | A | B | kd²/n | notes |
|---|---|---|---|---|---|---|---|---|---|
| 54 | 8 | 6 | P | 6 | Z3xZ9 | `1+y^2+y^4` | `y^3+x+x^2` | 5.33 |  |
| 98 | 6 | 12 | P | 6 | Z7xZ7 | `x^3+y^5+y^6` | `y^2+x^3+x^5` | 8.82 |  |
| 126 | 8 | 10 | P | 6 | Z3xZ21 | `1+y^2+y^10` | `y^3+x+x^2` | 6.35 |  |
| 150 | 16 | 8 | P | 6 | Z5xZ15 | `1+y^6+y^8` | `y^5+x+x^4` | 6.83 |  |
| 162 | 8 | 14 | P | 6 | Z3xZ27 | `1+y^10+y^14` | `y^12+x+x^2` | 9.68 |  |
| 180 | 8 | 16 | P | 6 | Z6xZ15 | `x^3+y+y^2` | `y^6+x^4+x^5` | 11.38 |  |

### Wang&Mueller 2408.10001 T2 (coprime, pi=xy)

| n | k | d | exact? | w | group | A | B | kd²/n | notes |
|---|---|---|---|---|---|---|---|---|---|
| 30 | 4 | 6 | P | 6 | Z3xZ5=Z15 | `1+pi+pi^2` | `1+pi^2+pi^7` | 4.80 |  |
| 42 | 6 | 6 | P | 6 | Z3xZ7=Z21 | `1+pi^2+pi^3` | `1+pi^2+pi^10` | 5.14 |  |
| 70 | 6 | 8 | P | 6 | Z5xZ7=Z35 | `1+pi+pi^5` | `1+pi+pi^12` | 5.49 |  |
| 108 | 12 | 6 | P | 6 | Z2xZ27=Z54 | `1+pi^3+pi^42` | `1+pi^6+pi^39` | 4.00 |  |
| 126 | 12 | 10 | P | 6 | Z7xZ9=Z63 | `1+pi+pi^58` | `1+pi^13+pi^41` | 9.52 | d_circ<=9 (vs gross d_circ<=10) |
| 154 | 6 | 16 | P | 6 | Z7xZ11=Z77 | `1+pi+pi^31` | `1+pi^19+pi^53` | 9.97 |  |

### Wang&Mueller 2408.10001 T3 (Alg1)

| n | k | d | exact? | w | group | A | B | kd²/n | notes |
|---|---|---|---|---|---|---|---|---|---|
| 18 | 4 | 4 | P | 6 | Z3xZ3 | `1+x+y` | `1+x^2+y^2` | 3.56 |  |
| 36 | 8 | 4 | P | 6 | Z3xZ6 | `1+y+y^2` | `x^3+y+y^2` | 3.56 |  |
| 36 | 4 | 6 | P | 6 | Z3xZ6 | `x+y^2+y^3` | `1+y+x^2` | 4.00 |  |
| 54 | 4 | 8 | P | 6 | Z3xZ9 | `x+y+y^3` | `1+y^2+x^2` | 4.74 |  |
| 196 | 18 | 8 | P | 6 | Z7xZ14 | `1+y+y^3` | `y^7+x+x^3` | 5.88 |  |

### Wang&Mueller 2408.10001 T4 (coprime pi=xy)

| n | k | d | exact? | w | group | A | B | kd²/n | notes |
|---|---|---|---|---|---|---|---|---|---|
| 28 | 6 | 4 | P | 6 | Z2xZ7 | `1+pi+pi^3` | `1+pi+pi^10` | 3.43 |  |
| 36 | 8 | 4 | P | 6 | Z2xZ9 | `1+pi^2+pi^10` | `1+pi^4+pi^8` | 3.56 |  |
| 42 | 10 | 4 | P | 6 | Z3xZ7 | `1+pi+pi^5` | `1+pi^2+pi^10` | 3.81 |  |
| 48 | 4 | 8 | P | 6 | Z3xZ8 | `1+pi+pi^2` | `1+pi^2+pi^10` | 5.33 |  |
| 60 | 16 | 4 | P | 6 | Z3xZ10 | `1+pi^2+pi^8` | `1+pi^4+pi^16` | 4.27 |  |
| 66 | 4 | 10 | P | 6 | Z3xZ11 | `1+pi+pi^5` | `1+pi+pi^23` | 6.06 |  |
| 56 | 6 | 8 | P | 6 | Z4xZ7 | `1+pi+pi^3` | `1+pi^5+pi^11` | 6.86 |  |
| 90 | 4 | 12 | P | 6 | Z5xZ9 | `1+pi+pi^4` | `1+pi^8+pi^34` | 6.40 |  |
| 90 | 8 | 8 | P | 6 | Z5xZ9 | `1+pi+pi^12` | `1+pi^2+pi^9` | 5.69 |  |
| 84 | 6 | 10 | P | 6 | Z6xZ7 | `1+pi+pi^3` | `1+pi^8+pi^31` | 7.14 |  |
| 132 | 4 | 14 | P | 6 | Z6xZ11 | `1+pi+pi^2` | `1+pi^11+pi^28` | 5.94 |  |
| 112 | 6 | 12 | P | 6 | Z7xZ8 | `1+pi+pi^3` | `1+pi^5+pi^25` | 7.71 |  |
| 126 | 6 | 14 | P | 6 | Z7xZ9 | `1+pi^4+pi^19` | `1+pi^6+pi^16` | 9.33 |  |
| 180 | 8 | 16 | P | 6 | Z9xZ10 | `1+pi+pi^4` | `1+pi^23+pi^62` | 11.38 |  |

### Wang&Mueller 2408.10001 T5 (coprime, weight 8)

| n | k | d | exact? | w | group | A | B | kd²/n | notes |
|---|---|---|---|---|---|---|---|---|---|
| 24 | 8 | 4 | P | 8 | Z3xZ4 | `1+pi+pi^3+pi^4` | `1+pi^2+pi^5+pi^9` | 5.33 |  |
| 30 | 10 | 4 | P | 8 | Z15 | `1+pi+pi^2+pi^7` | `1+pi+pi^4+pi^10` | 5.33 |  |
| 30 | 6 | 5 | P | 8 | Z15 | `1+pi+pi^3+pi^4` | `1+pi+pi^3+pi^7` | 5.00 |  |
| 42 | 12 | 5 | P | 8 | Z21 | `1+pi+pi^3+pi^13` | `1+pi+pi^4+pi^9` | 7.14 |  |
| 42 | 6 | 7 | P | 8 | Z21 | `1+pi+pi^3+pi^4` | `1+pi+pi^6+pi^10` | 7.00 |  |
| 48 | 6 | 8 | P | 8 | Z24 | `1+pi+pi^2+pi^3` | `1+pi^3+pi^9+pi^14` | 8.00 |  |
| 48 | 10 | 6 | P | 8 | Z24 | `1+pi+pi^3+pi^10` | `1+pi^3+pi^7+pi^16` | 7.50 |  |
| 40 | 14 | 4 | P | 8 | Z20 | `1+pi+pi^6+pi^15` | `1+pi^2+pi^5+pi^7` | 5.60 |  |
| 40 | 6 | 6 | P | 8 | Z20 | `1+pi+pi^2+pi^3` | `1+pi+pi^3+pi^10` | 5.40 |  |
| 40 | 8 | 5 | P | 8 | Z20 | `1+pi+pi^4+pi^5` | `1+pi+pi^4+pi^9` | 5.00 |  |
| 56 | 8 | 8 | P | 8 | Z28 | `1+pi+pi^2+pi^4` | `1+pi^2+pi^6+pi^19` | 9.14 |  |
| 56 | 14 | 6 | P | 8 | Z28 | `1+pi+pi^4+pi^9` | `1+pi+pi^17+pi^20` | 9.00 |  |
| 60 | 12 | 7 | P | 8 | Z30 | `1+pi+pi^2+pi^7` | `1+pi^3+pi^12+pi^25` | 9.80 |  |
| 60 | 6 | 9 | P | 8 | Z30 | `1+pi+pi^3+pi^4` | `1+pi^2+pi^11+pi^18` | 8.10 |  |
| 70 | 8 | 9 | P | 8 | Z35 | `1+pi+pi^2+pi^4` | `1+pi+pi^6+pi^24` | 9.26 |  |

### Voss+ 2406.19151 T2 (z=xy)

| n | k | d | exact? | w | group | A | B | kd²/n | notes |
|---|---|---|---|---|---|---|---|---|---|
| 30 | 6 | 4 | P | 6 | l,m=5,3 | `x^4+z^3` | `x^4+x+z^4+y` | 3.20 | weight-6 (2+4); p0=2.34%, pL(1e-4)=3e-7 (code-capacity-ish) |
| 48 | 6 | 6 | P | 6 | 4,6 | `x^2+y^4` | `x^3+z^3+y^2+y` | 4.50 | weight 6 (2+4) |
| 40 | 4 | 6 | P | 6 | 4,5 | `x^2+y` | `y^4+y^2+x^3+x` | 3.60 | weight 6 (2+4) |
| 48 | 4 | 6 | P | 6 | 4,6 | `x^3+y^5` | `x+z^5+y^5+y^2` | 3.00 | weight 6 (2+4); toric layout |

### Voss+ 2406.19151 T2 wt4/5/7

| n | k | d | exact? | w | group | A | B | kd²/n | notes |
|---|---|---|---|---|---|---|---|---|---|
| 112 | 8 | 5 | P | see notes | 7,8 | `z^2+z^6` | `x+x^6` | 1.79 | wt4 |
| 64 | 2 | 8 | P | see notes | 8,4 | `x+x^2` | `x^3+y` | 2.00 | wt4 |
| 72 | 2 | 8 | P | see notes | 4,9 | `x+y^2` | `x^2+y^2` | 1.78 | wt4 |
| 96 | 2 | 8 | P | see notes | 6,8 | `x^5+y^6` | `z+z^4` | 1.33 | wt4 |
| 112 | 2 | 10 | P | see notes | 7,8 | `z^6+x^5` | `z^2+y^5` | 1.79 | wt4 |
| 144 | 2 | 12 | P | see notes | 8,9 | `x^3+y^7` | `x+y^5` | 2.00 | wt4 |
| 30 | 4 | 5 | P | see notes | 3,5 | `x+z^4` | `x+y^2+z^2` | 3.33 | wt5 |
| 72 | 4 | 8 | P | see notes | 4,9 | `x+y^3` | `x^2+y+y^2` | 3.56 | wt5 |
| 96 | 4 | 8 | P | see notes | 8,6 | `x^6+x^3` | `z^5+x^5+y` | 2.67 | wt5 |
| 30 | 4 | 5 | P | see notes | 5,3 | `x^4+x^2` | `x+x^2+y+z^2+z^3` | 3.33 | wt7 |

### Postema&Kokkelmans 2502.17052 T1/T2 (d +-2!)

| n | k | d | exact? | w | group | A | B | kd²/n | notes |
|---|---|---|---|---|---|---|---|---|---|
| 150 | 4 | 10 | ~ | 6 | Z3xZ25 (z=xy) | `1+z+z^2` | `1+z^2+z^16` | 2.67 |  |
| 198 | 4 | 12 | ~ | 6 | Z9xZ11 | `1+z+z^2` | `1+z^5+z^37` | 2.91 |  |
| 270 | 4 | 16 | ~ | 6 | Z5xZ27 | `1+z+z^2` | `1+z^2+z^25` | 3.79 |  |
| 186 | 10 | 6 | ~ | 6 | Z3xZ31 | `1+z^2+z^5` | `1+z^2+z^36` | 1.94 |  |
| 18 | 4 | 2 | ~ | 6 | Z3xZ3 | `1+y+y^2` | `y+1+x` | 0.89 |  |
| 18 | 8 | 2 | ~ | 6 | Z3xZ3 | `1+y+y^2` | `1+x+x^2` | 1.78 |  |
| 36 | 4 | 4 | ~ | 6 | Z6xZ3 | `1+y+y^2` | `y+1+x` | 1.78 |  |
| 84 | 12 | 4 | ~ | 6 | Z7xZ6 | `1+y+y^2` | `1+x+x^3` | 2.29 |  |
| 98 | 6 | 8 | ~ | 6 | Z7xZ7 | `x^4+y+y^3` | `y^4+x+x^3` | 3.92 |  |
| 196 | 6 | 12 | ~ | 6 | Z14xZ7 | `x^4+y+y^3` | `y^4+x+x^3` | 4.41 |  |

### Berthusen+ 2404.17676 T1

| n | k | d | exact? | w | group | A | B | kd²/n | notes |
|---|---|---|---|---|---|---|---|---|---|
| 72 | 8 | 6 | Q | 6 | Z12xZ3 | `x^9+y+y^2` | `1+x+x^11` | 4.00 |  |
| 90 | 8 | 6 | Q | 6 | Z9xZ5 | `x^8+y^4+y` | `y^5+x^8+x^7` | 3.20 |  |
| 120 | 8 | 8 | Q | 6 | Z12xZ5 | `x^10+y^4+y` | `1+x+x^2` | 4.27 |  |
| 150 | 8 | 8 | Q | 6 | Z15xZ5 | `x^5+y^2+y^3` | `y^2+x^7+x^6` | 3.41 |  |
| 196 | 12 | 8 | Q | 6 | Z14xZ7 | `x^6+y^5+y^6` | `1+x^4+x^13` | 3.92 |  |

### Eberhardt&Steffan 2407.03973 T1

| n | k | d | exact? | w | group | A | B | kd²/n | notes |
|---|---|---|---|---|---|---|---|---|---|
| 90 | 8 | 10 | P | 6 | Z3xZ15 | `1+y+y^5` | `y^3+x+x^2` | 8.89 |  |
| 108 | 16 | 6 | P | 6 | Z6xZ9 | `1+y+y^2` | `y^3+x^2+x^4` | 5.33 |  |
| 162 | 4 | 16 | P | 6 | Z9xZ9 | `1+x+y` | `x^3+y+y^2` | 6.32 |  |
| 162 | 12 | 8 | P | 6 | Z9xZ9 | `1+x+y^6` | `y^3+x^2+x^3` | 4.74 |  |
| 162 | 24 | 6 | P | 6 | Z9xZ9 | `1+y+y^2` | `y^3+x^3+x^6` | 5.33 |  |
| 270 | 8 | 18 | P | 6 | Z9xZ15 | `x^3+y+y^2` | `y^3+x+x^2` | 9.60 |  |
| 98 | 6 | 12 | P | 6 | Z7xZ7 | `x+y^3+y^4` | `y+x^3+x^4` | 8.82 |  |
| 162 | 8 | 12 | P | 6 | Z9xZ9 | `x^3+y+y^2` | `y^3+x+x^2` | 7.11 |  |
| 128 | 14 | 12 | P | 8 | Z8xZ8 | `x^2+y+y^3+y^4` | `y^2+x+x^3+x^4` | 15.75 | weight 8 |

### Galimova 2603.17703 (trivariate)

| n | k | d | exact? | w | group | A | B | kd²/n | notes |
|---|---|---|---|---|---|---|---|---|---|
| 84 | 6 | 10 | Y | 6 | Z2xZ3xZ7=Z42 | `1+y^2z^4+xyz^5` | `1+z+xyz^3` | 7.14 | pL(1e-3)=2e-6/round, p0=0.53% |
| 140 | 6 | 14 | Y | 6 | Z2xZ5xZ7=Z70 | `1+yz^3+xyz^2` | `1+xy^4z^2+xy^4z^3` | 8.40 | pL(1e-3)=4e-8/round; p0=0.59% |
| 196 | 6 | 12 | Y | 6 | Z2xZ7xZ7 | `1+xz^2+xy^3z^6` | `1+xyz^6+xy^3z^2` | 4.41 | pL(1e-3)=7e-8 |
| 54 | 8 | 6 | Y | 6 | Z3^3 | `1+z^2+xz` | `1+xy+xy^2` | 5.33 | pL(1e-3)=1e-4 |
| 54 | 14 | 5 | Y | 8 | Z3^3 | `1+x+y+z` | `1+x^2+y^2+z^2` | 6.48 | weight 8, B=A^T |
| 128 | 20 | 8 | Y | 8 | Z4^3 | `1+x+y+z` | `1+x^3+y^3+z^3` | 10.00 | weight 8; pL(1e-3)=2e-5 |

### 2606.02418 (LLM evolutionary)

| n | k | d | exact? | w | group | A | B | kd²/n | notes |
|---|---|---|---|---|---|---|---|---|---|
| 288 | 24 | 12 | Y | 6 | Z12xZ12 | `x^6+y+y^2` | `y^3+x^2+x^4` | 12.00 | decomposable = 2x gross |
| 288 | 16 | 12 | Y | 6 | Z12xZ12 | `x^3+y+y^2` | `y^3+x+x^2` | 8.00 | indecomposable |
| 144 | 8 | 12 | Y | 6 | Z12xZ6 | `1+xy^2+xy^3` | `1+x^2y^3+x^3y^2` | 8.00 |  |
| 144 | 24 | 6 | Y | 6 | Z12xZ6 | `x^6+y+y^2` | `y^3+x^2+x^4` | 6.00 |  |
| 288 | 32 | 6 | Y | 6 | Z12xZ12 | `x^3+y^2+y^10` | `y^6+x+x^11` | 4.00 |  |
| 288 | 50 | 8 | Y | 8 | Z18xZ8 | `1+y^5+x+xy^5` | `1+y+x^5+x^5y` | 11.11 | weight 8 |
| 144 | 54 | 4 | Y | 8 | Z12xZ6 | `1+y^3+x^3+x^3y^3` | `1+y^3+x^3+x^9y^3` | 6.00 | weight 8 |

### Symons+ 2511.13560 (h-covers)

| n | k | d | exact? | w | group | A | B | kd²/n | notes |
|---|---|---|---|---|---|---|---|---|---|
| 216 | 12 | 12 | P | 6 | Z18xZ6 | `x^3+y+y^2` | `y^3+x+x^2` | 8.00 |  |
| 62 | 10 | 6 | P | 6 | Z31 | `1+x^6+x^27` | `1+x^15+x^24` | 5.81 |  |
| 124 | 10 | 10 | P | 6 | Z31xZ2 | `y+x^6+x^27` | `1+x^15+x^24` | 8.06 |  |
| 186 | 10 | 14 | P | 6 | Z31xZ3 | `y+x^6y^2+x^27` | `1+x^15y+x^24` | 10.54 |  |
| 248 | 10 | 18 | <= | 6 | Z31xZ4 | `1+x^6y+x^27` | `y^2+x^15y^3+x^24` | 13.06 |  |
| 72 | 8 | 8 | P | 6 | Z12xZ3 | `x^9+y+y^2` | `1+x^4+x^11` | 7.11 |  |
| 126 | 8 | 10 | P | 6 | Z21xZ3 | `x^9+y+y^2` | `1+x+x^8` | 6.35 |  |
| 144 | 8 | 12 | P | 6 | Z24xZ3 | `1+y+x^21y^2` | `1+x^22+x^17` | 8.00 |  |
| 162 | 8 | 14 | P | 6 | Z27xZ3 | `1+y+x^6y^2` | `1+x^25+x^20` | 9.68 |  |
| 180 | 8 | 16 | <= | 6 | Z15xZ6 | `x^9+y+x^6y^5` | `x^6y^3+x+x^2` | 11.38 |  |
| 198 | 8 | 16 | <= | 6 | Z33xZ3 | `x^12+y+y^2` | `1+x+x^8` | 10.34 |  |
| 28 | 6 | 4 | P | 6 | Z7xZ2 | `y+x^2+x^3` | `1+x^2+x^3` | 3.43 |  |
| 42 | 6 | 6 | P | 6 | Z7xZ3 | `1+x^2+x^3y` | `1+x^2+x^3y^2` | 5.14 |  |
| 56 | 6 | 8 | P | 6 | Z7xZ4 | `1+x^2+x^3y^2` | `1+x^2y+x^3` | 6.86 |  |
| 70 | 6 | 8 | P | 6 | Z7xZ5 | `y+x^2y^4+x^3y` | `y^4+x^2+x^3` | 5.49 |  |
| 84 | 6 | 10 | P | 6 | Z7xZ6 | `1+x^2y^3+x^3y^2` | `1+x^2+x^3y` | 7.14 |  |
| 98 | 6 | 12 | P | 6 | Z7xZ7 | `1+x^2y^2+x^3y` | `1+x^2+x^3y^2` | 8.82 |  |
| 112 | 6 | 12 | P | 6 | Z7xZ8 | `1+x^2y^5+x^3y` | `1+x^2y^6+x^3y^5` | 7.71 |  |
| 126 | 6 | 14 | P | 6 | Z7xZ9 | `1+x^2y^5+x^3y` | `1+x^2+x^3y^2` | 9.33 |  |
| 140 | 6 | 14 | P | 6 | Z7xZ10 | `1+x^2y^5+x^3y^9` | `1+x^2y^6+x^3y^3` | 8.40 |  |
| 154 | 6 | 16 | <= | 6 | Z7xZ11 | `1+x^2y^3+x^3y^4` | `1+x^2y^8+x^3y^7` | 9.97 |  |
| 288 | 20 | 6 | P | 6 | Z12xZ12 | `x^9y^9+x^6y^7+y^8` | `1+x^7y^6+x^2y^9` | 2.50 |  |
| 288 | 16 | 12 | P | 6 | Z12xZ12 | `x^3y^3+x^6y^7+y^8` | `x^6+xy^9+x^2y^9` | 8.00 |  |
| 252 | 20 | 4 | P | 6 | Z6xZ21 | `x^3y^18+y^4+y^11` | `y^18+xy^15+x^2y^9` | 1.27 |  |
| 252 | 14 | 12 | P | 6 | Z6xZ21 | `x^3+y^19+y^8` | `y^9+xy^6+x^2y^15` | 8.00 |  |
| 72 | 14 | 8 | P | 8 | Z6xZ6 | `y^4+x^5y^4+x^3+x^5y^3` | `y^5+x^2y+x^5y^5+x^3y^4` | 12.44 | wt8 |
| 144 | 14 | 14 | P | 8 | Z12xZ6 | `x^6y^4+x^5y^4+x^3+x^11y^3` | `y^5+x^8y+x^5y^5+x^9y^4` | 19.06 | wt8; kd^2/n=19.1 |
| 216 | 14 | 20 | <= | 8 | Z18xZ6 | `y^4+x^11y^4+x^3+x^11y^3` | `x^6y^5+x^2y+x^5y^5+x^15y^4` | 25.93 | wt8 |
| 288 | 14 | 24 | <= | 8 | Z12xZ12 | `x^6y^4+x^11y^4+x^3y^6+x^11y^3` | `y^11+x^2y^7+x^5y^11+x^9y^4` | 28.00 | wt8 |
| 64 | 14 | 8 | P | 8 | Z8xZ4 | `xy^3+1+x^6+x^3y^2` | `x^6y+x^4y+x^3+x^5y` | 14.00 | wt8 |
| 128 | 14 | 12 | P | 8 | Z8xZ8 | `xy^3+y^4+x^6y^4+x^3y^6` | `x^6y+x^4y^5+x^3+x^5y^5` | 15.75 | wt8 |
| 192 | 14 | 16 | <= | 8 | Z8xZ12 | `xy^3+y^8+x^6y^4+x^3y^2` | `x^6y+x^4y^5+x^3+x^5y` | 18.67 | wt8 |
| 256 | 14 | 22 | <= | 8 | Z16xZ8 | `x^9y^3+y^4+x^14+x^3y^6` | `x^14y+x^4y+x^3+x^13y` | 26.47 | wt8 |
| 64 | 12 | 8 | P | 8 | Z8xZ4 | `xy^3+1+x^6y^2+x^3` | `x^6y+x^4y+x^3y^2+x^5y` | 12.00 | wt8 |
| 96 | 12 | 10 | P | 8 | Z8xZ6 | `xy+y^2+x^6y^4+x^3y^4` | `x^6y+x^4y^5+x^3y^2+x^5y` | 12.50 | wt8 |
| 128 | 12 | 14 | ? | 8 | Z8xZ8 | `xy^3+1+x^6y^6+x^3y^2` | `x^6y^7+x^4y^7+x^3+x^5y` | 18.38 | wt8 (table says 14 but kd2/n<=18.4) |
| 160 | 12 | 16 | <= | 8 | Z8xZ10 | `xy^3+y^2+x^6y^8+x^3y^4` | `x^6y^7+x^4y+x^3+x^5y^3` | 19.20 | wt8 |
| 24 | 10 | 4 | P | 8 | Z6xZ2 | `1+x^5+x^3+x^5y` | `y+x^2y+x^5y+x^3` | 6.67 | wt8 |
| 48 | 10 | 6 | P | 8 | Z6xZ4 | `y^2+x^5+x^3+x^5y` | `y^3+x^2y^3+x^5y^3+x^3y^2` | 7.50 | wt8 |
| 72 | 10 | 8 | P | 8 | Z6xZ6 | `y^4+x^5+x^3y^4+x^5y` | `y^5+x^2y^5+x^5y+x^3y^2` | 8.89 | wt8 |
| 96 | 10 | 12 | P | 8 | Z12xZ4 | `y^2+x^11+x^9+x^5y` | `y^3+x^2y^3+x^5y^3+x^3y^2` | 15.00 | wt8 |
| 120 | 10 | 14 | ? | 8 | Z6xZ10 | `y^6+x^5+x^3y^2+x^5y^3` | `y+x^2y+x^5y^9+x^3y^2` | 16.33 | wt8 |
| 144 | 10 | 16 | <= | 8 | Z6xZ12 | `1+x^5y^4+x^3y^4+x^5y^3` | `y^5+x^2y^9+x^5y^11+x^3y^4` | 17.78 | wt8 |
| 56 | 8 | 8 | P | 8 | Z7xZ4 | `x^4y^2+y+x^5y+x^3y^3` | `x^5y^3+x^3y^3+x^4y^3+y^2` | 9.14 | wt8 |
| 84 | 8 | 10 | P | 8 | Z7xZ6 | `x^4y^4+y^5+x^5y^3+x^3y^5` | `x^5y+x^3y^3+x^4y^3+y^2` | 9.52 | wt8 |
| 112 | 8 | 14 | ? | 8 | Z7xZ8 | `x^4y^4+y^3+x^5y^7+x^3y^7` | `x^5y^7+x^3y^5+x^4y^7+1` | 14.00 | wt8 |
| 96 | 20 | 8 | P | 8 | Z12xZ4 | `xy^3+x^4+x^2+x^11y^2` | `x^6y+x^8y+x^11+x^9y` | 13.33 | wt8 |
| 160 | 24 | 8 | P | 8 | Z20xZ4 | `xy^3+x^16+x^14+x^19y^2` | `x^6y+x^8y+x^3+xy` | 9.60 | wt8 |
| 192 | 26 | 12 | P | 8 | Z24xZ4 | `x^9y^3+x^12+x^14+x^23y^2` | `x^10y+x^20y+x^11+xy` | 19.50 | wt8; kd2/n=19.5 |
| 192 | 30 | 8 | P | 8 | Z24xZ4 | `x^21y^3+x^4+x^10+x^15y^2` | `x^22y+x^4y+x^3+x^21y` | 10.00 | wt8 |
| 224 | 32 | 8 | P | 8 | Z4xZ28 | `xy^23+y^16+x^2y^4+x^3y^18` | `x^2y^9+y^21+x^3+xy^9` | 9.14 | wt8 |

### 2609.06572 (wt-8 BB census)

| n | k | d | exact? | w | group | A | B | kd²/n | notes |
|---|---|---|---|---|---|---|---|---|---|
| 144 | 16 | 10 | Y | 8 | Z12xZ6 | `1+x^11+y^5+x^9y^5` | `x^3y^2+x^8y^2+x^4y^3+x^9y^3` | 11.11 |  |
| 144 | 10 | 12 | Y | 8 | Z8xZ9 | `x^4y^2+xy^3+y^5+x^5y^8` | `1+x^4y^2+x^2y^3+x^6y^5` | 10.00 |  |
| 144 | 14 | 10 | Y | 8 | Z12xZ6 | `x^4+x^10y^2+x^3y^4+x^11y^4` | `1+x^7y^2+x^7y^3+x^10y^3` | 9.72 |  |
| 144 | 20 | 8 | Y | 8 | Z12xZ6 | `1+x^7y+x^5y^3+y^4` | `1+x^2+xy+y^4` | 8.89 |  |
| 144 | 6 | 15 | >= | 8 | Z8xZ9 | `1+x^4y^2+y^4+y^7` | `1+x^2y^6+x^5y^7+x^7y^8` | 9.38 |  |
| 72 | 14 | 8 | Y | 8 | Z6xZ6 | `1+x^4y^4+y^5+xy^5` | `1+x^4y+xy^2+x^3y^2` | 12.44 |  |
| 72 | 16 | 6 | Y | 8 | Z6xZ6 | `1+x^4y^2+x^4y^4+x^3y^5` | `x^4+xy+x^3y^2+x^3y^4` | 8.00 |  |
| 72 | 4 | 10 | Y | 8 | Z6xZ6 | `1+x^2+x^5y^4+x^5y^5` | `1+xy^2+xy^4+x^4y^5` | 5.56 |  |
| 200 | 20 | 9 | Y | 8 | Z10xZ10 | `1+x^2y^3+x^4y^3+x^8y^9` | `x^3+x^3y^2+x^7y^3+x^9y^9` | 8.10 |  |

### Lin&Pryadko 2502.19406 T1/T4 (d_S=3 subset)

| n | k | d | exact? | w | group | A | B | kd²/n | notes |
|---|---|---|---|---|---|---|---|---|---|
| 30 | 8 | 4 | P | 6 | Z15 | `1+x+x^4` | `1+x^2+x^8` | 4.27 |  |
| 62 | 10 | 6 | P | 6 | Z31 | `1+x+x^12` | `1+x^3+x^8` | 5.81 |  |
| 72 | 12 | 6 | P | 6 | Z6xZ6 | `1+y+x^3y^2` | `1+xy+x^5y^2` | 6.00 |  |
| 90 | 12 | 6 | P | 6 | Z15xZ3 | `1+x+x^2y` | `1+x^4y+x^11` | 4.80 |  |
| 96 | 12 | 6 | P | 6 | Z12xZ4 | `1+x+x^2y` | `1+xy^3+x^5` | 4.50 |  |
| 120 | 16 | 6 | P | 6 | Z10xZ6 | `1+y+x^5y^2` | `1+x^2y^2+x^4` | 4.80 |  |
| 120 | 16 | 6 | P | 6 | Z30xZ2 | `1+x+x^4y` | `1+x^4+x^16` | 4.80 |  |
| 124 | 20 | 6 | P | 6 | Z62 | `1+x^2+x^24` | `1+x^6+x^16` | 5.81 |  |
| 126 | 12 | 10 | P | 6 | Z63 | `1+x+x^6` | `1+x^11+x^25` | 9.52 | also T1: a=1+t^7+t^8,b=1+t^37+t^43, d_circ=10* |
| 126 | 14 | 6 | P | 6 | Z21xZ3 | `1+xy+x^5y^2` | `1+x+x^5` | 4.00 |  |
| 150 | 16 | 8 | P | 6 | Z5xZ15 | `1+x+x^3y^5` | `1+y+y^4` | 6.83 |  |
| 170 | 16 | 10 | P | 6 | Z85 | `1+x+x^16` | `1+x^4+x^64` | 9.41 |  |
| 186 | 14 | 10 | P | 6 | Z93 | `1+x+x^14` | `1+x^4+x^56` | 7.53 |  |
| 210 | 14 | 12 | P | 6 | Z105 | `1+x+x^12` | `1+x^16+x^87` | 9.60 |  |
| 288 | 16 | 12 | P | 6 | Z12xZ12 | `1+x+x^2y^3` | `1+y+x^3y^8` | 8.00 |  |
| 288 | 20 | 6 | P | 6 | Z12xZ12 | `1+x+x^5y^3` | `1+y+x^3y^5` | 2.50 |  |
| 294 | 18 | 10 | P | 6 | Z7xZ21 | `1+x+x^3y^3` | `1+y+y^17` | 6.12 |  |

### Panteleev&Kalachev 1904.02703 App.A (GB, cyclic Z_l)

| n | k | d | exact? | w | group | A | B | kd²/n | notes |
|---|---|---|---|---|---|---|---|---|---|
| 254 | 28 | 14-20 | range | 10 | Z127 | `1+x^15+x^20+x^28+x^66` | `1+x^58+x^59+x^100+x^121` | 0.00 | w=10 |
| 126 | 28 | 8 | Y | 10 | Z63 | `1+x+x^14+x^16+x^22` | `1+x^3+x^13+x^20+x^42` | 14.22 | w=10 |
| 48 | 6 | 8 | Y | 8 | Z24 | `1+x^2+x^8+x^15` | `1+x^2+x^12+x^17` | 8.00 | w=8 |
| 46 | 2 | 9 | Y | 8 | Z23 | `1+x^5+x^8+x^12` | `1+x+x^5+x^7` | 3.52 | w=8 |
| 180 | 10 | 15-18 | range | 8 | Z90 | `1+x^28+x^80+x^89` | `1+x^2+x^21+x^25` | 0.00 | w=8 |
| 900 | 50 | 15 | Y | 8 | Z450 | `1+x^97+x^372+x^425` | `1+x^50+x^265+x^390` | 12.50 | w=8,n>300 |

### Lin&Pryadko 2306.16400 T1 (2BGA, W=8)

| n | k | d | exact? | w | group | A | B | kd²/n | notes |
|---|---|---|---|---|---|---|---|---|---|
| 72 | 8 | 9 | P | 8 | C36 #2 | `1+r^28` | `1+r^9+r^18+r^12+r^29+r^14` | 9.00 | Wa=2,Wb=6 |
| 72 | 8 | 9 | P | 8 | C9|xC4 (36#1) | `1+r` | `1+s+r^6+s^3r+sr^7+s^3r^5` | 9.00 | nonabelian |
| 80 | 8 | 10 | P | 8 | C5|xC8 (40#1) | `1+sr^4` | `1+r+r^2+s+s^3r+s^2r^6` | 10.00 | nonabelian |
| 96 | 8 | 12 | P | 8 | (C3|xC8)|xC2 (48#10) | `1+sr^2` | `1+r+s^3+s^4+s^2r^5+s^4r^6` | 12.00 | nonabelian |
| 54 | 6 | 9 | P | 8 | C27 | `1+r+r^3+r^7` | `1+r+r^12+r^19` | 9.00 | 4+4 |
| 60 | 6 | 10 | P | 8 | C30 | `1+r^10+r^6+r^13` | `1+r^25+r^16+r^12` | 10.00 |  |
| 70 | 8 | 10 | P | 8 | C35 | `1+r^15+r^16+r^18` | `1+r+r^24+r^27` | 11.43 |  |
| 72 | 8 | 10 | P | 8 | C36 | `1+r^9+r^28+r^31` | `1+r+r^21+r^34` | 11.11 |  |
| 72 | 10 | 9 | P | 8 | C36 | `1+r^9+r^28+r^13` | `1+r+r^3+r^22` | 11.25 |  |
| 72 | 8 | 9 | P | 8 | C9|xC4 | `1+s+r+sr^6` | `1+s^2r+s^2r^6+r^2` | 9.00 |  |
| 80 | 8 | 10 | P | 8 | C5|xC8 | `1+r+s+s^3r^5` | `1+r^2+sr^4+s^3r^2` | 10.00 |  |
| 96 | 8 | 12 | P | 8 | C3|xC16 | `1+r+s+r^14` | `1+r^2+sr^4+r^11` | 12.00 |  |
| 80 | 9 | 9 | P | 8 | (C10xC2)|xC2 | `1+sr^5+r^5+sr^6` | `1+s^2+r+s^2r^3` | 9.11 |  |
| 84 | 10 | 9 | P | 8 | C7xS3 (42#3; paper prints n=82) | `1+r^7+r^8+sr^10` | `1+s+r^5+s^2r^13` | 9.64 |  |
| 96 | 10 | 12 | P | 8 | C12|xC4 (48#13) | `1+s+r^9+sr` | `1+s^2r^9+r^7+r^2` | 15.00 |  |
| 96 | 11 | 9 | P | 8 | C24|xC2 | `1+s+r^9+sr^13` | `1+r^9+sr^18+r^7` | 9.28 |  |
| 96 | 12 | 10 | P | 8 | C2x(C3|xC8) | `1+r+s^3r^2+s^2r^3` | `1+r+s^4r^6+s^5r^3` | 12.50 |  |

### Liang+ 2503.03827 GT T1-4

| n | k | d | exact? | w | group | A | B | kd²/n | notes |
|---|---|---|---|---|---|---|---|---|---|
| 12 | 4 | 2 | Y | 6 | Z^2/<(0,3),(2,1)> (order 6) | `1+x+xy` | `1+y+xy` | 1.33 | twisted torus a1,a2 |
| 14 | 6 | 2 | Y | 6 | Z^2/<(0,7),(1,2)> (order 7) | `1+x+y` | `1+y+x` | 1.71 | twisted torus a1,a2 |
| 18 | 4 | 4 | Y | 6 | Z^2/<(0,3),(3,0)> (order 9) | `1+x+xy` | `1+y+xy` | 3.56 | twisted torus a1,a2 |
| 24 | 4 | 4 | Y | 6 | Z^2/<(0,3),(4,2)> (order 12) | `1+x+xy` | `1+y+xy` | 2.67 | twisted torus a1,a2 |
| 28 | 6 | 4 | Y | 6 | Z^2/<(0,7),(2,3)> (order 14) | `1+x+x^-1y` | `1+y+xy` | 3.43 | twisted torus a1,a2 |
| 30 | 4 | 6 | Y | 6 | Z^2/<(0,3),(5,1)> (order 15) | `1+x+x^2` | `1+y+x^2` | 4.80 | twisted torus a1,a2 |
| 36 | 4 | 6 | Y | 6 | Z^2/<(0,9),(2,4)> (order 18) | `1+x+x^-1` | `1+y+y^-1` | 4.00 | twisted torus a1,a2 |
| 42 | 6 | 6 | Y | 6 | Z^2/<(0,7),(3,2)> (order 21) | `1+x+xy` | `1+y+xy^-1` | 5.14 | twisted torus a1,a2 |
| 48 | 4 | 8 | Y | 6 | Z^2/<(0,3),(8,1)> (order 24) | `1+x+x^2` | `1+y+x^2` | 5.33 | twisted torus a1,a2 |
| 54 | 8 | 6 | Y | 6 | Z^2/<(0,3),(9,0)> (order 27) | `1+x+x^-1` | `1+y+x^3y^2` | 5.33 | twisted torus a1,a2 |
| 56 | 6 | 8 | Y | 6 | Z^2/<(0,7),(4,3)> (order 28) | `1+x+y^-2` | `1+y+x^-2` | 6.86 | twisted torus a1,a2 |
| 60 | 8 | 6 | Y | 6 | Z^2/<(0,10),(3,3)> (order 30) | `1+x+y^-2` | `1+y+x^2` | 4.80 | twisted torus a1,a2 |
| 62 | 10 | 6 | Y | 6 | Z^2/<(0,31),(1,13)> (order 31) | `1+x+x^-1y` | `1+y+x^-1y^-1` | 5.81 | twisted torus a1,a2 |
| 66 | 4 | 10 | Y | 6 | Z^2/<(0,3),(11,2)> (order 33) | `1+x+x^-2y^-1` | `1+y+x^2y` | 6.06 | twisted torus a1,a2 |
| 70 | 6 | 8 | Y | 6 | Z^2/<(0,7),(5,1)> (order 35) | `1+x+xy` | `1+y+xy^-1` | 5.49 | twisted torus a1,a2 |
| 72 | 8 | 8 | Y | 6 | Z^2/<(0,12),(3,3)> (order 36) | `1+x+x^-1y^3` | `1+y+x^3y^-1` | 7.11 | twisted torus a1,a2 |
| 78 | 4 | 10 | Y | 6 | Z^2/<(0,3),(13,1)> (order 39) | `1+x+x^-2y^-1` | `1+y+x^2y` | 5.13 | twisted torus a1,a2 |
| 84 | 6 | 10 | Y | 6 | Z^2/<(0,14),(3,-6)> (order 42) | `1+x+x^-2` | `1+y+x^-2y^2` | 7.14 | twisted torus a1,a2 |
| 90 | 8 | 10 | Y | 6 | Z^2/<(0,15),(3,-6)> (order 45) | `1+x+x^-1y^-3` | `1+y+x^3y^-1` | 8.89 | twisted torus a1,a2 |
| 96 | 4 | 12 | Y | 6 | Z^2/<(0,12),(4,2)> (order 48) | `1+x+x^-2y` | `1+y+xy^-2` | 6.00 | twisted torus a1,a2 |
| 98 | 6 | 12 | Y | 6 | Z^2/<(0,7),(7,0)> (order 49) | `1+x+x^-1y^2` | `1+y+x^-2y^-1` | 8.82 | twisted torus a1,a2 |
| 102 | 4 | 12 | Y | 6 | Z^2/<(0,3),(17,2)> (order 51) | `1+x+x^-3y` | `1+y+x^3y^2` | 5.65 | twisted torus a1,a2 |
| 108 | 8 | 10 | Y | 6 | Z^2/<(0,9),(6,0)> (order 54) | `1+x+x^-1y^-3` | `1+y+x^3y^-1` | 7.41 | twisted torus a1,a2 |
| 112 | 6 | 12 | Y | 6 | Z^2/<(0,7),(8,2)> (order 56) | `1+x+x^-1y^2` | `1+y+x^-2y^-1` | 7.71 | twisted torus a1,a2 |
| 114 | 4 | 14 | Y | 6 | Z^2/<(0,3),(19,1)> (order 57) | `1+x+x^-3y` | `1+y+x^-5` | 6.88 | twisted torus a1,a2 |
| 120 | 8 | 12 | Y | 6 | Z^2/<(0,10),(6,4)> (order 60) | `1+x+x^-2y` | `1+y+xy^2` | 9.60 | twisted torus a1,a2 |
| 124 | 10 | 10 | Y | 6 | Z^2/<(0,31),(2,-12)> (order 62) | `1+x+x^-1y^2` | `1+y+x^-2y^-1` | 8.06 | twisted torus a1,a2 |
| 126 | 12 | 10 | Y | 6 | Z^2/<(0,9),(7,3)> (order 63) | `1+x+x^-1y^-2` | `1+y+xy^-1` | 9.52 | twisted torus a1,a2 |
| 132 | 4 | 14 | Y | 6 | Z^2/<(0,33),(2,-7)> (order 66) | `1+x+y^-2` | `1+y+x^-2` | 5.94 | twisted torus a1,a2 |
| 138 | 4 | 14 | Y | 6 | Z^2/<(0,3),(23,2)> (order 69) | `1+x+x^-3y` | `1+y+x^3y^2` | 5.68 | twisted torus a1,a2 |
| 140 | 6 | 14 | Y | 6 | Z^2/<(0,7),(10,1)> (order 70) | `1+x+x^-2` | `1+y+x^-2y^2` | 8.40 | twisted torus a1,a2 |
| 144 | 12 | 12 | Y | 6 | Z^2/<(0,12),(6,0)> (order 72) | `1+x+x^-1y^-3` | `1+y+x^3y^-1` | 12.00 | twisted torus a1,a2 |
| 146 | 18 | 4 | Y | 6 | Z^2/<(0,73),(1,16)> (order 73) | `1+x+y^2` | `1+y+x^-4y` | 1.97 | twisted torus a1,a2 |
| 150 | 8 | 12 | Y | 6 | Z^2/<(0,25),(3,7)> (order 75) | `1+x+x^-2y` | `1+y+xy^2` | 7.68 | twisted torus a1,a2 |
| 154 | 6 | 16 | Y | 6 | Z^2/<(0,77),(1,16)> (order 77) | `1+x+x^-1y^2` | `1+y+y^-4` | 9.97 | twisted torus a1,a2 |
| 156 | 4 | 16 | Y | 6 | Z^2/<(0,39),(2,-11)> (order 78) | `1+x+x^-2y` | `1+y+xy^-2` | 6.56 | twisted torus a1,a2 |
| 162 | 8 | 14 | Y | 6 | Z^2/<(0,9),(9,-3)> (order 81) | `1+x+x^-1y^-3` | `1+y+x^3y^-1` | 9.68 | twisted torus a1,a2 |
| 168 | 8 | 14 | Y | 6 | Z^2/<(0,42),(2,-16)> (order 84) | `1+x+x^-1y^-3` | `1+y+x^3y^-1` | 9.33 | twisted torus a1,a2 |
| 170 | 16 | 10 | Y | 6 | Z^2/<(0,17),(5,-7)> (order 85) | `1+x+y^-4` | `1+y+x^4` | 9.41 | twisted torus a1,a2 |
| 174 | 4 | 18 | Y | 6 | Z^2/<(0,3),(29,1)> (order 87) | `1+x+x^-8y` | `1+y+x^6y^2` | 7.45 | twisted torus a1,a2 |
| 180 | 8 | 16 | Y | 6 | Z^2/<(0,15),(6,6)> (order 90) | `1+x+x^-1y^-3` | `1+y+x^3y^-1` | 11.38 | twisted torus a1,a2 |
| 182 | 6 | 18 | Y | 6 | Z^2/<(0,7),(13,1)> (order 91) | `1+x+x^2y^3` | `1+y+x^4y` | 10.68 | twisted torus a1,a2 |
| 186 | 10 | 14 | Y | 6 | Z^2/<(0,31),(3,7)> (order 93) | `1+x+x^2y^3` | `1+y+x^2y^-2` | 10.54 | twisted torus a1,a2 |
| 192 | 8 | 16 | Y | 6 | Z^2/<(0,12),(8,2)> (order 96) | `1+x+x^-1y^3` | `1+y+x^3y^-1` | 10.67 | twisted torus a1,a2 |
| 196 | 6 | 18 | Y | 6 | Z^2/<(0,49),(2,-10)> (order 98) | `1+x+x^-1y^2` | `1+y+x^-2y^-1` | 9.92 | twisted torus a1,a2 |
| 198 | 8 | 16 | Y | 6 | Z^2/<(0,33),(3,9)> (order 99) | `1+x+x^-4` | `1+y+x^-3y^2` | 10.34 | twisted torus a1,a2 |
| 204 | 4 | 20 | Y | 6 | Z^2/<(0,51),(2,14)> (order 102) | `1+x+x^-3y` | `1+y+x^-1y^-2` | 7.84 | twisted torus a1,a2 |
| 210 | 10 | 16 | Y | 6 | Z^2/<(0,21),(5,10)> (order 105) | `1+x+x^-3y^2` | `1+y+x^-3y^-1` | 12.19 | twisted torus a1,a2 |
| 216 | 8 | 18 | Y | 6 | Z^2/<(0,54),(2,16)> (order 108) | `1+x+x^-2y^-5` | `1+y+x^-1y^-3` | 12.00 | twisted torus a1,a2 |
| 222 | 4 | 20 | Y | 6 | Z^2/<(0,3),(37,2)> (order 111) | `1+x+x^-6y^-1` | `1+y+x^5` | 7.21 | twisted torus a1,a2 |
| 224 | 6 | 20 | Y | 6 | Z^2/<(0,28),(4,-6)> (order 112) | `1+x+x^-3y^2` | `1+y+x^-3y^-1` | 10.71 | twisted torus a1,a2 |
| 228 | 4 | 20 | Y | 6 | Z^2/<(0,57),(2,10)> (order 114) | `1+x+x^-2y` | `1+y+xy^-2` | 7.02 | twisted torus a1,a2 |
| 234 | 8 | 18 | Y | 6 | Z^2/<(0,39),(3,-9)> (order 117) | `1+x+x^-1y^-3` | `1+y+x^3y^-1` | 11.08 | twisted torus a1,a2 |
| 238 | 6 | 20 | Y | 6 | Z^2/<(0,7),(17,1)> (order 119) | `1+x+x^-4` | `1+y+x^-3y^2` | 10.08 | twisted torus a1,a2 |
| 240 | 8 | 18 | Y | 6 | Z^2/<(0,10),(12,3)> (order 120) | `1+x+x^-2y` | `1+y+xy^2` | 10.80 | twisted torus a1,a2 |
| 246 | 4 | 22 | <= | 6 | Z^2/<(0,123),(1,22)> (order 123) | `1+x+x^3y` | `1+y+x^2y^-2` | 7.87 | twisted torus a1,a2 |
| 248 | 10 | 18 | Y | 6 | Z^2/<(0,62),(2,25)> (order 124) | `1+x+x^-2y` | `1+y+x^-3y^-2` | 13.06 | twisted torus a1,a2 |
| 252 | 12 | 16 | Y | 6 | Z^2/<(0,18),(7,7)> (order 126) | `1+x+x^-3y^-1` | `1+y+x^2y^-2` | 12.19 | twisted torus a1,a2 |
| 254 | 14 | 16 | Y | 6 | Z^2/<(0,127),(1,25)> (order 127) | `1+x+x^-1y^-3` | `1+y+y^-6` | 14.11 | twisted torus a1,a2 |
| 258 | 4 | 22 | <= | 6 | Z^2/<(0,3),(43,1)> (order 129) | `1+x+x^-8y^-1` | `1+y+x^5y` | 7.50 | twisted torus a1,a2 |
| 264 | 8 | 20 | Y | 6 | Z^2/<(0,66),(2,28)> (order 132) | `1+x+xy^-5` | `1+y+xy^4` | 12.12 | twisted torus a1,a2 |
| 266 | 6 | 22 | <= | 6 | Z^2/<(0,7),(19,2)> (order 133) | `1+x+x^-1y^-1` | `1+y+x^5` | 10.92 | twisted torus a1,a2 |
| 270 | 8 | 20 | Y | 6 | Z^2/<(0,15),(9,6)> (order 135) | `1+x+x^-1y^-3` | `1+y+x^3y^-1` | 11.85 | twisted torus a1,a2 |
| 276 | 4 | 24 | <= | 6 | Z^2/<(0,6),(23,5)> (order 138) | `1+x+x^-3y` | `1+y+x^3y^2` | 8.35 | twisted torus a1,a2 |
| 280 | 6 | 22 | <= | 6 | Z^2/<(0,28),(5,12)> (order 140) | `1+x+xy^3` | `1+y+x^2y^-2` | 10.37 | twisted torus a1,a2 |
| 282 | 4 | 24 | <= | 6 | Z^2/<(0,141),(1,7)> (order 141) | `1+x+x^-1y^3` | `1+y+x^3y^-1` | 8.17 | twisted torus a1,a2 |
| 288 | 16 | 12 | Y | 6 | Z^2/<(0,12),(12,0)> (order 144) | `1+x+x^-1y^3` | `1+y+x^3y^-1` | 8.00 | twisted torus a1,a2 |
| 288 | 12 | 18 | Y | 6 | Z^2/<(0,12),(12,0)> (order 144) | `1+x+x^-1y^-3` | `1+y+x^3y^-1` | 13.50 | twisted torus a1,a2 |
| 292 | 18 | 8 | Y | 6 | Z^2/<(0,73),(2,32)> (order 146) | `1+x+y^2` | `1+y+x^-4y` | 3.95 | twisted torus a1,a2 |
| 294 | 10 | 20 | Y | 6 | Z^2/<(0,21),(7,7)> (order 147) | `1+x+x^-3y` | `1+y+xy^-3` | 13.61 | twisted torus a1,a2 |
| 300 | 8 | 22 | <= | 6 | Z^2/<(0,75),(2,26)> (order 150) | `1+x+x^-1y^-4` | `1+y+x^-3y^3` | 12.91 | twisted torus a1,a2 |
| 310 | 10 | 22 | <= | 6 | Z^2/<(0,31),(5,11)> (order 155) | `1+x+x^3y^2` | `1+y+x^-4y^4` | 15.61 | twisted torus a1,a2 |
| 340 | 16 | 18 | Y | 6 | Z^2/<(0,34),(5,-7)> (order 170) | `1+x+y^-4` | `1+y+x^4` | 15.25 | twisted torus a1,a2 |
| 360 | 12 | 24 | <= | 6 | Z^2/<(0,30),(6,6)> (order 180) | `1+x+x^-1y^3` | `1+y+x^3y^-1` | 19.20 | twisted torus a1,a2 |
| 384 | 12 | 24 | <= | 6 | Z^2/<(0,48),(4,20)> (order 192) | `1+x+x^-4y^-3` | `1+y+x^3y^-1` | 18.00 | twisted torus a1,a2 |

### Liang+ 2503.03827 GB T5-8

| n | k | d | exact? | w | group | A | B | kd²/n | notes |
|---|---|---|---|---|---|---|---|---|---|
| 30 | 8 | 4 | Y | 6 | Z_15 | `1+y^2+y^8` | `1+y+y^4` | 4.27 | 1D GB (cyclic) |
| 42 | 10 | 4 | Y | 6 | Z_21 | `1+y^2+y^10` | `1+y+y^5` | 3.81 | 1D GB (cyclic) |
| 54 | 4 | 8 | Y | 6 | Z_27 | `1+y^5+y^7` | `1+y+y^5` | 4.74 | 1D GB (cyclic) |
| 72 | 4 | 10 | Y | 6 | Z_36 | `1+y^2+y^7` | `1+y+y^11` | 5.56 | 1D GB (cyclic) |
| 84 | 10 | 6 | Y | 6 | Z_42 | `1+y^11+y^13` | `1+y+y^5` | 4.29 | 1D GB (cyclic) |
| 90 | 8 | 8 | Y | 6 | Z_45 | `1+y^2+y^9` | `1+y+y^12` | 5.69 | 1D GB (cyclic) |
| 108 | 4 | 12 | Y | 6 | Z_54 | `1+y^8+y^10` | `1+y+y^8` | 5.33 | 1D GB (cyclic) |
| 144 | 4 | 16 | Y | 6 | Z_72 | `1+y^23+y^28` | `1+y+y^20` | 7.11 | 1D GB (cyclic) |
| 162 | 4 | 16 | Y | 6 | Z_81 | `1+y^7+y^11` | `1+y+y^14` | 6.32 | 1D GB (cyclic) |
| 168 | 10 | 12 | Y | 6 | Z_84 | `1+y^11+y^19` | `1+y+y^17` | 8.57 | 1D GB (cyclic) |
| 186 | 14 | 10 | Y | 6 | Z_93 | `1+y^8+y^19` | `1+y+y^14` | 7.53 | 1D GB (cyclic) |
| 192 | 4 | 18 | Y | 6 | Z_96 | `1+y^11+y^16` | `1+y+y^14` | 6.75 | 1D GB (cyclic) |
| 198 | 4 | 18 | Y | 6 | Z_99 | `1+y^11+y^16` | `1+y+y^14` | 6.55 | 1D GB (cyclic) |
| 210 | 14 | 12 | Y | 6 | Z_105 | `1+y^11+y^27` | `1+y+y^19` | 9.60 | 1D GB (cyclic) |
| 216 | 4 | 20 | Y | 6 | Z_108 | `1+y^14+y^22` | `1+y+y^20` | 7.41 | 1D GB (cyclic) |
| 234 | 4 | 22 | <= | 6 | Z_117 | `1+y^13+y^29` | `1+y+y^20` | 8.27 | 1D GB (cyclic) |
| 264 | 4 | 22 | <= | 6 | Z_132 | `1+y^13+y^20` | `1+y+y^17` | 7.33 | 1D GB (cyclic) |
| 288 | 4 | 24 | <= | 6 | Z_144 | `1+y^20+y^25` | `1+y+y^14` | 8.00 | 1D GB (cyclic) |
| 62 | 10 | 6 | Y | 6 | Z_31 | `1+y^3+y^8` | `1+y+y^12` | 5.81 | 1D GB (cyclic) |
| 66 | 4 | 10 | Y | 6 | Z_33 | `1+y^2+y^7` | `1+y+y^11` | 6.06 | 1D GB (cyclic) |
| 126 | 12 | 10 | Y | 6 | Z_63 | `1+y^12+y^23` | `1+y+y^8` | 9.52 | 1D GB (cyclic) |
| 140 | 6 | 14 | Y | 6 | Z_70 | `1+y^10+y^16` | `1+y+y^12` | 8.40 | 1D GB (cyclic) |
| 154 | 6 | 16 | Y | 6 | Z_77 | `1+y^4+y^34` | `1+y+y^19` | 9.97 | 1D GB (cyclic) |
| 170 | 16 | 10 | Y | 6 | Z_85 | `1+y^21+y^25` | `1+y+y^16` | 9.41 | 1D GB (cyclic) |
| 180 | 8 | 16 | Y | 6 | Z_90 | `1+y^8+y^47` | `1+y+y^34` | 11.38 | 1D GB (cyclic) |
| 182 | 6 | 18 | Y | 6 | Z_91 | `1+y^9+y^13` | `1+y+y^38` | 10.68 | 1D GB (cyclic) |
| 196 | 6 | 18 | Y | 6 | Z_98 | `1+y^12+y^22` | `1+y+y^19` | 9.92 | 1D GB (cyclic) |
| 204 | 4 | 20 | Y | 6 | Z_102 | `1+y^16+y^35` | `1+y+y^11` | 7.84 | 1D GB (cyclic) |
| 224 | 6 | 20 | Y | 6 | Z_112 | `1+y^3+y^22` | `1+y+y^31` | 10.71 | 1D GB (cyclic) |
| 238 | 6 | 20 | Y | 6 | Z_119 | `1+y^9+y^20` | `1+y+y^24` | 10.08 | 1D GB (cyclic) |
| 240 | 8 | 18 | Y | 6 | Z_120 | `1+y^13+y^21` | `1+y+y^19` | 10.80 | 1D GB (cyclic) |
| 248 | 10 | 18 | Y | 6 | Z_124 | `1+y^17+y^27` | `1+y+y^13` | 13.06 | 1D GB (cyclic) |
| 252 | 12 | 16 | Y | 6 | Z_126 | `1+y^25+y^30` | `1+y+y^8` | 12.19 | 1D GB (cyclic) |
| 254 | 14 | 16 | Y | 6 | Z_127 | `1+y^10+y^37` | `1+y+y^31` | 14.11 | 1D GB (cyclic) |
| 270 | 8 | 20 | Y | 6 | Z_135 | `1+y^6+y^23` | `1+y+y^27` | 11.85 | 1D GB (cyclic) |
| 294 | 10 | 20 | Y | 6 | Z_147 | `1+y^19+y^29` | `1+y+y^26` | 13.61 | 1D GB (cyclic) |
| 372 | 14 | 20 | Y | 6 | Z_186 | `1+y^26+y^34` | `1+y+y^20` | 15.05 | 1D GB (cyclic) |

### Liang&Chen 2510.05211 self-dual BB

| n | k | d | exact? | w | group | A | B | kd²/n | notes |
|---|---|---|---|---|---|---|---|---|---|
| 16 | 4 | 4 | Y | 8 | Z^2/<(0,4),(2,2)> | `1+x+y+y^-1` | `f^T (self-dual)` | 4.00 | weight 8 |
| 24 | 8 | 4 | Y | 8 | Z^2/<(0,6),(2,2)> | `1+x+x^-1y+xy` | `f^T (self-dual)` | 5.33 | weight 8 |
| 32 | 12 | 4 | Y | 8 | Z^2/<(0,4),(4,2)> | `1+x+x^2y+x^-1y` | `f^T (self-dual)` | 6.00 | weight 8 |
| 40 | 6 | 6 | Y | 8 | Z^2/<(0,4),(5,1)> | `1+x+xy^-1+x^-1` | `f^T (self-dual)` | 5.40 | weight 8 |
| 48 | 16 | 4 | Y | 8 | Z^2/<(0,4),(6,0)> | `1+x+y^2+x^-2` | `f^T (self-dual)` | 5.33 | weight 8 |
| 54 | 10 | 6 | Y | 8 | Z^2/<(0,9),(3,3)> | `1+x+x^-1y^-1+x^-1y` | `f^T (self-dual)` | 6.67 | weight 8 |
| 56 | 6 | 8 | Y | 8 | Z^2/<(0,7),(4,3)> | `1+x+x^2y+x^-1y` | `f^T (self-dual)` | 6.86 | weight 8 |
| 64 | 8 | 8 | Y | 8 | Z^2/<(0,8),(4,4)> | `1+x+y+y^-1` | `f^T (self-dual)` | 8.00 | weight 8 |
| 78 | 6 | 10 | Y | 8 | Z^2/<(0,13),(3,-3)> | `1+x+y^2+x^-2` | `f^T (self-dual)` | 7.69 | weight 8 |
| 80 | 10 | 8 | Y | 8 | Z^2/<(0,8),(5,4)> | `1+x+x^2y+x^-1y` | `f^T (self-dual)` | 8.00 | weight 8 |
| 90 | 18 | 6 | Y | 8 | Z^2/<(0,15),(3,6)> | `1+x+x^3y+x^-1y` | `f^T (self-dual)` | 7.20 | weight 8 |
| 96 | 12 | 8 | Y | 8 | Z^2/<(0,12),(4,4)> | `1+x+x^-1y^-1+x^-1y` | `f^T (self-dual)` | 8.00 | weight 8 |
| 104 | 6 | 12 | Y | 8 | Z^2/<(0,26),(2,8)> | `1+x+y+x^-1y^-1` | `f^T (self-dual)` | 8.31 | weight 8 |
| 108 | 20 | 6 | Y | 8 | Z^2/<(0,18),(3,6)> | `1+x+xy^2+x^-1y^2` | `f^T (self-dual)` | 6.67 | weight 8 |
| 120 | 8 | 12 | Y | 8 | Z^2/<(0,6),(10,0)> | `1+x+y+x^-2y^-2` | `f^T (self-dual)` | 9.60 | weight 8 |
| 126 | 22 | 6 | Y | 8 | Z^2/<(0,21),(3,6)> | `1+x+x^-2y+x^-1y^-2` | `f^T (self-dual)` | 6.29 | weight 8 |
| 128 | 16 | 8 | Y | 8 | Z^2/<(0,8),(8,4)> | `1+x+x^2y+x^-1y` | `f^T (self-dual)` | 8.00 | weight 8 |
| 132 | 8 | 12 | Y | 8 | Z^2/<(0,33),(2,11)> | `1+x+y^2+x^-1y^-1` | `f^T (self-dual)` | 8.73 | weight 8 |
| 136 | 6 | 14 | Y | 8 | Z^2/<(0,17),(4,4)> | `1+x+x^2y+x^-1y^2` | `f^T (self-dual)` | 8.65 | weight 8 |
| 152 | 6 | 16 | Y | 8 | Z^2/<(0,19),(4,6)> | `1+x+x^2y+x^-1y^2` | `f^T (self-dual)` | 10.11 | weight 8 |
| 160 | 8 | 16 | Y | 8 | Z^2/<(0,10),(8,0)> | `1+x+x^2y^2+x^-1y` | `f^T (self-dual)` | 12.80 | weight 8 |
| 176 | 8 | 16 | Y | 8 | Z^2/<(0,44),(2,30)> | `1+x+y+y^-2` | `f^T (self-dual)` | 11.64 | weight 8 |
| 192 | 12 | 12 | Y | 8 | Z^2/<(0,48),(2,21)> | `1+x+y^2+x^-1y^-1` | `f^T (self-dual)` | 9.00 | weight 8 |
| 200 | 12 | 12 | Y | 8 | Z^2/<(0,50),(2,14)> | `1+x+x^-1y+xy^2` | `f^T (self-dual)` | 8.64 | weight 8 |

### Other sources checked
* Scruby/Hillmann/Roffe 2406.14445 (now v2, "quantum radial codes"): lifted-product radial codes [[90,8,10]] (r,s)=(3,5), weight 6, and [[352,18,20]] (r,s)=(4,11), weight 8. Z word error at p=1e-3: ≈6e-6 and ≈3e-8. Base matrices (exponents mod 5 / mod 11): A1=[[3,2,1],[4,1,4],[1,2,3]]_5, A2=[[3,3,0],[1,0,1],[4,2,0]]_5; for 352: A1=[[10,10,1,6],[4,7,5,2],[8,10,6,9],[1,6,0,6]]_11, A2=[[9,5,8,3],[5,4,1,0],[0,4,6,10],[2,8,4,2]]_11. Also cites trivariate tricycle codes (Jacob et al. 2025) [[72,6,6]], [[126,6,8]], [[288,6,10]], [[180,12,8]], [[432,12,12]] (weights 6–9) — not BB.
* Tile codes (Steffan+ 2504.09171): planar with boundary, NOT BB: [[288,8,12]] w6, [[288,8,14]] w8, [[288,18,13]] w8.
* Olle+ 2502.14372 (RL): weight-reduction of HGP codes, no BB tables.
* Malcolm+ 2502.07150: SHYPS subsystem codes [[49,9,4]], [[225,16,8]], [[961,25,16]] (weight-3 gauge), not BB. (Corrected 2026-10-05: the [[81,9,3]] and [[784,16,7]] quoted here before are the paper's surface-code baselines.)
* Lin&Pryadko 2305.06890 (abelian/non-abelian two-block): theory, no code table in HTML. Wang&Pryadko 2203.17216 (GB distance bounds): no tables (figures only). Gong+ 2403.18901: uses only the IBM codes.
* 2606.02418 also lists non-CSS "perturbed BB" codes (e.g. [[144,12,12]], [[108,8,10]], [[72,4,8]], [[360,12,<=24]]) and cites Khesin mirror code non-CSS [[60,4,10]].

## Merged Pareto frontier — weight-6 codes, n ≤ 300

For each n: all (k,d) not dominated by another published weight-6 code at the same n (k'≥k and d'≥d). d marked * if only upper bound / ±2 / randomized. Sources abbreviated: IBM=2308.07915, GT=2503.03827 (twisted torus), GB1=2503.03827 1D GB tables, WM=2408.10001, SY=2511.13560, LP=2502.19406, TV=2603.17703, ES=2407.03973, PK=2502.17052, BE=2404.17676, LLM=2606.02418, VO=2406.19151.

| n | frontier (k,d) [sources] | best kd²/n |
|---|---|---|
| 12 | [[12,4,2]] GT | 1.33 |
| 14 | [[14,6,2]] GT | 1.71 |
| 18 | [[18,4,4]] GT/WM; [[18,8,2*]] PK | 3.56 |
| 24 | [[24,4,4]] GT | 2.67 |
| 28 | [[28,6,4]] GT/SY/WM | 3.43 |
| 30 | [[30,4,6]] GT/WM; [[30,8,4]] GB1/LP | 4.80 |
| 36 | [[36,4,6]] GT/WM; [[36,8,4]] WM | 4.00 |
| 40 | [[40,4,6]] VO | 3.60 |
| 42 | [[42,6,6]] GT/SY/WM; [[42,10,4]] GB1/WM | 5.14 |
| 48 | [[48,4,8]] GT/WM; [[48,6,6]] VO | 5.33 |
| 54 | [[54,4,8]] GB1/WM; [[54,8,6]] GT/TV/WM | 5.33 |
| 56 | [[56,6,8]] GT/SY/WM | 6.86 |
| 60 | [[60,8,6]] GT; [[60,16,4]] WM | 4.80 |
| 62 | [[62,10,6]] GB1/GT/LP/SY | 5.81 |
| 66 | [[66,4,10]] GB1/GT/WM | 6.06 |
| 70 | [[70,6,8]] GT/SY/WM | 5.49 |
| 72 | [[72,4,10]] GB1; [[72,8,8]] GT/SY; [[72,12,6]] IBM/LP | 7.11 |
| 78 | [[78,4,10]] GT | 5.13 |
| 84 | [[84,6,10]] GT/SY/TV/WM; [[84,10,6]] GB1; [[84,12,4*]] PK | 7.14 |
| 90 | [[90,4,12]] WM; [[90,8,10]] ES/GT/IBM; [[90,12,6]] LP | 8.89 |
| 96 | [[96,4,12]] GT; [[96,12,6]] LP | 6.00 |
| 98 | [[98,6,12]] ES/GT/SY/WM | 8.82 |
| 102 | [[102,4,12]] GT | 5.65 |
| 108 | [[108,4,12]] GB1; [[108,8,10]] GT/IBM; [[108,16,6]] ES | 7.41 |
| 112 | [[112,6,12]] GT/SY/WM | 7.71 |
| 114 | [[114,4,14]] GT | 6.88 |
| 120 | [[120,8,12]] GT; [[120,16,6]] LP | 9.60 |
| 124 | [[124,10,10]] GT/SY; [[124,20,6]] LP | 8.06 |
| 126 | [[126,6,14]] SY/WM; [[126,12,10]] GB1/GT/LP/WM; [[126,14,6]] LP | 9.52 |
| 132 | [[132,4,14]] GT/WM | 5.94 |
| 138 | [[138,4,14]] GT | 5.68 |
| 140 | [[140,6,14]] GB1/GT/SY/TV | 8.40 |
| 144 | [[144,4,16]] GB1; [[144,12,12]] GT/IBM; [[144,24,6]] LLM | 12.00 |
| 146 | [[146,18,4]] GT | 1.97 |
| 150 | [[150,8,12]] GT; [[150,16,8]] LP/WM | 7.68 |
| 154 | [[154,6,16]] GB1/GT/SY/WM | 9.97 |
| 156 | [[156,4,16]] GT | 6.56 |
| 162 | [[162,4,16]] ES/GB1; [[162,8,14]] GT/SY/WM; [[162,12,8]] ES; [[162,24,6]] ES | 9.68 |
| 168 | [[168,8,14]] GT; [[168,10,12]] GB1 | 9.33 |
| 170 | [[170,16,10]] GB1/GT/LP | 9.41 |
| 174 | [[174,4,18]] GT | 7.45 |
| 180 | [[180,8,16]] GB1/GT/SY/WM | 11.38 |
| 182 | [[182,6,18]] GB1/GT | 10.68 |
| 186 | [[186,10,14]] GT/SY; [[186,14,10]] GB1/LP | 10.54 |
| 192 | [[192,4,18]] GB1; [[192,8,16]] GT | 10.67 |
| 196 | [[196,6,18]] GB1/GT; [[196,18,8]] WM | 9.92 |
| 198 | [[198,4,18]] GB1; [[198,8,16]] GT/SY | 10.34 |
| 204 | [[204,4,20]] GB1/GT | 7.84 |
| 210 | [[210,10,16]] GT; [[210,14,12]] GB1/LP | 12.19 |
| 216 | [[216,4,20]] GB1; [[216,8,18]] GT; [[216,12,12]] SY | 12.00 |
| 222 | [[222,4,20]] GT | 7.21 |
| 224 | [[224,6,20]] GB1/GT | 10.71 |
| 228 | [[228,4,20]] GT | 7.02 |
| 234 | [[234,4,22*]] GB1; [[234,8,18]] GT | 11.08 |
| 238 | [[238,6,20]] GB1/GT | 10.08 |
| 240 | [[240,8,18]] GB1/GT | 10.80 |
| 246 | [[246,4,22*]] GT | 7.87 |
| 248 | [[248,10,18]] GB1/GT/SY | 13.06 |
| 252 | [[252,12,16]] GB1/GT; [[252,14,12]] SY; [[252,20,4]] SY | 12.19 |
| 254 | [[254,14,16]] GB1/GT | 14.11 |
| 258 | [[258,4,22*]] GT | 7.50 |
| 264 | [[264,4,22*]] GB1; [[264,8,20]] GT | 12.12 |
| 266 | [[266,6,22*]] GT | 10.92 |
| 270 | [[270,8,20]] GB1/GT | 11.85 |
| 276 | [[276,4,24*]] GT | 8.35 |
| 280 | [[280,6,22*]] GT | 10.37 |
| 282 | [[282,4,24*]] GT | 8.17 |
| 288 | [[288,4,24*]] GB1; [[288,12,18]] GT/IBM; [[288,24,12]] LLM; [[288,32,6]] LLM | 13.50 |
| 292 | [[292,18,8]] GT | 3.95 |
| 294 | [[294,10,20]] GB1/GT; [[294,18,10]] LP | 13.61 |
| 300 | [[300,8,22*]] GT | 12.91 |

## Weight-8 highlights (n ≤ 300) for reference
[[64,14,8]] (SY, kd²/n=14), [[72,14,8]] (SY/2609, 12.4), [[96,12,10]] (SY, 12.5), [[96,20,8]] (SY, 13.3), [[128,14,12]] (ES/SY, 15.8), [[144,14,14]] (SY, 19.1), [[144,16,10]] (2609, 11.1), [[160,8,16]] (self-dual, 12.8), [[192,26,12]] (SY, 19.5), [[288,50,8]] (LLM, 11.1), [[128,20,8]] (trivariate self-dual, 10), [[96,12,10]] 2BGA (LP 2306, nonabelian/abelian W=8), [[48,6,8]] & [[46,2,9]] GB (PK1904), upper-bound-only: [[216,14,<=20]], [[288,14,<=24]], [[288,20,<=22]], [[256,14,<=22]], [[192,20,<=16]].

## Supplement (2026-10-05): newer tables, incl. non-abelian and coset two-block codes

Compiled 2026-10-05 for `research/qec/code-discovery-2.md` by a literature sub-agent from the arXiv LaTeX sources, polynomials and GAP indices copied verbatim. Two corrections to the tables above found on the way: the Lin & Pryadko 2306.16400 rows were computed with QDistRnd (randomized), so their exactness flag should read Q, not P; and the SHYPS note (fixed in place). `known_codes.py` in `code-discovery-2/` reads the weight-6 rows of every two-block section below (sections marked NOT two-block are skipped).

Compiled 2026-10-05 from arXiv LaTeX sources (e-print tarballs) by the parent agent and three sub-agents (sections marked 'extracted by sub-agent');  polynomials copied verbatim (LaTeX -> ASCII). Column conventions as in literature.md. exact?: Y = paper states exact (ILP/MILP/exhaustive/deterministic enumeration), P = no method stated, Q = randomized (QDistRnd etc.), <= = upper bound only (kd²/n then uses the bound). Inside table cells ':' replaces '|' in group presentations (e.g. <r,s : r^m, s^2, (rs)^2>) so the markdown columns stay intact; ⋊ = semidirect product with the normal subgroup on the left (GAP StructureDescription); 'ord(G)' = group order. 489 table rows in total, n <= 400 except where flagged. Sections that are NOT two-block (3-block tricycle, 5-block ZSZ-LP, 6-block multicycle, LP matrices) are labelled as such in their headings.

### Priority 1 — Lin & Pryadko

**Convention for all 2306.16400 rows (Sec. III.A, Eqs. (2),(16)):** LP[a,b] with A = L(a), B = R(b): [L(a)]_{α,β} = Σ_g a_g δ_{α,gβ} (a multiplies group elements from the LEFT), [R(b)]_{α,β} = Σ_g b_g δ_{α,βg} (b multiplies from the RIGHT); H_X = (A, B), H_Z^T = (B ; -A) i.e. H_Z = (B^T, -A^T) (over F2: H_Z = (B^T | A^T)). Words like `s^3r` are group products s^3·r in GAP's internal order; terms sorted in GAP's element order, so `1` is always first. W = W_a + W_b. Distances: GAP package QDistRnd (randomized) -> exact? = Q for every row of this paper (literature.md marks its Table I rows P; QDistRnd is the actual method). arXiv has only v1 (28 Jun 2023); published PRA 109, 022407 (2024) has the same abstract ("19 pages, 9 figures, 3 tables"), full text paywalled.

**No W = 6 code is tabulated anywhere in 2306.16400.** W = 4..8 results (k = 2, 4, 6; W_a = 2, 3, 4) appear only as plots (Figs. 1-9, d vs sqrt(n), kd vs n). All three tables (I, II, III) are W = 8. Text-only mentions without polynomials: abelian [[64,18,8]] from C4xC4xC2 with W_a = W_b = 4 (kd/n > 2, kd²/n = 18.00); non-abelian W_a=2,W_b=6 sequences with kd = n for k = 6 or k ≡ 0 mod 4, and (k+2)d = n for k = 4s+2.

#### Lin & Pryadko 2306.16400 Table II [tab:Cmh-2+6] (abelian C_m x C_2, W_a=2, W_b=6 => w=8, largest-d codes with k | n=4m)

Group C_mh = C_m x C_2 = <x,s | x^m = s^2 = x s x^-1 s^-1 = 1>, ℓ = 2m, n = 4m. GAP ids from the authors' LaTeX comments: C4xC2 = SmallGroup(8,2), C6xC2 = (12,5), C8xC2 = (16,5), C10xC2 = (20,5), C12xC2 = (24,9), C14xC2 = (28,4). Paper: with a = a0(x)+s a1(x), these are index-4 quasi-quasi-cyclic codes with A = [[a0,a1],[a1,a0]] (circulant blocks). Bold d in the paper = codes that FAIL kd = n (marked 'bold' below).

| n | k | d | exact? | w | group | A | B | kd²/n | notes |
|---|---|---|---|---|---|---|---|---|---|
| 16 | 2 | 4 | Q | 8 | C4xC2 = SmallGroup(8,2), m=4 | `1+x` | `1+x+s+x^2+sx+sx^3` | 2.00 | Wa=2,Wb=6; bold d (kd != n) |
| 16 | 4 | 4 | Q | 8 | C4xC2 = SmallGroup(8,2), m=4 | `1+x` | `1+x+s+x^2+sx+x^3` | 4.00 | Wa=2,Wb=6 |
| 16 | 8 | 2 | Q | 8 | C4xC2 = SmallGroup(8,2), m=4 | `1+s` | `1+x+s+x^2+sx+sx^2` | 2.00 | Wa=2,Wb=6 |
| 24 | 4 | 5 | Q | 8 | C6xC2 = SmallGroup(12,5), m=6 | `1+x` | `1+x^3+s+x^4+x^2+sx` | 4.17 | Wa=2,Wb=6; bold d (kd != n) |
| 24 | 12 | 2 | Q | 8 | C6xC2 = SmallGroup(12,5), m=6 | `1+x^3` | `1+x^3+s+x^4+sx^3+x` | 2.00 | Wa=2,Wb=6 |
| 32 | 8 | 4 | Q | 8 | C8xC2 = SmallGroup(16,5), m=8 | `1+x^6` | `1+sx^7+sx^4+x^6+sx^5+sx^2` | 4.00 | Wa=2,Wb=6 |
| 32 | 16 | 2 | Q | 8 | C8xC2 = SmallGroup(16,5), m=8 | `1+sx^4` | `1+sx^7+sx^4+x^6+x^3+sx^2` | 2.00 | Wa=2,Wb=6 |
| 40 | 4 | 8 | Q | 8 | C10xC2 = SmallGroup(20,5), m=10 | `1+x` | `1+x^5+x^6+sx^6+x^7+sx^3` | 6.40 | Wa=2,Wb=6; bold d (kd != n) |
| 40 | 8 | 5 | Q | 8 | C10xC2 = SmallGroup(20,5), m=10 | `1+x^6` | `1+x^5+s+x^6+x+sx^2` | 5.00 | Wa=2,Wb=6 |
| 40 | 20 | 2 | Q | 8 | C10xC2 = SmallGroup(20,5), m=10 | `1+x^5` | `1+x^5+s+x^6+sx^5+x` | 2.00 | Wa=2,Wb=6 |
| 48 | 8 | 6 | Q | 8 | C12xC2 = SmallGroup(24,9), m=12 | `1+sx^10` | `1+x^3+sx^6+x^4+x^7+x^8` | 6.00 | Wa=2,Wb=6 |
| 48 | 12 | 4 | Q | 8 | C12xC2 = SmallGroup(24,9), m=12 | `1+x^3` | `1+x^3+sx^6+x^4+sx^9+x^7` | 4.00 | Wa=2,Wb=6 |
| 48 | 16 | 3 | Q | 8 | C12xC2 = SmallGroup(24,9), m=12 | `1+x^4` | `1+x^3+sx^6+x^4+x^7+sx^10` | 3.00 | Wa=2,Wb=6 |
| 48 | 24 | 2 | Q | 8 | C12xC2 = SmallGroup(24,9), m=12 | `1+sx^6` | `1+x^3+sx^6+x^4+sx^9+sx^10` | 2.00 | Wa=2,Wb=6 |
| 56 | 4 | 10 | Q | 8 | C14xC2 = SmallGroup(28,4), m=14 | `1+x` | `1+x^7+sx^8+x^2+x^3+sx^11` | 7.14 | Wa=2,Wb=6; bold d (kd != n) |
| 56 | 8 | 7 | Q | 8 | C14xC2 = SmallGroup(28,4), m=14 | `1+x^8` | `1+x^7+s+x^8+x^9+sx^4` | 7.00 | Wa=2,Wb=6 |
| 56 | 28 | 2 | Q | 8 | C14xC2 = SmallGroup(28,4), m=14 | `1+x^7` | `1+x^7+s+x^8+sx^7+x` | 2.00 | Wa=2,Wb=6 |

#### Lin & Pryadko 2306.16400 Table III [tab:D2m-2+6] (NON-ABELIAN dihedral D_m, W_a=2, W_b=6 => w=8, all satisfy kd = n)

Group D_m = C_m ⋉ C_2 = <r,s | r^m = s^2 = (rs)^2 = 1> (complete presentation), ℓ = 2m, n = 4m. GAP ids from the authors' LaTeX comments: D12 = SmallGroup(12,4), D16 = (16,7), D18 = (18,1), D20 = (20,4), D24 = (24,6), D28 = (28,3), D30 = (30,3), D32 = (32,18). a acts from the left (A = L(a)), b from the right (B = R(b)). Paper: with a = a0(r)+s a1(r) these are index-4 qQC codes, A = [[a0(x), conj(a1(x))],[a1(x), conj(a0(x))]] where conj(a0(r)) = a0(r^-1) = s a0(r) s. NOTE: these are weight-8 (2+6), not weight-6, and all have d <= 8; kd²/n <= 8.

| n | k | d | exact? | w | group | A | B | kd²/n | notes |
|---|---|---|---|---|---|---|---|---|---|
| 24 | 8 | 3 | Q | 8 | D12 = SmallGroup(12,4), m=6 | `1+r^4` | `1+sr^4+r^3+r^4+sr^2+r` | 3.00 | nonabelian; Wa=2,Wb=6; a left, b right |
| 24 | 12 | 2 | Q | 8 | D12 = SmallGroup(12,4), m=6 | `1+r^3` | `1+sr+r^3+r^4+sr^4+r` | 2.00 | nonabelian; Wa=2,Wb=6; a left, b right |
| 32 | 8 | 4 | Q | 8 | D16 = SmallGroup(16,7), m=8 | `1+r^2` | `1+sr^5+sr^4+r^2+sr^7+sr^6` | 4.00 | nonabelian; Wa=2,Wb=6; a left, b right |
| 32 | 16 | 2 | Q | 8 | D16 = SmallGroup(16,7), m=8 | `1+r^4` | `1+sr^3+sr^6+r^4+sr^7+sr^2` | 2.00 | nonabelian; Wa=2,Wb=6; a left, b right |
| 36 | 12 | 3 | Q | 8 | D18 = SmallGroup(18,1), m=9 | `1+r^3` | `1+s+r+r^3+sr^3+r^4` | 3.00 | nonabelian; Wa=2,Wb=6; a left, b right |
| 40 | 8 | 5 | Q | 8 | D20 = SmallGroup(20,4), m=10 | `1+r^2` | `1+sr^4+r^5+r^2+sr^6+r` | 5.00 | nonabelian; Wa=2,Wb=6; a left, b right |
| 40 | 20 | 2 | Q | 8 | D20 = SmallGroup(20,4), m=10 | `1+r^5` | `1+sr^2+r^5+r^6+sr^7+r` | 2.00 | nonabelian; Wa=2,Wb=6; a left, b right |
| 48 | 8 | 6 | Q | 8 | D24 = SmallGroup(24,6), m=12 | `1+r^10` | `1+sr^8+r^9+r^4+sr^2+r^5` | 6.00 | nonabelian; Wa=2,Wb=6; a left, b right |
| 48 | 12 | 4 | Q | 8 | D24 = SmallGroup(24,6), m=12 | `1+r^3` | `1+sr^7+r^3+r^4+sr^10+r^7` | 4.00 | nonabelian; Wa=2,Wb=6; a left, b right |
| 48 | 16 | 3 | Q | 8 | D24 = SmallGroup(24,6), m=12 | `1+r^8` | `1+sr^8+r^9+r^8+sr^4+r^5` | 3.00 | nonabelian; Wa=2,Wb=6; a left, b right |
| 48 | 24 | 2 | Q | 8 | D24 = SmallGroup(24,6), m=12 | `1+r^6` | `1+sr^11+r^6+sr^5+r+r^7` | 2.00 | nonabelian; Wa=2,Wb=6; a left, b right |
| 56 | 8 | 7 | Q | 8 | D28 = SmallGroup(28,3), m=14 | `1+r^4` | `1+sr^11+r^7+sr^5+r^12+r^9` | 7.00 | nonabelian; Wa=2,Wb=6; a left, b right |
| 56 | 28 | 2 | Q | 8 | D28 = SmallGroup(28,3), m=14 | `1+r^7` | `1+sr^2+r^7+r^8+sr^9+r` | 2.00 | nonabelian; Wa=2,Wb=6; a left, b right |
| 60 | 12 | 5 | Q | 8 | D30 = SmallGroup(30,3), m=15 | `1+r^12` | `1+sr^14+r^5+r^12+sr^11+r^14` | 5.00 | nonabelian; Wa=2,Wb=6; a left, b right |
| 60 | 20 | 3 | Q | 8 | D30 = SmallGroup(30,3), m=15 | `1+r^5` | `1+sr^13+r^5+r^12+sr^3+r^2` | 3.00 | nonabelian; Wa=2,Wb=6; a left, b right |
| 64 | 8 | 8 | Q | 8 | D32 = SmallGroup(32,18), m=16 | `1+r^6` | `1+sr^12+sr^9+r^6+s+sr` | 8.00 | nonabelian; Wa=2,Wb=6; a left, b right |
| 64 | 16 | 4 | Q | 8 | D32 = SmallGroup(32,18), m=16 | `1+r^4` | `1+sr^10+sr^3+r^4+sr^14+sr^7` | 4.00 | nonabelian; Wa=2,Wb=6; a left, b right |
| 64 | 32 | 2 | Q | 8 | D32 = SmallGroup(32,18), m=16 | `1+r^8` | `1+sr^11+sr^12+r^8+sr^3+sr^4` | 2.00 | nonabelian; Wa=2,Wb=6; a left, b right |

#### Lin & Pryadko 2306.16400 Example 1 (= Wang, Lin & Pryadko 2305.06890 Example) (NON-ABELIAN A4, essentially non-abelian odd-k code)

Same LP[a,b] convention (a left, b right). Odd k => not permutation-equivalent to any abelian/semi-abelian 2BGA code.

| n | k | d | exact? | w | group | A | B | kd²/n | notes |
|---|---|---|---|---|---|---|---|---|---|
| 24 | 5 | 3 | P | 8 | A4 = T = SmallGroup(12,3), <x,y : x^3 = (yx)^3 = y^2 = 1> | `1+x+y+x^-1yx` | `1+x+y+yx` | 1.88 | nonabelian; Wa=Wb=4; a left, b right; method not stated for the example |

#### Lin & Pryadko 2306.16400 Table I [tab:large-k] — presentations for the 17 rows ALREADY in literature.md (n<100, kd >= n, d > W; w=8)

Duplicates of literature.md rows, re-listed only to add the paper's verbatim group presentation ('shortest presentation' column) and flag typos. Group column = structure / SmallGroup(ℓ,#) / presentation as printed. Row blocks: (1) Wa=2,Wb=6 abelian; (2) Wa=2,Wb=6 non-abelian; (3) Wa=Wb=4 abelian; (4) Wa=Wb=4 non-abelian. a left (L(a)), b right (R(b)).

| n | k | d | exact? | w | group | A | B | kd²/n | notes |
|---|---|---|---|---|---|---|---|---|---|
| 72 | 8 | 9 | Q | 8 | C36 = SmallGroup(36,2), <r : r^36> | `1+r^28` | `1+r^9+r^18+r^12+r^29+r^14` | 9.00 | DUP; Wa=2,Wb=6 |
| 72 | 8 | 9 | Q | 8 | C9⋉C4 = SmallGroup(36,1), printed <r,s : s^4, r^6, s^-1rsr> | `1+r` | `1+s+r^6+s^3r+sr^7+s^3r^5` | 9.00 | DUP; nonabelian Wa=2,Wb=6; PRESENTATION TYPO: r^6 inconsistent with order 36 (and b contains r^7); the other 36#1 row prints r^9 -> read <r,s : s^4, r^9, s^-1rsr> (s inverts r) |
| 80 | 8 | 10 | Q | 8 | C5⋉C8 = SmallGroup(40,1), <r,s : s^5, r^8, r^-1srs> | `1+sr^4` | `1+r+r^2+s+s^3r+s^2r^6` | 10.00 | DUP; nonabelian Wa=2,Wb=6; complete presentation (r inverts s) |
| 96 | 8 | 12 | Q | 8 | (C3⋉C8)⋉C2 = SmallGroup(48,10), printed <r,s : s^6, r^8, (rs)^8> | `1+sr^2` | `1+r+s^3+s^4+s^2r^5+s^4r^6` | 12.00 | DUP; nonabelian Wa=2,Wb=6; printed relators are INCOMPLETE (a (6,8,8) triangle group is infinite) -> identification of r,s with GAP generators of SmallGroup(48,10) ambiguous |
| 54 | 6 | 9 | Q | 8 | C27 = SmallGroup(27,1), <r : r^27> | `1+r+r^3+r^7` | `1+r+r^12+r^19` | 9.00 | DUP; Wa=Wb=4 |
| 60 | 6 | 10 | Q | 8 | C30 = SmallGroup(30,4), <r : r^30> | `1+r^10+r^6+r^13` | `1+r^25+r^16+r^12` | 10.00 | DUP; Wa=Wb=4 |
| 70 | 8 | 10 | Q | 8 | C35 = SmallGroup(35,1), <r : r^35> | `1+r^15+r^16+r^18` | `1+r+r^24+r^27` | 11.43 | DUP; Wa=Wb=4 |
| 72 | 8 | 10 | Q | 8 | C36 = SmallGroup(36,2), <r : r^36> | `1+r^9+r^28+r^31` | `1+r+r^21+r^34` | 11.11 | DUP; Wa=Wb=4 |
| 72 | 10 | 9 | Q | 8 | C36 = SmallGroup(36,2), <r : r^36> | `1+r^9+r^28+r^13` | `1+r+r^3+r^22` | 11.25 | DUP; Wa=Wb=4 |
| 72 | 8 | 9 | Q | 8 | C9⋉C4 = SmallGroup(36,1), <r,s : s^4, r^9, s^-1rsr> | `1+s+r+sr^6` | `1+s^2r+s^2r^6+r^2` | 9.00 | DUP; nonabelian Wa=Wb=4; complete presentation (s inverts r) |
| 80 | 8 | 10 | Q | 8 | C5⋉C8 = SmallGroup(40,1), printed <r,s : s^5, r^8, s^-1rsr> | `1+r+s+s^3r^5` | `1+r^2+sr^4+s^3r^2` | 10.00 | DUP; nonabelian Wa=Wb=4; PRESENTATION TYPO: with s^5=1, s^-1rs=r^-1 forces r^2=1; the other 40#1 row prints r^-1srs -> read <r,s : s^5, r^8, r^-1srs> |
| 96 | 8 | 12 | Q | 8 | C3⋉C16 = SmallGroup(48,1), <r,s : s^3, r^16, r^-1srs> | `1+r+s+r^14` | `1+r^2+sr^4+r^11` | 12.00 | DUP; nonabelian Wa=Wb=4; complete presentation |
| 80 | 9 | 9 | Q | 8 | (C10xC2)⋉C2 = SmallGroup(40,8), printed <r,s : s^4, r^10, (rs)^2> | `1+sr^5+r^5+sr^6` | `1+s^2+r+s^2r^3` | 9.11 | DUP; nonabelian Wa=Wb=4; odd k; printed relators INCOMPLETE ((2,4,10) triangle group infinite) -> r,s ambiguous |
| 84 | 10 | 9 | Q | 8 | C7xS3 = SmallGroup(42,3), <r,s : s^3, r^14, r^-1srs> | `1+r^7+r^8+sr^10` | `1+s+r^5+s^2r^13` | 9.64 | DUP; nonabelian Wa=Wb=4; paper prints n=82 (typo, 2*42=84); complete presentation |
| 96 | 10 | 12 | Q | 8 | C12⋉C4 = SmallGroup(48,13), <r,s : s^4, r^12, s^-1rsr> | `1+s+r^9+sr` | `1+s^2r^9+r^7+r^2` | 15.00 | DUP; nonabelian Wa=Wb=4; complete presentation (s inverts r); best kd²/n in paper |
| 96 | 11 | 9 | Q | 8 | C24⋉C2 = SmallGroup(48,5), printed <r,s : s^2, r^24, (rs)^8> | `1+s+r^9+sr^13` | `1+r^9+sr^18+r^7` | 9.28 | DUP; nonabelian Wa=Wb=4; odd k; printed relators INCOMPLETE: (rs)^8=1 only forces srs=r^j with j in {5,11,17,23} mod 24 -> use GAP SmallGroup(48,5) action |
| 96 | 12 | 10 | Q | 8 | C2x(C3⋉C8) = SmallGroup(48,9), <r,s : s^6, r^8, r^-1srs> | `1+r+s^3r^2+s^2r^3` | `1+r+s^4r^6+s^5r^3` | 12.50 | DUP; nonabelian Wa=Wb=4; complete presentation (r inverts s) |

#### Lin, Liu, Lim & Pryadko 2502.19406 App. A longtable [tab:big-codes] — rows NOT yet in literature.md (abelian BB(3,3), all d_S = 3)

Convention: GB/BB two-block code H_X = (A,B), H_Z = (B^T, A^T) with A = a(x,y), B = b(x,y) on <x,y | x^{n_X} = y^{n_Y} = xyx^-1y^-1 = 1> (n_Y = 1: cyclic C_{n_X}); w = 3+3 = 6. Exhaustive enumeration over all groups with ℓ <= 75 (some larger). Distances: QDistRnd upper bounds then verified with vecdec; for d <= 10 verified with deterministic dist_m4ri -> Y for d <= 10, Q for d > 10. literature.md already has the 17 'best' rows; the rows below are the remaining n <= 400 rows (mostly dominated, low d). Also Table I of this paper lists the [[126,12,10]] code a = 1+t^7+t^8, b = 1+t^37+t^43 on C63 (circuit distance d_C = 10*), already noted in literature.md.

| n | k | d | exact? | w | group | A | B | kd²/n | notes |
|---|---|---|---|---|---|---|---|---|---|
| 42 | 10 | 4 | Y | 6 | C21 (nX=21,nY=1) | `1+x+x^5` | `1+x^2+x^10` | 3.81 | d_S=3 |
| 56 | 12 | 4 | Y | 6 | C14xC2 | `1+x+x^3y` | `1+x^2+x^6` | 3.43 | d_S=3 |
| 60 | 16 | 4 | Y | 6 | C30 | `1+x^2+x^8` | `1+x^4+x^16` | 4.27 | d_S=3 |
| 62 | 10 | 4 | Y | 6 | C31 | `1+x+x^12` | `1+x^2+x^24` | 2.58 | d_S=3 |
| 84 | 20 | 4 | Y | 6 | C42 | `1+x^2+x^10` | `1+x^4+x^20` | 3.81 | d_S=3 |
| 90 | 12 | 4 | Y | 6 | C15xC3 | `1+x+x^2y` | `1+xy^2+x^8` | 2.13 | d_S=3 |
| 90 | 24 | 4 | Y | 6 | C15xC3 | `1+x+x^4` | `1+x^2+x^8` | 4.27 | d_S=3 |
| 90 | 24 | 4 | Y | 6 | C45 | `1+x^3+x^12` | `1+x^6+x^24` | 4.27 | d_S=3 |
| 96 | 16 | 4 | Y | 6 | C12xC4 | `1+x+x^2y` | `1+x^2y^2+x^10` | 2.67 | d_S=3 |
| 98 | 12 | 4 | Y | 6 | C7xC7 | `1+y+x` | `1+y^2+x^2` | 1.96 | d_S=3 |
| 98 | 18 | 4 | Y | 6 | C7xC7 | `1+x+x^3` | `1+y+y^3` | 2.94 | d_S=3 |
| 112 | 24 | 4 | Y | 6 | C14xC4 | `1+x+x^3y^2` | `1+x^2+x^6` | 3.43 | d_S=3 |
| 112 | 24 | 4 | Y | 6 | C28xC2 | `1+x^2+x^6y` | `1+x^4+x^12` | 3.43 | d_S=3 |
| 120 | 16 | 4 | Y | 6 | C10xC6 | `1+x+x^3y` | `1+x^2+x^6y^2` | 2.13 | d_S=3 |
| 120 | 16 | 4 | Y | 6 | C30xC2 | `1+x+x^4y` | `1+x^2+x^8` | 2.13 | d_S=3 |
| 120 | 32 | 4 | Y | 6 | C10xC6 | `1+x^2+x^6y^2` | `1+x^2y^2+x^8y^2` | 4.27 | d_S=3 |
| 120 | 32 | 4 | Y | 6 | C30xC2 | `1+x^2+x^8` | `1+x^4+x^16` | 4.27 | d_S=3 |
| 120 | 32 | 4 | Y | 6 | C60 | `1+x^4+x^16` | `1+x^8+x^32` | 4.27 | d_S=3 |
| 124 | 20 | 4 | Y | 6 | C62 | `1+x^2+x^24` | `1+x^4+x^48` | 2.58 | d_S=3 |
| 126 | 12 | 4 | Y | 6 | C63 | `1+x+x^6` | `1+x^2+x^12` | 1.52 | d_S=3 |
| 126 | 12 | 6 | Y | 6 | C63 | `1+x+x^6` | `1+x^7+x^26` | 3.43 | d_S=3 |
| 126 | 12 | 8 | Y | 6 | C63 | `1+x+x^6` | `1+x^4+x^24` | 6.10 | d_S=3 |
| 126 | 16 | 4 | Y | 6 | C63 | `1+x+x^8` | `1+x^2+x^16` | 2.03 | d_S=3 |
| 146 | 18 | 4 | Y | 6 | C73 | `1+x+x^9` | `1+x^2+x^18` | 1.97 | d_S=3 |
| 170 | 16 | 4 | Y | 6 | C85 | `1+x+x^16` | `1+x^2+x^32` | 1.51 | d_S=3 |
| 186 | 14 | 4 | Y | 6 | C93 | `1+x+x^14` | `1+x^2+x^28` | 1.20 | d_S=3 |
| 186 | 14 | 6 | Y | 6 | C93 | `1+x+x^14` | `1+x^17+x^49` | 2.71 | d_S=3 |
| 210 | 14 | 10 | Y | 6 | C105 | `1+x+x^12` | `1+x^3+x^79` | 6.67 | d_S=3 |
| 210 | 14 | 4 | Y | 6 | C105 | `1+x+x^12` | `1+x^2+x^24` | 1.07 | d_S=3 |
| 210 | 14 | 8 | Y | 6 | C105 | `1+x+x^12` | `1+x^8+x^96` | 4.27 | d_S=3 |
| 294 | 30 | 4 | Y | 6 | C7xC21 | `1+x+x^3` | `1+y+y^5` | 1.63 | d_S=3 |
| 392 | 18 | 12 | Q | 6 | C14xC14 | `1+x+x^2y^5` | `1+y+x^7y^5` | 6.61 | d_S=3 |
| 392 | 24 | 12 | Q | 6 | C14xC14 | `1+x+x^3y^7` | `1+y+x^7y^5` | 8.82 | d_S=3 |

### Other sources found while checking Priority 1 (Lin-Pryadko group and GB follow-ups)

#### Lin, Lim, Kovalev & Pryadko 2506.16910 Table I [tab:toric-2222] (abelian multi-cycle AMC_2, D=4 'rotated 4D toric' codes — NOT two-block, 6 blocks, n = 6ℓ)

Convention: R = MBC(A,B,C,D) (Eq. 23); H_X = R_2 (rows [C B A | 0 0 0] and [-D 0 0 | B A 0], [0 -D 0 | -C 0 A], [0 0 -D | 0 -C -B]), H_Z = R_3^T; a_1..a_4 in the table (A..D presumably a_1..a_4). Stabilizer weight 6 (both types), highly redundant (single-shot), syndrome distance d_S = 4. Group C_ℓ = <x | x^ℓ>. Distance method not stated for the table (paper uses vecdec elsewhere) -> P. Columns A,B = a_1,a_2; a_3,a_4 in notes. Listed for completeness (weight-6 single-shot competitor), not BB.

| n | k | d | exact? | w | group | A | B | kd²/n | notes |
|---|---|---|---|---|---|---|---|---|---|
| 42 | 6 | 4 | P | 6 | C7 | `1+x` | `1+x^2` | 2.29 | a3=`1+x^3`, a4=`1+x^4`; d_S=4; confinement 4,4,4,6; 6-block AMC, not 2-block |
| 60 | 6 | 5 | P | 6 | C10 | `1+x` | `1+x^2` | 2.50 | a3=`1+x^3`, a4=`1+x^4`; d_S=4; confinement 4,6,6,6,4; 6-block AMC, not 2-block |
| 66 | 6 | 6 | P | 6 | C11 | `1+x` | `1+x^2` | 3.27 | a3=`1+x^3`, a4=`1+x^4`; d_S=4; confinement 4,6,6,6,4; 6-block AMC, not 2-block |
| 84 | 6 | 7 | P | 6 | C14 | `1+x` | `1+x^2` | 3.50 | a3=`1+x^5`, a4=`1+x^6`; d_S=4; confinement 4,6,6,6,4; 6-block AMC, not 2-block |
| 96 | 6 | 8 | P | 6 | C16 | `1+x` | `1+x^3` | 4.00 | a3=`1+x^5`, a4=`1+x^7`; d_S=4; confinement 4,6,8,8,4; 6-block AMC, not 2-block |
| 108 | 6 | 9 | P | 6 | C18 | `1+x` | `1+x^3` | 4.50 | a3=`1+x^5`, a4=`1+x^7`; d_S=4; confinement 4,6,8,8,4; 6-block AMC, not 2-block |
| 150 | 6 | 10 | P | 6 | C25 | `1+x` | `1+x^4` | 4.00 | a3=`1+x^6`, a4=`1+x^9`; d_S=4; confinement 4,6,8,8,4; 6-block AMC, not 2-block |
| 168 | 6 | 11 | P | 6 | C28 | `1+x` | `1+x^3` | 4.32 | a3=`1+x^7`, a4=`1+x^12`; d_S=4; confinement 4,6,8,8,4; 6-block AMC, not 2-block |
| 180 | 6 | 12 | P | 6 | C30 | `1+x^2` | `1+x^5` | 4.80 | a3=`1+x^8`, a4=`1+x^9`; d_S=4; confinement 4,6,8,8,4; 6-block AMC, not 2-block |
| 96 | 6 | 4 | P | 6 | C2^4 = <x1..x4> | `1+x1` | `1+x2` | 1.00 | a3=`1+x3`, a4=`1+x4`; conventional 4D toric code L=2; d_S=4 |
| 486 | 6 | 9 | P | 6 | C3^4 | `1+x1` | `1+x2` | 1.00 | n>400 (listed for completeness): conventional 4D toric L=3 |

#### Davenport, Blue & Chuang 2606.05044 Tables [tab:k2-mcr], [tab:k-large-mcr] ('MCR' GB codes as cyclic submodules; cyclic group, generator weight w from low-weight-generator search)

Convention: GB code over R_ℓ = F2[x]/(x^ℓ-1) with rowspace(H_X) = M_f(p,q) = {(p a, q a) | a in <f>} and rowspace(H_Z) = M_{rev f}(rev q, rev p) (Thm. 1). Equivalently (my reading) a GB code with a(x) = p f, b(x) = q f mod x^ℓ-1 — these a,b are DENSE; the paper's w is the minimum stabilizer-generator weight achievable for the same rowspace (found with low_weight_generator.sage, confirmed minimal by Gurobi ILP when no <=), so the matrices realizing w are not given as polynomials (repo: github.com/ajdav136/GBAutomorphisms). Distances: AB-reduction + Gurobi -> Y (k=2 table explicit; k>2 table says entries without <= ILP-validated). |Aut| and #automorphism gates in notes. Columns: A = `f` (p=1 in every row), B = `q`.

| n | k | d | exact? | w | group | A | B | kd²/n | notes |
|---|---|---|---|---|---|---|---|---|---|
| 18 | 2 | 5 | Y | 8 | C9 (ℓ=9) | `x+1` | `x^7+x^4+x^3+x` | 2.78 | p=1; #Aut=108; 10 gates; min generator weight w, NOT wt(a)+wt(b) |
| 22 | 2 | 6 | Y | 8 | C11 (ℓ=11) | `x+1` | `x^9+x^5+x^4+x^3+x` | 3.27 | p=1; #Aut=220; 10 gates; min generator weight w, NOT wt(a)+wt(b) |
| 30 | 2 | 7 | Y | 8 | C15 (ℓ=15) | `x+1` | `x^13+x^9+x^7+x^6+x^5+x^4+x+1` | 3.27 | p=1; #Aut=120; 9 gates; min generator weight w, NOT wt(a)+wt(b) |
| 50 | 2 | 9 | Y | 12 | C25 (ℓ=25) | `x+1` | `x^23+x^22+x^18+x^17+x^15+x^13+x^12+x^10+x^8+x^7+x^3+x^2+1` | 3.24 | p=1; #Aut=1000; min generator weight w, NOT wt(a)+wt(b) |
| 54 | 2 | 10 | Y | 16 | C27 (ℓ=27) | `x+1` | `x^25+x^22+x^21+x^19+x^18+x^16+x^13+x^12+x^10+x^7+x^4+x^3+x+1` | 3.70 | p=1; #Aut=972; min generator weight w, NOT wt(a)+wt(b) |
| 58 | 2 | 11 | Y | 12 | C29 (ℓ=29) | `x+1` | `x^27+x^26+x^21+x^19+x^18+x^17+x^15+x^14+x^12+x^11+x^10+x^8+x^3+x^2` | 4.17 | p=1; #Aut=1624; min generator weight w, NOT wt(a)+wt(b) |
| 66 | 2 | 13 | Y | 12 | C33 (ℓ=33) | `x+1` | `x^31+x^30+x^28+x^25+x^24+x^21+x^19+x^18+x^16+x^13+x^11+x^10+x^7+x^6+x^4+x` | 5.12 | p=1; #Aut=660; min generator weight w, NOT wt(a)+wt(b) |
| 30 | 6 | 5 | Y | 8 | C15 (ℓ=15) | `x^3+1` | `x^10+x^9+x^6+x^5+1` | 5.00 | p=1; #Aut=240; 20 gates; min generator weight w, NOT wt(a)+wt(b) |
| 66 | 6 | 8 | Y | 12 | C33 (ℓ=33) | `x^3+1` | `x^27+x^22+x^15+x^12+x^11+x^9+x^3` | 5.82 | p=1; #Aut=1320; min generator weight w, NOT wt(a)+wt(b) |
| 78 | 6 | 9 | Y | 12 | C39 (ℓ=39) | `x^3+1` | `x^33+x^26+x^24+x^21+x^18+x^15+x^13+x^6+1` | 6.23 | p=1; #Aut=1872; min generator weight w, NOT wt(a)+wt(b) |
| 90 | 10 | 10 | Y | <=18 | C45 (ℓ=45) | `x^5+x^3+x+1` | `x^38+x^36+x^35+x^32+x^27+x^23+x^20+x^18+x^17+x^15+x^9+x^8+x^5+x^2+1` | 11.11 | p=1; #Aut=1080; w only upper bound (<=18); min generator weight w, NOT wt(a)+wt(b) |
| 102 | 6 | 11 | Y | <=12 | C51 (ℓ=51) | `x^3+1` | `x^45+x^34+x^33+x^30+x^27+x^24+x^21+x^18+x^17+x^6+1` | 7.12 | p=1; #Aut=1632; min generator weight w, NOT wt(a)+wt(b) |
| 102 | 18 | 12 | <= | <=22 | C51 (ℓ=51) | `x^9+x^4+x^2+1` | `x^39+x^38+x^36+x^35+x^34+x^33+x^32+x^30+x^29+x^28+x^27+x^25+x^23+x^21+x^19+x^18+x^15+x^14+x^11+x^10+x^8+x^6+x^4+x^3+x` | 25.41 | p=1; #Aut=816; d<=12 upper bound; w<=22 (high weight); min generator weight w, NOT wt(a)+wt(b) |
| 110 | 10 | 10 | Y | <=16 | C55 (ℓ=55) | `x^5+1` | `x^45+x^44+x^33+x^25+x^22+x^20+x^15+x^11+x^5+1` | 9.09 | p=1; #Aut=4400; min generator weight w, NOT wt(a)+wt(b) |

### Priority 2 — non-abelian (and coset / cover) two-block codes

#### Aydin, Tamo & Barg 2606.17268 Table [tab:group_codes] (coset-based two-block codes Q_G^H(a,b), H NON-normal; mostly non-abelian G; w = 6 and 8)

Convention (Construction 1): G finite, H <= G, N = N_G(H); a in F2[G], b in F2[N]; H_X = [L(a) | R(b)], H_Z = [-R(b)^T | L(a)^T], where L(g) = permutation matrix of the LEFT action of g on the left cosets G/H (L(g)_{i,j} = 1 iff g x_j H = x_i H) and R(g), g in N_G(H), = matrix of the RIGHT action on G/H. n = 2[G:H]. For normal H this reduces to the 2BGA code over G/H (Lin-Pryadko convention: a left, b right). POLYNOMIALS ARE GIVEN ONLY AS INTEGER INDEX ARRAYS (verbatim below): a-indices refer to GAP 4.14.0 `LeftCosets(G, Core(G,H))`, b-indices to `LeftCosets(Normalizer(G,H), H)`; G = SmallGroup(ℓ,m), H = `Filtered(AllSubgroups(G), H -> not IsNormal(G,H))[s]`. Reconstruction requires GAP 4.14.0 orderings. 'All the distance values in this table are exact' -> Y. Circuit-level BP-OSD results (Table [tab:summarylogicalerrorrates]) in notes: pseudo-threshold p0, pL at p = 1e-3 / 1e-4.

| n | k | d | exact? | w | group | A | B | kd²/n | notes |
|---|---|---|---|---|---|---|---|---|---|
| 48 | 8 | 6 | Y | 6 | C3 x ((C16 ⋊ C2) ⋊ C4) = SmallGroup(384,512); H = C16 (s=53) | `[1, 34, 48]` | `[1, 6, 12]` | 6.00 | coset code (not 2BGA); p0=0.36%, pL(1e-3)=2e-4, pL(1e-4)=3e-7 |
| 96 | 8 | 10 | Y | 6 | C3 x ((C16 ⋊ C2) ⋊ C4) = SmallGroup(384,512); H = C8 (s=27) | `[1, 9, 87]` | `[1, 21, 23]` | 8.33 | coset code; beats best published BB/GT w6 at n=96 ([[96,4,12]], [[96,12,6]]); p0=0.47%, pL(1e-3)=2e-6, pL(1e-4)=8e-12 |
| 224 | 12 | 16 | Y | 6 | C7 x ((C4 x C4) ⋊ C2) = SmallGroup(224,53); H = C2 (s=1) | `[1, 81, 186]` | `[1, 16, 47]` | 13.71 | coset code; w6 kd²/n=13.71 > GT-optimal BB [[224,6,20]] (10.71); p0=0.54%, pL(1e-3)=7e-11, pL(1e-4)=2e-19 |
| 84 | 16 | 8 | Y | 8 | C21 x (C3 ⋊ C4) = SmallGroup(252,21); H = C6 (s=6) | `[1, 14, 71, 89]` | `[1, 6, 8, 19]` | 12.19 | coset code; p0=0.35%, pL(1e-3)=6e-5, pL(1e-4)=3e-9 |
| 112 | 16 | 10 | Y | 8 | C7 x ((C4 x C4) ⋊ C2) = SmallGroup(224,53); H = C4 (s=9) | `[1, 11, 59, 81]` | `[1, 15, 23, 25]` | 14.29 | coset code; p0=0.34%, pL(1e-3)=1e-5, pL(1e-4)=9e-11 |
| 128 | 16 | 12 | Y | 8 | (C8 ⋊ C2) ⋊ C8 = SmallGroup(128,10); H = C2 (s=1) | `[1, 47, 75, 88]` | `[1, 8, 12, 19]` | 18.00 | coset code; p0=0.36%, pL(1e-3)=6e-6, pL(1e-4)=9e-12 |
| 168 | 16 | 15 | Y | 8 | C7 x ((C6 x C2) ⋊ C2) = SmallGroup(168,33); H = C2 (s=1) | `[1, 72, 106, 109]` | `[1, 11, 18, 26]` | 21.43 | coset code; p0=0.36%, pL(1e-3)=2e-6, pL(1e-4)=2e-13 |

#### Aydin, Tamo & Barg 2606.17268 Table [tab:base_and_cover_codes] (2BGA base codes and their h-fold COVER 2BGA codes over cover groups G~ with G~/H = G; incl. NON-ABELIAN covers; w = 6, 7, 8)

Convention: cover codes are ordinary 2BGA codes Q_{G~}(a,b) (H trivial in Construction 1: H_X = [L(a) | R(b)], H_Z = [-R(b)^T | L(a)^T], a left-regular, b right-regular). a, b given as INTEGER INDICES into GAP 4.14.0 `Elements(G)` of the respective group (index 1 = identity) — verbatim, not converted (even for cyclic groups GAP's Elements order is not the power order in general). Base GAP id (l,m) = SmallGroup(l,m); cover id (l,m,s): G~ = SmallGroup(l,m), lift H = `Filtered(AllSubgroups(G~), h -> IsNormal(G~,h))[s]`. Distances: '<=' = QDistRnd upper bound (10^6 iterations), all others exact -> Y. w = wt(a)+wt(b). Base codes are previously known codes (e.g. [[102,22,9]] GB code from 'tripier2026' trapped-ion paper).

| n | k | d | exact? | w | group | A | B | kd²/n | notes |
|---|---|---|---|---|---|---|---|---|---|
| 40 | 8 | 5 | Y | 8 | C5 ⋊ C4 = SmallGroup(20,1) | `[1, 2, 4, 10]` | `[1, 3, 4, 11]` | 5.00 | BASE code; nonabelian |
| 80 | 8 | 10 | Y | 8 | C5 ⋊ C8 = SmallGroup(40,1); lift H=C2 (s=2) | `[1, 2, 5, 23]` | `[1, 9, 11, 25]` | 10.00 | 2-cover of [[40,8,5]]; nonabelian |
| 80 | 10 | 8 | Y | 8 | C2 x (C5 ⋊ C4) = SmallGroup(40,7); H=C2 (s=3) | `[1, 2, 10, 22]` | `[1, 4, 10, 25]` | 8.00 | 2-cover of [[40,8,5]]; nonabelian |
| 120 | 16 | 10 | Y | 8 | C3 x (C5 ⋊ C4) = SmallGroup(60,2); H=C3 (s=3) | `[1, 6, 20, 28]` | `[1, 10, 5, 43]` | 13.33 | 3-cover of [[40,8,5]]; nonabelian |
| 160 | 10 | 16 | <= | 8 | C2 x (C5 ⋊ C8) = SmallGroup(80,9); H=C2xC2 (s=6) | `[1, 2, 13, 39]` | `[1, 4, 16, 42]` | 16.00 | 4-cover of [[40,8,5]]; nonabelian; d<=16 |
| 62 | 12 | 7 | Y | 8 | C31 = SmallGroup(31,1) | `[1, 2, 3, 7]` | `[1, 2, 13, 18]` | 9.48 | BASE code |
| 124 | 12 | 14 | Y | 8 | C62 = SmallGroup(62,2); H=C2 (s=2) | `[1, 3, 5, 14]` | `[1, 4, 26, 36]` | 18.97 | 2-cover of [[62,12,7]]; cyclic (GB) |
| 124 | 14 | 12 | Y | 8 | C62 = SmallGroup(62,2); H=C2 (s=2) | `[1, 4, 6, 13]` | `[1, 3, 26, 36]` | 16.26 | 2-cover of [[62,12,7]]; cyclic (GB) |
| 186 | 12 | 19 | <= | 8 | C93 = SmallGroup(93,2); H=C3 (s=2) | `[1, 5, 8, 22]` | `[1, 7, 40, 55]` | 23.29 | 3-cover of [[62,12,7]]; d<=19 |
| 248 | 18 | 19 | <= | 8 | C124 = SmallGroup(124,2); H=C4 (s=3) | `[1, 10, 14, 23]` | `[1, 5, 49, 67]` | 26.20 | 4-cover of [[62,12,7]]; d<=19 |
| 248 | 16 | 21 | <= | 8 | C62 x C2 = SmallGroup(124,4); H=C2xC2 (s=5) | `[1, 6, 13, 27]` | `[1, 6, 50, 68]` | 28.45 | 4-cover of [[62,12,7]]; d<=21 |
| 56 | 12 | 4 | Y | 6 | C14 x C2 = SmallGroup(28,4) | `[1, 4, 12]` | `[1, 6, 15]` | 3.43 | BASE code (w6) |
| 112 | 12 | 8 | Y | 6 | C28 x C2 = SmallGroup(56,8); H=C2 (s=2) | `[1, 12, 28]` | `[1, 7, 25]` | 6.86 | 2-cover of [[56,12,4]] (w6) |
| 168 | 16 | 10 | Y | 6 | C14 x S3 = SmallGroup(84,13); H=C3 (s=3) | `[1, 12, 48]` | `[1, 7, 44]` | 9.52 | 3-cover of [[56,12,4]]; NONABELIAN w6 |
| 280 | 12 | 16 | Y | 6 | C14 x D10 = SmallGroup(140,9); H=C5 (s=3) | `[1, 4, 38]` | `[1, 7, 52]` | 10.97 | 5-cover of [[56,12,4]]; NONABELIAN w6; kd²/n 10.97 > GT [[280,6,<=22]] (10.37) |
| 48 | 6 | 6 | Y | 8 | C3 ⋊ C8 = SmallGroup(24,1) | `[1, 2, 3, 14]` | `[1, 2, 14, 17]` | 4.50 | BASE code; nonabelian |
| 96 | 12 | 10 | Y | 8 | C2 x (C3 ⋊ C8) = SmallGroup(48,9); H=C2 (s=3) | `[1, 2, 11, 22]` | `[1, 2, 33, 29]` | 12.50 | 2-cover of [[48,6,6]]; nonabelian (same params/group as Lin-Pryadko T1 [[96,12,10]]) |
| 192 | 14 | 16 | <= | 8 | C3 ⋊ ((C8 x C2) ⋊ C2) = SmallGroup(96,37); H=C2xC2 (s=9) | `[1, 2, 36, 30]` | `[1, 2, 57, 64]` | 18.67 | 4-cover of [[48,6,6]]; nonabelian; d<=16 |
| 36 | 4 | 6 | Y | 8 | (C3 x C3) ⋊ C2 = SmallGroup(18,4) | `[1, 2, 3, 6]` | `[1, 2, 8, 15]` | 4.00 | BASE code; nonabelian |
| 72 | 8 | 9 | Y | 8 | (C3 x C3) ⋊ C4 = SmallGroup(36,7); H=C2 (s=2) | `[1, 2, 4, 8]` | `[1, 2, 12, 27]` | 9.00 | 2-cover of [[36,4,6]]; nonabelian |
| 108 | 12 | 9 | Y | 8 | C3 x ((C3 x C3) ⋊ C2) = SmallGroup(54,13); H=C3 (s=6) | `[1, 6, 4, 17]` | `[1, 15, 13, 50]` | 9.00 | 3-cover of [[36,4,6]]; nonabelian |
| 144 | 10 | 14 | Y | 8 | (C3 x C3) ⋊ C8 = SmallGroup(72,13); H=C4 (s=7) | `[1, 2, 14, 23]` | `[1, 8, 33, 45]` | 13.61 | 4-cover of [[36,4,6]]; nonabelian |
| 70 | 16 | 6 | Y | 8 | C35 = SmallGroup(35,1) | `[1, 3, 9, 19]` | `[1, 3, 13, 23]` | 8.23 | BASE code |
| 140 | 16 | 12 | Y | 8 | C70 = SmallGroup(70,4); H=C2 (s=2) | `[1, 4, 15, 39]` | `[1, 6, 23, 43]` | 16.46 | 2-cover of [[70,16,6]] |
| 210 | 20 | 16 | <= | 8 | C105 = SmallGroup(105,2); H=C3 (s=2) | `[1, 7, 18, 54]` | `[1, 12, 29, 77]` | 24.38 | 3-cover of [[70,16,6]]; d<=16 |
| 280 | 22 | 18 | <= | 8 | C70 x C2 = SmallGroup(140,11); H=C2xC2 (s=5) | `[1, 8, 31, 74]` | `[1, 10, 63, 93]` | 25.46 | 4-cover of [[70,16,6]]; d<=18 |
| 84 | 16 | 8 | Y | 8 | C7 x S3 = SmallGroup(42,3) | `[1, 2, 3, 29]` | `[1, 4, 15, 27]` | 12.19 | BASE code; nonabelian |
| 168 | 20 | 14 | Y | 8 | C7 x (C3 ⋊ C4) = SmallGroup(84,3); H=C2 (s=2) | `[1, 2, 3, 51]` | `[1, 5, 36, 60]` | 23.33 | 2-cover of [[84,16,8]]; NONABELIAN; exact; kd²/n 23.33 |
| 252 | 16 | 21 | <= | 8 | C21 x S3 = SmallGroup(126,12); H=C3 (s=3) | `[1, 6, 21, 71]` | `[1, 11, 57, 78]` | 28.00 | 3-cover of [[84,16,8]]; nonabelian; d<=21 |
| 252 | 20 | 18 | <= | 8 | C21 x S3 = SmallGroup(126,12); H=C3 (s=3) | `[1, 15, 21, 71]` | `[1, 22, 57, 63]` | 25.71 | 3-cover of [[84,16,8]]; nonabelian; d<=18 |
| 252 | 24 | 17 | <= | 8 | C7 x ((C3 x C3) ⋊ C2) = SmallGroup(126,14); H=C3 (s=2) | `[1, 2, 11, 65]` | `[1, 4, 43, 98]` | 27.52 | 3-cover of [[84,16,8]]; nonabelian; d<=17 |
| 42 | 8 | 6 | Y | 8 | C21 = SmallGroup(21,2) | `[1, 3, 6, 14]` | `[1, 3, 8, 12]` | 6.86 | BASE code |
| 84 | 8 | 12 | Y | 8 | C42 = SmallGroup(42,6); H=C2 (s=2) | `[1, 6, 12, 29]` | `[1, 6, 14, 21]` | 13.71 | 2-cover of [[42,8,6]] |
| 126 | 8 | 15 | Y | 8 | C21 x C3 = SmallGroup(63,4); H=C3 (s=2) | `[1, 4, 15, 35]` | `[1, 12, 23, 33]` | 14.29 | 3-cover of [[42,8,6]] |
| 126 | 20 | 9 | Y | 8 | C63 = SmallGroup(63,2); H=C3 (s=2) | `[1, 18, 8, 31]` | `[1, 3, 23, 45]` | 12.86 | 3-cover of [[42,8,6]] |
| 168 | 10 | 18 | <= | 8 | C84 = SmallGroup(84,6); H=C4 (s=4) | `[1, 4, 24, 45]` | `[1, 7, 27, 48]` | 19.29 | 4-cover of [[42,8,6]]; d<=18 |
| 168 | 12 | 17 | <= | 8 | C84 = SmallGroup(84,6); H=C4 (s=4) | `[1, 7, 24, 64]` | `[1, 13, 21, 48]` | 20.64 | 4-cover of [[42,8,6]]; d<=17 |
| 168 | 14 | 16 | <= | 8 | C42 x C2 = SmallGroup(84,15); H=C2xC2 (s=6) | `[1, 10, 27, 56]` | `[1, 15, 23, 42]` | 21.33 | 4-cover of [[42,8,6]]; d<=16 |
| 62 | 10 | 7 | Y | 7 | C31 = SmallGroup(31,1) | `[1, 2, 13]` | `[1, 2, 3, 28]` | 7.90 | BASE code (w7 = 3+4) |
| 124 | 10 | 12 | Y | 7 | C62 = SmallGroup(62,2); H=C2 (s=2) | `[1, 3, 26]` | `[1, 4, 5, 56]` | 11.61 | 2-cover of [[62,10,7]] (w7) |
| 186 | 10 | 17 | <= | 7 | C93 = SmallGroup(93,2); H=C3 (s=2) | `[1, 5, 38]` | `[1, 3, 10, 83]` | 15.54 | 3-cover of [[62,10,7]] (w7); d<=17 |
| 248 | 10 | 21 | <= | 7 | C62 x C2 = SmallGroup(124,4); H=C2xC2 (s=5) | `[1, 6, 51]` | `[1, 9, 10, 113]` | 17.78 | 4-cover of [[62,10,7]] (w7); d<=21 |
| 42 | 10 | 5 | Y | 7 | C21 = SmallGroup(21,2) | `[1, 5, 13]` | `[1, 3, 10, 19]` | 5.95 | BASE code (w7) |
| 84 | 10 | 9 | Y | 7 | C42 = SmallGroup(42,6); H=C2 (s=2) | `[1, 8, 28]` | `[1, 6, 22, 37]` | 9.64 | 2-cover of [[42,10,5]] (w7) |
| 126 | 10 | 13 | Y | 7 | C63 = SmallGroup(63,2); H=C3 (s=2) | `[1, 6, 38]` | `[1, 3, 29, 46]` | 13.41 | 3-cover of [[42,10,5]] (w7) |
| 210 | 10 | 19 | <= | 7 | C105 = SmallGroup(105,2); H=C5 (s=3) | `[1, 14, 76]` | `[1, 41, 34, 102]` | 17.19 | 5-cover of [[42,10,5]] (w7); d<=19 |
| 102 | 22 | 9 | Y | 8 | C51 = SmallGroup(51,1) | `[1, 14, 45, 35]` | `[1, 27, 32, 50]` | 17.47 | BASE code (GB code from 'tripier2026') |
| 204 | 22 | 17 | <= | 8 | C102 = SmallGroup(102,4); H=C2 (s=2) | `[1, 26, 87, 71]` | `[1, 51, 65, 100]` | 31.17 | 2-cover of [[102,22,9]]; d<=17 |
| 204 | 24 | 15 | Y | 8 | C102 = SmallGroup(102,4); H=C2 (s=2) | `[1, 26, 87, 68]` | `[1, 54, 62, 100]` | 26.47 | 2-cover of [[102,22,9]]; exact; kd²/n 26.47 |
| 204 | 28 | 12 | Y | 8 | C102 = SmallGroup(102,4); H=C2 (s=2) | `[1, 26, 87, 68]` | `[1, 51, 65, 100]` | 19.76 | 2-cover of [[102,22,9]]; exact |
| 306 | 22 | 24 | <= | 8 | C153 = SmallGroup(153,1); H=C3 (s=2) | `[1, 31, 134, 104]` | `[1, 80, 105, 139]` | 41.41 | 3-cover of [[102,22,9]]; d<=24 |
| 306 | 26 | 18 | <= | 8 | C51 x C3 = SmallGroup(153,2); H=C3 (s=2) | `[1, 35, 138, 110]` | `[1, 84, 89, 151]` | 27.53 | 3-cover of [[102,22,9]]; d<=18 |
| 126 | 20 | 11 | Y | 8 | C63 = SmallGroup(63,2) | `[1, 6, 26, 60]` | `[1, 22, 53, 35]` | 19.21 | BASE code |
| 252 | 20 | 21 | <= | 8 | C126 = SmallGroup(126,6); H=C2 (s=2) | `[1, 10, 53, 122]` | `[1, 40, 98, 71]` | 35.00 | 2-cover of [[126,20,11]]; d<=21 |
| 252 | 22 | 18 | <= | 8 | C126 = SmallGroup(126,6); H=C2 (s=2) | `[1, 16, 44, 122]` | `[1, 40, 106, 71]` | 28.29 | 2-cover of [[126,20,11]]; d<=18 |

#### Aydin, Tamo & Barg 2606.17268 Table [tab:group_codes_additional] (App.; more coset-based Q_G^H(a,b) codes, H non-normal, G non-abelian; w = 6 and 8)

Same convention and index-array caveat as [tab:group_codes] above (a: indices into GAP 4.14.0 `LeftCosets(G, Core(G,H))`; b: into `LeftCosets(Normalizer(G,H), H)`; G = SmallGroup(ℓ,m), H = s-th non-normal subgroup in `Filtered(AllSubgroups(G), H -> not IsNormal(G,H))`). '<=' = QDistRnd 10^6 iterations upper bound, all others exact (Y). Parsed programmatically from the LaTeX source.

| n | k | d | exact? | w | group | A | B | kd²/n | notes |
|---|---|---|---|---|---|---|---|---|---|
| 36 | 4 | 6 | Y | 6 | C9 x D18 = SmallGroup(162,3); H = C9 (s=24) | `[1, 15, 36]` | `[1, 4, 8]` | 4.00 | coset code; nonabelian G |
| 48 | 4 | 8 | Y | 6 | C3 x (C9 ⋊ C8) = SmallGroup(216,12); H = C9 (s=14) | `[1, 26, 69]` | `[1, 8, 9]` | 5.33 | coset code; nonabelian G |
| 54 | 8 | 6 | Y | 6 | C36 x S3 = SmallGroup(216,47); H = C4 x C2 (s=23) | `[1, 13, 27]` | `[1, 4, 5]` | 5.33 | coset code; nonabelian G |
| 56 | 6 | 8 | Y | 6 | C7 x D8 = SmallGroup(56,9); H = C2 (s=1) | `[1, 5, 56]` | `[1, 2, 9]` | 6.86 | coset code; nonabelian G |
| 60 | 16 | 4 | Y | 6 | C15 x D10 = SmallGroup(150,8); H = C5 (s=6) | `[1, 11, 31]` | `[1, 10, 12]` | 4.27 | coset code; nonabelian G |
| 72 | 4 | 10 | Y | 6 | C3 x ((C6 x C2) ⋊ C2) = SmallGroup(72,30); H = C2 (s=1) | `[1, 12, 50]` | `[1, 4, 17]` | 5.56 | coset code; nonabelian G |
| 72 | 8 | 8 | Y | 6 | C36 x S3 = SmallGroup(216,47); H = C6 (s=17) | `[1, 6, 31]` | `[1, 7, 10]` | 7.11 | coset code; nonabelian G |
| 84 | 6 | 10 | Y | 6 | C21 x (C3 ⋊ C4) = SmallGroup(252,21); H = C6 (s=6) | `[1, 24, 34]` | `[1, 10, 20]` | 7.14 | coset code; nonabelian G |
| 90 | 8 | 10 | Y | 6 | C15 x ((C6 x C2) ⋊ C2) = SmallGroup(360,99); H = D8 (s=35) | `[1, 39, 41]` | `[1, 4, 14]` | 8.89 | coset code; nonabelian G |
| 96 | 4 | 12 | Y | 6 | (C3 x C3) ⋊ ((C4 x C4) ⋊ C2) = SmallGroup(288,489); H = S3 (s=93) | `[1, 19, 52]` | `[1, 14, 24]` | 6.00 | coset code; nonabelian G |
| 108 | 8 | 10 | Y | 6 | C36 x S3 = SmallGroup(216,47); H = C4 (s=12) | `[1, 13, 67]` | `[1, 8, 18]` | 7.41 | coset code; nonabelian G |
| 112 | 6 | 12 | Y | 6 | C7 x D8 x S3 = SmallGroup(336,188); H = S3 (s=59) | `[1, 76, 107]` | `[1, 13, 14]` | 7.71 | coset code; nonabelian G |
| 112 | 12 | 8 | Y | 6 | C7 x D8 x S3 = SmallGroup(336,188); H = S3 (s=59) | `[1, 44, 106]` | `[1, 12, 27]` | 6.86 | coset code; nonabelian G |
| 120 | 8 | 12 | Y | 6 | C15 x ((C6 x C2) ⋊ C2) = SmallGroup(360,99); H = C6 (s=18) | `[1, 162, 211]` | `[1, 7, 11]` | 9.60 | coset code; nonabelian G |
| 126 | 10 | 10 | Y | 6 | S3 x (C7 ⋊ C3) = SmallGroup(126,8); H = C2 (s=1) | `[1, 10, 81]` | `[1, 5, 19]` | 7.94 | coset code; nonabelian G |
| 144 | 4 | 16 | Y | 6 | C3 x (C9 ⋊ C8) = SmallGroup(216,12); H = C3 (s=1) | `[1, 26, 179]` | `[1, 22, 33]` | 7.11 | coset code; nonabelian G |
| 150 | 16 | 8 | Y | 6 | C15 x D10 = SmallGroup(150,8); H = C2 (s=1) | `[1, 4, 55]` | `[1, 7, 15]` | 6.83 | coset code; nonabelian G |
| 168 | 16 | 10 | Y | 6 | C7 x ((C6 x C2) ⋊ C2) = SmallGroup(168,33); H = C2 (s=1) | `[1, 92, 118]` | `[1, 14, 30]` | 9.52 | coset code; nonabelian G |
| 180 | 8 | 16 | Y | 6 | C15 x ((C6 x C2) ⋊ C2) = SmallGroup(360,99); H = C4 (s=14) | `[1, 84, 128]` | `[1, 5, 22]` | 11.38 | coset code; nonabelian G |
| 186 | 10 | 14 | Y | 6 | C31 x S3 = SmallGroup(186,3); H = C2 (s=1) | `[1, 74, 183]` | `[1, 21, 29]` | 10.54 | coset code; nonabelian G |
| 192 | 8 | 16 | Y | 6 | C3 x ((C16 ⋊ C2) ⋊ C4) = SmallGroup(384,512); H = C4 (s=8) | `[1, 226, 283]` | `[1, 9, 21]` | 10.67 | coset code; nonabelian G |
| 248 | 10 | 18 | <= | 6 | C31 x D8 = SmallGroup(248,9); H = C2 (s=1) | `[1, 122, 235]` | `[1, 15, 30]` | 13.06 | coset code; nonabelian G; d<=18 |
| 48 | 10 | 6 | Y | 8 | C3 x ((C6 x C2) ⋊ C2) = SmallGroup(72,30); H = C3 (s=9) | `[1, 16, 57, 58]` | `[1, 2, 3, 9]` | 7.50 | coset code; nonabelian G |
| 72 | 16 | 6 | Y | 8 | C3 x ((C6 x C2) ⋊ C2) = SmallGroup(72,30); H = C2 (s=1) | `[1, 3, 43, 47]` | `[1, 6, 8, 12]` | 8.00 | coset code; nonabelian G |
| 80 | 8 | 10 | Y | 8 | (C5 ⋊ C8) ⋊ C2 = SmallGroup(80,10); H = C2 (s=1) | `[1, 3, 43, 49]` | `[1, 9, 16, 18]` | 10.00 | coset code; nonabelian G |
| 80 | 10 | 8 | Y | 8 | (C5 ⋊ C8) ⋊ C2 = SmallGroup(80,10); H = C2 (s=1) | `[1, 10, 52, 58]` | `[1, 3, 9, 14]` | 8.00 | coset code; nonabelian G |
| 84 | 10 | 9 | Y | 8 | C21 x (C3 ⋊ C4) = SmallGroup(252,21); H = C6 (s=6) | `[1, 82, 101, 120]` | `[1, 8, 13, 16]` | 9.64 | coset code; nonabelian G |
| 90 | 18 | 7 | Y | 8 | C5 x ((C3 x C3) ⋊ C3) = SmallGroup(135,3); H = C3 (s=1) | `[1, 78, 111, 112]` | `[1, 4, 6, 7]` | 9.80 | coset code; nonabelian G |
| 96 | 10 | 12 | Y | 8 | (C3 x C3) ⋊ ((C4 x C4) ⋊ C2) = SmallGroup(288,489); H = S3 (s=93) | `[1, 18, 28, 84]` | `[1, 20, 22, 23]` | 15.00 | coset code; nonabelian G |
| 96 | 12 | 10 | Y | 8 | (C3 x C3) ⋊ ((C4 x C4) ⋊ C2) = SmallGroup(288,489); H = C6 (s=103) | `[1, 77, 85, 136]` | `[1, 13, 18, 21]` | 12.50 | coset code; nonabelian G |
| 96 | 18 | 8 | Y | 8 | (C3 x C3) ⋊ ((C4 x C4) ⋊ C2) = SmallGroup(288,489); H = S3 (s=93) | `[1, 19, 31, 82]` | `[1, 2, 18, 24]` | 12.00 | coset code; nonabelian G |
| 108 | 12 | 10 | Y | 8 | ((C9 x C3) ⋊ C3) ⋊ C2 = SmallGroup(162,4); H = C3 (s=13) | `[1, 29, 92, 97]` | `[1, 2, 12, 17]` | 11.11 | coset code; nonabelian G |
| 112 | 12 | 12 | Y | 8 | C7 x ((C4 x C4) ⋊ C2) = SmallGroup(224,53); H = C4 (s=8) | `[1, 46, 82, 107]` | `[1, 17, 20, 25]` | 15.43 | coset code; nonabelian G |
| 120 | 14 | 12 | Y | 8 | C15 x D8 = SmallGroup(120,32); H = C2 (s=3) | `[1, 13, 47, 100]` | `[1, 13, 16, 19]` | 16.80 | coset code; nonabelian G |
| 120 | 16 | 11 | Y | 8 | C15 x D8 = SmallGroup(120,32); H = C2 (s=3) | `[1, 9, 93, 101]` | `[1, 2, 15, 30]` | 16.13 | coset code; nonabelian G |
| 120 | 24 | 7 | Y | 8 | C5 x ((C3 x C3) ⋊ C4) = SmallGroup(180,23); H = C3 (s=12) | `[1, 32, 69, 73]` | `[1, 9, 20, 23]` | 9.80 | coset code; nonabelian G |
| 128 | 10 | 14 | Y | 8 | (C8 ⋊ C2) ⋊ C8 = SmallGroup(128,10); H = C2 (s=1) | `[1, 33, 92, 120]` | `[1, 18, 28, 30]` | 15.31 | coset code; nonabelian G |
| 128 | 18 | 10 | Y | 8 | (C8 ⋊ C2) ⋊ C8 = SmallGroup(128,10); H = C2 (s=1) | `[1, 40, 75, 107]` | `[1, 8, 12, 19]` | 14.06 | coset code; nonabelian G |
| 128 | 30 | 8 | Y | 8 | (C2 x (C4 ⋊ C4)) ⋊ C4 = SmallGroup(128,26); H = C2 (s=7) | `[1, 21, 78, 82]` | `[1, 12, 22, 27]` | 15.00 | coset code; nonabelian G |
| 136 | 8 | 15 | Y | 8 | C17 x D8 = SmallGroup(136,10); H = C2 (s=1) | `[1, 11, 55, 61]` | `[1, 12, 15, 29]` | 13.24 | coset code; nonabelian G |
| 144 | 10 | 15 | Y | 8 | C3 x (C9 ⋊ C8) = SmallGroup(216,12); H = C3 (s=1) | `[1, 17, 173, 200]` | `[1, 8, 33, 35]` | 15.62 | coset code; nonabelian G |
| 144 | 18 | 12 | Y | 8 | (C3 x C3) ⋊ ((C4 x C2 x C2) ⋊ C2) = SmallGroup(288,498); H = C4 (s=141) | `[1, 87, 88, 117]` | `[1, 7, 29, 35]` | 18.00 | coset code; nonabelian G |
| 144 | 24 | 8 | Y | 8 | (C3 x C3) ⋊ ((C4 x C4) ⋊ C2) = SmallGroup(288,489); H = C4 (s=15) | `[1, 112, 129, 134]` | `[1, 12, 17, 20]` | 10.67 | coset code; nonabelian G |
| 160 | 8 | 18 | <= | 8 | C5 ⋊ ((C8 x C4) ⋊ C2) = SmallGroup(320,17); H = C4 (s=61) | `[1, 28, 108, 113]` | `[1, 13, 17, 38]` | 16.20 | coset code; nonabelian G; d<=18 |
| 160 | 12 | 16 | Y | 8 | C5 ⋊ ((C4 ⋊ C4) ⋊ C4) = SmallGroup(320,10); H = C2 x C2 (s=59) | `[1, 65, 105, 128]` | `[1, 14, 28, 31]` | 19.20 | coset code; nonabelian G |
| 168 | 10 | 17 | <= | 8 | C7 x ((C6 x C2) ⋊ C2) = SmallGroup(168,33); H = C2 (s=1) | `[1, 60, 71, 108]` | `[1, 7, 19, 36]` | 17.20 | coset code; nonabelian G; d<=17 |
| 168 | 24 | 10 | Y | 8 | D8 x (C7 ⋊ C3) = SmallGroup(168,20); H = C2 (s=3) | `[1, 10, 48, 49]` | `[1, 12, 15, 25]` | 14.29 | coset code; nonabelian G |
| 186 | 22 | 12 | Y | 8 | C31 x S3 = SmallGroup(186,3); H = C2 (s=1) | `[1, 23, 90, 180]` | `[1, 2, 9, 26]` | 17.03 | coset code; nonabelian G |
| 186 | 36 | 7 | Y | 8 | C31 x S3 = SmallGroup(186,3); H = C2 (s=1) | `[1, 4, 19, 67]` | `[1, 6, 13, 19]` | 9.48 | coset code; nonabelian G |
| 200 | 28 | 10 | Y | 8 | C5 x C5 x D8 = SmallGroup(200,38); H = C2 (s=3) | `[1, 25, 157, 172]` | `[1, 14, 16, 33]` | 14.00 | coset code; nonabelian G |
| 208 | 10 | 20 | <= | 8 | C13 x ((C4 x C2) ⋊ C2) = SmallGroup(208,21); H = C2 (s=1) | `[1, 8, 172, 193]` | `[1, 24, 30, 50]` | 19.23 | coset code; nonabelian G; d<=20 |
| 216 | 14 | 18 | <= | 8 | C9 x ((C6 x C2) ⋊ C2) = SmallGroup(216,58); H = C2 (s=7) | `[1, 104, 196, 215]` | `[1, 6, 19, 35]` | 21.00 | coset code; nonabelian G; d<=18 |
| 240 | 16 | 20 | <= | 8 | C3 x ((C5 ⋊ C8) ⋊ C2) = SmallGroup(240,39); H = C2 (s=1) | `[1, 37, 136, 228]` | `[1, 11, 35, 52]` | 26.67 | coset code; nonabelian G; d<=20 |

#### Qian & Li 2608.08996 Tables [tab:codes] + [tab:structural-codes] (multi-agent discovery; coset-orbit balanced products, n <= 400; incl. NON-ABELIAN w6 [[336,12,20]])

Convention: entries of A and B are F2 sums of double cosets KgK. With K = {e} they are ordinary group-algebra elements; with normal K and 1x1 shape the code is effectively a 2BGA code over G/K (n = 2|G|/|K|). The left/right convention is given only as "balanced-product incidence rules" in Methods (not stated explicitly). w = max(stabilizer weight, qubit degree). exact = MILP-certified (Y); <= = QDistEvol 10^6. Matrices written (a b;c d). Extracted by sub-agent from LaTeX source.

| n | k | d | exact? | w | group | A | B | kd²/n | notes |
|---|---|---|---|---|---|---|---|---|---|
| 336 | 12 | 20 | Y | 6 | Z84⋊_29 Z4: x^84=s^4=e, sxs^-1=x^29 (order 336); K=<x^42> normal order 2 | `e+xs+x^3` | `s^3+x^2+x^4` | 14.29 | NONABELIAN w6, MILP-exact; effectively 2BGA over G/K (order 168). Paper: within 7% of [[340,16,18]] (15.25) |
| 400 | 16 | 22 | <= | 6 | Dic5xDic5: r_i^10=e, s_i^2=r_i^5, s_i r_i s_i^-1=r_i^-1; K=<r1^5 r2^5> normal order 2 | `s1r2s2+r1^9r2^2s2+r1^2s1r2^3s2` | `r1^8+r1^2r2+r2s2` | 19.36 | nonabelian w6; quotient (Dic5xDic5)/<(z1,z2)>; d<=22 |
| 288 | 16 | 18 | Y | 7 | Z12xZ48, K=<y^12> order 4 | `y^2+y^7+x` | `y^3+x+x^2+x^5y^9` | 18.00 | abelian (effectively BB on Z12xZ12), w7 |
| 384 | 16 | 24 | <= | 7 | Z12xZ48, K=<x^4> order 3 | `y^14+y^19+x` | `y^3+x+x^2+x^5y^21` | 24.00 | abelian, w7; d<=24 |
| 378 | 32 | 19 | Y | 8 | He(Z3)xZ7: x^3=y^3=z^3=u^7=e, z=xyx^-1y^-1 central; K={e} | `zu^3+y+xu+xyu^6` | `zu^4+yu^5+x+xyz^2u^3` | 30.56 | nonabelian 2BGA, n=2·ord(G)=378 |
| 288 | 24 | 18 | Y | 8 | Z12xZ48, K=<y^12> order 4 | `y^2+y^7+xy+x^3+x^11y^11` | `y^3+x+x^2` | 27.00 | abelian, 5+3 |
| 224 | 22 | 16 | Y | 8 | Z84⋊_29 Z4 (as above); K=<x^28> normal order 3 | `s^2+x+x^2s^3+x^5s^2` | `s+x^5+x^17s+x^20s^3` | 25.14 | G nonabelian but G/K = Z28xZ4 ABELIAN (29 = 1 mod 28), i.e. effectively a BB code on Z28xZ4 |
| 336 | 24 | 24 | <= | 8 | Z4xZ42, K={e} | `e+xy^2+x^2y^5+x^3y^6` | `e+xy^13+x^2y^9+x^3y^19` | 41.14 | abelian; d<=24 |
| 378 | 18 | 27 | <= | 8 | M27xZ7: x^9=s^3=u^7=e, sxs^-1=x^4; K={e} | `e+s^2u^3+x^3su+x^5u^6` | `s+x+x^2s^2u^5+x^5su^3` | 34.71 | nonabelian; d<=27 |
| 336 | 28 | 20 | <= | 8 | Z2xZ2xZ42 (x,y,z), K={e} | `e+yz^5+xz^2+xyz^6` | `e+yz^9+xz^13+xyz^19` | 33.33 | abelian; d<=20 |
| 288 | 18 | 18 | Y | 9 | SL(2,3)⋊_θ Z12: elements My^z, yMy^-1=PMP^-1, P=(1 0;0 2); K=<-I> | `(1 1;0 1)+y+(0 1;2 0)y^2+(1 0;1 1)y^3` | `(1 2;0 1)+(2 1;2 0)y+y^2+(1 0;2 1)y^3+(2 1;2 0)y^6` | 20.25 | nonabelian, w9 (4+5) |
| 256 | 18 | 16 | Y | 8 | Z8≀Z2: x1^8=x2^8=σ^2=e, x1x2=x2x1, σx1σ=x2; K={e} | `x2^2+x1+x1x2^6σ+x1^2x2^3σ` | `x2σ+x1x2^3+x1^2x2^4σ+x1^5` | 18.00 | nonabelian 2BGA |
| 384 | 18 | 28 | <= | 9 | SL(2,3)⋊_φ Z16, φ = conjugation by (1 1;2 1) in GL(2,3); K=<-I> | `e+(1 1;1 2)y+(1 0;1 1)y^3+(1 1;0 1)y^3+(0 1;2 0)y^6` | `e+(1 2;0 1)y+(1 0;1 1)y^2+(0 1;2 0)y^5` | 36.75 | nonabelian; d<=28 |
| 384 | 14 | 28 | <= | 9 | GL(2,3)xZ4, K={e} | `(0 1;1 0)+e+(0 1;2 0)y+(1 1;0 1)y^2+(2 0;0 2)y^3` | `e+(1 0;1 1)y+(1 1;0 1)y^2+(0 1;2 0)y^3` | 28.58 | nonabelian; d<=28 |
| 234 | 28 | 18 | Y | 10 | Z13xZ9, K={e} | `e+y^2+y^8+x^4y^4+x^6y^8` | `x^2y^8+x^5y^4+x^10y^2+x^11y^5+x^12` | 38.77 | abelian w10 |
| 372 | 44 | 18 | Y | 10 | Z31xZ6, K={e} | `x^5y+x^5y^3+x^7y^2+x^18y^2+x^30y^2` | `x^10y+x^10y^3+x^21y^5+x^24y^5+x^26y^5` | 38.32 | abelian w10 |
| 170 | 32 | 14 | Y | 10 | Z85, K={e} | `e+x^3+x^4+x^10+x^67` | `e+x^5+x^29+x^31+x^37` | 36.89 | abelian GB w10 |
| 390 | 32 | 32 | <= | 10 | Z195 | `e+x^5+x^28+x^155+x^186` | `e+x+x^64+x^86+x^161` | 84.02 | abelian GB w10; d<=32 |
| 390 | 36 | 30 | <= | 10 | Z39xZ5 | `e+x+x^7y+x^8y^3+x^25y^2` | `e+xy^2+x^3y^3+x^11y+x^12y^2` | 83.08 | abelian w10; d<=30 |
| 368 | 18 | 16 | Y | 9 | A6xZ2 (u generates Z2), K=<(1 2)(3 4)> non-normal | `(2 3 4 5 6)+(1 2 3)u` | `(2 3 4 5 6)u+(2 6 5 4 3)u+(1 3)(2 4)u` | 12.52 | coset-type code; n != 2ord(G)/ord(K) (qubits appear to be double cosets) - not 2BGA |
| 336 | 12 | 24 | <= | 9 | PSL(2,11), K=<(0 1;10 0)> non-normal | `(1 0;4 1)+(1 1;0 1)` | `e+(1 0;4 1)+(1 10;0 1)` | 20.57 | coset code; d<=24 |
| 248 | 12 | 18 | Y | 10 | PSL(2,13), K=<(3 0;0 9)> non-normal order 3 | `(1 0;4 1)+(1 1;0 1)` | `(0 1;12 0)+(1 12;0 1)` | 15.68 | coset code |
| 306 | 8 | 25 | <= | 10 | (Z17^2)⋊swap Z2: σxσ=y; K=<σ> non-normal | `y^4+x+x^3y^3σ` | `x^2y^3+x^5y+x^6y^6σ` | 16.34 | coset code; d<=25 |
| 320 | 24 | 16 | Y | 9 | Z41⋊Z8: x^41=s^8=e, sxs^-1=x^38; K=<x> normal | 4x4 circulant, rows (e x s s^6) shifted | 5x5 circulant, rows (e x^5s^3 s^2 x^15s^3 x^20s^3) shifted | 19.20 | NOT two-block (4x4 / 5x5 LP); K=<x> normal so effectively abelian quasi-cyclic over G/K = Z8; full matrices in SI |
| 400 | 26 | 16 | Y | 9 | Dic4: r^8=e, s^2=r^4, srs^-1=r^-1; K={e} | 3x4: `(e r r^2s r^5; r^2s e+r^4 r r^4s; r r^7s e r^2)` | 3x4: `(e r^2 r^5s r; rs e r^2 r^4; r^3 r^5s e r^5)` | 16.64 | LP, not two-block |
| 384 | 32 | 16 | Y | 8 | D6⋊_θ Z8: r^6=s^2=u^8=e, srs^-1=r^-1, uru^-1=r, usu^-1=r^3s; K=<r^3> | 2x2: `(r+su^2, u+r^4u^3; u+r^4u^3, r+su^2)` | 2x2: `(r+su^3, r^2u+r^5u^2; r^2u+r^5u^2, r+su^3)` | 21.33 | LP 2x2, not two-block |
| 396 | 8 | 32 | <= | 10 | Z31⋊Z6: x^31=s^6=e, sxs^-1=x^26; K=<s^2> non-normal | 3x3 circulant (e x s) | 3x3 circulant (e x^3s^3 s^5) | 20.69 | LP 3x3; d<=32 |

#### C. Liu 2609.36213 (examples, no table) (BBGA: weighted-shift bivariate bicycle codes over group algebras; non-abelian seeds)

Convention: H_X = [A|B], H_Z = [B^T|A^T]; lambda(z)u = zu (left), rho(z)u = uz (right). For p = q = 1: A = P = rho(a), i.e. a multiplies from the RIGHT; B = Q = lambda(b), b multiplies from the LEFT (note: opposite of Lin-Pryadko). Permutations act right-to-left. Extracted by sub-agent.

| n | k | d | exact? | w | group | A | B | kd²/n | notes |
|---|---|---|---|---|---|---|---|---|---|
| 144 | 16 | 12 | Y | 8 | A4 x C6; x=(1 2 3), y=(1 2)(3 4), t central order 6; x^3=y^2=(xy)^3=t^6=e, tx=xt, ty=yt | `t^5+xt+(xyx)t^4+(yxy)t^2` | `e+(xyx^2)t^5+(yxy)t^4+(yx^2)t` | 16.00 | exhaustive certificate (all weight<=10 supports); proven inequivalent to any abelian BB/2BGA code; same params as 2602.15372 reflection code |
| 144 | 16 | 12 | <= | 8 | D3 = <r,s : r^3=s^2=e, srs=r^-1>, p=3, q=4, seeds a=(e,e,s), b=(e,e,e,r) | `P^4+Q^2+Q^3+PQ` | `P^2+P^3+Q+(PQ)^5` | 16.00 | abelian realization <P,Q> = C6xC12 regular (x=Q order 12, y=P order 6); d<=12 |
| 18 | 4 | 3 | Y | 6/8 mixed | C3 = <r : r^3=e>, p=3, q=1, a=(e,e,c), b=(c), c=r+r^2 | `P^2+Q` | `P^2+PQ+P^2Q` | 2.00 | not an ordinary 2BGA code (proved); 6 checks of weight 6, 3 of weight 8 |

#### Hirasaki & Lee 2607.28621 Tables [Lifted_BBcodes] / [LiftedLPforCI] (lifts of 2BGA/LP codes via group extensions; NON-ABELIAN lifted groups; lifted polynomials NOT given)

Convention: A-modules free right R_G-modules, B-modules free left (balanced product: d_A(ar)=d_A(a)r, d_B(rb)=r d_B(b)); sub-agent reading: a multiplies from the left, b from the right. Distances QDistRnd unless verified by MIP ('d' exact, '<= d' upper bound). Only base-code polynomials are printed; the lifted codes' polynomials are not given, so these rows cannot be rebuilt from the paper alone. Extracted by sub-agent.

| n | k | d | exact? | w | group | A | B | kd²/n | notes |
|---|---|---|---|---|---|---|---|---|---|
| 216 | 12 | 14 | Y | 6 | Z2xZ2x((Z3xZ3)⋊Z3) | not given | not given | 10.89 | lift (K=Z3) of [[72,12,6]] Z6xZ6 `x^3+y+y^2`/`y^3+x+x^2`; beats Symons [[216,12,12]] |
| 216 | 12 | 14 | Y | 6 | Z6xZ6xZ3 (abelian) | not given | not given | 10.89 | same base |
| 128 | 16 | 12 | Y | 8 | Z16⋊Z4 | not given | not given | 18.00 | lift (K=Z2) of [[64,14,8]] Z8xZ4 `xy^3+1+x^6+x^3y^2`/`x^6y+x^4y+x^3+x^5y` |
| 54 | 8 | 6 | Y | 6 | (Z3xZ3)⋊Z3 | not given | not given | 5.33 | lift of [[18,8,2]] `1+y+y^2`/`1+x+x^2` |
| 54 | 8 | 6 | Y | 6 | Z9⋊Z3 | not given | not given | 5.33 | same base |
| 72 | 12 | 6 | Y | 6 | Z3xA4 | not given | not given | 6.00 | same base |
| 72 | 4 | 6 | Y | 6 | S3xS3, S3 = <r,s : r^3=s^2=rsrs=1>, subscripts 1,2 = factors | `1+s1s2r2+r2s2` | `1+r1+r2` | 2.00 | base code (Example 'Non-Abelian Group') |
| 144 | 4 | 12 | Y | 6 | (Z3⋊Z4)xS3 | not given | not given | 4.00 | lift K=Z2 of the S3xS3 code |
| 216 | 4 | 16 | <= | 6 | ((Z3xZ3)⋊Z3)⋊(Z2xZ2) | not given | not given | 4.74 | lift K=Z3; d<=16 |
| 288 | 4 | 22 | <= | 6 | (Z3xZ3)⋊QD16 | not given | not given | 6.72 | lift K=Z4; d<=22 |
| 288 | 4 | 24 | <= | 6 | (Z3⋊Z8)xS3 | not given | not given | 8.00 | lift K=Z4; d<=24 |
| 288 | 4 | 22 | <= | 6 | (Z3xZ3)⋊(Z8⋊Z2) | not given | not given | 6.72 | lift K=Z4; d<=22 |

#### Hong 2607.27644 Tables 1/3/5 + 'more ZSZ-LP' (ZSZ lifted-product codes over NON-ABELIAN metacyclic groups — 5-block, NOT two-block, check weight 9)

Convention: group Z_l1 ⋊_q Z_l2 = <x,y : x^l1 = y^l2 = y x y^-1 x^-q = 1>, order ℓ = l1·l2, n = 5ℓ. H_X = [[A,0,B,0,C^T],[0,A,0,B,D^T]], H_Z = [[C,D,0,0,A^T],[0,0,C,D,B^T]]; A = L[a], B = L[b] left-regular, C = R[c], D = R[d] right-regular (R[g]R[h] = R[hg]). Distances: pySATDist (exact, Y) or QDistEvol 10^6 (<=). A column = "a ; b", B column = "c ; d" (verbatim). Extracted by sub-agents (two independent extractions agree).

| n | k | d | exact? | w | group | A | B | kd²/n | notes |
|---|---|---|---|---|---|---|---|---|---|
| 160 | 32 | 10 | Y | 9 | Z16⋊_9 Z2 | `a=1+x^13+x^14y ; b=1+x^3y+x^13y` | `c=1+x^14+x^12y ; d=1+x^7+x^11` | 20.00 | ZSZ-LP-160; also fold-symmetric with phi(x)=xy, phi(y)=y (c=phi(abar), d=phi(bbar)) |
| 240 | 48 | 12 | Y | 9 | Z12⋊_7 Z4 | `a=1+x^8y+x^7y^2 ; b=1+x^2+x^5y` | `c=1+x^4y+x^2y^3 ; d=1+x^4+x^9` | 28.80 | ZSZ-LP-240; fold-symmetric with phi(x)=x^11, phi(y)=y^3 |
| 320 | 64 | 14 | Y | 9 | Z16⋊_3 Z4 | `a=1+x+y ; b=1+x^2+x^13y^3` | `c=1+x^12y+x^2y^3 ; d=1+x^12+x^11y^2` | 39.20 | ZSZ-LP-320 |
| 390 | 78 | 16 | <= | 9 | Z26⋊_3 Z3 | `a=1+x^17y+x^14y^2 ; b=1+x^16+x^3y` | `c=1+x^24+x^8y ; d=1+x^21+y` | 51.20 | ZSZ-LP-390; d<=16 |
| 390 | 78 | 15 | <= | 9 | Z26⋊_3 Z3 | `a=1+x^21y+x^5y^2 ; b=1+y+xy` | `c=phi(abar) ; d=phi(bbar)` | 45.00 | fold-symmetric variant, phi(x)=x^-1, phi(y)=y; d<=15 |
| 60 | 12 | 6 | Y | 9 | Z3⋊_2 Z4 | `a=1+y^3+x^2y^3 ; b=1+x+xy^2` | `c=1+y+xy^3 ; d=1+y^2+y^3` | 7.20 | more ZSZ-LP |
| 100 | 20 | 8 | Y | 9 | Z5⋊_4 Z4 | `a=1+x^2y+y^3 ; b=1+x^2+xy^3` | `c=1+y^3+x^4y^3 ; d=1+x^4y^2+xy^3` | 12.80 | more ZSZ-LP |
| 150 | 30 | 10 | Y | 9 | Z15⋊_11 Z2 | `a=1+x^11+x^12y ; b=1+y+x^6y` | `c=1+x^3+x^11 ; d=1+x^6+x^14y` | 20.00 | more ZSZ-LP |
| 210 | 42 | 12 | Y | 9 | Z21⋊_8 Z2 | `a=1+xy+x^3y ; b=1+x^12+x^17` | `c=1+x^4y+x^15y ; d=1+x^9y+x^18y` | 28.80 | more ZSZ-LP |

### Priority 3 — newer abelian / multivariate tables

#### Webster, Berent, Chandra, Hockings, Baspin, Thomsen, Smith & Cohen 2602.11457 Table [tab:codes] (Pinnacle architecture GB codes, weight 6, l = 2^m - 1)

Convention: S_X,j = prod_{a in A} X_(j+a),L · prod_{b in B} X_(j+b),R; S_Z,j = prod_{a in A} Z_(j-a),R · prod_{b in B} Z_(j-b),L, i.e. H_X = [A|B], H_Z = [B^T|A^T]; A, B supports in Z_l, each generating a simplex code. Method not stated (P). [[510,16,24]] omitted (n > 400). Extracted by sub-agent.

| n | k | d | exact? | w | group | A | B | kd²/n | notes |
|---|---|---|---|---|---|---|---|---|---|
| 30 | 8 | 4 | P | 6 | Z15 | `1+x^6+x^13` | `1+x+x^4` | 4.27 | A={0,6,13}, B={0,1,4} |
| 62 | 10 | 6 | P | 6 | Z31 | `1+x^6+x^15` | `1+x^5+x^7` | 5.81 | A={0,6,15}, B={0,5,7} |
| 126 | 12 | 10 | P | 6 | Z63 | `1+x^4+x^37` | `1+x^29+x^49` | 9.52 | A={0,4,37}, B={0,29,49} |
| 254 | 14 | 16 | P | 6 | Z127 | `1+x^32+x^100` | `1+x^28+x^49` | 14.11 | same params as GB1 254 in literature.md, different polynomials |

#### Liu, Xu & Xu 2602.15372 Tables 1-4 + End-Matter EM1-EM4 (self-dual 'stacked' codes: double-chain GB, double-layer BB, double-layer twisted BB, double-layer REFLECTION (non-abelian-like) codes; weight 8)

Convention: base code h_X = (A|B), h_Z = (B^T|A^T) with [A,B] = [A,A^T] = [B,B^T] = 0; stacked code H_X = H_Z = (U : U^T) with U = I_2 ⊗ A + σ_x ⊗ B^T, n = 4·(base size). For circulant/BB/twisted bases this is a self-dual 2BGA code on base x Z_2 = <..., s> with A' = a + s·b(x^-1), B' = A'^T. A, B columns = the paper's verbatim weight-2+2 base polynomials. Twisted BB: y^m = 1, x^l y^-gamma = 1. Reflection rows: p = M_x, q = M_y reflection permutation matrices ([M_l]_ij = δ_{i+j,l+1}); products in the order written; not group-algebra (regular) codes. Distances by integer programming (Y) unless <=. 'even' = only even-weight logicals. Flags: EM2 [[260,20,<=10]] has x^41 with l=5 (typo?); EM4 [[380,4,<=58]] implausible; m=1 rows still carry y exponents (y = I). Duplicate rows of EM tables that repeat main tables dropped. Extracted by sub-agent (reflection rows independently confirmed by a second sub-agent).

| n | k | d | exact? | w | group | A | B | kd²/n | notes |
|---|---|---|---|---|---|---|---|---|---|
| 36 | 4 | 6 | Y | 8 | base Z9 | `1+x^4` | `x^3+x^6` | 4.00 | double-chain T1 |
| 84 | 12 | 6 | Y | 8 | base Z21 | `x^2+x^5` | `x^5+x^14` | 5.14 | double-chain T1 |
| 100 | 12 | 8 | Y | 8 | base Z25 | `x^10+x^24` | `x^10+x^16` | 7.68 | double-chain T1 |
| 108 | 4 | 12 | Y | 8 | base Z27 | `x^22+x^24` | `x^12+x^22` | 5.33 | double-chain T1 |
| 132 | 8 | 12 | Y | 8 | base Z33 | `x^10+x^11` | `x^11+x^31` | 8.73 | double-chain T1 |
| 24 | 8 | 4 | Y | 8 | base Z6 | `1+x^2` | `x^3+x^4` | 5.33 | double-chain T1 (even) |
| 72 | 6 | 8 | Y | 8 | base Z18 | `x^2+x^17` | `x^4+x^5` | 5.33 | double-chain T1 (even) |
| 80 | 8 | 8 | Y | 8 | base Z20 | `1+x^17` | `x^8+x^17` | 6.40 | double-chain T1 (even) |
| 88 | 4 | 10 | Y | 8 | base Z22 | `x^13+x^18` | `x+x^5` | 4.55 | double-chain T1 (even) |
| 104 | 6 | 12 | Y | 8 | base Z26 | `x^6+x^11` | `x^5+x^14` | 8.31 | double-chain T1 (even) |
| 116 | 4 | 14 | Y | 8 | base Z29 | `1+x^3` | `x^20+x^25` | 6.76 | double-chain EM1 |
| 148 | 4 | 16 | Y | 8 | base Z37 | `x^21+x^24` | `x^17+x^22` | 6.92 | EM1 |
| 176 | 16 | 8 | Y | 8 | base Z44 | `x^11+x^31` | `1+x^8` | 5.82 | EM1 |
| 204 | 4 | 18 | <= | 8 | base Z51 | `1+x^14` | `x^32+x^40` | 6.35 | EM1 |
| 276 | 12 | 12 | <= | 8 | base Z69 | `x^27+x^33` | `x^15+x^54` | 6.26 | EM1 |
| 380 | 12 | 16 | <= | 8 | base Z95 | `x^2+x^25` | `x^58+x^75` | 8.08 | EM1 |
| 240 | 28 | 6 | <= | 8 | base Z60 | `x^41+x^59` | `x^4+x^26` | 4.20 | EM1 (even) |
| 248 | 4 | 20 | <= | 8 | base Z62 | `x^7+x^26` | `x^32+x^34` | 6.45 | EM1 (even) |
| 264 | 8 | 16 | <= | 8 | base Z66 | `x^4+x^18` | `x^42+x^53` | 7.76 | EM1 (even) |
| 280 | 4 | 22 | <= | 8 | base Z70 | `x^25+x^51` | `x^21+x^64` | 6.91 | EM1 (even) |
| 296 | 6 | 18 | <= | 8 | base Z74 | `x^6+x^65` | `x^8+x^25` | 6.57 | EM1 (even) |
| 312 | 6 | 20 | <= | 8 | base Z78 | `x^14+x^73` | `x^2+x^71` | 7.69 | EM1 (even) |
| 360 | 4 | 24 | <= | 8 | base Z90 | `x^22+x^24` | `x+x^52` | 6.40 | EM1 (even) |
| 60 | 12 | 5 | Y | 8 | base Z3xZ5 (l=3,m=5) | `x^2y^2+x^2y` | `x^2y^2+x^2` | 5.00 | double-layer BB T2 |
| 84 | 8 | 8 | Y | 8 | base Z3xZ7 | `x^2y+x` | `x^2y^2+x` | 6.10 | T2 |
| 100 | 12 | 8 | Y | 8 | base Z5xZ5 | `xy+x^4y` | `y^2+x^4y^3` | 7.68 | T2 |
| 108 | 16 | 6 | Y | 8 | base Z9xZ3 | `x^2y^6+x^3y^3` | `x^2y^4+x^4y` | 5.33 | T2 |
| 140 | 16 | 8 | Y | 8 | base Z7xZ5 | `y^4+x^2y^2` | `y^2+x^5y` | 7.31 | T2 |
| 56 | 6 | 8 | Y | 8 | base Z7xZ2 | `y^4+xy^3` | `x^5+x^2y^3` | 6.86 | T2 |
| 80 | 10 | 8 | Y | 8 | base Z5xZ4 | `y^2+y` | `x^2y+x^4` | 8.00 | T2 |
| 112 | 8 | 12 | Y | 8 | base Z14xZ2 | `x^3y^7+x^11y^4` | `y^2+x^5y^12` | 10.29 | T2 |
| 120 | 8 | 12 | Y | 8 | base Z6xZ5 | `x^4+x^5y^4` | `x+x^5y^3` | 9.60 | T2 |
| 160 | 20 | 8 | Y | 8 | base Z4xZ10 | `y^3+x^2y^2` | `x+x^3y^3` | 8.00 | T2 |
| 156 | 12 | 10 | Y | 8 | base Z13xZ3 | `y^6+x^10y^12` | `x^10y^11+x^8y^8` | 7.69 | EM2 |
| 204 | 8 | 16 | <= | 8 | base Z17xZ3 | `x+x^6y^10` | `x^2y^12+x^6y^11` | 10.04 | EM2 |
| 228 | 4 | 20 | <= | 8 | base Z19xZ3 | `x^17y^13+x^11y^13` | `x^13+x^18y` | 7.02 | EM2 |
| 260 | 20 | 10 | <= | 8 | base Z5xZ13 | `x^4y^2+x^41` | `x^2+x^2y^3` | 7.69 | EM2; x^41 with l=5 ambiguous (typo?) |
| 276 | 4 | 22 | <= | 8 | base Z23xZ3 | `x^20y^19+x^16y^12` | `x^17y^18+x^8y^9` | 7.01 | EM2 |
| 280 | 32 | 8 | <= | 8 | base Z7xZ10 | `xy^5+x^4y^3` | `x^3y^3+x^6y^3` | 7.31 | EM2 |
| 364 | 28 | 10 | <= | 8 | base Z7xZ13 | `y^5+y^4` | `xy+xy^6` | 7.69 | EM2 |
| 24 | 8 | 4 | Y | 8 | base Z3xZ2 | `xy^2+x^2` | `x^2y+xy^2` | 5.33 | EM2 |
| 32 | 12 | 4 | Y | 8 | base Z2xZ4 | `xy+x` | `x+y` | 6.00 | EM2 |
| 64 | 24 | 4 | Y | 8 | base Z4xZ4 | `x^3y^3+x` | `xy^2+xy^3` | 6.00 | EM2 |
| 88 | 4 | 12 | Y | 8 | base Z11xZ2 | `y^7+xy` | `x^9y^4+x^7y^7` | 6.55 | EM2 |
| 100 | 12 | 8 | Y | 8 | twisted l=5,m=5,gamma=3 | `xy^3+x^3y^3` | `x^2y^2+x^4y^3` | 7.68 | double-layer twisted BB T3 |
| 132 | 8 | 12 | Y | 8 | twisted l=3,m=11,gamma=9 | `1+x^2y^2` | `y^2+xy` | 8.73 | T3 |
| 140 | 16 | 8 | Y | 8 | twisted l=5,m=7,gamma=1 | `x^2y+y` | `x^2+x^4y^2` | 7.31 | T3 |
| 180 | 20 | 8 | Y | 8 | twisted l=5,m=9,gamma=4 | `y^3+x` | `x^4y+xy^2` | 7.11 | T3 |
| 204 | 8 | 16 | Y | 8 | twisted l=17,m=3,gamma=2 | `x^2y^5+x^14y` | `x^11y^16+x^13y^10` | 10.04 | T3 |
| 112 | 8 | 12 | Y | 8 | twisted l=2,m=14,gamma=6 | `y+xy` | `1+y` | 10.29 | T3 |
| 128 | 16 | 8 | Y | 8 | twisted l=4,m=8,gamma=4 | `xy+xy^3` | `x^2y^2+xy^2` | 8.00 | T3 |
| 144 | 8 | 12 | Y | 8 | twisted l=2,m=18,gamma=10 | `1+y` | `y+xy` | 8.00 | T3 |
| 176 | 10 | 12 | Y | 8 | twisted l=11,m=4,gamma=1 | `x^5y^2+x^2y^8` | `x^10y+y^2` | 8.18 | T3 |
| 208 | 8 | 16 | Y | 8 | twisted l=26,m=2,gamma=1 | `x^10y^21+x^7y^21` | `x^10y^4+xy^21` | 9.85 | T3 |
| 60 | 12 | 5 | Y | 8 | twisted l=3,m=5,gamma=4 | `x^2+x^2y` | `1+y^2` | 5.00 | EM3 |
| 84 | 8 | 8 | Y | 8 | twisted l=3,m=7,gamma=4 | `1+xy^2` | `y^2+x^2y^2` | 6.10 | EM3 |
| 220 | 12 | 12 | <= | 8 | twisted l=11,m=5,gamma=2 | `x^9y^9+x^2y^2` | `x^7+x^9y^5` | 7.85 | EM3 |
| 228 | 8 | 16 | <= | 8 | twisted l=3,m=19,gamma=1 | `y^2+x` | `1+x^2y^2` | 8.98 | EM3 |
| 252 | 4 | 20 | <= | 8 | twisted l=7,m=9,gamma=4 | `x^2y^4+x^2` | `x^3y+x^6y^4` | 6.35 | EM3 |
| 260 | 4 | 21 | <= | 8 | twisted l=5,m=13,gamma=10 | `x^4y+x^2y^4` | `x^4+1` | 6.78 | EM3 |
| 324 | 8 | 20 | <= | 8 | twisted l=3,m=27,gamma=9 | `x^2y+y^2` | `xy^2+x^2y^2` | 9.88 | EM3 |
| 340 | 4 | 22 | <= | 8 | twisted l=5,m=17,gamma=14 | `y^4+y^3` | `x^4+x^3y` | 5.69 | EM3 |
| 372 | 8 | 18 | <= | 8 | twisted l=3,m=31,gamma=16 | `xy^2+y^2` | `x^2y^2+y` | 6.97 | EM3 |
| 24 | 8 | 4 | Y | 8 | twisted l=3,m=2,gamma=1 | `y^2+x^2y` | `y+x` | 5.33 | EM3 |
| 32 | 12 | 4 | Y | 8 | twisted l=2,m=4,gamma=1 | `xy+1` | `x+1` | 6.00 | EM3 |
| 48 | 16 | 4 | Y | 8 | twisted l=2,m=6,gamma=2 | `y+xy` | `xy+x` | 5.33 | EM3 |
| 68 | 4 | 10 | Y | 8 | refl l=17,m=1 | `x^8y^5q+x^9y^12q` | `x^11y^2+x^15y^16q` | 5.88 | double-layer reflection T4 |
| 100 | 12 | 8 | Y | 8 | refl l=5,m=5 | `x^2py+x^2py^4` | `xy^2+x^3y` | 7.68 | T4 |
| 120 | 8 | 10 | Y | 8 | refl l=10,m=3 | `x^2py^4+xy^9` | `x^7y^4q+x^3y^7q` | 6.67 | T4 |
| 180 | 20 | 8 | Y | 8 | refl l=15,m=3 | `x^7y^8+x^2y^12q` | `x^14y^12q+x^10y^10` | 7.11 | T4 |
| 252 | 16 | 16 | Y | 8 | refl l=7,m=9 | `x^6y^3q+x^5y` | `xy^3+x^6y^6` | 16.25 | T4 |
| 64 | 16 | 8 | Y | 8 | refl l=4,m=4 | `xyq+xpy^3q` | `xyq+x^3py` | 16.00 | T4 (even) |
| 96 | 20 | 8 | Y | 8 | refl l=4,m=6 | `x^3py^3+xy` | `x^2y^2+x^2y^3` | 13.33 | T4 (even) |
| 120 | 14 | 10 | Y | 8 | refl l=15,m=2 | `x^11y^14q+x^7y^14` | `x^10y^6+x^11y^12q` | 11.67 | T4 (even) |
| 128 | 32 | 8 | Y | 8 | refl l=8,m=4 | `x^2py^6+x^6` | `x^2y^5+x^6yq` | 16.00 | T4 (even) |
| 144 | 16 | 12 | Y | 8 | refl l=18,m=2 | `x^11y^6+x^15y^13q` | `x^3q+x^13y^10` | 16.00 | T4 (even) |
| 28 | 4 | 5 | Y | 8 | refl l=7,m=1 | `x^2y^2+x^5y^6q` | `x^3y^3q+xy^6q` | 3.57 | EM4 |
| 36 | 12 | 4 | Y | 8 | refl l=3,m=3 | `pyq+x` | `pq+xy^2q` | 5.33 | EM4 |
| 44 | 4 | 7 | Y | 8 | refl l=11,m=1 | `x^9y^8+x^2y^3q` | `xy^2+x^7y^2` | 4.45 | EM4 |
| 60 | 12 | 5 | Y | 8 | refl l=15,m=1 | `y^2+x^6y^10` | `x^10y^9q+x^7y^7` | 5.00 | EM4 |
| 132 | 4 | 18 | Y | 8 | refl l=11,m=3 | `x^7y^2+x^6` | `x^10y^10q+x^7yq` | 9.82 | EM4 |
| 260 | 4 | 26 | <= | 8 | refl l=13,m=5 | `x^12y^7+x^7y^3` | `x^10y^2q+x^11y^9q` | 10.40 | EM4 |
| 300 | 4 | 28 | <= | 8 | refl l=25,m=3 | `x^3y^5+x^24y^4` | `x^14y^6+x^19y^2q` | 10.45 | EM4 |
| 324 | 4 | 28 | <= | 8 | refl l=27,m=3 | `x^2y^4+x^6y^17` | `x^22y^17q+x^16y^14q` | 9.68 | EM4 |
| 348 | 4 | 35 | <= | 8 | refl l=29,m=3 | `x^21y^5+x^24y^10q` | `x^27y^19q+x^22y^20` | 14.08 | EM4 |
| 380 | 4 | 58 | <= | 8 | refl l=5,m=19 | `x^4y^3+x^3py^4` | `xy+x^4y^3` | 35.41 | EM4; d<=58 implausible (typo / loose bound) — treat as unverified |
| 24 | 8 | 4 | Y | 8 | refl l=3,m=2 | `y^2q+xyq` | `x^2y+q` | 5.33 | EM4 (even) |
| 32 | 18 | 4 | Y | 8 | refl l=4,m=2 | `x^3y^3+xpy^3q` | `x^3py^3+x^3` | 9.00 | EM4 (even) |
| 112 | 8 | 14 | Y | 8 | refl l=14,m=2 | `x^5y^3+x^6q` | `x^10y^12q+x^5y^5q` | 14.00 | EM4 (even) |
| 160 | 8 | 16 | Y | 8 | refl l=10,m=4 | `x^2y^8+x^6y` | `x^7y^3q+y^5q` | 12.80 | EM4 (even) |
| 200 | 4 | 20 | <= | 8 | refl l=25,m=2 | `x^15y^2q+x^2y^15q` | `x^4y^22q+x^13y^4q` | 8.00 | EM4 (even) |
| 256 | 8 | 20 | <= | 8 | refl l=32,m=2 | `x^28y^6+x^9y^22q` | `x^13y^26q+x^15y^6q` | 12.50 | EM4 (even) |
| 312 | 6 | 30 | <= | 8 | refl l=13,m=6 | `x^11y^7q+y^5` | `xy^5+x^5y^7q` | 17.31 | EM4 (even) |
| 384 | 24 | 16 | <= | 8 | refl l=24,m=4 | `x^16y^8+x^5y^2` | `x^13y^11q+x^10y^11` | 16.00 | EM4 (even) |

#### Lu, Guo, Liu & Yang 2608.09115 (v3) Tables [newparam] / [newdecode] / [newk2] (HIGH-WEIGHT GB codes, 'divisor-driven search', cyclic Z_l)

Convention: H_X = [A|B], H_Z = [B^T|A^T], A = circ(a), B = circ(b), a = g·u, b = g·v with g | x^l - 1; w = wt(a)+wt(b) (weights 8-56, NOT low weight). 'Distances are exact' (Y). Polynomials transliterated from printed supports. Two Table-1 rows have no polynomials: [[42,12,4]] (l=21), [[62,12,4]] (l=31). Extracted by two sub-agents (agree).

| n | k | d | exact? | w | group | A | B | kd²/n | notes |
|---|---|---|---|---|---|---|---|---|---|
| 66 | 20 | 7 | Y | 19 | Z33 | `x+x^2+x^3+x^5+x^10+x^27+x^32` | `1+x+x^3+x^4+x^5+x^7+x^13+x^15+x^22+x^24+x^30+x^32` | 14.85 | g=1+x^3+x^5+x^7+x^10 |
| 42 | 16 | 6 | Y | 22 | Z21 | `1+x+x^2+x^8+x^9+x^11+x^17+x^18+x^20` | `1+x+x^2+x^3+x^4+x^6+x^8+x^11+x^12+x^14+x^18+x^19+x^20` | 13.71 | g=1+x+x^3+x^6+x^8 |
| 90 | 18 | 8 | Y | 38 | Z45 | `1+x+x^2+x^3+x^4+x^6+x^8+x^9+x^12+x^15+x^16+x^17+x^20+x^21+x^36+x^39+x^40+x^41+x^42+x^44` | `1+x+x^2+x^3+x^4+x^5+x^8+x^9+x^27+x^30+x^31+x^32+x^34+x^35+x^36+x^37+x^39+x^41` | 12.80 | g=1+x^3+x^4+x^5+x^8+x^9 |
| 42 | 14 | 6 | Y | 24 | Z21 | `x+x^3+x^5+x^6+x^7+x^9+x^12+x^13+x^14+x^16+x^17+x^19` | `x+x^2+x^4+x^6+x^11+x^12+x^13+x^14+x^15+x^18+x^19+x^20` | 12.00 | g=1+x+x^2+x^4+x^6 |
| 70 | 16 | 7 | Y | 38 | Z35 | `1+x+x^3+x^5+x^6+x^8+x^10+x^11+x^13+x^15+x^16+x^18+x^21+x^23+x^24+x^25+x^26+x^28+x^29+x^30` | `x+x^2+x^3+x^5+x^6+x^8+x^11+x^13+x^14+x^15+x^16+x^18+x^19+x^20+x^29+x^30+x^32+x^34` | 11.20 | g=1+x+x^3+x^5+x^6+x^8 |
| 90 | 20 | 7 | Y | 31 | Z45 | `1+x+x^3+x^5+x^7+x^8+x^9+x^10+x^36+x^39+x^40+x^42` | `1+x+x^2+x^4+x^5+x^8+x^10+x^21+x^22+x^23+x^25+x^26+x^29+x^30+x^32+x^34+x^35+x^38+x^40` | 10.89 | g=1+x+x^2+x^4+x^5+x^8+x^10 |
| 42 | 18 | 5 | Y | 22 | Z21 | `1+x+x^2+x^5+x^7+x^10+x^12+x^13+x^15+x^16+x^18+x^19` | `x+x^3+x^4+x^6+x^8+x^9+x^10+x^12+x^16+x^20` | 10.71 | g=1+x^2+x^3+x^4+x^6+x^7+x^8+x^9 |
| 54 | 16 | 6 | Y | 18 | Z27 | `1+x+x^2+x^3+x^4+x^5+x^6+x^8+x^25` | `1+x+x^3+x^5+x^6+x^8+x^11+x^22+x^25` | 10.67 | g=1+x^3+x^6 |
| 42 | 12 | 6 | Y | 20 | Z21 | `x+x^2+x^13+x^14+x^15+x^17+x^18+x^20` | `1+x+x^2+x^5+x^6+x^7+x^8+x^9+x^10+x^16+x^17+x^18` | 10.29 | g=1+x+x^2 |
| 70 | 20 | 6 | Y | 22 | Z35 | `x+x^2+x^3+x^4+x^5+x^6+x^7+x^8+x^9+x^11+x^13+x^15+x^30+x^32+x^34` | `1+x^2+x^5+x^8+x^10+x^11+x^34` | 10.29 | g=1+x^2+x^4+x^5+x^6+x^8+x^10 |
| 170 | 18 | 8 | Y | 56 | Z85 | `1+x^3+x^4+x^5+x^6+x^7+x^8+x^46+x^49+x^50+x^51+x^52+x^53+x^54+x^59+x^62+x^63+x^64+x^65+x^66+x^67+x^72+x^75+x^76+x^77+x^78+x^79+x^80` | `1+x^3+x^4+x^5+x^6+x^7+x^8+x^46+x^49+x^50+x^51+x^52+x^53+x^54+x^61+x^64+x^65+x^66+x^67+x^68+x^69+x^70+x^73+x^74+x^75+x^76+x^77+x^78` | 6.78 | g=1+x^3+x^4+x^5+x^6+x^7+x^8 |
| 170 | 18 | 7 | Y | 28 | Z85 | `1+x+x^3+x^9+x^22+x^23+x^25+x^31+x^38+x^39+x^41+x^47+x^69+x^70+x^72+x^78` | `1+x+x^3+x^9+x^11+x^12+x^14+x^20+x^22+x^23+x^25+x^31` | 5.19 | g=1+x+x^3+x^9 |
| 170 | 16 | 7 | Y | 31 | Z85 | `1+x+x^6+x^7+x^8+x^10+x^12+x^13+x^45+x^46+x^50+x^52+x^53` | `1+x+x^6+x^7+x^8+x^10+x^12+x^13+x^22+x^23+x^27+x^29+x^30+x^68+x^69+x^73+x^75+x^76` | 4.61 | g=1+x+x^5+x^7+x^8 |
| 170 | 18 | 6 | Y | 24 | Z85 | `1+x+x^3+x^9+x^39+x^40+x^42+x^46+x^47+x^48+x^49+x^55` | `1+x+x^3+x^9+x^31+x^32+x^34+x^40+x^54+x^55+x^57+x^63` | 3.81 | g=1+x+x^3+x^9 |
| 170 | 10 | 8 | Y | 12 | Z85 | `1+x+x^3+x^5+x^8+x^78+x^81+x^83` | `1+x+x^10+x^76` | 3.76 | g=1+x^5 |
| 42 | 6 | 7 | Y | 14 | Z21 | `1+x+x^2+x^3+x^4+x^5+x^14+x^17` | `1+x^2+x^11+x^12+x^13+x^19` | 7.00 | g=1+x+x^2 |
| 54 | 6 | 8 | Y | 16 | Z27 | `1+x+x^3+x^4+x^10+x^13+x^20+x^23` | `1+x^2+x^3+x^6+x^9+x^19+x^22+x^26` | 7.11 | g=1+x^3 |
| 66 | 6 | 8 | Y | 14 | Z33 | `1+x^3+x^5+x^8+x^14+x^17+x^23+x^26` | `x+x^2+x^3+x^4+x^5+x^30` | 5.82 | g=1+x^3 |
| 78 | 6 | 8 | Y | 12 | Z39 | `x+x^2+x^3+x^36+x^37+x^38` | `x^3+x^15+x^18+x^21+x^24+x^36` | 4.92 | g=1+x^3 |
| 90 | 4 | 10 | Y | 14 | Z45 | `x+x^3+x^4+x^43+x^44` | `1+x+x^2+x^7+x^8+x^9+x^38+x^39+x^40` | 4.44 | g=1+x+x^2 |
| 90 | 2 | 10 | Y | 12 | Z45 | `x+x^27+x^29+x^44` | `1+x+x^4+x^5+x^23+x^24+x^27+x^28` | 2.22 | g=1+x |
| 90 | 10 | 8 | Y | 20 | Z45 | `1+x^4+x^6+x^7+x^20+x^23+x^24+x^28+x^31+x^32` | `1+x^4+x^6+x^7+x^19+x^22+x^23+x^29+x^32+x^33` | 7.11 | g=1+x^3+x^4 |
| 170 | 10 | 6 | Y | 12 | Z85 | `1+x^5+x^11+x^16+x^74+x^79` | `1+x^5+x^27+x^32+x^58+x^63` | 2.12 | g=1+x^5 |
| 170 | 2 | 6 | Y | 8 | Z85 | `1+x+x^41+x^42+x^44+x^45` | `x^2+x^84` | 0.42 | g=1+x |
| 170 | 10 | 7 | Y | 12 | Z85 | `1+x+x^5+x^72+x^76+x^77` | `1+x^5+x^36+x^41+x^72+x^77` | 2.88 | g=1+x^5 |
| 46 | 2 | 8 | Y | 14 | Z23 | `1+x+x^12+x^13+x^17+x^19` | `1+x+x^4+x^5+x^8+x^9+x^12+x^13` | 2.78 | g=1+x |
| 50 | 2 | 9 | Y | 16 | Z25 | `1+x+x^10+x^11+x^14+x^15+x^21+x^22` | `1+x+x^10+x^11+x^16+x^17+x^19+x^20` | 3.24 | g=1+x |
| 54 | 2 | 9 | Y | 16 | Z27 | `1+x+x^2+x^3+x^13+x^14+x^16+x^17` | `1+x+x^2+x^3+x^10+x^11+x^19+x^20` | 3.00 | g=1+x |
| 66 | 2 | 9 | Y | 16 | Z33 | `1+x+x^4+x^5+x^21+x^22+x^25+x^26` | `1+x+x^5+x^6+x^20+x^21+x^25+x^26` | 2.45 | g=1+x |
| 70 | 2 | 10 | Y | 16 | Z35 | `1+x+x^3+x^4+x^11+x^12+x^14+x^15` | `1+x+x^14+x^15+x^18+x^19+x^31+x^32` | 2.86 | g=1+x |
| 78 | 2 | 9 | Y | 14 | Z39 | `1+x+x^16+x^18+x^33+x^34` | `1+x+x^14+x^15+x^19+x^20+x^33+x^34` | 2.08 | g=1+x |

Six further rows printed only as supports of g, u, v (a = g·u, b = g·v; NOT expanded here): [[66,4,8]] w=13: g{0,1,2}, u{0,1,17}, v{0,1,4,30}; [[66,20,5]] w=15: g{0,3,5,7,10}, u{1,3,4,8,9,11,12,14,19}, v{0,1,3,4,7,9,10,14,15,16,18,21}; [[66,20,6]] w=22: g{0,3,5,7,10}, u{0,1,7}, v{0,2,16}; [[90,16,6]] w=31: g{0,1,3,4,5,7,8}, u{5,6,10,15,16,21,25,30,31,36}, v{0,18,27}; [[90,18,6]] w=30: g{0,3,4,5,8,9}, u{0,2,3,5,9,17,18,20,32,33,35}, v{0,2,3,5,9,17,18,20,24,30,32,33,35}; [[90,20,6]] w=28: g{0,1,3,4,5,7,8}, u{4,5,6,11,15,20,21,26,30,35,36}, v{0,14,31}. The paper also gives a NON-ABELIAN coset-2BGA [[48,10,6]] w=8 (exact) on SmallGroup(48,10) with |H|=2, a=[1,6,26,28], b=[1,6,7,8] (GAP indices, ambiguous without GAP), and seven [[48,9,5]] classes.

#### Jacob, McLauchlan & Browne 2508.08191 (v3) Tab. [codes_2] + Tab. 1 (TRIVARIATE TRICYCLE codes, weight-3 polynomials; 3-block, not two-block)

Convention: G = Z_l x Z_m x Z_p, x = S_l⊗1⊗1, y = 1⊗S_m⊗1, z = 1⊗1⊗S_p. H_X = [A B C], H_Z = [[0,C^T,B^T],[C^T,0,A^T],[B^T,A^T,0]], meta-checks M_Z = [A^T B^T C^T], n = 3lmp. X-check weight 9, Z-check weight 6 (w = 9/6). d = d_Z (the smaller); dX in notes. Distances from DistM4RI connected-cluster, BP+OSD and QDistRnd, per-code method not stated (P; '<=' = upper bound). Third polynomial C in notes. Extracted by sub-agent.

| n | k | d | exact? | w | group | A | B | kd²/n | notes |
|---|---|---|---|---|---|---|---|---|---|
| 36 | 6 | 4 | P | 9/6 | Z3xZ2xZ2 | `1+x+x^2z` | `1+xy+x^2y` | 2.67 | C=`1+xyz+x^2`; dX=8 |
| 72 | 6 | 6 | P | 9/6 | Z4xZ3xZ2 | `1+y+xy^2` | `1+yz+x^2y^2` | 3.00 | C=`1+xy^2z+x^2y`; dX=12; d_circ<=4, pseudo-thr 0.9e-3 (X memory) |
| 81 | 6 | 6 | P | 9/6 | Z3xZ3xZ3 | `1+x+xy` | `1+y+yz` | 2.67 | C=`1+z+x`; dX=12 |
| 126 | 6 | 8 | P | 9/6 | Z7xZ3xZ2 | `1+y^2+x^4y` | `1+xy+x^4y^2z` | 3.05 | C=`1+x^2yz+x^2y^2`; dX<=22 |
| 135 | 12 | 6 | P | 9/6 | Z5xZ3xZ3 | `1+x+x^3y^2` | `1+xz^2+x^2yz` | 3.20 | C=`1+xy^2z+x^2yz`; dX=14 |
| 180 | 12 | 8 | P | 9/6 | Z5xZ4xZ3 | `1+x^2y^3z+x^4y` | `1+x^3+x^4z^2` | 4.27 | C=`1+x^3y^3+x^4yz^2`; dX<=20; d_circ<=7, pseudo-thr 1.7e-3 |
| 288 | 6 | 10 | P | 9/6 | Z8xZ4xZ3 | `1+yz+x^7yz^2` | `1+x^2yz+x^5z^2` | 2.08 | C=`1+x^2y^3z+x^6y^2z^2`; dX<=45 |

Above n = 400: [[432,12,12]] (6,6,4), d_circ <= 10; also n = 588, 648.

#### Jacob, McLauchlan & Browne 2508.08191 Tab. [222_CCZ] ((2,2,2) TT codes with transversal/logical CCZ; X-weight 6, Z-weight 4)

Same convention as above.

| n | k | d | exact? | w | group | A | B | kd²/n | notes |
|---|---|---|---|---|---|---|---|---|---|
| 21 | 3 | 3 | P | 6/4 | Z7 (l,m,p=7,1,1) | `1+x` | `1+x^5` | 1.29 | C=`1+x^4`; dX=6 |
| 36 | 3 | 3 | P | 6/4 | Z3xZ2xZ2 | `1+xyz` | `1+x^2z` | 0.75 | C=`1+x^2y`; dX=8 |
| 36 | 3 | 4 | P | 6/4 | Z12 | `1+x^3` | `1+x^11` | 1.33 | C=`1+x^7`; dX=8 |
| 48 | 3 | 4 | P | 6/4 | Z4xZ2xZ2 | `1+x` | `1+xz` | 1.00 | C=`1+xy`; dX=8 |
| 54 | 3 | 4 | P | 6/4 | Z3xZ3xZ2 | `1+yz` | `1+xz` | 0.89 | C=`1+xyz`; dX=9 |
| 60 | 3 | 4 | P | 6/4 | Z5xZ2xZ2 | `1+xz` | `1+xy` | 0.80 | C=`1+xyz`; dX=12 |
| 81 | 3 | 5 | P | 6/4 | Z9xZ3 | `1+x` | `1+x^8y` | 0.93 | C=`1+x^5y^2`; dX=15 |
| 90 | 3 | 5 | P | 6/4 | Z5xZ3xZ2 | `1+x` | `1+xy` | 0.83 | C=`1+x^2y^2z`; dX=15 |
| 114 | 3 | 6 | P | 6/4 | Z38 | `1+x` | `1+x^31` | 0.95 | C=`1+x^27`; dX=19 |
| 210 | 3 | 7 | P | 6/4 | Z70 | `1+x` | `1+x^16` | 0.70 | C=`1+x^25`; dX<=28 |

#### Jacob, McLauchlan & Browne 2508.08191 Tab. [high_rate_CCZ] (cup-product TT codes; <g> = sum_{k=1..ord g} g^k, A = A_in + A_out (+ A_free))

Same convention; w_X = wt(A)+wt(B)+wt(C) computed by sub-agent from stated (non-overlapping) terms.

| n | k | d | exact? | w | group | A | B | kd²/n | notes |
|---|---|---|---|---|---|---|---|---|---|
| 36 | 3 | 4 | P | X14 | Z12 | `A_in=<x^3>, A_out=x<x^3>, A_free=x^2(1+x^3)` | `1+x` | 1.33 | C=`1+x^5`; dX=6 |
| 60 | 12 | 3 | P | X16 | Z20 | `A_in=<x^5>, A_out=x<x^5>, A_free=x^3(1+x^15)+x^14(1+x^5)` | `1+x^8` | 1.80 | C=`1+x^16`; dX=5 |
| 96 | 24 | 4 | P | X18 | Z4xZ4xZ2 | `A_in=(1+z)(1+y), A_out=x<y>` | `B_in=(1+x)(1+x^2y^2), B_out=y<x>` | 4.00 | C=`1+y`; dX=4 |
| 96 | 12 | 4 | P | X12 | Z4xZ4xZ2 | `A_in=(1+z)(1+y), A_out=x<y>` | `1+x` | 2.00 | C=`1+xy^2z`; dX=8 |
| 144 | 24 | 3 | P | X12 | Z4xZ4xZ3 | `A_in=(1+y)(1+x^2), A_out=z<y>` | `1+z` | 1.50 | C=`1+x^2y^2z`; dX=6 |
| 192 | 3 | 8 | P | X22 | Z8xZ8 | `A_in=<y>, A_out=x<y>, A_free=x^2(1+y)` | `1+y` | 1.00 | C=`1+x`; dX=8 |

Also [[432,72,3]] (n > 400). Tab. [422_codes] (5 codes, all d_Z = 2) omitted.

#### Menon, Bonilla-Ataides, Mehta, Gu, Tan & Lukin 2508.10714 (v2, PRX 2026 'Magic tricycles') Tabs. [all_codes] + [code_polys] (tricycle codes with transversal CCZ)

Convention: H_X = [A^T B^T C^T], H_Z = [[C,0,A],[0,C,B],[B,A,0]] (transpose of the Jacob convention); G = Z_l x Z_m x Z_n, n_qubits = 3|G|, d = D_Z. Weights: 4-2-2: X 8, Z 6/4/6; 4-4-2: X 10, Z 6/6/8; 4-4-4: X 12, Z 8. Distances exact (SAT/MIP) when feasible, else biased Monte-Carlo estimate (>99.9% confidence); per-row method not stated -> 'Y/MC'. Third polynomial c in notes. Extracted by sub-agent.

| n | k | d | exact? | w | group | A | B | kd²/n | notes |
|---|---|---|---|---|---|---|---|---|---|
| 48 | 6 | 4 | Y/MC | 8 (4-2-2) | Z2xZ2xZ4 | `y+z+xz+xyz^2` | `yz^2+yz^3` | 2.00 | c=`y+xyz`; DX=8; CCZ depth 8 |
| 84 | 6 | 5 | Y/MC | 8 (4-2-2) | Z2xZ2xZ7 | `y+z+xz+xyz^2` | `z^3+xz^4` | 1.79 | c=`y+yz^4`; DX=12 |
| 108 | 6 | 6 | Y/MC | 8 (4-2-2) | Z3xZ3xZ4 | `x+z^2+yz+x^2yz^3` | `y^2z+x^2yz^3` | 2.00 | c=`x^2+x^2yz^2`; DX=12 |
| 240 | 6 | 8 | Y/MC | 8 (4-2-2) | Z4xZ4xZ5 | `xy^2z^3+xy^3z^4+x^2y^2z+x^2y^3z^2` | `y^3+x^2yz^2` | 1.60 | c=`xz^4+x^3y^3z`; DX<=22 |
| 108 | 12 | 4 | Y/MC | 10 (4-4-2) | Z3xZ3xZ4 | `z+xz^3+xyz^2+x^2y` | `y^2+y^2z^3+xy^2z+xy^2z^2` | 1.78 | c=`z+xyz^3`; DX=11 in Tab. all_codes but (6,4) in Tab. code_polys (paper inconsistent) |
| 180 | 12 | 6 | Y/MC | 10 (4-4-2) | Z3xZ4xZ5 | `yz^3+y^3+x^2yz^3+x^2y^3z` | `xyz^4+xy^2z^2+x^2yz+x^2y^2z^4` | 2.40 | c=`z^4+x^2z`; DX=15 |
| 108 | 15 | 6 | Y/MC | 12 (4-4-4) | Z3xZ3xZ4 | `y+y^2z+xyz^3+x^2y^2z^2` | `z^2+xy+xy^2z+x^2z^3` | 5.00 | c=`yz^3+y^2z+x^2+x^2y^2z^2`; DX=12; CCZ by numerical Leibniz rule |
| 270 | 24 | 8 | Y/MC | 12 (4-4-4) | Z3xZ5xZ6 | `z^4+y^3+xy^2+x^2yz^4` | `y^3+y^3z+xy^4+x^2y^2z` | 5.69 | c=`yz^4+y^2z+xy^2z+xy^3z^4`; DX=15 |
| 324 | 12 | 12 | <= | 12 (4-4-4) | Z3xZ4xZ9 | `y^2z+xyz+x^2z^6+x^2y^3z^5` | `z^4+z^5+xyz^7+x^2y^3z^2` | 5.33 | c=`yz^7+xz^7+xy^3+x^2y^2`; DX<=32 |

#### Menon+ 2508.10714 v1 Tab. [code_polys] (4-4-4 tricycle codes; polynomials dropped in v2)

Same convention as v2.

| n | k | d | exact? | w | group | A | B | kd²/n | notes |
|---|---|---|---|---|---|---|---|---|---|
| 72 | 6 | 6 | Y/MC | 12/8 | Z2xZ3xZ4 | `yz^2+xz+xyz^2+xy^2z` | `y+xz+yz^2+xyz^3` | 3.00 | c=`y+yz^3+xyz^2+xy^2z`; DX=12 |
| 144 | 6 | 10 | Y/MC | 12/8 | Z6xZ8 (bivariate) | `y^7+xy^3+x^4y^5+x^5y^6` | `x^5+xy+x^2y^6+x^4y^2` | 4.17 | c=`1+x^3y^7+x^4y^2+x^5y^2`; DX<=24 |
| 192 | 27 | 8 | Y/MC | 12/8 | Z4xZ4xZ4 | `x^2+x^3+y^3+z^3` | `x^3+y^2+y^3+z` | 9.00 | c=`x+y^3+z+z^2`; DX=16; v2 circuit sim pL(X/Z) at p=1e-3 = 2e-4 / 2e-5 |
| 375 | 15 | 15 | <= | 12/8 | Z5xZ5xZ5 | `y^4+z+z^3+z^5` | `x^2+x^3+x^5+z` | 9.00 | c=`x^2+x^3+z+z^3`; DX<=25; printed z^5, x^5 (= 1 in Z5) verbatim |

#### Mian, Gwilliam & Krastanov 2601.18879 (v2) Tabs. 2-5 (+6 compact) (multivariate MULTICYCLE codes; 6-block Koszul, not two-block; complete single-shot)

Convention: Koszul complex of t = 4 polynomials F,G,H,I over F2[w,x,y,z]/<w^l-1, x^m-1, y^p-1, z^r-1>; qubits in K_2, n = 6|G|; P_X = ∂_2^T, P_Z = ∂_3. A column = F, B column = G; H, I in notes. Tabs. 2-4, 6: QDistRnd DistRandCSS (Q); Tab. 5: Gurobi MIP (Y). Extracted by sub-agents (Tab. 2 independently by two).

| n | k | d | exact? | w | group | A | B | kd²/n | notes |
|---|---|---|---|---|---|---|---|---|---|
| 96 | 12 | 4 | Q | 6 | Z2xZ2xZ2xZ2 | `wz+xyz` | `w+xyz` | 2.00 | Tab2; H=wyz+xy; I=xyz+xz |
| 144 | 6 | 8 | Q | 6 | Z2xZ2xZ3xZ2 | `wxy+z` | `wx+wyz` | 2.67 | Tab2; H=wx+xy^2; I=xyz+y^2z |
| 192 | 6 | 8 | Q | 6 | Z4xZ2xZ2xZ2 | `w^2xz+z` | `w^2yz+wyz` | 2.00 | Tab2; H=w^3x+xyz; I=w^3xyz+w^2z |
| 216 | 6 | 12 | Q | 6 | Z2xZ2xZ3xZ3 | `wxy+xy^2` | `wy+xyz^2` | 4.00 | Tab2; H=xy^2+y^2z; I=wxyz^2+xz |
| 240 | 12 | 7 | Q | 6 | Z5xZ2xZ2xZ2 | `w^4y+w^3x` | `w^4x+w^3` | 2.45 | Tab2; H=w^4y+wx; I=w^4z+w^2yz |
| 240 | 6 | 12 | Q | 6 | Z5xZ2xZ2xZ2 | `w^4xy+xz` | `w^3+w^2z` | 3.60 | Tab2; H=w^3xyz+wy; I=w^2y+w |
| 288 | 6 | 12 | Q | 6 | Z2xZ2xZ3xZ4 | `wxy+xz^2` | `wxz^2+wy^2` | 3.00 | Tab2; H=xyz^3+z^2; I=wxy+w |
| 288 | 12 | 8 | Q | 6 | Z2xZ2xZ2xZ6 | `wz^3+xyz^2` | `wxz^2+xyz` | 2.67 | Tab2; H=wxz^2+w; I=wyz^5+xyz^3 |
| 324 | 6 | 12 | Q | 6 | Z2xZ3xZ3xZ3 | `wxz+yz^2` | `wxyz+xz` | 2.67 | Tab2; H=wx^2z+y^2z^2; I=wxy^2+wxz^2 |
| 336 | 6 | 12 | Q | 6 | Z2xZ2xZ2xZ7 | `wx+z^2` | `wxz^2+z^6` | 2.57 | Tab2; H=wx+wz^2; I=wxz^4+wyz^6 |
| 360 | 6 | 12 | Q | 6 | Z2xZ2xZ3xZ5 | `xz^3+y^2z` | `wx+wz^2` | 2.40 | Tab2; H=wxy^2+y^2z^2; I=wxyz^3+wxz^2 |
| 360 | 12 | 11 | Q | 6 | Z2xZ2xZ3xZ5 | `y^2z+yz^3` | `wy^2z^3+xy^2` | 4.03 | Tab2; H=wz^3+xyz^2; I=wyz^3+xy^2 |
| 384 | 6 | 12 | Q | 6 | Z2xZ2xZ2xZ8 | `wz^4+xz^7` | `xz^2+yz` | 2.25 | Tab2; H=wyz^5+wyz^4; I=xyz^7+x |
| 384 | 12 | 8 | Q | 6 | Z2xZ2xZ2xZ8 | `wyz+xyz^6` | `wxyz^4+wz^6` | 2.00 | Tab2; H=wxyz^6+wz^7; I=wxy+xz^5 |
| 144 | 12 | 8 | Q | 9 | Z3xZ2xZ2xZ2 | `w^2xyz+wy+yz` | `w^2y+wxz+x` | 5.33 | Tab3; H=w^2x+wz+xy; I=w^2yz+w+1 |
| 144 | 12 | 12 | Q | 9 | Z3xZ2xZ2xZ2 | `w^2xy+w+xyz` | `w^2x+wxy+z` | 12.00 | Tab3; H=w^2xz+wx+xyz; I=w^2yz+wxy+xyz |
| 216 | 12 | 12 | Q | 9 | Z2xZ2xZ3xZ3 | `wxy^2+wz^2+x` | `wxyz^2+wy^2z+x` | 8.00 | Tab3; H=wxy^2z^2+xyz^2+xz^2; I=wz+w+xy^2z |
| 216 | 24 | 9 | Q | 9 | Z2xZ2xZ3xZ3 | `wy^2z^2+x+yz^2` | `wxz^2+xy^2z+y` | 9.00 | Tab3; H=wxy^2z+wxyz^2+wxz^2; I=wyz+xy^2+z |
| 288 | 12 | 16 | Q | 9 | Z4xZ2xZ2xZ3 | `w^2y^3z+wxyz+y^2` | `w^2yz+w+xy^3` | 10.67 | Tab3; H=w^2y^3z+wy^3+y^3z; I=w^2x+wy^3+yz |
| 324 | 12 | 20 | Q | 9 | Z2xZ3xZ3xZ3 | `wx^2y+x^2y^2+z` | `wy+xy+y^2z^2` | 14.81 | Tab3; H=wx^2z+wyz+yz^2; I=wx^2y^2+wxyz^2+xy^2z; check independently |
| 336 | 18 | 12 | Q | 9 | Z2xZ2xZ2xZ7 | `wxyz^6+wx+z^2` | `wz^3+xyz^6+z^4` | 7.71 | Tab3; H=xy+xz^5+yz^4; I=wxyz^3+wyz+wy |
| 96 | 44 | 4 | Q | 12/12 | Z2^4 | `(1+x)(1+yz)` | `(1+y)(1+zw)` | 7.33 | Tab4 (w = median/max); H=(1+z)(1+wx); I=(1+w)(1+xy) |
| 144 | 12 | 8 | Q | 9/10 | Z2xZ2xZ2xZ3 | `wxy+xyz` | `y+zx+yx+zw` | 5.33 | Tab4; H=x+zyxw; I=z+y+x+zyx |
| 192 | 12 | 12 | Q | 12/12 | Z2xZ2xZ2xZ4 | `wxyz^3+xyz+xz^3+xz^2` | `wxz^2+wyz^3+w+1` | 9.00 | Tab4; H=wy+wz^2+xyz^2+xz; I=wxyz+wy+xz^3+z^3 |
| 216 | 12 | 12 | Q | 9/10 | Z2xZ2xZ3xZ3 | `wxy+xyz` | `y+zx+yx+zw` | 8.00 | Tab4; same polynomials as the 144 row |
| 216 | 12 | 14 | Q | 12/12 | Z2xZ2xZ3xZ3 | `wz^2+xy^2z+xy^2+y` | `wxy^2z^2+wxz^2+wy^2z+y^2z^2` | 10.89 | Tab4; H=wxy^2z^2+wxyz+xy^2z+x; I=wxz^2+wyz+wz^2+xz |
| 240 | 6 | 24 | Q | 12/12 | Z2xZ2xZ2xZ5 | `wxz^4+wyz^3+wy+xyz^4` | `wyz^4+wz+w+xz^3` | 14.40 | Tab4; H=wxz+wyz^4+wz^4+xz^4; I=wxyz^4+wxyz^3+wyz+w |
| 240 | 12 | 16 | Q | 12/12 | Z2xZ2xZ2xZ5 | `wxz+wz^4+xyz^3+yz` | `wz^3+xz^4+yz^2+z` | 12.80 | Tab4; H=wz^2+xz+yz^4+z; I=xz^4+yz^3+z^3+z^2 |
| 240 | 18 | 10 | Q | 12/12 | Z2xZ2xZ2xZ5 | `wxyz^3+wxz^2+wz^3+z` | `wxyz^4+wxy+wx+xy` | 7.50 | Tab4; H=wyz^4+wyz^3+wz^2+yz^3; I=wxyz+wx+wyz^4+wz |
| 288 | 24 | 12 | Q | 12/12 | Z2xZ2xZ2xZ6 | `wxyz^5+wyz^2+xyz^4+y` | `wxyz^5+wxy+wz+xyz` | 12.00 | Tab4; H=wxyz^4+wyz^5+wyz^4+z^4; I=wxz^4+wyz^4+wz^3+z^5 |
| 324 | 18 | 18 | Q | 12/12 | Z2xZ3xZ3xZ3 | `wx^2z+wy^2z+wz^2+x^2yz^2` | `wx^2y^2z+wx^2yz^2+wxy^2z^2+x^2z` | 18.00 | Tab4; H=wxy+wxz+wx+y^2z; I printed `wx^2y^2z+w*y^2+wz+w` (ambiguous '*') |
| 384 | 24 | 16 | Q | 12/12 | Z2xZ2xZ4xZ4 | `wxyz^3+wy^2z^3+xy^2+y^2z` | `wxyz^2+wxz^3+wy^3z+xyz^3` | 16.00 | Tab4; H=wxy^3z+wxz^2+wy^2z^3+xyz^2; I=wxy^3z+wxz^3+xy^2z^2+xy^2z |
| 324 | 6 | 49 | Q | 12/12 | Z2xZ3xZ3xZ3 | `wx^2z^2+x^2y^2+x^2yz^2+y` | `wxz+wx+x^2y^2z+xy^2z` | 44.46 | Tab4; SUSPECT/implausible (dX=51); H=wyz+x^2y^2z^2+x^2z+y; I=wx^2yz^2+wxy^2z+x^2y+y^2z^2 |
| 336 | 6 | 54 | Q | 12/12 | Z2xZ2xZ2xZ7 | `wxyz^4+wz^4+yz^6+yz^3` | `wyz^6+w+xz^3+yz^5` | 52.07 | Tab4; SUSPECT; H=wz^3+xyz^6+xz^3+xz; I=wxz^4+wz^2+xyz^2+1 |
| 336 | 12 | 49 | Q | 12/12 | Z2xZ2xZ2xZ7 | `wxyz^5+wxyz+wyz^3+wz^4` | `wxz^5+xy+yz^6+z^4` | 85.75 | Tab4; SUSPECT; H=wxyz^2+wz^5+xz^3+yz^5; I=wxyz^3+wxz^3+wyz^6+y |
| 96 | 12 | 6 | Y | 12 | Z2^4 | `wy+xz+yz+z` | `wxyz+wxz+w+xyz` | 4.50 | Tab5 (MIP); H=xyz+xz+yz+1; I=wxy+wy+wz+y |
| 96 | 12 | 8 | Y | 12 | Z2^4 | `wy+wz+w+1` | `wxz+wy+wz+w` | 8.00 | Tab5; H=wxyz+w+xyz+y; I=wxyz+w+yz+1 |
| 144 | 6 | 6 | Y | 12 | Z2xZ3xZ2xZ2 | `w+x^2yz+x^2z+yz` | `x^2y+x+yz+y` | 1.50 | Tab5; H=wxy+wxz+w+x^2; I=wx^2y+x^2y+xy+yz |
| 144 | 6 | 8 | Y | 12 | Z3xZ2xZ2xZ2 | `wxyz+wxy+wy+yz` | `w^2z+xyz+xy+yz` | 2.67 | Tab5; H=w^2x+w^2yz+w^2+1; I=w^2z+wy+w+xz |
| 144 | 6 | 10 | Y | 12 | Z3xZ2xZ2xZ2 | `w^2xz+w^2yz+wx+z` | `w^2yz+wxz+w+xy` | 4.17 | Tab5; H=wxy+wy+x+y; I=wxz+xy+x+z |
| 144 | 6 | 12 | Y | 12 | Z3xZ2xZ2xZ2 | `w^2+wyz+wz+w` | `w^2z+wxy+yz+z` | 6.00 | Tab5; H=w^2y+wz+w+1; I=w^2z+wx+wy+1 |
| 144 | 12 | 6 | Y | 12 | Z3xZ2xZ2xZ2 | `w^2z+wyz+w+y` | `w^2x+w^2+wxyz+wyz` | 3.00 | Tab5; H=w^2x+wz+x+y; I=w^2z+wxy+wz+y |
| 144 | 12 | 8 | Y | 12 | Z2xZ3xZ2xZ2 | `x^2z+x^2+yz+z` | `wx^2y+w+x^2+yz` | 5.33 | Tab5; H=wxy+w+x^2y+z; I=wxyz+wxy+x^2y+xz |
| 144 | 12 | 10 | Y | 12 | Z3xZ2xZ2xZ2 | `w^2x+xyz+x+y` | `w^2y+wxy+wx+z` | 8.33 | Tab5; H=w^2yz+w^2+yz+z; I=w^2x+wxyz+wyz+1 |
| 144 | 12 | 12 | Y | 12 | Z3xZ2xZ2xZ2 | `w^2xz+w^2yz+wy+xy` | `wxy+wyz+wz+xy` | 12.00 | Tab5; H=w^2xy+w^2+wyz+1; I=w^2xz+w^2z+wxz+xy |
| 144 | 18 | 6 | Y | 12 | Z3xZ2xZ2xZ2 | `w^2xy+w^2yz+w^2+z` | `w^2xy+w^2yz+w^2y+yz` | 4.50 | Tab5; H=w^2xy+wy+w+x; I=w^2yz+wz+w+xy |

Tab. 6 (collapsed 4D onto cyclic Z_{n/6}: x = y = z = 1, all four polynomials 1+w^e, w = 6, Q, dX = dZ), compact "n,k,d: e_F,e_G,e_H,e_I [kd²/n]": 156,6,10: 16,19,22,22 [3.85] (H = I as printed); 162,6,9: 7,13,15,24 [3.00]; 168,6,9: 4,7,13,19 [2.89]; 168,12,7: 4,6,8,18 [3.50]; 174,6,11: 2,8,20,24 [4.17]; 180,6,10: 3,13,21,28 [3.33]; 186,6,11: 1,14,19,23 [3.90]; 192,6,11: 8,25,26,29 [3.78]; 198,6,10: 3,7,16,21 [3.03]; 204,6,12: 5,8,9,23 [4.24]; 210,6,11: 13,25,29,34 [3.46]; 216,6,12: 3,21,26,35 [4.00]; 222,6,11: 4,17,22,34 [3.27]; 228,6,12: 3,24,30,37 [3.79]; 234,6,13: 1,14,30,36 [4.33]; 240,6,12: 2,5,11,26 [3.60]; 246,6,12: 1,12,23,37 [3.51]; 252,6,12: 9,17,19,30 [3.43]; 258,6,12: 15,16,34,39 [3.35]; 264,6,13: 6,7,21,26 [3.84]; 270,6,14: 9,24,32,38 [4.36]; 276,6,14: 19,32,34,43 [4.26]; 282,6,13: 23,25,28,39 [3.60]; 288,6,13: 22,23,41,43 [3.52]; 294,6,14: 9,15,22,48 [4.00]; 300,6,13: 12,16,21,23 [3.38]; 306,6,14: 1,36,39,46 [3.84]; 312,6,13: 8,17,41,42 [3.25]; 318,6,13: 1,19,21,29 [3.19]; 324,6,14: 9,11,44,48 [3.63]; 330,6,14: 15,39,46,51 [3.56]; 336,6,14: 13,41,49,43 [3.50]; 342,6,14: 1,15,46,49 [3.44]; 348,6,15: 22,43,55,56 [3.88]; 354,6,15: 33,39,42,47 [3.81]; 360,6,15: 4,18,25,44 [3.75]; 366,6,15: 30,38,47,48 [3.69]; 372,6,16: 42,46,55,57 [4.13]; 378,6,17: 11,44,46,56 [4.59]; 384,6,17: 6,10,43,51 [4.52]; 390,6,15: 8,15,39,59 [3.46]; 396,6,15: 12,13,32,38 [3.41]. (Tabs. 7-11, collapsed 5D-9D, weights 8-12, kd²/n <~ 3.5, skipped.)

#### Koukoulekidis, Šimkovic, Leib & Pereira 2401.07583 (text; extended-GB family base code)

| n | k | d | exact? | w | group | A | B | kd²/n | notes |
|---|---|---|---|---|---|---|---|---|---|
| 10 | 2 | 3 | Y | 6 | Z5 | `1+x^4` | `1+x+x^2+x^4` | 1.80 | smallest GB code with d>=3 (exhaustive search over l<=10); H = [(A:B),0; 0,(B^T:A^T)]; extended family l=5m has no tabulated distances |

### Highlights (most important new rows; verbatim from the tables above)

Weight 6 (the main target):
* [[336,12,20]] w6, NON-ABELIAN, MILP-exact, kd²/n 14.29 — Qian & Li 2608.08996: Z84⋊_29 Z4 (x^84=s^4=e, sxs^-1=x^29), K=<x^42>; A = `e+xs+x^3`, B = `s^3+x^2+x^4` (left/right convention not stated in the paper).
* [[224,12,16]] w6, coset code (G = SmallGroup(224,53) = C7 x ((C4 x C4) ⋊ C2), H = C2 non-normal, s=1), exact, kd²/n 13.71 — Aydin-Tamo-Barg 2606.17268; a = `[1, 81, 186]`, b = `[1, 16, 47]` (GAP 4.14.0 coset indices). Beats GT-optimal BB [[224,6,20]] (10.71).
* [[280,12,16]] w6, NON-ABELIAN 2BGA on C14 x D10 = SmallGroup(140,9) (5-cover of [[56,12,4]]), exact, kd²/n 10.97 — 2606.17268; a = `[1, 4, 38]`, b = `[1, 7, 52]` (GAP Elements indices). Beats GT [[280,6,<=22]].
* [[216,12,14]] w6 (non-abelian Z2xZ2x((Z3xZ3)⋊Z3) and abelian Z6xZ6xZ3 lifts of the [[72,12,6]] BB code), exact, kd²/n 10.89 — Hirasaki & Lee 2607.28621; lifted polynomials NOT printed.
* [[168,16,10]] w6, NON-ABELIAN 2BGA on C14 x S3 = SmallGroup(84,13), exact, 9.52 — 2606.17268; a = `[1, 12, 48]`, b = `[1, 7, 44]`.
* [[96,8,10]] w6 coset code (SmallGroup(384,512), H = C8), exact, 8.33 — 2606.17268; a = `[1, 9, 87]`, b = `[1, 21, 23]`. Beats best published w6 BB at n = 96.
* Lin & Pryadko 2306.16400 has NO W = 6 table; its new rows (Tables II/III) are W = 8 with kd = n and d <= 10 (best: D32 [[64,8,8]], kd²/n 8.00; C14xC2 [[56,4,10]], 7.14).

Weight 8 (exact unless noted): [[168,20,14]] NON-ABELIAN C7 x (C3 ⋊ C4) (23.33, 2606.17268); [[204,24,15]] cyclic C102 (26.47, 2606.17268); [[378,32,19]] He(Z3) x Z7 (30.56, 2608.08996); [[288,24,18]] effectively Z12xZ12 (w 5+3, 27.00, 2608.08996); [[224,22,16]] effectively Z28xZ4 (25.14, 2608.08996); [[168,16,15]] coset SmallGroup(168,33) (21.43); [[160,12,16]] coset SmallGroup(320,10) (19.20); [[124,12,14]] C62 (18.97); [[144,16,12]] A4 x C6 (16.00, 2609.36213, a right / b left).

### Checked sources

| arXiv id | title | usable table? | rows extracted |
|---|---|---|---|
| 2306.16400 | Lin & Pryadko, Quantum two-block group algebra codes (PRA 109, 022407) | yes: Tables I-III + Example 1 (all W=8; no W=6 table; arXiv v1 only, PRA text paywalled) | 36 new (T2 17, T3 18, Ex 1) + 17 T1 re-listings with presentations/typo flags |
| 2305.06890 | Wang, Lin & Pryadko, Abelian and non-abelian quantum two-block codes | no table; only the A4 [[24,5,3]] example (shared with 2306.16400) | 0 new |
| 2502.19406 | Lin, Liu, Lim & Pryadko, Single-shot and two-shot decoding with generalized bicycle codes | yes (App. longtable) | 33 rows not in literature.md |
| 2506.16910 | Lin, Lim, Kovalev & Pryadko, Abelian multi-cycle codes for single-shot error correction | yes (6-block, not two-block) | 11 |
| 2606.05044 | Davenport, Blue & Chuang, GB codes as cyclic submodules and their automorphism structure | yes (MCR GB; w = min generator weight) | 14 |
| 2401.07583 | Koukoulekidis et al., Small quantum codes from algebraic extensions of GB codes | no (figures); 1 base code from text | 1 |
| 2606.17268 | Aydin, Tamo & Barg, Breaking the bicycle frame: coset-based quantum LDPC codes | yes (3 tables; elements as GAP 4.14.0 index arrays only) | 7 + 55 + 53 = 115 |
| 2608.08996 | Qian & Li, Multi-agent discovery of practical quantum LDPC codes | yes | 27 |
| 2609.36213 | C. Liu, Bivariate bicycle codes over group algebras | examples only | 3 |
| 2607.28621 | Hirasaki & Lee, Lifting lifted product codes | partial (lifted polynomials not given) | 12 |
| 2607.27644 | Hong, QLDPC codes with design rate 1/5 ... below 1000 physical qubits (ZSZ-LP) | yes (5-block, non-abelian metacyclic) | 9 |
| 2602.15372 | Liu, Xu & Xu, Self-dual stacked quantum LDPC codes | yes (T1-T4, EM1-EM4) | 94 |
| 2602.11457 | Webster et al., The Pinnacle architecture (GB codes) | yes | 4 |
| 2608.09115 | Lu, Guo, Liu & Yang, Quantum bicycle LDPC codes with high kd²/n from divisor-driven search | yes (high-weight GB) | 31 (+6 support-only, +2 without polynomials) |
| 2508.08191 | Jacob, McLauchlan & Browne, Single-shot decoding and FT gates with trivariate tricycle codes | yes (3-block) | 23 (5 rows with d=2 omitted) |
| 2508.10714 | Menon et al., Magic tricycles (PRX 2026) | yes (v2 + v1 polynomials) | 13 |
| 2601.18879 | Mian, Gwilliam & Krastanov, Multivariate multicycle codes for complete single-shot decoding | yes (6-block) | 46 + 42 compact (Tab. 6) |
| 2310.15092 | Willenborg, Borello, Horlemann & Islam, Dihedral quantum codes | no (only qudit [[360,192,8]]_11 MDPC) | 0 |
| 2409.09830 / 2503.03936 | Pacenti & Vasic, Quantum Margulis codes / Construction and decoding of ... | no (n,k, girth only; no d, no generators) | 0 |
| 2602.12228 | Non-abelian QLDPC codes and non-Clifford operations from gauging logical gates | no | 0 |
| 2606.24808 | LLM discovery of QLDPC codes through structured concept evolution | no rows with n <= 400 | 0 |
| 2608.27565 | Spectral theory of semisimple bivariate bicycle codes | no (theory) | 0 |
| 2606.08771 | Algebra of bivariate-bicycle surface codes | planar BB-surface codes with boundary; comparison table only | 0 |
| 2607.27521 | Floquet abelian multicycle codes | Floquet, not static two-block | 0 |
| 2504.09171 | Steffan et al., Tile codes | parameters only, layouts as pictures; new [[512,18,19]] w8 (n > 400) | 0 |
| 2502.07150 | Malcolm et al., SHYPS | parameters only: SHYPS(3)=[[49,9,4]], SHYPS(4)=[[225,16,8]], SHYPS(5)=[[961,25,16]] (subsystem, weight-3 gauge). literature.md note is wrong: [[81,9,3]] and [[784,16,7]] are surface-code baselines | 0 |
| 2603.17703 | Galimova, (independent) trivariate bicycle codes | already in literature.md | 0 |
| 2504.18360, 2508.09082, 2602.04443, 2608.15754 | GB-Kitaev generalization; low-connectivity GB; qudit twisted torus; RL decoder-fit codes | no usable binary tables | 0 |

Not found / not opened: Guo-Hong-Kaufman-Lucas ZSZ codes (PRX Quantum 2026; w6 [[80,2,8]], [[144,12,8]], [[288,12,8]] quoted in 2601.18879), arXiv id not located; 'tripier2026' trapped-ion paper (source of GB [[102,22,9]]) not checked; Kasai 2504.17790 (non-binary affine-permutation 2BGA) not checked.
Further leads found at the end, NOT extracted (time): Okada & Kasai 2607.14091 'Pair-partition constructions for CPM-based QLDPC codes' (36 exact-distance CPM codes; cited by 2608.08996 as the strongest w8 benchmark), Okada & Kasai 2605.23894 / 2606.27130 (two-branch finite-field CSS LDPC bases, cited as strongest exact w10), 2609.24201 'High-rate quasi-dyadic QLDPC codes'. These are quasi-cyclic CPM / multi-block constructions, not necessarily two-block.
