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
* Malcolm+ 2502.07150: SHYPS codes ([49,9,4] classical → [[81,9,3]], [[784,16,7]]), not BB.
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
