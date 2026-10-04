#!/usr/bin/env python3
"""Theorem 3 bounds vs the measured single-fault table of research/shor-noise.md.

For each of the 15 depolarizing instances (n, r from the shor-noise main table)
compute, per window (start: i < floor(t - 2 log2 r); end: last nu2(r) rounds;
middle: the rest):
  * T3(a) end window: P(ok | any fault) = S0 exactly.
  * T3(b) phase-type faults: lower bound L_i (8/pi^2 if D'>=1, 1-1/(2(ceil D'-3)) if ceil D'>=4),
    D' = 2^(t-i-2)/r^2, averaged over the rounds of the window.
  * T3(d) persistent-dephasing X/Y faults: upper bound U_i = min(1, 1/r + r_odd 2^(i+1+nu-t)).
Rounds are weighted equally (every round has the same oracle size up to O(1)).
"""
import math
inst = [  # n, N, r, S0 (from the shor-noise main table)
 (10,899,210,.995),(11,1711,406,1.0),(12,2867,690,.993),(13,4343,350,1.0),(14,8453,4134,.980),
 (15,29083,4788,.997),(16,34387,17000,.983),(17,88433,14630,1.0),(18,200479,16632,1.0),
 (19,471203,117432,.990),(20,821749,34164,1.0),(21,1128437,187666,.993),(22,2957047,369200,1.0),
 (23,6226057,86400,1.0),(24,10161323,1692480,1.0)]
def lower(t,i,r):
    if i+2>t: return 0.0
    d=2.0**(t-i-2)/(r*r); c=math.ceil(d); b=0.0
    if d>=1: b=8/math.pi**2
    if c>=4: b=max(b,1-1/(2*(c-3)))
    return b
tot={'s':[0,0,0],'m':[0,0,0],'e':[0,0,0]}
print("n  r        nu  start  mid  end | Zstart_lower | XY_start_upper XY_mid_upper | S0")
for n,N,r,s0 in inst:
    t=2*n; nu=(r&-r).bit_length()-1; ro=r>>nu
    ws=math.floor(t-2*math.log2(r)); we=t-nu
    S=[i for i in range(t) if i<ws]; E=[i for i in range(t) if i>=we]; M=[i for i in range(t) if ws<=i<we]
    zl=sum(lower(t,i,r) for i in S)/max(1,len(S))
    up=lambda i:min(1.0,1/r+ro*2.0**(i+1+nu-t))
    us=sum(up(i) for i in S)/max(1,len(S)); um=sum(up(i) for i in M)/max(1,len(M))
    print(f"{n:2d} {r:8d} {nu:2d} {len(S):5d} {len(M):4d} {len(E):4d} | {zl:12.3f} | {us:14.4f} {um:12.4f} | {s0}")
    for k,L,v in (('s',S,zl),('m',M,um),('e',E,s0)):
        tot[k][0]+=len(L); tot[k][1]+=v*len(L)
    tot['s'][2]+=us*len(S)
print(f"pooled: Z start lower bound {tot['s'][1]/tot['s'][0]:.3f} (measured 0.976);"
      f" X/Y start upper (dephasing) {tot['s'][2]/tot['s'][0]:.4f} (measured 0.032);"
      f" X/Y middle upper {tot['m'][1]/tot['m'][0]:.3f} (measured 0.055);"
      f" end = S0 avg {tot['e'][1]/tot['e'][0]:.3f} (measured X/Y .997, Z .997)")
