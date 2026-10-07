"""Assign each CZ to an epoch: per wire, its CZ sequence splits into stretches of ring-nn (N) and non-ring (X/L) CZs.
Epoch index of a stretch counted per wire (N stretches even, non-ring odd). Check agreement between the 2 wires of each CZ."""
from parse import *
from collections import Counter
G=load()
czs=[(i,g[1]) for i,g in enumerate(G) if g[0]=='cz']
kind={i:(cls(*q)=='N') for i,q in czs}
seq={q:[] for q in range(62)}
for i,(a,b) in czs: seq[a].append(i); seq[b].append(i)
ep={}  # (cz index, wire) -> epoch
for q in range(62):
    e=None; prev=None
    for i in seq[q]:
        k=kind[i]
        if prev is None: e=0 if k else 1
        elif k!=prev: e+=1
        prev=k; ep[(i,q)]=e
agree=Counter(); diffs=Counter()
for i,(a,b) in czs:
    d=ep[(i,a)]-ep[(i,b)]; diffs[d]+=1
print('epoch diff between wires of a CZ:',sorted(diffs.items()))
print('stretches per wire:',sorted(Counter(max(ep[(i,q)] for i in seq[q])+1 for q in range(62)).items()))
