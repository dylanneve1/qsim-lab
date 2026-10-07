"""Segment P6 into ring bursts (B) and gaps (Q) using CZ ASAP layers.
Burst = maximal run of CZ layers where N-CZs dominate (>= 2/3). Each CZ gets epoch: 2k for burst k, 2k+1 for the gap after it.
N-CZs inside gaps / X-CZs inside bursts are assigned by their layer (reported). u3 gates are assigned to the epoch of the
next CZ on their wire (the last u3 on each wire -> final epoch)."""
from parse import *
from collections import Counter
G=load()
lay=[0]*62; CL={}
for i,g in enumerate(G):
    if g[0]=='cz':
        a,b=g[1]; l=max(lay[a],lay[b]); CL[i]=l; lay[a]=lay[b]=l+1
D=max(lay)
cN=Counter(); cX=Counter()
for i,l in CL.items():
    (cN if cls(*G[i][1])=='N' else cX)[l]+=1
isb=[cN[l]>=2*cX[l] and cN[l]>0 for l in range(D)]
# epoch per layer
ep=[]; e=0 if isb[0] else 1
for l in range(D):
    if l>0 and isb[l]!=isb[l-1]: e+=1
    ep.append(e)
def epochs():
    E={}
    for i,l in CL.items(): E[i]=ep[l]
    # u3 -> next cz epoch on wire
    nxt={}
    for i in range(len(G)-1,-1,-1):
        g=G[i]
        if g[0]=='cz':
            for q in g[1]: nxt[q]=E[i]
        else:
            E[i]=nxt.get(g[1][0], ep[-1]+1)
    return E
if __name__=='__main__':
    print('layers',D,'epochs',ep[-1]+1)
    print(''.join('B' if b else '.' for b in isb))
    E=epochs()
    # misfits
    mis=Counter()
    for i,l in CL.items():
        k=cls(*G[i][1]); mis[(ep[l]%2, k)]+=1
    print('epoch parity (0=burst) x class:',dict(mis))
    # monotonic per wire?
    bad=0
    last={q:-1 for q in range(62)}
    for i,g in enumerate(G):
        for q in g[1]:
            if E[i]<last[q]: bad+=1
            last[q]=E[i]
    print('non-monotone ops',bad)

def epochs2():
    """Class-based: N CZs -> nearest burst epoch, X/L CZs -> nearest gap epoch (by ASAP layer); u3 -> epoch of next CZ on wire."""
    # layer ranges per epoch
    rng={}
    for l,e in enumerate(ep): rng.setdefault(e,[l,l]); rng[e][1]=l
    def nearest(l,parity):
        best=None
        for e,(a,b) in rng.items():
            if e%2!=parity: continue
            d=0 if a<=l<=b else min(abs(l-a),abs(l-b))
            if best is None or d<best[0] or (d==best[0] and e<best[1]): best=(d,e)
        return best[1]
    E={}
    for i,l in CL.items():
        k=cls(*G[i][1]); E[i]=nearest(l,0 if k=='N' else 1)
    nxt={}
    for i in range(len(G)-1,-1,-1):
        g=G[i]
        if g[0]=='cz':
            for q in g[1]: nxt[q]=E[i]
        else: E[i]=nxt.get(g[1][0], ep[-1]+1)
    return E

def epochs3(verbose=False):
    """epochs() with non-ring CZs at the edge of a burst moved into the adjacent gap when causally allowed
    (last CZ on both wires within the burst -> next gap; first on both wires -> previous gap). u3 re-assigned to next CZ."""
    E=epochs()
    czi=[i for i,g in enumerate(G) if g[0]=='cz']
    moved=1; total=0
    while moved:
        moved=0
        seq={q:[] for q in range(62)}
        for i in czi:
            for q in G[i][1]: seq[q].append(i)
        pos_={}
        for q in range(62):
            for j,i in enumerate(seq[q]): pos_[(i,q)]=j
        for i in czi:
            e=E[i]
            if e%2 or cls(*G[i][1])=='N': continue
            a,b=G[i][1]
            def nxt(q):
                j=pos_[(i,q)]; return E[seq[q][j+1]] if j+1<len(seq[q]) else 10**9
            def prv(q):
                j=pos_[(i,q)]; return E[seq[q][j-1]] if j>0 else -1
            if nxt(a)>e and nxt(b)>e: E[i]=e+1; moved+=1
            elif prv(a)<e and prv(b)<e and e>0: E[i]=e-1; moved+=1
        total+=moved
    nxt={}
    for i in range(len(G)-1,-1,-1):
        g=G[i]
        if g[0]=='cz':
            for q in g[1]: nxt[q]=E[i]
        else: E[i]=nxt.get(g[1][0], ep[-1]+1)
    if verbose: print('moved',total)
    return E

def epochs4(verbose=False):
    """epochs3() plus: ring (N) CZs sitting at the edge of a gap are moved into the adjacent burst when causally allowed."""
    E=epochs3()
    czi=[i for i,g in enumerate(G) if g[0]=='cz']
    moved=1; total=0
    while moved:
        moved=0
        seq={q:[] for q in range(62)}
        for i in czi:
            for q in G[i][1]: seq[q].append(i)
        pos_={}
        for q in range(62):
            for j,i in enumerate(seq[q]): pos_[(i,q)]=j
        for i in czi:
            e=E[i]
            if e%2==0 or cls(*G[i][1])!='N': continue
            a,b=G[i][1]
            def nxt(q):
                j=pos_[(i,q)]; return E[seq[q][j+1]] if j+1<len(seq[q]) else 10**9
            def prv(q):
                j=pos_[(i,q)]; return E[seq[q][j-1]] if j>0 else -1
            if prv(a)<e and prv(b)<e: E[i]=e-1; moved+=1
            elif nxt(a)>e and nxt(b)>e and e+1<=ep[-1]: E[i]=e+1; moved+=1
        total+=moved
    nxt={}
    for i in range(len(G)-1,-1,-1):
        g=G[i]
        if g[0]=='cz':
            for q in g[1]: nxt[q]=E[i]
        else: E[i]=nxt.get(g[1][0], ep[-1]+1)
    if verbose: print('moved N',total)
    return E
