"""Heisenberg images of Z_w / X_w (w = output wire) through a window of the circuit (CZ block layers [lo,hi],
1q gates strictly between window CZs on each wire). Reports the weight of the image on single-qubit terms.
usage: heis.py QASM lo hi eps"""
import sys, numpy as np, collections
sys.path.insert(0,'/tmp/peaked-amp/pauli'); sys.path.insert(0,'/tmp/peaked-p5p6')
import pprop as PP, relabel as RL
from blocks2 import u3m
fn, lo, hi, eps = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
n, ops = RL.parse_lines(fn); lay = RL.cz_layers(n, ops)
# window cz indices
win = [i for i in range(len(ops)) if ops[i][0]=='cz' and lo <= lay[i] <= hi]
first = {}; last = {}
for i in win:
    for q in ops[i][1]:
        first.setdefault(q, i); last[q] = i
out = []
for i, o in enumerate(ops):
    if o[0]=='cz':
        if lo <= lay[i] <= hi: out.append(('cz', o[1], None))
    else:
        q = o[1][0]
        if q in first and first[q] < i < last[q]:
            t,p,l=[eval(x,{'pi':np.pi}) for x in o[2].split(',')]
            out.append(('1', q, u3m(t,p,l)))
def image(q0, which):
    one=np.uint64(1)
    x=np.zeros(1,np.uint64); z=np.zeros(1,np.uint64); c=np.ones(1)
    if which=='Z': z[0]=one<<np.uint64(q0)
    else: x[0]=one<<np.uint64(q0)
    # reuse expect_Z loop body by temporarily calling with custom start: copy of PP.expect_Z
    for k,q,M in reversed(out):
        if k=='cz':
            a,b=np.uint64(q[0]),np.uint64(q[1])
            xa=(x>>a)&one; xb=(x>>b)&one; za=(z>>a)&one; zb=(z>>b)&one
            sgn=(xa&xb&(za^zb)).astype(bool); z=z^(xb<<a)^(xa<<b); c=np.where(sgn,-c,c)
        else:
            T=PP.transfer(M); s=np.uint64(q)
            xs=(x>>s)&one; zs=(z>>s)&one
            loc=np.where(xs==1,np.where(zs==1,2,1),np.where(zs==1,3,0)); nz=loc!=0
            if not nz.any(): continue
            px,pz,pc=[x[~nz]],[z[~nz]],[c[~nz]]
            xb_,zb_,cb,lb=x[nz]&~(one<<s),z[nz]&~(one<<s),c[nz],loc[nz]
            for j in (1,2,3):
                cj=cb*T[lb,j]; m=np.abs(cj)>=eps
                if m.any():
                    bx,bz=PP.BITS[j]; px.append(xb_[m]|(np.uint64(bx)<<s)); pz.append(zb_[m]|(np.uint64(bz)<<s)); pc.append(cj[m])
            x=np.concatenate(px); z=np.concatenate(pz); c=np.concatenate(pc)
            order=np.lexsort((z,x)); x,z,c=x[order],z[order],c[order]
            new=np.ones(len(x),bool); new[1:]=(x[1:]!=x[:-1])|(z[1:]!=z[:-1])
            idx=np.flatnonzero(new); c=np.add.reduceat(c,idx); x,z=x[idx],z[idx]
            m=np.abs(c)>=eps; x,z,c=x[m],z[m],c[m]
            if len(c)>3e6: return None
    supp=x|z
    w=np.array([bin(int(s)).count('1') for s in supp])
    tot=np.sum(c**2)
    single=collections.Counter()
    for s,cc,ww in zip(supp,c,w):
        if ww==1: single[int(s).bit_length()-1]+=cc**2
    bq,bw=(single.most_common(1)[0] if single else (-1,0))
    return len(c), tot, bq, bw/tot, np.sum(c[w<=1]**2)/tot
pi=None
import os
if os.path.exists('P5_granite_summit.qasm.bw.npy.pi.npy'): pi=np.load('P5_granite_summit.qasm.bw.npy.pi.npy')
print('window',lo,hi,'ops',len(out))
for q in range(n):
    r=[image(q,wh) for wh in 'ZX']
    print(q, ' '.join('%s: terms %d norm %.3f best q%d w1 %.4f wle1 %.4f'%(wh,*rr) if rr else wh+': blowup' for wh,rr in zip('ZX',r)), '| pi^-1(q)=%d'%int(np.argsort(pi)[q]) if pi is not None else '', flush=True)
