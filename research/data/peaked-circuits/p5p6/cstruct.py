import re,sys,collections
import numpy as np
def load(fn):
    ops=[]
    for line in open(fn):
        line=line.strip()
        m=re.match(r'u3\(([^)]*)\)\s+\w+\[(\d+)\];',line)
        if m:
            ang=[eval(x,{'pi':np.pi}) for x in m.group(1).split(',')]
            ops.append(('u3',(int(m.group(2)),),ang));continue
        m=re.match(r'cz\s+\w+\[(\d+)\],\s*\w+\[(\d+)\];',line)
        if m: ops.append(('cz',(int(m.group(1)),int(m.group(2))),None))
    return ops
if __name__=='__main__':
    fn=sys.argv[1]
    ops=load(fn); n=max(max(o[1]) for o in ops)+1
    czs=[o for o in ops if o[0]=='cz']
    print('n',n,'ops',len(ops),'cz',len(czs))
    # group into blocks: track for each cz whether the previous 2q op on both wires was cz on same pair
    last={}
    blocks=[]  # list of (pair, ncz)
    cur=None
    for i,o in enumerate(ops):
        if o[0]!='cz': continue
        a,b=o[1]; p=tuple(sorted((a,b)))
        if last.get(a)==('blk',len(blocks)-1) and last.get(b)==('blk',len(blocks)-1) and blocks and blocks[-1][0]==p:
            blocks[-1][1]+=1
        else:
            # check if previous 2q on both wires is same block (non-consecutive in file)
            if last.get(a) is not None and last.get(a)==last.get(b) and blocks[last[a][1]][0]==p:
                blocks[last[a][1]][1]+=1; continue
            blocks.append([p,1])
        last[a]=('blk',len(blocks)-1); last[b]=('blk',len(blocks)-1)
    print('blocks',len(blocks), collections.Counter(b[1] for b in blocks))
    # ASAP layering of blocks
    depth=[0]*n; lay=[]
    for p,k in blocks:
        l=max(depth[p[0]],depth[p[1]]); lay.append(l); depth[p[0]]=depth[p[1]]=l+1
    L=max(lay)+1
    print('block depth',L)
    cnt=collections.Counter(lay)
    print('layer sizes',[cnt[i] for i in range(L)])
    pc=collections.Counter(b[0] for b in blocks)
    print('distinct pairs',len(pc),'repeat hist',sorted(collections.Counter(pc.values()).items()))
    deg=collections.Counter()
    for p in pc: deg[p[0]]+=1; deg[p[1]]+=1
    print('graph degree hist',sorted(collections.Counter(deg.values()).items()))
    # per-qubit block counts
    qc=collections.Counter()
    for p,k in blocks: qc[p[0]]+=1; qc[p[1]]+=1
    print('per-qubit block counts',[qc[i] for i in range(n)])
    np.save(fn.split('/')[-1]+'.blocks.npy',np.array([(p[0],p[1],l) for (p,k),l in zip(blocks,lay)]))
