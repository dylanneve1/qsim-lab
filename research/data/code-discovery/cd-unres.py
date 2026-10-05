import json,collections,sys
sys.path.insert(0, '/Users/dylan/qsim-cd/research/data/code-discovery')
recs=[json.loads(l) for l in open(sys.argv[1])]
best=collections.defaultdict(int)
for r in recs:
    if r.get('skipped'): continue
    if r['d_lo']==r['d_up']: best[(r['n'],r['k'])]=max(best[(r['n'],r['k'])],r['d_up'])
un=[]
for r in recs:
    if r.get('skipped') or r['d_lo']==r['d_up']: continue
    b=best.get((r['n'],r['k']),0)
    if b==0: continue   # (n,k) with no exact decision (k below min_k)
    if r['d_up']>b: un.append((r['n'],r['k'],b,r['d_lo'],r['d_up'],r['l'],r['m'],r['A'],r['B']))
un.sort(key=lambda t:(t[0],t[1]))
for u in un: print(json.dumps(u))
print(len(un), file=sys.stderr)
