import csv, sys
from collections import defaultdict
def load(fn):
    return [r for r in csv.DictReader(l for l in open(fn) if not l.startswith('#'))]
rows=[]
for fn in sys.argv[1:]: rows+=load(fn)
d=defaultdict(list)
for r in rows: d[(r['n'],r['k'])].append(r)
for (n,k),v in sorted(d.items(), key=lambda x:(int(x[0][0]),int(x[0][1]))):
    m=len(v); s=sum(int(r['order_strict']) for r in v); f=sum(int(r['factor_ok']) for r in v); c=sum(r['capped_round']!='-1' for r in v)
    dirty=[r for r in v if r['dirty_from']!='-1']
    ds=sum(int(r['order_strict']) for r in dirty)
    print(f"n={n} k={k} M={m} strict={s/m:.3f} factor={f/m:.3f} capped={c} dirty={len(dirty)} dirty_strict={ds} maxpeak={max(int(r['peak']) for r in v)} mean_s={sum(float(r['secs']) for r in v)/m:.3f}")
