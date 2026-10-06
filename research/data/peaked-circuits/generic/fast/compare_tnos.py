"""Compare bench_tnos.py runs: trajectory equality on the common prefix and time to reach each step.
usage: compare_tnos.py LOG_A LOG_B [...]   (first log is the reference)"""
import sys, json
runs = []
for f in sys.argv[1:]:
    for line in open(f):
        if line.startswith('RESULT {'):
            runs.append((f, json.loads(line[7:])))
ref_f, ref = runs[0]
key = ('lo', 'hi', 'side', 'elems', 'max_bond', 'nbonds', 'moved')
for f, r in runs:
    m = min(len(r['hist']), len(ref['hist']))
    same = all(tuple(a[k] for k in key) == tuple(b[k] for k in key) for a, b in zip(r['hist'][:m], ref['hist'][:m]))
    print(f"{r['variant']:7s} steps {len(r['hist']):3d} wall {r['wall']:7.1f}s  final {r['hist'][-1]['elems']:8d} elems "
          f"[{r['hist'][-1]['lo']},{r['hist'][-1]['hi']})  trajectory == {ref['variant']} on {m} common steps: {same}  rss {r['rss_mb']} MB")
print("time (s) to reach step [lo,hi) / elems:")
print("  step              elems   " + "  ".join(f"{r['variant']:>8s}" for _, r in runs))
n = max(len(r['hist']) for _, r in runs)
for i in range(n):
    h = next(r['hist'][i] for _, r in runs if len(r['hist']) > i)
    cells = [f"{r['hist'][i]['t']:8.1f}" if len(r['hist']) > i else "       -" for _, r in runs]
    print(f"  [{h['lo']:2d},{h['hi']:2d}) {h['side']:6s} {h['elems']:8d}   " + "  ".join(cells))
