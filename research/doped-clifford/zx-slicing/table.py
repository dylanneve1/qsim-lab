"""Compile tables from lean.jsonl (all slicing runs: chain-seed and kahypar-seeded)."""
import json, collections
rows = [json.loads(l) for l in open('/tmp/doped-zx/lean.jsonl')]
best = collections.defaultdict(dict)   # (D) -> width -> (log2C, log2slices, src)
for r in rows:
    D = r['D']; src = 'chain' if r['src'] == 'seed' else 'kahypar:' + r['src']
    for w in range(10, 60):
        if r['width'] <= w + 1e-9:
            cur = best[D].get(w)
            if cur is None or r['log2C'] < cur[0]: best[D][w] = (r['log2C'], r['log2slices'], src)
chain0 = {r['D']: (r['width'], r['log2C']) for r in rows if r['src'] == 'seed' and r['log2slices'] == 0}
print('| D | chain-sweep width / log2C (unsliced) | best log2C @w<=31 (log2 slices, tree) | @w<=30 | @w<=28 |')
print('|---|---|---|---|---|')
for D in sorted(best):
    cells = []
    for w in (31, 30, 28):
        b = best[D].get(w); cells.append(f'{b[0]:.1f} ({b[1]:.0f}, {b[2].split(":")[0]})' if b else '-')
    print(f'| {D} | {chain0[D][0]} / {chain0[D][1]:.1f} | ' + ' | '.join(cells) + ' |')
print()
print('Overhead to go k bits below the chain-sweep width w0 (log2 of total cost / unsliced chain cost):')
print('| D | w0 | k=1 | k=2 | k=3 | k=4 | k=5 | k=7 |'); print('|---|---|---|---|---|---|---|---|')
for D in sorted(best):
    w0, c0 = chain0[D]; cells = []
    for k in (1, 2, 3, 4, 5, 7):
        b = best[D].get(w0 - k); cells.append(f'+{b[0]-c0:.1f}' if b else '-')
    print(f'| {D} | {w0} | ' + ' | '.join(cells) + ' |')
