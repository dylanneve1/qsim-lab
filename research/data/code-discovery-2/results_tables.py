"""Markdown tables for code-discovery-2.md §5 from the search outputs.

usage: python results_tables.py <search.jsonl.xz or .jsonl> <search.log>

Expects the concatenated outputs of all search runs (see README in this
folder): one JSON line per connected class with k > 0, and the per-group
summary lines. Prints: per-order totals, status counts, the new codes, and
the (n, k) points where the space reaches the known threshold exactly.
"""
import json
import lzma
import re
import sys
from collections import Counter, defaultdict

jpath, lpath = sys.argv[1], sys.argv[2]
opener = lzma.open if jpath.endswith(".xz") else open
rows = [json.loads(l) for l in opener(jpath, "rt")]
groups = {}
for line in open(lpath):
    m = re.match(r"N=(\d+) id=(\d+) SmallGroup\(\d+,\d+\) (\S+) \|Z\|=(\d+) tclasses=(\d+) orbits=(\d+) "
                 r"k>0=(\d+) disconnected=(\d+) k>128=(\d+)", line)
    if m:
        key = (int(m.group(1)), int(m.group(2)))
        groups[key] = dict(name=m.group(3), orbits=int(m.group(6)), kpos=int(m.group(7)),
                           disc=int(m.group(8)), bigk=int(m.group(9)))

print(f"groups: {len(groups)}; inequivalent codes (pair orbits): {sum(g['orbits'] for g in groups.values())}; "
      f"k > 0: {sum(g['kpos'] for g in groups.values())}, of which disconnected "
      f"{sum(g['disc'] for g in groups.values())}, k > 128 {sum(g['bigk'] for g in groups.values())}; "
      f"connected classes classified: {len(rows)}")
st = Counter(r["status"] for r in rows)
print("status:", dict(st))
print()
print("| N (n = 2N) | groups | inequivalent codes | connected, k > 0 | below | le_T / tie | new | undecided |")
print("|---|---|---|---|---|---|---|---|")
byN = defaultdict(lambda: Counter())
gN = Counter()
oN = Counter()
for (N, _), g in groups.items():
    gN[N] += 1
    oN[N] += g["orbits"]
for r in rows:
    byN[r["N"]][r["status"]] += 1
bands = [(6, 47), (48, 71), (72, 95), (96, 96), (97, 119), (120, 143), (144, 144), (145, 150)]
for lo, hi in bands:
    sel = [N for N in gN if lo <= N <= hi]
    c = Counter()
    for N in sel:
        c.update(byN[N])
    label = f"{lo}–{hi}" if lo != hi else f"{lo}"
    print(f"| {label} | {sum(gN[N] for N in sel)} | {sum(oN[N] for N in sel)} | {sum(c.values())} | "
          f"{c['below']} | {c['le_T'] + c['tie']} | {c['new'] + c['new_bounds']} | {c['undecided']} |")
print()
print("New codes (d > T(n, k)):")
print()
print("| [[n,k,d]] | k·d²/n | T(n,k) | groups (GAP id: A ; B) |")
print("|---|---|---|---|")
new = defaultdict(list)
for r in rows:
    if r["status"] in ("new", "new_bounds"):
        new[(r["n"], r["k"], r["d_up"], r["T"])].append(f"({r['N']},{r['id']}) {groups[(r['N'], r['id'])]['name']}: {r['A']} ; {r['B']}")
for (n, k, d, t), v in sorted(new.items()):
    print(f"| [[{n},{k},{d}]] | {k * d * d / n:.2f} | {t} | {'<br>'.join(v)} |")
print()
print("Undecided:")
for r in rows:
    if r["status"] == "undecided":
        print(f"- [[{r['n']},{r['k']},{r['d_lo']}..{r['d_up']}]] T = {r['T']}: ({r['N']},{r['id']}) "
              f"{groups[(r['N'], r['id'])]['name']} A = {r['A']} B = {r['B']}")
print()
print("(n, k) where some class reaches T exactly (tie proven) or has a logical of weight T only (le_T):")
reach = defaultdict(set)
for r in rows:
    if r["status"] in ("tie", "le_T"):
        reach[(r["n"], r["k"], r["T"])].add(r["status"])
print(", ".join(f"[[{n},{k},{t}]]{'' if 'tie' in s else '≤'}" for (n, k, t), s in sorted(reach.items())))
