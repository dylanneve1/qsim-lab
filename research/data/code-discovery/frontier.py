"""Pareto frontier of the exhaustive two-block code search vs the literature.

usage: python frontier.py <search.jsonl>... [--lit literature.md] [--out prefix]

Reads the JSON lines of `bb_codes search`, keeps codes with exact distance
(d_lo == d_up), and compares against the merged weight-6 frontier table of
literature.md. A code (n, k, d) dominates (n', k', d') if n <= n', k >= k',
d >= d' with one strict inequality.
"""
import json
import re
import sys
from collections import defaultdict

args = sys.argv[1:]
lit_path = "literature.md"
out = "frontier"
files = []
i = 0
while i < len(args):
    if args[i] == "--lit":
        lit_path = args[i + 1]
        i += 2
    elif args[i] == "--out":
        out = args[i + 1]
        i += 2
    else:
        files.append(args[i])
        i += 1

codes = []
inexact = []
skipped = 0
for f in files:
    for line in open(f):
        r = json.loads(line)
        if r.get("skipped"):
            skipped += 1
            continue
        if r["d_lo"] == r["d_up"]:
            codes.append(r)
        elif r["d_lo"] > 0:
            inexact.append(r)

def mono(t):
    a = b = 0
    t = t.replace("*", " ").strip()
    if t == "1":
        return (0, 0)
    for f in t.split():
        v, _, e = f.partition("^")
        e = int(e) if e else 1
        if v == "x":
            a += e
        else:
            b += e
    return (a, b)


def connected(r):
    """True if A and B (both containing 1) generate the whole group;
    otherwise the code is |G/H| disjoint copies of a code over H."""
    l, m = r["l"], r["m"]
    gens = [mono(t) for t in r["A"].split("+") + r["B"].split("+")]
    seen = {(0, 0)}
    stack = [(0, 0)]
    while stack:
        i, j = stack.pop()
        for a, b in gens:
            for s in (1, -1):
                v = ((i + s * a) % l, (j + s * b) % m)
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
    return len(seen) == l * m


ndec = sum(1 for r in codes if not connected(r))
codes = [r for r in codes if connected(r)]
# best d per (n, k) over indecomposable codes
best = {}
for r in codes:
    key = (r["n"], r["k"])
    if key not in best or r["d_up"] > best[key]["d_up"]:
        best[key] = r
# per-n frontier
by_n = defaultdict(list)
for (n, k), r in best.items():
    by_n[n].append((k, r["d_up"], r))
front = {}
for n, v in by_n.items():
    v.sort(key=lambda t: (-t[0], -t[1]))
    keep = []
    for k, d, r in v:
        if not any(k2 >= k and d2 >= d and (k2, d2) != (k, d) for k2, d2, _ in v):
            keep.append((k, d, r))
    front[n] = sorted(keep)

# literature merged frontier
lit = []
sec = open(lit_path).read().split("## Merged Pareto frontier")[1].split("## Weight-8")[0]
for line in sec.splitlines():
    m = re.match(r"\|\s*(\d+)\s*\|(.*)\|\s*[\d.]+\s*\|", line)
    if not m:
        continue
    for nn, kk, dd, star in re.findall(r"\[\[(\d+),(\d+),(\d+)(\*?)\]\]", m.group(2)):
        lit.append((int(nn), int(kk), int(dd), star == "*"))


def dominates(a, b):
    return a[0] <= b[0] and a[1] >= b[1] and a[2] >= b[2] and a != b


ours = [(n, k, d) for n, v in front.items() for k, d, _ in v]
rows = []
for n in sorted(front):
    for k, d, r in front[n]:
        lit_dom = [x for x in lit if dominates(x[:3], (n, k, d)) or x[:3] == (n, k, d)]
        beats = [x for x in lit if dominates((n, k, d), x[:3])]
        rows.append((n, k, d, r, lit_dom, beats))

with open(out + ".md", "w") as fo:
    fo.write(f"indecomposable codes with exact d: {len(codes)}; decomposable (copies of a smaller code, dropped): {ndec}; bounded only: {len(inexact)}; k>128 skipped: {skipped}\n\n")
    fo.write("| n | k | d | kd^2/n | group | A | B | literature status |\n|---|---|---|---|---|---|---|---|\n")
    for n, k, d, r, lit_dom, beats in rows:
        same = [x for x in lit_dom if x[:3] == (n, k, d)]
        if same:
            st = "published" + ("" if not same[0][3] else " (lit d was a bound)")
        elif lit_dom:
            st = "dominated by " + ", ".join(f"[[{a},{b},{c}]]" for a, b, c, _ in lit_dom[:3])
        else:
            st = "**not in lit frontier**"
        if beats:
            st += "; dominates lit " + ", ".join(f"[[{a},{b},{c}{'*' if s else ''}]]" for a, b, c, s in beats[:4])
        g = f"Z{r['l']}xZ{r['m']}" if r["m"] > 1 else f"Z{r['l']}"
        fo.write(f"| {n} | {k} | {d} | {k*d*d/n:.2f} | {g} | `{r['A']}` | `{r['B']}` | {st} |\n")
    # literature entries not reproduced
    fo.write("\n## Literature frontier entries vs this search\n\n")
    for x in lit:
        n, k, d, star = x
        if n > max(front):
            continue
        ok = any(a >= k and b >= d for a, b, _ in front.get(n, []))
        beaten = [(n2, k2, d2) for (n2, k2, d2) in ours if dominates((n2, k2, d2), (n, k, d))]
        fo.write(f"- [[{n},{k},{d}{'*' if star else ''}]]: {'matched/exceeded at same n' if ok else 'NOT reached in this space'}"
                 + (f"; dominated by ours {beaten[:3]}" if beaten else "") + "\n")
json.dump([dict(n=n, k=k, d=d, A=r["A"], B=r["B"], l=r["l"], m=r["m"]) for n, k, d, r, _, _ in rows],
          open(out + ".json", "w"), indent=0)
print(open(out + ".md").read()[:3000])
