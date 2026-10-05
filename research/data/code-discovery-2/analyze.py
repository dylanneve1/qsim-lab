"""Summarise `group_codes search` / `csearch` output against the known frontier.

usage: python analyze.py [--out prefix] <file.jsonl> <file.log> [<file.jsonl> <file.log> ...]

Each .jsonl has one line per connected class with k > 0, the matching .log
the per-group (or per (K, H, m)) summary lines on stderr. Statuses (see
examples/group_codes.rs): below (a logical of weight < T exists), le_T (one
of weight T exists: d <= T), tie (exhaustive proof that d = T; first,
slower search rule only), new / new_bounds (no logical of weight <= T:
d > T), undecided (node limit).
Writes <prefix>.md (tables) and <prefix>_top.jsonl (every class with
d_up >= T, i.e. le_T, tie, new, undecided).
"""
import json
import re
import sys
from collections import Counter, defaultdict

args = sys.argv[1:]
out = "summary"
if "--out" in args:
    i = args.index("--out")
    out = args[i + 1]
    args = args[:i] + args[i + 2:]
pairs = list(zip(args[0::2], args[1::2]))

names = {}
summ = []
rows = []
for jl, lg in pairs:
    for line in open(lg):
        m = re.match(r"N=(\d+) id=(\d+) SmallGroup\(\d+,\d+\) (\S+) .* t=([\d.]+)s", line)
        if m:
            names[(int(m.group(1)), int(m.group(2)))] = m.group(3)
            summ.append(("2bga", float(m.group(4)), line))
        m = re.match(r"K=(\S+) \|H\|=(\d+) .* t=([\d.]+)s", line)
        if m:
            summ.append(("coset", float(m.group(3)), line))
    for line in open(jl):
        r = json.loads(line)
        if "K" in r:
            r["family"] = "coset"
            r["group"] = f"Z{r['m']} x {r['K']} / H(|H|={len(r['H'].split(','))})"
        else:
            r["family"] = "2bga"
            r["group"] = f"SmallGroup({r['N']},{r['id']}) {names.get((r['N'], r['id']), '?')}"
        rows.append(r)

status = Counter((r["family"], r["status"]) for r in rows)
top = [r for r in rows if r["status"] != "below"]
with open(out + "_top.jsonl", "w") as f:
    for r in sorted(top, key=lambda r: (r["n"], -r["k"], -r["d_up"])):
        f.write(json.dumps(r) + "\n")


def kd2n(r, d=None):
    d = r["d_up"] if d is None else d
    return r["k"] * d * d / r["n"]


with open(out + ".md", "w") as f:
    cpu = sum(t for _, t, _ in summ)
    f.write(f"jobs (groups or (K,H,m)): {len(summ)}; classes with k>0, connected: {len(rows)}; CPU {cpu:.0f} s\n\n")
    f.write("| family | status | classes |\n|---|---|---|\n")
    for (fam, st), v in sorted(status.items()):
        f.write(f"| {fam} | {st} | {v} |\n")
    for title, sel in (("New (d > T)", ("new", "new_bounds")), ("Undecided", ("undecided",)),
                       ("Exact ties (d = T proven)", ("tie",))):
        v = [r for r in rows if r["status"] in sel]
        f.write(f"\n## {title}: {len(v)}\n\n")
        if not v:
            continue
        f.write("| n | k | d | T | kd²/n | group | A | B | d_Z / d_X | roots |\n|---|---|---|---|---|---|---|---|---|---|\n")
        best = {}
        for r in v:
            key = (r["n"], r["k"], r["family"])
            if key not in best or r["d_up"] > best[key]["d_up"]:
                best[key] = r
        for r in sorted(best.values(), key=lambda r: (r["n"], -r["k"])):
            d = f"{r['d_lo']}..{r['d_up']}" if r["d_lo"] != r["d_up"] else str(r["d_up"])
            f.write(f"| {r['n']} | {r['k']} | {d} | {r['T']} | {kd2n(r):.2f} | {r['group']} | {r['A']} | {r['B']} | {r['dz']} / {r['dx']} | {r['roots']} |\n")
    # per (n, k): how close the space comes to the threshold
    f.write("\n## Per (n, k): best upper bound found vs T (all classes)\n\n")
    f.write("| n | k | T | best d_up | classes | at T (le_T/tie) | family |\n|---|---|---|---|---|---|---|\n")
    by = defaultdict(list)
    for r in rows:
        by[(r["n"], r["k"], r["family"])].append(r)
    for (n, k, fam), v in sorted(by.items()):
        bu = max(r["d_up"] for r in v)
        at = sum(1 for r in v if r["status"] in ("le_T", "tie"))
        if bu >= v[0]["T"] - 1 and v[0]["T"] >= 6:
            f.write(f"| {n} | {k} | {v[0]['T']} | {bu} | {len(v)} | {at} | {fam} |\n")
print(f"wrote {out}.md, {out}_top.jsonl; {dict(status)}")
