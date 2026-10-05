"""Summarise `group_codes search` output against the known frontier.

usage: python analyze.py <search.jsonl> <search.log> [--out prefix]

search.jsonl: one line per connected class with k > 0 (status below / tie /
new / undecided, see examples/group_codes.rs). search.log: the per-group
summary lines on stderr (group names, class counts, timings).
Writes <prefix>.md (summary tables) and <prefix>_ties.jsonl (every class
with d >= T, i.e. tie or new, with its group name).
"""
import json
import re
import sys
from collections import Counter, defaultdict

args = sys.argv[1:]
out = "nonabelian_w6"
if "--out" in args:
    i = args.index("--out")
    out = args[i + 1]
    args = args[:i] + args[i + 2:]
jsonl, log = args[0], args[1]

names = {}
groups = []
for line in open(log):
    m = re.match(r"N=(\d+) id=(\d+) SmallGroup\(\d+,\d+\) (\S+) \|Z\|=(\d+) tclasses=(\d+) orbits=(\d+) "
                 r"k>0=(\d+) disconnected=(\d+) k>128=(\d+) below=(\d+) tie=(\d+) new=(\d+) undecided=(\d+) t=([\d.]+)s", line)
    if m:
        N, gid = int(m.group(1)), int(m.group(2))
        names[(N, gid)] = m.group(3)
        groups.append(dict(N=N, id=gid, name=m.group(3), Z=int(m.group(4)), tclasses=int(m.group(5)),
                           orbits=int(m.group(6)), kpos=int(m.group(7)), disc=int(m.group(8)),
                           bigk=int(m.group(9)), below=int(m.group(10)), tie=int(m.group(11)),
                           new=int(m.group(12)), und=int(m.group(13)), t=float(m.group(14))))

rows = [json.loads(l) for l in open(jsonl)]
status = Counter(r["status"] for r in rows)
by_nk = defaultdict(list)
for r in rows:
    by_nk[(r["n"], r["k"])].append(r)

ties = [r for r in rows if r["status"] in ("tie", "new")]
for r in ties:
    r["group"] = names.get((r["N"], r["id"]), "?")
und = [r for r in rows if r["status"] == "undecided"]

with open(out + "_ties.jsonl", "w") as f:
    for r in sorted(ties, key=lambda r: (r["n"], -r["k"], -r["d_up"])):
        f.write(json.dumps(r) + "\n")

with open(out + ".md", "w") as f:
    f.write(f"groups searched: {len(groups)}; orbits (inequivalent codes) {sum(g['orbits'] for g in groups)}; "
            f"connected k>0 classes: {len(rows)}; disconnected k>0: {sum(g['disc'] for g in groups)}; "
            f"k>128 skipped: {sum(g['bigk'] for g in groups)}; CPU {sum(g['t'] for g in groups):.0f} s\n\n")
    f.write("status counts: " + ", ".join(f"{k} {v}" for k, v in sorted(status.items())) + "\n\n")
    f.write("## Classes reaching the known threshold (d >= T(n, k))\n\n")
    f.write("| n | k | d | T | kd²/n | status | group | A | B | d_Z / d_X | roots |\n|---|---|---|---|---|---|---|---|---|---|---|\n")
    for r in sorted(ties, key=lambda r: (r["n"], -r["k"])):
        d = r["d_up"]
        f.write(f"| {r['n']} | {r['k']} | {d} | {r['T']} | {r['k']*d*d/r['n']:.2f} | {r['status']} | "
                f"SmallGroup({r['N']},{r['id']}) {r['group']} | {r['A']} | {r['B']} | {r['dz']} / {r['dx']} | {r['roots']} |\n")
    f.write("\n## Undecided (node limit)\n\n")
    for r in und:
        f.write(f"- [[{r['n']},{r['k']},{r['d_lo']}..{r['d_up']}]] T={r['T']} SmallGroup({r['N']},{r['id']}) "
                f"{names.get((r['N'], r['id']), '?')} A={r['A']} B={r['B']}\n")
    f.write("\n## Per n: largest k with a class at the threshold, and how close the rest came\n\n")
    f.write("| n | classes | k with d >= T | best d_up below T, per k (k:d_up/T) |\n|---|---|---|---|\n")
    for n in sorted({r["n"] for r in rows}):
        ks = sorted({r["k"] for r in rows if r["n"] == n})
        at = sorted({r["k"] for r in ties if r["n"] == n})
        near = []
        for k in ks:
            v = [r for r in by_nk[(n, k)] if r["status"] == "below"]
            if v:
                near.append(f"{k}:{max(r['d_up'] for r in v)}/{v[0]['T']}")
        f.write(f"| {n} | {sum(len(by_nk[(n, k)]) for k in ks)} | {at} | {' '.join(near)} |\n")
print(f"wrote {out}.md, {out}_ties.jsonl; status {dict(status)}")
