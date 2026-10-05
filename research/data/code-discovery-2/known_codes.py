"""Known weight-6 two-block codes and the domination threshold T(n, k).

usage: python known_codes.py [--extra extra.tsv] > threshold.tsv

Known = the merged weight-6 frontier of ../code-discovery/literature.md and
every weight-6 row of the two-block sections of its 2026-10-05 supplement
(published; d taken as listed, also when it is only an upper bound), the
exact abelian rank<=2 frontier ../code-discovery/frontier_w6.json and
../code-discovery/certified.jsonl, plus optional extra "n k d" rows, and every
direct sum of up to 5 of these ([[n1+n2, k1+k2, min d]]).

T(n, k) = max d' over known (n', k', d') with n' <= n and k' >= k (0 if none):
a new [[n, k, d]] code is not dominated by any known code iff d > T(n, k).
Output: one line "n k T" per even n in [4, 300] and k in [1, 150] with T > 0.
"""
import json
import os
import re
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
CD = os.path.join(HERE, "..", "code-discovery")
NMAX = 300


def published():
    text = open(os.path.join(CD, "literature.md")).read()
    sec = text.split("## Merged Pareto frontier")[1].split("## Weight-8")[0]
    out = supplement(text)
    for line in sec.splitlines():
        m = re.match(r"\|\s*(\d+)\s*\|(.*)\|\s*[\d.]+\s*\|", line)
        if not m:
            continue
        for nn, kk, dd, _star in re.findall(r"\[\[(\d+),(\d+),(\d+)(\*?)\]\]", m.group(2)):
            out.append((int(nn), int(kk), int(dd)))
    return out


def supplement(text):
    """Weight-6 rows (n <= NMAX) of the two-block tables in the supplement."""
    out = []
    if "## Supplement (2026-10-05)" not in text:
        return out
    part = text.split("## Supplement (2026-10-05)")[1]
    two_block = False
    for line in part.splitlines():
        if line.startswith("#"):
            low = line.lower()
            two_block = "not two-block" not in low
            continue
        if line.startswith("## Highlights") or not two_block:
            continue
        m = re.match(r"\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|", line)
        if not m or m.group(5) != "6":
            continue
        d = re.search(r"(\d+)", m.group(3))
        n = int(m.group(1))
        if d and n <= NMAX:
            out.append((n, int(m.group(2)), int(d.group(1))))
    return out


def ours():
    out = [(r["n"], r["k"], r["d"]) for r in json.load(open(os.path.join(CD, "frontier_w6.json")))]
    for line in open(os.path.join(CD, "certified.jsonl")):
        r = json.loads(line)
        if r["d_lo"] == r["d_up"]:
            out.append((r["n"], r["k"], r["d_up"]))
    return out


def pareto(items):
    items = sorted(set(items), key=lambda t: (-t[0], -t[1]))
    keep = []
    for k, d in items:
        if not any(k2 >= k and d2 >= d for k2, d2 in keep):
            keep.append((k, d))
    return keep


def closure(base, parts=5):
    """Pareto (k, d) sets at each n reachable by sums of <= parts base codes."""
    level = defaultdict(list)
    for n, k, d in base:
        if n <= NMAX:
            level[n].append((k, d))
    level = {n: pareto(v) for n, v in level.items()}
    total = defaultdict(list)
    for n, v in level.items():
        total[n].extend(v)
    for _ in range(parts - 1):
        nxt = defaultdict(list)
        for n, v in level.items():
            for n2, k2, d2 in base:
                if n + n2 <= NMAX:
                    for k, d in v:
                        nxt[n + n2].append((k + k2, min(d, d2)))
        level = {n: pareto(v) for n, v in nxt.items()}
        for n, v in level.items():
            total[n].extend(v)
    return {n: pareto(v) for n, v in total.items()}


def main():
    base = published() + ours()
    args = sys.argv[1:]
    if args[:1] == ["--extra"]:
        for line in open(args[1]):
            if line.strip() and not line.startswith("#"):
                n, k, d = map(int, line.split()[:3])
                base.append((n, k, d))
    base = sorted(set(base))
    reach = closure(base)
    for n in range(4, NMAX + 1, 2):
        for k in range(1, 151):
            t = 0
            for n2, v in reach.items():
                if n2 <= n:
                    for k2, d2 in v:
                        if k2 >= k and d2 > t:
                            t = d2
            if t > 0:
                print(n, k, t)


main()
