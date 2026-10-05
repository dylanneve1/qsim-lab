"""Validates paper_model.exact_model against the paper's own Monte-Carlo
data (assets/masked_success_stats.csv of the release): the paper samples a
uniformly random base g in [2, N-2) coprime to N per shot; here the same
model is averaged exactly over all such g. Prints exact vs sampled mean and
the sampled binomial standard error."""
import math
import pathlib
import sys
import collections

import gidney_env as ge
from paper_model import exact_model

csv = pathlib.Path(ge.GIDNEY_SRC).parent / "assets" / "masked_success_stats.csv"
data = collections.defaultdict(lambda: [0, 0.0])
for line in csv.read_text().splitlines():
    if not line or line.startswith("modulus"):
        continue
    n, p, shots, succ = line.split(",")
    d = data[(int(n), float(p))]
    d[0] += int(shots)
    d[1] += float(succ)
for n in map(int, sys.argv[1:]):
    gs = [g for g in range(2, n - 2) if math.gcd(g, n) == 1]
    for p in (0.0, 0.01, 0.1, 0.5):
        if (n, p) not in data:
            continue
        w = max(1, round(n * p))
        ex = sum(exact_model(n, g, w)[0] for g in gs) / len(gs)
        shots, succ = data[(n, p)]
        mean = succ / shots
        se = math.sqrt(max(mean * (1 - mean), 1e-12) / shots)
        print(f"N={n} p={p} W={w}: exact model {ex:.5f}  paper's sampled {mean:.5f} +- {se:.5f} ({shots} shots)  z={(ex - mean) / se:+.2f}")
