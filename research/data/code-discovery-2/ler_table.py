"""Table of circuit-level logical error rates and paired ratios.

usage: python ler_table.py <ler.jsonl> [<label>=<file.json> ...]

Each input line is the JSON printed by `group_codes ler` / `bb_codes ler`
(shots, fails, rounds, k, p, Wilson intervals). Prints a markdown table with
the block logical error rate per round, the rate per logical qubit per round
(block rate / k), and for consecutive pairs of rows with the same p and
rounds the ratio of per-logical-qubit rates with a 95% Katz (log) interval
on the ratio of the two binomial proportions.
"""
import json
import math
import sys

rows = []
for arg in sys.argv[1:]:
    label, _, path = arg.partition("=") if "=" in arg else (None, None, arg)
    for line in open(path):
        line = line.strip()
        if line.startswith("{"):
            r = json.loads(line)
            r["label"] = label or r.get("group", f"[[{r['n']},{r['k']}]]")
            rows.append(r)

print("| code | p | rounds | shots | fails | block p_L / round [95% CI] | per logical qubit / round | OSD calls |")
print("|---|---|---|---|---|---|---|---|")
for r in rows:
    lo, hi = r["ci95_round"]
    print(f"| {r['label']} | {r['p']} | {r['rounds']} | {r['shots']} | {r['fails']} | "
          f"{r['p_L_round']:.2e} [{lo:.2e}, {hi:.2e}] | {r['p_L_round'] / r['k']:.2e} | "
          f"{r['osd_calls'] / r['shots']:.0%} |")

print()
for a, b in zip(rows[0::2], rows[1::2]):
    if (a["p"], a["rounds"]) != (b["p"], b["rounds"]):
        continue
    fa, na, fb, nb = a["fails"], a["shots"], b["fails"], b["shots"]
    if fa == 0 or fb == 0:
        print(f"{a['label']} vs {b['label']}: a zero count, no ratio")
        continue
    # per-logical-qubit block rates: p_L / k (block failure probabilities, same rounds)
    ra, rb = fa / na / a["k"], fb / nb / b["k"]
    rr = rb / ra
    se = math.sqrt(1 / fa - 1 / na + 1 / fb - 1 / nb)
    print(f"{b['label']} / {a['label']} per logical qubit (p = {a['p']}, {a['rounds']} rounds): "
          f"{rr:.2f} [95% CI {rr * math.exp(-1.96 * se):.2f}, {rr * math.exp(1.96 * se):.2f}]")
