"""Merge the search runs into one output, each group once.

usage: python merge_runs.py <out prefix> <run1.jsonl> <run1.log> [<run2.jsonl> <run2.log> ...]

The search was restarted several times; every run skipped the groups already
completed by earlier runs, and a group's lines are written only when the group
completes. This keeps, for each group (N, id), the lines of the first run whose
log reports it complete, drops lines of groups no run completed, and writes
<prefix>.jsonl, <prefix>.log and <prefix>_groups.txt (one "N id" per completed
group).
"""
import json
import re
import sys

out = sys.argv[1]
pairs = list(zip(sys.argv[2::2], sys.argv[3::2]))
done = {}
for run, (jl, lg) in enumerate(pairs):
    for line in open(lg):
        m = re.match(r"N=(\d+) id=(\d+) ", line)
        if m:
            key = (int(m.group(1)), int(m.group(2)))
            if key not in done:
                done[key] = (run, line)
rows = []
for run, (jl, lg) in enumerate(pairs):
    for line in open(jl):
        r = json.loads(line)
        key = (r["N"], r["id"])
        if key in done and done[key][0] == run:
            rows.append(line if line.endswith("\n") else line + "\n")
with open(out + ".jsonl", "w") as f:
    f.writelines(rows)
with open(out + ".log", "w") as f:
    for key in sorted(done):
        f.write(done[key][1] if done[key][1].endswith("\n") else done[key][1] + "\n")
with open(out + "_groups.txt", "w") as f:
    for key in sorted(done):
        f.write(f"{key[0]} {key[1]}\n")
print(f"{len(done)} groups, {len(rows)} class lines")
