#!/usr/bin/env python3
"""Markdown tables for research/shor/shor-xl.md from the logs in this directory.
usage: python3 tables.py"""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))


def runs(path):
    """Parse timed.sh / ge_shor logs: one dict per process (multiple EH runs merged)."""
    out, cur = [], None
    for line in open(os.path.join(HERE, path)):
        if line.startswith("== "):
            cur = {"cmd": line.strip()}
            out.append(cur)
        elif line.startswith("before:"):
            cur["load_before"] = re.search(r"load average: ([\d.]+)", line).group(1)
        elif line.startswith("after:"):
            cur["load_after"] = re.search(r"load average: ([\d.]+)", line).group(1)
        elif line.startswith("EH run") or line.startswith("Shor-odd run"):
            kv = dict(re.findall(r"(\w+)=(\S+)", line))
            cur.setdefault("runs", []).append(kv)
        elif line.startswith("qubits="):
            cur.update(dict(re.findall(r"(\w+)=(\S+)", line)))
        elif line.startswith("wall"):
            cur["wall"] = float(line.split()[1])
            cur["rss_gb"] = int(re.search(r"maxrss (\d+) KB", line).group(1)) * 1024 / 1e9
        elif line.startswith("time "):
            cur["prof"] = line.strip()
    return out


def orders():
    o = {}
    for line in open(os.path.join(HERE, "orders_seed1.txt")):
        kv = dict(re.findall(r"(\w+(?:\(\w+\))?)=(\S+)", line))
        kv["lambda_odd"] = re.search(r" lambda_odd=(\d+)", line).group(1)
        o[int(kv["N"])] = kv
    return o


def frontier():
    o = orders()
    rows = []
    for path in ("frontier_22_33.log", "g43_B_seed1.log"):
        for r in runs(path):
            n = int(r["runs"][0]["N"])
            kv = o[n]
            ok = [x for x in r["runs"] if x.get("factors", "None") != "None"]
            rows.append((int(kv["bits"]), n, kv["ord(g)"], kv["lambda_odd"], r["qubits"], r["toffoli"],
                         r["peak_branches"], f"{float(r['gate_branch_ops']):.2e}", f"{r['wall']:.1f}",
                         f"{r['rss_gb']:.2f}", f"{r['load_before']}/{r['load_after']}",
                         f"run {len(r['runs'])}" if ok else "failed"))
    print("| bits | N | ord(g) | λ_odd | qubits | Toffolis | peak branches | gate·branch ops | time (s) | peak RSS (GB) | load before/after | factored |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(rows):
        print("| " + " | ".join(str(x) for x in r) + " |")


def skipped():
    print("| bits | N | λ_odd | ord(g), seed-1 base | window memory needed (2·ord(g)·16 B) |")
    print("|---|---|---|---|---|")
    for n, kv in sorted(orders().items(), key=lambda t: int(t[1]["bits"])):
        r = int(kv["ord(g)"])
        need = 2 * r * 16 / 1e9
        if need > 10 and n != 549755813701:
            print(f"| {kv['bits']} | {n} | {int(kv['lambda_odd']):.3e} | {r:.3e} | {need:,.0f} GB |")




def ab(path):
    """min wall / eval per arm of an ab.sh log: prints one markdown row per label."""
    cur, data = None, {}
    for line in open(os.path.join(HERE, path)):
        m = re.match(r"-- (\S+) arm (\w) rep (\d+)", line)
        if m:
            cur = (m.group(1), m.group(2))
            data.setdefault(cur, {"wall": [], "eval": [], "load": []})
        elif line.startswith("before:") and cur:
            data[cur]["load"].append(float(re.search(r"load average: ([\d.]+)", line).group(1)))
        elif line.startswith("wall") and cur:
            data[cur]["wall"].append(float(line.split()[1]))
        elif "sliced profile" in line and cur:
            e = re.search(r"eval c=1 ([\d.]+)s  eval c=0 ([\d.]+)s", line)
            data[cur]["eval"].append(float(e.group(1)) + float(e.group(2)))
        elif line.startswith("time ") and "eval" in line and cur:
            data[cur]["eval"].append(float(re.search(r"eval ([\d.]+)", line).group(1)))
    labels = sorted({k[0] for k in data})
    for lab in labels:
        a, b = data[(lab, "A")], data[(lab, "B")]
        fmt = lambda v: " / ".join(f"{x:.2f}" for x in v)
        print(f"| {lab} | {fmt(a['wall'])} | {fmt(b['wall'])} | {min(a['wall']) / min(b['wall']):.3f} | "
              f"{min(a['eval']):.2f} → {min(b['eval']):.2f} ({min(a['eval']) / min(b['eval']):.2f}×) | "
              f"{min(a['load'] + b['load']):.0f}–{max(a['load'] + b['load']):.0f} |")


def ab_all():
    print("| run | AVX2 tier: wall (s) | AVX-512 tier: wall (s) | min/min | gate evaluation, min (s) | 1-min load before runs |")
    print("|---|---|---|---|---|---|")
    for p in ("ab_28.log", "ab_31_qsim.log", "ab_31_eh.log", "ab_nw.log"):
        if os.path.exists(os.path.join(HERE, p)):
            ab(p)


if __name__ == "__main__":
    frontier()
    print()
    skipped()
    print()
    ab_all()
