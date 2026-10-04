#!/usr/bin/env python3
"""Tables and plots for research/magic-atlas.md.

  python3 plots.py DATADIR [MAC_DIR]
DATADIR has atlas.csv, recycle.csv, magic.csv, profiles/, magic/; MAC_DIR has
the Mac jsonl files (magic-law.jsonl, magic-eng.jsonl, magic-demo.jsonl).
Writes PNGs and tables.md into DATADIR.
"""
import csv, glob, json, math, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = sys.argv[1]
MAC = sys.argv[2] if len(sys.argv) > 2 else D
# reference categorical palette (light), fixed order
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
    "legend.frameon": False, "figure.dpi": 130,
})

def rows(fn):
    p = os.path.join(D, fn)
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []

atlas = rows("atlas.csv")
rec = {r["spec"]: r for r in rows("recycle.csv")}
magic = rows("magic.csv")

def fnum(x, default=float("nan")):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default

def frec(spec):
    r = rec.get(spec)
    if not r:
        return None
    if r.get("f_rec") == ">16":
        return ">16"
    if r.get("f") not in (None, ""):
        return int(float(r["f"]))
    return None

# ---------------------------------------------------------------- tables
out = []
def table(title, specs_rows):
    out.append(f"\n### {title}\n")
    out.append("| instance | n | gates | T-count | non-Cl. rot. | d | d/n | f | f_rec | support | E_stab | log2 W_d | log2 W_f |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in specs_rows:
        n = int(r["n"])
        fr = frec(r["spec"])
        out.append("| `{}` | {} | {} | {} | {} | {} | {:.2f} | {} | {} | {} | {} | {:.1f} | {:.1f} |".format(
            r["spec"], n, r["gates"], r["t_count"], r["rotations"], r["d"], int(r["d"]) / n, r["f"],
            "–" if fr is None else fr, r["support"], r["e_stab_max"], fnum(r["log2_work"]), fnum(r["log2_work_f"])))

by_fam = {}
for r in atlas:
    by_fam.setdefault(r["family"], []).append(r)

def pick(fam, pred=lambda r: True):
    return [r for r in by_fam.get(fam, []) if pred(r)]

table("QFT (exact), three inputs", pick("qft", lambda r: int(r["n"]) in (64, 256, 1024)))
table("Approximate QFT (cut = max kept |j-k|)", pick("aqft", lambda r: int(r["n"]) == 256 and r["spec"].split("cut=")[1] in ("1", "2", "4", "16")))
for fam in ["cuccaro", "gidney", "draper"]:
    table(f"Adder: {fam}", pick(fam, lambda r: ("bits=64," in r["spec"] or "bits=256," in r["spec"]) and "cut" not in r["spec"]))
table("Draper with angle cutoff (bits = 64)", pick("draper", lambda r: "cut" in r["spec"]))
table("Shor: one controlled-U_a of the windowed oracle (control |+>)", pick("shorwin", lambda r: "w=4" in r["spec"]))
table("Shor: full order finding (2n counting qubits, windowed oracle w=2, inverse QFT)", pick("shor"))
table("Grover (Toffoli-ladder oracle + diffusion)", pick("grover", lambda r: r["spec"].endswith("it=1") or r["spec"].endswith("it=8")))
table("Trotter: transverse-field Ising (J=h=1, dt=0.1), from |0>", pick("ising", lambda r: int(r["n"]) in (64, 256) and any(r["spec"].endswith(f"steps={s},dt=0.1") for s in (1, 2, 16))))
table("Trotter: Ising step-size sweep (n=64, 4 steps)", pick("ising-dt") + pick("ising-du"))
table("Trotter: XXX Heisenberg from Néel", pick("heis", lambda r: int(r["n"]) in (64, 256) and any(r["spec"].endswith(f"steps={s},dt=0.1") for s in (1, 16))))
table("QAOA MaxCut", pick("qaoa", lambda r: int(r["n"]) in (64, 256)))
table("VQE hardware-efficient ansatz", pick("hea", lambda r: int(r["n"]) in (64, 256)))
table("Phase estimation, stabilizer eigenstate (GHZ, commuting Pauli rotations)", pick("qpe", lambda r: r["spec"].endswith("s=256,kind=stab") or r["spec"].startswith("qpe:t=23")))
table("Phase estimation of an Ising Trotter step on |0> (U^{2^k} by repetition)", pick("qpe-trotter", lambda r: "s=32" in r["spec"]))
table("Coined quantum walk on a 2^m cycle", pick("walk", lambda r: "steps=1" in r["spec"] or "steps=8" in r["spec"]))
table("Toy HHL (exact eigenvalue inversion)", pick("hhl"))
table("Random Clifford+T baseline (L = n/2 layers)", pick("rct", lambda r: int(r["n"]) in (64, 256)))

# magic (ground truth)
if magic:
    out.append("\n### Ground truth at n ≤ 12: stabilizer nullity ν and SRE M2 vs the cheap invariants\n")
    out.append("| instance | n | T-count | d | f | f_rec | max Σlive | ν max | ν final | M2 max | M2 final |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in magic:
        if "error" in r and r["error"]:
            continue
        out.append("| `{}` | {} | {} | {} | {} | {} | {} | {:.0f} | {:.0f} | {:.2f} | {:.2f} |".format(
            r["spec"], r["n"], r["t_count"], r["d"], r["f"], r.get("f_recycled", ""), r.get("live_max", ""),
            fnum(r["nullity_max"]), fnum(r["nullity_final"]), fnum(r["m2_max"]), fnum(r["m2_final"])))

# Mac timings
def jl(name):
    p = os.path.join(MAC, name)
    if not os.path.exists(p):
        return []
    res = []
    for line in open(p):
        line = line.strip()
        if line.startswith("{"):
            try:
                res.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return res

law = [r for r in jl("magic-law.jsonl") if "engine" in r]
eng = [r for r in jl("magic-eng.jsonl") if "engine" in r]
demo = [r for r in jl("magic-demo.jsonl") if "demo" in r]

def minof(rs, key):
    best = {}
    for r in rs:
        k = key(r)
        if k not in best or r["secs"] < best[k]["secs"]:
            best[k] = r
    return best

atlas_by_spec = {r["spec"]: r for r in atlas}
if eng:
    import subprocess
    best = minof(eng, lambda r: (r["spec"], r["engine"]))
    specs = []
    for r in eng:
        if r["spec"] not in specs:
            specs.append(r["spec"])
    out.append("\n### Measured (Mac M1 Pro, 1 thread, min of 3): seconds per engine, with the predicted work\n")
    out.append("| instance | n | d | f | f_rec | log2 G·2^n | log2 W_d | log2 W_f | SV s | cstate s | factored s | recycled s | best/SV |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for s in specs:
        g = {e: best.get((s, e)) for e in ["sv", "cstate", "factored", "recycled"]}
        a = atlas_by_spec.get(s)
        n = g["sv"]["n"] if g["sv"] else (a and int(a["n"]))
        fr = g["recycled"]["f"] if g["recycled"] else "–"
        ts = {e: (g[e]["secs"] if g[e] else None) for e in g}
        bestt = min(t for t in ts.values() if t is not None)
        lg = math.log2(int(a["lowered"])) + int(a["n"]) if a else float("nan")
        out.append("| `{}` | {} | {} | {} | {} | {:.1f} | {} | {} | {} | {} | {} | {} | {} |".format(
            s, n, a["d"] if a else "", a["f"] if a else "", fr, lg,
            a["log2_work"] if a else "", a["log2_work_f"] if a else "",
            *[("%.4f" % ts[e]) if ts[e] is not None else "fail" for e in ["sv", "cstate", "factored", "recycled"]],
            ("%.2g×" % (ts["sv"] / bestt)) if ts["sv"] else "–"))

if demo:
    out.append("\n### Beyond-SV demonstrations (Mac, default threads, min over repeats)\n")
    bd = minof([dict(r, secs=r["sim_secs"]) for r in demo], lambda r: (r["demo"], r.get("n")))
    for k, r in sorted(bd.items(), key=lambda x: str(x[0])):
        out.append("- `" + json.dumps(r) + "`")

open(os.path.join(D, "tables.md"), "w").write("\n".join(out) + "\n")

# ---------------------------------------------------------------- plots
# 1. summary bars: d/n, f/n, f_rec/n, support/n for one representative per family
reps = [
    ("QFT |x> basis, n=256", "qft:n=256,in=basis"),
    ("QFT graph-state in, n=256", "qft:n=256,in=graph"),
    ("AQFT cut=4, graph in, n=256", "qft:n=256,in=graph,cut=4"),
    ("Cuccaro, classical in, 128 b", "cuccaro:bits=128,in=basis"),
    ("Cuccaro, a=|+>, 128 b", "cuccaro:bits=128,in=plusa"),
    ("Gidney, classical in, 128 b", "gidney:bits=128,in=basis"),
    ("Draper, classical in, 128 b", "draper:bits=128,in=basis"),
    ("Draper, a,b=|+>, 128 b", "draper:bits=128,in=plusab"),
    ("Shor ctrl-U_a, 32-bit N", "shorwin:nbits=32,w=4,in=one"),
    ("Shor ctrl-U_a, x half-superposed", "shorwin:nbits=32,w=4,in=half"),
    ("Shor full, 10-bit N", "shor:nbits=10,w=2"),
    ("Grover n=64, 4 it", "grover:n=64,it=4"),
    ("Ising Trotter n=256, 1 step", "ising:n=256,steps=1,dt=0.1"),
    ("Ising Trotter n=256, 16 steps", "ising:n=256,steps=16,dt=0.1"),
    ("Heisenberg n=256, 16 steps", "heis:n=256,steps=16,dt=0.1"),
    ("QAOA p=3 n=256", "qaoa:n=256,p=3,graph=reg3"),
    ("HEA 4 layers n=256", "hea:n=256,layers=4"),
    ("QPE t=23, stab. eigenstate s=256", "qpe:t=23,s=256,kind=stab"),
    ("QPE t=5, Trotter U, s=32", "qpe:t=5,s=32,kind=trotter"),
    ("Quantum walk m=32, 8 steps", "walk:m=32,steps=8"),
    ("HHL t=8 m=7", "hhl:t=8,m=7"),
    ("Clifford+T n=256, t=64", "rct:n=256,L=128,t=64"),
    ("Clifford+T n=256, t=512", "rct:n=256,L=128,t=512"),
]
labels, vals = [], []
for lab, s in reps:
    a = atlas_by_spec.get(s)
    if not a:
        continue
    n = int(a["n"])
    fr = frec(s)
    frv = (17 / n if fr == ">16" else (fr / n if isinstance(fr, int) else float("nan")))
    labels.append(f"{lab}  (n={n})")
    vals.append((int(a["d"]) / n, int(a["f"]) / n, frv, int(a["support"]) / n))
if labels:
    fig, ax = plt.subplots(figsize=(8.2, 0.34 * len(labels) + 1.3))
    names = ["d/n (active dim.)", "f/n (factored)", "f_rec/n (recycled, simulated)", "support bound/n"]
    h = 0.19
    for j in range(4):
        ys = [i + (j - 1.5) * h for i in range(len(labels))]
        xs = [v[j] for v in vals]
        ax.barh(ys, xs, height=h * 0.85, color=C[j], label=names[j])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("fraction of n (1 = no structural shortcut left)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=2, fontsize=8)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(D, "atlas_summary.png"))
    plt.close(fig)

# 2. "when": d_k/n and f_k/n vs fraction of the circuit
profs = sorted(glob.glob(os.path.join(D, "profiles", "*.csv")))
if profs:
    k = len(profs)
    cols = 4
    rws = math.ceil(k / cols)
    fig, axs = plt.subplots(rws, cols, figsize=(11, 2.1 * rws), sharey=True)
    axs = axs.flatten()
    for ax, p in zip(axs, profs):
        name = os.path.basename(p)[:-4]
        spec = name.replace("_", ",").replace(",", ":", 1)
        rowsr = [r for r in csv.DictReader(open(p)) if r["kind"] == "rot"]
        cks = [r for r in csv.DictReader(open(p)) if r["kind"] == "ck"]
        a = None
        for s, r in atlas_by_spec.items():
            if s.replace(":", "_").replace(",", "_").replace("=", "") == name:
                a = r
        if not a or not cks:
            continue
        n, G = int(a["n"]), int(a["gates"])
        xs = [int(r["gate"]) / G for r in rowsr]
        ax.plot(xs, [int(r["d"]) / n for r in rowsr], color=C[0], label="d_k/n")
        ax.plot(xs, [int(r["f"]) / n for r in rowsr], color=C[1], linewidth=1.5, label="f_k/n")
        ax.set_title(a["spec"], fontsize=7.5, color=INK)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
    for ax in axs[k:]:
        ax.axis("off")
    axs[0].legend(fontsize=7, loc="lower right")
    fig.supxlabel("fraction of the circuit (original gates)")
    fig.supylabel("fraction of n")
    fig.tight_layout()
    fig.savefig(os.path.join(D, "when_profiles.png"))
    plt.close(fig)

# 3. ground truth: nullity vs d, and three timelines
if magic:
    fig, axs = plt.subplots(1, 4, figsize=(12, 3.1))
    ax = axs[0]
    xs = [int(r["d"]) for r in magic if r.get("d")]
    ys = [fnum(r["nullity_max"]) for r in magic if r.get("d")]
    ax.plot([0, 14], [0, 14], color=INK2, linewidth=1, linestyle="--")
    ax.scatter(xs, ys, s=22, color=C[0], edgecolor="white", linewidth=0.8, zorder=3)
    for r in magic:
        if r.get("d") and fnum(r["nullity_max"]) <= int(r["d"]) - 4:
            ax.annotate(r["spec"].split(":")[0], (int(r["d"]), fnum(r["nullity_max"])), fontsize=6.5,
                        color=INK2, xytext=(3, -3), textcoords="offset points")
    ax.set_xlabel("active dimension d (final)")
    ax.set_ylabel("max stabilizer nullity ν along the circuit")
    ax.set_title("ν ≤ d always; gap = classical structure", fontsize=8)
    for ax, (spec, title) in zip(axs[1:], [
            ("shorwin:nbits=2,w=1,in=one", "Shor ctrl-U_a (N=3), n=13"),
            ("grover:n=6,it=3", "Grover n=6 (+4 anc), 3 it"),
            ("qft:n=12,in=basis", "QFT|x>, n=12")]):
        p = os.path.join(D, "magic", spec.replace(":", "_").replace(",", "_").replace("=", "") + ".csv")
        if not os.path.exists(p):
            continue
        rr = [r for r in csv.DictReader(open(p)) if int(r["gate"]) >= 0]
        g = [int(r["gate"]) for r in rr]
        ax.plot(g, [int(r["d"]) for r in rr], color=C[0], label="d_k")
        ax.plot(g, [int(r["live"]) for r in rr], color=C[1], linewidth=1.5, label="Σ live (recycled)")
        ax.plot(g, [fnum(r["nullity"]) for r in rr], color=C[2], linewidth=1.5, label="ν_k (exact)")
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("gate")
    axs[1].legend(fontsize=7, loc="center right")
    fig.tight_layout()
    fig.savefig(os.path.join(D, "nullity_vs_d.png"))
    plt.close(fig)

# 4. cost law
if law:
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    for j, (e, lab) in enumerate([("cstate", "compressed state: Σ_j 2^{d_j}"), ("factored", "factored: Σ_j 2^{|factor_j|}")]):
        best = minof([r for r in law if r["engine"] == e], lambda r: r["spec"])
        pts = sorted((r["element_ops"], r["evolve_secs"]) for r in best.values() if r.get("element_ops"))
        if pts:
            ax.loglog([p[0] for p in pts], [p[1] for p in pts], "o-", color=C[j], markersize=5, linewidth=1.5, label=lab)
    ax.set_xlabel("predicted amplitude updates (from the O(gates·n) profile)")
    ax.set_ylabel("measured evolve seconds (1 thread)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(D, "cost_law.png"))
    plt.close(fig)

# 5. engines dot plot
if eng:
    best = minof(eng, lambda r: (r["spec"], r["engine"]))
    specs = []
    for r in eng:
        if r["spec"] not in specs:
            specs.append(r["spec"])
    fig, ax = plt.subplots(figsize=(7.5, 0.33 * len(specs) + 1.2))
    for j, e in enumerate(["sv", "cstate", "factored", "recycled"]):
        xs, ys = [], []
        for i, s in enumerate(specs):
            r = best.get((s, e))
            if r:
                xs.append(max(r["secs"], 1e-5))
                ys.append(i)
        ax.scatter(xs, ys, s=30, color=C[j], edgecolor="white", linewidth=0.8, label=e, zorder=3)
    ax.set_xscale("log")
    ax.set_yticks(range(len(specs)))
    ax.set_yticklabels(specs, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("seconds (Mac M1 Pro, 1 thread, min of 3, incl. compile)")
    ax.legend(fontsize=7, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    fig.savefig(os.path.join(D, "engines.png"))
    plt.close(fig)
print("ok")
