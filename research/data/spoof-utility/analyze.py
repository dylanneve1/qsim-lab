#!/usr/bin/env python3
"""Analysis for research/spoof-utility.md.

Reads the SPD campaign JSONL files in this directory (campaign*.jsonl), the
Kim et al. exact / MPS curves (kim/*.txt, from the authors' data repository)
and the experiment values extracted by extract_kim.py (kim_*_experiment.csv);
writes tables (tables.md) and figures (*.png).
"""
import glob, json, math, os, sys
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

# colours: reference categorical palette, light mode, fixed order
C_SPD, C_EXP, C_UNMIT, C_MPS = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
C_EXACT = "#52514e"

FIGS = {
    "3a": ("$M_z$ (5 steps)", "kim/fig3a_exact.txt", +1),
    "3b": ("weight-10 $X_{13,29,31}Y_{9,30}Z_{8,12,17,28,32}$ (5 steps)", "kim/fig3b_exact.txt", +1),
    "3c": ("weight-17 $X_{8}Y_{1}Z_{8}$ (5 steps)", "kim/fig3c_exact.txt", +1),
    "4a": ("weight-17 $X_{8}Y_{8}Z_{1}$ (5 steps + RX)", None, +1),
    "4b": ("$\\langle Z_{62}\\rangle$ (20 steps)", None, +1),
}


def load_runs():
    runs = []
    for f in sorted(glob.glob("campaign*.jsonl")):
        for line in open(f):
            line = line.strip()
            if line.startswith("{"):
                r = json.loads(line)
                r["src"] = f
                runs.append(r)
    return runs


def load_xy(path):
    d = {}
    for line in open(path):
        a, b = line.split(",")[:2]
        d[round(float(a), 4)] = float(b)
    return d


def load_exp(fig):
    rows = np.genfromtxt(f"kim_fig{fig}_experiment.csv", delimiter=",", names=True)
    return rows


def converged(runs, fig, lattice=127, steps=None, depol=0.0, max_weight=-1):
    """theta -> list of (delta, value, seconds, run) sorted by decreasing delta."""
    by = defaultdict(list)
    for r in runs:
        if r["fig"] != fig or r["lattice"] != lattice or r["aborted"] or r["depol"] != depol:
            continue
        if steps is not None and r["steps"] != steps:
            continue
        if r.get("max_weight", -1) != max_weight:
            continue
        by[round(r["theta"], 4)].append((r["delta"], r["value"], r["seconds"], r))
    for k in by:
        # keep the latest run per delta
        dd = {}
        for t in by[k]:
            dd[t[0]] = t
        by[k] = sorted(dd.values(), key=lambda t: -t[0])
    return dict(sorted(by.items()))


def best_and_err(lst):
    """Smallest-delta value and convergence error estimate |v(δ_min) − v(δ_prev)|."""
    v = lst[-1][1]
    e = abs(lst[-1][1] - lst[-2][1]) if len(lst) > 1 else float("nan")
    return v, e, lst[-1][0], lst[-1][2]


def main():
    runs = load_runs()
    out = []
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig_all, axes = plt.subplots(1, 5, figsize=(22, 4.4))
    for ax, (fig, (title, exact_path, _)) in zip(axes, FIGS.items()):
        conv = converged(runs, fig)
        exact = load_xy(exact_path) if exact_path else {}
        exp = load_exp(fig)
        out.append(f"\n### Fig. {fig}: {title}\n")
        out.append("| θ_h | SPD (δ_min) | conv. err | δ_min | time/pt (s) | exact | SPD−exact | ZNE exp. | exp. 68% CI | exp−SPD |")
        out.append("|---|---|---|---|---|---|---|---|---|---|")
        expmap = {round(float(r["theta_h"]), 4): r for r in exp}
        xs, ys, es = [], [], []
        maxerr = 0.0
        for th, lst in conv.items():
            v, e, d, sec = best_and_err(lst)
            xs.append(th)
            ys.append(v)
            es.append(e if not math.isnan(e) else 0)
            ex = exact.get(th)
            ecell = f"{ex:+.4f}" if ex is not None else "–"
            dcell = f"{v - ex:+.1e}" if ex is not None else "–"
            if ex is not None:
                maxerr = max(maxerr, abs(v - ex))
            er = expmap.get(th)
            if er is not None:
                expcell = f"{er['mitigated']:+.3f}"
                cicell = f"[{er['boot_lo68']:+.3f}, {er['boot_hi68']:+.3f}]"
                diff = f"{er['mitigated'] - v:+.3f}"
            else:
                expcell = cicell = diff = "–"
            out.append(
                f"| {th:.4f} | {v:+.5f} | {e:.1e} | {d:.0e} | {sec:.2f} | {ecell} | {dcell} | {expcell} | {cicell} | {diff} |"
            )
        if exact:
            out.append(f"\nmax |SPD − exact| over the θ grid: **{maxerr:.1e}**\n")
        # plot
        if exact:
            ex_x = sorted(exact)
            ax.plot(ex_x, [exact[x] for x in ex_x], color=C_EXACT, lw=2.0, label="exact (Kim et al.)", zorder=1)
        if fig in ("4a", "4b") and os.path.exists(f"kim/fig{fig}_MPS.txt"):
            m = load_xy(f"kim/fig{fig}_MPS.txt")
            mx = sorted(m)
            ax.plot(mx, [m[x] for x in mx], color=C_MPS, lw=1.5, ls="--", label="MPS (Kim et al.)", zorder=1)
        ax.errorbar(xs, ys, yerr=es, fmt="o", ms=5, color=C_SPD, label="SPD (this work)", zorder=3)
        th = exp["theta_h"]
        lo = exp["mitigated"] - exp["boot_lo68"]
        hi = exp["boot_hi68"] - exp["mitigated"]
        ax.errorbar(th, exp["mitigated"], yerr=[np.abs(lo), np.abs(hi)], fmt="s", ms=5, color=C_EXP, capsize=2, label="experiment, ZNE", zorder=2)
        ax.plot(th, exp["unmitigated"], "^", ms=5, mfc="none", color=C_UNMIT, label="experiment, unmitigated", zorder=2)
        ax.set_title(f"Fig. {fig}: " + title, fontsize=9)
        ax.set_xlabel("$\\theta_h$")
        ax.axhline(0, color="#ccc", lw=0.8, zorder=0)
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig_all.tight_layout()
    fig_all.savefig("figures_kim.png", dpi=130)

    # convergence plot: |SPD(δ) − exact| (max over θ) vs δ, for 3a–3c
    fc, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for fig, col in zip(["3a", "3b", "3c"], [C_SPD, C_EXP, C_UNMIT]):
        conv = converged(runs, fig)
        exact = load_xy(FIGS[fig][1])
        errs = defaultdict(float)
        for th, lst in conv.items():
            if th not in exact:
                continue
            for d, v, s, r in lst:
                errs[d] = max(errs[d], abs(v - exact[th]))
        ds = sorted(errs)
        ax[0].loglog(ds, [max(errs[d], 1e-12) for d in ds], "o-", color=col, label=f"Fig. {fig}")
    ax[0].set_xlabel("threshold δ")
    ax[0].set_ylabel("max over θ of |SPD − exact|")
    ax[0].invert_xaxis()
    ax[0].legend(frameon=False)
    ax[0].grid(alpha=0.3, which="both")
    conv = converged(runs, "4b")
    sel = [0.3, 0.5, 0.6, 0.7, 0.8, 1.0]
    cols = [C_SPD, C_EXP, C_UNMIT, C_MPS, "#e87ba4", "#008300"]
    for th, col in zip(sel, cols):
        if th in conv:
            lst = conv[th]
            ax[1].semilogx([t[0] for t in lst], [t[1] for t in lst], "o-", color=col, label=f"θ={th}")
    ax[1].invert_xaxis()
    ax[1].set_xlabel("threshold δ")
    ax[1].set_ylabel("$\\langle Z_{62}\\rangle$, 20 steps")
    ax[1].legend(frameon=False, fontsize=8, ncol=2)
    ax[1].grid(alpha=0.3, which="both")
    fc.tight_layout()
    fc.savefig("convergence.png", dpi=130)

    open("tables.md", "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
