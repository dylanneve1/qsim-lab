"""Figures for research/shor/approx-modexp.md from the CSVs in out/.

python3 plot_results.py   (writes PNGs next to this script)
"""

from __future__ import annotations

import csv
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "out"


def read(path: pathlib.Path) -> list[dict]:
    with open(path) as fh:
        return list(csv.DictReader(line for line in fh if not line.startswith("#")))


def peaks(tag: str) -> None:
    p = OUT / f"peaks_{tag}.csv"
    if not p.exists():
        return
    rows = read(p)
    k = np.array([int(r["k"]) for r in rows])
    fig, ax = plt.subplots(figsize=(8, 3.2))
    for key, lab, st in (("unmasked", "exact, unmasked", "-"), ("ideal_masked", "exact arithmetic, masked", "-"),
                         ("actual", "approximate circuit, masked", "-")):
        y = np.array([float(r[key]) for r in rows])
        ax.plot(k, y * len(k), st, lw=0.8, label=lab)
    ax.set_xlabel("frequency peak k (j ≈ k·2^m/r)")
    ax.set_ylabel("P(peak k) × r")
    ax.set_yscale("log")
    ax.legend(fontsize=7)
    ax.set_title(f"Per-peak structure ({tag})")
    fig.tight_layout()
    fig.savefig(HERE / f"peaks_{tag}.png", dpi=130)
    plt.close(fig)


def sweep(name: str, xkey: str, xlabel: str) -> None:
    p = OUT / name
    if not p.exists():
        return
    rows = read(p)
    x = [float(r["val"]) for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ax[0].semilogy(x, [float(r["tv_actual_ideal"]) for r in rows], "o-", label="TV(actual, ideal masked)")
    ax[0].semilogy(x, [float(r["infidelity_shift"]) for r in rows], "s-", label="1 − F (best shift)")
    ax[0].semilogy(x, [float(r["infidelity"]) for r in rows], "^-", label="1 − F (no shift)")
    ax[0].semilogy(x, [float(r["paper_eps"]) / float(r["paper_S"]) for r in rows], "--", label="paper ε/S")
    ax[0].set_xlabel(xlabel)
    ax[0].legend(fontsize=7)
    ax[1].plot(x, [float(r["succ_actual"]) for r in rows], "o-", label="approximate circuit")
    ax[1].plot(x, [float(r["succ_ideal"]) for r in rows], "s-", label="exact arithmetic, same mask")
    ax[1].set_xlabel(xlabel)
    ax[1].set_ylabel("P(success, one run)")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(HERE / name.replace(".csv", ".png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    peaks("n10")
    peaks("n12")
    sweep("sweep_mask_f10.csv", "mask", "mask bits (W = 2^mask of T)")
    sweep("sweep_f_mask4.csv", "f", "accumulator bits f")
    sweep("sweep_w1.csv", "w1", "window1")
    sweep("sweep_w3.csv", "w3", "window3a = window3b")
    sweep("sweep_w4.csv", "w4", "window4")
    print("plots written")
