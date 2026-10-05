"""Markdown tables for research/shor/approx-modexp.md from out/verify.txt,
out/dist.txt and out/moon.txt (python3 summarize.py)."""

from __future__ import annotations

import pathlib
import re

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "out"


def kv(line: str) -> dict:
    return dict(re.findall(r"([A-Za-z_\[\]()0-9|<>.^~+-]+)=([^ ]+)", line))


def verify_table(path: pathlib.Path) -> None:
    if not path.exists():
        return
    print(f"\n#### {path.name}\n")
    print("| N | mode | m | f | mask | ℓ | \\|P\\| | additions | branches / stream | streams | bad sign | dirty | bad residue | bad acc | max\\|δ\\| | mean\\|δ\\| | δ range | max Δ_N |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    cur = None
    rows = []
    hist = None
    for line in path.read_text().splitlines():
        if line.startswith("N="):
            cur = kv(line)
            cur["streams"] = 0
            cur["bad"] = [0, 0, 0, 0]
            rows.append(cur)
        elif line.startswith("dev_hist") and cur is not None:
            ks = [int(x.split(":")[0]) for x in line.split()[1].split(",")]
            cur["range"] = f"[{min(ks)}, {max(ks)}]"
        elif line.startswith("verify[") and cur is not None:
            d = kv(line)
            cur["streams"] += 1
            for i, k in enumerate(("bad_sign", "dirty", "bad_residue", "bad_acc")):
                cur["bad"][i] += int(d[k])
            cur["branches"] = d["branches"]
            cur["max_dev"] = d["max_dev"]
            cur["mean_dev"] = d["mean_dev"]
            cur["md"] = d["max_mod_dev"]
    for r in rows:
        mode = "EH" if "regs" in r and r["regs"].count("(") > 1 else "Shor"
        print(
            f"| {r['N']} | {mode} | {r['m']} | {r['f']} | {r['mask']} | {r['ell']} | {r['|P|']} | {r['additions']} | "
            f"{int(r.get('branches', 0)):,} | {r['streams']} | {r['bad'][0]} | {r['bad'][1]} | {r['bad'][2]} | {r['bad'][3]} | "
            f"{r.get('max_dev', '-')} | {r.get('mean_dev', '-')} | {r.get('range', '-')} | {r.get('md', '-')} |"
        )


if __name__ == "__main__":
    verify_table(OUT / "verify.txt")
    verify_table(OUT / "moon.txt")
