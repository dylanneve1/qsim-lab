"""Cross-check of the Rust precomputation (src/shor/approx.rs) against the
paper's own precomputation code (Gidney 2025 release, facto/algorithm/prep).

For each configuration: the Rust driver dumps its prime set and tables; the
paper's code recomputes generators and all tables for the same prime set and
the paper's `_verify_rns_solution` checks the prime set; every table entry must
match exactly (table1 compared mod 2^32, the paper's uint32 dtype). The paper's
own prime search (`find_rns_for_conf`) is also run to show it succeeds at the
same size (it is randomised/parallel, so its primes may differ).

Usage: python3 xcheck_tables.py   (prints one line per configuration)
"""

from __future__ import annotations

import math
import sys

import numpy as np

import gidney_env as ge
import sympy
from facto.algorithm.prep._precompute_rns import FindRnsSolution, _verify_rns_solution, find_rns_for_conf

CONFIGS = [
    # tag, driver args
    ("n8_shor", ["N=143", "g=2", "m=10", "f=6", "mask=2"]),
    ("n10_shor", ["N=899", "g=2", "m=20", "f=8", "mask=paper"]),
    ("n12_shor_w", ["N=3127", "g=3122", "m=24", "f=10", "mask=paper", "w1=4", "w3a=2", "w3b=3", "w4=4"]),
    ("n14_shor", ["N=11663", "g=2", "m=20", "f=12", "mask=paper", "w1=5", "w3a=1", "w3b=2", "w4=4"]),
    ("n16_eh", ["N=56759", "g=2", "mode=eh", "f=12", "mask=paper", "w1=4", "w3a=2", "w3b=3", "w4=5"]),
]


def check(tag: str, args: list[str]) -> bool:
    out = ge.HERE / "xcheck" / f"tables_{tag}"
    ge.run_driver("dump", *args, f"out={out}")
    rc = ge.read_rust_config(out / "rust_config.txt")
    eh = "mode=eh" in args
    mult = ge.eh_multipliers(rc, 1) if eh else None
    conf = ge.paper_exec_config(rc, multipliers=mult)
    ok = True
    msgs = []
    if list(map(int, conf.generators)) != rc["generators"]:
        ok = False
        msgs.append("generators differ")
    t1 = (np.asarray(conf.table1, dtype=np.uint64) & np.uint64(0xFFFFFFFF)).ravel().tolist()
    for name, mine, theirs in [
        ("table1", rc["table1"], t1),
        ("table3a", rc["table3a"], np.asarray(conf.table3a).ravel().tolist()),
        ("table3b", rc["table3b"], np.asarray(conf.table3b).ravel().tolist()),
        ("table3c", rc["table3c"], np.asarray(conf.table3c).ravel().tolist()),
        ("table4", rc["table4"], np.asarray(conf.table4).ravel().tolist()),
    ]:
        if len(mine) != len(theirs):
            ok = False
            msgs.append(f"{name}: length {len(mine)} vs {len(theirs)}")
            continue
        bad = sum(1 for a, b in zip(mine, theirs) if int(a) != int(b))
        if bad:
            ok = False
            msgs.append(f"{name}: {bad} entries differ")
    # the paper's verifier on the Rust prime set
    periods = rc["periods"]
    ell = rc["rns_primes_bit_length"]
    lo, hi = min(periods), max(periods)
    in_range = list(sympy.primerange(lo, hi + 1))
    sol = FindRnsSolution(
        periods=tuple(periods),
        primes_bit_length=ell,
        primes_range_start=lo,
        primes_range_stop=hi + 1,
        primes_extra=(),
        primes_skipped=tuple(p for p in in_range if p not in set(periods)),
    )
    try:
        _verify_rns_solution(conf.conf, sol)
        ver = "paper-verifier-ok"
    except AssertionError as e:  # pragma: no cover
        ok = False
        ver = f"paper-verifier-FAILED: {e}"
    L = math.prod(periods)
    dev = L % rc["modulus"]
    dev = min(dev, rc["modulus"] - dev)
    # the paper's own (randomised) search at the same size
    paper_primes = "-"
    if not eh:
        try:
            from facto.algorithm.prep._precompute_multipliers import find_multipliers_for_conf

            pc = conf.conf.with_edits(rns_primes_range_start=None, rns_primes_range_stop=None)
            s = find_rns_for_conf(pc, not_nil_constraints=find_multipliers_for_conf(pc).values())
            Lp = math.prod(s.periods)
            dp = Lp % rc["modulus"]
            paper_primes = f"{len(s.periods)} primes, dev={min(dp, rc['modulus'] - dp)}"
        except Exception as e:  # noqa: BLE001
            paper_primes = f"paper search failed: {type(e).__name__}: {e}"
    if tag == "n8_shor":
        # reference tables for tests/shor/approx_modexp.rs, produced by the
        # PAPER's code for the Rust prime set
        ref = ge.REPO / "tests" / "data" / "approx-modexp"
        ref.mkdir(parents=True, exist_ok=True)
        with open(ref / "paper_tables_n8.txt", "w") as fh:
            fh.write("# Gidney 2025 release (CC-BY-4.0, doi:10.5281/zenodo.15347487), facto/algorithm/prep,\n")
            fh.write("# tables for N=143 g=2 m=10 w=(2,2,2,2) f=6 mask=2 gap=6 and the prime set below\n")
            fh.write(f"periods = {list(map(int, conf.periods))}\n")
            fh.write(f"generators = {list(map(int, conf.generators))}\n")
            fh.write(f"table1 = {t1}\n")
            for name, arr in (("table3a", conf.table3a), ("table3b", conf.table3b), ("table3c", conf.table3c), ("table4", conf.table4)):
                fh.write(f"{name} = {np.asarray(arr).ravel().tolist()}\n")
    print(
        f"{tag}: |P|={len(periods)} ell={ell} L mod N dev={dev} (< N>>gap={rc['modulus'] >> rc['min_wraparound_gap']}) "
        f"{ver}; tables {'IDENTICAL' if ok else 'DIFFER: ' + '; '.join(msgs)}; paper's own search: {paper_primes}"
    )
    return ok


def main() -> int:
    ok = all([check(t, a) for t, a in CONFIGS])
    print(f"TABLES_OK={ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
