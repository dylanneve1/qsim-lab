#!/usr/bin/env python3
"""Run the cotengra configurations of quimb_paths.py directly on a network JSON file.

Network file (index ids are ints; written by the Rust engine or by --export below):

    {"inputs": [[int, ...], ...],   one list of index ids per tensor, in axis order
     "output": [int, ...],          open indices ([] for an amplitude)
     "size_dict": {"<int>": dim}}   dimension of every index id

An index id may appear in more than two tensors (a hyperindex); cotengra handles that natively.

Environment and usage:

    export PYTHONPATH=/dev/shm/qsim/ext/tn-py OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
    PY=/dev/shm/qsim/venv/bin/python

    # search (same two configs/budgets as quimb_paths.py); prints one JSON line per config
    nice -n 15 prlimit --as=3000000000 $PY cotengra_on_network.py networks/quimb_syc53_m10_s0.json \
        --configs unsliced sliced --budget std --parallel 2 [--out cotengra_network_test.jsonl]

    # export quimb's ADCRS-simplified <0^n|C|0^n> network of a circuit file (structure only, plus
    # a sidecar <name>.arrays.json with the tensor entries and quimb's norm exponent, so the same
    # network can be contracted elsewhere)
    nice -n 15 $PY cotengra_on_network.py --export circuits/syc53_m10_s0.txt networks/quimb_syc53_m10_s0.json

Fields printed: the network fields and tree fields of quimb_paths.py (n_tensors, n_inds,
n_hyper_inds, cost, log10_cost, total_flops, log2_max_size, contraction_width, nslices, ...), plus
search_time_s (wall time of HyperOptimizer.search only), trials, best_method and versions.
"""

import argparse
import datetime
import json
import math
import os
import platform
import sys
import time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import quimb_paths as qp  # noqa: E402


def load_network(path):
    with open(path) as f:
        d = json.load(f)
    inputs = [tuple(int(i) for i in term) for term in d["inputs"]]
    output = tuple(int(i) for i in d["output"])
    size_dict = {int(k): int(v) for k, v in d["size_dict"].items()}
    for term in inputs:
        for ix in term:
            if ix not in size_dict:
                raise ValueError(f"index {ix} has no size")
    return inputs, output, size_dict


def network_stats(inputs, output, size_dict):
    counts = Counter(ix for term in inputs for ix in term)
    for ix in output:
        counts[ix] += 1
    sizes = Counter(size_dict[ix] for ix in counts)
    return {
        "n_tensors": len(inputs),
        "n_inds": len(counts),
        "n_hyper_inds": sum(1 for c in counts.values() if c > 2),
        "max_ind_appearances": max(counts.values()),
        "ind_size_hist": {str(k): v for k, v in sorted(sizes.items())},
        "max_tensor_rank": max(len(t) for t in inputs),
        "log2_max_tensor_size": max(sum(math.log2(size_dict[i]) for i in t) for t in inputs),
    }


def as_str_labels(inputs, output, size_dict):
    """cotengra's greedy driver (cotengrust) needs single-character str labels (quimb does the same
    mapping internally): int id i -> cotengra.get_symbol(i)."""
    from cotengra import get_symbol as sym

    return (
        [tuple(sym(i) for i in t) for t in inputs],
        tuple(sym(i) for i in output),
        {sym(k): v for k, v in size_dict.items()},
    )


def search(path, config, budget, parallel, out):
    inputs, output, size_dict = load_network(path)
    opt, kw = qp.make_opt(config, budget, parallel)
    before = qp.load_snapshot()
    cpu0 = qp.children_cpu_s()
    t0 = time.perf_counter()
    tree = opt.search(*as_str_labels(inputs, output, size_dict))
    t1 = time.perf_counter()
    qp.kill_pool()
    worker_cpu = qp.children_cpu_s() - cpu0
    tree, fix = qp.enforce_target(tree, config)
    rec = {
        "network": os.path.relpath(os.path.abspath(path), HERE),
        "config": config,
        "budget": budget,
        "optimizer": kw,
        **network_stats(inputs, output, size_dict),
        **qp.tree_stats(tree),
        "post_sliced_to_target": bool(fix),
        **({"pre_fix": fix["pre_fix"]} if fix else {}),
        **qp.resource_info(),
        "search_time_s": t1 - t0,
        "trials": len(opt.scores),
        "worker_cpu_s": worker_cpu,
        "worker_cpu_share": worker_cpu / max(parallel * (t1 - t0), 1e-9),
        "best_method": opt.best.get("params", {}).get("method"),
        "versions": qp.versions(),
        "load_before": before,
        "load_after": qp.load_snapshot(),
        "date": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "host": platform.node(),
    }
    line = json.dumps(rec)
    print(line, flush=True)
    if out:
        with open(out, "a") as f:
            f.write(line + "\n")
    return rec


def export(circuit_path, out_path):
    import numpy as np

    circ, n, _ = qp.build_circuit(circuit_path)
    tn = circ.amplitude_tn("0" * n, simplify_sequence="ADCRS")
    ids = {}
    inputs = [[ids.setdefault(ix, len(ids)) for ix in t.inds] for t in tn]
    size_dict = {str(i): int(tn.ind_size(ix)) for ix, i in ids.items()}
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"inputs": inputs, "output": [], "size_dict": size_dict}, f, separators=(",", ":"))
        f.write("\n")

    arrays = [np.asarray(t.data, dtype=np.complex128) for t in tn]
    exponent = float(getattr(tn, "exponent", 0.0))
    side = {
        "network": os.path.basename(out_path),
        "source": f"quimb {qp.versions()['quimb']} Circuit.amplitude_tn('0'*{n}, simplify_sequence='ADCRS') of "
        + os.path.relpath(os.path.abspath(circuit_path), HERE),
        "tensor_order": "same order as 'inputs' in the network file; entries row-major over the listed index ids",
        "tensors": [{"shape": list(a.shape), "re": a.real.ravel().tolist(), "im": a.imag.ravel().tolist()} for a in arrays],
        "exponent": exponent,
        "value_rule": "amplitude = (full contraction of the tensors) * 10**exponent",
        "amplitude": None,
        "amplitude_note": "not computed here; quimb_timings.jsonl holds <0^53|C|0^53> contracted by quimb from "
        "the identical simplified network (same circuit, same deterministic ADCRS simplification)",
    }
    side_path = out_path[: -len(".json")] + ".arrays.json"
    with open(side_path, "w") as f:
        json.dump(side, f, separators=(",", ":"))
        f.write("\n")
    print(json.dumps({"network": out_path, "arrays": side_path, **network_stats(
        [tuple(r) for r in inputs], (), {int(k): v for k, v in size_dict.items()}),
        "exponent": exponent}))


def main(argv=None):
    ap = argparse.ArgumentParser(description="cotengra configs on a network JSON file")
    ap.add_argument("network", nargs="?")
    ap.add_argument("--export", nargs=2, metavar=("CIRCUIT", "OUT_JSON"))
    ap.add_argument("--configs", nargs="+", default=["unsliced", "sliced"])
    ap.add_argument("--budget", default="standard", choices=sorted(qp.BUDGETS))
    ap.add_argument("--parallel", type=int, default=2, help="cotengra worker processes (machine rule: <= 2)")
    ap.add_argument("--out", default=None, help="JSONL file to append to")
    args = ap.parse_args(argv)
    if args.export:
        export(*args.export)
        return 0
    if not args.network:
        ap.error("network file required")
    for config in args.configs:
        search(args.network, config, args.budget, args.parallel, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
