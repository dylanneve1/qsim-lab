"""Shared helpers for the approx-modexp cross-checks: locate the paper's code
release (Gidney 2025, Zenodo 10.5281/zenodo.15347487, CC-BY-4.0), keep its
multiprocessing polite on a shared machine, build the paper's ExecutionConfig
for a given prime set, and read the tables dumped by the Rust driver
(`cargo run --release --example approx_modexp -- dump ...`).

Set GIDNEY_SRC to the release's `src/` directory (default:
/dev/shm/qsim/ext/gidney25/x/src).
"""

from __future__ import annotations

import ast
import multiprocessing
import os
import pathlib
import subprocess
import sys

GIDNEY_SRC = os.environ.get("GIDNEY_SRC", "/dev/shm/qsim/ext/gidney25/x/src")
sys.path.insert(0, GIDNEY_SRC)

# keep the paper's pools small (shared machine)
os.cpu_count = lambda: 4  # type: ignore[assignment]
_real_pool = multiprocessing.Pool


def _small_pool(processes=None, *a, **k):
    return _real_pool(min(processes or 4, 4), *a, **k)


multiprocessing.Pool = _small_pool  # type: ignore[assignment]

import numpy as np  # noqa: E402

from facto.algorithm.prep import ExecutionConfig, ProblemConfig  # noqa: E402
from facto.algorithm.prep._precompute_generators import precompute_generators  # noqa: E402
from facto.algorithm.prep._precompute_multipliers import find_multipliers_for_conf  # noqa: E402
from facto.algorithm.prep._precompute_table1 import precompute_table1  # noqa: E402
from facto.algorithm.prep._precompute_table3 import precompute_table3  # noqa: E402
from facto.algorithm.prep._precompute_table4 import precompute_table4  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent
TARGET = os.environ.get("CARGO_TARGET_DIR", str(REPO / "target"))
DRIVER = pathlib.Path(TARGET) / "release" / "examples" / "approx_modexp"


def run_driver(*args: str) -> str:
    """Runs the Rust driver and returns stdout."""
    out = subprocess.run([str(DRIVER), *args], check=True, capture_output=True, text=True)
    return out.stdout


def read_rust_config(path: str | pathlib.Path) -> dict:
    """Parses rust_config.txt written by the driver's `dump` command."""
    d: dict = {}
    for line in pathlib.Path(path).read_text().splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip()
        d[k] = ast.literal_eval(v) if v.startswith("[") else int(v)
    return d


def paper_exec_config(rc: dict, multipliers: dict | None = None) -> ExecutionConfig:
    """The paper's ExecutionConfig for the Rust prime set `rc['periods']`
    (the paper's own table code; only the prime search is bypassed)."""
    conf = ProblemConfig(
        modulus=rc["modulus"],
        num_input_qubits=rc["num_input_qubits"],
        generator=rc["generator"],
        mask_bits=rc["mask_bits"],
        window1=rc["window1"],
        window3a=rc["window3a"],
        window3b=rc["window3b"],
        window4=rc["window4"],
        min_wraparound_gap=rc["min_wraparound_gap"],
        len_accumulator=rc["len_accumulator"],
        parallelism=1,
        num_shots=1,
        rns_primes_bit_length=rc["rns_primes_bit_length"],
        rns_primes_range_start=None,
        rns_primes_range_stop=None,
        rns_primes_extra=(),
        rns_primes_skipped=(),
    )
    periods = list(rc["periods"])
    if multipliers is None:
        multipliers = find_multipliers_for_conf(conf)
    conf = conf.with_edits(
        rns_primes_range_start=min(periods) - 1,
        rns_primes_range_stop=max(periods) + 1,
    )
    generators = precompute_generators(periods=periods)
    t1 = precompute_table1(
        periods=periods,
        generators=generators,
        values=multipliers.values(),
        period_dtype=np.uint32,
    ).reshape((len(periods) + 1, len(multipliers) >> conf.window1, 1 << conf.window1))
    t3a, t3b, t3c = precompute_table3(
        conf=conf, periods=periods, generators=generators, period_dtype=np.uint32, print_progress=False
    )
    t4 = precompute_table4(conf=conf, periods=periods)
    return ExecutionConfig(
        conf=conf,
        periods=np.array(periods, dtype=np.uint32),
        generators=np.array(generators, dtype=np.uint32),
        table1=t1,
        table3a=t3a,
        table3b=t3b,
        table3c=t3c,
        table4=t4,
    )


def eh_multipliers(rc: dict, s: int = 1) -> dict:
    """Window multipliers for the Ekera-Hastad exponent g^a y^-b (registers a:
    m + l qubits, b: l qubits, y = g^((N-1)/2), m = ceil(n/2), l = ceil(m/s)),
    keyed like the paper's `find_multipliers_for_conf` (bit offset, value)."""
    n = rc["modulus"]
    g = rc["generator"]
    m = (n.bit_length() + 1) // 2
    l = -(-m // s)
    y = pow(g, (n - 1) // 2, n)
    yi = pow(y, -1, n)
    w1 = rc["window1"]
    mult = {}
    for reg_len, base, off in ((m + l, g, 0), (l, yi, m + l)):
        for start in range(0, reg_len, w1):
            b = pow(base, 1 << start, n)
            for k in range(1 << w1):
                mult[(off + start, k)] = pow(b, k, n)
    return mult
