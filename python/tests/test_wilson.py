"""Tests for qsimlab.qec.wilson_interval against an independent reference:
the Wilson interval is the set of p where |p_hat - p| <= z*sqrt(p(1-p)/n),
so its endpoints are the roots of that equality, found here by bisection."""
import math
import pytest

from qsimlab.qec import wilson_interval

Z95 = 1.959963984540054


def _root(f, lo, hi):
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if (f(lo) > 0) == (f(mid) > 0):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _reference(k, n, z=Z95):
    ph = k / n
    g = lambda p: abs(ph - p) - z * math.sqrt(p * (1 - p) / n)
    lo = 0.0 if k == 0 else _root(g, 0.0, ph)
    hi = 1.0 if k == n else _root(g, ph, 1.0)
    return lo, hi


@pytest.mark.parametrize("k,n", [(0, 10), (1, 10), (5, 10), (10, 10), (3, 1000),
                                 (0, 1), (1, 1), (17, 123456), (499, 1000)])
def test_matches_root_finding_reference(k, n):
    lo, hi = wilson_interval(k, n)
    rlo, rhi = _reference(k, n)
    assert lo == pytest.approx(rlo, abs=1e-10)
    assert hi == pytest.approx(rhi, abs=1e-10)
    assert 0.0 <= lo <= k / n <= hi <= 1.0


def test_zero_shots_is_uninformative():
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_symmetry():
    lo, hi = wilson_interval(3, 20)
    lo2, hi2 = wilson_interval(17, 20)
    assert lo == pytest.approx(1 - hi2) and hi == pytest.approx(1 - lo2)


def test_narrows_with_shots():
    w = [b - a for a, b in (wilson_interval(n // 10, n) for n in (10, 100, 1000))]
    assert w[0] > w[1] > w[2]
