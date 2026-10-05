import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))


@pytest.fixture
def rng():
    return np.random.default_rng(12345)


def tv_distance(counts: dict, probs: dict, shots: int) -> float:
    keys = set(counts) | set(probs)
    return 0.5 * sum(abs(counts.get(k, 0) / shots - probs.get(k, 0.0)) for k in keys)
