from __future__ import annotations

from math import isclose

import numpy as np
from scipy.stats import beta


def best_belief(successes: int, failures: int, epsilon: float = 0.05) -> float:
    """Return the paper's epsilon lower quantile of a Beta(1+S, 1+F) posterior."""
    if successes < 0 or failures < 0:
        raise ValueError("success and failure counts must be non-negative")
    if not 0.0 < epsilon < 1.0:
        raise ValueError("epsilon must lie strictly between zero and one")
    return float(beta.ppf(epsilon, 1 + successes, 1 + failures))


def thompson_draw(successes: int, failures: int, rng: np.random.Generator) -> float:
    """Sample a Beta working posterior used by CMP Thompson selection."""
    return float(rng.beta(1 + successes, 1 + failures))


def strictly_better(candidate: float, incumbent: float, *, atol: float = 1e-12) -> bool:
    """Ties retain the incumbent, including numerically indistinguishable values."""
    return candidate > incumbent and not isclose(candidate, incumbent, abs_tol=atol)
