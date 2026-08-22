import pytest
from scipy.stats import beta

from rqgm.statistics import best_belief, strictly_better


def test_best_belief_is_exact_beta_lower_quantile() -> None:
    assert best_belief(8, 2, 0.05) == pytest.approx(beta.ppf(0.05, 9, 3))


def test_best_belief_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        best_belief(-1, 0)
    with pytest.raises(ValueError):
        best_belief(1, 1, 1.0)


def test_strict_improvement_keeps_numerical_ties() -> None:
    assert not strictly_better(0.5 + 1e-13, 0.5)
    assert strictly_better(0.6, 0.5)
