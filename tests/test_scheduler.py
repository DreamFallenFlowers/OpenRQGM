import pytest

from rqgm.scheduler import exponential_checkpoints, normalize_checkpoints, should_expand


def test_ucb_air_gate_follows_archive_growth_rule() -> None:
    assert not should_expand(0, 1, 0.5)
    assert should_expand(1, 1, 0.5)
    assert not should_expand(3, 2, 0.5)
    assert should_expand(4, 2, 0.5)


def test_exponential_checkpoint_schedule() -> None:
    assert exponential_checkpoints(4, 2.0, 20) == (4, 8, 16)


def test_checkpoint_validation_rejects_duplicates_and_budget_edge() -> None:
    with pytest.raises(ValueError):
        normalize_checkpoints((4, 4), 10)
    with pytest.raises(ValueError):
        normalize_checkpoints((10,), 10)
