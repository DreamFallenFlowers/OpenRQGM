from __future__ import annotations


def should_expand(validation_outcomes: int, archive_size: int, alpha: float) -> bool:
    """UCB-Air archive-growth gate from Algorithm 1.

    The seed must be evaluated before the first expansion; using max(N, 1) would
    incorrectly create a child before any validation evidence exists.
    """
    if validation_outcomes < 0 or archive_size < 1:
        raise ValueError("invalid scheduler state")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    return validation_outcomes > 0 and validation_outcomes**alpha >= archive_size


def exponential_checkpoints(start: int, ratio: float, budget: int) -> tuple[int, ...]:
    if start <= 0 or budget <= 0:
        raise ValueError("start and budget must be positive")
    if ratio <= 1.0:
        raise ValueError("checkpoint ratio must exceed one")
    checkpoints: list[int] = []
    value = start
    while value < budget:
        if not checkpoints or value > checkpoints[-1]:
            checkpoints.append(value)
        value = max(value + 1, int(round(value * ratio)))
    return tuple(checkpoints)


def normalize_checkpoints(checkpoints: tuple[int, ...], budget: int) -> tuple[int, ...]:
    if budget <= 0:
        raise ValueError("budget must be positive")
    if any(value <= 0 or value >= budget for value in checkpoints):
        raise ValueError("checkpoints must be strictly inside the validation budget")
    if tuple(sorted(set(checkpoints))) != checkpoints:
        raise ValueError("checkpoints must be unique and increasing")
    return checkpoints
