from __future__ import annotations

from dataclasses import dataclass

from .scheduler import exponential_checkpoints, normalize_checkpoints


@dataclass(frozen=True, slots=True)
class RQGMConfig:
    validation_budget: int = 128
    epsilon: float = 0.05
    expansion_alpha: float = 0.5
    checkpoints: tuple[int, ...] = ()
    checkpoint_start: int = 32
    checkpoint_ratio: float = 2.0
    minimum_anchor_outcomes: int = 5
    expansion_budget: int | None = None
    training_budget: int | None = None
    validation_call_budget: int | None = None
    random_seed: int = 0

    def __post_init__(self) -> None:
        if self.validation_budget <= 0:
            raise ValueError("validation_budget must be positive")
        if not 0.0 < self.epsilon < 1.0:
            raise ValueError("epsilon must lie strictly between zero and one")
        if not 0.0 < self.expansion_alpha < 1.0:
            raise ValueError("expansion_alpha must lie strictly between zero and one")
        if self.minimum_anchor_outcomes <= 0:
            raise ValueError("minimum_anchor_outcomes must be positive")
        if self.checkpoints:
            normalize_checkpoints(self.checkpoints, self.validation_budget)

    @property
    def resolved_checkpoints(self) -> tuple[int, ...]:
        if self.checkpoints:
            return self.checkpoints
        return exponential_checkpoints(
            self.checkpoint_start,
            self.checkpoint_ratio,
            self.validation_budget,
        )
