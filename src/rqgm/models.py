from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal
from uuid import uuid4

EvaluationKind = Literal["fixed", "learned"]


@dataclass(frozen=True, slots=True)
class RoleTask:
    role_id: str
    task_id: str
    kind: EvaluationKind
    evaluator_slot: str | None = None

    def __post_init__(self) -> None:
        if self.kind == "learned" and not self.evaluator_slot:
            raise ValueError("learned tasks require an evaluator slot")
        if self.kind == "fixed" and self.evaluator_slot is not None:
            raise ValueError("fixed tasks cannot name an evaluator slot")


@dataclass(frozen=True, slots=True)
class EvaluatorCandidate:
    candidate_id: str
    slot_id: str
    artifact: Any
    source: str
    parent_id: str | None = None

    @classmethod
    def create(
        cls,
        slot_id: str,
        artifact: Any,
        *,
        source: str,
        parent_id: str | None = None,
        candidate_id: str | None = None,
    ) -> EvaluatorCandidate:
        return cls(candidate_id or str(uuid4()), slot_id, artifact, source, parent_id)


@dataclass(slots=True)
class EvaluatorSlot:
    slot_id: str
    role_id: str
    incumbent: EvaluatorCandidate
    epoch: int = 1

    def __post_init__(self) -> None:
        if self.incumbent.slot_id != self.slot_id:
            raise ValueError("incumbent belongs to a different evaluator slot")


@dataclass(slots=True)
class WorkspaceNode:
    node_id: str
    workspace: Any
    parent_id: str | None
    created_at_step: int
    valid: bool = True
    cached_artifacts: dict[str, Any] = field(default_factory=dict)
    training_feedback: list[Any] = field(default_factory=list)


@dataclass(slots=True)
class UtilityRecord:
    record_id: str
    node_id: str
    role_id: str
    task_id: str
    outcome: int
    epoch_vector: dict[str, int]
    evaluator_slot: str | None
    evaluator_id: str | None
    artifact_key: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    valid: bool = True
    invalidated_at_checkpoint: int | None = None
    invalidation_reason: str | None = None

    def __post_init__(self) -> None:
        if self.outcome not in (0, 1):
            raise ValueError("RQGM search utility must be binary")


@dataclass(frozen=True, slots=True)
class EvaluationOutcome:
    outcome: int
    artifact_key: str | None = None
    artifact: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.outcome not in (0, 1):
            raise ValueError("evaluation outcome must be zero or one")


@dataclass(frozen=True, slots=True)
class AnchorExample:
    example_id: str
    artifact: Any
    expected: Any


@dataclass(slots=True)
class AnchorScore:
    candidate_id: str
    successes: int
    failures: int
    best_belief: float


@dataclass(slots=True)
class CheckpointEvent:
    checkpoint: int
    old_epoch_vector: dict[str, int]
    new_epoch_vector: dict[str, int]
    replacements: dict[str, tuple[str, str]]
    erased_records: int
    anchor_scores: dict[str, list[AnchorScore]]


@dataclass(frozen=True, slots=True)
class Endpoint:
    node_id: str
    successes: int
    failures: int
    best_belief: float


@dataclass(slots=True)
class RunResult:
    validation_outcomes: int
    endpoint: Endpoint
    specialists: dict[str, Endpoint]
    epoch_vector: dict[str, int]
    checkpoints: list[CheckpointEvent]
    archive_size: int
