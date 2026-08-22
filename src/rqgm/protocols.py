from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from .models import (
    AnchorExample,
    EvaluationOutcome,
    EvaluatorCandidate,
    EvaluatorSlot,
    RoleTask,
    WorkspaceNode,
)


class WorkspaceEditor(Protocol):
    async def edit(
        self,
        parent: WorkspaceNode,
        archive: Sequence[WorkspaceNode],
        budget: int | None,
    ) -> Any | None: ...


class TaskEvaluator(Protocol):
    async def evaluate(
        self,
        node: WorkspaceNode,
        task: RoleTask,
        evaluator: EvaluatorCandidate | None,
        cached_artifact: Any | None,
        budget: int | None,
    ) -> EvaluationOutcome: ...


class TrainingFeedback(Protocol):
    async def collect(
        self,
        node: WorkspaceNode,
        tasks: Sequence[RoleTask],
        evaluators: dict[str, EvaluatorCandidate],
        budget: int | None,
    ) -> Any: ...


class ChallengerSource(Protocol):
    async def challengers(
        self,
        slot: EvaluatorSlot,
        archive: Sequence[WorkspaceNode],
    ) -> Sequence[EvaluatorCandidate]: ...


class AnchorProvider(Protocol):
    @property
    def provider_id(self) -> str: ...

    @property
    def version(self) -> str: ...

    @property
    def fingerprint(self) -> str: ...

    async def examples(self, slot_id: str) -> Sequence[AnchorExample]: ...


class AnchorEvaluator(Protocol):
    async def evaluate(
        self,
        candidate: EvaluatorCandidate,
        example: AnchorExample,
    ) -> int: ...


@dataclass(slots=True)
class Runtime:
    editor: WorkspaceEditor
    task_evaluator: TaskEvaluator
    challenger_source: ChallengerSource
    anchor_provider: AnchorProvider
    anchor_evaluator: AnchorEvaluator
    training_feedback: TrainingFeedback | None = None


@dataclass(slots=True)
class CallableWorkspaceEditor:
    function: Callable[[WorkspaceNode, Sequence[WorkspaceNode], int | None], Awaitable[Any | None]]

    async def edit(self, parent, archive, budget):  # type: ignore[no-untyped-def]
        return await self.function(parent, archive, budget)


@dataclass(slots=True)
class CallableTaskEvaluator:
    function: Callable[
        [WorkspaceNode, RoleTask, EvaluatorCandidate | None, Any | None, int | None],
        Awaitable[EvaluationOutcome],
    ]

    async def evaluate(self, node, task, evaluator, cached_artifact, budget):  # type: ignore[no-untyped-def]
        return await self.function(node, task, evaluator, cached_artifact, budget)
