from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from .config import RQGMConfig
from .engine import RQGM
from .models import (
    AnchorExample,
    EvaluationOutcome,
    EvaluatorCandidate,
    EvaluatorSlot,
    RoleTask,
    WorkspaceNode,
)
from .protocols import Runtime


@dataclass(slots=True)
class ToyEditor:
    async def edit(self, parent, archive, budget):  # type: ignore[no-untyped-def]
        del budget
        highest_solution = max(node.workspace["solution"] for node in archive)
        highest_threshold = max(node.workspace["judge_threshold"] for node in archive)
        return {
            "solution": min(5, max(parent.workspace["solution"], highest_solution + 1)),
            "judge_threshold": min(
                3, max(parent.workspace["judge_threshold"], highest_threshold + 1)
            ),
        }


@dataclass(slots=True)
class ToyTaskEvaluator:
    async def evaluate(self, node, task, evaluator, cached_artifact, budget):  # type: ignore[no-untyped-def]
        del budget
        if task.task_id == "solve-open-task":
            artifact = (
                cached_artifact if cached_artifact is not None else node.workspace["solution"]
            )
            threshold = evaluator.artifact["threshold"]
            return EvaluationOutcome(
                int(artifact >= threshold),
                artifact_key=f"solution:{node.node_id}",
                artifact=artifact,
            )
        if task.task_id == "judge-grounded-task":
            return EvaluationOutcome(int(node.workspace["judge_threshold"] == 3))
        raise KeyError(task.task_id)


@dataclass(slots=True)
class ToyChallengers:
    async def challengers(
        self, slot: EvaluatorSlot, archive: Sequence[WorkspaceNode]
    ) -> Sequence[EvaluatorCandidate]:
        return [
            EvaluatorCandidate.create(
                slot.slot_id,
                {"threshold": node.workspace["judge_threshold"]},
                source=f"workspace:{node.node_id}",
                parent_id=slot.incumbent.candidate_id,
                candidate_id=f"judge-threshold-{node.workspace['judge_threshold']}",
            )
            for node in archive
        ]


@dataclass(slots=True)
class ToyAnchorProvider:
    provider_id: str = "toy-ground-truth"
    version: str = "v1"
    fingerprint: str = "sha256:toy-values-0-through-5-threshold-3"

    async def examples(self, slot_id: str) -> Sequence[AnchorExample]:
        if slot_id != "judge":
            raise KeyError(slot_id)
        return [AnchorExample(str(value), value, int(value >= 3)) for value in range(6)]


@dataclass(slots=True)
class ToyAnchorEvaluator:
    async def evaluate(self, candidate: EvaluatorCandidate, example: AnchorExample) -> int:
        prediction = int(example.artifact >= candidate.artifact["threshold"])
        return int(prediction == example.expected)


@dataclass(slots=True)
class ToyTrainingFeedback:
    async def collect(self, node, tasks, evaluators, budget):  # type: ignore[no-untyped-def]
        del tasks, evaluators, budget
        return {"hint": "increase solution quality and calibrate the judge", "node": node.node_id}


def build_toy_engine(*, budget: int = 24, seed: int = 7) -> RQGM:
    incumbent = EvaluatorCandidate.create(
        "judge",
        {"threshold": 1},
        source="seed",
        candidate_id="judge-threshold-1",
    )
    return RQGM(
        seed_workspace={"solution": 0, "judge_threshold": 1},
        tasks=[
            RoleTask("solver", "solve-open-task", "learned", "judge"),
            RoleTask("judge", "judge-grounded-task", "fixed"),
        ],
        slots=[EvaluatorSlot("judge", "judge", incumbent)],
        runtime=Runtime(
            editor=ToyEditor(),
            task_evaluator=ToyTaskEvaluator(),
            challenger_source=ToyChallengers(),
            anchor_provider=ToyAnchorProvider(),
            anchor_evaluator=ToyAnchorEvaluator(),
            training_feedback=ToyTrainingFeedback(),
        ),
        config=RQGMConfig(
            validation_budget=budget,
            checkpoints=(8, 16) if budget > 16 else (max(1, budget // 2),),
            minimum_anchor_outcomes=5,
            random_seed=seed,
        ),
    )


def result_to_dict(result: Any) -> dict[str, Any]:
    return asdict(result)
