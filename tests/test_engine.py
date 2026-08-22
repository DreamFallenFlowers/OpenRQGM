from collections.abc import Sequence
from dataclasses import dataclass

import pytest

from rqgm import (
    RQGM,
    AnchorExample,
    EvaluationOutcome,
    EvaluatorCandidate,
    EvaluatorSlot,
    RoleTask,
    RQGMConfig,
    Runtime,
)
from rqgm.models import WorkspaceNode


@dataclass
class NoopEditor:
    async def edit(self, parent, archive, budget):  # type: ignore[no-untyped-def]
        return None


@dataclass
class AlternatingEvaluator:
    calls: int = 0

    async def evaluate(self, node, task, evaluator, cached_artifact, budget):  # type: ignore[no-untyped-def]
        self.calls += 1
        artifact = cached_artifact if cached_artifact is not None else {"generated": self.calls}
        if task.kind == "fixed":
            return EvaluationOutcome(1, artifact_key="fixed", artifact=artifact)
        return EvaluationOutcome(
            int(evaluator.artifact["quality"] >= 2), artifact_key="open", artifact=artifact
        )


@dataclass
class Source:
    challenger: EvaluatorCandidate

    async def challengers(
        self, slot: EvaluatorSlot, archive: Sequence[WorkspaceNode]
    ) -> Sequence[EvaluatorCandidate]:
        return [self.challenger]


@dataclass
class Anchors:
    provider_id: str = "private"
    version: str = "1"
    fingerprint: str = "sha256:test-anchor"

    async def examples(self, slot_id: str):  # type: ignore[no-untyped-def]
        return [AnchorExample(str(i), i, True) for i in range(6)]


@dataclass
class AnchorJudge:
    async def evaluate(self, candidate, example):  # type: ignore[no-untyped-def]
        del example
        return int(candidate.artifact["quality"] >= 2)


@dataclass
class Training:
    async def collect(self, node, tasks, evaluators, budget):  # type: ignore[no-untyped-def]
        return {"private_training_hint": node.node_id}


def build_engine() -> tuple[RQGM, AlternatingEvaluator]:
    incumbent = EvaluatorCandidate.create(
        "judge", {"quality": 1}, source="seed", candidate_id="incumbent"
    )
    challenger = EvaluatorCandidate.create(
        "judge", {"quality": 2}, source="workspace", candidate_id="challenger"
    )
    evaluator = AlternatingEvaluator()
    engine = RQGM(
        seed_workspace={"code": "seed"},
        tasks=[
            RoleTask("maker", "open", "learned", "judge"),
            RoleTask("judge", "fixed", "fixed"),
        ],
        slots=[EvaluatorSlot("judge", "judge", incumbent)],
        runtime=Runtime(
            editor=NoopEditor(),
            task_evaluator=evaluator,
            challenger_source=Source(challenger),
            anchor_provider=Anchors(),
            anchor_evaluator=AnchorJudge(),
            training_feedback=Training(),
        ),
        config=RQGMConfig(
            validation_budget=8,
            checkpoints=(4,),
            minimum_anchor_outcomes=5,
            random_seed=3,
        ),
    )
    return engine, evaluator


@pytest.mark.asyncio
async def test_checkpoint_promotes_anchor_winner_and_selectively_erases() -> None:
    engine, _ = build_engine()
    result = await engine.run()
    event = result.checkpoints[0]
    assert event.replacements == {"judge": ("incumbent", "challenger")}
    assert result.epoch_vector == {"judge": 2}
    assert event.erased_records > 0
    fixed = [record for record in engine.archive.records if record.task_id == "fixed"]
    learned_old = [
        record
        for record in engine.archive.records
        if record.task_id == "open" and record.epoch_vector["judge"] == 1
    ]
    assert fixed and all(record.valid for record in fixed)
    assert learned_old and all(not record.valid for record in learned_old)


@pytest.mark.asyncio
async def test_training_feedback_never_enters_validation_records() -> None:
    engine, _ = build_engine()
    await engine.run()
    assert engine.archive.nodes["seed"].training_feedback
    assert all("private_training_hint" not in record.metadata for record in engine.archive.records)
    assert len(engine.archive.records) == engine.config.validation_budget


@pytest.mark.asyncio
async def test_checkpoint_does_not_overshoot_validation_boundary() -> None:
    engine, _ = build_engine()
    await engine.run()
    event = engine.checkpoint_events[0]
    old_epoch_records = [
        record for record in engine.archive.records if record.epoch_vector["judge"] == 1
    ]
    assert len(old_epoch_records) == 4
    assert event.checkpoint == 4


@pytest.mark.asyncio
async def test_runtime_cannot_mutate_frozen_incumbent() -> None:
    engine, _ = build_engine()
    engine.config = RQGMConfig(validation_budget=2, checkpoint_start=32, random_seed=2)

    @dataclass
    class MutatingEvaluator:
        async def evaluate(self, node, task, evaluator, cached_artifact, budget):  # type: ignore[no-untyped-def]
            if evaluator is not None:
                evaluator.artifact["quality"] = 999
            return EvaluationOutcome(1)

    engine.runtime.task_evaluator = MutatingEvaluator()
    await engine.run()
    assert engine.slot_map["judge"].incumbent.artifact == {"quality": 1}


@pytest.mark.asyncio
async def test_multi_slot_replacements_are_planned_from_old_epoch_vector() -> None:
    incumbents = [
        EvaluatorCandidate.create(
            slot_id, {"quality": 1}, source="seed", candidate_id=f"{slot_id}-old"
        )
        for slot_id in ("a", "b")
    ]
    challengers = {
        slot_id: EvaluatorCandidate.create(
            slot_id, {"quality": 2}, source="workspace", candidate_id=f"{slot_id}-new"
        )
        for slot_id in ("a", "b")
    }

    @dataclass
    class RecordingSource:
        seen_epochs: list[tuple[str, int]]

        async def challengers(self, slot, archive):  # type: ignore[no-untyped-def]
            self.seen_epochs.append((slot.slot_id, slot.epoch))
            return [challengers[slot.slot_id]]

    source = RecordingSource([])
    engine = RQGM(
        seed_workspace={},
        tasks=[
            RoleTask("ra", "ta", "learned", "a"),
            RoleTask("rb", "tb", "learned", "b"),
        ],
        slots=[
            EvaluatorSlot("a", "ra", incumbents[0]),
            EvaluatorSlot("b", "rb", incumbents[1]),
        ],
        runtime=Runtime(
            editor=NoopEditor(),
            task_evaluator=AlternatingEvaluator(),
            challenger_source=source,
            anchor_provider=Anchors(),
            anchor_evaluator=AnchorJudge(),
        ),
        config=RQGMConfig(validation_budget=8, checkpoints=(4,)),
    )
    await engine._checkpoint(4)
    assert source.seen_epochs == [("a", 1), ("b", 1)]
    assert engine.epoch_vector == {"a": 2, "b": 2}
    assert set(engine.checkpoint_events[0].replacements) == {"a", "b"}
