from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from uuid import uuid4

import numpy as np

from .archive import Archive
from .config import RQGMConfig
from .models import (
    AnchorScore,
    CheckpointEvent,
    EvaluatorCandidate,
    EvaluatorSlot,
    RoleTask,
    RunResult,
    UtilityRecord,
    WorkspaceNode,
)
from .protocols import Runtime
from .scheduler import should_expand
from .statistics import best_belief, strictly_better


@dataclass(slots=True)
class RQGM:
    """Standalone implementation of the paper's Algorithm 1 control flow."""

    seed_workspace: object
    tasks: Sequence[RoleTask]
    slots: Sequence[EvaluatorSlot]
    runtime: Runtime
    config: RQGMConfig = field(default_factory=RQGMConfig)
    seed_node_id: str = "seed"
    archive: Archive = field(init=False)
    slot_map: dict[str, EvaluatorSlot] = field(init=False)
    rng: np.random.Generator = field(init=False, repr=False)
    validation_outcomes: int = field(init=False, default=0)
    checkpoint_events: list[CheckpointEvent] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self._validate_topology()
        seed = WorkspaceNode(self.seed_node_id, deepcopy(self.seed_workspace), None, 0)
        self.archive = Archive(seed)
        self.slot_map = {slot.slot_id: deepcopy(slot) for slot in self.slots}
        self.rng = np.random.default_rng(self.config.random_seed)
        self.validation_outcomes = 0
        self.checkpoint_events: list[CheckpointEvent] = []

    def _validate_topology(self) -> None:
        slot_ids = [slot.slot_id for slot in self.slots]
        if len(set(slot_ids)) != len(slot_ids):
            raise ValueError("evaluator slot ids must be unique")
        task_ids = [task.task_id for task in self.tasks]
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("task ids must be globally unique")
        known_slots = set(slot_ids)
        for task in self.tasks:
            if task.evaluator_slot and task.evaluator_slot not in known_slots:
                raise ValueError(f"task {task.task_id} references an unknown evaluator slot")

    @property
    def epoch_vector(self) -> dict[str, int]:
        return {slot_id: slot.epoch for slot_id, slot in sorted(self.slot_map.items())}

    @property
    def frozen_evaluators(self) -> dict[str, EvaluatorCandidate]:
        return {slot_id: slot.incumbent for slot_id, slot in self.slot_map.items()}

    async def run(self) -> RunResult:
        await self._collect_training_feedback(self.archive.nodes[self.seed_node_id])
        checkpoints = self.config.resolved_checkpoints
        checkpoint_index = 0

        while self.validation_outcomes < self.config.validation_budget:
            next_boundary = (
                checkpoints[checkpoint_index]
                if checkpoint_index < len(checkpoints)
                else self.config.validation_budget
            )
            await self._run_epoch_until(next_boundary)
            if next_boundary < self.config.validation_budget:
                await self._checkpoint(next_boundary)
                checkpoint_index += 1

        roles = sorted({task.role_id for task in self.tasks})
        return RunResult(
            validation_outcomes=self.validation_outcomes,
            endpoint=self.archive.endpoint(self.config.epsilon),
            specialists={
                role: self.archive.endpoint(self.config.epsilon, role_id=role) for role in roles
            },
            epoch_vector=self.epoch_vector,
            checkpoints=list(self.checkpoint_events),
            archive_size=len(self.archive.nodes),
        )

    async def _run_epoch_until(self, boundary: int) -> None:
        while self.validation_outcomes < boundary:
            if should_expand(
                self.validation_outcomes,
                len(self.archive.nodes),
                self.config.expansion_alpha,
            ):
                await self._expand_once()
            await self._evaluate_once()

    async def _expand_once(self) -> None:
        parent = self.archive.sample_node_by_cmp(self.rng)
        workspace = await self.runtime.editor.edit(
            deepcopy(parent),
            tuple(deepcopy(node) for node in self.archive.nodes.values()),
            self.config.expansion_budget,
        )
        if workspace is None:
            return
        child = WorkspaceNode(
            node_id=str(uuid4()),
            workspace=workspace,
            parent_id=parent.node_id,
            created_at_step=self.validation_outcomes,
        )
        self.archive.add_node(child)
        await self._collect_training_feedback(child)

    async def _collect_training_feedback(self, node: WorkspaceNode) -> None:
        if self.runtime.training_feedback is None:
            return
        feedback = await self.runtime.training_feedback.collect(
            deepcopy(node),
            tuple(self.tasks),
            deepcopy(self.frozen_evaluators),
            self.config.training_budget,
        )
        self.archive.training_feedback[node.node_id].append(feedback)
        node.training_feedback.append(feedback)

    async def _evaluate_once(self) -> None:
        node = self.archive.sample_node_by_cmp(self.rng)
        task = self.archive.least_measured_cell(node.node_id, self.tasks, self.rng)
        evaluator = self.slot_map[task.evaluator_slot].incumbent if task.evaluator_slot else None
        cache_key = task.task_id
        cached_artifact = node.cached_artifacts.get(cache_key)
        outcome = await self.runtime.task_evaluator.evaluate(
            deepcopy(node),
            task,
            deepcopy(evaluator),
            deepcopy(cached_artifact),
            self.config.validation_call_budget,
        )
        if outcome.artifact_key and outcome.artifact is not None:
            node.cached_artifacts[cache_key] = outcome.artifact
        self.archive.add_record(
            UtilityRecord(
                record_id=str(uuid4()),
                node_id=node.node_id,
                role_id=task.role_id,
                task_id=task.task_id,
                outcome=outcome.outcome,
                epoch_vector=self.epoch_vector,
                evaluator_slot=task.evaluator_slot,
                evaluator_id=evaluator.candidate_id if evaluator else None,
                artifact_key=outcome.artifact_key,
                metadata=dict(outcome.metadata),
            )
        )
        self.validation_outcomes += 1

    async def _score_candidate(
        self,
        candidate: EvaluatorCandidate,
        slot_id: str,
    ) -> AnchorScore | None:
        examples = tuple(await self.runtime.anchor_provider.examples(slot_id))
        if len(examples) < self.config.minimum_anchor_outcomes:
            return None
        outcomes = [
            await self.runtime.anchor_evaluator.evaluate(deepcopy(candidate), deepcopy(example))
            for example in examples
        ]
        if any(outcome not in (0, 1) for outcome in outcomes):
            raise ValueError("anchor evaluator must return binary outcomes")
        successes = sum(outcomes)
        failures = len(outcomes) - successes
        return AnchorScore(
            candidate_id=candidate.candidate_id,
            successes=successes,
            failures=failures,
            best_belief=best_belief(successes, failures, self.config.epsilon),
        )

    async def _checkpoint(self, checkpoint: int) -> None:
        old_vector = self.epoch_vector
        planned: dict[str, EvaluatorCandidate] = {}
        score_log: dict[str, list[AnchorScore]] = {}

        # Every slot decision observes the same pre-transition archive and epoch vector.
        for slot_id in sorted(self.slot_map):
            slot = self.slot_map[slot_id]
            challengers = tuple(
                await self.runtime.challenger_source.challengers(
                    deepcopy(slot),
                    tuple(deepcopy(node) for node in self.archive.nodes.values()),
                )
            )
            unique: dict[str, EvaluatorCandidate] = {slot.incumbent.candidate_id: slot.incumbent}
            for candidate in challengers:
                if candidate.slot_id != slot_id:
                    raise ValueError("challenger belongs to a different evaluator slot")
                unique[candidate.candidate_id] = candidate

            scored: list[tuple[EvaluatorCandidate, AnchorScore]] = []
            for candidate in unique.values():
                score = await self._score_candidate(candidate, slot_id)
                if score is not None:
                    scored.append((candidate, score))
            score_log[slot_id] = [score for _, score in scored]
            incumbent_pair = next(
                (
                    (candidate, score)
                    for candidate, score in scored
                    if candidate.candidate_id == slot.incumbent.candidate_id
                ),
                None,
            )
            if incumbent_pair is None:
                continue
            incumbent_score = incumbent_pair[1].best_belief
            winner, winner_score = max(
                scored,
                key=lambda pair: (
                    pair[1].best_belief,
                    pair[0].candidate_id == slot.incumbent.candidate_id,
                    pair[0].candidate_id,
                ),
            )
            if winner.candidate_id != slot.incumbent.candidate_id and strictly_better(
                winner_score.best_belief, incumbent_score
            ):
                planned[slot_id] = winner

        replacements: dict[str, tuple[str, str]] = {}
        erased = 0
        for slot_id, winner in planned.items():
            slot = self.slot_map[slot_id]
            old_id = slot.incumbent.candidate_id
            slot.incumbent = deepcopy(winner)
            slot.epoch += 1
            replacements[slot_id] = (old_id, winner.candidate_id)
        for slot_id in planned:
            erased += self.archive.invalidate_slot(slot_id, checkpoint)

        self.checkpoint_events.append(
            CheckpointEvent(
                checkpoint=checkpoint,
                old_epoch_vector=old_vector,
                new_epoch_vector=self.epoch_vector,
                replacements=replacements,
                erased_records=erased,
                anchor_scores=score_log,
            )
        )
