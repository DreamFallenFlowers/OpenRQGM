from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence

import numpy as np

from .models import Endpoint, RoleTask, UtilityRecord, WorkspaceNode
from .statistics import best_belief, thompson_draw


class Archive:
    """Workspace tree plus epoch-local utility records and lineage-only feedback."""

    def __init__(self, seed: WorkspaceNode) -> None:
        if seed.parent_id is not None:
            raise ValueError("the seed node cannot have a parent")
        self.nodes: dict[str, WorkspaceNode] = {seed.node_id: seed}
        self.children: dict[str, list[str]] = defaultdict(list)
        self.records: list[UtilityRecord] = []
        self.training_feedback: dict[str, list[object]] = defaultdict(list)

    def add_node(self, node: WorkspaceNode) -> None:
        if node.node_id in self.nodes:
            raise ValueError(f"duplicate node id: {node.node_id}")
        if node.parent_id not in self.nodes:
            raise ValueError("a child must reference an existing parent")
        self.nodes[node.node_id] = node
        self.children[node.parent_id].append(node.node_id)

    def add_record(self, record: UtilityRecord) -> None:
        if record.node_id not in self.nodes:
            raise ValueError("utility record references an unknown node")
        self.records.append(record)

    def descendants(self, node_id: str) -> set[str]:
        result: set[str] = set()
        stack = [node_id]
        while stack:
            current = stack.pop()
            if current in result:
                continue
            result.add(current)
            stack.extend(self.children.get(current, ()))
        return result

    def valid_records(
        self,
        *,
        node_ids: Iterable[str] | None = None,
        role_id: str | None = None,
        task_id: str | None = None,
    ) -> list[UtilityRecord]:
        allowed = set(node_ids) if node_ids is not None else None
        return [
            record
            for record in self.records
            if record.valid
            and (allowed is None or record.node_id in allowed)
            and (role_id is None or record.role_id == role_id)
            and (task_id is None or record.task_id == task_id)
        ]

    def counts(
        self,
        *,
        node_ids: Iterable[str] | None = None,
        role_id: str | None = None,
        task_id: str | None = None,
    ) -> tuple[int, int]:
        records = self.valid_records(node_ids=node_ids, role_id=role_id, task_id=task_id)
        successes = sum(record.outcome for record in records)
        return successes, len(records) - successes

    def clade_counts(self, node_id: str) -> tuple[int, int]:
        return self.counts(node_ids=self.descendants(node_id))

    def sample_node_by_cmp(self, rng: np.random.Generator) -> WorkspaceNode:
        candidates = [node for node in self.nodes.values() if node.valid]
        if not candidates:
            raise RuntimeError("archive contains no valid nodes")
        draws = {
            node.node_id: thompson_draw(*self.clade_counts(node.node_id), rng)
            for node in candidates
        }
        best_draw = max(draws.values())
        tied = sorted(node_id for node_id, draw in draws.items() if draw == best_draw)
        return self.nodes[tied[int(rng.integers(0, len(tied)))]]

    def least_measured_cell(
        self,
        node_id: str,
        tasks: Sequence[RoleTask],
        rng: np.random.Generator,
    ) -> RoleTask:
        by_role: dict[str, list[RoleTask]] = defaultdict(list)
        for task in tasks:
            by_role[task.role_id].append(task)
        if not by_role:
            raise RuntimeError("at least one role/task cell is required")

        role_counts = {
            role: sum(
                len(self.valid_records(node_ids=[node_id], task_id=task.task_id)) for task in cells
            )
            for role, cells in by_role.items()
        }
        minimum_role = min(role_counts.values())
        roles = sorted(role for role, count in role_counts.items() if count == minimum_role)
        role = roles[int(rng.integers(0, len(roles)))]

        task_counts = {
            task.task_id: len(self.valid_records(node_ids=[node_id], task_id=task.task_id))
            for task in by_role[role]
        }
        minimum_task = min(task_counts.values())
        task_ids = sorted(
            task_id for task_id, count in task_counts.items() if count == minimum_task
        )
        task_id = task_ids[int(rng.integers(0, len(task_ids)))]
        return next(task for task in by_role[role] if task.task_id == task_id)

    def invalidate_slot(self, slot_id: str, checkpoint: int) -> int:
        erased = 0
        for record in self.records:
            if record.valid and record.evaluator_slot == slot_id:
                record.valid = False
                record.invalidated_at_checkpoint = checkpoint
                record.invalidation_reason = f"evaluator slot {slot_id} changed"
                erased += 1
        return erased

    def endpoint(self, epsilon: float, *, role_id: str | None = None) -> Endpoint:
        scored: list[Endpoint] = []
        for node in self.nodes.values():
            if not node.valid:
                continue
            successes, failures = self.counts(node_ids=[node.node_id], role_id=role_id)
            scored.append(
                Endpoint(
                    node_id=node.node_id,
                    successes=successes,
                    failures=failures,
                    best_belief=best_belief(successes, failures, epsilon),
                )
            )
        if not scored:
            raise RuntimeError("cannot select an endpoint from an empty archive")
        return max(scored, key=lambda item: (item.best_belief, -item.failures, item.node_id))
