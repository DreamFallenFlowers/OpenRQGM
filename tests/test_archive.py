import numpy as np

from rqgm.archive import Archive
from rqgm.models import RoleTask, UtilityRecord, WorkspaceNode


def record(node: str, role: str, task: str, outcome: int, slot: str | None = None):
    return UtilityRecord(
        f"{node}-{task}-{outcome}",
        node,
        role,
        task,
        outcome,
        {"judge": 1},
        slot,
        "e1" if slot else None,
    )


def test_clade_counts_include_descendants_and_respect_erasure() -> None:
    archive = Archive(WorkspaceNode("root", {}, None, 0))
    archive.add_node(WorkspaceNode("child", {}, "root", 1))
    archive.add_record(record("root", "fixed", "a", 1))
    archive.add_record(record("child", "open", "b", 0, "judge"))
    assert archive.clade_counts("root") == (1, 1)
    assert archive.invalidate_slot("judge", checkpoint=8) == 1
    assert archive.clade_counts("root") == (1, 0)
    assert archive.records[0].valid


def test_least_measured_selection_balances_role_then_task() -> None:
    archive = Archive(WorkspaceNode("root", {}, None, 0))
    tasks = [
        RoleTask("writer", "draft-a", "fixed"),
        RoleTask("writer", "draft-b", "fixed"),
        RoleTask("reviewer", "review", "fixed"),
    ]
    archive.add_record(record("root", "writer", "draft-a", 1))
    chosen = archive.least_measured_cell("root", tasks, np.random.default_rng(1))
    assert chosen.role_id == "reviewer"
