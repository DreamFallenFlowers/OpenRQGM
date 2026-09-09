import sys
from pathlib import Path

import pytest

EXAMPLE = Path(__file__).parents[1] / "examples" / "paper_coding"
sys.path.insert(0, str(EXAMPLE))

from agent_workspace import seed_workspace, validate_workspace, workspace_from_files  # noqa: E402
from polyglot import (  # noqa: E402
    LANGUAGES,
    TEST_COMMANDS,
    PolyglotTask,
    editable_files,
    safe_relative,
    split_balanced,
    split_counts,
)


@pytest.fixture
def polyglot_root(tmp_path: Path) -> Path:
    editable = {
        "cpp": Path("solution.cpp"),
        "go": Path("solution.go"),
        "java": Path("src/main/java/Solution.java"),
        "javascript": Path("solution.js"),
        "python": Path("solution.py"),
        "rust": Path("src/lib.rs"),
    }
    for language in LANGUAGES:
        for index in range(40):
            task = tmp_path / language / "exercises" / "practice" / f"task-{index:02d}"
            source = task / editable[language]
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text("", encoding="utf-8")
    return tmp_path


def test_balanced_split_contains_six_disjoint_languages(polyglot_root: Path) -> None:
    train, validation, test = split_balanced(polyglot_root, 7, 1, 2, 1)
    assert {task.language for task in train} == set(LANGUAGES)
    assert {task.language for task in validation} == set(LANGUAGES)
    assert {task.language for task in test} == set(LANGUAGES)
    ids = [{task.task_id for task in split} for split in (train, validation, test)]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])


def test_global_count_split_matches_paper_cardinalities(polyglot_root: Path) -> None:
    train, validation, test = split_counts(polyglot_root, 7, 10, 49, 166)
    assert (len(train), len(validation), len(test)) == (10, 49, 166)
    ids = [{task.task_id for task in split} for split in (train, validation, test)]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])


def test_each_language_has_a_bounded_edit_surface(polyglot_root: Path) -> None:
    train, _, _ = split_balanced(polyglot_root, 7, 1, 1, 1)
    for task in train:
        paths = [path.relative_to(task.path).as_posix() for path in editable_files(task)]
        assert paths
        assert all(not path.startswith(".meta/") and "test" not in path.lower() for path in paths)


@pytest.mark.parametrize("path", ["../secret", "/host", "a/../../secret", "C:\\secret"])
def test_path_traversal_is_rejected(path: str) -> None:
    with pytest.raises(ValueError):
        safe_relative(path)


def test_meta_agent_can_replace_whole_codebase_but_not_escape() -> None:
    workspace = workspace_from_files(
        [
            {"path": "agent.py", "content": seed_workspace()["files"]["agent.py"]},
            {"path": "helpers/policy.py", "content": "POLICY = 'changed'\n"},
        ]
    )
    validate_workspace(workspace)
    assert set(workspace["files"]) == {"agent.py", "helpers/policy.py"}
    with pytest.raises(ValueError):
        workspace_from_files([{"path": "../engine.py", "content": "owned"}])


def test_polyglot_task_identity_includes_language() -> None:
    task = PolyglotTask("python", Path("exercise"))
    assert task.task_id == "python/exercise"


def test_java_runner_enables_and_requires_the_complete_suite() -> None:
    command = TEST_COMMANDS["java"][2]
    assert "s/@Disabled" in command
    assert "tests started" in command
    assert "0 tests skipped" in command
    assert "test_status" in command
