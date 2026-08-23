import asyncio
import importlib.util
import sys
from pathlib import Path

MODULE_PATH = Path(__file__).parents[1] / "examples" / "paper_coding" / "run.py"
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("paper_coding_run", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_canonical_hash_is_order_independent() -> None:
    assert MODULE.canonical_hash({"a": 1, "b": 2}) == MODULE.canonical_hash({"b": 2, "a": 1})


def test_review_example_does_not_include_label() -> None:
    row = {
        "pull_request_title": "title",
        "patch": "diff",
        "description": "description",
        "hint": "hint",
        "label": "APPROVE",
    }
    assert "label" not in MODULE.review_example("example", row)


def test_codex_discovery_avoids_windows_store_alias() -> None:
    assert "WindowsApps" not in MODULE.find_codex_executable()


def test_challenger_source_skips_incumbent_artifact() -> None:
    incumbent_workspace = MODULE.seed_workspace()
    incumbent = MODULE.EvaluatorCandidate.create(
        "code-reviewer",
        {"workspace": incumbent_workspace},
        source="seed",
        candidate_id="seed",
    )
    slot = MODULE.EvaluatorSlot("code-reviewer", "reviewer", incumbent)
    changed = MODULE.seed_workspace()
    changed["files"]["README.md"] = "changed"
    nodes = [
        MODULE.WorkspaceNode("a", incumbent_workspace, None, 0),
        MODULE.WorkspaceNode("b", changed, "a", 1),
    ]
    candidates = asyncio.run(MODULE.ChallengerSource().challengers(slot, nodes))
    assert [candidate.artifact["workspace"] for candidate in candidates] == [changed]


class FakeClient:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def json(self, prompt, schema, purpose):  # type: ignore[no-untyped-def]
        del schema, purpose
        self.prompts.append(prompt)
        return {"label": "APPROVE"}


class FakeAgentRunner:
    def __init__(self) -> None:
        self.contexts: list[dict[str, object]] = []

    def prompt(self, workspace, operation, context):  # type: ignore[no-untyped-def]
        del workspace
        self.contexts.append(context)
        return f"{operation}: {context}"


def code_artifact(
    task: str, *, fixed: bool = False, reviews: dict[str, str] | None = None
) -> dict[str, object]:
    return {
        "kind": "polyglot-solution",
        "phase": "validation",
        "task": task,
        "task_path": task,
        "language": "python",
        "instructions": f"instructions for {task}",
        "starters": {f"{task}.py": "def solve(): pass\n"},
        "tests": {f"{task}_test.py": "def test_solve(): assert solve() == 1\n"},
        "replacements": {f"{task}.py": "def solve(): return 1\n"},
        "patch": f"patch for {task}",
        "test_outcome": 1,
        "sandbox": {"returncode": 0},
        "attempts": 1,
        "fixed_recorded": fixed,
        "reviews": reviews or {},
    }


def make_evaluator(client: FakeClient) -> object:
    return MODULE.CodingTaskEvaluator(
        client=client,
        runner=object(),
        agent_runner=FakeAgentRunner(),
        validation_tasks=[],
        crave_validation=[],
        random_seed=7,
        repair_attempts=0,
    )


def test_fixed_validation_consumes_distinct_cached_samples() -> None:
    evaluator = make_evaluator(FakeClient())
    node = MODULE.WorkspaceNode(
        "node",
        MODULE.seed_workspace(),
        None,
        0,
        cached_artifacts={
            "polyglot:validation:a": code_artifact("a"),
            "polyglot:validation:b": code_artifact("b"),
        },
    )
    task = MODULE.RoleTask("coder", "coder-polyglot-tests", "fixed")
    first = asyncio.run(evaluator.evaluate(node, task, None, None, None))
    second = asyncio.run(evaluator.evaluate(node, task, None, None, None))
    assert first.artifact_key == "polyglot:validation:a"
    assert second.artifact_key == "polyglot:validation:b"
    assert first.metadata["sample_id"] != second.metadata["sample_id"]


def test_new_evaluator_reuses_patch_with_full_task_context() -> None:
    client = FakeClient()
    evaluator = make_evaluator(client)
    artifact = code_artifact("a", fixed=True, reviews={"old": "APPROVE"})
    node = MODULE.WorkspaceNode(
        "node",
        MODULE.seed_workspace(),
        None,
        0,
        cached_artifacts={"polyglot:validation:a": artifact},
    )
    task = MODULE.RoleTask("coder", "coder-learned-review", "learned", "code-reviewer")
    candidate = MODULE.EvaluatorCandidate.create(
        "code-reviewer",
        {"workspace": MODULE.seed_workspace()},
        source="test",
        candidate_id="new",
    )
    outcome = asyncio.run(evaluator.evaluate(node, task, candidate, None, None))
    assert outcome.artifact_key == "polyglot:validation:a"
    assert outcome.artifact["reviews"] == {"old": "APPROVE", "new": "APPROVE"}
    context = evaluator.agent_runner.contexts[0]
    assert "instructions for a" in str(context)
    assert "def solve(): pass" in str(context)
    assert "def test_solve()" in str(context)
    assert "patch for a" in str(context)


def test_crave_validation_does_not_repeat_per_node() -> None:
    client = FakeClient()
    rows = [
        {
            "pull_request_title": f"title-{index}",
            "patch": f"patch-{index}",
            "description": "description",
            "hint": "hint",
            "label": "APPROVE",
        }
        for index in range(2)
    ]
    evaluator = MODULE.CodingTaskEvaluator(
        client=client,
        runner=object(),
        agent_runner=FakeAgentRunner(),
        validation_tasks=[],
        crave_validation=rows,
        random_seed=7,
        repair_attempts=0,
    )
    node = MODULE.WorkspaceNode("node", MODULE.seed_workspace(), None, 0)
    task = MODULE.RoleTask("reviewer", "reviewer-crave-validation", "fixed")
    first = asyncio.run(evaluator.evaluate(node, task, None, None, None))
    node.cached_artifacts[first.artifact_key] = first.artifact
    second = asyncio.run(evaluator.evaluate(node, task, None, None, None))
    assert first.metadata["sample_id"] != second.metadata["sample_id"]


def test_polyglot_order_is_language_round_robin() -> None:
    evaluator = MODULE.CodingTaskEvaluator(
        client=object(),
        runner=object(),
        agent_runner=object(),
        validation_tasks=[],
        crave_validation=[],
        random_seed=7,
    )
    tasks = [
        MODULE.PolyglotTask(language, Path(f"{language}-{index}"))
        for language in MODULE.LANGUAGES
        for index in range(2)
    ]
    ordered = evaluator._ordered_tasks("node", tasks, "validation")
    width = len(MODULE.LANGUAGES)
    assert len({task.language for task in ordered[:width]}) == width
    assert len({task.language for task in ordered[width : 2 * width]}) == width


def test_task_material_includes_repository_tests(tmp_path) -> None:  # type: ignore[no-untyped-def]
    task = tmp_path / "exercise"
    docs = task / ".docs"
    docs.mkdir(parents=True)
    (docs / "instructions.md").write_text("Implement solve.", encoding="utf-8")
    (task / "exercise.py").write_text("def solve(): pass\n", encoding="utf-8")
    (task / "exercise_test.py").write_text(
        "def test_solve(): assert solve() == 1\n", encoding="utf-8"
    )
    evaluator = MODULE.CodingTaskEvaluator(FakeClient(), object(), FakeAgentRunner(), [], [], 7, 0)
    material = evaluator._task_material(MODULE.PolyglotTask("python", task))
    assert material["instructions"] == "Implement solve."
    assert "exercise_test.py" in material["tests"]
    assert "def test_solve()" in material["tests"]["exercise_test.py"]
