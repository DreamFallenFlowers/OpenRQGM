import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import patch

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


def test_context_filter_is_deterministic_and_keeps_complete_examples() -> None:
    rows = [
        {
            "pull_request_title": "small",
            "patch": "x" * 10,
            "description": "description",
            "hint": "hint",
            "label": "APPROVE",
        },
        {
            "pull_request_title": "large",
            "patch": "x" * 500,
            "description": "description",
            "hint": "hint",
            "label": "REQUEST_CHANGES",
        },
    ]
    limit = MODULE.review_example_size(rows[0])
    assert MODULE.context_eligible_rows(rows, limit) == [rows[0]]
    assert rows[0]["patch"] == "x" * 10


def test_payload_batches_respect_count_and_byte_caps() -> None:
    items = ["a" * 20, "b" * 20, "c" * 20]
    single = len(json.dumps(items[0]).encode("utf-8"))
    batches = MODULE.payload_batches(
        items,
        max_items=3,
        max_payload_bytes=single * 2,
        payload=lambda item: item,
    )
    assert batches == [items[:2], items[2:]]


def test_codex_discovery_uses_explicit_cli_path(tmp_path) -> None:
    executable = tmp_path / "codex"
    executable.write_text("", encoding="utf-8")
    with patch.dict(MODULE.os.environ, {"CODEX_CLI_PATH": str(executable)}, clear=True):
        assert MODULE.find_codex_executable() == str(executable)


def test_codex_jsonl_token_usage_is_recorded_in_both_metrics() -> None:
    output = "\n".join(
        [
            "non-json warning",
            json.dumps(
                {
                    "type": "turn.completed",
                    "usage": {
                        "input_tokens": 100,
                        "cached_input_tokens": 20,
                        "output_tokens": 10,
                        "reasoning_output_tokens": 2,
                    },
                }
            ),
        ]
    )
    usage = MODULE.token_usage_from_jsonl(output)
    assert usage["raw_total_tokens"] == 110
    assert usage["blended_tokens"] == 150


def test_model_call_ledger_survives_process_restart(tmp_path) -> None:
    ledger = tmp_path / "model-calls.jsonl"
    first = MODULE.CodexCli("test-model", 1, tmp_path, ledger)
    first._record_call({"purpose": "one", "raw_total_tokens": 3})
    second = MODULE.CodexCli("test-model", 1, tmp_path, ledger)
    assert second.calls == [{"purpose": "one", "raw_total_tokens": 3}]


def test_app_server_usage_mapping_preserves_token_metrics() -> None:
    usage = MODULE.AppServerWorker._usage_fields(
        {
            "totalTokens": 110,
            "inputTokens": 100,
            "cachedInputTokens": 20,
            "cacheWriteInputTokens": 0,
            "outputTokens": 10,
            "reasoningOutputTokens": 2,
        }
    )
    assert usage["raw_total_tokens"] == 110
    assert usage["blended_tokens"] == 150


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


def test_ablation_condition_topologies_are_distinct_and_intentional() -> None:
    workspace = MODULE.seed_workspace()
    incumbent = MODULE.EvaluatorCandidate.create(
        "code-reviewer",
        {"workspace": workspace},
        source="seed",
        candidate_id="reviewer-seed",
    )
    verifier = MODULE.condition_components("verifier_only", incumbent)
    fixed = MODULE.condition_components("fixed_reviewer", incumbent)
    coevolving = MODULE.condition_components("coevolving_reviewer", incumbent)

    assert [task.task_id for task in verifier[0]] == ["coder-polyglot-tests"]
    assert verifier[1] == [] and verifier[2] is None and verifier[4] is False
    assert len(fixed[0]) == len(coevolving[0]) == 3
    assert fixed[1] == [] and fixed[2] is incumbent and fixed[4] is True
    assert len(coevolving[1]) == 1 and coevolving[2] is None and coevolving[4] is True
    fixed_learned = next(task for task in fixed[0] if task.task_id == "coder-learned-review")
    evolved_learned = next(task for task in coevolving[0] if task.task_id == "coder-learned-review")
    assert fixed_learned.kind == "fixed"
    assert evolved_learned.kind == "learned"
    assert fixed_learned.evaluator_slot is None
    assert evolved_learned.evaluator_slot == "code-reviewer"


def test_anchor_inference_is_chunked_without_changing_binary_examples() -> None:
    rows = [
        {
            "pull_request_title": f"title-{index}",
            "patch": f"patch-{index}",
            "description": "description",
            "hint": "hint",
            "label": "APPROVE",
        }
        for index in range(5)
    ]
    provider = MODULE.PrivateCraveAnchors(rows)
    runner = FakeAgentRunner()

    class BatchClient:
        def __init__(self) -> None:
            self.purposes: list[str] = []

        async def json(self, prompt, schema, purpose):  # type: ignore[no-untyped-def]
            del prompt, schema
            self.purposes.append(purpose)
            examples = runner.contexts[-1]["examples"]
            return {
                "predictions": [{"id": example["id"], "label": "APPROVE"} for example in examples]
            }

    client = BatchClient()
    evaluator = MODULE.BatchedAnchorEvaluator(client, provider, runner, batch_size=2)
    candidate = MODULE.EvaluatorCandidate.create(
        "code-reviewer",
        {"workspace": MODULE.seed_workspace()},
        source="test",
        candidate_id="candidate",
    )
    examples = asyncio.run(provider.examples("code-reviewer"))
    assert asyncio.run(evaluator.evaluate(candidate, examples[0])) == 1
    assert client.purposes == [
        "anchor:candidate:batch-0",
        "anchor:candidate:batch-1",
        "anchor:candidate:batch-2",
    ]
    assert len(evaluator.predictions["candidate"]) == 5


def test_parallel_anchor_batches_preserve_all_predictions() -> None:
    rows = [
        {
            "pull_request_title": f"title-{index}",
            "patch": f"patch-{index}",
            "description": "description",
            "hint": "hint",
            "label": "APPROVE",
        }
        for index in range(8)
    ]
    provider = MODULE.PrivateCraveAnchors(rows)

    class JsonAgentRunner:
        def prompt(self, workspace, operation, context):  # type: ignore[no-untyped-def]
            del workspace, operation
            return json.dumps(context)

    class ConcurrentClient:
        def __init__(self) -> None:
            self.active = 0
            self.max_active = 0

        async def json(self, prompt, schema, purpose):  # type: ignore[no-untyped-def]
            del schema, purpose
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            await asyncio.sleep(0.01)
            examples = json.loads(prompt)["examples"]
            self.active -= 1
            return {
                "predictions": [{"id": example["id"], "label": "APPROVE"} for example in examples]
            }

    client = ConcurrentClient()
    evaluator = MODULE.BatchedAnchorEvaluator(
        client, provider, JsonAgentRunner(), batch_size=2, parallelism=4
    )
    candidate = MODULE.EvaluatorCandidate.create(
        "code-reviewer",
        {"workspace": MODULE.seed_workspace()},
        source="test",
        candidate_id="candidate",
    )
    examples = asyncio.run(provider.examples("code-reviewer"))
    assert asyncio.run(evaluator.evaluate(candidate, examples[0])) == 1
    assert client.max_active == 4
    assert len(evaluator.predictions["candidate"]) == 8


def test_matched_ablation_configs_only_vary_declared_fields() -> None:
    config_dir = MODULE_PATH.parent / "configs" / "ablations"
    configs = [
        json.loads(path.read_text(encoding="utf-8")) for path in sorted(config_dir.glob("*.json"))
    ]
    assert len(configs) == 6
    allowed = {
        "claim",
        "experiment_condition",
        "validation_budget",
        "checkpoints",
        "output",
    }
    projections = [
        {key: value for key, value in config.items() if key not in allowed} for config in configs
    ]
    assert all(projection == projections[0] for projection in projections)
    assert {config["validation_budget"] for config in configs} == {512, 1024}
    assert {config["experiment_condition"] for config in configs} == {
        "verifier_only",
        "fixed_reviewer",
        "coevolving_reviewer",
    }


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


def test_parallel_training_preserves_deterministic_result_order() -> None:
    evaluator = MODULE.CodingTaskEvaluator(
        client=object(),
        runner=object(),
        agent_runner=object(),
        validation_tasks=[],
        crave_validation=[],
        random_seed=7,
        parallelism=3,
    )
    tasks = [MODULE.PolyglotTask("python", Path(f"task-{index}")) for index in range(3)]
    active = 0
    max_active = 0

    async def fake_generate(self, node, task, *, phase, purpose_prefix):  # type: ignore[no-untyped-def]
        nonlocal active, max_active
        del self, node, phase, purpose_prefix
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.01 * (4 - int(task.path.name.rsplit("-", 1)[1])))
        active -= 1
        return {
            "test_outcome": 1,
            "attempts": 1,
            "sandbox": {"returncode": 0},
        }

    node = MODULE.WorkspaceNode("node", MODULE.seed_workspace(), None, 0)
    expected = [task.task_id for task in evaluator._ordered_tasks("node", tasks, "training")]
    with patch.object(MODULE.CodingTaskEvaluator, "_generate_artifact", fake_generate):
        results = asyncio.run(evaluator.training_samples(node, tasks, 3))
    assert max_active == 3
    assert [item["task"] for item in results] == expected


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
