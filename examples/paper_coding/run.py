from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import tempfile
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

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
from rqgm.persistence import save_state

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "paper-coding"
POLYGLOT = ROOT / "data" / "polyglot-benchmark"


def canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def load_rows(split: str) -> list[dict[str, Any]]:
    path = DATA / f"crave-{split}.json"
    if not path.exists():
        raise FileNotFoundError(f"run prepare_data.py first; missing {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def python_tasks(seed: int) -> list[Path]:
    base = POLYGLOT / "python" / "exercises" / "practice"
    tasks = sorted(path for path in base.iterdir() if path.is_dir())
    random.Random(seed).shuffle(tasks)
    return tasks


def sample_rows(rows: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    if count > len(rows):
        raise ValueError(f"requested {count} rows from a split of size {len(rows)}")
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    return [rows[index] for index in indices[:count]]


def review_example(example_id: str, row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": example_id,
        "title": row["pull_request_title"],
        "patch": row["patch"],
        "description": row["description"],
        "hint": row["hint"],
    }


def find_codex_executable() -> str:
    configured = os.environ.get("CODEX_CLI_PATH")
    if configured and Path(configured).is_file():
        return configured
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        bundled = sorted(
            (Path(local_app_data) / "OpenAI" / "Codex" / "bin").glob("*/codex.exe"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if bundled:
            return str(bundled[0])
    command = shutil.which("codex.cmd")
    if command:
        return command
    raise FileNotFoundError("Codex desktop CLI is not available")


@dataclass(slots=True)
class CodexCli:
    model: str
    timeout: int
    workdir: Path
    calls: list[dict[str, Any]] = field(default_factory=list)

    async def json(self, prompt: str, schema: dict[str, Any], purpose: str) -> dict[str, Any]:
        return await asyncio.to_thread(self._json_sync, prompt, schema, purpose)

    def _json_sync(self, prompt: str, schema: dict[str, Any], purpose: str) -> dict[str, Any]:
        executable = find_codex_executable()
        self.workdir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="openrqgm-codex-") as temp:
            temp_path = Path(temp)
            schema_path = temp_path / "schema.json"
            output_path = temp_path / "output.json"
            schema_path.write_text(json.dumps(schema), encoding="utf-8")
            command = [
                executable,
                "exec",
                "--ephemeral",
                "--ignore-user-config",
                "--skip-git-repo-check",
                "--sandbox",
                "read-only",
                "--disable",
                "apps",
                "--disable",
                "browser_use",
                "--disable",
                "computer_use",
                "--disable",
                "plugins",
                "-m",
                self.model,
                "-C",
                str(self.workdir),
                "--output-schema",
                str(schema_path),
                "-o",
                str(output_path),
                "-",
            ]
            print(f"[model:start] {purpose}", flush=True)
            started = time.monotonic()
            completed = subprocess.run(
                command,
                input=prompt,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                timeout=self.timeout,
                check=False,
            )
            event = {
                "purpose": purpose,
                "model": self.model,
                "returncode": completed.returncode,
                "prompt_sha256": canonical_hash(prompt),
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }
            token_match = re.search(r"tokens used\s+([\d,]+)", completed.stdout)
            if token_match:
                event["tokens_used"] = int(token_match.group(1).replace(",", ""))
            self.calls.append(event)
            print(f"[model:done] {purpose} rc={completed.returncode}", flush=True)
            if completed.returncode != 0 or not output_path.exists():
                raise RuntimeError(
                    f"Codex call failed for {purpose}: rc={completed.returncode}; "
                    f"stderr_tail={completed.stderr[-600:]}"
                )
            return json.loads(output_path.read_text(encoding="utf-8"))


LABEL_SCHEMA = {
    "type": "object",
    "properties": {"label": {"type": "string", "enum": ["APPROVE", "REQUEST_CHANGES"]}},
    "required": ["label"],
    "additionalProperties": False,
}


def reviewer_prompt(policy: str, example: dict[str, Any]) -> str:
    return f"""You are a code-review classifier. Follow this frozen policy:
<policy>
{policy}
</policy>

Classify the pull request as APPROVE or REQUEST_CHANGES. Use only the supplied
text. Do not browse, search, call tools, or rely on repository access.

<pull_request>
{json.dumps(example, ensure_ascii=False)}
</pull_request>
Return the required JSON only."""


@dataclass(slots=True)
class WorkspaceEditor:
    client: CodexCli

    async def edit(self, parent, archive, budget):  # type: ignore[no-untyped-def]
        del budget
        feedback = parent.training_feedback[-3:]
        prompt = f"""Improve a two-role coding workspace by making one bounded prompt edit.
The coder solves Python Exercism tasks. The reviewer predicts whether a patch
should pass review. Training feedback may be used; validation and anchor labels
are unavailable. Do not browse, call tools, or name benchmark answers.

Current workspace:
{json.dumps(parent.workspace, ensure_ascii=False)}
Lineage training feedback:
{json.dumps(feedback, ensure_ascii=False)}
Archive prompt hashes:
{json.dumps([canonical_hash(node.workspace) for node in archive])}

Return complete replacement prompts as JSON."""
        schema = {
            "type": "object",
            "properties": {
                "coder_prompt": {"type": "string"},
                "reviewer_prompt": {"type": "string"},
            },
            "required": ["coder_prompt", "reviewer_prompt"],
            "additionalProperties": False,
        }
        try:
            return await self.client.json(prompt, schema, "workspace-edit")
        except (
            OSError,
            RuntimeError,
            subprocess.SubprocessError,
            ValueError,
            json.JSONDecodeError,
        ):
            return None


@dataclass(slots=True)
class DockerPythonRunner:
    timeout: int
    image: str = "python:3.12-slim"

    def source_file(self, task: Path) -> Path:
        candidates = sorted(
            path
            for path in task.glob("*.py")
            if not path.name.endswith("_test.py") and not path.name.startswith("test_")
        )
        if len(candidates) != 1:
            raise RuntimeError(f"expected one editable Python file in {task}, got {candidates}")
        return candidates[0]

    def run(self, task: Path, replacement: str) -> tuple[int, dict[str, Any]]:
        with tempfile.TemporaryDirectory(prefix="openrqgm-task-") as temp:
            copied = Path(temp) / task.name
            shutil.copytree(task, copied)
            target = copied / self.source_file(task).name
            target.write_text(replacement, encoding="utf-8")
            command = [
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                "--read-only",
                "--memory",
                "512m",
                "--cpus",
                "1",
                "--pids-limit",
                "128",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges",
                "--user",
                "65534:65534",
                "-e",
                "PYTHONDONTWRITEBYTECODE=1",
                "--tmpfs",
                "/tmp:rw,noexec,nosuid,size=64m",
                "-v",
                f"{copied.resolve()}:/work:ro",
                "-w",
                "/work",
                self.image,
                "python",
                "-m",
                "unittest",
                "discover",
                "-p",
                "*_test.py",
            ]
            try:
                completed = subprocess.run(
                    command,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    capture_output=True,
                    timeout=self.timeout,
                    check=False,
                )
                metadata = {
                    "returncode": completed.returncode,
                    "stdout_tail": completed.stdout[-500:],
                    "stderr_tail": completed.stderr[-500:],
                }
                return int(completed.returncode == 0), metadata
            except subprocess.TimeoutExpired:
                return 0, {"timeout": True}
            except OSError as error:
                return 0, {"sandbox_error": type(error).__name__}


@dataclass(slots=True)
class CodingTaskEvaluator:
    client: CodexCli
    runner: DockerPythonRunner
    validation_tasks: list[Path]
    crave_validation: list[dict[str, Any]]
    rng: random.Random
    artifacts: dict[str, dict[str, Any]] = field(default_factory=dict)
    review_index: int = 0

    async def _coder_artifact(self, node: WorkspaceNode) -> dict[str, Any]:
        if node.node_id in self.artifacts:
            return self.artifacts[node.node_id]
        task = self.validation_tasks[self.rng.randrange(len(self.validation_tasks))]
        source_path = self.runner.source_file(task)
        instructions = next((task / ".docs").glob("instruction*.md"), None)
        task_text = instructions.read_text(encoding="utf-8") if instructions else task.name
        starter = source_path.read_text(encoding="utf-8")
        prompt = f"""Act as a coding task solver under this frozen policy:
{node.workspace["coder_prompt"]}

Solve the task using only the supplied instructions and starter file. Do not
browse, search, call tools, or read external files. Return the complete Python
source file, not a diff.

<instructions>
{task_text}
</instructions>
<starter filename={source_path.name!r}>
{starter}
</starter>"""
        schema = {
            "type": "object",
            "properties": {"replacement": {"type": "string"}},
            "required": ["replacement"],
            "additionalProperties": False,
        }
        result = await self.client.json(prompt, schema, f"coder:{task.name}")
        artifact = {
            "task": task.name,
            "task_path": str(task),
            "source_file": source_path.name,
            "replacement": result["replacement"],
        }
        self.artifacts[node.node_id] = artifact
        return artifact

    async def evaluate(self, node, task, evaluator, cached_artifact, budget):  # type: ignore[no-untyped-def]
        del cached_artifact, budget
        if task.task_id == "coder-polyglot-tests":
            try:
                artifact = await self._coder_artifact(node)
            except (
                OSError,
                RuntimeError,
                subprocess.SubprocessError,
                ValueError,
                json.JSONDecodeError,
            ) as error:
                return EvaluationOutcome(0, metadata={"error": type(error).__name__})
            outcome, details = await asyncio.to_thread(
                self.runner.run, Path(artifact["task_path"]), artifact["replacement"]
            )
            return EvaluationOutcome(
                outcome,
                artifact_key=f"coder:{node.node_id}",
                artifact=artifact,
                metadata={"task": artifact["task"], "sandbox": details},
            )
        if task.task_id == "coder-learned-review":
            if evaluator is None:
                raise RuntimeError("learned reviewer task has no frozen evaluator")
            try:
                artifact = await self._coder_artifact(node)
            except (
                OSError,
                RuntimeError,
                subprocess.SubprocessError,
                ValueError,
                json.JSONDecodeError,
            ) as error:
                return EvaluationOutcome(0, metadata={"error": type(error).__name__})
            example = {
                "id": artifact["task"],
                "title": f"Implement {artifact['task']}",
                "patch": artifact["replacement"],
                "description": "Candidate solution for the supplied programming task.",
                "hint": "Approve only if the code appears correct, complete, and robust.",
            }
            try:
                result = await self.client.json(
                    reviewer_prompt(evaluator.artifact["prompt"], example),
                    LABEL_SCHEMA,
                    f"learned-review:{artifact['task']}",
                )
            except (
                OSError,
                RuntimeError,
                subprocess.SubprocessError,
                ValueError,
                json.JSONDecodeError,
            ) as error:
                return EvaluationOutcome(0, metadata={"error": type(error).__name__})
            return EvaluationOutcome(
                int(result["label"] == "APPROVE"),
                artifact_key=f"coder:{node.node_id}",
                artifact=artifact,
                metadata={"task": artifact["task"], "review_label": result["label"]},
            )
        if task.task_id == "reviewer-crave-validation":
            row = self.crave_validation[self.review_index % len(self.crave_validation)]
            example_id = f"validation-{self.review_index % len(self.crave_validation)}"
            self.review_index += 1
            try:
                result = await self.client.json(
                    reviewer_prompt(
                        node.workspace["reviewer_prompt"], review_example(example_id, row)
                    ),
                    LABEL_SCHEMA,
                    f"crave-validation:{example_id}",
                )
            except (
                OSError,
                RuntimeError,
                subprocess.SubprocessError,
                ValueError,
                json.JSONDecodeError,
            ) as error:
                return EvaluationOutcome(
                    0, metadata={"example_id": example_id, "error": type(error).__name__}
                )
            return EvaluationOutcome(
                int(result["label"] == row["label"]),
                metadata={"example_id": example_id, "prediction": result["label"]},
            )
        raise KeyError(task.task_id)


@dataclass(slots=True)
class ChallengerSource:
    async def challengers(
        self, slot: EvaluatorSlot, archive: Sequence[WorkspaceNode]
    ) -> Sequence[EvaluatorCandidate]:
        candidates: dict[str, EvaluatorCandidate] = {}
        for node in archive:
            prompt = node.workspace["reviewer_prompt"]
            digest = hashlib.sha256(prompt.encode()).hexdigest()[:16]
            candidate_id = f"reviewer-{digest}"
            candidates[candidate_id] = EvaluatorCandidate.create(
                slot.slot_id,
                {"prompt": prompt},
                source=f"workspace:{node.node_id}",
                parent_id=slot.incumbent.candidate_id,
                candidate_id=candidate_id,
            )
        return list(candidates.values())


@dataclass(slots=True)
class PrivateCraveAnchors:
    rows: list[dict[str, Any]]
    provider_id: str = "TuringEnterprises/CRAVE:test-withheld"
    version: str = "public-dataset-reconstruction-v1"

    @property
    def fingerprint(self) -> str:
        return canonical_hash(self.rows)

    async def examples(self, slot_id: str) -> Sequence[AnchorExample]:
        if slot_id != "code-reviewer":
            raise KeyError(slot_id)
        return [
            AnchorExample(f"anchor-{index}", review_example(f"anchor-{index}", row), row["label"])
            for index, row in enumerate(self.rows)
        ]


@dataclass(slots=True)
class BatchedAnchorEvaluator:
    client: CodexCli
    provider: PrivateCraveAnchors
    predictions: dict[str, dict[str, str]] = field(default_factory=dict)

    async def _predict(self, candidate: EvaluatorCandidate) -> dict[str, str]:
        examples = await self.provider.examples(candidate.slot_id)
        prompt = f"""You are a code-review classifier. Follow this frozen policy:
<policy>
{candidate.artifact["prompt"]}
</policy>

Classify every supplied pull request as APPROVE or REQUEST_CHANGES. Use only
the supplied text. Do not browse, search, call tools, or use repository access.
Do not omit or reorder ids.

<examples>
{json.dumps([example.artifact for example in examples], ensure_ascii=False)}
</examples>
Return the required JSON only."""
        schema = {
            "type": "object",
            "properties": {
                "predictions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "label": {
                                "type": "string",
                                "enum": ["APPROVE", "REQUEST_CHANGES"],
                            },
                        },
                        "required": ["id", "label"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["predictions"],
            "additionalProperties": False,
        }
        try:
            result = await self.client.json(prompt, schema, f"anchor:{candidate.candidate_id}")
            return {item["id"]: item["label"] for item in result["predictions"]}
        except (
            OSError,
            RuntimeError,
            subprocess.SubprocessError,
            ValueError,
            json.JSONDecodeError,
        ):
            return {}

    async def evaluate(self, candidate: EvaluatorCandidate, example: AnchorExample) -> int:
        if candidate.candidate_id not in self.predictions:
            self.predictions[candidate.candidate_id] = await self._predict(candidate)
        prediction = self.predictions[candidate.candidate_id].get(example.example_id)
        return int(prediction == example.expected)


@dataclass(slots=True)
class TrainingFeedback:
    client: CodexCli
    rows: list[dict[str, Any]]
    samples_per_node: int
    index: int = 0

    async def collect(self, node, tasks, evaluators, budget):  # type: ignore[no-untyped-def]
        del tasks, evaluators, budget
        feedback = []
        for _ in range(self.samples_per_node):
            row = self.rows[self.index % len(self.rows)]
            example_id = f"train-{self.index % len(self.rows)}"
            self.index += 1
            try:
                result = await self.client.json(
                    reviewer_prompt(
                        node.workspace["reviewer_prompt"], review_example(example_id, row)
                    ),
                    LABEL_SCHEMA,
                    f"crave-training:{example_id}",
                )
            except (
                OSError,
                RuntimeError,
                subprocess.SubprocessError,
                ValueError,
                json.JSONDecodeError,
            ) as error:
                feedback.append(
                    {
                        "task": "reviewer-crave-training",
                        "example_id": example_id,
                        "error": type(error).__name__,
                        "enters_validation_utility": False,
                    }
                )
                continue
            feedback.append(
                {
                    "task": "reviewer-crave-training",
                    "example_id": example_id,
                    "prediction": result["label"],
                    "expected": row["label"],
                    "correct": result["label"] == row["label"],
                }
            )
        return {"samples": feedback, "enters_validation_utility": False}


def build_engine(config: dict[str, Any], client: CodexCli) -> tuple[RQGM, CodingTaskEvaluator]:
    seed = int(config["random_seed"])
    tasks = python_tasks(seed)
    validation_tasks = tasks[: int(config["polyglot_validation_tasks"])]
    crave_train = sample_rows(load_rows("train"), 32, seed + 1)
    crave_validation = sample_rows(
        load_rows("validation"), int(config["crave_validation_examples"]), seed + 2
    )
    crave_anchors = sample_rows(load_rows("test"), int(config["crave_anchor_examples"]), seed + 3)
    provider = PrivateCraveAnchors(crave_anchors)
    task_evaluator = CodingTaskEvaluator(
        client,
        DockerPythonRunner(int(config["container_timeout_seconds"])),
        validation_tasks,
        crave_validation,
        random.Random(seed + 4),
    )
    initial_reviewer = (
        "Approve only when the patch is correct, complete, scoped to the request, and has no "
        "clear regression, unsafe behavior, or missing edge case. Otherwise request changes."
    )
    incumbent = EvaluatorCandidate.create(
        "code-reviewer",
        {"prompt": initial_reviewer},
        source="seed",
        candidate_id="reviewer-seed",
    )
    engine = RQGM(
        seed_workspace={
            "coder_prompt": (
                "Produce a minimal correct implementation. Preserve the required API, handle edge "
                "cases explicitly, and return syntactically valid Python without test-specific "
                "hacks."
            ),
            "reviewer_prompt": initial_reviewer,
        },
        tasks=[
            RoleTask("coder", "coder-polyglot-tests", "fixed"),
            RoleTask("coder", "coder-learned-review", "learned", "code-reviewer"),
            RoleTask("reviewer", "reviewer-crave-validation", "fixed"),
        ],
        slots=[EvaluatorSlot("code-reviewer", "reviewer", incumbent)],
        runtime=Runtime(
            editor=WorkspaceEditor(client),
            task_evaluator=task_evaluator,
            challenger_source=ChallengerSource(),
            anchor_provider=provider,
            anchor_evaluator=BatchedAnchorEvaluator(client, provider),
            training_feedback=TrainingFeedback(
                client, crave_train, int(config["training_samples_per_node"])
            ),
        ),
        config=RQGMConfig(
            validation_budget=int(config["validation_budget"]),
            checkpoints=tuple(config["checkpoints"]),
            epsilon=float(config["epsilon"]),
            expansion_alpha=float(config["expansion_alpha"]),
            minimum_anchor_outcomes=int(config["minimum_anchor_outcomes"]),
            random_seed=seed,
        ),
    )
    return engine, task_evaluator


async def execute(config_path: Path) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = ROOT / config["output"]
    output.mkdir(parents=True, exist_ok=True)
    client = CodexCli(config["model"], int(config["model_timeout_seconds"]), DATA / "codex-empty")
    engine, _ = build_engine(config, client)
    started = time.monotonic()
    result = await engine.run()
    summary = {
        "claim": config["claim"],
        "result": asdict(result),
        "anchor_provider": {
            "provider_id": engine.runtime.anchor_provider.provider_id,
            "version": engine.runtime.anchor_provider.version,
            "fingerprint": engine.runtime.anchor_provider.fingerprint,
        },
        "model_calls": client.calls,
        "paper_comparison_valid": False,
        "paper_reported_rqgm_endpoint": "119/166",
        "wall_time_seconds": round(time.monotonic() - started, 3),
        "limitations": [
            "Python-only public subset",
            "paper split and production prompts are unpublished",
            "gpt-5.6-sol differs from the paper's GPT-5 low endpoint",
            "anchor inference is batched as a declared cost-saving approximation",
            "training feedback currently grounds the reviewer role only",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    save_state(engine, output / "state.json")
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(execute(args.config.resolve()))


if __name__ == "__main__":
    main()
