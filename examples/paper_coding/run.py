from __future__ import annotations

import argparse
import asyncio
import difflib
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
        prompt = f"""Improve a two-role coding workspace by making one bounded strategy edit.
The coder solves Python Exercism tasks. The reviewer predicts whether a patch
should pass review. Training feedback may be used; validation and anchor labels
are unavailable. Coder training feedback contains sandbox test outcomes and may
be used to improve the generation or repair policy. Do not browse, call tools,
or name benchmark answers.

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
                "repair_prompt": {"type": "string"},
            },
            "required": ["coder_prompt", "reviewer_prompt", "repair_prompt"],
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
                    "stdout_tail": completed.stdout[-4000:],
                    "stderr_tail": completed.stderr[-8000:],
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
    random_seed: int
    repair_attempts: int = 1

    def _task_material(self, task: Path) -> dict[str, str]:
        source_path = self.runner.source_file(task)
        instructions = next((task / ".docs").glob("instruction*.md"), None)
        task_text = instructions.read_text(encoding="utf-8") if instructions else task.name
        starter = source_path.read_text(encoding="utf-8")
        test_files = sorted(
            path
            for path in task.glob("*.py")
            if path.name.endswith("_test.py") or path.name.startswith("test_")
        )
        tests = "\n\n".join(
            f"# file: {path.name}\n{path.read_text(encoding='utf-8')}" for path in test_files
        )
        return {
            "task": task.name,
            "task_path": str(task),
            "source_file": source_path.name,
            "instructions": task_text,
            "starter": starter,
            "tests": tests,
        }

    def _ordered_tasks(self, node_id: str, tasks: Sequence[Path], phase: str) -> list[Path]:
        return sorted(
            tasks,
            key=lambda task: hashlib.sha256(
                f"{self.random_seed}:{phase}:{node_id}:{task.name}".encode()
            ).digest(),
        )

    async def _generate_artifact(
        self,
        node: WorkspaceNode,
        task: Path,
        *,
        phase: str,
        purpose_prefix: str,
    ) -> dict[str, Any]:
        material = self._task_material(task)
        prompt = f"""Act as a coding task solver under this frozen policy:
{node.workspace["coder_prompt"]}

Solve the task using only the supplied instructions and starter file. Return
the complete Python source file, not a diff. A sandboxed test runner will return
concrete feedback and you may receive a bounded repair turn. Do not browse or
search the web.

<instructions>
{material["instructions"]}
</instructions>
<starter filename={material["source_file"]!r}>
{material["starter"]}
</starter>
<repository_tests>
{material["tests"]}
</repository_tests>"""
        schema = {
            "type": "object",
            "properties": {"replacement": {"type": "string"}},
            "required": ["replacement"],
            "additionalProperties": False,
        }
        result = await self.client.json(prompt, schema, f"{purpose_prefix}:{task.name}:attempt-0")
        replacement = result["replacement"]
        outcome, sandbox = await asyncio.to_thread(self.runner.run, task, replacement)
        attempts = 1
        for repair_index in range(self.repair_attempts):
            if outcome:
                break
            repair_policy = node.workspace.get(
                "repair_prompt",
                "Use the concrete test failure to repair the implementation without changing "
                "the API.",
            )
            repair_prompt = f"""Repair a Python solution under this frozen policy:
{repair_policy}

Use only the task, starter, current implementation, and sandbox output below.
Do not browse or search the web. Return the complete corrected source file.

<instructions>
{material["instructions"]}
</instructions>
<starter filename={material["source_file"]!r}>
{material["starter"]}
</starter>
<repository_tests>
{material["tests"]}
</repository_tests>
<current>
{replacement}
</current>
<sandbox_result>
{json.dumps(sandbox, ensure_ascii=False)}
</sandbox_result>"""
            repaired = await self.client.json(
                repair_prompt,
                schema,
                f"{purpose_prefix}:{task.name}:repair-{repair_index + 1}",
            )
            replacement = repaired["replacement"]
            outcome, sandbox = await asyncio.to_thread(self.runner.run, task, replacement)
            attempts += 1
        patch = "".join(
            difflib.unified_diff(
                material["starter"].splitlines(keepends=True),
                replacement.splitlines(keepends=True),
                fromfile=f"a/{material['source_file']}",
                tofile=f"b/{material['source_file']}",
            )
        )
        artifact = {
            "kind": "polyglot-solution",
            "phase": phase,
            **material,
            "replacement": replacement,
            "patch": patch,
            "test_outcome": outcome,
            "sandbox": sandbox,
            "attempts": attempts,
            "fixed_recorded": False,
            "reviews": {},
        }
        return artifact

    def _cached_code_artifacts(self, node: WorkspaceNode) -> list[tuple[str, dict[str, Any]]]:
        return sorted(
            (
                (key, value)
                for key, value in node.cached_artifacts.items()
                if isinstance(value, dict)
                and value.get("kind") == "polyglot-solution"
                and value.get("phase") == "validation"
            ),
            key=lambda pair: pair[0],
        )

    async def _artifact_for_dimension(
        self,
        node: WorkspaceNode,
        dimension: str,
        evaluator_id: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        artifacts = self._cached_code_artifacts(node)
        for key, artifact in artifacts:
            if dimension == "fixed" and not artifact.get("fixed_recorded", False):
                return key, artifact
            if dimension == "learned" and evaluator_id not in artifact.get("reviews", {}):
                return key, artifact

        used = {artifact["task"] for _, artifact in artifacts}
        task = next(
            (
                candidate
                for candidate in self._ordered_tasks(
                    node.node_id, self.validation_tasks, "validation"
                )
                if candidate.name not in used
            ),
            None,
        )
        if task is None:
            raise RuntimeError(f"validation task pool exhausted for node {node.node_id}")
        artifact = await self._generate_artifact(
            node,
            task,
            phase="validation",
            purpose_prefix="coder-validation",
        )
        return f"polyglot:validation:{task.name}", artifact

    def _next_crave_example(self, node: WorkspaceNode) -> tuple[str, dict[str, Any]]:
        used = {
            value["example_id"]
            for value in node.cached_artifacts.values()
            if isinstance(value, dict) and value.get("kind") == "crave-validation"
        }
        ordered = sorted(
            enumerate(self.crave_validation),
            key=lambda pair: hashlib.sha256(
                f"{self.random_seed}:crave:{node.node_id}:{pair[0]}".encode()
            ).digest(),
        )
        selected = next(
            ((index, row) for index, row in ordered if f"validation-{index}" not in used),
            None,
        )
        if selected is None:
            raise RuntimeError(f"CRAVE validation pool exhausted for node {node.node_id}")
        index, row = selected
        return f"validation-{index}", row

    async def training_samples(
        self,
        node: WorkspaceNode,
        tasks: Sequence[Path],
        count: int,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        ordered = self._ordered_tasks(node.node_id, tasks, "training")
        for task in ordered[:count]:
            try:
                artifact = await self._generate_artifact(
                    node,
                    task,
                    phase="training",
                    purpose_prefix="coder-training",
                )
                results.append(
                    {
                        "task": task.name,
                        "outcome": artifact["test_outcome"],
                        "attempts": artifact["attempts"],
                        "sandbox": artifact["sandbox"],
                    }
                )
            except (
                OSError,
                RuntimeError,
                subprocess.SubprocessError,
                ValueError,
                json.JSONDecodeError,
            ) as error:
                results.append({"task": task.name, "outcome": 0, "error": type(error).__name__})
        return results

    async def heldout_results(
        self,
        node: WorkspaceNode,
        tasks: Sequence[Path],
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for task in tasks:
            try:
                artifact = await self._generate_artifact(
                    node,
                    task,
                    phase="heldout",
                    purpose_prefix="coder-heldout",
                )
                results.append(
                    {
                        "task": task.name,
                        "outcome": artifact["test_outcome"],
                        "attempts": artifact["attempts"],
                        "sandbox": artifact["sandbox"],
                    }
                )
            except (
                OSError,
                RuntimeError,
                subprocess.SubprocessError,
                ValueError,
                json.JSONDecodeError,
            ) as error:
                results.append({"task": task.name, "outcome": 0, "error": type(error).__name__})
        return results

    async def evaluate(self, node, task, evaluator, cached_artifact, budget):  # type: ignore[no-untyped-def]
        del cached_artifact, budget
        if task.task_id == "coder-polyglot-tests":
            try:
                artifact_key, artifact = await self._artifact_for_dimension(node, "fixed")
            except (
                OSError,
                RuntimeError,
                subprocess.SubprocessError,
                ValueError,
                json.JSONDecodeError,
            ) as error:
                return EvaluationOutcome(0, metadata={"error": type(error).__name__})
            artifact["fixed_recorded"] = True
            return EvaluationOutcome(
                artifact["test_outcome"],
                artifact_key=artifact_key,
                artifact=artifact,
                metadata={
                    "sample_id": artifact["task"],
                    "task": artifact["task"],
                    "sandbox": artifact["sandbox"],
                    "attempts": artifact["attempts"],
                },
            )
        if task.task_id == "coder-learned-review":
            if evaluator is None:
                raise RuntimeError("learned reviewer task has no frozen evaluator")
            try:
                artifact_key, artifact = await self._artifact_for_dimension(
                    node, "learned", evaluator.candidate_id
                )
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
                "instructions": artifact["instructions"],
                "starter": artifact["starter"],
                "tests": artifact["tests"],
                "patch": artifact["patch"],
                "description": "Judge this patch against the complete task specification.",
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
            artifact.setdefault("reviews", {})[evaluator.candidate_id] = result["label"]
            return EvaluationOutcome(
                int(result["label"] == "APPROVE"),
                artifact_key=artifact_key,
                artifact=artifact,
                metadata={
                    "sample_id": artifact["task"],
                    "task": artifact["task"],
                    "review_label": result["label"],
                },
            )
        if task.task_id == "reviewer-crave-validation":
            try:
                example_id, row = self._next_crave_example(node)
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
                artifact_key=f"crave:{example_id}",
                artifact={
                    "kind": "crave-validation",
                    "example_id": example_id,
                    "prediction": result["label"],
                },
                metadata={"sample_id": example_id, "prediction": result["label"]},
            )
        raise KeyError(task.task_id)


@dataclass(slots=True)
class ChallengerSource:
    async def challengers(
        self, slot: EvaluatorSlot, archive: Sequence[WorkspaceNode]
    ) -> Sequence[EvaluatorCandidate]:
        candidates: dict[str, EvaluatorCandidate] = {}
        incumbent_prompt = slot.incumbent.artifact["prompt"]
        for node in archive:
            prompt = node.workspace["reviewer_prompt"]
            if prompt == incumbent_prompt:
                continue
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
    reviewer_samples_per_node: int
    coding_evaluator: CodingTaskEvaluator
    coder_tasks: list[Path]
    coder_samples_per_node: int
    index: int = 0

    async def collect(self, node, tasks, evaluators, budget):  # type: ignore[no-untyped-def]
        del tasks, evaluators, budget
        selected: list[tuple[str, dict[str, Any]]] = []
        for _ in range(self.reviewer_samples_per_node):
            row = self.rows[self.index % len(self.rows)]
            example_id = f"train-{self.index % len(self.rows)}"
            self.index += 1
            selected.append((example_id, row))
        prompt = f"""You are a code-review classifier. Follow this frozen policy:
<policy>
{node.workspace["reviewer_prompt"]}
</policy>

Classify every supplied pull request as APPROVE or REQUEST_CHANGES. Use only
the supplied text. Do not browse, search, call tools, or use repository access.
Do not omit or reorder ids.

<examples>
{json.dumps([review_example(example_id, row) for example_id, row in selected], ensure_ascii=False)}
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
        reviewer_feedback: list[dict[str, Any]] = []
        if selected:
            try:
                result = await self.client.json(
                    prompt,
                    schema,
                    f"crave-training:{selected[0][0]}..{selected[-1][0]}",
                )
                predictions = {item["id"]: item["label"] for item in result["predictions"]}
                for example_id, row in selected:
                    prediction = predictions.get(example_id)
                    reviewer_feedback.append(
                        {
                            "task": "reviewer-crave-training",
                            "example_id": example_id,
                            "prediction": prediction,
                            "expected": row["label"],
                            "correct": prediction == row["label"],
                        }
                    )
            except (
                OSError,
                RuntimeError,
                subprocess.SubprocessError,
                ValueError,
                json.JSONDecodeError,
            ) as error:
                reviewer_feedback = [
                    {
                        "task": "reviewer-crave-training",
                        "example_id": example_id,
                        "error": type(error).__name__,
                    }
                    for example_id, _ in selected
                ]
        coder_feedback = await self.coding_evaluator.training_samples(
            node,
            self.coder_tasks,
            self.coder_samples_per_node,
        )
        return {
            "reviewer_samples": reviewer_feedback,
            "coder_samples": coder_feedback,
            "enters_validation_utility": False,
        }


def build_engine(
    config: dict[str, Any], client: CodexCli
) -> tuple[RQGM, CodingTaskEvaluator, list[Path]]:
    seed = int(config["random_seed"])
    tasks = python_tasks(seed)
    train_count = int(config["polyglot_train_tasks"])
    validation_count = int(config["polyglot_validation_tasks"])
    test_count = int(config["polyglot_test_tasks"])
    if train_count + validation_count + test_count > len(tasks):
        raise ValueError("requested Polyglot train/validation/test splits overlap")
    coder_train_tasks = tasks[:train_count]
    validation_tasks = tasks[train_count : train_count + validation_count]
    heldout_tasks = tasks[
        train_count + validation_count : train_count + validation_count + test_count
    ]
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
        seed + 4,
        int(config["coder_repair_attempts"]),
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
            "repair_prompt": (
                "Use the concrete sandbox failure to correct the algorithm while preserving the "
                "required public API. Make the smallest complete fix and avoid test-specific hacks."
            ),
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
                client,
                crave_train,
                int(config["training_samples_per_node"]),
                task_evaluator,
                coder_train_tasks,
                int(config["coder_training_samples_per_node"]),
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
    return engine, task_evaluator, heldout_tasks


async def execute(config_path: Path) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = ROOT / config["output"]
    output.mkdir(parents=True, exist_ok=True)
    client = CodexCli(config["model"], int(config["model_timeout_seconds"]), DATA / "codex-empty")
    engine, task_evaluator, heldout_tasks = build_engine(config, client)
    started = time.monotonic()
    result = await engine.run()
    endpoint_ids = {
        "generalist": result.endpoint.node_id,
        "coder_specialist": result.specialists["coder"].node_id,
    }
    heldout: dict[str, Any] = {}
    heldout_by_node: dict[str, list[dict[str, Any]]] = {}
    for endpoint_name, node_id in endpoint_ids.items():
        node = engine.archive.nodes[node_id]
        if node_id not in heldout_by_node:
            heldout_by_node[node_id] = await task_evaluator.heldout_results(node, heldout_tasks)
        outcomes = heldout_by_node[node_id]
        heldout[endpoint_name] = {
            "node_id": node_id,
            "passes": sum(item["outcome"] for item in outcomes),
            "total": len(outcomes),
            "outcomes": outcomes,
        }
    summary = {
        "claim": config["claim"],
        "result": asdict(result),
        "anchor_provider": {
            "provider_id": engine.runtime.anchor_provider.provider_id,
            "version": engine.runtime.anchor_provider.version,
            "fingerprint": engine.runtime.anchor_provider.fingerprint,
        },
        "model_calls": client.calls,
        "heldout": heldout,
        "paper_comparison_valid": False,
        "paper_reported_rqgm_endpoint": "119/166",
        "wall_time_seconds": round(time.monotonic() - started, 3),
        "limitations": [
            "Python-only public subset",
            "paper split and production prompts are unpublished",
            "gpt-5.6-sol differs from the paper's GPT-5 low endpoint",
            "anchor inference is batched as a declared cost-saving approximation",
            "structured prompt/repair strategy evolution is narrower than arbitrary codebase edits",
            "only the public Python subset is implemented",
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
