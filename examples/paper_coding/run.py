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

from agent_workspace import (
    AgentWorkspaceRunner,
    seed_workspace,
    validate_workspace,
    workspace_from_files,
)
from polyglot import (
    DEFAULT_IMAGES,
    LANGUAGES,
    DockerPolyglotRunner,
    PolyglotTask,
    replacements_from_json,
    split_balanced,
)
from polyglot import (
    material as task_material,
)

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


def source_identity() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}
    return {"commit": revision, "dirty": dirty}


def token_usage_from_jsonl(output: str) -> dict[str, int]:
    """Extract the final Codex turn usage while tolerating non-JSON warnings."""
    usage: dict[str, int] = {}
    for line in output.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") != "turn.completed" or not isinstance(event.get("usage"), dict):
            continue
        usage = {key: int(value) for key, value in event["usage"].items() if isinstance(value, int)}
    if usage:
        usage["raw_total_tokens"] = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
        usage["blended_tokens"] = usage.get("input_tokens", 0) + 5 * usage.get("output_tokens", 0)
    return usage


def load_rows(split: str) -> list[dict[str, Any]]:
    path = DATA / f"crave-{split}.json"
    if not path.exists():
        raise FileNotFoundError(f"run prepare_data.py first; missing {path}")
    return json.loads(path.read_text(encoding="utf-8"))


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
                "--json",
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
            usage = token_usage_from_jsonl(completed.stdout)
            if usage:
                event.update(usage)
            else:
                token_match = re.search(
                    r"tokens used\s+([\d,]+)", completed.stdout + "\n" + completed.stderr
                )
                if token_match:
                    event["raw_total_tokens"] = int(token_match.group(1).replace(",", ""))
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

EXPERIMENT_CONDITIONS = {
    "verifier_only",
    "fixed_reviewer",
    "coevolving_reviewer",
}


@dataclass(slots=True)
class WorkspaceEditor:
    client: CodexCli
    agent_runner: AgentWorkspaceRunner

    async def edit(self, parent, archive, budget):  # type: ignore[no-untyped-def]
        del budget
        feedback = parent.training_feedback[-3:]
        prompt = f"""You are the meta-agent for a two-role polyglot coding system. Improve the
entire agent codebase by one bounded change. You may add, remove, or replace any
text file inside this codebase, including agent.py, helper modules, prompts,
parsing, and role logic. The fixed JSON protocol is stdin
{{"operation":"coder|repair|reviewer","context":{{...}}}} and stdout
{{"prompt":"..."}}. Code executes without network, credentials, repository,
validation labels, or private anchors. Never attempt to change the RQGM engine,
sandbox policy, benchmark tests, or host. Training feedback is the only outcome
evidence available. The six task languages are cpp, go, java, javascript,
python, and rust. Do not browse, call tools, or include benchmark answers.

Current agent codebase:
{json.dumps(parent.workspace["files"], ensure_ascii=False)}
Lineage training feedback:
{json.dumps(feedback, ensure_ascii=False)}
Archive codebase hashes:
{json.dumps([canonical_hash(node.workspace) for node in archive])}

Return the complete next codebase as a list of UTF-8 text files. Include every
file that should exist; omitted files are deleted."""
        schema = {
            "type": "object",
            "properties": {
                "files": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                        "required": ["path", "content"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["files"],
            "additionalProperties": False,
        }
        try:
            result = await self.client.json(prompt, schema, "workspace-edit")
            workspace = workspace_from_files(result["files"])
            if canonical_hash(workspace) == canonical_hash(parent.workspace):
                return None
            for operation in ("coder", "repair", "reviewer"):
                await asyncio.to_thread(
                    self.agent_runner.prompt, workspace, operation, {"protocol_smoke": True}
                )
            return workspace
        except (
            OSError,
            RuntimeError,
            subprocess.SubprocessError,
            ValueError,
            json.JSONDecodeError,
        ):
            return None


@dataclass(slots=True)
class CodingTaskEvaluator:
    client: CodexCli
    runner: DockerPolyglotRunner
    agent_runner: AgentWorkspaceRunner
    validation_tasks: list[PolyglotTask]
    crave_validation: list[dict[str, Any]]
    random_seed: int
    repair_attempts: int = 1
    fixed_evaluator: EvaluatorCandidate | None = None

    def _task_material(self, task: PolyglotTask) -> dict[str, Any]:
        return task_material(task)

    def _ordered_tasks(
        self, node_id: str, tasks: Sequence[PolyglotTask], phase: str
    ) -> list[PolyglotTask]:
        by_language: dict[str, list[PolyglotTask]] = {}
        for task in tasks:
            by_language.setdefault(task.language, []).append(task)
        for language, language_tasks in by_language.items():
            by_language[language] = sorted(
                language_tasks,
                key=lambda task: hashlib.sha256(
                    f"{self.random_seed}:{phase}:{node_id}:{task.task_id}".encode()
                ).digest(),
            )
        languages = sorted(
            by_language,
            key=lambda language: hashlib.sha256(
                f"{self.random_seed}:{phase}:{node_id}:language:{language}".encode()
            ).digest(),
        )
        ordered: list[PolyglotTask] = []
        for index in range(max((len(items) for items in by_language.values()), default=0)):
            ordered.extend(
                by_language[language][index]
                for language in languages
                if index < len(by_language[language])
            )
        return ordered

    async def _generate_artifact(
        self,
        node: WorkspaceNode,
        task: PolyglotTask,
        *,
        phase: str,
        purpose_prefix: str,
    ) -> dict[str, Any]:
        material = self._task_material(task)
        prompt = await asyncio.to_thread(
            self.agent_runner.prompt,
            node.workspace,
            "coder",
            {
                "language": material["language"],
                "instructions": material["instructions"],
                "editable_files": material["starters"],
                "repository_tests": material["tests"],
                "response_contract": "Return complete contents for editable files only.",
            },
        )
        schema = {
            "type": "object",
            "properties": {
                "files": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                        "required": ["path", "content"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["files"],
            "additionalProperties": False,
        }
        result = await self.client.json(
            prompt, schema, f"{purpose_prefix}:{task.task_id}:attempt-0"
        )
        replacements = replacements_from_json(result["files"])
        outcome, sandbox = await asyncio.to_thread(self.runner.run, task, replacements)
        attempts = 1
        for repair_index in range(self.repair_attempts):
            if outcome:
                break
            repair_prompt = await asyncio.to_thread(
                self.agent_runner.prompt,
                node.workspace,
                "repair",
                {
                    "language": material["language"],
                    "instructions": material["instructions"],
                    "starter_files": material["starters"],
                    "repository_tests": material["tests"],
                    "current_files": replacements,
                    "sandbox_result": sandbox,
                },
            )
            repaired = await self.client.json(
                repair_prompt,
                schema,
                f"{purpose_prefix}:{task.task_id}:repair-{repair_index + 1}",
            )
            replacements = replacements_from_json(repaired["files"])
            outcome, sandbox = await asyncio.to_thread(self.runner.run, task, replacements)
            attempts += 1
        import difflib

        patch = "".join(
            line
            for path, replacement in replacements.items()
            for line in difflib.unified_diff(
                material["starters"].get(path, "").splitlines(keepends=True),
                replacement.splitlines(keepends=True),
                fromfile=f"a/{path}",
                tofile=f"b/{path}",
            )
        )
        artifact = {
            "kind": "polyglot-solution",
            "phase": phase,
            **material,
            "replacements": replacements,
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
                if candidate.task_id not in used
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
        return f"polyglot:validation:{task.task_id}", artifact

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
        tasks: Sequence[PolyglotTask],
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
                        "task": task.task_id,
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
                results.append({"task": task.task_id, "outcome": 0, "error": type(error).__name__})
        return results

    async def heldout_results(
        self,
        node: WorkspaceNode,
        tasks: Sequence[PolyglotTask],
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
                        "task": task.task_id,
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
                results.append({"task": task.task_id, "outcome": 0, "error": type(error).__name__})
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
            evaluator = evaluator or self.fixed_evaluator
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
                "starter": artifact["starters"],
                "tests": artifact["tests"],
                "patch": artifact["patch"],
                "description": "Judge this patch against the complete task specification.",
            }
            try:
                prompt = await asyncio.to_thread(
                    self.agent_runner.prompt,
                    evaluator.artifact["workspace"],
                    "reviewer",
                    {"examples": [example], "response_contract": "Return one label for the id."},
                )
                result = await self.client.json(
                    prompt,
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
            example_id = "unselected"
            try:
                example_id, row = self._next_crave_example(node)
                prompt = await asyncio.to_thread(
                    self.agent_runner.prompt,
                    node.workspace,
                    "reviewer",
                    {
                        "examples": [review_example(example_id, row)],
                        "response_contract": "Return one label for the id.",
                    },
                )
                result = await self.client.json(
                    prompt,
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
        incumbent_hash = canonical_hash(slot.incumbent.artifact["workspace"])
        for node in archive:
            digest_full = canonical_hash(node.workspace)
            if digest_full == incumbent_hash:
                continue
            digest = digest_full.removeprefix("sha256:")[:16]
            candidate_id = f"reviewer-{digest}"
            candidates[candidate_id] = EvaluatorCandidate.create(
                slot.slot_id,
                {"workspace": node.workspace},
                source=f"workspace:{node.node_id}",
                parent_id=slot.incumbent.candidate_id,
                candidate_id=candidate_id,
            )
        return list(candidates.values())


@dataclass(slots=True)
class NoChallengers:
    """Explicit fixed-evaluator baseline: no candidate can enter an election."""

    async def challengers(
        self, slot: EvaluatorSlot, archive: Sequence[WorkspaceNode]
    ) -> Sequence[EvaluatorCandidate]:
        del slot, archive
        return ()


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
    agent_runner: AgentWorkspaceRunner
    batch_size: int = 20
    predictions: dict[str, dict[str, str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("anchor batch size must be positive")

    async def _predict(self, candidate: EvaluatorCandidate) -> dict[str, str]:
        examples = await self.provider.examples(candidate.slot_id)
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
        predictions: dict[str, str] = {}
        for start in range(0, len(examples), self.batch_size):
            batch = examples[start : start + self.batch_size]
            try:
                prompt = await asyncio.to_thread(
                    self.agent_runner.prompt,
                    candidate.artifact["workspace"],
                    "reviewer",
                    {
                        "examples": [example.artifact for example in batch],
                        "response_contract": "Return a prediction for every id without reordering.",
                    },
                )
                result = await self.client.json(
                    prompt,
                    schema,
                    f"anchor:{candidate.candidate_id}:batch-{start // self.batch_size}",
                )
                predictions.update({item["id"]: item["label"] for item in result["predictions"]})
            except (
                OSError,
                RuntimeError,
                subprocess.SubprocessError,
                ValueError,
                json.JSONDecodeError,
            ):
                continue
        return predictions

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
    coder_tasks: list[PolyglotTask]
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
                prompt = await asyncio.to_thread(
                    self.coding_evaluator.agent_runner.prompt,
                    node.workspace,
                    "reviewer",
                    {
                        "examples": [
                            review_example(example_id, row) for example_id, row in selected
                        ],
                        "response_contract": "Return a prediction for every id without reordering.",
                    },
                )
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


def condition_components(
    condition: str, incumbent: EvaluatorCandidate
) -> tuple[
    list[RoleTask],
    list[EvaluatorSlot],
    EvaluatorCandidate | None,
    ChallengerSource | NoChallengers,
    bool,
]:
    if condition not in EXPERIMENT_CONDITIONS:
        raise ValueError(f"unknown experiment condition: {condition}")
    if condition == "verifier_only":
        return (
            [RoleTask("coder", "coder-polyglot-tests", "fixed")],
            [],
            None,
            NoChallengers(),
            False,
        )
    shared_tasks = [
        RoleTask("coder", "coder-polyglot-tests", "fixed"),
        RoleTask(
            "coder",
            "coder-learned-review",
            "learned" if condition == "coevolving_reviewer" else "fixed",
            "code-reviewer" if condition == "coevolving_reviewer" else None,
        ),
        RoleTask("reviewer", "reviewer-crave-validation", "fixed"),
    ]
    if condition == "fixed_reviewer":
        return shared_tasks, [], incumbent, NoChallengers(), True
    return (
        shared_tasks,
        [EvaluatorSlot("code-reviewer", "reviewer", incumbent)],
        None,
        ChallengerSource(),
        True,
    )


def build_engine(
    config: dict[str, Any], client: CodexCli
) -> tuple[RQGM, CodingTaskEvaluator, list[PolyglotTask]]:
    condition = config.get("experiment_condition", "coevolving_reviewer")
    seed = int(config["random_seed"])
    coder_train_tasks, validation_tasks, heldout_tasks = split_balanced(
        POLYGLOT,
        seed,
        int(config["polyglot_train_tasks_per_language"]),
        int(config["polyglot_validation_tasks_per_language"]),
        int(config["polyglot_test_tasks_per_language"]),
    )
    crave_train = sample_rows(load_rows("train"), 32, seed + 1)
    crave_validation = sample_rows(
        load_rows("validation"), int(config["crave_validation_examples"]), seed + 2
    )
    crave_anchors = sample_rows(load_rows("test"), int(config["crave_anchor_examples"]), seed + 3)
    provider = PrivateCraveAnchors(crave_anchors)
    images = config.get("polyglot_images", DEFAULT_IMAGES)
    agent_runner = AgentWorkspaceRunner(
        int(config["agent_timeout_seconds"]), config.get("agent_image", "python:3.12-slim")
    )
    initial_workspace = seed_workspace()
    validate_workspace(initial_workspace)
    incumbent = EvaluatorCandidate.create(
        "code-reviewer",
        {"workspace": initial_workspace},
        source="seed",
        candidate_id="reviewer-seed",
    )
    tasks, slots, fixed_evaluator, challenger_source, reviewer_training = condition_components(
        condition, incumbent
    )
    task_evaluator = CodingTaskEvaluator(
        client,
        DockerPolyglotRunner(int(config["container_timeout_seconds"]), images),
        agent_runner,
        validation_tasks,
        crave_validation,
        seed + 4,
        int(config["coder_repair_attempts"]),
        fixed_evaluator,
    )
    engine = RQGM(
        seed_workspace=initial_workspace,
        tasks=tasks,
        slots=slots,
        runtime=Runtime(
            editor=WorkspaceEditor(client, agent_runner),
            task_evaluator=task_evaluator,
            challenger_source=challenger_source,
            anchor_provider=provider,
            anchor_evaluator=BatchedAnchorEvaluator(
                client,
                provider,
                agent_runner,
                int(config.get("anchor_batch_size", len(crave_anchors))),
            ),
            training_feedback=TrainingFeedback(
                client,
                crave_train,
                int(config["training_samples_per_node"]) if reviewer_training else 0,
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
        by_language = {
            language: {
                "passes": sum(
                    item["outcome"]
                    for item in outcomes
                    if item["task"].split("/", 1)[0] == language
                ),
                "total": sum(item["task"].split("/", 1)[0] == language for item in outcomes),
            }
            for language in LANGUAGES
        }
        language_rates = [
            scores["passes"] / scores["total"] for scores in by_language.values() if scores["total"]
        ]
        heldout[endpoint_name] = {
            "node_id": node_id,
            "passes": sum(item["outcome"] for item in outcomes),
            "total": len(outcomes),
            "by_language": by_language,
            "macro_average": sum(language_rates) / len(language_rates),
            "outcomes": outcomes,
        }
    token_fields = (
        "input_tokens",
        "cached_input_tokens",
        "cache_write_input_tokens",
        "output_tokens",
        "reasoning_output_tokens",
        "raw_total_tokens",
        "blended_tokens",
    )
    token_totals = {
        field: sum(int(call.get(field, 0)) for call in client.calls) for field in token_fields
    }
    token_totals["metered_calls"] = sum("raw_total_tokens" in call for call in client.calls)
    token_totals["unmetered_calls"] = len(client.calls) - token_totals["metered_calls"]
    data_split = {
        "train": [task.task_id for task in engine.runtime.training_feedback.coder_tasks],
        "validation": [task.task_id for task in task_evaluator.validation_tasks],
        "heldout": [task.task_id for task in heldout_tasks],
    }
    summary = {
        "claim": config["claim"],
        "experiment_condition": config.get("experiment_condition", "coevolving_reviewer"),
        "ablation_id": config.get("ablation_id"),
        "source": source_identity(),
        "config_fingerprint": canonical_hash(config),
        "data_split_fingerprint": canonical_hash(data_split),
        "result": asdict(result),
        "anchor_provider": {
            "provider_id": engine.runtime.anchor_provider.provider_id,
            "version": engine.runtime.anchor_provider.version,
            "fingerprint": engine.runtime.anchor_provider.fingerprint,
        },
        "model_calls": client.calls,
        "token_totals": token_totals,
        "heldout": heldout,
        "polyglot": {
            "languages": ["cpp", "go", "java", "javascript", "python", "rust"],
            "images": config.get("polyglot_images", DEFAULT_IMAGES),
            "agent_image": config.get("agent_image", "python:3.12-slim"),
            "dataset_commit": "7e0611e77b54e2dea774cdc0aa00cf9f7ed6144f",
            "agent_workspace_format": "openrqgm-agent-codebase-v1",
        },
        "paper_comparison_valid": False,
        "paper_reported_rqgm_endpoint": "119/166",
        "wall_time_seconds": round(time.monotonic() - started, 3),
        "limitations": [
            "paper split and production prompts are unpublished",
            "gpt-5.6-sol differs from the paper's GPT-5 low endpoint",
            "anchor inference is batched as a declared cost-saving approximation",
            "the meta-agent may modify the complete sandboxed agent codebase but not the "
            "trusted RQGM engine, private anchors, benchmark data, or sandbox boundary",
            "public Aider Polyglot tasks are balanced across six languages; this is not the "
            "paper's unpublished exact split",
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
