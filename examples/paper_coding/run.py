from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import queue
import random
import re
import shutil
import subprocess
import tempfile
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from contextlib import suppress
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
    split_counts,
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
from rqgm.persistence import restore_state, save_state

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "paper-coding"
POLYGLOT = ROOT / "data" / "polyglot-benchmark"


def canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


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


def review_example_size(row: dict[str, Any]) -> int:
    """Return the exact UTF-8 JSON payload size used for context eligibility."""
    return len(
        json.dumps(
            review_example("eligibility", row),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def context_eligible_rows(
    rows: Sequence[dict[str, Any]], max_example_bytes: int
) -> list[dict[str, Any]]:
    """Keep only complete CRAVE examples that fit the preregistered input cap."""
    if max_example_bytes <= 0:
        raise ValueError("CRAVE example byte limit must be positive")
    return [row for row in rows if review_example_size(row) <= max_example_bytes]


def payload_batches(
    items: Sequence[Any],
    *,
    max_items: int,
    max_payload_bytes: int,
    payload: Callable[[Any], object],
) -> list[list[Any]]:
    """Pack items deterministically without exceeding count or payload limits."""
    if max_items <= 0 or max_payload_bytes <= 0:
        raise ValueError("payload batch limits must be positive")
    batches: list[list[Any]] = []
    current: list[Any] = []
    current_bytes = 0
    for item in items:
        item_bytes = len(
            json.dumps(payload(item), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        )
        if item_bytes > max_payload_bytes:
            raise ValueError("one review example exceeds the prompt payload limit")
        if current and (
            len(current) >= max_items or current_bytes + item_bytes > max_payload_bytes
        ):
            batches.append(current)
            current = []
            current_bytes = 0
        current.append(item)
        current_bytes += item_bytes
    if current:
        batches.append(current)
    return batches


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


class ModelCallError(Exception):
    """Transient model-service failure that must not become a benchmark outcome."""


class AppServerWorker:
    """One persistent Codex app-server process serving one request at a time."""

    def __init__(self, executable: str, max_calls: int = 200) -> None:
        self.executable = executable
        self.max_calls = max_calls
        self.process: subprocess.Popen[str] | None = None
        self.stdout_queue: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self.stderr_tail: deque[str] = deque(maxlen=40)
        self.request_id = 0
        self.call_count = 0
        self.lock = threading.Lock()

    def _pump_stdout(self, stream: Any) -> None:
        for line in stream:
            try:
                self.stdout_queue.put(json.loads(line))
            except json.JSONDecodeError:
                continue
        self.stdout_queue.put(None)

    def _pump_stderr(self, stream: Any) -> None:
        for line in stream:
            self.stderr_tail.append(line.rstrip())

    def _start(self, timeout: int) -> None:
        if self.process is not None and self.process.poll() is None:
            return
        self.stop()
        self.stdout_queue = queue.Queue()
        self.stderr_tail = deque(maxlen=40)
        self.process = subprocess.Popen(
            [
                self.executable,
                "app-server",
                "--stdio",
                "--disable",
                "apps",
                "--disable",
                "browser_use",
                "--disable",
                "computer_use",
                "--disable",
                "plugins",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert self.process.stdout is not None and self.process.stderr is not None
        threading.Thread(target=self._pump_stdout, args=(self.process.stdout,), daemon=True).start()
        threading.Thread(target=self._pump_stderr, args=(self.process.stderr,), daemon=True).start()
        response = self._request(
            "initialize",
            {"clientInfo": {"name": "openrqgm", "version": "1"}},
            time.monotonic() + timeout,
        )
        if "error" in response:
            raise ModelCallError(f"Codex app-server initialize failed: {response['error']}")
        self._notify("initialized", {})
        self.call_count = 0

    def _write(self, message: dict[str, Any]) -> None:
        if self.process is None or self.process.stdin is None or self.process.poll() is not None:
            raise ModelCallError("Codex app-server is not running")
        self.process.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
        self.process.stdin.flush()

    def _notify(self, method: str, params: dict[str, Any]) -> None:
        self._write({"method": method, "params": params})

    def _next_message(self, deadline: float) -> dict[str, Any]:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ModelCallError("Codex app-server request timed out")
        try:
            message = self.stdout_queue.get(timeout=remaining)
        except queue.Empty as error:
            raise ModelCallError("Codex app-server request timed out") from error
        if message is None:
            tail = "\n".join(self.stderr_tail)[-1200:]
            raise ModelCallError(f"Codex app-server exited unexpectedly: {tail}")
        return message

    def _request(
        self,
        method: str,
        params: dict[str, Any],
        deadline: float,
        observe: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        self.request_id += 1
        request_id = self.request_id
        self._write({"id": request_id, "method": method, "params": params})
        while True:
            message = self._next_message(deadline)
            if observe is not None:
                observe(message)
            if message.get("id") == request_id:
                return message

    @staticmethod
    def _usage_fields(usage: dict[str, Any]) -> dict[str, int]:
        mapping = {
            "totalTokens": "raw_total_tokens",
            "inputTokens": "input_tokens",
            "cachedInputTokens": "cached_input_tokens",
            "cacheWriteInputTokens": "cache_write_input_tokens",
            "outputTokens": "output_tokens",
            "reasoningOutputTokens": "reasoning_output_tokens",
        }
        result = {
            target: int(usage.get(source, 0))
            for source, target in mapping.items()
            if isinstance(usage.get(source), int)
        }
        result["blended_tokens"] = result.get("input_tokens", 0) + 5 * result.get(
            "output_tokens", 0
        )
        return result

    def call(
        self,
        prompt: str,
        schema: dict[str, Any],
        model: str,
        reasoning_effort: str | None,
        workdir: Path,
        timeout: int,
    ) -> tuple[dict[str, Any], dict[str, int]]:
        with self.lock:
            deadline = time.monotonic() + timeout
            try:
                self._start(timeout)
                thread_response = self._request(
                    "thread/start",
                    {
                        "model": model,
                        "cwd": str(workdir),
                        "approvalPolicy": "never",
                        "sandbox": "read-only",
                        "ephemeral": True,
                        "config": {"model_reasoning_effort": reasoning_effort or "low"},
                    },
                    deadline,
                )
                if "error" in thread_response:
                    raise ModelCallError(str(thread_response["error"]))
                thread_id = thread_response["result"]["thread"]["id"]
                usage: dict[str, int] = {}

                def observe(message: dict[str, Any]) -> None:
                    nonlocal usage
                    if message.get("method") == "thread/tokenUsage/updated":
                        raw = message.get("params", {}).get("tokenUsage", {}).get("last", {})
                        usage = self._usage_fields(raw)

                turn_response = self._request(
                    "turn/start",
                    {
                        "threadId": thread_id,
                        "input": [{"type": "text", "text": prompt}],
                        "model": model,
                        "effort": reasoning_effort,
                        "outputSchema": schema,
                    },
                    deadline,
                    observe,
                )
                if "error" in turn_response:
                    raise ModelCallError(str(turn_response["error"]))
                turn_id = turn_response["result"]["turn"]["id"]
                completed: dict[str, Any] | None = None
                while completed is None:
                    message = self._next_message(deadline)
                    observe(message)
                    if (
                        message.get("method") == "turn/completed"
                        and message.get("params", {}).get("turn", {}).get("id") == turn_id
                    ):
                        completed = message["params"]["turn"]
                if completed.get("status") != "completed":
                    raise ModelCallError(
                        f"Codex turn failed: {completed.get('error') or completed.get('status')}"
                    )
                texts = [
                    item["text"]
                    for item in completed.get("items", [])
                    if item.get("type") == "agentMessage" and item.get("phase") == "final_answer"
                ]
                if not texts:
                    raise ModelCallError("Codex turn completed without a final answer")
                try:
                    result = json.loads(texts[-1])
                except json.JSONDecodeError as error:
                    raise ModelCallError("Codex returned invalid structured output") from error
                self.call_count += 1
                if self.call_count >= self.max_calls:
                    self.stop()
                return result, usage
            except (OSError, BrokenPipeError, KeyError, TypeError) as error:
                tail = "\n".join(self.stderr_tail)[-1200:]
                self.stop()
                raise ModelCallError(f"Codex app-server failure: {error}; {tail}") from error
            except ModelCallError:
                self.stop()
                raise

    def stop(self) -> None:
        process, self.process = self.process, None
        if process is None:
            return
        if process.stdin is not None:
            with suppress(OSError):
                process.stdin.close()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()


@dataclass(slots=True)
class CodexAppServer:
    model: str
    timeout: int
    workdir: Path
    ledger_path: Path | None = None
    reasoning_effort: str | None = None
    concurrency: int = 4
    worker_max_calls: int = 200
    calls: list[dict[str, Any]] = field(default_factory=list)
    workers: list[AppServerWorker] = field(default_factory=list)
    available: asyncio.Queue[AppServerWorker] | None = None
    record_lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        if self.concurrency <= 0:
            raise ValueError("model concurrency must be positive")
        if self.ledger_path is not None and self.ledger_path.exists():
            for line in self.ledger_path.read_text(encoding="utf-8").splitlines():
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict):
                    self.calls.append(event)

    def _record_call(self, event: dict[str, Any]) -> None:
        with self.record_lock:
            self.calls.append(event)
            if self.ledger_path is None:
                return
            self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
            with self.ledger_path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(event, sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())

    def _ensure_pool(self) -> asyncio.Queue[AppServerWorker]:
        if self.available is None:
            executable = find_codex_executable()
            self.workers = [
                AppServerWorker(executable, self.worker_max_calls) for _ in range(self.concurrency)
            ]
            self.available = asyncio.Queue()
            for worker in self.workers:
                self.available.put_nowait(worker)
        return self.available

    async def json(self, prompt: str, schema: dict[str, Any], purpose: str) -> dict[str, Any]:
        self.workdir.mkdir(parents=True, exist_ok=True)
        last_error: ModelCallError | None = None
        for attempt in range(3):
            worker = await self._ensure_pool().get()
            print(f"[model:start] {purpose} backend=app-server", flush=True)
            started = time.monotonic()
            try:
                result, usage = await asyncio.to_thread(
                    worker.call,
                    prompt,
                    schema,
                    self.model,
                    self.reasoning_effort,
                    self.workdir,
                    self.timeout,
                )
                event: dict[str, Any] = {
                    "purpose": purpose,
                    "model": self.model,
                    "reasoning_effort": self.reasoning_effort,
                    "backend": "app-server",
                    "returncode": 0,
                    "prompt_sha256": canonical_hash(prompt),
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                    **usage,
                }
                self._record_call(event)
                print(f"[model:done] {purpose} rc=0 backend=app-server", flush=True)
                return result
            except ModelCallError as error:
                last_error = error
                self._record_call(
                    {
                        "purpose": purpose,
                        "model": self.model,
                        "reasoning_effort": self.reasoning_effort,
                        "backend": "app-server",
                        "returncode": 1,
                        "prompt_sha256": canonical_hash(prompt),
                        "elapsed_seconds": round(time.monotonic() - started, 3),
                        "error": str(error)[-1200:],
                    }
                )
                print(f"[model:done] {purpose} rc=1 backend=app-server", flush=True)
            finally:
                self._ensure_pool().put_nowait(worker)
            if attempt < 2:
                await asyncio.sleep(30 * (2**attempt))
        assert last_error is not None
        raise last_error

    async def close(self) -> None:
        await asyncio.gather(*(asyncio.to_thread(worker.stop) for worker in self.workers))


@dataclass(slots=True)
class CodexCli:
    model: str
    timeout: int
    workdir: Path
    ledger_path: Path | None = None
    reasoning_effort: str | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)
    record_lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        if self.ledger_path is None or not self.ledger_path.exists():
            return
        for line in self.ledger_path.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                self.calls.append(event)

    def _record_call(self, event: dict[str, Any]) -> None:
        with self.record_lock:
            self.calls.append(event)
            if self.ledger_path is None:
                return
            self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
            with self.ledger_path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(event, sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())

    async def json(self, prompt: str, schema: dict[str, Any], purpose: str) -> dict[str, Any]:
        last_error: ModelCallError | None = None
        for attempt in range(3):
            try:
                return await asyncio.to_thread(self._json_sync, prompt, schema, purpose)
            except ModelCallError as error:
                last_error = error
                if attempt < 2:
                    await asyncio.sleep(30 * (2**attempt))
        assert last_error is not None
        raise last_error

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
            ]
            if self.reasoning_effort:
                command.extend(["-c", f'model_reasoning_effort="{self.reasoning_effort}"'])
            command.extend(
                [
                    "-C",
                    str(self.workdir),
                    "--output-schema",
                    str(schema_path),
                    "-o",
                    str(output_path),
                    "-",
                ]
            )
            print(f"[model:start] {purpose}", flush=True)
            started = time.monotonic()
            try:
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
            except subprocess.TimeoutExpired as error:
                self._record_call(
                    {
                        "purpose": purpose,
                        "model": self.model,
                        "reasoning_effort": self.reasoning_effort,
                        "returncode": None,
                        "prompt_sha256": canonical_hash(prompt),
                        "elapsed_seconds": round(time.monotonic() - started, 3),
                        "error": "timeout",
                    }
                )
                raise ModelCallError(
                    f"Codex call timed out for {purpose} after {self.timeout}s"
                ) from error
            event = {
                "purpose": purpose,
                "model": self.model,
                "reasoning_effort": self.reasoning_effort,
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
            self._record_call(event)
            print(f"[model:done] {purpose} rc={completed.returncode}", flush=True)
            if completed.returncode != 0 or not output_path.exists():
                raise ModelCallError(
                    f"Codex call failed for {purpose}: rc={completed.returncode}; "
                    f"stderr_tail={completed.stderr[-600:]}"
                )
            try:
                return json.loads(output_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise ModelCallError(
                    f"Codex returned invalid structured output for {purpose}"
                ) from error


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
    client: CodexCli | CodexAppServer
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
    client: CodexCli | CodexAppServer
    runner: DockerPolyglotRunner
    agent_runner: AgentWorkspaceRunner
    validation_tasks: list[PolyglotTask]
    crave_validation: list[dict[str, Any]]
    random_seed: int
    repair_attempts: int = 1
    fixed_evaluator: EvaluatorCandidate | None = None
    parallelism: int = 1

    def __post_init__(self) -> None:
        if self.parallelism <= 0:
            raise ValueError("task parallelism must be positive")

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
        ordered = self._ordered_tasks(node.node_id, tasks, "training")
        semaphore = asyncio.Semaphore(self.parallelism)

        async def generate(task: PolyglotTask) -> dict[str, Any]:
            try:
                async with semaphore:
                    artifact = await self._generate_artifact(
                        node,
                        task,
                        phase="training",
                        purpose_prefix="coder-training",
                    )
                return {
                    "task": task.task_id,
                    "outcome": artifact["test_outcome"],
                    "attempts": artifact["attempts"],
                    "sandbox": artifact["sandbox"],
                }
            except (
                OSError,
                RuntimeError,
                subprocess.SubprocessError,
                ValueError,
                json.JSONDecodeError,
            ) as error:
                return {"task": task.task_id, "outcome": 0, "error": type(error).__name__}

        return list(await asyncio.gather(*(generate(task) for task in ordered[:count])))

    async def heldout_results(
        self,
        node: WorkspaceNode,
        tasks: Sequence[PolyglotTask],
    ) -> list[dict[str, Any]]:
        semaphore = asyncio.Semaphore(self.parallelism)

        async def generate(task: PolyglotTask) -> dict[str, Any]:
            try:
                async with semaphore:
                    artifact = await self._generate_artifact(
                        node,
                        task,
                        phase="heldout",
                        purpose_prefix="coder-heldout",
                    )
                return {
                    "task": task.task_id,
                    "outcome": artifact["test_outcome"],
                    "attempts": artifact["attempts"],
                    "sandbox": artifact["sandbox"],
                }
            except (
                OSError,
                RuntimeError,
                subprocess.SubprocessError,
                ValueError,
                json.JSONDecodeError,
            ) as error:
                return {"task": task.task_id, "outcome": 0, "error": type(error).__name__}

        return list(await asyncio.gather(*(generate(task) for task in tasks)))

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
    client: CodexCli | CodexAppServer
    provider: PrivateCraveAnchors
    agent_runner: AgentWorkspaceRunner
    batch_size: int = 20
    max_payload_bytes: int = 160_000
    parallelism: int = 1
    predictions: dict[str, dict[str, str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("anchor batch size must be positive")
        if self.parallelism <= 0:
            raise ValueError("anchor parallelism must be positive")
        if self.max_payload_bytes <= 0:
            raise ValueError("anchor payload limit must be positive")

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
        batches = payload_batches(
            examples,
            max_items=self.batch_size,
            max_payload_bytes=self.max_payload_bytes,
            payload=lambda example: example.artifact,
        )

        async def predict_batch(batch_index: int) -> dict[str, str]:
            batch = batches[batch_index]
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
                    f"anchor:{candidate.candidate_id}:batch-{batch_index}",
                )
                return {item["id"]: item["label"] for item in result["predictions"]}
            except (
                OSError,
                RuntimeError,
                subprocess.SubprocessError,
                ValueError,
                json.JSONDecodeError,
            ):
                return {}

        semaphore = asyncio.Semaphore(self.parallelism)

        async def limited(batch_index: int) -> dict[str, str]:
            async with semaphore:
                return await predict_batch(batch_index)

        predictions: dict[str, str] = {}
        for batch_predictions in await asyncio.gather(
            *(limited(batch_index) for batch_index in range(len(batches)))
        ):
            predictions.update(batch_predictions)
        return predictions

    async def evaluate(self, candidate: EvaluatorCandidate, example: AnchorExample) -> int:
        if candidate.candidate_id not in self.predictions:
            self.predictions[candidate.candidate_id] = await self._predict(candidate)
        prediction = self.predictions[candidate.candidate_id].get(example.example_id)
        return int(prediction == example.expected)


@dataclass(slots=True)
class TrainingFeedback:
    client: CodexCli | CodexAppServer
    rows: list[dict[str, Any]]
    reviewer_samples_per_node: int
    coding_evaluator: CodingTaskEvaluator
    coder_tasks: list[PolyglotTask]
    coder_samples_per_node: int
    max_payload_bytes: int = 160_000
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

        async def collect_reviewer() -> list[dict[str, Any]]:
            reviewer_feedback: list[dict[str, Any]] = []
            if not selected:
                return reviewer_feedback
            try:
                predictions: dict[str, str] = {}
                batches = payload_batches(
                    selected,
                    max_items=max(1, self.reviewer_samples_per_node),
                    max_payload_bytes=self.max_payload_bytes,
                    payload=lambda item: review_example(item[0], item[1]),
                )
                for batch_index, batch in enumerate(batches):
                    prompt = await asyncio.to_thread(
                        self.coding_evaluator.agent_runner.prompt,
                        node.workspace,
                        "reviewer",
                        {
                            "examples": [
                                review_example(example_id, row) for example_id, row in batch
                            ],
                            "response_contract": (
                                "Return a prediction for every id without reordering."
                            ),
                        },
                    )
                    result = await self.client.json(
                        prompt,
                        schema,
                        f"crave-training:batch-{batch_index}:{batch[0][0]}..{batch[-1][0]}",
                    )
                    predictions.update(
                        {item["id"]: item["label"] for item in result["predictions"]}
                    )
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
                return [
                    {
                        "task": "reviewer-crave-training",
                        "example_id": example_id,
                        "error": type(error).__name__,
                    }
                    for example_id, _ in selected
                ]
            return reviewer_feedback

        reviewer_feedback, coder_feedback = await asyncio.gather(
            collect_reviewer(),
            self.coding_evaluator.training_samples(
                node,
                self.coder_tasks,
                self.coder_samples_per_node,
            ),
        )
        return {
            "reviewer_samples": reviewer_feedback,
            "coder_samples": coder_feedback,
            "enters_validation_utility": False,
        }


@dataclass(slots=True)
class RunSnapshotObserver:
    state_path: Path
    progress_path: Path
    interval: int = 8
    prior_wall_seconds: float = 0.0
    attempt_started: float = field(default_factory=time.monotonic)

    def active_wall_seconds(self) -> float:
        return self.prior_wall_seconds + time.monotonic() - self.attempt_started

    async def update(self, engine: RQGM, event: str) -> None:
        should_save = event in {"initialized", "checkpoint", "completed"} or (
            event == "evaluated" and engine.validation_outcomes % self.interval == 0
        )
        if should_save:
            await asyncio.to_thread(save_state, engine, self.state_path)
            await asyncio.to_thread(
                atomic_json,
                self.progress_path,
                {
                    "active_wall_seconds": self.active_wall_seconds(),
                    "validation_outcomes": engine.validation_outcomes,
                },
            )
            print(
                f"[state:saved] event={event} outcomes={engine.validation_outcomes}",
                flush=True,
            )


def restore_runtime_progress(engine: RQGM) -> None:
    feedback = engine.runtime.training_feedback
    if not isinstance(feedback, TrainingFeedback):
        return
    feedback.index = sum(
        len(item.get("reviewer_samples", []))
        for items in engine.archive.training_feedback.values()
        for item in items
        if isinstance(item, dict)
    )


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
    config: dict[str, Any], client: CodexCli | CodexAppServer
) -> tuple[RQGM, CodingTaskEvaluator, list[PolyglotTask]]:
    condition = config.get("experiment_condition", "coevolving_reviewer")
    seed = int(config["random_seed"])
    if "polyglot_train_tasks" in config:
        coder_train_tasks, validation_tasks, heldout_tasks = split_counts(
            POLYGLOT,
            seed,
            int(config["polyglot_train_tasks"]),
            int(config["polyglot_validation_tasks"]),
            int(config["polyglot_test_tasks"]),
        )
    else:
        coder_train_tasks, validation_tasks, heldout_tasks = split_balanced(
            POLYGLOT,
            seed,
            int(config["polyglot_train_tasks_per_language"]),
            int(config["polyglot_validation_tasks_per_language"]),
            int(config["polyglot_test_tasks_per_language"]),
        )
    crave_example_max_bytes = int(config.get("crave_example_max_bytes", 80_000))
    crave_prompt_max_bytes = int(config.get("crave_prompt_max_bytes", 160_000))
    crave_train_rows = context_eligible_rows(load_rows("train"), crave_example_max_bytes)
    crave_train_count = config.get("crave_training_pool_examples", 32)
    crave_train = (
        sample_rows(crave_train_rows, len(crave_train_rows), seed + 1)
        if crave_train_count == "all"
        else sample_rows(crave_train_rows, int(crave_train_count), seed + 1)
    )
    crave_validation = sample_rows(
        context_eligible_rows(load_rows("validation"), crave_example_max_bytes),
        int(config["crave_validation_examples"]),
        seed + 2,
    )
    crave_anchors = sample_rows(
        context_eligible_rows(load_rows("test"), crave_example_max_bytes),
        int(config["crave_anchor_examples"]),
        seed + 3,
    )
    provider = PrivateCraveAnchors(crave_anchors)
    images = config.get("polyglot_images", DEFAULT_IMAGES)
    agent_runner = AgentWorkspaceRunner(
        int(config["agent_timeout_seconds"]), config.get("agent_image", "python:3.12-slim")
    )
    initial_workspace = seed_workspace(config.get("seed_workspace_profile", "reconstruction_v3"))
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
        client=client,
        runner=DockerPolyglotRunner(int(config["container_timeout_seconds"]), images),
        agent_runner=agent_runner,
        validation_tasks=validation_tasks,
        crave_validation=crave_validation,
        random_seed=seed + 4,
        repair_attempts=int(config["coder_repair_attempts"]),
        fixed_evaluator=fixed_evaluator,
        parallelism=int(config.get("task_concurrency", 1)),
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
                client=client,
                provider=provider,
                agent_runner=agent_runner,
                batch_size=int(config.get("anchor_batch_size", len(crave_anchors))),
                max_payload_bytes=crave_prompt_max_bytes,
                parallelism=int(config.get("anchor_concurrency", 1)),
            ),
            training_feedback=TrainingFeedback(
                client,
                crave_train,
                int(config["training_samples_per_node"]) if reviewer_training else 0,
                task_evaluator,
                coder_train_tasks,
                int(config["coder_training_samples_per_node"]),
                crave_prompt_max_bytes,
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
    state_path = output / "state.json"
    metadata_path = output / "run-metadata.json"
    progress_path = output / "runtime-progress.json"
    client_class = (
        CodexAppServer if config.get("model_backend", "cli") == "app_server" else CodexCli
    )
    client_options: dict[str, Any] = {}
    if client_class is CodexAppServer:
        client_options = {
            "concurrency": int(config.get("model_concurrency", 4)),
            "worker_max_calls": int(config.get("persistent_worker_max_calls", 200)),
        }
    client = client_class(
        model=config["model"],
        timeout=int(config["model_timeout_seconds"]),
        workdir=DATA / "codex-empty",
        ledger_path=output / "model-calls.jsonl",
        reasoning_effort=config.get("reasoning_effort"),
        **client_options,
    )
    engine, task_evaluator, heldout_tasks = build_engine(config, client)
    prior_wall_seconds = 0.0
    if progress_path.exists():
        prior_wall_seconds = float(
            json.loads(progress_path.read_text(encoding="utf-8"))["active_wall_seconds"]
        )
    progress_observer = RunSnapshotObserver(
        state_path,
        progress_path,
        int(config.get("snapshot_interval", 8)),
        prior_wall_seconds,
    )
    engine.runtime.progress_observer = progress_observer
    data_split = {
        "train": [task.task_id for task in engine.runtime.training_feedback.coder_tasks],
        "validation": [task.task_id for task in task_evaluator.validation_tasks],
        "heldout": [task.task_id for task in heldout_tasks],
    }
    run_metadata = {
        "ablation_id": config.get("ablation_id"),
        "condition": config.get("experiment_condition", "coevolving_reviewer"),
        "config_fingerprint": canonical_hash(config),
        "data_split_fingerprint": canonical_hash(data_split),
        "crave_training_fingerprint": canonical_hash(engine.runtime.training_feedback.rows),
        "crave_validation_fingerprint": canonical_hash(task_evaluator.crave_validation),
        "anchor_fingerprint": engine.runtime.anchor_provider.fingerprint,
        "source": source_identity(),
    }
    if metadata_path.exists():
        saved_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if saved_metadata != run_metadata:
            raise ValueError("run metadata changed; refusing an incommensurate resume")
    else:
        atomic_json(metadata_path, run_metadata)
    if state_path.exists():
        restore_state(engine, state_path, require_resumable=True)
        restore_runtime_progress(engine)
        print(
            f"[state:restored] outcomes={engine.validation_outcomes} "
            f"checkpoints={len(engine.checkpoint_events)}",
            flush=True,
        )
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
    summary = {
        "claim": config["claim"],
        "experiment_condition": config.get("experiment_condition", "coevolving_reviewer"),
        "ablation_id": config.get("ablation_id"),
        "source": source_identity(),
        "config_fingerprint": canonical_hash(config),
        "data_split_fingerprint": canonical_hash(data_split),
        "crave_training_fingerprint": canonical_hash(engine.runtime.training_feedback.rows),
        "crave_validation_fingerprint": canonical_hash(task_evaluator.crave_validation),
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
        "wall_time_seconds": round(progress_observer.active_wall_seconds(), 3),
        "limitations": config.get(
            "limitations",
            [
                "the paper's exact task identities and production harness are unpublished",
                "the model provider revision and complete per-role prompts are unavailable",
                "the public adapter uses bounded generate-test-repair calls instead of "
                "the paper's exact tool loop",
            ],
        ),
    }
    atomic_json(output / "summary.json", summary)
    save_state(engine, output / "state.json")
    if isinstance(client, CodexAppServer):
        await client.close()
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(execute(args.config.resolve()))


if __name__ == "__main__":
    main()
