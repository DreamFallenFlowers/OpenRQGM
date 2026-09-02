from __future__ import annotations

import gzip
import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LIVE_CODE_BENCH_COMMIT = "28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24"
LIVE_CODE_BENCH_RELEASE = "release_v6"
LIVE_CODE_BENCH_PROBLEMS = 1055
LIVE_CODE_BENCH_DATASET = "livecodebench/code_generation_lite"
DEFAULT_IMAGE = "openrqgm-livecodebench:2026-09-02"


@dataclass(frozen=True, slots=True)
class LiveCodeBenchTask:
    question_id: str
    question_content: str
    starter_code: str
    contest_date: str
    difficulty: str
    evaluation_sample: dict[str, str]

    @property
    def task_id(self) -> str:
        return f"livecodebench/{self.question_id}"


def load_snapshot(path: Path) -> tuple[list[LiveCodeBenchTask], dict[str, Any]]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            payload = json.load(stream)
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
    metadata = payload["metadata"]
    if metadata["repository_commit"] != LIVE_CODE_BENCH_COMMIT:
        raise ValueError("LiveCodeBench repository commit drift")
    if metadata["release_version"] != LIVE_CODE_BENCH_RELEASE:
        raise ValueError("LiveCodeBench release drift")
    if metadata["dataset"] != LIVE_CODE_BENCH_DATASET:
        raise ValueError("LiveCodeBench dataset drift")
    tasks = [LiveCodeBenchTask(**row) for row in payload["tasks"]]
    if len(tasks) != LIVE_CODE_BENCH_PROBLEMS:
        raise ValueError(f"expected {LIVE_CODE_BENCH_PROBLEMS} LiveCodeBench tasks")
    return tasks, metadata


def material(task: LiveCodeBenchTask) -> dict[str, Any]:
    return {
        "task": task.task_id,
        "language": "python",
        "instructions": task.question_content,
        "starters": {"solution.py": task.starter_code or "# YOUR CODE HERE\n"},
        # Official LiveCodeBench generation prompts expose the statement and
        # starter code, not the executable public/private evaluator payload.
        "tests": {},
    }


@dataclass(slots=True)
class DockerLiveCodeBenchRunner:
    timeout: int = 30
    image: str = DEFAULT_IMAGE

    def run(self, task: LiveCodeBenchTask, code: str) -> tuple[int, dict[str, Any]]:
        with tempfile.TemporaryDirectory(prefix="openrqgm-livecodebench-") as temp:
            root = Path(temp)
            (root / "request.json").write_text(
                json.dumps(
                    {"sample": task.evaluation_sample, "code": code, "timeout": 6},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            command = [
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                "--read-only",
                "--memory",
                "4g",
                "--cpus",
                "1",
                "--pids-limit",
                "128",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges",
                "--tmpfs",
                "/tmp:rw,exec,nosuid,size=128m",
                "-v",
                f"{root.resolve()}:/input:ro",
                self.image,
            ]
            try:
                completed = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=self.timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return 0, {"timeout": True, "image": self.image}
            except OSError as error:
                return 0, {"sandbox_error": type(error).__name__, "image": self.image}
            try:
                result = json.loads(completed.stdout)
            except json.JSONDecodeError:
                return 0, {
                    "returncode": completed.returncode,
                    "stderr_tail": completed.stderr[-1000:],
                    "invalid_runner_output": True,
                    "image": self.image,
                }
            # The official evaluator metadata can contain hidden inputs and
            # expected outputs. Never persist those in an OpenRQGM run.
            safe_metadata = {
                key: value
                for key, value in result.get("metadata", {}).items()
                if key in {"error_code", "error_message", "execution time"}
            }
            return int(completed.returncode == 0 and result.get("passed") is True), {
                "returncode": completed.returncode,
                "tests": int(result.get("tests", 0)),
                "metadata": safe_metadata,
                "image": self.image,
            }
