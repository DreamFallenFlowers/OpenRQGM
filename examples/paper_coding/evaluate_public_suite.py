from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from agent_workspace import AgentWorkspaceRunner
from livecodebench import DockerLiveCodeBenchRunner, load_snapshot
from livecodebench import material as lcb_material
from polyglot import (
    DEFAULT_IMAGES,
    DockerPolyglotRunner,
    discover_tasks,
    replacements_from_json,
)
from polyglot import (
    material as polyglot_material,
)
from run import CodexCli, canonical_hash

ROOT = Path(__file__).resolve().parents[2]
POLYGLOT = ROOT / "data" / "polyglot-benchmark"
LCB_SNAPSHOT = ROOT / "data" / "paper-coding" / "livecodebench-release-v6-all-tasks.json.gz"

FILES_SCHEMA = {
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


class PublicSuiteModelError(Exception):
    """Abort a resumable run instead of recording API failures as wrong answers."""


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def load_results(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return {
        row["task"]: row
        for line in path.read_text(encoding="utf-8").splitlines()
        if (row := json.loads(line)).get("task")
    }


def select_workspaces(state_path: Path, summary_path: Path) -> dict[str, dict[str, Any]]:
    state = json.loads(state_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    nodes = {node["node_id"]: node["workspace"] for node in state["nodes"]}
    selected: dict[str, dict[str, Any]] = {}
    for name in ("generalist", "coder_specialist"):
        node_id = summary["heldout"][name]["node_id"]
        selected.setdefault(node_id, nodes[node_id])
    return selected


async def generate(
    client: CodexCli,
    agent_runner: AgentWorkspaceRunner,
    workspace: dict[str, Any],
    material: dict[str, Any],
    purpose: str,
) -> dict[str, str]:
    prompt = await asyncio.to_thread(
        agent_runner.prompt,
        workspace,
        "coder",
        {
            "language": material["language"],
            "instructions": material["instructions"],
            "editable_files": material["starters"],
            "repository_tests": material["tests"],
            "response_contract": "Return complete contents for editable files only.",
        },
    )
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            result = await client.json(prompt, FILES_SCHEMA, purpose)
            break
        except (OSError, RuntimeError, subprocess.SubprocessError) as error:
            last_error = error
            if attempt < 2:
                await asyncio.sleep(30 * (2**attempt))
    else:
        raise PublicSuiteModelError(f"model call failed after retries: {purpose}") from last_error
    return replacements_from_json(result["files"])


async def evaluate_node(
    *,
    node_id: str,
    workspace: dict[str, Any],
    config: dict[str, Any],
    output: Path,
    benchmarks: set[str],
) -> dict[str, Any]:
    node_output = output / node_id
    client = CodexCli(
        config["model"],
        int(config["model_timeout_seconds"]),
        ROOT / "data" / "paper-coding" / "codex-empty",
        node_output / "model-calls.jsonl",
        config.get("reasoning_effort"),
    )
    agent_runner = AgentWorkspaceRunner(
        int(config["agent_timeout_seconds"]), config.get("agent_image", "python:3.12-slim")
    )
    report: dict[str, Any] = {"node_id": node_id, "workspace_hash": canonical_hash(workspace)}

    if "polyglot" in benchmarks:
        path = node_output / "aider-polyglot-full.jsonl"
        saved = load_results(path)
        tasks = discover_tasks(POLYGLOT, int(config["random_seed"]))
        runner = DockerPolyglotRunner(
            int(config["container_timeout_seconds"]),
            config.get("polyglot_images", DEFAULT_IMAGES),
        )
        for index, task in enumerate(tasks, 1):
            if task.task_id in saved:
                continue
            try:
                replacements = await generate(
                    client,
                    agent_runner,
                    workspace,
                    polyglot_material(task),
                    f"public-suite:polyglot:{task.task_id}",
                )
                outcome, sandbox = await asyncio.to_thread(runner.run, task, replacements)
                artifact_hash = canonical_hash(replacements)
            except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as error:
                outcome, sandbox = 0, {"error": type(error).__name__}
                artifact_hash = None
            row = {
                "task": task.task_id,
                "outcome": outcome,
                "artifact_hash": artifact_hash,
                "sandbox": sandbox,
            }
            append_jsonl(path, row)
            saved[task.task_id] = row
            print(f"[polyglot {index}/225] node={node_id} score={outcome}", flush=True)
        report["aider_polyglot_full"] = {
            "passes": sum(row["outcome"] for row in saved.values()),
            "total": len(tasks),
            "protocol": "pass@1_ground_truth_blind",
            "dataset_commit": "7e0611e77b54e2dea774cdc0aa00cf9f7ed6144f",
        }

    if "livecodebench" in benchmarks:
        path = node_output / "livecodebench-release-v6-all-tasks.jsonl"
        saved = load_results(path)
        tasks, metadata = load_snapshot(LCB_SNAPSHOT)
        runner = DockerLiveCodeBenchRunner(
            int(config.get("livecodebench_container_timeout_seconds", 60)),
            config.get("livecodebench_image", "openrqgm-livecodebench:2026-09-02"),
        )
        for index, task in enumerate(tasks, 1):
            if task.task_id in saved:
                continue
            try:
                replacements = await generate(
                    client,
                    agent_runner,
                    workspace,
                    lcb_material(task),
                    f"public-suite:livecodebench:{task.question_id}",
                )
                code = replacements.get("solution.py")
                if code is None or set(replacements) != {"solution.py"}:
                    raise ValueError("LiveCodeBench response must contain only solution.py")
                outcome, sandbox = await asyncio.to_thread(runner.run, task, code)
                artifact_hash = "sha256:" + hashlib.sha256(code.encode()).hexdigest()
            except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as error:
                outcome, sandbox = 0, {"error": type(error).__name__}
                artifact_hash = None
            row = {
                "task": task.task_id,
                "contest_date": task.contest_date,
                "difficulty": task.difficulty,
                "outcome": outcome,
                "artifact_hash": artifact_hash,
                "sandbox": sandbox,
            }
            append_jsonl(path, row)
            saved[task.task_id] = row
            print(f"[livecodebench {index}/1055] node={node_id} score={outcome}", flush=True)
        report["livecodebench_release_v6_all_tasks"] = {
            "passes": sum(row["outcome"] for row in saved.values()),
            "total": len(tasks),
            "protocol": "official_default_codegeneration_lite_pass@1_all_tasks",
            **metadata,
        }
    return report


async def execute(args: argparse.Namespace) -> None:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    workspaces = select_workspaces(args.state, args.summary)
    reports = []
    for node_id, workspace in workspaces.items():
        reports.append(
            await evaluate_node(
                node_id=node_id,
                workspace=workspace,
                config=config,
                output=args.output,
                benchmarks=set(args.benchmark),
            )
        )
    final = {
        "claim": "Full public-suite external evaluation; not paper-result reproduction.",
        "search_feedback_used": False,
        "benchmarks": sorted(set(args.benchmark)),
        "nodes": reports,
    }
    (args.output / "summary.json").write_text(json.dumps(final, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--benchmark",
        action="append",
        choices=("polyglot", "livecodebench"),
        default=[],
    )
    args = parser.parse_args()
    if not args.benchmark:
        args.benchmark = ["polyglot", "livecodebench"]
    asyncio.run(execute(args))


if __name__ == "__main__":
    main()
