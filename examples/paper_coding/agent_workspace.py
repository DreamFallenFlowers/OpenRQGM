from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from polyglot import safe_relative

SEED_AGENT = r"""import json, sys

def coder(context):
    return ("You are a careful polyglot coding agent. Solve the supplied task without browsing.\n"
            "Preserve every required API and handle edge cases. Return complete contents "
            "for only the editable files.\n\n"
            + json.dumps(context, ensure_ascii=False))

def repair(context):
    return ("Repair the candidate using the concrete sandbox failure. Do not browse, "
            "change tests, or use test-specific hacks.\n"
            "Return complete contents for only editable files.\n\n"
            + json.dumps(context, ensure_ascii=False))

def reviewer(context):
    return ("Classify the supplied code review examples as APPROVE or REQUEST_CHANGES. "
            "Approve only correct, complete, safe changes. "
            "Use only supplied text and preserve ids.\n\n"
            + json.dumps(context, ensure_ascii=False))

request = json.load(sys.stdin)
operation = request["operation"]
functions = {"coder": coder, "repair": repair, "reviewer": reviewer}
print(json.dumps({"prompt": functions[operation](request["context"])}, ensure_ascii=False))
"""


def seed_workspace() -> dict[str, Any]:
    return {
        "format": "openrqgm-agent-codebase-v1",
        "entrypoint": "agent.py",
        "files": {
            "agent.py": SEED_AGENT,
            "README.md": (
                "Sandboxed prompt-building agent. JSON stdin/stdout protocol: "
                "coder, repair, reviewer."
            ),
        },
    }


def validate_workspace(workspace: dict[str, Any]) -> None:
    if (
        workspace.get("format") != "openrqgm-agent-codebase-v1"
        or workspace.get("entrypoint") != "agent.py"
    ):
        raise ValueError("invalid agent workspace format or entrypoint")
    files = workspace.get("files")
    if not isinstance(files, dict) or "agent.py" not in files or not files:
        raise ValueError("agent workspace must contain agent.py")
    if len(files) > 32:
        raise ValueError("agent workspace file limit exceeded")
    total = 0
    for name, content in files.items():
        safe_relative(name)
        if not isinstance(content, str):
            raise ValueError("workspace files must be UTF-8 text")
        total += len(content.encode("utf-8"))
    if total > 256_000:
        raise ValueError("agent workspace byte limit exceeded")


@dataclass(slots=True)
class AgentWorkspaceRunner:
    timeout: int
    image: str = "python:3.12-slim"

    def prompt(self, workspace: dict[str, Any], operation: str, context: dict[str, Any]) -> str:
        validate_workspace(workspace)
        if operation not in {"coder", "repair", "reviewer"}:
            raise ValueError(operation)
        with tempfile.TemporaryDirectory(prefix="openrqgm-agent-") as temp:
            root = Path(temp)
            for relative, content in workspace["files"].items():
                target = root.joinpath(*safe_relative(relative).parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
            command = [
                "docker",
                "run",
                "--rm",
                "-i",
                "--network",
                "none",
                "--read-only",
                "--memory",
                "256m",
                "--cpus",
                "0.5",
                "--pids-limit",
                "64",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges",
                "--user",
                "65534:65534",
                "--tmpfs",
                "/tmp:rw,noexec,nosuid,size=32m",
                "-v",
                f"{root.resolve()}:/agent:ro",
                "-w",
                "/agent",
                self.image,
                "python3",
                workspace["entrypoint"],
            ]
            completed = subprocess.run(
                command,
                input=json.dumps({"operation": operation, "context": context}, ensure_ascii=False),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(f"agent workspace failed: {completed.stderr[-1000:]}")
            result = json.loads(completed.stdout)
            if (
                set(result) != {"prompt"}
                or not isinstance(result["prompt"], str)
                or not result["prompt"].strip()
            ):
                raise ValueError("agent workspace returned invalid protocol output")
            return result["prompt"]


def workspace_from_files(files: list[dict[str, str]]) -> dict[str, Any]:
    workspace = {
        "format": "openrqgm-agent-codebase-v1",
        "entrypoint": "agent.py",
        "files": {item["path"]: item["content"] for item in files},
    }
    validate_workspace(workspace)
    return workspace
