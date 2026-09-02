from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "podman_docker_compat", ROOT / "scripts" / "remote" / "podman_docker_compat.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_run_limits_become_oci_rlimits() -> None:
    translated = MODULE.translate(
        [
            "run",
            "--rm",
            "--memory",
            "256m",
            "--cpus",
            "0.5",
            "--pids-limit",
            "64",
            "python:3.12-slim",
        ]
    )
    assert "--memory" not in translated
    assert "--cpus" not in translated
    assert "--pids-limit" not in translated
    assert "as=268435456:268435456" in translated
    assert "nproc=64:64" in translated


def test_non_run_commands_are_unchanged() -> None:
    assert MODULE.translate(["build", "-t", "image", "."]) == [
        "build",
        "-t",
        "image",
        ".",
    ]
