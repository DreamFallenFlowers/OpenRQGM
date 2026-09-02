#!/usr/bin/env python3
"""Small Docker-CLI compatibility shim for rootless Podman on GPUFree.

The outer Kubernetes container does not delegate cgroups. Convert Docker's
memory/PID flags to OCI rlimits while preserving the benchmark's wall-clock
timeouts, network isolation, read-only mounts, dropped capabilities, and
no-new-privileges policy.
"""

from __future__ import annotations

import os
import subprocess
import sys

PODMAN_PREFIX = [
    "/usr/bin/podman",
    "--root",
    "/root/gpufree-data/containers",
    "--runroot",
    "/run/user/1000/containers",
    "--storage-driver",
    "overlay",
    "--storage-opt",
    "overlay.mount_program=/usr/bin/fuse-overlayfs",
]


def bytes_from_limit(value: str) -> int:
    suffixes = {"k": 1024, "m": 1024**2, "g": 1024**3}
    normalized = value.strip().lower()
    if normalized[-1:] in suffixes:
        return int(float(normalized[:-1]) * suffixes[normalized[-1]])
    return int(normalized)


def translate(arguments: list[str]) -> list[str]:
    if not arguments or arguments[0] != "run":
        return arguments
    translated = ["run"]
    limits: list[str] = []
    index = 1
    while index < len(arguments):
        argument = arguments[index]
        if argument == "--memory" and index + 1 < len(arguments):
            memory = bytes_from_limit(arguments[index + 1])
            limits.extend(["--ulimit", f"as={memory}:{memory}"])
            index += 2
            continue
        if argument == "--pids-limit" and index + 1 < len(arguments):
            pids = int(arguments[index + 1])
            limits.extend(["--ulimit", f"nproc={pids}:{pids}"])
            index += 2
            continue
        if argument == "--cpus" and index + 1 < len(arguments):
            # CPU quota requires delegated cgroups. The caller's subprocess
            # timeout remains the hard CPU/wall-time termination boundary.
            index += 2
            continue
        translated.append(argument)
        index += 1
    translated[1:1] = limits
    return translated


def main() -> int:
    environment = os.environ.copy()
    environment["XDG_RUNTIME_DIR"] = "/run/user/1000"
    return subprocess.run(PODMAN_PREFIX + translate(sys.argv[1:]), env=environment).returncode


if __name__ == "__main__":
    raise SystemExit(main())
