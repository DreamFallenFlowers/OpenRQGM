#!/usr/bin/env python3
"""Small Docker-CLI compatibility shim for Podman on GPUFree.

The outer Kubernetes container does not delegate cgroups or permit creating a
network namespace.  Convert Docker's cgroup limits to OCI rlimits and replace
``--network none`` with host networking plus a seccomp policy that denies all
socket syscalls.  The benchmark still preserves wall-clock timeouts, read-only
mounts, dropped capabilities, and no-new-privileges policy.
"""

from __future__ import annotations

import os
import subprocess
import sys

PODMAN_PREFIX = [
    "/usr/bin/podman",
    "--root",
    "/root/gpufree-data/root-containers",
    "--runroot",
    "/run/root/containers",
    "--storage-driver",
    "vfs",
]
SECCOMP_NO_NETWORK = os.environ.get(
    "OPENRQGM_SECCOMP_NO_NETWORK",
    "/root/gpufree-data/openrqgm/seccomp-no-network.json",
)


def bytes_from_limit(value: str) -> int:
    suffixes = {"k": 1024, "m": 1024**2, "g": 1024**3}
    normalized = value.strip().lower()
    if normalized[-1:] in suffixes:
        return int(float(normalized[:-1]) * suffixes[normalized[-1]])
    return int(normalized)


def translate(arguments: list[str]) -> list[str]:
    if arguments and arguments[0] == "build":
        # Image definitions are trusted repository inputs. Buildah's chroot
        # isolation avoids the outer Kubernetes container's unavailable cgroup
        # delegation; candidate execution still uses the hardened run path.
        return ["build", "--isolation", "chroot", "--layers", *arguments[1:]]
    if not arguments or arguments[0] != "run":
        return arguments
    translated = [
        "run",
        "--cgroups=disabled",
        "--security-opt",
        f"seccomp={SECCOMP_NO_NETWORK}",
    ]
    limits: list[str] = []
    index = 1
    while index < len(arguments):
        argument = arguments[index]
        if argument == "--memory" and index + 1 < len(arguments):
            memory = bytes_from_limit(arguments[index + 1])
            limits.extend(["--ulimit", f"rss={memory}:{memory}"])
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
        if argument == "--network" and index + 1 < len(arguments):
            network = arguments[index + 1]
            translated.extend(["--network", "host" if network == "none" else network])
            index += 2
            continue
        translated.append(argument)
        index += 1
    translated[1:1] = limits
    return translated


def main() -> int:
    environment = os.environ.copy()
    environment["XDG_RUNTIME_DIR"] = "/run/root"
    return subprocess.run(PODMAN_PREFIX + translate(sys.argv[1:]), env=environment).returncode


if __name__ == "__main__":
    raise SystemExit(main())
