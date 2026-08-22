#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

ADAPTER = """from rqgm import EvaluationOutcome


class WorkspaceEditor:
    async def edit(self, parent, archive, budget):
        raise NotImplementedError("return a child workspace; never expose anchors here")


class TaskEvaluator:
    async def evaluate(self, node, task, evaluator, cached_artifact, budget):
        raise NotImplementedError("return EvaluationOutcome(0 or 1)")


class ChallengerSource:
    async def challengers(self, slot, archive):
        raise NotImplementedError("return evaluator candidates derived without anchor access")


class PrivateAnchorProvider:
    provider_id = "your-domain-private-anchor"
    version = "v1"
    fingerprint = "sha256:compute-this-from-canonical-private-anchor-bytes"

    async def examples(self, slot_id):
        raise NotImplementedError("load held-out examples at runtime; do not serialize them")


class AnchorEvaluator:
    async def evaluate(self, candidate, example):
        raise NotImplementedError("return exact binary agreement with ground truth")
"""

RUN = """import asyncio

from rqgm import RQGM, RQGMConfig, Runtime


async def main():
    raise NotImplementedError("construct roles, slots, runtime and seed workspace")


if __name__ == "__main__":
    asyncio.run(main())
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Create an RQGM domain-adapter skeleton")
    parser.add_argument("target", type=Path)
    args = parser.parse_args()
    target = args.target.resolve()
    target.mkdir(parents=True, exist_ok=True)
    files = {"adapter.py": ADAPTER, "run.py": RUN}
    for name, content in files.items():
        path = target / name
        if path.exists():
            raise SystemExit(f"refusing to overwrite {path}")
        path.write_text(content, encoding="utf-8")
    print(f"created RQGM adapter skeleton in {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
