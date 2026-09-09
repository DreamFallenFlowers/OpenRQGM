from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHECKOUT = ROOT / "data" / "LiveCodeBench"
OUTPUT = ROOT / "data" / "paper-coding" / "livecodebench-release-v6-all-tasks.json.gz"
EXPECTED_COMMIT = "28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24"
EXPECTED_COUNT = 1055


def canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=CHECKOUT, capture_output=True, text=True, check=True
    ).stdout.strip()
    if commit != EXPECTED_COMMIT:
        raise SystemExit(f"LiveCodeBench checkout must be pinned to {EXPECTED_COMMIT}")
    sys.path.insert(0, str(CHECKOUT))
    from lcb_runner.benchmarks.code_generation import load_code_generation_dataset

    problems = load_code_generation_dataset("release_v6")
    problems = [
        problem for problem in problems if problem.contest_date.date().isoformat() <= "2025-04-30"
    ]
    if len(problems) != EXPECTED_COUNT:
        raise SystemExit(f"expected {EXPECTED_COUNT} release_v6 problems, found {len(problems)}")
    rows = [
        {
            "question_id": problem.question_id,
            "question_content": problem.question_content,
            "starter_code": problem.starter_code,
            "contest_date": problem.contest_date.isoformat(),
            "difficulty": problem.difficulty.value,
            "evaluation_sample": problem.get_evaluation_sample(),
        }
        for problem in sorted(problems, key=lambda item: (item.contest_date, item.question_id))
    ]
    payload = {
        "metadata": {
            "repository": "https://github.com/LiveCodeBench/LiveCodeBench",
            "repository_commit": commit,
            "dataset": "livecodebench/code_generation_lite",
            "release_version": "release_v6",
            "scenario": "codegeneration",
            "not_fast": False,
            "coverage": "all_1055_release_v6_tasks_with_official_pruned_test_cases",
            "problem_count": len(rows),
            "task_identity_hash": canonical_hash([row["question_id"] for row in rows]),
            "private_payload_hash": canonical_hash([row["evaluation_sample"] for row in rows]),
        },
        "tasks": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output, "wt", encoding="utf-8", compresslevel=6) as stream:
        json.dump(payload, stream, ensure_ascii=False)
    print(json.dumps(payload["metadata"], indent=2))


if __name__ == "__main__":
    main()
