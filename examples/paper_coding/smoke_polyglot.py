from __future__ import annotations

import argparse
import json
from pathlib import Path

from polyglot import DockerPolyglotRunner, PolyglotTask

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "polyglot-benchmark"


def reference_cases() -> list[tuple[PolyglotTask, dict[str, str]]]:
    definitions = {
        "cpp": (
            "all-your-base",
            {"all_your_base.cpp": ".meta/example.cpp", "all_your_base.h": ".meta/example.h"},
        ),
        "go": ("beer-song", {"beer_song.go": ".meta/example.go"}),
        "java": (
            "affine-cipher",
            {"src/main/java/AffineCipher.java": ".meta/src/reference/java/AffineCipher.java"},
        ),
        "javascript": ("affine-cipher", {"affine-cipher.js": ".meta/proof.ci.js"}),
        "python": ("affine-cipher", {"affine_cipher.py": ".meta/example.py"}),
        "rust": ("accumulate", {"src/lib.rs": ".meta/example.rs"}),
    }
    cases: list[tuple[PolyglotTask, dict[str, str]]] = []
    for language, (name, mapping) in definitions.items():
        task_path = DATA / language / "exercises" / "practice" / name
        replacements = {
            target: (task_path / source).read_text(encoding="utf-8")
            for target, source in mapping.items()
        }
        cases.append((PolyglotTask(language, task_path), replacements))
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    runner = DockerPolyglotRunner(args.timeout)
    results = []
    for task, replacements in reference_cases():
        outcome, metadata = runner.run(task, replacements)
        results.append({"task": task.task_id, "outcome": outcome, "metadata": metadata})
        print(f"[{task.language}] {task.path.name}: {'PASS' if outcome else 'FAIL'}", flush=True)
    output = ROOT / "runs" / "paper-coding-six-language-smoke.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    if not all(item["outcome"] for item in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
