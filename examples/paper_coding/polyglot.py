from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

LANGUAGES = ("cpp", "go", "java", "javascript", "python", "rust")
DEFAULT_IMAGES = {language: f"openrqgm-polyglot-{language}:2026-08-23" for language in LANGUAGES}
TEST_COMMANDS: dict[str, list[str]] = {
    "cpp": [
        "bash",
        "-lc",
        "mkdir -p build && cd build && "
        "cmake -DEXERCISM_RUN_ALL_TESTS=1 -G 'Unix Makefiles' .. && "
        "make && ctest --output-on-failure",
    ],
    "go": ["go", "test", "./..."],
    "java": [
        "bash",
        "-lc",
        "cp -a /opt/gradle-cache /home/sandbox/.gradle && gradle --no-daemon test",
    ],
    "javascript": [
        "bash",
        "-lc",
        "ln -s /opt/npm/node_modules node_modules 2>/dev/null || true; "
        "sed -i 's/\\bxtest(/test(/g; s/\\bxit(/it(/g' *.spec.js; npm run test",
    ],
    "python": ["python3", "-m", "pytest", "-q"],
    "rust": ["cargo", "test", "--", "--include-ignored"],
}


@dataclass(frozen=True, slots=True)
class PolyglotTask:
    language: str
    path: Path

    @property
    def task_id(self) -> str:
        return f"{self.language}/{self.path.name}"


def discover_tasks(root: Path, seed: int) -> list[PolyglotTask]:
    import random

    by_language: dict[str, list[PolyglotTask]] = {}
    for language in LANGUAGES:
        base = root / language / "exercises" / "practice"
        tasks = [PolyglotTask(language, path) for path in sorted(base.iterdir()) if path.is_dir()]
        random.Random(f"{seed}:{language}").shuffle(tasks)
        by_language[language] = tasks
    result: list[PolyglotTask] = []
    for index in range(max(map(len, by_language.values()))):
        for language in LANGUAGES:
            tasks = by_language[language]
            if index < len(tasks):
                result.append(tasks[index])
    return result


def split_balanced(
    root: Path,
    seed: int,
    train_per_language: int,
    validation_per_language: int,
    test_per_language: int,
) -> tuple[list[PolyglotTask], list[PolyglotTask], list[PolyglotTask]]:
    tasks = discover_tasks(root, seed)
    grouped = {
        language: [task for task in tasks if task.language == language] for language in LANGUAGES
    }
    train: list[PolyglotTask] = []
    validation: list[PolyglotTask] = []
    test: list[PolyglotTask] = []
    required = train_per_language + validation_per_language + test_per_language
    for language in LANGUAGES:
        items = grouped[language]
        if len(items) < required:
            raise ValueError(f"{language} has {len(items)} tasks, need {required}")
        train.extend(items[:train_per_language])
        validation.extend(items[train_per_language : train_per_language + validation_per_language])
        test.extend(items[train_per_language + validation_per_language : required])
    return train, validation, test


def editable_files(task: PolyglotTask) -> list[Path]:
    root = task.path
    if task.language == "python":
        files = [
            p
            for p in root.glob("*.py")
            if not p.name.endswith("_test.py") and not p.name.startswith("test_")
        ]
    elif task.language == "go":
        files = [p for p in root.glob("*.go") if not p.name.endswith("_test.go")]
    elif task.language == "java":
        files = list((root / "src" / "main" / "java").rglob("*.java"))
    elif task.language == "javascript":
        files = [
            p
            for p in root.glob("*.js")
            if not p.name.endswith(".spec.js") and p.name not in {"babel.config.js"}
        ]
    elif task.language == "rust":
        files = list((root / "src").rglob("*.rs"))
    elif task.language == "cpp":
        files = [
            p
            for p in root.glob("*")
            if p.suffix in {".cpp", ".cc", ".h", ".hpp"} and not p.name.endswith("_test.cpp")
        ]
    else:
        raise ValueError(task.language)
    if not files:
        raise RuntimeError(f"no editable files for {task.task_id}")
    return sorted(files)


def public_test_files(task: PolyglotTask) -> list[Path]:
    editable = set(editable_files(task))
    files: list[Path] = []
    for path in task.path.rglob("*"):
        if (
            not path.is_file()
            or path in editable
            or any(part.startswith(".") for part in path.relative_to(task.path).parts)
        ):
            continue
        if path.stat().st_size <= 200_000:
            files.append(path)
    return sorted(files)


def safe_relative(path: str) -> PurePosixPath:
    candidate = PurePosixPath(path.replace("\\", "/"))
    if (
        candidate.is_absolute()
        or not candidate.parts
        or ":" in candidate.parts[0]
        or any(part in {"", ".", ".."} for part in candidate.parts)
    ):
        raise ValueError(f"unsafe relative path: {path!r}")
    return candidate


@dataclass(slots=True)
class DockerPolyglotRunner:
    timeout: int
    images: dict[str, str] | None = None

    def run(self, task: PolyglotTask, replacements: dict[str, str]) -> tuple[int, dict[str, Any]]:
        image = (self.images or DEFAULT_IMAGES)[task.language]
        allowed = {path.relative_to(task.path).as_posix() for path in editable_files(task)}
        if not replacements or set(replacements) - allowed:
            return 0, {"sandbox_error": "invalid_replacement_paths", "allowed": sorted(allowed)}
        with tempfile.TemporaryDirectory(prefix="openrqgm-polyglot-") as temp:
            # Some official harnesses (notably Exercism C++) derive the exercise
            # name from the source directory basename, so preserve it in-container.
            copied = Path(temp) / task.path.name
            container_workdir = f"/workspace/{task.path.name}"
            shutil.copytree(task.path, copied)
            for relative, content in replacements.items():
                target = copied.joinpath(*safe_relative(relative).parts)
                target.write_text(content, encoding="utf-8")
            command = [
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                "--read-only",
                "--memory",
                "1g",
                "--cpus",
                "1",
                "--pids-limit",
                "256",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges",
                "--tmpfs",
                # Compilers for Go/Rust/C++ materialize test executables in /tmp.
                # The container remains networkless, capability-free, and disposable.
                "/tmp:rw,exec,nosuid,size=128m",
                "--tmpfs",
                "/home/sandbox:rw,nosuid,size=256m",
                "-e",
                "HOME=/home/sandbox",
                "-e",
                "GRADLE_USER_HOME=/home/sandbox/.gradle",
                "-v",
                f"{copied.resolve()}:{container_workdir}:rw",
                "-w",
                container_workdir,
                image,
                *TEST_COMMANDS[task.language],
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
                return int(completed.returncode == 0), {
                    "returncode": completed.returncode,
                    "stdout_tail": completed.stdout[-6000:],
                    "stderr_tail": completed.stderr[-6000:],
                    "language": task.language,
                    "image": image,
                    "command": TEST_COMMANDS[task.language],
                }
            except subprocess.TimeoutExpired:
                return 0, {"timeout": True, "language": task.language}
            except OSError as error:
                return 0, {"sandbox_error": type(error).__name__, "language": task.language}


def material(task: PolyglotTask) -> dict[str, Any]:
    docs = sorted((task.path / ".docs").glob("instruction*.md"))
    starters = {
        path.relative_to(task.path).as_posix(): path.read_text(encoding="utf-8")
        for path in editable_files(task)
    }
    tests = {
        path.relative_to(task.path).as_posix(): path.read_text(encoding="utf-8", errors="replace")
        for path in public_test_files(task)
    }
    return {
        "task": task.task_id,
        "task_path": str(task.path),
        "language": task.language,
        "instructions": "\n\n".join(path.read_text(encoding="utf-8") for path in docs)
        or task.path.name,
        "starters": starters,
        "tests": tests,
    }


def replacements_from_json(items: list[dict[str, str]]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in items:
        path = safe_relative(item["path"]).as_posix()
        if path in result:
            raise ValueError(f"duplicate replacement: {path}")
        result[path] = item["content"]
    return result
