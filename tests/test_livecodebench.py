from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples" / "paper_coding"))

import livecodebench as lcb  # noqa: E402


def test_material_never_exposes_evaluation_payload() -> None:
    task = lcb.LiveCodeBenchTask(
        "q1",
        "Add two integers.",
        "",
        "2025-01-01T00:00:00",
        "easy",
        {"input_output": json.dumps({"inputs": ["1 2"], "outputs": ["3"]})},
    )
    material = lcb.material(task)
    assert material["tests"] == {}
    assert "1 2" not in json.dumps(material)
    assert task.task_id == "livecodebench/q1"


def test_snapshot_rejects_wrong_commit(tmp_path: Path) -> None:
    path = tmp_path / "snapshot.json"
    path.write_text(
        json.dumps(
            {
                "metadata": {
                    "repository_commit": "wrong",
                    "release_version": lcb.LIVE_CODE_BENCH_RELEASE,
                },
                "tasks": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="commit drift"):
        lcb.load_snapshot(path)
