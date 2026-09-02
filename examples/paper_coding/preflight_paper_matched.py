from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from polyglot import DEFAULT_IMAGES, split_counts  # noqa: E402
from run import CodexCli  # noqa: E402


async def model_smoke(config: dict[str, object]) -> None:
    client = CodexCli(
        str(config["model"]),
        int(config["model_timeout_seconds"]),
        ROOT / "data" / "paper-coding" / "codex-empty",
        ROOT / "runs" / "paper-matched-v2-preflight-model-calls.jsonl",
        str(config["reasoning_effort"]),
    )
    schema = {
        "type": "object",
        "properties": {"status": {"type": "string", "enum": ["ok"]}},
        "required": ["status"],
        "additionalProperties": False,
    }
    result = await client.json("Return status ok.", schema, "paper-v2-model-smoke")
    assert result == {"status": "ok"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-smoke", action="store_true")
    args = parser.parse_args()
    config_dir = HERE / "configs"
    paths = [
        config_dir / "paper_matched_v2_rqgm.json",
        config_dir / "paper_matched_v2_hgm_h.json",
    ]
    configs = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    allowed = {"claim", "experiment_condition", "output", "limitations"}
    projections = [
        {key: value for key, value in config.items() if key not in allowed}
        for config in configs
    ]
    assert projections[0] == projections[1], "RQGM and HGM-H cells are not matched"
    config = configs[0]
    assert (config["model"], config["reasoning_effort"]) == ("gpt-5.5", "low")
    assert config["validation_budget"] == 12_288
    assert config["checkpoints"] == [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    train, validation, test = split_counts(
        ROOT / "data" / "polyglot-benchmark",
        int(config["random_seed"]),
        int(config["polyglot_train_tasks"]),
        int(config["polyglot_validation_tasks"]),
        int(config["polyglot_test_tasks"]),
    )
    assert (len(train), len(validation), len(test)) == (10, 49, 166)
    ids = [{task.task_id for task in group} for group in (train, validation, test)]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    for image in DEFAULT_IMAGES.values():
        subprocess.run(["docker", "image", "inspect", image], check=True, capture_output=True)
    if args.model_smoke:
        asyncio.run(model_smoke(config))
    print("PASS: matched paper-v2 public configuration and 10/49/166 replacement split")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
