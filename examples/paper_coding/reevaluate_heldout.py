from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import run as paper_run

from rqgm.persistence import restore_state


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def make_client(config: dict[str, Any], run_dir: Path) -> Any:
    cli_class = paper_run.CodexCli
    app_server_class = getattr(paper_run, "CodexAppServer", None)
    use_app_server = config.get("model_backend") == "app_server" and app_server_class
    client_class = app_server_class if use_app_server else cli_class
    options: dict[str, Any] = {}
    if use_app_server:
        options = {
            "concurrency": int(config.get("model_concurrency", 4)),
            "worker_max_calls": int(config.get("persistent_worker_max_calls", 200)),
        }
    return client_class(
        model=config["model"],
        timeout=int(config["model_timeout_seconds"]),
        workdir=paper_run.DATA / "codex-empty",
        ledger_path=run_dir / "model-calls-heldout-complete.jsonl",
        reasoning_effort=config.get("reasoning_effort"),
        **options,
    )


def snapshot(
    destination: Path,
    source_summary: dict[str, Any],
    languages: set[str],
    endpoint_results: dict[str, list[dict[str, Any]]],
) -> None:
    endpoints: dict[str, Any] = {}
    for name, outcomes in endpoint_results.items():
        original = source_summary["heldout"][name]["outcomes"]
        untouched = [row for row in original if row["task"].split("/", 1)[0] not in languages]
        endpoints[name] = {
            "node_id": source_summary["heldout"][name]["node_id"],
            "complete_suite_passes": sum(row["outcome"] for row in outcomes),
            "complete_suite_total": len(outcomes),
            "combined_passes": sum(row["outcome"] for row in untouched + outcomes),
            "combined_total": len(untouched) + len(outcomes),
            "outcomes": outcomes,
        }
    paper_run.atomic_json(
        destination,
        {
            "status": "complete"
            if all(
                value["complete_suite_total"]
                == sum(
                    row["task"].split("/", 1)[0] in languages
                    for row in source_summary["heldout"][name]["outcomes"]
                )
                for name, value in endpoints.items()
            )
            else "running",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "languages": sorted(languages),
            "suite_policy": {
                "java": (
                    "remove Exercism @Disabled annotations; require >=1 started and 0 skipped tests"
                ),
            },
            "note": (
                "This is a fresh endpoint re-evaluation with the saved agents. "
                "It replaces affected language rows but cannot retroactively "
                "reuse the original generated patches."
            ),
            "endpoints": endpoints,
        },
    )


async def execute(config_path: Path, run_dir: Path, languages: set[str], batch_size: int) -> None:
    config = load_json(config_path)
    source_summary = load_json(run_dir / "summary.json")
    destination = run_dir / "heldout-complete-supplement.json"
    prior = load_json(destination) if destination.exists() else {"endpoints": {}}
    client = make_client(config, run_dir)
    engine, evaluator, heldout = paper_run.build_engine(config, client)
    restore_state(engine, run_dir / "state.json")
    tasks = [task for task in heldout if task.language in languages]
    endpoint_ids = {
        name: source_summary["heldout"][name]["node_id"]
        for name in ("coder_specialist", "generalist")
    }
    endpoint_results: dict[str, list[dict[str, Any]]] = {
        name: list(prior.get("endpoints", {}).get(name, {}).get("outcomes", []))
        for name in endpoint_ids
    }
    try:
        for name, node_id in endpoint_ids.items():
            completed = {row["task"] for row in endpoint_results[name]}
            remaining = [task for task in tasks if task.task_id not in completed]
            for start in range(0, len(remaining), batch_size):
                batch = remaining[start : start + batch_size]
                rows = await evaluator.heldout_results(engine.archive.nodes[node_id], batch)
                endpoint_results[name].extend(rows)
                snapshot(destination, source_summary, languages, endpoint_results)
                passed = sum(row["outcome"] for row in endpoint_results[name])
                print(
                    f"[heldout-complete] endpoint={name} "
                    f"done={len(endpoint_results[name])}/{len(tasks)} passes={passed}",
                    flush=True,
                )
    finally:
        close = getattr(client, "close", None)
        if close is not None:
            await close()
    snapshot(destination, source_summary, languages, endpoint_results)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--languages", default="java")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    languages = {item.strip() for item in args.languages.split(",") if item.strip()}
    asyncio.run(execute(args.config.resolve(), args.run_dir.resolve(), languages, args.batch_size))


if __name__ == "__main__":
    main()
