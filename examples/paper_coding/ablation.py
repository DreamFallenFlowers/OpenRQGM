from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from run import EXPERIMENT_CONDITIONS, build_engine, canonical_hash, execute

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = Path(__file__).resolve().parent / "configs" / "ablations"
BUDGETS = (512, 1024)
CONDITIONS = ("verifier_only", "fixed_reviewer", "coevolving_reviewer")
ALLOWED_CELL_FIELDS = {
    "claim",
    "experiment_condition",
    "validation_budget",
    "checkpoints",
    "output",
}


def config_path(budget: int, condition: str) -> Path:
    return CONFIG_DIR / f"{budget}-{condition}.json"


def load_matrix() -> dict[tuple[int, str], dict[str, Any]]:
    return {
        (budget, condition): json.loads(config_path(budget, condition).read_text(encoding="utf-8"))
        for budget in BUDGETS
        for condition in CONDITIONS
    }


def matched_projection(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if key not in ALLOWED_CELL_FIELDS}


def git_identity() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )
    return {"commit": commit, "dirty": dirty}


def preflight_matrix() -> dict[str, Any]:
    matrix = load_matrix()
    projections = {canonical_hash(matched_projection(config)) for config in matrix.values()}
    if len(projections) != 1:
        raise ValueError("ablation cells differ outside the declared condition/budget fields")

    cells: list[dict[str, Any]] = []
    data_fingerprints: set[str] = set()
    anchor_fingerprints: set[str] = set()
    for (budget, condition), config in matrix.items():
        if config["ablation_id"] != "polyglot-matched-v1":
            raise ValueError("unexpected ablation id")
        if config["validation_budget"] != budget:
            raise ValueError("filename and validation budget disagree")
        if config["experiment_condition"] != condition:
            raise ValueError("filename and experiment condition disagree")
        if condition not in EXPERIMENT_CONDITIONS:
            raise ValueError(condition)
        if config["minimum_anchor_outcomes"] != config["crave_anchor_examples"]:
            raise ValueError("anchor eligibility must require the complete configured anchor")

        engine, evaluator, heldout = build_engine(config, object())
        train = engine.runtime.training_feedback.coder_tasks
        split_identity = {
            "train": [task.task_id for task in train],
            "validation": [task.task_id for task in evaluator.validation_tasks],
            "heldout": [task.task_id for task in heldout],
        }
        data_fingerprint = canonical_hash(split_identity)
        anchor_fingerprint = engine.runtime.anchor_provider.fingerprint
        data_fingerprints.add(data_fingerprint)
        anchor_fingerprints.add(anchor_fingerprint)

        expected_tasks = 1 if condition == "verifier_only" else 3
        expected_slots = 1 if condition == "coevolving_reviewer" else 0
        if len(engine.tasks) != expected_tasks or len(engine.slots) != expected_slots:
            raise ValueError(f"unexpected topology for {condition}")
        if len(heldout) != 60:
            raise ValueError("heldout must contain ten tasks for each of six languages")

        cells.append(
            {
                "budget": budget,
                "condition": condition,
                "config": str(config_path(budget, condition)),
                "config_fingerprint": canonical_hash(config),
                "data_fingerprint": data_fingerprint,
                "anchor_fingerprint": anchor_fingerprint,
                "tasks": [task.task_id for task in engine.tasks],
                "evaluator_slots": [slot.slot_id for slot in engine.slots],
                "heldout_count": len(heldout),
            }
        )

    if len(data_fingerprints) != 1 or len(anchor_fingerprints) != 1:
        raise ValueError("ablation cells do not share identical data and anchor splits")
    return {
        "ablation_id": "polyglot-matched-v1",
        "status": "PASS",
        "git": git_identity(),
        "matched_projection_fingerprint": next(iter(projections)),
        "data_fingerprint": next(iter(data_fingerprints)),
        "anchor_fingerprint": next(iter(anchor_fingerprints)),
        "cells": cells,
    }


def assert_fresh_output(config: dict[str, Any]) -> None:
    output = ROOT / config["output"]
    if (output / "summary.json").exists() or (output / "state.json").exists():
        raise FileExistsError(f"refusing to overwrite completed run: {output}")


def audit_state(state: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "skills" / "rqgm-reproduction" / "scripts" / "audit_run.py"),
            str(state),
        ],
        cwd=ROOT,
        check=True,
    )


async def run_cells(selected: list[tuple[int, str]]) -> None:
    report = preflight_matrix()
    if report["git"]["dirty"]:
        raise RuntimeError("refusing a registered ablation from a dirty worktree")
    manifest = ROOT / "runs" / "polyglot-matched-v1-manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    matrix = load_matrix()
    for cell in selected:
        config = matrix[cell]
        assert_fresh_output(config)
        print(f"[ablation:start] budget={cell[0]} condition={cell[1]}", flush=True)
        await execute(config_path(*cell))
        state = ROOT / config["output"] / "state.json"
        audit_state(state)
        print(f"[ablation:done] budget={cell[0]} condition={cell[1]}", flush=True)


def report_budget(budget: int) -> dict[str, Any]:
    matrix = load_matrix()
    rows: list[dict[str, Any]] = []
    expected_data: set[str] = set()
    expected_anchor: set[str] = set()
    expected_revisions: set[str] = set()
    for condition in CONDITIONS:
        config = matrix[(budget, condition)]
        summary_path = ROOT / config["output"] / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"missing completed cell: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("ablation_id") != "polyglot-matched-v1":
            raise ValueError(f"wrong ablation id in {summary_path}")
        if summary.get("experiment_condition") != condition:
            raise ValueError(f"condition mismatch in {summary_path}")
        if summary.get("config_fingerprint") != canonical_hash(config):
            raise ValueError(f"configuration drift in {summary_path}")
        if summary["result"]["validation_outcomes"] != budget:
            raise ValueError(f"incomplete validation budget in {summary_path}")
        if summary["token_totals"]["blended_tokens"] <= 0:
            raise ValueError(f"missing token accounting in {summary_path}")
        if summary["token_totals"].get("unmetered_calls") != 0:
            raise ValueError(f"incomplete token accounting in {summary_path}")
        if summary.get("source", {}).get("dirty") is not False:
            raise ValueError(f"cell did not run from a clean worktree: {summary_path}")
        audit_state(ROOT / config["output"] / "state.json")
        expected_data.add(summary["data_split_fingerprint"])
        expected_anchor.add(summary["anchor_provider"]["fingerprint"])
        expected_revisions.add(summary["source"]["commit"])
        result = summary["result"]
        heldout = {
            endpoint: {
                "passes": scores["passes"],
                "total": scores["total"],
                "macro_average": scores["macro_average"],
                "by_language": scores["by_language"],
            }
            for endpoint, scores in summary["heldout"].items()
        }
        rows.append(
            {
                "condition": condition,
                "validation_outcomes": result["validation_outcomes"],
                "archive_size": result["archive_size"],
                "heldout": heldout,
                "raw_total_tokens": summary["token_totals"]["raw_total_tokens"],
                "blended_tokens": summary["token_totals"]["blended_tokens"],
                "model_calls": len(summary["model_calls"]),
                "wall_time_seconds": summary["wall_time_seconds"],
                "replacements": sum(
                    len(checkpoint["replacements"]) for checkpoint in result["checkpoints"]
                ),
                "erased_records": sum(
                    checkpoint["erased_records"] for checkpoint in result["checkpoints"]
                ),
            }
        )
    if len(expected_data) != 1 or len(expected_anchor) != 1 or len(expected_revisions) != 1:
        raise ValueError("completed cells were not evaluated on matched code, data, and anchors")
    by_condition = {row["condition"]: row for row in rows}
    fixed = by_condition["fixed_reviewer"]
    evolved = by_condition["coevolving_reviewer"]

    def endpoint_deltas(left: dict[str, Any], right: dict[str, Any]) -> dict[str, int]:
        return {
            endpoint: left["heldout"][endpoint]["passes"] - right["heldout"][endpoint]["passes"]
            for endpoint in ("generalist", "coder_specialist")
        }

    return {
        "ablation_id": "polyglot-matched-v1",
        "budget": budget,
        "data_split_fingerprint": next(iter(expected_data)),
        "anchor_fingerprint": next(iter(expected_anchor)),
        "source_revision": next(iter(expected_revisions)),
        "cells": rows,
        "contrasts": {
            "fixed_reviewer_minus_verifier_only": {
                "heldout_pass_delta": endpoint_deltas(fixed, by_condition["verifier_only"]),
                "blended_token_delta": fixed["blended_tokens"]
                - by_condition["verifier_only"]["blended_tokens"],
            },
            "coevolution_minus_fixed_reviewer": {
                "heldout_pass_delta": endpoint_deltas(evolved, fixed),
                "blended_token_delta": evolved["blended_tokens"] - fixed["blended_tokens"],
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight or run the matched RQGM ablation")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--confirm-expensive-matrix", action="store_true")
    parser.add_argument("--budget", type=int, choices=BUDGETS)
    parser.add_argument("--condition", choices=CONDITIONS)
    args = parser.parse_args()

    if args.report:
        if args.budget is None:
            raise SystemExit("--report requires --budget")
        report = report_budget(args.budget)
        output = ROOT / "runs" / f"polyglot-matched-v1-{args.budget}-report.json"
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return

    if args.run_all:
        if not args.confirm_expensive_matrix:
            raise SystemExit("--run-all requires --confirm-expensive-matrix")
        asyncio.run(
            run_cells([(budget, condition) for budget in BUDGETS for condition in CONDITIONS])
        )
        return
    if args.run:
        if args.budget is None or args.condition is None:
            raise SystemExit("--run requires --budget and --condition")
        asyncio.run(run_cells([(args.budget, args.condition)]))
        return
    print(json.dumps(preflight_matrix(), indent=2))


if __name__ == "__main__":
    main()
