from __future__ import annotations

import json
import os
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .archive import Archive
from .engine import RQGM
from .models import (
    AnchorScore,
    CheckpointEvent,
    EvaluatorCandidate,
    EvaluatorSlot,
    UtilityRecord,
    WorkspaceNode,
)

STATE_VERSION = 2


def snapshot(engine: RQGM) -> dict[str, Any]:
    """Create a JSON-compatible audit snapshot without materializing anchor examples."""
    return {
        "state_version": STATE_VERSION,
        "initialized": engine.initialized,
        "validation_outcomes": engine.validation_outcomes,
        "rng_state": deepcopy(engine.rng.bit_generator.state),
        "config": asdict(engine.config),
        "tasks": [asdict(task) for task in engine.tasks],
        "slots": [
            {
                "slot_id": slot.slot_id,
                "role_id": slot.role_id,
                "epoch": slot.epoch,
                "incumbent": {
                    "candidate_id": slot.incumbent.candidate_id,
                    "slot_id": slot.incumbent.slot_id,
                    "artifact": slot.incumbent.artifact,
                    "source": slot.incumbent.source,
                    "parent_id": slot.incumbent.parent_id,
                },
            }
            for slot in engine.slot_map.values()
        ],
        "nodes": [
            {
                "node_id": node.node_id,
                "workspace": node.workspace,
                "parent_id": node.parent_id,
                "created_at_step": node.created_at_step,
                "valid": node.valid,
                "cached_artifacts": node.cached_artifacts,
                "training_feedback": node.training_feedback,
            }
            for node in engine.archive.nodes.values()
        ],
        "records": [
            {
                "record_id": record.record_id,
                "node_id": record.node_id,
                "role_id": record.role_id,
                "task_id": record.task_id,
                "outcome": record.outcome,
                "epoch_vector": record.epoch_vector,
                "evaluator_slot": record.evaluator_slot,
                "evaluator_id": record.evaluator_id,
                "artifact_key": record.artifact_key,
                "metadata": record.metadata,
                "valid": record.valid,
                "invalidated_at_checkpoint": record.invalidated_at_checkpoint,
                "invalidation_reason": record.invalidation_reason,
            }
            for record in engine.archive.records
        ],
        "training_feedback": dict(engine.archive.training_feedback),
        "checkpoints": [asdict(event) for event in engine.checkpoint_events],
        "anchor_provider": {
            "provider_id": engine.runtime.anchor_provider.provider_id,
            "version": engine.runtime.anchor_provider.version,
            "fingerprint": engine.runtime.anchor_provider.fingerprint,
        },
    }


def save_state(engine: RQGM, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(snapshot(engine), stream, indent=2, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, output)
    return output


def restore_state(engine: RQGM, path: str | Path, *, require_resumable: bool = False) -> RQGM:
    """Restore mutable search state into an equivalently configured runtime.

    Callers must supply the runtime again so private anchors and executable
    evaluators never enter the state file.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    version = data.get("state_version")
    if version not in (1, STATE_VERSION):
        raise ValueError("unsupported RQGM state version")
    if require_resumable and version != STATE_VERSION:
        raise ValueError("state predates deterministic resume support")
    provider = data.get("anchor_provider", {})
    if provider != {
        "provider_id": engine.runtime.anchor_provider.provider_id,
        "version": engine.runtime.anchor_provider.version,
        "fingerprint": engine.runtime.anchor_provider.fingerprint,
    }:
        raise ValueError("anchor provider identity/version changed")
    expected_tasks = [asdict(task) for task in engine.tasks]
    if data.get("tasks") != expected_tasks:
        raise ValueError("role/task topology changed")
    expected_config = asdict(engine.config)
    saved_config = data.get("config", {})
    saved_config["checkpoints"] = tuple(saved_config.get("checkpoints", ()))
    if saved_config != expected_config:
        raise ValueError("RQGM statistical/search configuration changed")

    nodes = [WorkspaceNode(**raw) for raw in data["nodes"]]
    seed = next((node for node in nodes if node.parent_id is None), None)
    if seed is None:
        raise ValueError("state has no seed node")
    archive = Archive(seed)
    pending = [node for node in nodes if node.node_id != seed.node_id]
    while pending:
        progress = False
        for node in list(pending):
            if node.parent_id in archive.nodes:
                archive.add_node(node)
                pending.remove(node)
                progress = True
        if not progress:
            raise ValueError("state archive contains a broken or cyclic lineage")
    archive.records = [UtilityRecord(**raw) for raw in data["records"]]
    archive.training_feedback.update(data.get("training_feedback", {}))

    engine.archive = archive
    engine.slot_map = {}
    for raw in data["slots"]:
        incumbent = EvaluatorCandidate(**raw["incumbent"])
        slot = EvaluatorSlot(raw["slot_id"], raw["role_id"], incumbent, raw["epoch"])
        engine.slot_map[slot.slot_id] = slot
    engine.validation_outcomes = int(data["validation_outcomes"])
    engine.initialized = bool(data.get("initialized", engine.validation_outcomes > 0))
    if "rng_state" in data:
        engine.rng.bit_generator.state = data["rng_state"]
    engine.checkpoint_events = []
    for raw in data.get("checkpoints", []):
        raw["anchor_scores"] = {
            slot_id: [AnchorScore(**score) for score in scores]
            for slot_id, scores in raw["anchor_scores"].items()
        }
        raw["replacements"] = {
            slot_id: tuple(value) for slot_id, value in raw["replacements"].items()
        }
        engine.checkpoint_events.append(CheckpointEvent(**raw))
    return engine
