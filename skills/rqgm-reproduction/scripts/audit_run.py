#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def audit(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    errors: list[str] = []
    records = data.get("records", [])
    budget = data.get("validation_outcomes")
    if len(records) != budget:
        errors.append(f"record count {len(records)} does not equal validation budget {budget}")
    if any(record.get("outcome") not in (0, 1) for record in records):
        errors.append("non-binary utility outcome found")
    anchor_metadata = data.get("anchor_provider", {})
    if "examples" in anchor_metadata:
        errors.append("private anchor examples were serialized")
    if not all(anchor_metadata.get(key) for key in ("provider_id", "version", "fingerprint")):
        errors.append("anchor identity/version/fingerprint metadata is incomplete")

    epochs = {slot["slot_id"]: slot["epoch"] for slot in data.get("slots", [])}
    for record in records:
        vector = record.get("epoch_vector", {})
        if not set(vector).issubset(epochs):
            errors.append(f"record {record.get('record_id')} has an unknown epoch slot")
        if not record.get("valid", True) and record.get("invalidated_at_checkpoint") is None:
            errors.append(f"record {record.get('record_id')} is invalid without a checkpoint")

    checkpoints = data.get("checkpoints", [])
    positions = [event["checkpoint"] for event in checkpoints]
    if positions != sorted(set(positions)):
        errors.append("checkpoint positions are not unique and increasing")
    for event in checkpoints:
        changed = set(event.get("replacements", {}))
        if changed and event.get("erased_records", 0) <= 0:
            errors.append(f"checkpoint {event['checkpoint']} replaced a slot but erased no records")

    valid_by_task = Counter(record["task_id"] for record in records if record.get("valid", True))
    if not valid_by_task:
        errors.append("no valid endpoint evidence remains")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit an RQGM state snapshot")
    parser.add_argument("state", type=Path)
    args = parser.parse_args()
    errors = audit(args.state)
    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        return 1
    print("PASS: structural RQGM state invariants hold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
