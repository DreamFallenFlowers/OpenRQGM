from __future__ import annotations

import hashlib
import json
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "paper-coding"
POLYGLOT = ROOT / "data" / "polyglot-benchmark"
POLYGLOT_URL = "https://github.com/Aider-AI/polyglot-benchmark.git"
CRAVE_DATASET = "TuringEnterprises/CRAVE"


def fetch_rows(split: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    offset = 0
    while True:
        query = urllib.parse.urlencode(
            {
                "dataset": CRAVE_DATASET,
                "config": "default",
                "split": split,
                "offset": offset,
                "length": 100,
            }
        )
        with urllib.request.urlopen(
            f"https://datasets-server.huggingface.co/rows?{query}", timeout=60
        ) as response:
            payload = json.load(response)
        page = [item["row"] for item in payload["rows"]]
        rows.extend(page)
        offset += len(page)
        if not page or offset >= int(payload["num_rows_total"]):
            return rows


def stable_fingerprint(rows: list[dict[str, object]]) -> str:
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    if not POLYGLOT.exists():
        subprocess.run(
            ["git", "clone", "--filter=blob:none", POLYGLOT_URL, str(POLYGLOT)],
            check=True,
        )
    commit = subprocess.check_output(
        ["git", "-C", str(POLYGLOT), "rev-parse", "HEAD"], text=True
    ).strip()
    metadata: dict[str, object] = {"polyglot_commit": commit, "crave": {}}
    for split in ("train", "validation", "test"):
        rows = fetch_rows(split)
        path = DATA / f"crave-{split}.json"
        path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        metadata["crave"][split] = {  # type: ignore[index]
            "rows": len(rows),
            "fingerprint": stable_fingerprint(rows),
        }
    (DATA / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
