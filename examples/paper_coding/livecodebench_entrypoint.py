from __future__ import annotations

import json
from pathlib import Path

from testing_util import run_test


def main() -> None:
    request = json.loads(Path("/input/request.json").read_text(encoding="utf-8"))
    results, metadata = run_test(
        request["sample"], test=request["code"], debug=False, timeout=int(request["timeout"])
    )
    values = list(results or [])
    passed = bool(values) and all(value is True for value in values)
    print(json.dumps({"passed": passed, "tests": len(values), "metadata": metadata}))


if __name__ == "__main__":
    main()
