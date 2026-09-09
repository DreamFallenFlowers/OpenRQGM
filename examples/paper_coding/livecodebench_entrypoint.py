from __future__ import annotations

import json
import os
import resource
from pathlib import Path

from testing_util import run_test


def main() -> None:
    memory_bytes = int(os.environ.get("OPENRQGM_MEMORY_BYTES", 4 * 1024**3))
    process_limit = int(os.environ.get("OPENRQGM_PROCESS_LIMIT", 128))
    resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
    resource.setrlimit(resource.RLIMIT_NPROC, (process_limit, process_limit))
    request = json.loads(Path("/input/request.json").read_text(encoding="utf-8"))
    results, metadata = run_test(
        request["sample"], test=request["code"], debug=False, timeout=int(request["timeout"])
    )
    values = list(results or [])
    passed = bool(values) and all(value is True for value in values)
    print(json.dumps({"passed": passed, "tests": len(values), "metadata": metadata}))


if __name__ == "__main__":
    main()
